"""Broker-neutral execution operations for paper and gated live deployment.

This module contains no strategy research logic.  It implements the plumbing
that a live deployment needs around already-approved contract-level intents:

* a persist-before-send, hash-chained outbox;
* idempotent broker submission and event replay;
* a deterministic paper broker for forward shadow testing;
* position/cash reconciliation;
* heartbeat monitoring; and
* a latched, persisted kill switch with dual-control reset.

The included broker is intentionally a paper adapter.  A real-capital launch
still requires a separately certified adapter for the selected broker and
verified evidence for every readiness gate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..marketdata.contracts import (
    SERIAL_CONTRACT_COLUMNS,
    ContractDataError,
    SerialDataLimits,
    validated_serial_snapshot,
)
from ..controls.production import (
    ArtifactBackedReadinessCheck,
    ProductionLimits,
    RuntimeHealth,
    overall_readiness_status,
)


PASS = "PASS"
BLOCKED = "BLOCKED"
READY = "READY"


RuntimeHealthCheck = Callable[[pd.Timestamp], RuntimeHealth]
SerialSnapshotProvider = Callable[[pd.Timestamp], pd.DataFrame]


class OperationsError(RuntimeError):
    """Raised when an operational invariant would be violated."""


def _utc(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OperationsError(f"{label} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise OperationsError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _finite(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _utc(value, "timestamp").isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True)
class OrderRequest:
    """A validated exchange-contract order request."""

    order_id: str
    contract_id: str
    root_symbol: str
    side: str
    quantity: int
    decision_timestamp: pd.Timestamp
    order_type: str = "MARKETABLE_LIMIT"
    limit_price: float | None = None
    reduce_only: bool = False
    roll_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("order_id", "contract_id", "root_symbol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.contract_id == self.root_symbol:
            raise ValueError("contract_id must identify an expiry, not a continuous root")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity < 1:
            raise ValueError("quantity must be a positive integer")
        _utc(self.decision_timestamp, "decision_timestamp")
        if self.order_type not in {"LIMIT", "MARKETABLE_LIMIT", "TWAP", "VWAP"}:
            raise ValueError("unsupported order_type")
        if self.order_type in {"LIMIT", "MARKETABLE_LIMIT"}:
            if self.limit_price is None or not _finite(self.limit_price):
                raise ValueError("limit orders require a finite limit_price")
        elif self.limit_price is not None and not _finite(self.limit_price):
            raise ValueError("limit_price must be finite when supplied")
        if type(self.reduce_only) is not bool:
            raise ValueError("reduce_only must be an explicit boolean")

    def payload(self) -> dict[str, Any]:
        values = asdict(self)
        values["decision_timestamp"] = _utc(
            self.decision_timestamp, "decision_timestamp"
        ).isoformat()
        return values


def _sha256_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class BrokerDeploymentIdentity:
    """Canonical identity of the broker deployment selected for an account.

    The build digest must identify the independently certified adapter build,
    not a mutable package version string.  Live routing compares the digest of
    this complete identity with artifact-backed readiness evidence.
    """

    broker_id: str
    adapter_id: str
    adapter_build_sha256: str
    account_id: str
    environment: str

    def __post_init__(self) -> None:
        for name in ("broker_id", "adapter_id", "account_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
        _sha256_digest(self.adapter_build_sha256, "adapter_build_sha256")
        if self.environment not in {"paper", "staging", "production"}:
            raise ValueError(
                "environment must be paper, staging, or production"
            )

    def payload(self) -> dict[str, str]:
        return {
            "broker_id": self.broker_id,
            "adapter_id": self.adapter_id,
            "adapter_build_sha256": self.adapter_build_sha256,
            "account_id": self.account_id,
            "environment": self.environment,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical(self.payload()).encode("utf-8")
        ).hexdigest()


def position_snapshot_sha256(positions: Mapping[Any, Any]) -> str:
    """Hash an exact integer broker-position snapshot deterministically."""

    if not isinstance(positions, Mapping):
        raise TypeError("positions must be a mapping")
    normalized: dict[str, int] = {}
    for raw_contract, raw_quantity in positions.items():
        contract = str(raw_contract).strip()
        if not contract or contract in normalized:
            raise OperationsError("position snapshot has blank or duplicate contracts")
        if (
            isinstance(raw_quantity, bool)
            or not isinstance(raw_quantity, (int, np.integer))
        ):
            raise OperationsError("position snapshot quantities must be integers")
        normalized[contract] = int(raw_quantity)
    payload = [[contract, normalized[contract]] for contract in sorted(normalized)]
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def serial_snapshot_sha256(snapshot: pd.DataFrame) -> str:
    """Hash a validated serial snapshot using its normalized canonical values."""

    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty:
        raise OperationsError("serial snapshot must be a non-empty DataFrame")
    missing = [column for column in SERIAL_CONTRACT_COLUMNS if column not in snapshot]
    if missing:
        raise OperationsError(
            "serial snapshot is missing columns: " + ", ".join(missing)
        )
    ordered = snapshot.loc[:, SERIAL_CONTRACT_COLUMNS].sort_values(
        ["root_symbol", "contract_id"]
    )
    records: list[dict[str, Any]] = []
    for raw in ordered.to_dict("records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or pd.isna(value):
                row[key] = None
            elif isinstance(value, pd.Timestamp):
                row[key] = (
                    value.isoformat()
                    if value.tzinfo is None
                    else value.tz_convert("UTC").isoformat()
                )
            else:
                row[key] = _json_value(value)
        records.append(row)
    return hashlib.sha256(_canonical(records).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PortfolioComplianceRequest:
    """Exact live batch presented to an external portfolio/compliance gate."""

    orders: tuple[OrderRequest, ...]
    broker_positions: Mapping[str, int]
    serial_snapshot: pd.DataFrame
    nav_usd: float
    as_of: pd.Timestamp
    broker_identity: BrokerDeploymentIdentity

    def __post_init__(self) -> None:
        if (
            not isinstance(self.orders, tuple)
            or not self.orders
            or any(not isinstance(order, OrderRequest) for order in self.orders)
        ):
            raise TypeError("orders must be a non-empty tuple of OrderRequest objects")
        if not isinstance(self.broker_positions, Mapping):
            raise TypeError("broker_positions must be a mapping")
        position_snapshot_sha256(self.broker_positions)
        serial_snapshot_sha256(self.serial_snapshot)
        if not _finite(self.nav_usd) or float(self.nav_usd) <= 0:
            raise ValueError("nav_usd must be finite and positive")
        _utc(self.as_of, "portfolio compliance as_of")
        if not isinstance(self.broker_identity, BrokerDeploymentIdentity):
            raise TypeError("broker_identity must be a BrokerDeploymentIdentity")

    def payload(self) -> dict[str, Any]:
        return {
            "orders": [order.payload() for order in self.orders],
            "position_snapshot_sha256": position_snapshot_sha256(
                self.broker_positions
            ),
            "serial_snapshot_sha256": serial_snapshot_sha256(
                self.serial_snapshot
            ),
            "nav_usd": float(self.nav_usd),
            "as_of": _utc(self.as_of, "portfolio compliance as_of").isoformat(),
            "broker_identity_sha256": self.broker_identity.sha256,
        }

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(self.payload()).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class PortfolioComplianceDecision:
    """Request-bound decision returned by the external live control plane."""

    decision_id: str
    evaluated_at: pd.Timestamp
    request_sha256: str
    policy_sha256: str
    approved: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.decision_id, str)
            or not self.decision_id.strip()
            or self.decision_id != self.decision_id.strip()
        ):
            raise ValueError("decision_id must be a non-empty trimmed string")
        _utc(self.evaluated_at, "portfolio compliance evaluated_at")
        _sha256_digest(self.request_sha256, "request_sha256")
        _sha256_digest(self.policy_sha256, "policy_sha256")
        if type(self.approved) is not bool:
            raise ValueError("approved must be an explicit boolean")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in self.reasons
        ):
            raise ValueError("reasons must be a tuple of non-empty strings")
        if not self.approved and not self.reasons:
            raise ValueError("a denied decision must contain a reason")


PortfolioComplianceCheck = Callable[
    [PortfolioComplianceRequest], PortfolioComplianceDecision
]


@dataclass(frozen=True)
class CertifiedIntent:
    """Authenticated exact order intent produced by the upstream risk service.

    Live routing still recomputes serial, participation, gross and margin
    controls.  The certificate additionally prevents an arbitrary caller from
    substituting a different alpha order after those controls were approved.
    """

    order: OrderRequest
    issued_at: pd.Timestamp
    expires_at: pd.Timestamp
    model_fingerprint_sha256: str
    config_fingerprint_sha256: str
    source_fingerprint_sha256: str
    serial_snapshot_sha256: str
    position_snapshot_sha256: str
    broker_identity_sha256: str
    compliance_policy_sha256: str
    projected_participation: float
    projected_gross_notional_multiple: float
    projected_margin_fraction: float
    key_id: str
    signature_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.order, OrderRequest):
            raise TypeError("order must be an OrderRequest")
        issued = _utc(self.issued_at, "intent issued_at")
        expires = _utc(self.expires_at, "intent expires_at")
        if expires <= issued:
            raise ValueError("intent expires_at must be after issued_at")
        for name in (
            "model_fingerprint_sha256",
            "config_fingerprint_sha256",
            "source_fingerprint_sha256",
            "serial_snapshot_sha256",
            "position_snapshot_sha256",
            "broker_identity_sha256",
            "compliance_policy_sha256",
            "signature_sha256",
        ):
            _sha256_digest(getattr(self, name), name)
        for name in (
            "projected_participation",
            "projected_gross_notional_multiple",
            "projected_margin_fraction",
        ):
            value = getattr(self, name)
            if not _finite(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise ValueError("key_id must be non-empty")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "order": self.order.payload(),
            "issued_at": _utc(self.issued_at, "intent issued_at").isoformat(),
            "expires_at": _utc(self.expires_at, "intent expires_at").isoformat(),
            "model_fingerprint_sha256": self.model_fingerprint_sha256,
            "config_fingerprint_sha256": self.config_fingerprint_sha256,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "serial_snapshot_sha256": self.serial_snapshot_sha256,
            "position_snapshot_sha256": self.position_snapshot_sha256,
            "broker_identity_sha256": self.broker_identity_sha256,
            "compliance_policy_sha256": self.compliance_policy_sha256,
            "projected_participation": float(self.projected_participation),
            "projected_gross_notional_multiple": float(
                self.projected_gross_notional_multiple
            ),
            "projected_margin_fraction": float(self.projected_margin_fraction),
            "key_id": self.key_id,
        }


def certify_order_intent(
    order: OrderRequest,
    *,
    issued_at: Any,
    expires_at: Any,
    model_fingerprint_sha256: str,
    config_fingerprint_sha256: str,
    source_fingerprint_sha256: str,
    serial_snapshot_digest: str,
    position_snapshot_digest: str,
    broker_identity_digest: str,
    compliance_policy_digest: str,
    projected_participation: float,
    projected_gross_notional_multiple: float,
    projected_margin_fraction: float,
    key_id: str,
    signing_key: bytes,
) -> CertifiedIntent:
    """Create an HMAC-authenticated compact intent for a separate risk service."""

    if not isinstance(signing_key, bytes) or len(signing_key) < 32:
        raise ValueError("signing_key must contain at least 32 bytes")
    values = {
        "order": order,
        "issued_at": _utc(issued_at, "intent issued_at"),
        "expires_at": _utc(expires_at, "intent expires_at"),
        "model_fingerprint_sha256": _sha256_digest(
            model_fingerprint_sha256, "model_fingerprint_sha256"
        ),
        "config_fingerprint_sha256": _sha256_digest(
            config_fingerprint_sha256, "config_fingerprint_sha256"
        ),
        "source_fingerprint_sha256": _sha256_digest(
            source_fingerprint_sha256, "source_fingerprint_sha256"
        ),
        "serial_snapshot_sha256": _sha256_digest(
            serial_snapshot_digest, "serial_snapshot_digest"
        ),
        "position_snapshot_sha256": _sha256_digest(
            position_snapshot_digest, "position_snapshot_digest"
        ),
        "broker_identity_sha256": _sha256_digest(
            broker_identity_digest, "broker_identity_digest"
        ),
        "compliance_policy_sha256": _sha256_digest(
            compliance_policy_digest, "compliance_policy_digest"
        ),
        "projected_participation": float(projected_participation),
        "projected_gross_notional_multiple": float(
            projected_gross_notional_multiple
        ),
        "projected_margin_fraction": float(projected_margin_fraction),
        "key_id": str(key_id),
        "signature_sha256": "0" * 64,
    }
    unsigned = CertifiedIntent(**values).unsigned_payload()
    signature = hmac.new(
        signing_key,
        _canonical(unsigned).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return CertifiedIntent(**{**values, "signature_sha256": signature})


@dataclass(frozen=True)
class BrokerEvent:
    """Normalized broker acknowledgement, fill, rejection, or cancellation."""

    event_id: str
    event_type: str
    order_id: str
    broker_order_id: str
    timestamp: pd.Timestamp
    fill_quantity: int = 0
    fill_price: float | None = None
    fees_usd: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.event_type not in {"ACK", "FILL", "REJECT", "CANCEL"}:
            raise ValueError("unsupported broker event type")
        for name in ("event_id", "order_id", "broker_order_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _utc(self.timestamp, "broker event timestamp")
        if (
            isinstance(self.fill_quantity, bool)
            or not isinstance(self.fill_quantity, int)
            or self.fill_quantity < 0
        ):
            raise ValueError("fill_quantity must be a nonnegative integer")
        if self.event_type == "FILL":
            if self.fill_quantity < 1 or not _finite(self.fill_price):
                raise ValueError("fill events require positive quantity and finite price")
        elif self.fill_quantity != 0 or self.fill_price is not None:
            raise ValueError("non-fill events cannot contain fill economics")
        if not _finite(self.fees_usd) or self.fees_usd < 0:
            raise ValueError("fees_usd must be finite and nonnegative")

    def payload(self) -> dict[str, Any]:
        values = asdict(self)
        values["timestamp"] = _utc(self.timestamp, "timestamp").isoformat()
        return values


@runtime_checkable
class BrokerAdapter(Protocol):
    """Minimal broker contract required by :class:`ExecutionService`."""

    def is_connected(self) -> bool: ...

    def submit_order(self, order: OrderRequest, *, timestamp: Any = None) -> BrokerEvent: ...

    def cancel_order(self, order_id: str, *, timestamp: Any) -> BrokerEvent: ...

    def positions(self) -> Mapping[str, int]: ...

    def cash_balance(self) -> float: ...

    def open_orders(self) -> Mapping[str, int]: ...


@runtime_checkable
class DeploymentBoundBrokerAdapter(BrokerAdapter, Protocol):
    """Broker adapter that can attest its selected deployment and account."""

    def deployment_identity(self) -> BrokerDeploymentIdentity: ...


def _read_broker_deployment_identity(
    broker: BrokerAdapter,
) -> BrokerDeploymentIdentity:
    if not isinstance(broker, DeploymentBoundBrokerAdapter):
        raise OperationsError(
            "live broker does not expose a deployment identity"
        )
    try:
        identity = broker.deployment_identity()
    except Exception as exc:
        raise OperationsError(
            f"broker deployment identity cannot be read: {exc}"
        ) from exc
    if not isinstance(identity, BrokerDeploymentIdentity):
        raise OperationsError(
            "broker returned an invalid deployment identity"
        )
    return identity


def _broker_identity_blockers(
    identity: BrokerDeploymentIdentity,
    *,
    certified_identity_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if identity.environment != "production":
        reasons.append("live broker environment is not production")
    if not hmac.compare_digest(identity.sha256, certified_identity_sha256):
        reasons.append(
            "runtime broker identity does not match the certified deployment"
        )
    return reasons


class HashChainedJournal:
    """Append-only JSONL journal with idempotency keys and a SHA-256 chain."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous_hash = "0" * 64
        keys: set[str] = set()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise OperationsError(f"journal cannot be read: {exc}") from exc
        for expected_sequence, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OperationsError("journal contains invalid JSON") from exc
            required = {
                "sequence",
                "timestamp",
                "event_type",
                "idempotency_key",
                "payload",
                "previous_hash",
                "record_hash",
            }
            if not isinstance(record, dict) or not required.issubset(record):
                raise OperationsError("journal record is malformed")
            if record["sequence"] != expected_sequence:
                raise OperationsError("journal sequence is not contiguous")
            _utc(record["timestamp"], "journal timestamp")
            if record["previous_hash"] != previous_hash:
                raise OperationsError("journal hash chain is broken")
            key = record["idempotency_key"]
            if not isinstance(key, str) or not key or key in keys:
                raise OperationsError("journal idempotency key is blank or duplicated")
            unsigned = {key_: value for key_, value in record.items() if key_ != "record_hash"}
            expected_hash = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
            if record["record_hash"] != expected_hash:
                raise OperationsError("journal record hash does not match")
            previous_hash = expected_hash
            keys.add(key)
            records.append(record)
        return records

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        timestamp: Any,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        event_time = _utc(timestamp, "journal timestamp")
        normalized_payload = _json_value(dict(payload))
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        try:
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                # One process owns read/check/sequence assignment/append/fsync as
                # a single critical section.  This is still a single-writer
                # local journal, not a substitute for a transactional OMS DB.
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    records = self.read()
                    existing = [
                        record
                        for record in records
                        if record["idempotency_key"] == idempotency_key
                    ]
                    if existing:
                        record = existing[0]
                        if (
                            record["event_type"] != event_type
                            or record["payload"] != normalized_payload
                        ):
                            raise OperationsError(
                                "idempotency key was reused with different content"
                            )
                        return record
                    unsigned = {
                        "sequence": len(records) + 1,
                        "timestamp": event_time.isoformat(),
                        "event_type": event_type,
                        "idempotency_key": idempotency_key,
                        "payload": normalized_payload,
                        "previous_hash": (
                            records[-1]["record_hash"] if records else "0" * 64
                        ),
                    }
                    record = {
                        **unsigned,
                        "record_hash": hashlib.sha256(
                            _canonical(unsigned).encode("utf-8")
                        ).hexdigest(),
                    }
                    encoded = _canonical(record) + "\n"
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    # Re-read after fsync so a partial/corrupt append fails now.
                    self.read()
                    return record
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise OperationsError(f"journal append failed: {exc}") from exc


class PaperBroker:
    """Deterministic idempotent broker used for prospective shadow trading."""

    def __init__(self, *, initial_cash: float = 0.0) -> None:
        if not _finite(initial_cash):
            raise ValueError("initial_cash must be finite")
        self._connected = True
        self._initial_cash = float(initial_cash)
        self._cash = float(initial_cash)
        self._orders: dict[str, OrderRequest] = {}
        self._acks: dict[str, BrokerEvent] = {}
        self._events: dict[str, BrokerEvent] = {}
        self._filled: dict[str, int] = {}
        self._positions: dict[str, int] = {}
        self._terminal: dict[str, str] = {}

    def set_connected(self, connected: bool) -> None:
        if type(connected) is not bool:
            raise ValueError("connected must be a boolean")
        self._connected = connected

    def deployment_identity(self) -> BrokerDeploymentIdentity:
        """Identify the reference adapter explicitly as non-production."""

        return BrokerDeploymentIdentity(
            broker_id="delta1-in-memory-paper",
            adapter_id="delta1_strategy.execution.PaperBroker",
            adapter_build_sha256=hashlib.sha256(
                b"delta1_strategy.execution.PaperBroker:v1"
            ).hexdigest(),
            account_id="in-memory-paper-account",
            environment="paper",
        )

    def is_connected(self) -> bool:
        return self._connected

    def submit_order(self, order: OrderRequest, *, timestamp: Any = None) -> BrokerEvent:
        if not self._connected:
            raise OperationsError("paper broker is disconnected")
        if not isinstance(order, OrderRequest):
            raise TypeError("order must be an OrderRequest")
        existing = self._orders.get(order.order_id)
        if existing is not None:
            if existing != order:
                raise OperationsError("broker order ID reused with different content")
            return self._acks[order.order_id]
        broker_order_id = "paper-" + hashlib.sha256(
            _canonical(order.payload()).encode("utf-8")
        ).hexdigest()[:20]
        ack = BrokerEvent(
            event_id=f"ack-{broker_order_id}",
            event_type="ACK",
            order_id=order.order_id,
            broker_order_id=broker_order_id,
            timestamp=(
                _utc(order.decision_timestamp, "decision_timestamp")
                if timestamp is None
                else _utc(timestamp, "broker acknowledgement timestamp")
            ),
        )
        self._orders[order.order_id] = order
        self._acks[order.order_id] = ack
        self._events[ack.event_id] = ack
        self._filled[order.order_id] = 0
        return ack

    def fill_order(
        self,
        order_id: str,
        *,
        fill_id: str,
        quantity: int,
        price: float,
        fees_usd: float,
        timestamp: Any,
    ) -> BrokerEvent:
        if not self._connected:
            raise OperationsError("paper broker is disconnected")
        if fill_id in self._events:
            event = self._events[fill_id]
            if event.order_id != order_id:
                raise OperationsError("fill ID reused for another order")
            return event
        if order_id not in self._orders:
            raise OperationsError("cannot fill an unknown order")
        if order_id in self._terminal:
            raise OperationsError("cannot fill a terminal order")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError("fill quantity must be a positive integer")
        remaining = self._orders[order_id].quantity - self._filled[order_id]
        if quantity > remaining:
            raise OperationsError("fill would overfill the order")
        event = BrokerEvent(
            event_id=fill_id,
            event_type="FILL",
            order_id=order_id,
            broker_order_id=self._acks[order_id].broker_order_id,
            timestamp=_utc(timestamp, "fill timestamp"),
            fill_quantity=quantity,
            fill_price=float(price),
            fees_usd=float(fees_usd),
        )
        order = self._orders[order_id]
        signed = quantity if order.side == "BUY" else -quantity
        self._positions[order.contract_id] = self._positions.get(order.contract_id, 0) + signed
        self._cash -= float(fees_usd)
        self._filled[order_id] += quantity
        if self._filled[order_id] == order.quantity:
            self._terminal[order_id] = "FILLED"
        self._events[fill_id] = event
        return event

    def reject_order(self, order_id: str, *, reason: str, timestamp: Any) -> BrokerEvent:
        if order_id not in self._orders:
            raise OperationsError("cannot reject an unknown order")
        if order_id in self._terminal:
            raise OperationsError("order is already terminal")
        event = BrokerEvent(
            event_id=f"reject-{self._acks[order_id].broker_order_id}",
            event_type="REJECT",
            order_id=order_id,
            broker_order_id=self._acks[order_id].broker_order_id,
            timestamp=_utc(timestamp, "reject timestamp"),
            reason=str(reason),
        )
        self._terminal[order_id] = "REJECTED"
        self._events[event.event_id] = event
        return event

    def cancel_order(self, order_id: str, *, timestamp: Any) -> BrokerEvent:
        if order_id not in self._orders:
            raise OperationsError("cannot cancel an unknown order")
        if order_id in self._terminal:
            terminal = self._terminal[order_id]
            if terminal == "CANCELLED":
                return next(
                    event
                    for event in self._events.values()
                    if event.order_id == order_id and event.event_type == "CANCEL"
                )
            raise OperationsError("order is already terminal")
        event = BrokerEvent(
            event_id=f"cancel-{self._acks[order_id].broker_order_id}",
            event_type="CANCEL",
            order_id=order_id,
            broker_order_id=self._acks[order_id].broker_order_id,
            timestamp=_utc(timestamp, "cancel timestamp"),
        )
        self._terminal[order_id] = "CANCELLED"
        self._events[event.event_id] = event
        return event

    def positions(self) -> Mapping[str, int]:
        return dict(self._positions)

    def cash_balance(self) -> float:
        return self._cash

    def open_orders(self) -> Mapping[str, int]:
        return {
            order_id: order.quantity - self._filled[order_id]
            for order_id, order in self._orders.items()
            if order_id not in self._terminal
        }


@dataclass(frozen=True)
class KillSwitchState:
    status: str
    updated_at: str
    reason: str
    actor: str
    approver: str | None


class PersistentKillSwitch:
    """Fail-closed kill switch that remains latched across process restarts."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, state: KillSwitchState) -> KillSwitchState:
        payload = asdict(state)
        wrapper = {
            "state": payload,
            "sha256": hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest(),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(_canonical(wrapper) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError as exc:
            raise OperationsError(f"kill-switch state cannot be persisted: {exc}") from exc
        return self.read()

    def read(self) -> KillSwitchState:
        if not self.path.exists():
            return KillSwitchState(
                status="HALTED",
                updated_at=pd.Timestamp.now(tz="UTC").isoformat(),
                reason="kill switch has not been commissioned",
                actor="SYSTEM",
                approver=None,
            )
        try:
            wrapper = json.loads(self.path.read_text(encoding="utf-8"))
            payload = wrapper["state"]
            observed = wrapper["sha256"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OperationsError("kill-switch state is malformed") from exc
        expected = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        if observed != expected:
            raise OperationsError("kill-switch state checksum does not match")
        try:
            state = KillSwitchState(**payload)
        except TypeError as exc:
            raise OperationsError("kill-switch state fields are invalid") from exc
        if state.status not in {"ACTIVE", "HALTED"}:
            raise OperationsError("kill-switch state has an invalid status")
        _utc(state.updated_at, "kill-switch updated_at")
        return state

    def commission(self, *, requested_by: str, approved_by: str, timestamp: Any) -> KillSwitchState:
        return self._dual_control_reset(
            requested_by=requested_by,
            approved_by=approved_by,
            timestamp=timestamp,
            reason="kill switch commissioned after test",
        )

    def trigger(self, *, actor: str, reason: str, timestamp: Any) -> KillSwitchState:
        if not str(actor).strip() or not str(reason).strip():
            raise ValueError("actor and reason must be non-empty")
        return self._write(
            KillSwitchState(
                status="HALTED",
                updated_at=_utc(timestamp, "kill-switch timestamp").isoformat(),
                reason=str(reason).strip(),
                actor=str(actor).strip(),
                approver=None,
            )
        )

    def reset(
        self,
        *,
        requested_by: str,
        approved_by: str,
        reason: str,
        timestamp: Any,
    ) -> KillSwitchState:
        if self.read().status != "HALTED":
            raise OperationsError("kill switch is not halted")
        return self._dual_control_reset(
            requested_by=requested_by,
            approved_by=approved_by,
            timestamp=timestamp,
            reason=reason,
        )

    def _dual_control_reset(
        self,
        *,
        requested_by: str,
        approved_by: str,
        timestamp: Any,
        reason: str,
    ) -> KillSwitchState:
        requester = str(requested_by).strip()
        approver = str(approved_by).strip()
        if not requester or not approver or not str(reason).strip():
            raise ValueError("requester, approver, and reason must be non-empty")
        if requester.casefold() == approver.casefold():
            raise OperationsError("kill-switch reset requires independent approval")
        return self._write(
            KillSwitchState(
                status="ACTIVE",
                updated_at=_utc(timestamp, "kill-switch timestamp").isoformat(),
                reason=str(reason).strip(),
                actor=requester,
                approver=approver,
            )
        )


def _mapping(values: Mapping[Any, Any], label: str) -> pd.Series:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    series = pd.Series(dict(values), dtype=float)
    series.index = series.index.map(str)
    if series.index.has_duplicates or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{label} is duplicated or non-finite")
    return series


def reconciliation_report(
    internal_positions: Mapping[Any, Any],
    broker_positions: Mapping[Any, Any],
    *,
    internal_cash: float,
    broker_cash: float,
    cash_tolerance: float = 0.01,
) -> pd.DataFrame:
    """Reconcile contract positions and cash; any unresolved break blocks risk."""

    if not _finite(internal_cash) or not _finite(broker_cash):
        raise ValueError("cash balances must be finite")
    if not _finite(cash_tolerance) or cash_tolerance < 0:
        raise ValueError("cash_tolerance must be finite and nonnegative")
    internal = _mapping(internal_positions, "internal_positions")
    broker = _mapping(broker_positions, "broker_positions")
    symbols = sorted(set(internal.index).union(broker.index))
    rows = []
    for symbol in symbols:
        expected = float(internal.get(symbol, 0.0))
        observed = float(broker.get(symbol, 0.0))
        difference = observed - expected
        rows.append(
            {
                "item": f"position:{symbol}",
                "category": "position",
                "expected": expected,
                "observed": observed,
                "difference": difference,
                "status": PASS if difference == 0 else BLOCKED,
                "reason": "reconciled" if difference == 0 else "position break",
            }
        )
    cash_difference = float(broker_cash) - float(internal_cash)
    rows.append(
        {
            "item": "cash:USD",
            "category": "cash",
            "expected": float(internal_cash),
            "observed": float(broker_cash),
            "difference": cash_difference,
            "status": PASS if abs(cash_difference) <= cash_tolerance else BLOCKED,
            "reason": (
                "reconciled"
                if abs(cash_difference) <= cash_tolerance
                else "cash break exceeds tolerance"
            ),
        }
    )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class MonitoringPolicy:
    market_data_max_age_seconds: float = 300.0
    broker_max_age_seconds: float = 60.0
    reconciliation_max_age_seconds: float = 86_400.0
    journal_max_age_seconds: float = 900.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not _finite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def monitoring_report(
    heartbeats: Mapping[str, Any],
    *,
    as_of: Any,
    policy: MonitoringPolicy = MonitoringPolicy(),
) -> pd.DataFrame:
    """Evaluate required service heartbeats using a caller-supplied clock."""

    if not isinstance(policy, MonitoringPolicy):
        raise TypeError("policy must be a MonitoringPolicy")
    if not isinstance(heartbeats, Mapping):
        raise TypeError("heartbeats must be a mapping")
    now = _utc(as_of, "as_of")
    limits = {
        "market_data": policy.market_data_max_age_seconds,
        "broker": policy.broker_max_age_seconds,
        "reconciliation": policy.reconciliation_max_age_seconds,
        "journal": policy.journal_max_age_seconds,
    }
    rows = []
    for component, maximum_age in limits.items():
        value = heartbeats.get(component)
        reasons: list[str] = []
        age: float | None = None
        try:
            heartbeat = _utc(value, f"{component} heartbeat")
        except OperationsError as exc:
            reasons.append(str(exc))
        else:
            age = float((now - heartbeat).total_seconds())
            if age < 0:
                reasons.append("heartbeat is in the future")
            elif age > maximum_age:
                reasons.append("heartbeat is stale")
        rows.append(
            {
                "component": component,
                "status": BLOCKED if reasons else PASS,
                "age_seconds": age,
                "max_age_seconds": maximum_age,
                "reason": "; ".join(reasons) if reasons else "heartbeat is healthy",
            }
        )
    return pd.DataFrame(rows)


def replay_order_states(journal: HashChainedJournal) -> pd.DataFrame:
    """Rebuild OMS state from the verified event journal after a restart."""

    if not isinstance(journal, HashChainedJournal):
        raise TypeError("journal must be a HashChainedJournal")
    states: dict[str, dict[str, Any]] = {}
    event_ids: set[str] = set()
    for record in journal.read():
        event_type = record["event_type"]
        payload = record["payload"]
        if event_type == "ORDER_OUTBOX":
            order_id = payload["order_id"]
            if order_id in states:
                raise OperationsError("duplicate outbox order")
            states[order_id] = {
                "order_id": order_id,
                "contract_id": payload["contract_id"],
                "requested_quantity": int(payload["quantity"]),
                "filled_quantity": 0,
                "remaining_quantity": int(payload["quantity"]),
                "status": "PERSISTED",
            }
        elif event_type.startswith("BROKER_"):
            order_id = payload.get("order_id")
            if order_id not in states:
                raise OperationsError("broker event precedes persisted order")
            event_id = payload.get("event_id")
            if event_id in event_ids:
                raise OperationsError("duplicate broker event")
            event_ids.add(event_id)
            event_name = payload.get("event_type")
            if states[order_id]["status"] in {"FILLED", "REJECTED", "CANCELLED"}:
                raise OperationsError("broker event follows a terminal event")
            if event_name == "ACK":
                if states[order_id]["status"] != "PERSISTED":
                    raise OperationsError("acknowledgement is out of sequence")
                states[order_id]["status"] = "ACKNOWLEDGED"
            elif event_name == "FILL":
                if states[order_id]["status"] not in {"ACKNOWLEDGED", "PARTIAL"}:
                    raise OperationsError("fill precedes broker acknowledgement")
                states[order_id]["filled_quantity"] += int(payload["fill_quantity"])
                requested = states[order_id]["requested_quantity"]
                filled = states[order_id]["filled_quantity"]
                if filled > requested:
                    raise OperationsError("journal replay detected an overfill")
                states[order_id]["remaining_quantity"] = requested - filled
                states[order_id]["status"] = "FILLED" if filled == requested else "PARTIAL"
            elif event_name in {"REJECT", "CANCEL"}:
                states[order_id]["status"] = (
                    "REJECTED" if event_name == "REJECT" else "CANCELLED"
                )
            else:
                raise OperationsError("journal contains an unknown broker event")
    return pd.DataFrame(states.values()).sort_values("order_id").reset_index(drop=True)


def _runtime_health_blockers(
    health: Any,
    *,
    route_time: pd.Timestamp,
    limits: ProductionLimits,
) -> list[str]:
    """Validate a freshly evaluated health object at the routing boundary."""

    if not isinstance(health, RuntimeHealth):
        return ["runtime health provider returned an invalid object"]
    reasons: list[str] = []
    if health.healthy is not True or health.status != PASS:
        reasons.append("runtime health is not PASS")
    if health.reasons:
        reasons.append("runtime health contains blocking reasons")
    for label, value in (
        ("broker connection", health.broker_connected),
        ("broker reconciliation", health.broker_reconciled),
        ("monitoring", health.monitoring_healthy),
        ("kill-switch readiness", health.kill_switch_ready),
    ):
        if value is not True:
            reasons.append(f"runtime {label} is not healthy")

    try:
        evaluated_at = _utc(health.evaluated_at, "runtime health evaluated_at")
        data_timestamp = _utc(health.data_timestamp, "runtime health data_timestamp")
        reconciled_at = _utc(health.reconciled_at, "runtime health reconciled_at")
        monitoring_at = _utc(
            health.monitoring_evaluated_at,
            "runtime health monitoring_evaluated_at",
        )
        kill_switch_at = _utc(
            health.kill_switch_checked_at,
            "runtime health kill_switch_checked_at",
        )
    except OperationsError as exc:
        reasons.append(str(exc))
    else:
        evaluation_age = float((route_time - evaluated_at).total_seconds())
        data_age = float((route_time - data_timestamp).total_seconds())
        if evaluation_age < 0:
            reasons.append("runtime health evaluation is in the future")
        elif evaluation_age > limits.max_runtime_health_age_seconds:
            reasons.append("runtime health evaluation is stale")
        if data_age < 0:
            reasons.append("runtime market data is in the future")
        elif data_age > limits.max_data_age_seconds:
            reasons.append("runtime market data is stale")
        ordered = (
            data_timestamp <= reconciled_at <= monitoring_at <= kill_switch_at <= evaluated_at
        )
        if not ordered:
            reasons.append("runtime evidence timestamps are out of order")
        reconciliation_age = float((route_time - reconciled_at).total_seconds())
        if reconciliation_age < 0:
            reasons.append("runtime reconciliation is in the future")
        elif reconciliation_age > limits.max_reconciliation_age_seconds:
            reasons.append("runtime reconciliation is stale")
        for label, timestamp in (
            ("monitoring", monitoring_at),
            ("kill-switch check", kill_switch_at),
        ):
            age = float((route_time - timestamp).total_seconds())
            if age < 0:
                reasons.append(f"runtime {label} is in the future")
            elif age > limits.max_runtime_health_age_seconds:
                reasons.append(f"runtime {label} is stale")

    try:
        _sha256_digest(
            health.position_snapshot_sha256,
            "runtime position_snapshot_sha256",
        )
    except ValueError as exc:
        reasons.append(str(exc))

    risk_values = (
        ("gross notional", health.gross_notional_multiple, limits.max_gross_notional_multiple),
        ("margin", health.margin_fraction, limits.max_margin_fraction),
        ("drawdown", health.drawdown_fraction, limits.max_drawdown_fraction),
    )
    for label, value, limit in risk_values:
        if not _finite(value) or float(value) < 0:
            reasons.append(f"runtime {label} is missing or invalid")
        elif float(value) > limit:
            reasons.append(f"runtime {label} limit is breached")
    absolute_risk_values = (
        ("NAV", health.nav_usd, True),
        ("peak NAV", health.peak_nav_usd, True),
        ("gross notional USD", health.gross_notional_usd, False),
        ("margin requirement USD", health.margin_requirement_usd, False),
    )
    for label, value, strictly_positive in absolute_risk_values:
        if (
            not _finite(value)
            or (strictly_positive and float(value) <= 0)
            or (not strictly_positive and float(value) < 0)
        ):
            reasons.append(f"runtime {label} is missing or invalid")
    return list(dict.fromkeys(reasons))


def _portfolio_compliance_blockers(
    decision: Any,
    *,
    expected_request_sha256: str,
    expected_policy_sha256: str,
    route_time: pd.Timestamp,
    limits: ProductionLimits,
) -> list[str]:
    """Validate a fresh external decision against the exact live request."""

    if not isinstance(decision, PortfolioComplianceDecision):
        return ["portfolio/compliance provider returned an invalid decision"]
    reasons: list[str] = []
    if not hmac.compare_digest(
        decision.request_sha256,
        expected_request_sha256,
    ):
        reasons.append(
            "portfolio/compliance decision does not match the live request"
        )
    if not hmac.compare_digest(
        decision.policy_sha256,
        expected_policy_sha256,
    ):
        reasons.append(
            "portfolio/compliance policy does not match the certified deployment"
        )
    try:
        evaluated_at = _utc(
            decision.evaluated_at,
            "portfolio compliance evaluated_at",
        )
    except OperationsError as exc:
        reasons.append(str(exc))
    else:
        age = float((route_time - evaluated_at).total_seconds())
        if age < 0:
            reasons.append("portfolio/compliance decision is timestamped in the future")
        elif age > limits.max_runtime_health_age_seconds:
            reasons.append("portfolio/compliance decision is stale")
    if decision.approved is not True:
        reason = "; ".join(decision.reasons) or "no reason supplied"
        reasons.append("portfolio/compliance decision denied: " + reason)
    return list(dict.fromkeys(reasons))


def _verified_reduce_only(
    order: OrderRequest,
    positions: Mapping[str, int],
) -> tuple[bool, int | None]:
    """Recompute reduction from broker positions; never trust the order flag."""

    value = positions.get(order.contract_id, 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return False, None
    current = int(value)
    signed_quantity = order.quantity if order.side == "BUY" else -order.quantity
    projected = current + signed_quantity
    if current == 0:
        return False, projected
    same_side = projected == 0 or math.copysign(1, projected) == math.copysign(1, current)
    return bool(same_side and abs(projected) < abs(current)), projected


def _liquidation_deadline(row: pd.Series) -> pd.Timestamp:
    return _utc(row["broker_liquidation_cutoff"], "broker_liquidation_cutoff")


@dataclass(frozen=True)
class OrderBatchAssessment:
    """Deterministic route-time serial and portfolio-risk calculation."""

    serial_snapshot_sha256: str
    position_snapshot_sha256: str
    current_gross_notional_usd: float
    current_margin_requirement_usd: float
    projected_gross_notional_usd: float
    projected_margin_requirement_usd: float
    projected_gross_notional_multiple: float
    projected_margin_fraction: float
    participation_by_order: Mapping[str, float]
    verified_reduce_only_by_order: Mapping[str, bool]
    blockers_by_order: Mapping[str, tuple[str, ...]]

    @property
    def passed(self) -> bool:
        return not any(self.blockers_by_order.values())


def assess_order_batch(
    orders: list[OrderRequest],
    *,
    broker_positions: Mapping[str, int],
    serial_snapshot: pd.DataFrame,
    nav_usd: float,
    as_of: Any,
    limits: ProductionLimits = ProductionLimits(),
) -> OrderBatchAssessment:
    """Recompute expiry, participation, gross and margin controls as one batch.

    ``serial_snapshot`` must already have passed
    :func:`contracts.validated_serial_snapshot`.  This function is public so an
    upstream risk service can create an exact certificate from the same
    calculation; :class:`ExecutionService` always reruns it independently.
    """

    if not isinstance(limits, ProductionLimits):
        raise TypeError("limits must be a ProductionLimits instance")
    if not orders or any(not isinstance(order, OrderRequest) for order in orders):
        raise TypeError("orders must be a non-empty list of OrderRequest objects")
    route_time = _utc(as_of, "pre-trade as_of")
    if not _finite(nav_usd) or float(nav_usd) <= 0:
        raise OperationsError("pre-trade NAV must be finite and positive")
    missing = [column for column in SERIAL_CONTRACT_COLUMNS if column not in serial_snapshot]
    if missing:
        raise OperationsError(
            "validated serial snapshot is missing columns: " + ", ".join(missing)
        )
    if serial_snapshot["contract_id"].duplicated().any():
        raise OperationsError("validated serial snapshot has duplicate contract IDs")
    lookup = serial_snapshot.set_index("contract_id", drop=False)

    positions: dict[str, int] = {}
    for raw_contract, raw_quantity in broker_positions.items():
        contract = str(raw_contract)
        if (
            isinstance(raw_quantity, bool)
            or not isinstance(raw_quantity, (int, np.integer))
        ):
            raise OperationsError("broker positions contain a non-integer quantity")
        positions[contract] = int(raw_quantity)
    position_digest = position_snapshot_sha256(positions)
    snapshot_digest = serial_snapshot_sha256(serial_snapshot)

    blockers: dict[str, list[str]] = {order.order_id: [] for order in orders}
    ids = [order.order_id for order in orders]
    if len(set(ids)) != len(ids):
        for order in orders:
            blockers[order.order_id].append("batch contains a duplicate order_id")
    contracts = [order.contract_id for order in orders]
    if len(set(contracts)) != len(contracts):
        for order in orders:
            blockers[order.order_id].append(
                "batch contains multiple orders for one contract"
            )

    participation: dict[str, float] = {}
    reductions: dict[str, bool] = {}
    projected_positions = dict(positions)
    safety_days = SerialDataLimits().roll_calendar_days_before_deadline
    safety_time = route_time + pd.Timedelta(days=safety_days)
    for order in orders:
        order_blockers = blockers[order.order_id]
        decision_time = _utc(order.decision_timestamp, "order decision_timestamp")
        decision_age = float((route_time - decision_time).total_seconds())
        if decision_age < 0:
            order_blockers.append("order decision timestamp is in the future")
        elif decision_age > limits.max_data_age_seconds:
            order_blockers.append("order decision is stale")

        if order.contract_id not in lookup.index:
            order_blockers.append("contract is absent from the validated serial snapshot")
            participation[order.order_id] = math.inf
            reductions[order.order_id] = False
            continue
        row = lookup.loc[order.contract_id]
        if str(row["root_symbol"]) != order.root_symbol:
            order_blockers.append("contract root does not match the order root")
        volume = float(row["volume"])
        projected_participation = order.quantity / volume if volume > 0 else math.inf
        participation[order.order_id] = projected_participation
        if (
            not math.isfinite(projected_participation)
            or projected_participation > limits.max_order_participation
        ):
            order_blockers.append("order participation exceeds the live limit")

        if order.order_type == "MARKETABLE_LIMIT":
            if order.side == "BUY" and float(order.limit_price) < float(row["ask"]):
                order_blockers.append("buy marketable-limit price is below the current ask")
            if order.side == "SELL" and float(order.limit_price) > float(row["bid"]):
                order_blockers.append("sell marketable-limit price is above the current bid")

        verified_reduction, _ = _verified_reduce_only(order, positions)
        reductions[order.order_id] = bool(order.reduce_only and verified_reduction)
        if order.reduce_only and not verified_reduction:
            order_blockers.append(
                "reduce-only flag does not reduce the broker position without crossing"
            )
        if not reductions[order.order_id]:
            deadline = _liquidation_deadline(row)
            if deadline < safety_time:
                order_blockers.append(
                    "risk-increasing contract is inside the delivery safety horizon"
                )

        signed = order.quantity if order.side == "BUY" else -order.quantity
        projected_positions[order.contract_id] = (
            projected_positions.get(order.contract_id, 0) + signed
        )

    # Every held or projected contract must have exact USD notional and margin
    # values in the same validated point-in-time snapshot.
    missing_position_contracts = sorted(
        contract
        for contract in set(positions).union(projected_positions)
        if (
            (positions.get(contract, 0) != 0 or projected_positions.get(contract, 0) != 0)
            and contract not in lookup.index
        )
    )
    if missing_position_contracts:
        reason = (
            "broker/projected positions are absent from the serial snapshot: "
            + ", ".join(missing_position_contracts)
        )
        for order in orders:
            blockers[order.order_id].append(reason)

    def portfolio_values(values: Mapping[str, int]) -> tuple[float, float]:
        gross = 0.0
        margin = 0.0
        for contract, quantity in values.items():
            if quantity == 0 or contract not in lookup.index:
                continue
            row = lookup.loc[contract]
            gross += abs(quantity) * float(row["notional_per_contract_usd"])
            margin += abs(quantity) * float(row["margin_per_contract_usd"])
        return gross, margin

    current_gross, current_margin = portfolio_values(positions)
    projected_gross, projected_margin = portfolio_values(projected_positions)
    gross_multiple = projected_gross / float(nav_usd)
    margin_fraction = projected_margin / float(nav_usd)
    if gross_multiple > limits.max_gross_notional_multiple:
        for order in orders:
            if not reductions.get(order.order_id, False):
                blockers[order.order_id].append(
                    "projected batch gross notional exceeds the live limit"
                )
    if margin_fraction > limits.max_margin_fraction:
        for order in orders:
            if not reductions.get(order.order_id, False):
                blockers[order.order_id].append(
                    "projected batch margin exceeds the live limit"
                )

    # A roll certificate represents an inseparable close/open pair.  Routing an
    # opening leg alone or into a less-safe contract is blocked.
    roll_groups: dict[str, list[OrderRequest]] = {}
    for order in orders:
        if order.roll_id is not None:
            roll_groups.setdefault(order.roll_id, []).append(order)
    for roll_id, group in roll_groups.items():
        roll_errors: list[str] = []
        if len(group) != 2:
            roll_errors.append("roll group must contain exactly two legs")
        elif (
            group[0].root_symbol != group[1].root_symbol
            or group[0].contract_id == group[1].contract_id
            or group[0].quantity != group[1].quantity
            or group[0].side == group[1].side
        ):
            roll_errors.append("roll legs do not form a matched root/quantity pair")
        else:
            reducing = [order for order in group if reductions.get(order.order_id, False)]
            opening = [order for order in group if not reductions.get(order.order_id, False)]
            if len(reducing) != 1 or len(opening) != 1:
                roll_errors.append("roll group must contain one close and one opening leg")
            else:
                source = lookup.loc[reducing[0].contract_id]
                destination = lookup.loc[opening[0].contract_id]
                if (
                    str(source["exchange"]) != str(destination["exchange"])
                    or str(source["currency"]) != str(destination["currency"])
                    or _liquidation_deadline(destination)
                    <= _liquidation_deadline(source)
                ):
                    roll_errors.append(
                        "roll destination is not a compatible later delivery-safe contract"
                    )
        if roll_errors:
            for order in group:
                blockers[order.order_id].extend(roll_errors)

    return OrderBatchAssessment(
        serial_snapshot_sha256=snapshot_digest,
        position_snapshot_sha256=position_digest,
        current_gross_notional_usd=current_gross,
        current_margin_requirement_usd=current_margin,
        projected_gross_notional_usd=projected_gross,
        projected_margin_requirement_usd=projected_margin,
        projected_gross_notional_multiple=gross_multiple,
        projected_margin_fraction=margin_fraction,
        participation_by_order=dict(participation),
        verified_reduce_only_by_order=dict(reductions),
        blockers_by_order={
            order_id: tuple(dict.fromkeys(values))
            for order_id, values in blockers.items()
        },
    )


class ExecutionService:
    """Route approved intents through persistent, idempotent broker submission."""

    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        journal: HashChainedJournal,
        kill_switch: PersistentKillSwitch,
        mode: str = "paper",
        readiness_check: ArtifactBackedReadinessCheck | None = None,
        runtime_health_check: RuntimeHealthCheck | None = None,
        serial_snapshot_provider: SerialSnapshotProvider | None = None,
        portfolio_compliance_check: PortfolioComplianceCheck | None = None,
        intent_verification_keys: Mapping[str, bytes] | None = None,
        limits: ProductionLimits = ProductionLimits(),
    ) -> None:
        if mode not in {"paper", "live"}:
            raise ValueError("mode must be paper or live")
        if not isinstance(broker, BrokerAdapter):
            raise TypeError("broker does not implement BrokerAdapter")
        if not isinstance(limits, ProductionLimits):
            raise TypeError("limits must be a ProductionLimits instance")
        if mode == "live" and not isinstance(
            readiness_check, ArtifactBackedReadinessCheck
        ):
            raise TypeError(
                "live mode requires an ArtifactBackedReadinessCheck"
            )
        if runtime_health_check is not None and not callable(runtime_health_check):
            raise TypeError("runtime_health_check must be callable")
        if mode == "live" and runtime_health_check is None:
            raise TypeError("live mode requires a runtime health check")
        if serial_snapshot_provider is not None and not callable(serial_snapshot_provider):
            raise TypeError("serial_snapshot_provider must be callable")
        if mode == "live" and serial_snapshot_provider is None:
            raise TypeError("live mode requires a serial snapshot provider")
        if portfolio_compliance_check is not None and not callable(
            portfolio_compliance_check
        ):
            raise TypeError("portfolio_compliance_check must be callable")
        if mode == "live" and portfolio_compliance_check is None:
            raise TypeError(
                "live mode requires a portfolio/compliance check"
            )
        keys = dict(intent_verification_keys or {})
        for key_id, key in keys.items():
            if not isinstance(key_id, str) or not key_id.strip():
                raise ValueError("intent verification key IDs must be non-empty")
            if not isinstance(key, bytes) or len(key) < 32:
                raise ValueError("intent verification keys must contain at least 32 bytes")
        if mode == "live" and not keys:
            raise TypeError("live mode requires intent verification keys")
        if mode == "live":
            identity = _read_broker_deployment_identity(broker)
            identity_reasons = _broker_identity_blockers(
                identity,
                certified_identity_sha256=(
                    readiness_check.certified_broker_identity_sha256
                ),
            )
            if identity_reasons:
                raise OperationsError("; ".join(identity_reasons))
        self.broker = broker
        self.journal = journal
        self.kill_switch = kill_switch
        self.mode = mode
        self.readiness_check = readiness_check
        self.runtime_health_check = runtime_health_check
        self.serial_snapshot_provider = serial_snapshot_provider
        self.portfolio_compliance_check = portfolio_compliance_check
        self.intent_verification_keys = keys
        self.limits = limits

    def route(
        self,
        orders: list[OrderRequest | CertifiedIntent],
        *,
        timestamp: Any,
    ) -> pd.DataFrame:
        """Route a batch only after controls are recomputed at the boundary.

        Live risk-increasing orders must be wrapped in an authenticated
        :class:`CertifiedIntent`.  Emergency reductions may be unsigned, but
        only when fresh broker positions prove that they reduce without
        crossing and the current serial snapshot is valid.
        """

        route_time = _utc(timestamp, "route timestamp")
        entries: list[tuple[OrderRequest, CertifiedIntent | None]] = []
        for value in orders:
            if isinstance(value, CertifiedIntent):
                entries.append((value.order, value))
            elif isinstance(value, OrderRequest):
                entries.append((value, None))
            else:
                raise TypeError(
                    "orders must contain OrderRequest or CertifiedIntent objects"
                )
        if not entries:
            return pd.DataFrame(
                columns=["order_id", "contract_id", "status", "broker_order_id", "reason"]
            )
        order_values = [order for order, _ in entries]
        order_ids = [order.order_id for order in order_values]
        if len(set(order_ids)) != len(order_ids):
            return pd.DataFrame(
                [
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": BLOCKED,
                        "broker_order_id": None,
                        "reason": "batch contains duplicate order IDs",
                    }
                    for order in order_values
                ]
            )

        switch = self.kill_switch.read()
        connected = self.broker.is_connected()
        broker_identity: BrokerDeploymentIdentity | None = None
        broker_identity_blockers: list[str] = []
        if self.mode == "live":
            try:
                broker_identity = _read_broker_deployment_identity(self.broker)
            except OperationsError as exc:
                broker_identity_blockers.append(str(exc))
            else:
                broker_identity_blockers.extend(
                    _broker_identity_blockers(
                        broker_identity,
                        certified_identity_sha256=(
                            self.readiness_check.certified_broker_identity_sha256
                        ),
                    )
                )
        readiness_status = BLOCKED
        readiness_failure = "artifact-backed production readiness is not READY"
        if self.mode != "live":
            readiness_status = READY
            readiness_failure = ""
        else:
            try:
                readiness_report = self.readiness_check(as_of=route_time)
                readiness_status = overall_readiness_status(readiness_report)
            except Exception as exc:
                readiness_failure = f"route-time readiness verification failed: {exc}"

        runtime_health: RuntimeHealth | None = None
        runtime_blockers = ["runtime health check is not configured"]
        if self.runtime_health_check is not None:
            try:
                runtime_health = self.runtime_health_check(route_time)
            except Exception as exc:
                runtime_blockers = [f"route-time runtime health check failed: {exc}"]
            else:
                runtime_blockers = _runtime_health_blockers(
                    runtime_health,
                    route_time=route_time,
                    limits=self.limits,
                )

        try:
            broker_positions = dict(self.broker.positions())
            open_orders = dict(self.broker.open_orders())
        except Exception as exc:
            broker_positions = {}
            open_orders = {"UNKNOWN": 1}
            position_failure = f"broker position/open-order query failed: {exc}"
        else:
            position_failure = ""
        preexisting_open_orders = bool(open_orders)

        try:
            observed_position_digest = position_snapshot_sha256(broker_positions)
        except Exception as exc:
            observed_position_digest = None
            position_failure = position_failure or f"broker position snapshot is invalid: {exc}"

        reconciliation_verified = False
        if (
            isinstance(runtime_health, RuntimeHealth)
            and runtime_health.broker_reconciled
            and observed_position_digest is not None
        ):
            try:
                reconciled_at = _utc(
                    runtime_health.reconciled_at,
                    "runtime health reconciled_at",
                )
            except OperationsError:
                pass
            else:
                reconciliation_age = float((route_time - reconciled_at).total_seconds())
                reconciliation_verified = bool(
                    0 <= reconciliation_age <= self.limits.max_reconciliation_age_seconds
                    and hmac.compare_digest(
                        str(runtime_health.position_snapshot_sha256 or ""),
                        observed_position_digest,
                    )
                )

        assessment: OrderBatchAssessment | None = None
        snapshot: pd.DataFrame | None = None
        nav_usd: float | None = None
        serial_failure = ""
        if self.serial_snapshot_provider is not None:
            try:
                raw_snapshot = self.serial_snapshot_provider(route_time)
                serial_limits = SerialDataLimits(
                    max_data_age_seconds=self.limits.max_data_age_seconds,
                    max_order_participation=self.limits.max_order_participation,
                )
                snapshot = validated_serial_snapshot(
                    raw_snapshot,
                    as_of=route_time,
                    limits=serial_limits,
                )
                nav_usd = (
                    runtime_health.nav_usd
                    if isinstance(runtime_health, RuntimeHealth)
                    else None
                )
                assessment = assess_order_batch(
                    order_values,
                    broker_positions=broker_positions,
                    serial_snapshot=snapshot,
                    nav_usd=nav_usd,
                    as_of=route_time,
                    limits=self.limits,
                )
            except (Exception, ContractDataError) as exc:
                serial_failure = f"route-time serial/pre-trade validation failed: {exc}"
        elif self.mode == "live":
            serial_failure = "route-time serial snapshot provider is not configured"

        risk_snapshot_failure = ""
        if assessment is not None and isinstance(runtime_health, RuntimeHealth):
            current_values = (
                (
                    assessment.current_gross_notional_usd,
                    runtime_health.gross_notional_usd,
                    "gross notional",
                ),
                (
                    assessment.current_margin_requirement_usd,
                    runtime_health.margin_requirement_usd,
                    "margin requirement",
                ),
            )
            mismatches = []
            for observed, reported, label in current_values:
                if not _finite(reported) or not math.isclose(
                    observed,
                    float(reported),
                    rel_tol=1e-9,
                    abs_tol=0.01,
                ):
                    mismatches.append(label)
            if mismatches:
                risk_snapshot_failure = (
                    "runtime health does not reconcile to serial-valued broker "
                    "positions: " + ", ".join(mismatches)
                )

        compliance_decision: PortfolioComplianceDecision | None = None
        compliance_request_sha256: str | None = None
        compliance_blockers: list[str] = []
        if self.mode == "live":
            if (
                assessment is None
                or snapshot is None
                or nav_usd is None
                or broker_identity is None
            ):
                compliance_blockers.append(
                    "portfolio/compliance decision cannot be evaluated without "
                    "valid serial, position, NAV, and broker-identity state"
                )
            else:
                request = PortfolioComplianceRequest(
                    orders=tuple(order_values),
                    broker_positions=dict(broker_positions),
                    serial_snapshot=snapshot.copy(deep=True),
                    nav_usd=float(nav_usd),
                    as_of=route_time,
                    broker_identity=broker_identity,
                )
                compliance_request_sha256 = request.request_sha256
                try:
                    decision = self.portfolio_compliance_check(request)
                except Exception as exc:
                    compliance_blockers.append(
                        f"portfolio/compliance check failed: {exc}"
                    )
                else:
                    try:
                        observed_request_sha256 = request.request_sha256
                    except Exception as exc:
                        compliance_blockers.append(
                            "portfolio/compliance provider mutated the live request "
                            f"into invalid state: {exc}"
                        )
                    else:
                        if observed_request_sha256 != compliance_request_sha256:
                            compliance_blockers.append(
                                "portfolio/compliance provider mutated the live request"
                            )
                    if not isinstance(decision, PortfolioComplianceDecision):
                        compliance_blockers.append(
                            "portfolio/compliance provider returned an invalid decision"
                        )
                    else:
                        compliance_blockers.extend(
                            _portfolio_compliance_blockers(
                                decision,
                                expected_request_sha256=compliance_request_sha256,
                                expected_policy_sha256=(
                                    self.readiness_check.compliance_policy_sha256
                                ),
                                route_time=route_time,
                                limits=self.limits,
                            )
                        )
                        compliance_decision = decision

        def certificate_blockers(
            certificate: CertifiedIntent | None,
            order: OrderRequest,
        ) -> list[str]:
            if certificate is None:
                return ["live risk increase lacks an authenticated intent certificate"]
            reasons: list[str] = []
            key = self.intent_verification_keys.get(certificate.key_id)
            if key is None:
                reasons.append("intent certificate key is not trusted")
            else:
                expected = hmac.new(
                    key,
                    _canonical(certificate.unsigned_payload()).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(expected, certificate.signature_sha256):
                    reasons.append("intent certificate signature does not match")
            issued = _utc(certificate.issued_at, "intent issued_at")
            expires = _utc(certificate.expires_at, "intent expires_at")
            if route_time < issued or route_time > expires:
                reasons.append("intent certificate is not effective at route time")
            elif (route_time - issued).total_seconds() > self.limits.max_data_age_seconds:
                reasons.append("intent certificate is stale")
            if isinstance(self.readiness_check, ArtifactBackedReadinessCheck):
                expected_fingerprints = {
                    "model_fingerprint_sha256": self.readiness_check.model_fingerprint_sha256,
                    "config_fingerprint_sha256": self.readiness_check.config_fingerprint_sha256,
                    "source_fingerprint_sha256": self.readiness_check.source_fingerprint_sha256,
                }
                for name, expected_value in expected_fingerprints.items():
                    if getattr(certificate, name) != expected_value:
                        reasons.append(f"intent certificate {name} is not deployment-bound")
                if certificate.broker_identity_sha256 != (
                    self.readiness_check.certified_broker_identity_sha256
                ):
                    reasons.append(
                        "intent certificate broker identity is not deployment-bound"
                    )
                if certificate.compliance_policy_sha256 != (
                    self.readiness_check.compliance_policy_sha256
                ):
                    reasons.append(
                        "intent certificate compliance policy is not deployment-bound"
                    )
            if (
                broker_identity is not None
                and certificate.broker_identity_sha256 != broker_identity.sha256
            ):
                reasons.append(
                    "intent certificate broker identity does not match runtime"
                )
            if (
                compliance_decision is not None
                and certificate.compliance_policy_sha256
                != compliance_decision.policy_sha256
            ):
                reasons.append(
                    "intent certificate compliance policy does not match runtime"
                )
            if assessment is None:
                reasons.append("intent certificate cannot be checked without pre-trade assessment")
            else:
                if certificate.serial_snapshot_sha256 != assessment.serial_snapshot_sha256:
                    reasons.append("intent certificate serial snapshot does not match")
                if certificate.position_snapshot_sha256 != assessment.position_snapshot_sha256:
                    reasons.append("intent certificate position snapshot does not match")
                observed_participation = assessment.participation_by_order.get(
                    order.order_id, math.inf
                )
                comparisons = (
                    (
                        observed_participation,
                        certificate.projected_participation,
                        "participation",
                    ),
                    (
                        assessment.projected_gross_notional_multiple,
                        certificate.projected_gross_notional_multiple,
                        "gross notional",
                    ),
                    (
                        assessment.projected_margin_fraction,
                        certificate.projected_margin_fraction,
                        "margin",
                    ),
                )
                for observed, approved, label in comparisons:
                    if observed > float(approved) + 1e-12:
                        reasons.append(
                            f"route-time {label} exceeds the certified projection"
                        )
            return reasons

        submitted_contracts: set[str] = set()

        rows: list[dict[str, Any]] = []
        for order, certificate in entries:
            verified_reduction, projected = _verified_reduce_only(order, broker_positions)
            if assessment is not None:
                verified_reduction = assessment.verified_reduce_only_by_order.get(
                    order.order_id, False
                )
            reduce_only = bool(
                order.reduce_only
                and verified_reduction
                and not preexisting_open_orders
                and order.contract_id not in submitted_contracts
                and not position_failure
                and reconciliation_verified
            )
            reasons: list[str] = []
            if not connected:
                reasons.append("broker is disconnected")
            if self.mode == "live":
                reasons.extend(broker_identity_blockers)
                reasons.extend(compliance_blockers)
            if self.mode == "live" and serial_failure:
                reasons.append(serial_failure)
            if self.mode == "live" and risk_snapshot_failure:
                reasons.append(risk_snapshot_failure)
            if self.mode == "live" and assessment is not None:
                reasons.extend(assessment.blockers_by_order.get(order.order_id, ()))
            if self.mode == "live" and preexisting_open_orders:
                reasons.append("live routing is blocked while broker orders are open")
            if self.mode == "live" and order.roll_id is not None:
                reasons.append(
                    "live roll submission requires a certified atomic/contingent "
                    "selected-broker adapter; the reference router will not send "
                    "individual roll legs"
                )
            if order.reduce_only and not reduce_only:
                if position_failure:
                    reasons.append(position_failure)
                elif preexisting_open_orders:
                    reasons.append(
                        "reduce-only cannot be verified while broker orders are open"
                    )
                elif order.contract_id in submitted_contracts:
                    reasons.append(
                        "reduce-only cannot follow another submitted order for the contract"
                    )
                elif not reconciliation_verified:
                    reasons.append(
                        "reduce-only requires a fresh broker reconciliation"
                    )
                else:
                    reasons.append(
                        "reduce-only flag does not reduce the reconciled broker position"
                    )
            if self.mode == "live" and not reduce_only:
                reasons.extend(certificate_blockers(certificate, order))
                if readiness_status != READY:
                    reasons.append(readiness_failure)
                if runtime_blockers:
                    reasons.extend(runtime_blockers)
                if switch.status != "ACTIVE":
                    reasons.append("kill switch is halted")
            elif self.mode != "live":
                if runtime_blockers and not reduce_only:
                    reasons.extend(runtime_blockers)
                if switch.status != "ACTIVE" and not reduce_only:
                    reasons.append("kill switch is halted")
            if self.mode == "live" and not reasons:
                try:
                    latest_identity = _read_broker_deployment_identity(self.broker)
                except OperationsError as exc:
                    reasons.append(str(exc))
                else:
                    reasons.extend(
                        _broker_identity_blockers(
                            latest_identity,
                            certified_identity_sha256=(
                                self.readiness_check.certified_broker_identity_sha256
                            ),
                        )
                    )
                    if (
                        broker_identity is None
                        or latest_identity.sha256 != broker_identity.sha256
                    ):
                        reasons.append(
                            "runtime broker identity changed after compliance approval"
                        )
            if reasons:
                rows.append(
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": BLOCKED,
                        "broker_order_id": None,
                        "reason": "; ".join(reasons),
                    }
                )
                continue
            outbox_payload = order.payload()
            if self.mode == "live":
                outbox_payload.update(
                    {
                        "broker_identity_sha256": broker_identity.sha256,
                        "compliance_policy_sha256": (
                            compliance_decision.policy_sha256
                        ),
                    }
                )
            outbox = self.journal.append(
                "ORDER_OUTBOX",
                outbox_payload,
                timestamp=route_time,
                idempotency_key=f"outbox:{order.order_id}",
            )

            # If a prior call already persisted and acknowledged this exact
            # order, return idempotently without asking the broker twice.
            try:
                existing_states = replay_order_states(self.journal)
            except Exception as exc:
                rows.append(
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": BLOCKED,
                        "broker_order_id": None,
                        "reason": f"journal replay failed before submission: {exc}",
                    }
                )
                continue
            existing = existing_states.loc[
                existing_states["order_id"].eq(order.order_id)
            ]
            if not existing.empty and existing.iloc[0]["status"] != "PERSISTED":
                rows.append(
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": "ROUTED",
                        "broker_order_id": None,
                        "reason": "exact order was already submitted; no duplicate send",
                    }
                )
                continue
            if self.mode == "live":
                submission_identity_reasons: list[str] = []
                try:
                    submission_identity = _read_broker_deployment_identity(
                        self.broker
                    )
                except OperationsError as exc:
                    submission_identity_reasons.append(str(exc))
                else:
                    submission_identity_reasons.extend(
                        _broker_identity_blockers(
                            submission_identity,
                            certified_identity_sha256=(
                                self.readiness_check.certified_broker_identity_sha256
                            ),
                        )
                    )
                    if submission_identity.sha256 != broker_identity.sha256:
                        submission_identity_reasons.append(
                            "runtime broker identity changed after durable outbox"
                        )
                if submission_identity_reasons:
                    reason = "; ".join(
                        dict.fromkeys(submission_identity_reasons)
                    )
                    try:
                        self.kill_switch.trigger(
                            actor="SYSTEM",
                            reason=reason,
                            timestamp=route_time,
                        )
                    except Exception:
                        pass
                    rows.append(
                        {
                            "order_id": order.order_id,
                            "contract_id": order.contract_id,
                            "status": BLOCKED,
                            "broker_order_id": None,
                            "reason": reason,
                        }
                    )
                    continue
            try:
                event = self.broker.submit_order(order, timestamp=route_time)
            except Exception as exc:
                rows.append(
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": BLOCKED,
                        "broker_order_id": None,
                        "reason": f"broker submission failed after durable outbox: {exc}",
                    }
                )
                continue
            ack_reasons: list[str] = []
            if not isinstance(event, BrokerEvent):
                ack_reasons.append("broker returned an invalid acknowledgement object")
            else:
                if event.event_type != "ACK":
                    ack_reasons.append("broker submission did not return an ACK")
                if event.order_id != order.order_id:
                    ack_reasons.append("broker ACK order_id does not match the submission")
                event_time = _utc(event.timestamp, "broker ACK timestamp")
                outbox_time = _utc(outbox["timestamp"], "outbox timestamp")
                if event_time < outbox_time:
                    ack_reasons.append("broker ACK predates the durable outbox")
                if (event_time - route_time).total_seconds() > self.limits.max_data_age_seconds:
                    ack_reasons.append("broker ACK timestamp is implausibly far in the future")
                for record in self.journal.read():
                    payload = record.get("payload", {})
                    if (
                        record.get("event_type") == "BROKER_ACK"
                        and payload.get("broker_order_id") == event.broker_order_id
                        and payload.get("order_id") != order.order_id
                    ):
                        ack_reasons.append(
                            "broker_order_id is already bound to another order"
                        )
                        break
            if ack_reasons:
                try:
                    self.kill_switch.trigger(
                        actor="SYSTEM",
                        reason="; ".join(dict.fromkeys(ack_reasons)),
                        timestamp=route_time,
                    )
                except Exception:
                    pass
                rows.append(
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": BLOCKED,
                        "broker_order_id": None,
                        "reason": "; ".join(dict.fromkeys(ack_reasons)),
                    }
                )
                continue
            try:
                self.record_broker_event(event)
                replay_order_states(self.journal)
            except Exception as exc:
                try:
                    self.kill_switch.trigger(
                        actor="SYSTEM",
                        reason=f"broker event persistence/replay failed: {exc}",
                        timestamp=route_time,
                    )
                except Exception:
                    pass
                rows.append(
                    {
                        "order_id": order.order_id,
                        "contract_id": order.contract_id,
                        "status": BLOCKED,
                        "broker_order_id": None,
                        "reason": f"broker event persistence/replay failed: {exc}",
                    }
                )
                continue
            rows.append(
                {
                    "order_id": order.order_id,
                    "contract_id": order.contract_id,
                    "status": "ROUTED",
                    "broker_order_id": event.broker_order_id,
                    "reason": "persisted before idempotent broker acknowledgement",
                }
            )
            if projected is not None:
                # Reserve the complete submitted quantity inside this batch so
                # multiple reduce-only orders cannot cross through zero.
                broker_positions[order.contract_id] = projected
            submitted_contracts.add(order.contract_id)
        return pd.DataFrame(rows)

    def record_broker_event(self, event: BrokerEvent) -> dict[str, Any]:
        if not isinstance(event, BrokerEvent):
            raise TypeError("event must be a BrokerEvent")
        return self.journal.append(
            f"BROKER_{event.event_type}",
            event.payload(),
            timestamp=event.timestamp,
            idempotency_key=f"broker-event:{event.event_id}",
        )

    def cancel_all(self, *, timestamp: Any) -> pd.DataFrame:
        rows = []
        for order_id in sorted(self.broker.open_orders()):
            event = self.broker.cancel_order(order_id, timestamp=timestamp)
            self.record_broker_event(event)
            rows.append(event.payload())
        return pd.DataFrame(rows)


def disaster_recovery_report(
    journal: HashChainedJournal,
    broker: BrokerAdapter,
) -> pd.DataFrame:
    """Verify journal replay and open-order agreement after a cold restart."""

    rows: list[dict[str, Any]] = []
    try:
        states = replay_order_states(journal)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "gate": "journal_replay",
                    "status": BLOCKED,
                    "observed": None,
                    "reason": str(exc),
                },
                {
                    "gate": "open_order_reconciliation",
                    "status": BLOCKED,
                    "observed": None,
                    "reason": "journal replay failed",
                },
            ]
        )
    rows.append(
        {
            "gate": "journal_replay",
            "status": PASS,
            "observed": len(states),
            "reason": "hash chain and order state replay passed",
        }
    )
    internal_open = (
        {}
        if states.empty
        else {
            row.order_id: int(row.remaining_quantity)
            for row in states.itertuples(index=False)
            if row.status in {"PERSISTED", "ACKNOWLEDGED", "PARTIAL"}
        }
    )
    broker_open = {str(key): int(value) for key, value in broker.open_orders().items()}
    matched = internal_open == broker_open
    rows.append(
        {
            "gate": "open_order_reconciliation",
            "status": PASS if matched else BLOCKED,
            "observed": len(broker_open),
            "reason": "open orders reconcile" if matched else "open-order break",
        }
    )
    return pd.DataFrame(rows)


__all__ = [
    "BLOCKED",
    "PASS",
    "READY",
    "BrokerAdapter",
    "BrokerDeploymentIdentity",
    "BrokerEvent",
    "CertifiedIntent",
    "DeploymentBoundBrokerAdapter",
    "ExecutionService",
    "HashChainedJournal",
    "KillSwitchState",
    "MonitoringPolicy",
    "OperationsError",
    "OrderRequest",
    "OrderBatchAssessment",
    "PaperBroker",
    "PortfolioComplianceCheck",
    "PortfolioComplianceDecision",
    "PortfolioComplianceRequest",
    "PersistentKillSwitch",
    "assess_order_batch",
    "certify_order_intent",
    "disaster_recovery_report",
    "monitoring_report",
    "position_snapshot_sha256",
    "reconciliation_report",
    "replay_order_states",
    "serial_snapshot_sha256",
]
