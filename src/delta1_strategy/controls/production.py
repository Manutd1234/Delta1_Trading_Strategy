"""Fail-closed controls for taking the research strategy toward production.

This module deliberately does not import :mod:`strategy`.  It accepts plain
objects and pandas containers so that operational controls remain usable if
the research implementation is reorganised.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd


PASS = "PASS"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ProductionLimits:
    """Independent hard limits used by the order and health gates."""

    max_order_participation: float = 0.02
    max_gross_notional_multiple: float = 6.0
    max_margin_fraction: float = 0.35
    max_drawdown_fraction: float = 0.15
    max_data_age_seconds: float = 300.0
    max_reconciliation_age_seconds: float = 30.0
    max_runtime_health_age_seconds: float = 30.0

    def __post_init__(self) -> None:
        bounded = {
            "max_order_participation": self.max_order_participation,
            "max_margin_fraction": self.max_margin_fraction,
            "max_drawdown_fraction": self.max_drawdown_fraction,
        }
        for name, value in bounded.items():
            if not _finite_number(value) or not 0 < float(value) <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in (
            "max_gross_notional_multiple",
            "max_data_age_seconds",
            "max_reconciliation_age_seconds",
            "max_runtime_health_age_seconds",
        ):
            value = getattr(self, name)
            if not _finite_number(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RuntimeHealth:
    """Immutable result of a point-in-time operational health check."""

    healthy: bool = False
    status: str = BLOCKED
    evaluated_at: pd.Timestamp | None = None
    data_timestamp: pd.Timestamp | None = None
    reconciled_at: pd.Timestamp | None = None
    monitoring_evaluated_at: pd.Timestamp | None = None
    kill_switch_checked_at: pd.Timestamp | None = None
    position_snapshot_sha256: str | None = None
    data_age_seconds: float | None = None
    reconciliation_age_seconds: float | None = None
    monitoring_age_seconds: float | None = None
    kill_switch_age_seconds: float | None = None
    nav_usd: float | None = None
    peak_nav_usd: float | None = None
    gross_notional_usd: float | None = None
    margin_requirement_usd: float | None = None
    gross_notional_multiple: float | None = None
    margin_fraction: float | None = None
    drawdown_fraction: float | None = None
    broker_connected: bool = False
    broker_reconciled: bool = False
    monitoring_healthy: bool = False
    kill_switch_ready: bool = False
    reasons: tuple[str, ...] = ("health check has not been evaluated",)


ORDER_INTENT_COLUMNS = [
    "order_id",
    "symbol",
    "decision_timestamp",
    "current_contracts",
    "target_contracts",
    "requested_quantity",
    "capacity_contracts",
    "capped_quantity",
    "quantity",
    "side",
    "projected_contracts",
    "projected_participation",
    "projected_notional_usd",
    "projected_margin_usd",
    "portfolio_gross_notional_multiple",
    "portfolio_margin_fraction",
    "style",
    "approved",
    "status",
    "reason",
]


EXTERNAL_EVIDENCE_LABELS = {
    "serial_contract_data": "Tradeable serial-contract data and roll liquidity",
    "timestamped_live_market_data": "Timestamped live market data and calendars",
    "effective_dated_contract_specs": "Effective-dated contract specifications",
    "effective_dated_margin_and_fees": "Effective-dated margin and fee schedules",
    "calibrated_execution_cost_model": "Quote/fill-calibrated execution costs",
    "point_in_time_universe": "Point-in-time universe and security master",
    "cash_margin_funding_controls": "Cash, variation margin, collateral, and funding controls",
    "frozen_model_and_change_control": "Frozen model and approved change-control record",
    "independent_holdout": "Genuinely post-freeze independent holdout",
    "forward_paper_trading": "Prospective paper/shadow-trading acceptance record",
    "certified_broker_adapter": "Certified adapter for the selected broker",
    "broker_reconciliation": "Broker and clearing reconciliation",
    "monitoring_and_alerting": "Monitoring, alerting, and incident response",
    "disaster_recovery_drill": "Successful disaster-recovery drill",
    "tested_kill_switch": "Tested, latched kill switch",
    "compliance_approval": "Compliance, market-access, delivery, and position-limit approval",
    "independent_model_validation": "Independent model-risk validation and launch approval",
}


HISTORICAL_CRITICAL_GATES = frozenset(
    {
        "backtest_daily_available",
        "required_daily_columns",
        "finite_positive_nav",
        "finite_net_returns",
        "nonnegative_costs",
        "historical_gross_notional",
        "historical_rebalance_participation",
        "historical_order_participation",
        "historical_drawdown",
        "terminal_pending_orders",
    }
)
INFORMATIONAL_READINESS_GATES = frozenset(
    {
        "historical_static_margin_proxy",
        "continuous_roll_participation_proxy",
    }
)
REQUIRED_CRITICAL_GATES = frozenset(EXTERNAL_EVIDENCE_LABELS).union(
    HISTORICAL_CRITICAL_GATES
)
EXPECTED_READINESS_GATES = REQUIRED_CRITICAL_GATES.union(
    INFORMATIONAL_READINESS_GATES
)


def _finite_number(value: Any) -> bool:
    """Return ``True`` only for real, finite scalar numbers (not booleans)."""

    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number)


def _normalise_timestamp(value: Any) -> tuple[pd.Timestamp | None, str]:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None, "INVALID_TIMESTAMP"
    if pd.isna(timestamp):
        return None, "INVALID_TIMESTAMP"
    if timestamp.tzinfo is None:
        return None, "INVALID_TIMESTAMP"
    timestamp = timestamp.tz_convert("UTC")
    return timestamp, timestamp.isoformat()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def daily_frame_fingerprint(daily: Any) -> str:
    """Return a deterministic fingerprint of the exact in-memory daily ledger."""

    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("daily ledger must be a non-empty DataFrame")
    if daily.index.has_duplicates or daily.columns.has_duplicates:
        raise ValueError("daily ledger index and columns must be unique")
    if not all(isinstance(column, str) and column for column in daily.columns):
        raise ValueError("daily ledger columns must be non-empty strings")
    metadata = {
        "columns": list(daily.columns),
        "dtypes": [str(dtype) for dtype in daily.dtypes],
        "index_class": type(daily.index).__name__,
        "index_dtype": str(daily.index.dtype),
        "index_name": (
            str(daily.index.name) if daily.index.name is not None else None
        ),
        "shape": [int(daily.shape[0]), int(daily.shape[1])],
    }
    try:
        row_hashes = pd.util.hash_pandas_object(
            daily,
            index=True,
            categorize=True,
        ).to_numpy(dtype="<u8", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("daily ledger cannot be deterministically hashed") from exc
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _named_series(values: Any, name: str) -> pd.Series:
    """Convert a mapping-like object to a symbol-indexed Series."""

    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, Mapping):
        series = pd.Series(dict(values), dtype=object)
    else:
        raise TypeError(f"{name} must be a pandas Series or mapping")
    if series.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate symbols")
    series.index = series.index.map(str)
    if series.index.has_duplicates:
        raise ValueError(f"{name} contains symbols that collide as strings")
    return series.rename(name)


def _contract_number(value: Any) -> tuple[float | None, str | None]:
    if not _finite_number(value):
        return None, "contract count is missing or non-finite"
    number = float(value)
    if not number.is_integer():
        return None, "contract count is not an integer"
    return number, None


def _positive_number(value: Any, label: str) -> tuple[float | None, str | None]:
    if not _finite_number(value) or float(value) <= 0:
        return None, f"{label} is missing, non-finite, or non-positive"
    return float(value), None


def _order_id(
    decision_text: str,
    symbol: str,
    current: Any,
    target: Any,
    capped_quantity: Any,
) -> str:
    payload = json.dumps(
        [decision_text, symbol, current, target, capped_quantity],
        separators=(",", ":"),
        default=str,
    )
    return "d1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _is_reduce_only(current: float, projected: float, quantity: float) -> bool:
    if quantity == 0 or current == 0:
        return False
    same_side = projected == 0 or math.copysign(1.0, projected) == math.copysign(
        1.0, current
    )
    return same_side and abs(projected) < abs(current)


def build_order_intents(
    current_contracts: pd.Series | Mapping[Any, Any],
    target_contracts: pd.Series | Mapping[Any, Any],
    notional_per_contract_usd: pd.Series | Mapping[Any, Any],
    margin_per_contract: pd.Series | Mapping[Any, Any],
    adv: pd.Series | Mapping[Any, Any],
    nav: float,
    decision_timestamp: Any,
    limits: ProductionLimits = ProductionLimits(),
) -> pd.DataFrame:
    """Build deterministic, participation-capped and risk-gated order intents.

    Missing inputs never result in an approved order.  When aggregate exposure
    cannot be valued or breaches a hard limit, only an order that strictly
    reduces an existing position without crossing through zero may proceed.
    """

    if not isinstance(limits, ProductionLimits):
        raise TypeError("limits must be a ProductionLimits instance")

    inputs = {
        "current_contracts": _named_series(current_contracts, "current_contracts"),
        "target_contracts": _named_series(target_contracts, "target_contracts"),
        "notional_per_contract_usd": _named_series(
            notional_per_contract_usd,
            "notional_per_contract_usd",
        ),
        "margin_per_contract": _named_series(
            margin_per_contract, "margin_per_contract"
        ),
        "adv": _named_series(adv, "adv"),
    }
    # Metadata can legitimately contain symbols outside today's book.  Only
    # current or requested positions create an intent; missing metadata for
    # one of those intents is handled below as a blocked row.
    symbols = sorted(
        set(inputs["current_contracts"].index).union(
            inputs["target_contracts"].index
        )
    )
    if not symbols:
        return pd.DataFrame(columns=ORDER_INTENT_COLUMNS)

    decision_time, decision_text = _normalise_timestamp(decision_timestamp)
    nav_valid = _finite_number(nav) and float(nav) > 0
    nav_number = float(nav) if nav_valid else math.nan

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        errors: list[str] = []
        current, error = _contract_number(inputs["current_contracts"].get(symbol, np.nan))
        if error:
            errors.append(f"current {error}")
        target, error = _contract_number(inputs["target_contracts"].get(symbol, np.nan))
        if error:
            errors.append(f"target {error}")
        contract_notional, error = _positive_number(
            inputs["notional_per_contract_usd"].get(symbol, np.nan),
            "USD notional per contract",
        )
        if error:
            errors.append(error)
        margin, error = _positive_number(
            inputs["margin_per_contract"].get(symbol, np.nan),
            "margin per contract",
        )
        if error:
            errors.append(error)
        daily_volume, error = _positive_number(inputs["adv"].get(symbol, np.nan), "ADV")
        if error:
            errors.append(error)
        if decision_time is None:
            errors.append("decision timestamp is missing or invalid")
        if not nav_valid:
            errors.append("NAV is missing, non-finite, or non-positive")

        requested = target - current if current is not None and target is not None else math.nan
        capacity = (
            math.floor(limits.max_order_participation * daily_volume)
            if daily_volume is not None
            else 0
        )
        capped = (
            math.copysign(min(abs(requested), capacity), requested)
            if _finite_number(requested) and requested != 0 and capacity > 0
            else 0.0
        )
        projected = current + capped if current is not None else math.nan
        participation = abs(capped) / daily_volume if daily_volume is not None else math.nan
        notional = (
            abs(projected) * contract_notional
            if _finite_number(projected) and contract_notional is not None
            else math.nan
        )
        projected_margin = (
            abs(projected) * margin
            if _finite_number(projected) and margin is not None
            else math.nan
        )
        side = "BUY" if capped > 0 else "SELL" if capped < 0 else "HOLD"
        rows.append(
            {
                "symbol": symbol,
                "decision_timestamp": decision_time,
                "current_contracts": current if current is not None else math.nan,
                "target_contracts": target if target is not None else math.nan,
                "requested_quantity": requested,
                "capacity_contracts": int(capacity),
                "capped_quantity": int(capped) if float(capped).is_integer() else capped,
                "quantity": abs(int(capped)) if float(capped).is_integer() else abs(capped),
                "side": side,
                "projected_contracts": projected,
                "projected_participation": participation,
                "projected_notional_usd": notional,
                "projected_margin_usd": projected_margin,
                "_errors": errors,
                "_reduce_only": (
                    _is_reduce_only(current, projected, capped)
                    if current is not None and _finite_number(projected)
                    else False
                ),
            }
        )

    valuation_valid = all(
        _finite_number(row["projected_notional_usd"])
        and _finite_number(row["projected_margin_usd"])
        for row in rows
    )
    if nav_valid and valuation_valid:
        gross_multiple = sum(row["projected_notional_usd"] for row in rows) / nav_number
        margin_fraction = sum(row["projected_margin_usd"] for row in rows) / nav_number
    else:
        gross_multiple = math.nan
        margin_fraction = math.nan

    aggregate_reasons: list[str] = []
    if not valuation_valid:
        aggregate_reasons.append("aggregate exposure cannot be valued")
    if not nav_valid:
        aggregate_reasons.append("aggregate exposure cannot be compared with NAV")
    if _finite_number(gross_multiple) and gross_multiple > limits.max_gross_notional_multiple:
        aggregate_reasons.append("projected gross notional exceeds limit")
    if _finite_number(margin_fraction) and margin_fraction > limits.max_margin_fraction:
        aggregate_reasons.append("projected margin exceeds limit")

    for row in rows:
        row["portfolio_gross_notional_multiple"] = gross_multiple
        row["portfolio_margin_fraction"] = margin_fraction
        requested = row["requested_quantity"]
        capped = row["capped_quantity"]
        errors = row.pop("_errors")
        reduce_only = row.pop("_reduce_only")

        if errors:
            row.update(
                style="BLOCKED",
                approved=False,
                status=BLOCKED,
                reason="; ".join(dict.fromkeys(errors)),
            )
        elif requested == 0:
            row.update(
                style="NONE",
                approved=False,
                status="NO_ACTION",
                reason="position already equals target",
            )
        elif capped == 0:
            row.update(
                style="BLOCKED",
                approved=False,
                status=BLOCKED,
                reason="participation limit permits fewer than one contract",
            )
        elif aggregate_reasons and not reduce_only:
            row.update(
                style="BLOCKED",
                approved=False,
                status=BLOCKED,
                reason="; ".join(aggregate_reasons),
            )
        elif aggregate_reasons:
            row.update(
                style="REDUCE_ONLY",
                approved=True,
                status="APPROVED_REDUCE_ONLY",
                reason="risk-reducing order allowed while " + "; ".join(aggregate_reasons),
            )
        elif abs(capped) < abs(requested):
            row.update(
                style="PARTICIPATION_CAPPED",
                approved=True,
                status="APPROVED_PARTIAL",
                reason="quantity capped by the participation limit",
            )
        else:
            row.update(
                style="PARTICIPATION",
                approved=True,
                status="APPROVED",
                reason="all order and portfolio gates passed",
            )
        row["order_id"] = _order_id(
            decision_text,
            row["symbol"],
            row["current_contracts"],
            row["target_contracts"],
            row["capped_quantity"],
        )

    return pd.DataFrame(rows).reindex(columns=ORDER_INTENT_COLUMNS)


def evaluate_runtime_health(
    *,
    data_timestamp: Any = None,
    reconciled_at: Any = None,
    monitoring_evaluated_at: Any = None,
    kill_switch_checked_at: Any = None,
    position_snapshot_sha256: Any = None,
    nav: Any = None,
    gross_notional: Any = None,
    margin_requirement: Any = None,
    peak_nav: Any = None,
    broker_connected: bool = False,
    broker_reconciled: bool = False,
    monitoring_healthy: bool = False,
    kill_switch_ready: bool = False,
    limits: ProductionLimits = ProductionLimits(),
    current_time: Any = None,
) -> RuntimeHealth:
    """Evaluate a live snapshot; absent, stale, or malformed state blocks trading.

    A passing snapshot is an ordered evidence chain: market data was observed,
    broker positions were reconciled to a content-addressed snapshot,
    monitoring evaluated that reconciled state, the kill switch was checked,
    and only then was this health result evaluated.  Every timestamp must be
    timezone-aware and sufficiently fresh under ``limits``.
    """

    if not isinstance(limits, ProductionLimits):
        raise TypeError("limits must be a ProductionLimits instance")
    reasons: list[str] = []

    data_time, _ = _normalise_timestamp(data_timestamp)
    reconciliation_time, _ = _normalise_timestamp(reconciled_at)
    monitoring_time, _ = _normalise_timestamp(monitoring_evaluated_at)
    kill_switch_time, _ = _normalise_timestamp(kill_switch_checked_at)
    now, _ = _normalise_timestamp(
        pd.Timestamp.now(tz="UTC") if current_time is None else current_time
    )
    data_age: float | None = None
    reconciliation_age: float | None = None
    monitoring_age: float | None = None
    kill_switch_age: float | None = None
    if data_time is None:
        reasons.append("data timestamp is missing or invalid")
    if reconciliation_time is None:
        reasons.append("reconciliation timestamp is missing or invalid")
    if monitoring_time is None:
        reasons.append("monitoring timestamp is missing or invalid")
    if kill_switch_time is None:
        reasons.append("kill-switch timestamp is missing or invalid")
    if now is None:
        reasons.append("current timestamp is missing or invalid")
    if data_time is not None and now is not None:
        data_age = float((now - data_time).total_seconds())
        if data_age < 0:
            reasons.append("data timestamp is in the future")
        elif data_age > limits.max_data_age_seconds:
            reasons.append("market data is stale")

    freshness_checks = (
        (
            reconciliation_time,
            limits.max_reconciliation_age_seconds,
            "reconciliation timestamp is in the future",
            "broker reconciliation is stale",
            "reconciliation",
        ),
        (
            monitoring_time,
            limits.max_runtime_health_age_seconds,
            "monitoring timestamp is in the future",
            "monitoring evaluation is stale",
            "monitoring",
        ),
        (
            kill_switch_time,
            limits.max_runtime_health_age_seconds,
            "kill-switch timestamp is in the future",
            "kill-switch check is stale",
            "kill_switch",
        ),
    )
    timestamp_ages: dict[str, float] = {}
    if now is not None:
        for timestamp, maximum_age, future_reason, stale_reason, key in freshness_checks:
            if timestamp is None:
                continue
            age = float((now - timestamp).total_seconds())
            timestamp_ages[key] = age
            if age < 0:
                reasons.append(future_reason)
            elif age > maximum_age:
                reasons.append(stale_reason)
    reconciliation_age = timestamp_ages.get("reconciliation")
    monitoring_age = timestamp_ages.get("monitoring")
    kill_switch_age = timestamp_ages.get("kill_switch")

    ordered_pairs = (
        (
            data_time,
            reconciliation_time,
            "reconciliation timestamp predates market data",
        ),
        (
            reconciliation_time,
            monitoring_time,
            "monitoring timestamp predates reconciliation",
        ),
        (
            monitoring_time,
            kill_switch_time,
            "kill-switch timestamp predates monitoring evaluation",
        ),
    )
    for earlier, later, reason in ordered_pairs:
        if earlier is not None and later is not None and earlier > later:
            reasons.append(reason)

    position_digest = (
        position_snapshot_sha256
        if isinstance(position_snapshot_sha256, str)
        else None
    )
    if not _valid_sha256(position_snapshot_sha256):
        reasons.append("position snapshot digest is missing or not lowercase SHA-256")

    nav_valid = _finite_number(nav) and float(nav) > 0
    gross_valid = _finite_number(gross_notional) and float(gross_notional) >= 0
    margin_valid = _finite_number(margin_requirement) and float(margin_requirement) >= 0
    peak_valid = _finite_number(peak_nav) and float(peak_nav) > 0
    if not nav_valid:
        reasons.append("NAV is missing, non-finite, or non-positive")
    if not gross_valid:
        reasons.append("gross notional is missing, non-finite, or negative")
    if not margin_valid:
        reasons.append("margin requirement is missing, non-finite, or negative")
    if not peak_valid:
        reasons.append("peak NAV is missing, non-finite, or non-positive")

    gross_multiple: float | None = None
    margin_fraction: float | None = None
    drawdown_fraction: float | None = None
    if nav_valid and gross_valid:
        gross_multiple = float(gross_notional) / float(nav)
        if gross_multiple > limits.max_gross_notional_multiple:
            reasons.append("gross notional limit breached")
    if nav_valid and margin_valid:
        margin_fraction = float(margin_requirement) / float(nav)
        if margin_fraction > limits.max_margin_fraction:
            reasons.append("margin limit breached")
    if nav_valid and peak_valid:
        if float(nav) > float(peak_nav):
            reasons.append("NAV exceeds the supplied peak NAV")
        else:
            drawdown_fraction = 1.0 - float(nav) / float(peak_nav)
            if drawdown_fraction > limits.max_drawdown_fraction:
                reasons.append("drawdown limit breached")

    operational = {
        "broker connection is not healthy": broker_connected,
        "broker reconciliation is not healthy": broker_reconciled,
        "monitoring and alerting are not healthy": monitoring_healthy,
        "kill switch is not ready": kill_switch_ready,
    }
    for reason, value in operational.items():
        if value is not True:
            reasons.append(reason)

    unique_reasons = tuple(dict.fromkeys(reasons))
    healthy = not unique_reasons
    return RuntimeHealth(
        healthy=healthy,
        status=PASS if healthy else BLOCKED,
        evaluated_at=now,
        data_timestamp=data_time,
        reconciled_at=reconciliation_time,
        monitoring_evaluated_at=monitoring_time,
        kill_switch_checked_at=kill_switch_time,
        position_snapshot_sha256=position_digest,
        data_age_seconds=data_age,
        reconciliation_age_seconds=reconciliation_age,
        monitoring_age_seconds=monitoring_age,
        kill_switch_age_seconds=kill_switch_age,
        nav_usd=float(nav) if nav_valid else None,
        peak_nav_usd=float(peak_nav) if peak_valid else None,
        gross_notional_usd=float(gross_notional) if gross_valid else None,
        margin_requirement_usd=float(margin_requirement) if margin_valid else None,
        gross_notional_multiple=gross_multiple,
        margin_fraction=margin_fraction,
        drawdown_fraction=drawdown_fraction,
        broker_connected=broker_connected is True,
        broker_reconciled=broker_reconciled is True,
        monitoring_healthy=monitoring_healthy is True,
        kill_switch_ready=kill_switch_ready is True,
        reasons=unique_reasons,
    )


def _report_row(
    gate: str,
    category: str,
    passed: bool,
    observed: Any,
    limit: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "category": category,
        "critical": True,
        "status": PASS if passed else BLOCKED,
        "observed": observed,
        "limit": limit,
        "reason": reason if not passed else "gate passed",
    }


def _finite_series(frame: pd.DataFrame, column: str) -> tuple[bool, pd.Series | None]:
    if column not in frame.columns:
        return False, None
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(len(values) and np.isfinite(values.to_numpy(dtype=float)).all()), values


def production_readiness_report(
    result: Any,
    limits: ProductionLimits = ProductionLimits(),
    *,
    evidence_registry: str | PathLike[str] | None = None,
    model_fingerprint_sha256: str | None = None,
    config_fingerprint_sha256: str | None = None,
    source_fingerprint_sha256: str | None = None,
    certified_broker_identity_sha256: str | None = None,
    compliance_policy_sha256: str | None = None,
    evidence_as_of: Any = None,
) -> pd.DataFrame:
    """Return critical historical and external-evidence launch gates.

    A backtest can demonstrate that numerical limits were respected; it cannot
    demonstrate live data, broker, cost calibration, or holdout readiness.
    External gates must therefore be loaded from artifact-backed records using
    :mod:`evidence`.  Naked booleans are no longer accepted.  A missing registry
    or fingerprint context produces one explicit ``BLOCKED`` row per gate.
    """

    if not isinstance(limits, ProductionLimits):
        raise TypeError("limits must be a ProductionLimits instance")

    rows: list[dict[str, Any]] = []
    daily = getattr(result, "daily", None)
    daily_valid = isinstance(daily, pd.DataFrame) and not daily.empty
    rows.append(
        _report_row(
            "backtest_daily_available",
            "backtest",
            daily_valid,
            len(daily) if isinstance(daily, pd.DataFrame) else None,
            "> 0 rows",
            "result.daily is missing, not a DataFrame, or empty",
        )
    )

    required = (
        "nav",
        "equity",
        "net_return",
        "cost",
        "gross_notional_multiple",
        "static_margin_fraction",
        "max_order_participation",
        "max_rebalance_participation",
        "max_roll_participation_proxy",
        "pending_markets",
    )
    columns_present = daily_valid and all(column in daily.columns for column in required)
    missing = [column for column in required if not daily_valid or column not in daily.columns]
    rows.append(
        _report_row(
            "required_daily_columns",
            "backtest",
            columns_present,
            ", ".join(missing) if missing else "all present",
            "all required columns",
            "required daily columns are missing",
        )
    )

    safe_daily = daily if daily_valid else pd.DataFrame()
    nav_finite, nav_values = _finite_series(safe_daily, "nav")
    nav_ok = nav_finite and bool((nav_values > 0).all())
    rows.append(
        _report_row(
            "finite_positive_nav",
            "ledger",
            nav_ok,
            float(nav_values.min()) if nav_finite else None,
            "> 0",
            "NAV contains missing, non-finite, or non-positive values",
        )
    )

    returns_ok, _ = _finite_series(safe_daily, "net_return")
    rows.append(
        _report_row(
            "finite_net_returns",
            "ledger",
            returns_ok,
            "finite" if returns_ok else None,
            "all finite",
            "net returns contain missing or non-finite values",
        )
    )

    costs_finite, cost_values = _finite_series(safe_daily, "cost")
    costs_ok = costs_finite and bool((cost_values >= 0).all())
    rows.append(
        _report_row(
            "nonnegative_costs",
            "ledger",
            costs_ok,
            float(cost_values.min()) if costs_finite else None,
            ">= 0",
            "costs contain missing, non-finite, or negative values",
        )
    )

    metric_gates = (
        (
            "historical_gross_notional",
            "gross_notional_multiple",
            limits.max_gross_notional_multiple,
            "historical gross notional exceeds the production limit",
        ),
        (
            "historical_rebalance_participation",
            "max_rebalance_participation",
            limits.max_order_participation,
            "historical rebalance participation exceeds the production limit",
        ),
        (
            # Rebalance participation alone misses the roll, whose two legs are
            # the larger order in most sessions.  A book can satisfy the
            # rebalance limit and still have demanded many times a thin
            # session's volume to transfer a delivery.
            "historical_order_participation",
            "max_order_participation",
            limits.max_order_participation,
            "historical two-leg order participation exceeds the production limit",
        ),
    )
    for gate, column, limit, failure_reason in metric_gates:
        finite, values = _finite_series(safe_daily, column)
        observed = float(values.max()) if finite else None
        passed = finite and bool((values >= 0).all()) and observed <= limit
        rows.append(
            _report_row(gate, "risk", passed, observed, limit, failure_reason)
        )

    margin_finite, margin_proxy = _finite_series(
        safe_daily, "static_margin_fraction"
    )
    rows.append(
        {
            "gate": "historical_static_margin_proxy",
            "category": "research limitation",
            "critical": False,
            "status": "INFORMATIONAL",
            "observed": (
                float(margin_proxy.max()) if margin_finite else None
            ),
            "limit": "not a point-in-time historical gate",
            "reason": (
                "computed from a current static catalogue snapshot and must not "
                "resize historical positions; live margin is governed by "
                "effective_dated_margin_and_fees evidence and runtime controls"
            ),
        }
    )

    equity_finite, equity = _finite_series(safe_daily, "equity")
    max_drawdown: float | None = None
    if equity_finite and bool((equity > 0).all()):
        # Canonical normalised equity has an implicit 1.0 starting value even
        # when the frame begins on the first return observation.  Always
        # include it so a first-session loss cannot establish its own peak.
        # Bare NAV is intentionally not accepted as a fallback because its
        # pre-observation initial capital is ambiguous.
        path = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
        running_peak = path.cummax()
        max_drawdown = float((1.0 - path / running_peak).max())
    drawdown_ok = max_drawdown is not None and max_drawdown <= limits.max_drawdown_fraction
    rows.append(
        _report_row(
            "historical_drawdown",
            "risk",
            drawdown_ok,
            max_drawdown,
            limits.max_drawdown_fraction,
            "historical drawdown exceeds the production limit or cannot be computed",
        )
    )

    pending_finite, pending = _finite_series(safe_daily, "pending_markets")
    pending_ok = pending_finite and bool((pending >= 0).all()) and float(pending.iloc[-1]) == 0
    rows.append(
        _report_row(
            "terminal_pending_orders",
            "execution",
            pending_ok,
            float(pending.iloc[-1]) if pending_finite else None,
            0,
            "the backtest ends with pending market orders or invalid pending-order data",
        )
    )

    roll_finite, roll_proxy = _finite_series(
        safe_daily, "max_roll_participation_proxy"
    )
    rows.append(
        {
            "gate": "continuous_roll_participation_proxy",
            "category": "research limitation",
            "critical": False,
            "status": "INFORMATIONAL",
            "observed": float(roll_proxy.max()) if roll_finite else None,
            "limit": "not a live-capacity gate",
            "reason": (
                "continuous-contract roll turnover is only a diagnostic proxy; "
                "production roll capacity is governed by serial_contract_data evidence"
            ),
        }
    )

    evidence_gates = tuple(EXTERNAL_EVIDENCE_LABELS)
    fingerprint_values = (
        model_fingerprint_sha256,
        config_fingerprint_sha256,
        source_fingerprint_sha256,
    )
    if evidence_registry is None or any(value is None for value in fingerprint_values):
        for gate, label in EXTERNAL_EVIDENCE_LABELS.items():
            rows.append(
                _report_row(
                    gate,
                    "external evidence",
                    False,
                    None,
                    "valid artifact-backed approval",
                    f"{label} has no verified evidence record",
                )
            )
    else:
        from .evidence import verify_evidence_registry

        evidence_report = verify_evidence_registry(
            evidence_registry,
            required_gates=evidence_gates,
            expected_model_fingerprint=model_fingerprint_sha256,
            expected_config_fingerprint=config_fingerprint_sha256,
            expected_source_fingerprint=source_fingerprint_sha256,
            expected_subject_sha256_by_gate={
                "certified_broker_adapter": certified_broker_identity_sha256,
                "compliance_approval": compliance_policy_sha256,
            },
            as_of=evidence_as_of,
        )
        rows.extend(evidence_report.to_dict("records"))

    return pd.DataFrame(rows).reindex(
        columns=["gate", "category", "critical", "status", "observed", "limit", "reason"]
    )


@dataclass(frozen=True)
class ArtifactBackedReadinessCheck:
    """Re-verify the full launch report and evidence registry at route time.

    The object binds a specific research result, evidence registry and frozen
    fingerprints.  Calling it re-reads and re-hashes the evidence artifacts at
    the supplied UTC time, so an order router never relies on a caller-provided
    ``"READY"`` string or a stale cached pass/fail value.
    """

    result: Any
    evidence_registry: str | PathLike[str]
    model_fingerprint_sha256: str
    config_fingerprint_sha256: str
    source_fingerprint_sha256: str
    certified_broker_identity_sha256: str
    compliance_policy_sha256: str
    limits: ProductionLimits = ProductionLimits()
    run_manifest_path: str | PathLike[str] | None = None
    deployment_root: str | PathLike[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.limits, ProductionLimits):
            raise TypeError("limits must be a ProductionLimits instance")
        if not isinstance(self.evidence_registry, (str, PathLike)):
            raise TypeError("evidence_registry must be a filesystem path")
        if self.run_manifest_path is not None and not isinstance(
            self.run_manifest_path,
            (str, PathLike),
        ):
            raise TypeError("run_manifest_path must be a filesystem path")
        if self.deployment_root is not None and not isinstance(
            self.deployment_root,
            (str, PathLike),
        ):
            raise TypeError("deployment_root must be a filesystem path")
        for name in (
            "model_fingerprint_sha256",
            "config_fingerprint_sha256",
            "source_fingerprint_sha256",
            "certified_broker_identity_sha256",
            "compliance_policy_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @staticmethod
    def _resolve_declared_file(
        *,
        root: Path,
        base: Path,
        relative_name: Any,
        label: str,
    ) -> Path:
        if not isinstance(relative_name, str) or not relative_name.strip():
            raise ValueError(f"{label} contains an invalid path")
        relative = Path(relative_name)
        if relative.is_absolute():
            raise ValueError(f"{label} paths must be relative")
        try:
            candidate = (base / relative).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{label} path escapes or is unavailable: {relative_name}") from exc
        if not candidate.is_file():
            raise ValueError(f"{label} path is not a regular file: {relative_name}")
        return candidate

    def _verify_deployment_bundle(self) -> str | None:
        if self.run_manifest_path is None or self.deployment_root is None:
            return "run_manifest_path and deployment_root are required"
        try:
            root = Path(self.deployment_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            return f"deployment root is unavailable: {exc}"
        if not root.is_dir():
            return "deployment root is not a directory"
        manifest_input = Path(self.run_manifest_path).expanduser()
        if not manifest_input.is_absolute():
            manifest_input = root / manifest_input
        try:
            manifest_path = manifest_input.resolve(strict=True)
            manifest_path.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            return f"run manifest escapes or is unavailable: {exc}"
        if not manifest_path.is_file():
            return "run manifest is not a regular file"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"run manifest cannot be parsed: {exc}"
        if not isinstance(manifest, dict):
            return "run manifest must be a JSON object"

        manifest_fingerprints = {
            "implementation_fingerprint_sha256": self.model_fingerprint_sha256,
            "config_sha256": self.config_fingerprint_sha256,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
        }
        for field, expected in manifest_fingerprints.items():
            observed = manifest.get(field)
            if not _valid_sha256(observed):
                return f"run manifest {field} is missing or invalid"
            if observed != expected:
                return f"run manifest {field} does not match the approved fingerprint"

        verified_maps: dict[str, dict[str, Path]] = {}
        for field, base in (
            ("implementation_files", root),
            ("output_files", manifest_path.parent),
        ):
            declared = manifest.get(field)
            if not isinstance(declared, dict) or not declared:
                return f"run manifest {field} is missing or empty"
            resolved: dict[str, Path] = {}
            resolved_targets: set[Path] = set()
            for relative_name, declared_hash in declared.items():
                if not _valid_sha256(declared_hash):
                    return f"run manifest {field} contains an invalid SHA-256"
                try:
                    file_path = self._resolve_declared_file(
                        root=root,
                        base=base,
                        relative_name=relative_name,
                        label=field,
                    )
                    observed_hash = _sha256_file(file_path)
                except (OSError, ValueError) as exc:
                    return str(exc)
                if file_path in resolved_targets:
                    return f"run manifest {field} resolves duplicate file paths"
                if observed_hash != declared_hash:
                    return f"deployment file SHA-256 does not match: {relative_name}"
                resolved[str(relative_name)] = file_path
                resolved_targets.add(file_path)
            verified_maps[field] = resolved

        implementation_hashes = manifest["implementation_files"]
        observed_implementation_fingerprint = hashlib.sha256(
            "\n".join(
                f"{name}:{digest}"
                for name, digest in sorted(implementation_hashes.items())
            ).encode("utf-8")
        ).hexdigest()
        if observed_implementation_fingerprint != self.model_fingerprint_sha256:
            return "implementation-file hashes do not reproduce the model fingerprint"

        required_outputs = {
            "source_manifest.csv",
            "strategy_config.json",
            "strategy_daily.csv",
        }
        output_paths = verified_maps["output_files"]
        missing_outputs = sorted(required_outputs.difference(output_paths))
        if missing_outputs:
            return "run manifest omits required outputs: " + ", ".join(missing_outputs)

        try:
            config_bytes = output_paths["strategy_config.json"].read_bytes()
        except OSError as exc:
            return f"strategy configuration cannot be read: {exc}"
        config_payload = config_bytes[:-1] if config_bytes.endswith(b"\n") else config_bytes
        observed_config_fingerprint = hashlib.sha256(config_payload).hexdigest()
        if observed_config_fingerprint != self.config_fingerprint_sha256:
            return "strategy configuration does not reproduce the config fingerprint"

        try:
            source_manifest = pd.read_csv(output_paths["source_manifest.csv"])
        except Exception as exc:
            return f"source manifest cannot be parsed: {exc}"
        if (
            source_manifest.empty
            or not {"relative_path", "sha256"}.issubset(source_manifest.columns)
            or source_manifest["relative_path"].isna().any()
            or source_manifest["relative_path"].astype(str).duplicated().any()
            or not source_manifest["sha256"].map(_valid_sha256).all()
        ):
            return "source manifest is malformed"
        observed_source_fingerprint = hashlib.sha256(
            "\n".join(
                source_manifest["relative_path"].astype(str)
                + ":"
                + source_manifest["sha256"].astype(str)
            ).encode("utf-8")
        ).hexdigest()
        if observed_source_fingerprint != self.source_fingerprint_sha256:
            return "source manifest does not reproduce the source fingerprint"

        declared_daily_fingerprint = manifest.get("daily_fingerprint_sha256")
        if not _valid_sha256(declared_daily_fingerprint):
            return "run manifest daily_fingerprint_sha256 is missing or invalid"
        try:
            observed_daily_fingerprint = daily_frame_fingerprint(
                getattr(self.result, "daily", None)
            )
        except ValueError as exc:
            return str(exc)
        if observed_daily_fingerprint != declared_daily_fingerprint:
            return "in-memory daily ledger does not match the frozen run manifest"
        return None

    def _blocked_bundle_report(self, reason: str) -> pd.DataFrame:
        report = production_readiness_report(self.result, self.limits)
        external = report["category"].eq("external evidence")
        report.loc[external, "status"] = BLOCKED
        report.loc[external, "observed"] = None
        report.loc[external, "reason"] = (
            "deployment bundle verification failed: " + reason
        )
        return report

    def __call__(self, *, as_of: Any) -> pd.DataFrame:
        bundle_error = self._verify_deployment_bundle()
        if bundle_error is not None:
            return self._blocked_bundle_report(bundle_error)
        report = production_readiness_report(
            self.result,
            self.limits,
            evidence_registry=self.evidence_registry,
            model_fingerprint_sha256=self.model_fingerprint_sha256,
            config_fingerprint_sha256=self.config_fingerprint_sha256,
            source_fingerprint_sha256=self.source_fingerprint_sha256,
            certified_broker_identity_sha256=self.certified_broker_identity_sha256,
            compliance_policy_sha256=self.compliance_policy_sha256,
            evidence_as_of=as_of,
        )
        # The manifest is the root of trust for output and in-memory ledger
        # hashes.  Bind its exact bytes to the independently approved frozen-
        # model artifact; otherwise an operator could rewrite the manifest and
        # all self-declared output hashes together.
        try:
            root = Path(self.deployment_root).expanduser().resolve(strict=True)
            manifest_path = Path(self.run_manifest_path).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = root / manifest_path
            manifest_hash = _sha256_file(manifest_path.resolve(strict=True))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._blocked_bundle_report(
                f"run manifest cannot be re-hashed: {exc}"
            )
        frozen_model = report.loc[
            report["gate"].eq("frozen_model_and_change_control")
        ]
        if (
            len(frozen_model) != 1
            or frozen_model.iloc[0]["status"] != PASS
            or frozen_model.iloc[0]["observed"] != manifest_hash
        ):
            return self._blocked_bundle_report(
                "frozen_model_and_change_control evidence must hash the exact run manifest"
            )
        return report


def overall_readiness_status(report: Any) -> str:
    """Return ``READY`` only for the complete, unique readiness schema.

    This deliberately rejects caller-constructed subsets.  A single ``PASS``
    row, a missing/duplicated gate, an extra critical gate, a non-boolean
    critical marker, or a malformed informational row all fail closed.
    """

    if not isinstance(report, pd.DataFrame) or report.empty:
        return BLOCKED
    required_columns = {"gate", "critical", "status"}
    if not required_columns.issubset(report.columns):
        return BLOCKED
    gates = report["gate"]
    if (
        gates.isna().any()
        or not gates.map(lambda value: isinstance(value, str) and bool(value)).all()
        or gates.duplicated(keep=False).any()
        or set(gates) != EXPECTED_READINESS_GATES
    ):
        return BLOCKED
    if not report["critical"].map(
        lambda value: type(value) is bool or isinstance(value, np.bool_)
    ).all():
        return BLOCKED

    indexed = report.set_index("gate")
    critical_rows = indexed.loc[list(REQUIRED_CRITICAL_GATES)]
    if (
        not critical_rows["critical"].map(bool).all()
        or not critical_rows["status"].eq(PASS).all()
    ):
        return BLOCKED
    informational = indexed.loc[list(INFORMATIONAL_READINESS_GATES)]
    if (
        informational["critical"].map(bool).any()
        or not informational["status"].eq("INFORMATIONAL").all()
    ):
        return BLOCKED
    return "READY"


__all__ = [
    "ArtifactBackedReadinessCheck",
    "EXPECTED_READINESS_GATES",
    "EXTERNAL_EVIDENCE_LABELS",
    "HISTORICAL_CRITICAL_GATES",
    "INFORMATIONAL_READINESS_GATES",
    "ProductionLimits",
    "REQUIRED_CRITICAL_GATES",
    "RuntimeHealth",
    "build_order_intents",
    "daily_frame_fingerprint",
    "evaluate_runtime_health",
    "production_readiness_report",
    "overall_readiness_status",
]
