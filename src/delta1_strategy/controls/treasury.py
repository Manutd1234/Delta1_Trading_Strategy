"""Research-neutral treasury evidence records and validation.

This module deliberately does not calculate strategy returns, resize positions,
route orders, or qualify the bundled paper broker for live trading.  It defines
an immutable boundary for externally sourced cash, collateral, settlement,
margin, and funding evidence.  Live integration remains fail-closed until a
broker or clearing adapter can supply and reconcile these records.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd


ZERO_SHA256 = "0" * 64
JOURNAL_EVENT_TYPES = frozenset(
    {
        "VARIATION_MARGIN",
        "COMMISSION_FEE",
        "COLLATERAL_TRANSFER",
        "COLLATERAL_INTEREST",
        "FUNDING_EXPENSE",
        "DEPOSIT_WITHDRAWAL",
        "FX_TRANSLATION",
        "LIQUIDATION_COST",
        "ADJUSTMENT",
    }
)


class TreasuryValidationError(ValueError):
    """Raised when treasury evidence is incomplete, stale, or inconsistent."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _currency(value: Any, label: str) -> str:
    currency = _text(value, label).upper()
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError(f"{label} must be a three-letter currency code")
    return currency


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _fraction(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0 <= result <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _utc(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _date(value: Any, label: str) -> date:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError(f"{label} is invalid")
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO calendar date") from exc


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _utc(value, "timestamp").isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def record_sha256(record: Any) -> str:
    """Return a deterministic hash for one immutable treasury record."""

    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("record must be a dataclass instance")
    encoded = json.dumps(
        {
            "record_type": record.__class__.__qualname__,
            "payload": _json_value(asdict(record)),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SettlementMark:
    """Official serial-contract settlement and its point-in-time FX evidence."""

    contract_id: str
    session_date: date
    settlement_timestamp: pd.Timestamp
    prior_settlement: float
    current_settlement: float
    settlement_currency: str
    base_currency: str
    point_value: float
    fx_to_base: float
    fx_timestamp: pd.Timestamp
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _text(self.contract_id, "contract_id"))
        object.__setattr__(self, "session_date", _date(self.session_date, "session_date"))
        object.__setattr__(
            self,
            "settlement_timestamp",
            _utc(self.settlement_timestamp, "settlement_timestamp"),
        )
        object.__setattr__(self, "prior_settlement", _finite(self.prior_settlement, "prior_settlement"))
        object.__setattr__(self, "current_settlement", _finite(self.current_settlement, "current_settlement"))
        object.__setattr__(
            self,
            "settlement_currency",
            _currency(self.settlement_currency, "settlement_currency"),
        )
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        object.__setattr__(self, "point_value", _positive(self.point_value, "point_value"))
        object.__setattr__(self, "fx_to_base", _positive(self.fx_to_base, "fx_to_base"))
        object.__setattr__(self, "fx_timestamp", _utc(self.fx_timestamp, "fx_timestamp"))
        if self.settlement_currency == self.base_currency and not math.isclose(
            self.fx_to_base, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("base-currency settlements must use fx_to_base=1")
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))


@dataclass(frozen=True)
class MarginQuote:
    """A broker/clearing portfolio-margin quote bound to exact snapshots."""

    account_id: str
    base_currency: str
    generated_at: pd.Timestamp
    effective_from: pd.Timestamp
    expires_at: pd.Timestamp
    position_snapshot_sha256: str
    open_orders_snapshot_sha256: str
    initial_margin_base: float
    maintenance_margin_base: float
    house_addon_base: float
    concentration_addon_base: float
    open_order_reserve_base: float
    margin_model: str
    model_version: str
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _text(self.account_id, "account_id"))
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        object.__setattr__(self, "generated_at", _utc(self.generated_at, "generated_at"))
        object.__setattr__(self, "effective_from", _utc(self.effective_from, "effective_from"))
        object.__setattr__(self, "expires_at", _utc(self.expires_at, "expires_at"))
        if self.effective_from >= self.expires_at:
            raise ValueError("margin quote effective interval must be positive")
        if self.generated_at > self.expires_at:
            raise ValueError("margin quote cannot be generated after it expires")
        for name in ("position_snapshot_sha256", "open_orders_snapshot_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in (
            "initial_margin_base",
            "maintenance_margin_base",
            "house_addon_base",
            "concentration_addon_base",
            "open_order_reserve_base",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.initial_margin_base < self.maintenance_margin_base:
            raise ValueError("initial margin must not be below maintenance margin")
        object.__setattr__(self, "margin_model", _text(self.margin_model, "margin_model"))
        object.__setattr__(self, "model_version", _text(self.model_version, "model_version"))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))

    @property
    def total_initial_margin_base(self) -> float:
        return (
            self.initial_margin_base
            + self.house_addon_base
            + self.concentration_addon_base
            + self.open_order_reserve_base
        )

    @property
    def total_maintenance_margin_base(self) -> float:
        return (
            self.maintenance_margin_base
            + self.house_addon_base
            + self.concentration_addon_base
            + self.open_order_reserve_base
        )


@dataclass(frozen=True)
class CashBalance:
    """One currency subledger in an account snapshot."""

    currency: str
    base_currency: str
    settled_amount: float
    fx_to_base: float
    fx_timestamp: pd.Timestamp
    margin_eligible: bool
    margin_haircut_fraction: float
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _currency(self.currency, "currency"))
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        object.__setattr__(self, "settled_amount", _finite(self.settled_amount, "settled_amount"))
        object.__setattr__(self, "fx_to_base", _positive(self.fx_to_base, "fx_to_base"))
        object.__setattr__(self, "fx_timestamp", _utc(self.fx_timestamp, "fx_timestamp"))
        if type(self.margin_eligible) is not bool:
            raise ValueError("margin_eligible must be a boolean")
        object.__setattr__(
            self,
            "margin_haircut_fraction",
            _fraction(self.margin_haircut_fraction, "margin_haircut_fraction"),
        )
        if self.currency == self.base_currency and not math.isclose(
            self.fx_to_base, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("base-currency cash must use fx_to_base=1")
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))

    @property
    def market_value_base(self) -> float:
        return self.settled_amount * self.fx_to_base

    @property
    def eligible_margin_value_base(self) -> float:
        if not self.margin_eligible or self.settled_amount <= 0:
            return 0.0
        return self.market_value_base * (1.0 - self.margin_haircut_fraction)


@dataclass(frozen=True)
class CollateralLot:
    """An effective-dated non-cash collateral lot with explicit haircut evidence."""

    collateral_id: str
    account_id: str
    asset_id: str
    asset_type: str
    currency: str
    base_currency: str
    market_value_native: float
    fx_to_base: float
    haircut_fraction: float
    eligible: bool
    valuation_timestamp: pd.Timestamp
    effective_from: pd.Timestamp
    effective_to: pd.Timestamp
    concentration_group: str | None
    concentration_limit_base: float | None
    custodian: str
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        for name in ("collateral_id", "account_id", "asset_id", "asset_type", "custodian", "source_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.asset_type.upper() == "CASH":
            raise ValueError("cash belongs in CashBalance, not CollateralLot")
        object.__setattr__(self, "currency", _currency(self.currency, "currency"))
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        object.__setattr__(
            self,
            "market_value_native",
            _nonnegative(self.market_value_native, "market_value_native"),
        )
        object.__setattr__(self, "fx_to_base", _positive(self.fx_to_base, "fx_to_base"))
        object.__setattr__(self, "haircut_fraction", _fraction(self.haircut_fraction, "haircut_fraction"))
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be a boolean")
        object.__setattr__(
            self,
            "valuation_timestamp",
            _utc(self.valuation_timestamp, "valuation_timestamp"),
        )
        object.__setattr__(self, "effective_from", _utc(self.effective_from, "effective_from"))
        object.__setattr__(self, "effective_to", _utc(self.effective_to, "effective_to"))
        if self.effective_from >= self.effective_to:
            raise ValueError("collateral effective interval must be positive")
        if not self.effective_from <= self.valuation_timestamp < self.effective_to:
            raise ValueError("collateral valuation is outside its effective interval")
        if (self.concentration_group is None) != (self.concentration_limit_base is None):
            raise ValueError("concentration group and limit must be supplied together")
        if self.concentration_group is not None:
            object.__setattr__(
                self,
                "concentration_group",
                _text(self.concentration_group, "concentration_group"),
            )
            object.__setattr__(
                self,
                "concentration_limit_base",
                _nonnegative(self.concentration_limit_base, "concentration_limit_base"),
            )
        if self.currency == self.base_currency and not math.isclose(
            self.fx_to_base, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("base-currency collateral must use fx_to_base=1")
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))

    @property
    def market_value_base(self) -> float:
        return self.market_value_native * self.fx_to_base

    @property
    def post_haircut_value_base(self) -> float:
        if not self.eligible:
            return 0.0
        return self.market_value_base * (1.0 - self.haircut_fraction)


@dataclass(frozen=True)
class LedgerPosting:
    """One positive debit or credit posting; signs are represented by ``side``."""

    ledger_account: str
    currency: str
    base_currency: str
    side: str
    amount_native: float
    fx_to_base: float
    contract_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ledger_account", _text(self.ledger_account, "ledger_account"))
        object.__setattr__(self, "currency", _currency(self.currency, "currency"))
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        side = _text(self.side, "side").upper()
        if side not in {"DEBIT", "CREDIT"}:
            raise ValueError("side must be DEBIT or CREDIT")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "amount_native", _positive(self.amount_native, "amount_native"))
        object.__setattr__(self, "fx_to_base", _positive(self.fx_to_base, "fx_to_base"))
        if self.currency == self.base_currency and not math.isclose(
            self.fx_to_base, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("base-currency postings must use fx_to_base=1")
        if self.contract_id is not None:
            object.__setattr__(self, "contract_id", _text(self.contract_id, "contract_id"))

    @property
    def amount_base(self) -> float:
        return self.amount_native * self.fx_to_base


@dataclass(frozen=True)
class JournalEntry:
    """A balanced, append-only treasury journal entry."""

    sequence: int
    entry_id: str
    idempotency_key: str
    account_id: str
    base_currency: str
    event_type: str
    effective_at: pd.Timestamp
    recorded_at: pd.Timestamp
    value_date: date
    postings: tuple[LedgerPosting, ...]
    previous_entry_sha256: str
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        for name in ("entry_id", "idempotency_key", "account_id", "source_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        event_type = _text(self.event_type, "event_type").upper()
        if event_type not in JOURNAL_EVENT_TYPES:
            raise ValueError("unsupported treasury journal event_type")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "effective_at", _utc(self.effective_at, "effective_at"))
        object.__setattr__(self, "recorded_at", _utc(self.recorded_at, "recorded_at"))
        if self.recorded_at < self.effective_at:
            raise ValueError("recorded_at cannot predate effective_at")
        object.__setattr__(self, "value_date", _date(self.value_date, "value_date"))
        try:
            postings = tuple(self.postings)
        except TypeError as exc:
            raise ValueError("postings must be an iterable of LedgerPosting records") from exc
        if len(postings) < 2 or any(not isinstance(posting, LedgerPosting) for posting in postings):
            raise ValueError("a journal entry requires at least two LedgerPosting records")
        if any(posting.base_currency != self.base_currency for posting in postings):
            raise ValueError("posting base currencies must match the journal entry")
        object.__setattr__(self, "postings", postings)
        object.__setattr__(
            self,
            "previous_entry_sha256",
            _sha256(self.previous_entry_sha256, "previous_entry_sha256"),
        )
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))

        currencies = {posting.currency for posting in postings}
        for currency in currencies:
            debit = sum(
                posting.amount_native
                for posting in postings
                if posting.currency == currency and posting.side == "DEBIT"
            )
            credit = sum(
                posting.amount_native
                for posting in postings
                if posting.currency == currency and posting.side == "CREDIT"
            )
            if not math.isclose(debit, credit, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(f"journal entry is not balanced in {currency}")
        debit_base = sum(posting.amount_base for posting in postings if posting.side == "DEBIT")
        credit_base = sum(posting.amount_base for posting in postings if posting.side == "CREDIT")
        if not math.isclose(debit_base, credit_base, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("journal entry is not balanced in base currency")


@dataclass(frozen=True)
class AccountSnapshot:
    """Point-in-time account, liquidity, and margin state from the broker."""

    account_id: str
    base_currency: str
    as_of: pd.Timestamp
    cash_balances: tuple[CashBalance, ...]
    collateral_lots: tuple[CollateralLot, ...]
    unsettled_vm_receivable_base: float
    unsettled_vm_payable_base: float
    pending_fees_base: float
    other_assets_liabilities_base: float
    account_equity_base: float
    initial_margin_requirement_base: float
    maintenance_margin_requirement_base: float
    broker_available_funds_base: float
    broker_excess_liquidity_base: float
    margin_call_base: float
    position_snapshot_sha256: str
    open_orders_snapshot_sha256: str
    margin_quote_sha256: str
    journal_head_sha256: str
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _text(self.account_id, "account_id"))
        object.__setattr__(self, "base_currency", _currency(self.base_currency, "base_currency"))
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        try:
            cash = tuple(self.cash_balances)
            collateral = tuple(self.collateral_lots)
        except TypeError as exc:
            raise ValueError("cash_balances and collateral_lots must be iterable") from exc
        if any(not isinstance(item, CashBalance) for item in cash):
            raise ValueError("cash_balances must contain CashBalance records")
        if any(not isinstance(item, CollateralLot) for item in collateral):
            raise ValueError("collateral_lots must contain CollateralLot records")
        if len({item.currency for item in cash}) != len(cash):
            raise ValueError("cash_balances contains duplicate currencies")
        if len({item.collateral_id for item in collateral}) != len(collateral):
            raise ValueError("collateral_lots contains duplicate collateral IDs")
        if any(item.base_currency != self.base_currency for item in cash + collateral):
            raise ValueError("cash and collateral base currencies must match the account")
        if any(item.account_id != self.account_id for item in collateral):
            raise ValueError("collateral account IDs must match the account snapshot")
        if any(item.fx_timestamp > self.as_of for item in cash):
            raise ValueError("cash FX evidence cannot postdate the account snapshot")
        if any(item.valuation_timestamp > self.as_of for item in collateral):
            raise ValueError("collateral valuation cannot postdate the account snapshot")

        concentration_limits: dict[str, float] = {}
        for lot in collateral:
            if lot.concentration_group is None:
                continue
            prior = concentration_limits.setdefault(
                lot.concentration_group, float(lot.concentration_limit_base)
            )
            if not math.isclose(
                prior,
                float(lot.concentration_limit_base),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError("one concentration group has conflicting limits")
        object.__setattr__(self, "cash_balances", cash)
        object.__setattr__(self, "collateral_lots", collateral)

        for name in (
            "unsettled_vm_receivable_base",
            "unsettled_vm_payable_base",
            "pending_fees_base",
            "initial_margin_requirement_base",
            "maintenance_margin_requirement_base",
            "margin_call_base",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "other_assets_liabilities_base",
            "account_equity_base",
            "broker_available_funds_base",
            "broker_excess_liquidity_base",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.initial_margin_requirement_base < self.maintenance_margin_requirement_base:
            raise ValueError("account initial margin must not be below maintenance margin")
        for name in (
            "position_snapshot_sha256",
            "open_orders_snapshot_sha256",
            "margin_quote_sha256",
            "journal_head_sha256",
            "source_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))

    @property
    def settled_cash_base(self) -> float:
        return sum(balance.market_value_base for balance in self.cash_balances)

    @property
    def collateral_market_value_base(self) -> float:
        return sum(lot.market_value_base for lot in self.collateral_lots)

    @property
    def eligible_collateral_base(self) -> float:
        uncapped = sum(
            lot.post_haircut_value_base
            for lot in self.collateral_lots
            if lot.concentration_group is None
        )
        grouped: dict[str, float] = {}
        limits: dict[str, float] = {}
        for lot in self.collateral_lots:
            if lot.concentration_group is None:
                continue
            grouped[lot.concentration_group] = (
                grouped.get(lot.concentration_group, 0.0) + lot.post_haircut_value_base
            )
            limits[lot.concentration_group] = float(lot.concentration_limit_base)
        return uncapped + sum(min(value, limits[group]) for group, value in grouped.items())

    @property
    def eligible_margin_resources_base(self) -> float:
        eligible_cash = sum(balance.eligible_margin_value_base for balance in self.cash_balances)
        return eligible_cash + self.eligible_collateral_base

    @property
    def computed_account_equity_base(self) -> float:
        return (
            self.settled_cash_base
            + self.collateral_market_value_base
            + self.unsettled_vm_receivable_base
            - self.unsettled_vm_payable_base
            - self.pending_fees_base
            + self.other_assets_liabilities_base
        )


@dataclass(frozen=True)
class TreasuryEvidenceBundle:
    """Evidence that must validate as one content-addressed account state."""

    account_snapshot: AccountSnapshot
    margin_quote: MarginQuote
    settlement_marks: tuple[SettlementMark, ...]
    journal_entries: tuple[JournalEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.account_snapshot, AccountSnapshot):
            raise TypeError("account_snapshot must be an AccountSnapshot")
        if not isinstance(self.margin_quote, MarginQuote):
            raise TypeError("margin_quote must be a MarginQuote")
        marks = tuple(self.settlement_marks)
        entries = tuple(self.journal_entries)
        if any(not isinstance(mark, SettlementMark) for mark in marks):
            raise TypeError("settlement_marks must contain SettlementMark records")
        if any(not isinstance(entry, JournalEntry) for entry in entries):
            raise TypeError("journal_entries must contain JournalEntry records")
        if len({mark.contract_id for mark in marks}) != len(marks):
            raise ValueError("settlement_marks contains duplicate contract IDs")
        object.__setattr__(self, "settlement_marks", marks)
        object.__setattr__(self, "journal_entries", entries)


@dataclass(frozen=True)
class TreasuryFreshnessLimits:
    """Caller-approved evidence ages; no production limits are assumed here."""

    max_account_age_seconds: float
    max_margin_quote_age_seconds: float
    max_settlement_age_seconds: float
    max_fx_age_seconds: float
    max_collateral_age_seconds: float
    monetary_tolerance_base: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "max_account_age_seconds",
            "max_margin_quote_age_seconds",
            "max_settlement_age_seconds",
            "max_fx_age_seconds",
            "max_collateral_age_seconds",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "monetary_tolerance_base",
            _nonnegative(self.monetary_tolerance_base, "monetary_tolerance_base"),
        )


@runtime_checkable
class TreasuryEvidenceProvider(Protocol):
    """Read-only boundary for a future broker/clearing implementation."""

    def account_snapshot(self, *, as_of: pd.Timestamp) -> AccountSnapshot: ...

    def portfolio_margin_quote(
        self,
        *,
        as_of: pd.Timestamp,
        position_snapshot_sha256: str,
        open_orders_snapshot_sha256: str,
    ) -> MarginQuote: ...

    def settlement_marks(
        self, *, as_of: pd.Timestamp, contract_ids: Sequence[str]
    ) -> Sequence[SettlementMark]: ...

    def journal_entries(self, *, through: pd.Timestamp) -> Sequence[JournalEntry]: ...


def validate_journal_chain(
    entries: Sequence[JournalEntry],
    *,
    expected_previous_sha256: str = ZERO_SHA256,
    expected_start_sequence: int = 1,
) -> str:
    """Validate order, identity, and hash continuity; return the head digest."""

    previous = _sha256(expected_previous_sha256, "expected_previous_sha256")
    if (
        isinstance(expected_start_sequence, bool)
        or not isinstance(expected_start_sequence, int)
        or expected_start_sequence < 1
    ):
        raise TreasuryValidationError("expected_start_sequence must be a positive integer")
    entry_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    prior_recorded_at: pd.Timestamp | None = None
    for offset, entry in enumerate(entries):
        if not isinstance(entry, JournalEntry):
            raise TreasuryValidationError("journal chain contains a non-JournalEntry value")
        if entry.sequence != expected_start_sequence + offset:
            raise TreasuryValidationError("journal sequence is not contiguous")
        if entry.entry_id in entry_ids or entry.idempotency_key in idempotency_keys:
            raise TreasuryValidationError("journal identity or idempotency key is duplicated")
        if entry.previous_entry_sha256 != previous:
            raise TreasuryValidationError("journal previous-entry hash does not reconcile")
        if prior_recorded_at is not None and entry.recorded_at < prior_recorded_at:
            raise TreasuryValidationError("journal recorded timestamps are not monotonic")
        entry_ids.add(entry.entry_id)
        idempotency_keys.add(entry.idempotency_key)
        prior_recorded_at = entry.recorded_at
        previous = record_sha256(entry)
    return previous


def _fresh(timestamp: pd.Timestamp, *, as_of: pd.Timestamp, maximum_age: float, label: str) -> None:
    age = float((as_of - timestamp).total_seconds())
    if age < 0:
        raise TreasuryValidationError(f"{label} is in the future")
    if age > maximum_age:
        raise TreasuryValidationError(f"{label} is stale")


def validate_treasury_evidence(
    bundle: TreasuryEvidenceBundle,
    *,
    as_of: Any,
    limits: TreasuryFreshnessLimits,
    expected_position_snapshot_sha256: str,
    expected_open_orders_snapshot_sha256: str,
    expected_contract_ids: Sequence[str],
) -> TreasuryEvidenceBundle:
    """Validate one treasury state or raise; no partial/pass-with-warning result exists."""

    if not isinstance(bundle, TreasuryEvidenceBundle):
        raise TypeError("bundle must be a TreasuryEvidenceBundle")
    if not isinstance(limits, TreasuryFreshnessLimits):
        raise TypeError("limits must be TreasuryFreshnessLimits")
    evaluation_time = _utc(as_of, "as_of")
    expected_position = _sha256(
        expected_position_snapshot_sha256, "expected_position_snapshot_sha256"
    )
    expected_orders = _sha256(
        expected_open_orders_snapshot_sha256, "expected_open_orders_snapshot_sha256"
    )
    account = bundle.account_snapshot
    quote = bundle.margin_quote

    if quote.effective_from > evaluation_time:
        raise TreasuryValidationError("margin quote is not yet effective")
    if evaluation_time >= quote.expires_at:
        raise TreasuryValidationError("margin quote has expired")
    _fresh(
        quote.generated_at,
        as_of=evaluation_time,
        maximum_age=limits.max_margin_quote_age_seconds,
        label="margin quote",
    )
    _fresh(
        account.as_of,
        as_of=evaluation_time,
        maximum_age=limits.max_account_age_seconds,
        label="account snapshot",
    )
    if account.as_of < quote.generated_at:
        raise TreasuryValidationError("account snapshot predates its margin quote")

    if account.account_id != quote.account_id or account.base_currency != quote.base_currency:
        raise TreasuryValidationError("account and margin quote identity do not match")
    for label, observed in (
        ("account position snapshot", account.position_snapshot_sha256),
        ("margin quote position snapshot", quote.position_snapshot_sha256),
    ):
        if observed != expected_position:
            raise TreasuryValidationError(f"{label} hash does not match expected state")
    for label, observed in (
        ("account open-orders snapshot", account.open_orders_snapshot_sha256),
        ("margin quote open-orders snapshot", quote.open_orders_snapshot_sha256),
    ):
        if observed != expected_orders:
            raise TreasuryValidationError(f"{label} hash does not match expected state")
    if account.margin_quote_sha256 != record_sha256(quote):
        raise TreasuryValidationError("account margin-quote hash does not reconcile")

    journal_head = validate_journal_chain(bundle.journal_entries)
    if account.journal_head_sha256 != journal_head:
        raise TreasuryValidationError("account journal-head hash does not reconcile")
    for entry in bundle.journal_entries:
        if entry.account_id != account.account_id or entry.base_currency != account.base_currency:
            raise TreasuryValidationError("journal identity does not match the account")
        if entry.effective_at > evaluation_time or entry.recorded_at > evaluation_time:
            raise TreasuryValidationError("journal contains a future event")
        if entry.recorded_at > account.as_of:
            raise TreasuryValidationError("account snapshot predates a journal event")

    tolerance = limits.monetary_tolerance_base
    if not math.isclose(
        account.initial_margin_requirement_base,
        quote.total_initial_margin_base,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise TreasuryValidationError("account initial margin does not match the margin quote")
    if not math.isclose(
        account.maintenance_margin_requirement_base,
        quote.total_maintenance_margin_base,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise TreasuryValidationError("account maintenance margin does not match the margin quote")
    if not math.isclose(
        account.account_equity_base,
        account.computed_account_equity_base,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise TreasuryValidationError("account equity roll-forward does not reconcile")

    try:
        required_contracts = tuple(_text(value, "expected_contract_id") for value in expected_contract_ids)
    except TypeError as exc:
        raise TreasuryValidationError("expected_contract_ids must be iterable") from exc
    if len(set(required_contracts)) != len(required_contracts):
        raise TreasuryValidationError("expected_contract_ids contains duplicates")
    marks = {mark.contract_id: mark for mark in bundle.settlement_marks}
    missing = sorted(set(required_contracts).difference(marks))
    if missing:
        raise TreasuryValidationError(
            "settlement evidence is missing contracts: " + ", ".join(missing)
        )
    for mark in bundle.settlement_marks:
        if mark.base_currency != account.base_currency:
            raise TreasuryValidationError("settlement base currency does not match the account")
        _fresh(
            mark.settlement_timestamp,
            as_of=evaluation_time,
            maximum_age=limits.max_settlement_age_seconds,
            label=f"settlement mark {mark.contract_id}",
        )
        _fresh(
            mark.fx_timestamp,
            as_of=evaluation_time,
            maximum_age=limits.max_fx_age_seconds,
            label=f"settlement FX {mark.contract_id}",
        )

    for balance in account.cash_balances:
        _fresh(
            balance.fx_timestamp,
            as_of=evaluation_time,
            maximum_age=limits.max_fx_age_seconds,
            label=f"cash FX {balance.currency}",
        )
    for lot in account.collateral_lots:
        if not lot.effective_from <= evaluation_time < lot.effective_to:
            raise TreasuryValidationError(
                f"collateral {lot.collateral_id} is outside its effective interval"
            )
        _fresh(
            lot.valuation_timestamp,
            as_of=evaluation_time,
            maximum_age=limits.max_collateral_age_seconds,
            label=f"collateral valuation {lot.collateral_id}",
        )

    return bundle


__all__ = [
    "AccountSnapshot",
    "CashBalance",
    "CollateralLot",
    "JOURNAL_EVENT_TYPES",
    "JournalEntry",
    "LedgerPosting",
    "MarginQuote",
    "SettlementMark",
    "TreasuryEvidenceBundle",
    "TreasuryEvidenceProvider",
    "TreasuryFreshnessLimits",
    "TreasuryValidationError",
    "ZERO_SHA256",
    "record_sha256",
    "validate_journal_chain",
    "validate_treasury_evidence",
]
