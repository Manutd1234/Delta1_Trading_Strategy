"""Serial-contract market-data validation and delivery-safe roll planning.

The research engine uses continuous futures because that is the only history
shipped with the case data.  This module defines the stricter contract-level
boundary required by a paper or live execution service.  It deliberately
refuses continuous symbols: every order must identify an exchange-listed
expiry and must be checked against the liquidity of both roll legs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd


PASS = "PASS"
BLOCKED = "BLOCKED"


SERIAL_CONTRACT_COLUMNS = (
    "timestamp",
    "session_date",
    "root_symbol",
    "contract_id",
    "exchange",
    "currency",
    "delivery_type",
    "expiry",
    "first_notice_date",
    "last_trade_date",
    "broker_liquidation_cutoff",
    "bid",
    "ask",
    "settlement",
    "volume",
    "open_interest",
    "point_value",
    "notional_per_contract_usd",
    "tick_size",
    "margin_per_contract",
    "margin_per_contract_usd",
    "commission_per_contract",
    "exchange_fee_per_contract",
    "spec_effective_from",
    "spec_effective_to",
)


ROLL_PLAN_COLUMNS = (
    "roll_id",
    "decision_timestamp",
    "root_symbol",
    "from_contract",
    "to_contract",
    "position_contracts",
    "requested_contracts",
    "capacity_contracts",
    "roll_contracts",
    "from_side",
    "to_side",
    "from_volume",
    "to_volume",
    "from_participation",
    "to_participation",
    "liquidation_deadline",
    "destination_liquidation_deadline",
    "trigger_date",
    "approved",
    "status",
    "reason",
)


ROLL_LEG_COLUMNS = (
    "roll_id",
    "decision_timestamp",
    "root_symbol",
    "leg",
    "contract_id",
    "side",
    "quantity",
    "reported_volume",
    "projected_participation",
    "liquidation_deadline",
    "approved",
    "status",
)


class ContractDataError(ValueError):
    """Raised when a serial-contract snapshot cannot be used safely."""


@dataclass(frozen=True)
class SerialDataLimits:
    """Hard limits applied to serial market data and roll plans."""

    max_data_age_seconds: float = 300.0
    max_order_participation: float = 0.02
    roll_calendar_days_before_deadline: int = 7

    def __post_init__(self) -> None:
        if not _finite(self.max_data_age_seconds) or self.max_data_age_seconds <= 0:
            raise ValueError("max_data_age_seconds must be finite and positive")
        if (
            not _finite(self.max_order_participation)
            or not 0 < self.max_order_participation <= 1
        ):
            raise ValueError("max_order_participation must be in (0, 1]")
        if (
            isinstance(self.roll_calendar_days_before_deadline, bool)
            or not isinstance(self.roll_calendar_days_before_deadline, int)
            or self.roll_calendar_days_before_deadline < 1
        ):
            raise ValueError("roll_calendar_days_before_deadline must be a positive integer")


def _finite(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _aware_utc(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractDataError(f"{label} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ContractDataError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _row(gate: str, passed: bool, observed: Any, requirement: Any, reason: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "critical": True,
        "status": PASS if passed else BLOCKED,
        "observed": observed,
        "requirement": requirement,
        "reason": "gate passed" if passed else reason,
    }


def serial_snapshot_validation_report(
    snapshot: Any,
    *,
    as_of: Any,
    limits: SerialDataLimits = SerialDataLimits(),
) -> pd.DataFrame:
    """Return fail-closed validation gates for one point-in-time snapshot.

    The report is intentionally strict.  In particular, naive timestamps and
    present-day specifications without an effective interval are rejected.
    Negative futures prices are allowed, but crossed or non-finite quotes are
    not.
    """

    if not isinstance(limits, SerialDataLimits):
        raise TypeError("limits must be a SerialDataLimits instance")
    rows: list[dict[str, Any]] = []
    frame_ok = isinstance(snapshot, pd.DataFrame) and not snapshot.empty
    rows.append(
        _row(
            "serial_snapshot_available",
            frame_ok,
            len(snapshot) if isinstance(snapshot, pd.DataFrame) else None,
            "> 0 rows",
            "serial-contract snapshot is missing or empty",
        )
    )
    frame = snapshot.copy() if frame_ok else pd.DataFrame()
    missing = [column for column in SERIAL_CONTRACT_COLUMNS if column not in frame]
    columns_ok = frame_ok and not missing
    rows.append(
        _row(
            "serial_required_columns",
            columns_ok,
            ", ".join(missing) if missing else "all present",
            "all serial-contract columns",
            "required serial-contract columns are missing",
        )
    )
    try:
        evaluation_time = _aware_utc(as_of, "as_of")
        as_of_ok = True
    except ContractDataError:
        evaluation_time = None
        as_of_ok = False
    rows.append(
        _row(
            "serial_as_of_timestamp",
            as_of_ok,
            evaluation_time.isoformat() if evaluation_time is not None else None,
            "timezone-aware UTC-convertible timestamp",
            "as_of is invalid or timezone-naive",
        )
    )
    if not columns_ok:
        return pd.DataFrame(rows)

    contract_text = frame["contract_id"].astype("string").str.strip()
    root_text = frame["root_symbol"].astype("string").str.strip()
    identifiers_ok = bool(
        contract_text.notna().all()
        and root_text.notna().all()
        and contract_text.ne("").all()
        and root_text.ne("").all()
        and ~contract_text.duplicated().any()
        and contract_text.ne(root_text).all()
    )
    rows.append(
        _row(
            "serial_contract_identifiers",
            identifiers_ok,
            int(contract_text.nunique(dropna=True)),
            "unique non-continuous contract IDs",
            "contract IDs are blank, duplicated, or equal to continuous root symbols",
        )
    )

    exchange_text = frame["exchange"].astype("string").str.strip()
    currency_text = frame["currency"].astype("string").str.strip()
    venue_currency_ok = bool(
        exchange_text.notna().all()
        and exchange_text.ne("").all()
        and currency_text.notna().all()
        and currency_text.ne("").all()
    )
    rows.append(
        _row(
            "serial_venue_currency",
            venue_currency_ok,
            "present" if venue_currency_ok else None,
            "nonblank exchange and currency for every contract",
            "exchange or currency is missing or blank",
        )
    )

    raw_timestamps = frame["timestamp"]
    timezone_aware = True
    parsed_timestamps: list[pd.Timestamp] = []
    for value in raw_timestamps:
        try:
            parsed_timestamps.append(_aware_utc(value, "snapshot timestamp"))
        except ContractDataError:
            timezone_aware = False
            break
    rows.append(
        _row(
            "serial_timestamp_timezone",
            timezone_aware,
            "timezone-aware" if timezone_aware else None,
            "all timestamps timezone-aware",
            "one or more market-data timestamps are invalid or timezone-naive",
        )
    )
    age_ok = False
    maximum_age: float | None = None
    if timezone_aware and evaluation_time is not None:
        ages = np.array(
            [(evaluation_time - timestamp).total_seconds() for timestamp in parsed_timestamps],
            dtype=float,
        )
        maximum_age = float(np.max(ages))
        age_ok = bool((ages >= 0).all() and (ages <= limits.max_data_age_seconds).all())
    rows.append(
        _row(
            "serial_data_freshness",
            age_ok,
            maximum_age,
            limits.max_data_age_seconds,
            "market data is stale or timestamped in the future",
        )
    )

    session_dates = pd.to_datetime(frame["session_date"], errors="coerce")
    session_ok = bool(session_dates.notna().all())
    if session_ok and parsed_timestamps:
        utc_dates = pd.Series([timestamp.tz_convert("UTC").date() for timestamp in parsed_timestamps])
        # A venue session can end after midnight UTC, so tolerate one calendar day.
        deltas = np.abs(
            (pd.to_datetime(utc_dates) - session_dates.reset_index(drop=True).dt.normalize())
            .dt.days.to_numpy()
        )
        session_ok = bool((deltas <= 1).all())
    rows.append(
        _row(
            "serial_session_dates",
            session_ok,
            "valid" if session_ok else None,
            "valid session date within one UTC day",
            "session dates are invalid or inconsistent with timestamps",
        )
    )

    numeric = {}
    numeric_columns = (
        "bid",
        "ask",
        "settlement",
        "volume",
        "open_interest",
        "point_value",
        "notional_per_contract_usd",
        "tick_size",
        "margin_per_contract",
        "margin_per_contract_usd",
        "commission_per_contract",
        "exchange_fee_per_contract",
    )
    for column in numeric_columns:
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_numeric = bool(
        all(np.isfinite(values.to_numpy(dtype=float)).all() for values in numeric.values())
    )
    market_values_ok = bool(
        finite_numeric
        and (numeric["ask"] >= numeric["bid"]).all()
        and (numeric["volume"] >= 0).all()
        and (numeric["open_interest"] >= 0).all()
        and (numeric["point_value"] > 0).all()
        and (numeric["notional_per_contract_usd"] > 0).all()
        and (numeric["tick_size"] > 0).all()
        and (numeric["margin_per_contract"] > 0).all()
        and (numeric["margin_per_contract_usd"] > 0).all()
        and (numeric["commission_per_contract"] >= 0).all()
        and (numeric["exchange_fee_per_contract"] >= 0).all()
    )
    rows.append(
        _row(
            "serial_market_values",
            market_values_ok,
            "finite and internally consistent" if market_values_ok else None,
            (
                "finite quotes/specs, ask >= bid, positive point value/USD notional/"
                "native and USD margin, "
                "nonnegative liquidity/costs"
            ),
            "quotes, liquidity, or specifications are invalid",
        )
    )

    expiry = pd.to_datetime(frame["expiry"], errors="coerce").dt.normalize()
    notice = pd.to_datetime(frame["first_notice_date"], errors="coerce").dt.normalize()
    last_trade = pd.to_datetime(frame["last_trade_date"], errors="coerce").dt.normalize()
    expiry_ok = bool(
        expiry.notna().all()
        and last_trade.notna().all()
        and (last_trade <= expiry).all()
        and ((notice.isna()) | (notice <= expiry)).all()
    )
    rows.append(
        _row(
            "serial_expiry_fields",
            expiry_ok,
            "valid" if expiry_ok else None,
            "dated expiry and last-trade; optional first notice no later than expiry",
            "expiry, first-notice, or last-trade fields are invalid",
        )
    )

    delivery_type = frame["delivery_type"].astype("string").str.upper().str.strip()
    cutoffs: list[pd.Timestamp] = []
    cutoff_ok = True
    for value in frame["broker_liquidation_cutoff"]:
        try:
            cutoffs.append(_aware_utc(value, "broker_liquidation_cutoff"))
        except ContractDataError:
            cutoff_ok = False
            break
    delivery_ok = bool(
        cutoff_ok
        and delivery_type.isin(["PHYSICAL", "CASH_SETTLED"]).all()
        and len(cutoffs) == len(frame)
    )
    if delivery_ok:
        for row_number, cutoff in enumerate(cutoffs):
            cutoff_date = cutoff.tz_localize(None).normalize()
            row_delivery = str(delivery_type.iloc[row_number])
            row_notice = notice.iloc[row_number]
            row_last_trade = last_trade.iloc[row_number]
            if pd.isna(row_last_trade) or cutoff_date > row_last_trade:
                delivery_ok = False
                break
            if row_delivery == "PHYSICAL" and (
                pd.isna(row_notice) or cutoff_date >= row_notice
            ):
                delivery_ok = False
                break
    rows.append(
        _row(
            "serial_delivery_controls",
            delivery_ok,
            "valid" if delivery_ok else None,
            (
                "recognized delivery type and aware broker cutoff no later than "
                "last trade; physical cutoff strictly before first notice"
            ),
            "delivery type, first notice, or broker liquidation cutoff is unsafe",
        )
    )

    effective_from: list[pd.Timestamp] = []
    effective_to: list[pd.Timestamp] = []
    effective_ok = evaluation_time is not None
    if effective_ok:
        for start, end in zip(
            frame["spec_effective_from"], frame["spec_effective_to"], strict=True
        ):
            try:
                start_time = _aware_utc(start, "spec_effective_from")
                end_time = _aware_utc(end, "spec_effective_to")
            except ContractDataError:
                effective_ok = False
                break
            effective_from.append(start_time)
            effective_to.append(end_time)
            if not start_time <= evaluation_time < end_time:
                effective_ok = False
                break
    rows.append(
        _row(
            "serial_effective_dated_specs",
            effective_ok,
            "effective at as_of" if effective_ok else None,
            "spec_effective_from <= as_of < spec_effective_to",
            "contract specifications are stale, future-dated, or lack aware effective dates",
        )
    )
    return pd.DataFrame(rows)


def validated_serial_snapshot(
    snapshot: pd.DataFrame,
    *,
    as_of: Any,
    limits: SerialDataLimits = SerialDataLimits(),
) -> pd.DataFrame:
    """Validate and normalize a serial snapshot, or raise ``ContractDataError``."""

    report = serial_snapshot_validation_report(snapshot, as_of=as_of, limits=limits)
    blocked = report.loc[report["status"].ne(PASS), "gate"].astype(str).tolist()
    if blocked:
        raise ContractDataError("serial snapshot failed: " + ", ".join(blocked))
    frame = snapshot.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    for column in ("expiry", "first_notice_date", "last_trade_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["broker_liquidation_cutoff"] = pd.to_datetime(
        frame["broker_liquidation_cutoff"], utc=True
    )
    for column in ("spec_effective_from", "spec_effective_to"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in (
        "bid",
        "ask",
        "settlement",
        "volume",
        "open_interest",
        "point_value",
        "notional_per_contract_usd",
        "tick_size",
        "margin_per_contract",
        "margin_per_contract_usd",
        "commission_per_contract",
        "exchange_fee_per_contract",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in (
        "contract_id",
        "root_symbol",
        "exchange",
        "currency",
        "delivery_type",
    ):
        frame[column] = frame[column].astype(str).str.strip()
    frame["delivery_type"] = frame["delivery_type"].str.upper()
    return frame.sort_values(["root_symbol", "last_trade_date", "contract_id"]).reset_index(
        drop=True
    )


def _positions(values: pd.Series | Mapping[Any, Any]) -> pd.Series:
    if isinstance(values, pd.Series):
        positions = values.copy()
    elif isinstance(values, Mapping):
        positions = pd.Series(dict(values), dtype=object)
    else:
        raise TypeError("positions must be a pandas Series or mapping")
    positions.index = positions.index.map(str)
    if positions.index.has_duplicates:
        raise ValueError("positions contains duplicate contract IDs")
    numeric = pd.to_numeric(positions, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("positions contains missing or non-finite quantities")
    if not np.equal(numeric.to_numpy(), np.rint(numeric.to_numpy())).all():
        raise ValueError("positions must contain integer contract quantities")
    return numeric.astype(int)


def build_roll_plans(
    positions: pd.Series | Mapping[Any, Any],
    snapshot: pd.DataFrame,
    *,
    as_of: Any,
    limits: SerialDataLimits = SerialDataLimits(),
) -> pd.DataFrame:
    """Build two-leg roll plans capped by the thinner serial contract.

    A plan is emitted for every held contract.  Before its trigger date it is
    marked ``NOT_DUE``.  At or after the liquidation deadline it is blocked
    rather than pretending a late roll is safe.  Partial plans can be carried
    by the caller into the next execution cycle.
    """

    decision_time = _aware_utc(as_of, "as_of")
    frame = validated_serial_snapshot(snapshot, as_of=decision_time, limits=limits)
    held = _positions(positions)
    held = held.loc[held.ne(0)]
    if held.empty:
        return pd.DataFrame(columns=ROLL_PLAN_COLUMNS)
    lookup = frame.set_index("contract_id", drop=False)
    rows: list[dict[str, Any]] = []
    for contract_id, position in held.sort_index().items():
        base = {
            "decision_timestamp": decision_time,
            "from_contract": contract_id,
            "position_contracts": int(position),
            "requested_contracts": abs(int(position)),
        }
        if contract_id not in lookup.index:
            rows.append(
                {
                    **base,
                    "roll_id": f"roll-missing-{contract_id}-{decision_time:%Y%m%dT%H%M%SZ}",
                    "root_symbol": None,
                    "to_contract": None,
                    "capacity_contracts": 0,
                    "roll_contracts": 0,
                    "from_side": "SELL" if position > 0 else "BUY",
                    "to_side": "BUY" if position > 0 else "SELL",
                    "from_volume": np.nan,
                    "to_volume": np.nan,
                    "from_participation": np.nan,
                    "to_participation": np.nan,
                    "liquidation_deadline": pd.NaT,
                    "destination_liquidation_deadline": pd.NaT,
                    "trigger_date": pd.NaT,
                    "approved": False,
                    "status": BLOCKED,
                    "reason": "held contract is absent from the serial snapshot",
                }
            )
            continue
        current = lookup.loc[contract_id]
        root = str(current["root_symbol"])
        deadline = pd.Timestamp(current["broker_liquidation_cutoff"])
        trigger = deadline - pd.Timedelta(days=limits.roll_calendar_days_before_deadline)
        minimum_destination_deadline = decision_time + pd.Timedelta(
            days=limits.roll_calendar_days_before_deadline
        )
        candidate_deadline = frame["broker_liquidation_cutoff"]
        candidates = frame.loc[
            frame["root_symbol"].eq(root)
            & frame["exchange"].eq(current["exchange"])
            & frame["currency"].eq(current["currency"])
            & frame["last_trade_date"].gt(current["last_trade_date"])
            & candidate_deadline.gt(deadline)
            & candidate_deadline.ge(minimum_destination_deadline)
        ].sort_values(["last_trade_date", "contract_id"])
        next_row = candidates.iloc[0] if not candidates.empty else None
        destination_deadline = (
            pd.Timestamp(candidate_deadline.loc[next_row.name])
            if next_row is not None
            else pd.NaT
        )
        from_volume = float(current["volume"])
        to_volume = float(next_row["volume"]) if next_row is not None else math.nan
        requested = abs(int(position))
        old_capacity = math.floor(limits.max_order_participation * from_volume)
        new_capacity = (
            math.floor(limits.max_order_participation * to_volume)
            if next_row is not None
            else 0
        )
        capacity = min(old_capacity, new_capacity)
        quantity = min(requested, capacity)
        from_participation = quantity / from_volume if from_volume > 0 else math.nan
        to_participation = quantity / to_volume if to_volume > 0 else math.nan
        due = decision_time >= trigger
        too_late = decision_time >= deadline
        if next_row is None:
            approved = False
            status = BLOCKED
            reason = "no delivery-safe later serial contract is available"
            quantity = 0
        elif too_late:
            approved = False
            status = BLOCKED
            reason = "liquidation deadline has been reached; trigger emergency escalation"
            quantity = 0
        elif not due:
            approved = False
            status = "NOT_DUE"
            reason = "roll trigger date has not been reached"
            quantity = 0
        elif capacity < 1:
            approved = False
            status = BLOCKED
            reason = "two-leg participation capacity permits fewer than one contract"
            quantity = 0
        elif quantity < requested:
            approved = True
            status = "APPROVED_PARTIAL"
            reason = "roll capped by the thinner leg; residual must remain pending"
        else:
            approved = True
            status = "APPROVED"
            reason = "both roll legs pass delivery and participation controls"
        if quantity == 0:
            from_participation = 0.0 if from_volume > 0 else math.nan
            to_participation = 0.0 if to_volume > 0 else math.nan
        to_contract = str(next_row["contract_id"]) if next_row is not None else None
        roll_id = (
            f"roll-{root}-{contract_id}-{to_contract or 'missing'}-"
            f"{decision_time:%Y%m%dT%H%M%SZ}"
        )
        rows.append(
            {
                **base,
                "roll_id": roll_id,
                "root_symbol": root,
                "to_contract": to_contract,
                "capacity_contracts": int(capacity),
                "roll_contracts": int(quantity),
                "from_side": "SELL" if position > 0 else "BUY",
                "to_side": "BUY" if position > 0 else "SELL",
                "from_volume": from_volume,
                "to_volume": to_volume,
                "from_participation": from_participation,
                "to_participation": to_participation,
                "liquidation_deadline": deadline,
                "destination_liquidation_deadline": destination_deadline,
                "trigger_date": trigger,
                "approved": approved,
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).reindex(columns=ROLL_PLAN_COLUMNS)


def expand_roll_legs(plans: pd.DataFrame) -> pd.DataFrame:
    """Expand approved roll pairs into individually auditable physical legs."""

    if not isinstance(plans, pd.DataFrame):
        raise TypeError("plans must be a DataFrame")
    missing = [column for column in ROLL_PLAN_COLUMNS if column not in plans]
    if missing:
        raise ValueError("roll plans missing columns: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    for plan in plans.itertuples(index=False):
        if not bool(plan.approved) or int(plan.roll_contracts) <= 0:
            continue
        common = {
            "roll_id": plan.roll_id,
            "decision_timestamp": plan.decision_timestamp,
            "root_symbol": plan.root_symbol,
            "quantity": int(plan.roll_contracts),
            "liquidation_deadline": plan.liquidation_deadline,
            "approved": True,
            "status": plan.status,
        }
        rows.extend(
            [
                {
                    **common,
                    "leg": "CLOSE_EXPIRING",
                    "contract_id": plan.from_contract,
                    "side": plan.from_side,
                    "reported_volume": plan.from_volume,
                    "projected_participation": plan.from_participation,
                },
                {
                    **common,
                    "leg": "OPEN_DEFERRED",
                    "contract_id": plan.to_contract,
                    "side": plan.to_side,
                    "reported_volume": plan.to_volume,
                    "projected_participation": plan.to_participation,
                },
            ]
        )
    return pd.DataFrame(rows).reindex(columns=ROLL_LEG_COLUMNS)


__all__ = [
    "BLOCKED",
    "PASS",
    "ContractDataError",
    "SerialDataLimits",
    "SERIAL_CONTRACT_COLUMNS",
    "ROLL_PLAN_COLUMNS",
    "ROLL_LEG_COLUMNS",
    "serial_snapshot_validation_report",
    "validated_serial_snapshot",
    "build_roll_plans",
    "expand_roll_legs",
]
