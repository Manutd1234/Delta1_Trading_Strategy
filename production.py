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
from typing import Any, Mapping

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

    def __post_init__(self) -> None:
        bounded = {
            "max_order_participation": self.max_order_participation,
            "max_margin_fraction": self.max_margin_fraction,
            "max_drawdown_fraction": self.max_drawdown_fraction,
        }
        for name, value in bounded.items():
            if not _finite_number(value) or not 0 < float(value) <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in ("max_gross_notional_multiple", "max_data_age_seconds"):
            value = getattr(self, name)
            if not _finite_number(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ReadinessEvidence:
    """Evidence that must be supplied independently of a historical run."""

    serial_contracts: bool = False
    timestamped_live_data: bool = False
    dated_contract_specs: bool = False
    dated_margin_and_fees: bool = False
    calibrated_cost_model: bool = False
    independent_holdout: bool = False
    paper_trading: bool = False
    broker_adapter: bool = False
    broker_reconciliation: bool = False
    monitoring_and_alerts: bool = False
    tested_kill_switch: bool = False


@dataclass(frozen=True)
class RuntimeHealth:
    """Immutable result of a point-in-time operational health check."""

    healthy: bool = False
    status: str = BLOCKED
    data_age_seconds: float | None = None
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


EVIDENCE_LABELS = {
    "serial_contracts": "Tradeable serial-contract history",
    "timestamped_live_data": "Timestamped live market data",
    "dated_contract_specs": "Dated contract specifications",
    "dated_margin_and_fees": "Dated margin and fee schedule",
    "calibrated_cost_model": "Calibrated execution-cost model",
    "independent_holdout": "Independent holdout evaluation",
    "paper_trading": "Paper-trading evidence",
    "broker_adapter": "Broker adapter",
    "broker_reconciliation": "Broker reconciliation",
    "monitoring_and_alerts": "Monitoring and alerting",
    "tested_kill_switch": "Tested kill switch",
}


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
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp, timestamp.isoformat()


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


def _nonzero_number(value: Any, label: str) -> tuple[float | None, str | None]:
    if not _finite_number(value) or float(value) == 0:
        return None, f"{label} is missing, non-finite, or zero"
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
    raw_prices: pd.Series | Mapping[Any, Any],
    point_values: pd.Series | Mapping[Any, Any],
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
        "raw_prices": _named_series(raw_prices, "raw_prices"),
        "point_values": _named_series(point_values, "point_values"),
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
        # Futures prices can be negative.  Exposure is still valued from the
        # absolute price, while a zero price is unusable for a risk check.
        price, error = _nonzero_number(
            inputs["raw_prices"].get(symbol, np.nan), "raw price"
        )
        if error:
            errors.append(error)
        point_value, error = _positive_number(
            inputs["point_values"].get(symbol, np.nan), "point value"
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
            abs(projected * price * point_value)
            if _finite_number(projected) and price is not None and point_value is not None
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
    """Evaluate a live snapshot; absent, stale, or malformed state blocks trading."""

    if not isinstance(limits, ProductionLimits):
        raise TypeError("limits must be a ProductionLimits instance")
    reasons: list[str] = []

    data_time, _ = _normalise_timestamp(data_timestamp)
    now, _ = _normalise_timestamp(
        pd.Timestamp.now(tz="UTC") if current_time is None else current_time
    )
    data_age: float | None = None
    if data_time is None:
        reasons.append("data timestamp is missing or invalid")
    if now is None:
        reasons.append("current timestamp is missing or invalid")
    if data_time is not None and now is not None:
        data_age = float((now - data_time).total_seconds())
        if data_age < 0:
            reasons.append("data timestamp is in the future")
        elif data_age > limits.max_data_age_seconds:
            reasons.append("market data is stale")

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
        data_age_seconds=data_age,
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
    evidence: ReadinessEvidence = ReadinessEvidence(),
) -> pd.DataFrame:
    """Return critical historical and external-evidence launch gates.

    A backtest can demonstrate that numerical limits were respected; it cannot
    demonstrate live data, broker, cost calibration, or holdout readiness.
    Those gates therefore require explicit external evidence and default to
    ``BLOCKED``.
    """

    if not isinstance(limits, ProductionLimits):
        raise TypeError("limits must be a ProductionLimits instance")
    if not isinstance(evidence, ReadinessEvidence):
        raise TypeError("evidence must be a ReadinessEvidence instance")

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
        "net_return",
        "cost",
        "gross_notional_multiple",
        "static_margin_fraction",
        "max_order_participation",
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
            "historical_margin",
            "static_margin_fraction",
            limits.max_margin_fraction,
            "historical static margin exceeds the production limit",
        ),
        (
            "historical_order_participation",
            "max_order_participation",
            limits.max_order_participation,
            "historical order participation exceeds the production limit",
        ),
    )
    for gate, column, limit, failure_reason in metric_gates:
        finite, values = _finite_series(safe_daily, column)
        observed = float(values.max()) if finite else None
        passed = finite and bool((values >= 0).all()) and observed <= limit
        rows.append(
            _report_row(gate, "risk", passed, observed, limit, failure_reason)
        )

    equity_column = "equity" if "equity" in safe_daily.columns else "nav"
    equity_finite, equity = _finite_series(safe_daily, equity_column)
    max_drawdown: float | None = None
    if equity_finite and bool((equity > 0).all()):
        if equity_column == "equity":
            # Normalised equity has an implicit 1.0 starting value even when
            # the supplied frame begins on the first return observation.
            path = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
        else:
            path = equity.reset_index(drop=True)
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

    for field, label in EVIDENCE_LABELS.items():
        supplied = getattr(evidence, field, False) is True
        rows.append(
            _report_row(
                field,
                "external evidence",
                supplied,
                supplied,
                True,
                f"{label} has not been evidenced",
            )
        )

    return pd.DataFrame(rows).reindex(
        columns=["gate", "category", "critical", "status", "observed", "limit", "reason"]
    )


def overall_readiness_status(report: Any) -> str:
    """Return ``READY`` only when every critical gate explicitly passes."""

    if not isinstance(report, pd.DataFrame) or report.empty:
        return BLOCKED
    if not {"critical", "status"}.issubset(report.columns):
        return BLOCKED
    critical = report.loc[report["critical"].eq(True), "status"]
    if critical.empty or not critical.eq(PASS).all():
        return BLOCKED
    return "READY"


__all__ = [
    "ProductionLimits",
    "ReadinessEvidence",
    "RuntimeHealth",
    "build_order_intents",
    "evaluate_runtime_health",
    "production_readiness_report",
    "overall_readiness_status",
]
