"""Quote/fill execution-cost calibration with fail-closed evidence checks.

The historical strategy's configured frictions are assumptions.  This module
provides the production boundary: realized paper/live fills are joined to the
arrival quote embedded in each record, decomposed into spread, slippage, and
explicit fees, and assessed against pre-declared sample and freshness rules.
Every fill must provide its effective fill-time USD point value; native point
values cannot be mixed with USD fees or labelled as USD execution shortfall.
It never marks the bundled research data as calibrated because that dataset
contains neither executable quotes nor broker fills.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


PASS = "PASS"
BLOCKED = "BLOCKED"


FILL_COLUMNS = (
    "fill_id",
    "order_id",
    "contract_id",
    "root_symbol",
    "arrival_timestamp",
    "timestamp",
    "side",
    "quantity",
    "order_type",
    "interval_volume",
    "volatility_regime",
    "arrival_bid",
    "arrival_ask",
    "fill_price",
    "point_value_usd",
    "commissions_and_fees_usd",
    "is_roll",
)


CALIBRATION_COLUMNS = (
    "root_symbol",
    "raw_fill_sha256",
    "fill_count",
    "buy_fill_count",
    "sell_fill_count",
    "roll_fill_count",
    "order_count",
    "buy_order_count",
    "sell_order_count",
    "roll_order_count",
    "session_count",
    "delivery_cycle_count",
    "order_type_count",
    "volatility_regime_count",
    "participation_bucket_count",
    "first_fill_timestamp",
    "last_fill_timestamp",
    "median_latency_ms",
    "p95_latency_ms",
    "minimum_participation_rate",
    "maximum_participation_rate",
    "median_quoted_half_spread_per_contract_usd",
    "median_residual_slippage_per_contract_usd",
    "p75_residual_slippage_per_contract_usd",
    "median_fees_per_contract_usd",
    "median_total_cost_per_contract_usd",
    "p75_total_cost_per_contract_usd",
    "median_total_cost_bps",
    "p75_total_cost_bps",
    "estimated_impact_bps_at_full_participation",
)


class CostCalibrationError(ValueError):
    """Raised when quote/fill observations cannot be calibrated safely."""


@dataclass(frozen=True)
class CostCalibrationPolicy:
    """Pre-declared acceptance policy for a root-level calibration."""

    # The three ``minimum_*fills*`` names are retained as compatibility aliases.
    # They now set unique-order floors unless the corresponding explicit order
    # field is supplied.  Fill fragmentation must never satisfy a readiness gate.
    minimum_fills_per_root: int = 30
    minimum_fills_per_side: int = 5
    minimum_roll_fills_per_root: int = 5
    minimum_orders_per_root: int | None = None
    minimum_orders_per_side: int | None = None
    minimum_roll_orders_per_root: int | None = None
    minimum_sessions_per_root: int = 10
    minimum_recent_orders_per_root: int = 20
    minimum_recent_sessions_per_root: int = 10
    minimum_delivery_cycles_per_root: int = 2
    recent_coverage_window_days: float = 30.0
    minimum_order_types_per_root: int = 2
    minimum_volatility_regimes_per_root: int = 2
    minimum_participation_buckets_per_root: int = 3
    minimum_calendar_span_days: float = 10.0
    maximum_age_days: float = 90.0
    maximum_p95_latency_ms: float = 60_000.0
    maximum_observed_participation: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "minimum_fills_per_root",
            "minimum_fills_per_side",
            "minimum_roll_fills_per_root",
            "minimum_sessions_per_root",
            "minimum_recent_orders_per_root",
            "minimum_recent_sessions_per_root",
            "minimum_delivery_cycles_per_root",
            "minimum_order_types_per_root",
            "minimum_volatility_regimes_per_root",
            "minimum_participation_buckets_per_root",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "minimum_orders_per_root",
            "minimum_orders_per_side",
            "minimum_roll_orders_per_root",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        for name in (
            "recent_coverage_window_days",
            "minimum_calendar_span_days",
            "maximum_age_days",
            "maximum_p95_latency_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.maximum_observed_participation)
            or not 0 < self.maximum_observed_participation <= 1
        ):
            raise ValueError(
                "maximum_observed_participation must be finite and in (0, 1]"
            )

    @property
    def required_orders_per_root(self) -> int:
        return self.minimum_orders_per_root or self.minimum_fills_per_root

    @property
    def required_orders_per_side(self) -> int:
        return self.minimum_orders_per_side or self.minimum_fills_per_side

    @property
    def required_roll_orders_per_root(self) -> int:
        return self.minimum_roll_orders_per_root or self.minimum_roll_fills_per_root


def _aware_utc(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CostCalibrationError(f"{label} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise CostCalibrationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _validated_fills(fills: Any) -> pd.DataFrame:
    if not isinstance(fills, pd.DataFrame) or fills.empty:
        raise CostCalibrationError("fills must be a non-empty DataFrame")
    missing = [column for column in FILL_COLUMNS if column not in fills]
    if missing:
        raise CostCalibrationError("fills missing columns: " + ", ".join(missing))
    frame = fills.loc[:, FILL_COLUMNS].copy()
    for identifier in ("fill_id", "order_id", "contract_id", "root_symbol"):
        frame[identifier] = frame[identifier].astype("string").str.strip()
        if frame[identifier].isna().any() or frame[identifier].eq("").any():
            raise CostCalibrationError(f"{identifier} contains blank values")
    if frame["fill_id"].duplicated().any():
        raise CostCalibrationError("fill_id must be unique")
    arrivals = [
        _aware_utc(value, "arrival timestamp")
        for value in frame["arrival_timestamp"]
    ]
    timestamps = [_aware_utc(value, "fill timestamp") for value in frame["timestamp"]]
    frame["arrival_timestamp"] = pd.DatetimeIndex(arrivals)
    frame["timestamp"] = pd.DatetimeIndex(timestamps)
    if (frame["timestamp"] < frame["arrival_timestamp"]).any():
        raise CostCalibrationError("fill timestamp cannot precede arrival timestamp")
    frame["side"] = frame["side"].astype("string").str.upper().str.strip()
    if not frame["side"].isin(["BUY", "SELL"]).all():
        raise CostCalibrationError("side must be BUY or SELL")
    frame["order_type"] = frame["order_type"].astype("string").str.upper().str.strip()
    allowed_order_types = {"LIMIT", "MARKETABLE_LIMIT", "TWAP", "VWAP"}
    if not frame["order_type"].isin(allowed_order_types).all():
        raise CostCalibrationError("order_type is missing or unsupported")
    frame["volatility_regime"] = (
        frame["volatility_regime"].astype("string").str.upper().str.strip()
    )
    if (
        frame["volatility_regime"].isna().any()
        or frame["volatility_regime"].eq("").any()
    ):
        raise CostCalibrationError("volatility_regime contains blank values")
    numeric_columns = (
        "quantity",
        "interval_volume",
        "arrival_bid",
        "arrival_ask",
        "fill_price",
        "point_value_usd",
        "commissions_and_fees_usd",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise CostCalibrationError("numeric fill fields must be finite")
    if (
        (frame["quantity"] <= 0).any()
        or not np.equal(frame["quantity"], np.rint(frame["quantity"])).all()
    ):
        raise CostCalibrationError("quantity must be a positive integer")
    if (
        (frame["interval_volume"] <= 0).any()
        or (frame["quantity"] > frame["interval_volume"]).any()
    ):
        raise CostCalibrationError(
            "interval_volume must be positive and no smaller than fill quantity"
        )
    if (frame["arrival_ask"] < frame["arrival_bid"]).any():
        raise CostCalibrationError("arrival quote is crossed")
    if (frame["point_value_usd"] <= 0).any():
        raise CostCalibrationError("point_value_usd must be positive")
    if (frame["commissions_and_fees_usd"] < 0).any():
        raise CostCalibrationError("commissions and fees must be nonnegative")
    if not frame["is_roll"].map(type).eq(bool).all():
        raise CostCalibrationError("is_roll must contain explicit booleans")
    invariant_order_fields = (
        "root_symbol",
        "contract_id",
        "side",
        "order_type",
        "volatility_regime",
        "is_roll",
    )
    variation = frame.groupby("order_id", sort=False)[list(invariant_order_fields)].nunique(
        dropna=False
    )
    inconsistent = variation.gt(1)
    if inconsistent.any().any():
        bad_fields = [
            field for field in invariant_order_fields if bool(inconsistent[field].any())
        ]
        bad_orders = inconsistent.index[inconsistent.any(axis=1)].astype(str).tolist()[:5]
        raise CostCalibrationError(
            "order_id must map to invariant "
            + ", ".join(bad_fields)
            + f"; offending orders: {bad_orders}"
        )
    frame["quantity"] = frame["quantity"].astype(int)
    return frame


def _canonical_fill_sha256(frame: pd.DataFrame) -> str:
    """Hash validated fill contents independently of row and column ordering."""

    canonical = frame.sort_values("fill_id", kind="mergesort").loc[:, FILL_COLUMNS]
    timestamp_columns = {"arrival_timestamp", "timestamp"}
    integer_columns = {"quantity"}
    boolean_columns = {"is_roll"}
    records: list[list[Any]] = []
    for values in canonical.itertuples(index=False, name=None):
        record: list[Any] = []
        for column, value in zip(FILL_COLUMNS, values, strict=True):
            if column in timestamp_columns:
                record.append(pd.Timestamp(value).tz_convert("UTC").isoformat())
            elif column in integer_columns:
                record.append(int(value))
            elif column in boolean_columns:
                record.append(bool(value))
            elif column in {
                "interval_volume",
                "arrival_bid",
                "arrival_ask",
                "fill_price",
                "point_value_usd",
                "commissions_and_fees_usd",
            }:
                record.append(float(value).hex())
            else:
                record.append(str(value))
        records.append(record)
    payload = json.dumps(
        {"schema": "execution-cost-raw-fills-v2-usd", "columns": FILL_COLUMNS, "rows": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_fill_sha256(fills: pd.DataFrame) -> str:
    """Return the deterministic SHA-256 of normalized, validated raw fills."""

    return _canonical_fill_sha256(_validated_fills(fills))


def execution_cost_observations(fills: pd.DataFrame) -> pd.DataFrame:
    """Return fill-level realized cost decomposition.

    Price improvement can make realized shortfall or residual slippage
    negative.  Calibration percentiles are clipped only when converted into a
    conservative model parameter; the observation ledger preserves reality.
    """

    frame = _validated_fills(fills)
    direction = frame["side"].map({"BUY": 1.0, "SELL": -1.0})
    midpoint = (frame["arrival_bid"] + frame["arrival_ask"]) / 2.0
    quantity = frame["quantity"].astype(float)
    point_value = frame["point_value_usd"]
    shortfall = direction * (frame["fill_price"] - midpoint) * point_value * quantity
    quoted_half_spread = (
        (frame["arrival_ask"] - frame["arrival_bid"])
        / 2.0
        * point_value
        * quantity
    )
    residual = shortfall - quoted_half_spread
    total = shortfall + frame["commissions_and_fees_usd"]
    notional = midpoint.abs() * point_value * quantity
    per_contract = quantity.replace(0, np.nan)
    observations = frame.copy()
    observations["arrival_mid"] = midpoint
    observations["latency_ms"] = (
        observations["timestamp"] - observations["arrival_timestamp"]
    ).dt.total_seconds() * 1_000.0
    observations["participation_rate"] = (
        quantity / frame["interval_volume"].astype(float)
    )
    observations["quoted_half_spread_usd"] = quoted_half_spread
    observations["implementation_shortfall_usd"] = shortfall
    observations["residual_slippage_usd"] = residual
    observations["total_execution_cost_usd"] = total
    observations["quoted_half_spread_per_contract_usd"] = quoted_half_spread / per_contract
    observations["residual_slippage_per_contract_usd"] = residual / per_contract
    observations["fees_per_contract_usd"] = (
        frame["commissions_and_fees_usd"] / per_contract
    )
    observations["total_cost_per_contract_usd"] = total / per_contract
    observations["total_cost_bps"] = np.divide(
        total.to_numpy(dtype=float) * 10_000.0,
        notional.to_numpy(dtype=float),
        out=np.full(len(frame), np.nan),
        where=notional.to_numpy(dtype=float) > 0,
    )
    observations["residual_slippage_bps"] = np.divide(
        residual.to_numpy(dtype=float) * 10_000.0,
        notional.to_numpy(dtype=float),
        out=np.full(len(frame), np.nan),
        where=notional.to_numpy(dtype=float) > 0,
    )
    observations["participation_bucket"] = pd.cut(
        observations["participation_rate"],
        bins=[0.0, 0.0025, 0.01, 0.02, 1.0],
        labels=["<=0.25%", "0.25%-1%", "1%-2%", ">2%"],
        include_lowest=False,
        right=True,
    ).astype("string")
    return observations


def _order_cost_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Collapse partial fills so each parent order receives one calibration vote."""

    rows: list[dict[str, Any]] = []
    for order_id, group in observations.groupby("order_id", sort=True):
        quantity = float(group["quantity"].sum())
        # ``interval_volume`` can be repeated on every partial fill.  Using the
        # largest recorded interval denominator is conservative and invariant to
        # merely splitting one fill record into many rows.
        interval_volume = float(group["interval_volume"].max())
        notional = float(
            (
                group["arrival_mid"].abs()
                * group["point_value_usd"]
                * group["quantity"]
            ).sum()
        )
        quoted_half_spread = float(group["quoted_half_spread_usd"].sum())
        residual_slippage = float(group["residual_slippage_usd"].sum())
        fees = float(group["commissions_and_fees_usd"].sum())
        total_cost = float(group["total_execution_cost_usd"].sum())
        participation = quantity / interval_volume
        rows.append(
            {
                "order_id": str(order_id),
                "root_symbol": str(group["root_symbol"].iloc[0]),
                "contract_id": str(group["contract_id"].iloc[0]),
                "side": str(group["side"].iloc[0]),
                "order_type": str(group["order_type"].iloc[0]),
                "volatility_regime": str(group["volatility_regime"].iloc[0]),
                "is_roll": bool(group["is_roll"].iloc[0]),
                "fill_count": len(group),
                "quantity": quantity,
                "first_fill_timestamp": group["timestamp"].min(),
                "last_fill_timestamp": group["timestamp"].max(),
                "latency_ms": float(group["latency_ms"].max()),
                "participation_rate": participation,
                "quoted_half_spread_per_contract_usd": (
                    quoted_half_spread / quantity
                ),
                "residual_slippage_per_contract_usd": (
                    residual_slippage / quantity
                ),
                "fees_per_contract_usd": fees / quantity,
                "total_cost_per_contract_usd": total_cost / quantity,
                "total_cost_bps": (
                    total_cost * 10_000.0 / notional if notional > 0 else np.nan
                ),
                "residual_slippage_bps": (
                    residual_slippage * 10_000.0 / notional
                    if notional > 0
                    else np.nan
                ),
            }
        )
    orders = pd.DataFrame(rows)
    if orders.empty:
        return orders
    orders["participation_bucket"] = pd.cut(
        orders["participation_rate"],
        bins=[0.0, 0.0025, 0.01, 0.02, 1.0],
        labels=["<=0.25%", "0.25%-1%", "1%-2%", ">2%"],
        include_lowest=False,
        right=True,
    ).astype("string")
    return orders


def calibrate_execution_costs(fills: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unique orders into raw-fill-bound, versionable root parameters."""

    observations = execution_cost_observations(fills)
    orders = _order_cost_observations(observations)
    digest = _canonical_fill_sha256(observations.loc[:, FILL_COLUMNS])
    rows: list[dict[str, Any]] = []
    for root, group in orders.groupby("root_symbol", sort=True):
        fill_group = observations.loc[observations["root_symbol"].eq(root)]
        per_contract = group["total_cost_per_contract_usd"]
        residual = group["residual_slippage_per_contract_usd"]
        total_bps = group["total_cost_bps"].dropna()
        impact_samples = np.divide(
            group["residual_slippage_bps"].to_numpy(dtype=float),
            np.sqrt(group["participation_rate"].to_numpy(dtype=float)),
            out=np.full(len(group), np.nan),
            where=group["participation_rate"].to_numpy(dtype=float) > 0,
        )
        finite_impact = impact_samples[np.isfinite(impact_samples)]
        rows.append(
            {
                "root_symbol": str(root),
                "raw_fill_sha256": digest,
                "fill_count": len(fill_group),
                "buy_fill_count": int(fill_group["side"].eq("BUY").sum()),
                "sell_fill_count": int(fill_group["side"].eq("SELL").sum()),
                "roll_fill_count": int(fill_group["is_roll"].sum()),
                "order_count": len(group),
                "buy_order_count": int(group["side"].eq("BUY").sum()),
                "sell_order_count": int(group["side"].eq("SELL").sum()),
                "roll_order_count": int(group["is_roll"].sum()),
                "session_count": int(
                    fill_group["timestamp"].dt.normalize().nunique()
                ),
                "delivery_cycle_count": int(group["contract_id"].nunique()),
                "order_type_count": int(group["order_type"].nunique()),
                "volatility_regime_count": int(
                    group["volatility_regime"].nunique()
                ),
                "participation_bucket_count": int(
                    group["participation_bucket"].nunique(dropna=True)
                ),
                "first_fill_timestamp": fill_group["timestamp"].min(),
                "last_fill_timestamp": fill_group["timestamp"].max(),
                "median_latency_ms": float(group["latency_ms"].median()),
                "p95_latency_ms": float(group["latency_ms"].quantile(0.95)),
                "minimum_participation_rate": float(
                    group["participation_rate"].min()
                ),
                "maximum_participation_rate": float(
                    group["participation_rate"].max()
                ),
                "median_quoted_half_spread_per_contract_usd": float(
                    group["quoted_half_spread_per_contract_usd"].median()
                ),
                "median_residual_slippage_per_contract_usd": float(residual.median()),
                "p75_residual_slippage_per_contract_usd": float(residual.quantile(0.75)),
                "median_fees_per_contract_usd": float(
                    group["fees_per_contract_usd"].median()
                ),
                "median_total_cost_per_contract_usd": float(per_contract.median()),
                "p75_total_cost_per_contract_usd": float(
                    max(0.0, per_contract.quantile(0.75))
                ),
                "median_total_cost_bps": (
                    float(total_bps.median()) if not total_bps.empty else np.nan
                ),
                "p75_total_cost_bps": (
                    float(max(0.0, total_bps.quantile(0.75)))
                    if not total_bps.empty
                    else np.nan
                ),
                "estimated_impact_bps_at_full_participation": (
                    float(max(0.0, np.quantile(finite_impact, 0.75)))
                    if len(finite_impact)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).reindex(columns=CALIBRATION_COLUMNS)


def _calibration_rows_match(reported: pd.Series, expected: pd.Series) -> bool:
    """Compare a persisted calibration row with a raw-fill recomputation."""

    text_columns = {"root_symbol", "raw_fill_sha256"}
    timestamp_columns = {"first_fill_timestamp", "last_fill_timestamp"}
    for column in CALIBRATION_COLUMNS:
        left = reported[column]
        right = expected[column]
        if column in text_columns:
            if str(left) != str(right):
                return False
        elif column in timestamp_columns:
            try:
                if _aware_utc(left, column) != _aware_utc(right, column):
                    return False
            except CostCalibrationError:
                return False
        else:
            left_number = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
            right_number = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
            if pd.isna(left_number) or pd.isna(right_number):
                if pd.isna(left_number) and pd.isna(right_number):
                    continue
                return False
            if not (
                np.isfinite(float(left_number))
                and np.isfinite(float(right_number))
                and np.isclose(
                    float(left_number),
                    float(right_number),
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                return False
    return True


def cost_calibration_readiness_report(
    calibration: Any,
    *,
    required_roots: Iterable[str],
    as_of: Any,
    policy: CostCalibrationPolicy = CostCalibrationPolicy(),
    raw_fills: Any = None,
) -> pd.DataFrame:
    """Return raw-fill-bound, conjunctive readiness rows by futures root.

    A free-standing aggregate can never pass.  Readiness validates the supplied
    raw fills, recomputes every aggregate, verifies the embedded SHA-256, and
    then evaluates unique-order, distinct-session, recency, delivery-cycle, and
    execution-quality requirements.
    """

    if not isinstance(policy, CostCalibrationPolicy):
        raise TypeError("policy must be a CostCalibrationPolicy")
    try:
        evaluation_time = _aware_utc(as_of, "as_of")
    except CostCalibrationError as exc:
        evaluation_time = None
        as_of_error = str(exc)
    else:
        as_of_error = None

    roots = [str(root).strip() for root in required_roots]
    roots = list(dict.fromkeys(root for root in roots if root))
    if not roots:
        roots = ["__required_universe__"]

    valid_frame = isinstance(calibration, pd.DataFrame) and set(
        CALIBRATION_COLUMNS
    ).issubset(calibration.columns)
    frame = calibration if valid_frame else pd.DataFrame()
    raw_error: str | None = None
    raw_observations = pd.DataFrame()
    raw_orders = pd.DataFrame()
    expected = pd.DataFrame()
    try:
        raw_observations = execution_cost_observations(raw_fills)
        raw_orders = _order_cost_observations(raw_observations)
        expected = calibrate_execution_costs(raw_fills)
    except (CostCalibrationError, TypeError, ValueError) as exc:
        raw_error = f"raw fills are missing or invalid: {exc}"

    rows: list[dict[str, Any]] = []
    for root in roots:
        reasons: list[str] = []
        if not valid_frame:
            reasons.append("calibration table is missing or malformed")
            reported_matches = pd.DataFrame()
        else:
            reported_matches = frame.loc[frame["root_symbol"].astype(str).eq(root)]
            if len(reported_matches) != 1:
                reasons.append(
                    "root calibration is missing"
                    if reported_matches.empty
                    else "root calibration is duplicated"
                )

        if raw_error is not None:
            reasons.append(raw_error)
            expected_matches = pd.DataFrame()
        else:
            expected_matches = expected.loc[expected["root_symbol"].astype(str).eq(root)]
            if len(expected_matches) != 1:
                reasons.append("raw fills do not contain exactly one required-root calibration")

        row: pd.Series | None = None
        if len(expected_matches) == 1:
            row = expected_matches.iloc[0]
            if len(reported_matches) == 1:
                reported = reported_matches.iloc[0]
                if str(reported["raw_fill_sha256"]) != str(row["raw_fill_sha256"]):
                    reasons.append("raw-fill SHA256 does not match calibration")
                elif not _calibration_rows_match(reported, row):
                    reasons.append("calibration aggregate does not match raw fills")

        observed: Any = None
        if row is not None:
            observed = (
                f"{int(row['order_count'])} orders / "
                f"{int(row['session_count'])} UTC sessions / "
                f"{int(row['delivery_cycle_count'])} delivery cycles"
            )
            numeric_requirements = (
                ("order_count", policy.required_orders_per_root, "total orders"),
                ("buy_order_count", policy.required_orders_per_side, "buy orders"),
                ("sell_order_count", policy.required_orders_per_side, "sell orders"),
                (
                    "roll_order_count",
                    policy.required_roll_orders_per_root,
                    "roll orders",
                ),
                ("session_count", policy.minimum_sessions_per_root, "sessions"),
                (
                    "delivery_cycle_count",
                    policy.minimum_delivery_cycles_per_root,
                    "delivery cycles",
                ),
                (
                    "order_type_count",
                    policy.minimum_order_types_per_root,
                    "order types",
                ),
                (
                    "volatility_regime_count",
                    policy.minimum_volatility_regimes_per_root,
                    "volatility regimes",
                ),
                (
                    "participation_bucket_count",
                    policy.minimum_participation_buckets_per_root,
                    "participation buckets",
                ),
            )
            for column, minimum, label in numeric_requirements:
                value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                if not np.isfinite(value) or value < minimum:
                    reasons.append(f"insufficient {label}")

            try:
                first_fill = _aware_utc(row["first_fill_timestamp"], "first fill")
                last_fill = _aware_utc(row["last_fill_timestamp"], "last fill")
            except CostCalibrationError as exc:
                reasons.append(str(exc))
            else:
                span_days = (last_fill - first_fill).total_seconds() / 86_400.0
                if span_days < policy.minimum_calendar_span_days:
                    reasons.append("calibration history is too short")
                if evaluation_time is not None:
                    age_days = (evaluation_time - last_fill).total_seconds() / 86_400.0
                    if age_days < 0:
                        reasons.append("last fill is in the future")
                    elif age_days > policy.maximum_age_days:
                        reasons.append("calibration is stale")

            if evaluation_time is not None and not raw_orders.empty:
                recent_start = evaluation_time - pd.Timedelta(
                    days=policy.recent_coverage_window_days
                )
                root_orders = raw_orders.loc[raw_orders["root_symbol"].eq(root)]
                recent_orders = root_orders.loc[
                    root_orders["last_fill_timestamp"].between(
                        recent_start, evaluation_time, inclusive="both"
                    )
                ]
                root_fills = raw_observations.loc[
                    raw_observations["root_symbol"].eq(root)
                ]
                recent_fills = root_fills.loc[
                    root_fills["timestamp"].between(
                        recent_start, evaluation_time, inclusive="both"
                    )
                ]
                recent_session_count = int(
                    recent_fills["timestamp"].dt.normalize().nunique()
                )
                if len(recent_orders) < policy.minimum_recent_orders_per_root:
                    reasons.append("insufficient recent orders")
                if recent_session_count < policy.minimum_recent_sessions_per_root:
                    reasons.append("insufficient recent sessions")

            p75 = pd.to_numeric(
                pd.Series([row["p75_total_cost_per_contract_usd"]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(p75) or p75 < 0:
                reasons.append("conservative cost parameter is invalid")
            p95_latency = pd.to_numeric(
                pd.Series([row["p95_latency_ms"]]), errors="coerce"
            ).iloc[0]
            if (
                not np.isfinite(p95_latency)
                or p95_latency < 0
                or p95_latency > policy.maximum_p95_latency_ms
            ):
                reasons.append("fill latency is missing or outside policy")
            minimum_participation = pd.to_numeric(
                pd.Series([row["minimum_participation_rate"]]), errors="coerce"
            ).iloc[0]
            maximum_participation = pd.to_numeric(
                pd.Series([row["maximum_participation_rate"]]), errors="coerce"
            ).iloc[0]
            if (
                not np.isfinite(minimum_participation)
                or minimum_participation <= 0
                or not np.isfinite(maximum_participation)
                or maximum_participation < minimum_participation
                or maximum_participation > policy.maximum_observed_participation
            ):
                reasons.append("participation observations are invalid or outside policy")
            impact = pd.to_numeric(
                pd.Series([row["estimated_impact_bps_at_full_participation"]]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(impact) or impact < 0:
                reasons.append("square-root impact coefficient is invalid")
        if as_of_error:
            reasons.append(as_of_error)
        rows.append(
            {
                "gate": f"cost_calibration:{root}",
                "category": "cost calibration",
                "critical": True,
                "status": BLOCKED if reasons else PASS,
                "observed": observed,
                "limit": (
                    f">={policy.required_orders_per_root} unique orders, >="
                    f"{policy.required_orders_per_side}/side, >="
                    f"{policy.required_roll_orders_per_root} roll, >="
                    f"{policy.minimum_sessions_per_root} UTC sessions, >="
                    f"{policy.minimum_recent_orders_per_root} recent orders across >="
                    f"{policy.minimum_recent_sessions_per_root} recent sessions, >="
                    f"{policy.minimum_delivery_cycles_per_root} delivery cycles, <="
                    f"{policy.maximum_age_days:g} days old"
                ),
                "reason": "; ".join(dict.fromkeys(reasons)) if reasons else "gate passed",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BLOCKED",
    "PASS",
    "CALIBRATION_COLUMNS",
    "FILL_COLUMNS",
    "CostCalibrationError",
    "CostCalibrationPolicy",
    "execution_cost_observations",
    "raw_fill_sha256",
    "calibrate_execution_costs",
    "cost_calibration_readiness_report",
]
