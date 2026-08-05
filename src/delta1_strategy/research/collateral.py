"""Funded-account reporting view for the DELTA1 excess-return ledger.

The research ledger is an excess-return ledger: cash collateral earns nothing,
so reported CAGR is what the futures positions produced and not what a funded
account would have shown.  Over 1990-2014 the front Fed Funds contract implies
an average financing rate of about 3.27%, so the gap between the two bases is
several percentage points of compound return.  Disclosing it is honest
accounting.

Two things about that disclosure are dangerous enough to be designed against.

The first is a definitional trap.  Adding the financing rate to the numerator
and continuing to label the ratio "rf = 0" produces a Sharpe of roughly 2.00 on
a strategy whose risk-adjusted return has not changed at all, and 2.00 happens
to be the exact figure the committee has named as an aspiration.  A number that
arrives at the goal by rearranging a definition is worse than no number, so
``funded_performance_report`` emits the excess-of-financing Sharpe, which is
identical to the excess ledger's Sharpe by construction, and
``collateral_reconciliation_report`` checks that identity rather than trusting
it.

The second is regime blindness.  The 3.27% average spans 8% money in 1990 and
0.12% money in 2014.  A single blended uplift implies a forward expectation
that the data does not support, so the summary reports the contribution per
rate regime and the caller is expected to publish it that way.

This module never mutates the research ledger.  It reads ``result.daily`` and
returns new frames, so the ten ledger identities, the daily fingerprint, and
the frozen bundle are all untouched.  ``drawdown`` is the precedent: a
clearly-labelled diagnostic layer that sits beside the canonical result rather
than inside it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_DAILY_COLUMNS = ("net_return",)

FUNDED_STATUS = "diagnostic_not_ledger_integrated"
FUNDED_BASIS_LABEL = "funded_collateral_diagnostic"
EXCESS_BASIS_LABEL = "excess_return_research_ledger"

# The yield leg only.  A funded account also pays variation-margin financing and
# can be liquidated; neither is modelled here, so this view is optimistic.
FUNDED_LIMITATION = (
    "collateral_yield_leg_only; variation_margin_funding_and_liquidation_absent"
)


@dataclass(frozen=True)
class CollateralConfig:
    """Financing convention for the funded view.

    Defaults describe the front CBOT 30-Day Federal Funds contract, whose
    ``100 - price`` quote is an overnight-linked rate.  An overnight benchmark
    is the right instrument for daily cash accrual on a margin account; a
    three-month bill would overstate the credit whenever the curve slopes up.
    """

    rate_file: str = "&ZQ.csv"
    rate_instrument: str = (
        "CBOT 30-Day Federal Funds futures, front delivery month"
    )
    rate_transform: str = "(100 - settlement) / 100"
    daycount_basis: int = 360
    cash_rate_spread_bps: float = 0.0
    max_stale_rate_sessions: int = 5
    annualization: int = 252

    def __post_init__(self) -> None:
        if self.daycount_basis not in {360, 365}:
            raise ValueError("daycount_basis must be 360 or 365")
        if (
            not np.isfinite(self.cash_rate_spread_bps)
            or self.cash_rate_spread_bps < 0
        ):
            raise ValueError("cash_rate_spread_bps must be finite and nonnegative")
        if (
            isinstance(self.max_stale_rate_sessions, bool)
            or not isinstance(self.max_stale_rate_sessions, int)
            or self.max_stale_rate_sessions < 1
        ):
            raise ValueError("max_stale_rate_sessions must be a positive integer")
        if (
            isinstance(self.annualization, bool)
            or not isinstance(self.annualization, int)
            or self.annualization <= 0
        ):
            raise ValueError("annualization must be a positive integer")


def load_financing_rate(
    data_dir: Path,
    config: CollateralConfig = CollateralConfig(),
) -> pd.DataFrame:
    """Load the implied financing rate from the unadjusted Fed Funds file.

    The unadjusted contract is mandatory.  Back-adjustment shifts the whole
    price series to make returns continuous, which destroys the *level* that
    ``100 - price`` depends on; the back-adjusted file implies a 14% rate in
    1988.  Bounds below reject that failure mode rather than propagating it.
    """

    from .strategy import _load_column

    path = Path(data_dir) / "Futures Data" / config.rate_file
    settlement = _load_column(path, "Close", "settlement_price").dropna()
    if settlement.empty:
        raise ValueError(f"{path} contains no usable settlements")
    if (settlement > 100.0).any():
        raise ValueError(
            f"{path} implies a negative financing rate; the unadjusted "
            "contract is required and a back-adjusted file will fail here"
        )
    implied = (100.0 - settlement) / 100.0
    if implied.max() > 0.20:
        raise ValueError(
            f"{path} implies a financing rate above 20%; the file does not "
            "look like an unadjusted Fed Funds series"
        )
    # Back-adjustment shifts the whole series by the accumulated roll gaps,
    # which preserves returns but destroys the level this transform depends
    # on.  The tell is the floor: policy rates reach the low single digits at
    # some point in any multi-decade window, so a long series that never does
    # has been shifted.  The back-adjusted companion file bottoms out near 6%.
    span_years = (implied.index[-1] - implied.index[0]).days / 365.2425
    if span_years > 10.0 and implied.min() > 0.03:
        raise ValueError(
            f"{path} spans {span_years:.1f} years but its implied rate never "
            f"falls below {implied.min():.2%}; this is the signature of a "
            "back-adjusted continuous series, which cannot be used as a rate "
            "level. Use the unadjusted contract file."
        )
    frame = pd.DataFrame(
        {
            "settlement_price": settlement,
            "implied_annual_rate": implied,
        }
    )
    frame.attrs["source_path"] = str(path)
    frame.attrs["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame


def _validated_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    missing = [column for column in REQUIRED_DAILY_COLUMNS if column not in daily]
    if missing:
        raise ValueError(f"daily is missing required columns: {missing}")
    if daily.empty:
        raise ValueError("daily cannot be empty")
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be unique and increasing")
    validated = daily.loc[:, list(REQUIRED_DAILY_COLUMNS)].astype(float).copy()
    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("daily net returns must all be finite")
    if (validated["net_return"] <= -1).any():
        raise ValueError("net returns must be greater than -100%")
    return validated


def funded_ledger(
    daily: pd.DataFrame,
    rate: pd.DataFrame,
    config: CollateralConfig = CollateralConfig(),
    *,
    period_start: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Accrue collateral interest alongside, never inside, the research ledger.

    Interest accrues on calendar days rather than sessions, so a Friday-to-
    Monday step earns three days and a long weekend earns four.  The rate used
    on session ``t`` is the one observed at the close of ``t - 1``: cash earns
    at a rate that was already known, and the strict inequality is checked in
    the reconciliation report rather than assumed here.
    """

    validated = _validated_daily(daily)
    if period_start is not None:
        validated = validated.loc[pd.Timestamp(period_start):]
        if validated.empty:
            raise ValueError("no sessions remain after period_start")
    index = validated.index

    observed = rate["implied_annual_rate"].reindex(index.union(rate.index)).sort_index()
    aligned = observed.ffill(limit=config.max_stale_rate_sessions).reindex(index)
    # A session "used" a forward-filled rate when the union series carried a
    # value into it that the vendor did not print on that date.  Counting the
    # residual NaNs instead would report zero precisely when filling is doing
    # the most work.
    printed = rate["implied_annual_rate"].reindex(index)
    filled = aligned.notna() & printed.isna()

    # Carry the date the vendor actually printed the rate, not the date it was
    # consumed, so the causality check compares real observation times.
    printed_dates = pd.Series(index, index=index).where(
        printed.notna()
    ).ffill(limit=config.max_stale_rate_sessions)

    # Shift after alignment so the value carried into session t is the last
    # rate genuinely observed strictly before it.
    lagged = aligned.shift(1)
    observation_date = printed_dates.shift(1)
    if lagged.isna().iloc[1:].any():
        stale = int(lagged.isna().iloc[1:].sum())
        raise ValueError(
            f"{stale} sessions have no financing rate within "
            f"{config.max_stale_rate_sessions} sessions; refusing to invent one"
        )

    calendar_days = pd.Series(index, index=index).diff().dt.days.astype(float)
    if period_start is not None:
        calendar_days.iloc[0] = float(
            (index[0] - pd.Timestamp(period_start)).days
        )
    else:
        calendar_days.iloc[0] = 0.0
    if (calendar_days < 0).any():
        raise ValueError("calendar day counts cannot be negative")

    spread = config.cash_rate_spread_bps / 10_000.0
    collateral_rate = (lagged - spread).clip(lower=0.0).fillna(0.0)
    collateral_return = collateral_rate * calendar_days / float(config.daycount_basis)

    excess = validated["net_return"]
    funded_return = excess + collateral_return
    funded_nav = (1.0 + funded_return).cumprod()
    funded_peak = funded_nav.cummax().clip(lower=1.0)

    return pd.DataFrame(
        {
            "excess_net_return": excess,
            "accrual_days": calendar_days,
            "rate_observation_date": observation_date,
            "implied_annual_rate": lagged.fillna(0.0),
            "collateral_rate_annual": collateral_rate,
            "collateral_return": collateral_return,
            "funded_net_return": funded_return,
            "funded_nav": funded_nav,
            "funded_peak_nav": funded_peak,
            "funded_drawdown_fraction": funded_nav / funded_peak - 1.0,
            "rate_forward_filled": filled.astype(int),
        },
        index=index,
    )


def _path_metrics(returns: pd.Series, annualization: int) -> dict[str, float]:
    values = returns.dropna()
    if values.empty:
        raise ValueError("cannot summarise an empty return path")
    years = max((values.index[-1] - values.index[0]).days / 365.2425, 1e-9)
    equity = (1.0 + values).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    deviation = float(values.std(ddof=0))
    return {
        "CAGR": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "Annualized volatility": deviation * np.sqrt(annualization),
        "Max drawdown": float((equity / peak - 1.0).min()),
    }


def funded_performance_report(
    funded: pd.DataFrame,
    config: CollateralConfig = CollateralConfig(),
) -> pd.DataFrame:
    """Compare the two accounting bases without inventing a Sharpe improvement.

    The Sharpe column is deliberately *excess of financing* on both rows.  On
    the funded row that means subtracting back the same collateral return that
    was added, which is the correct treatment and which leaves the ratio equal
    to the excess ledger's.  Reporting a funded Sharpe against a zero
    risk-free rate would show roughly 2.00 and would be meaningless.
    """

    excess = funded["excess_net_return"]
    collateral = funded["collateral_return"]
    total = funded["funded_net_return"]
    annualization = config.annualization

    def sharpe(returns: pd.Series) -> float:
        # Both the numerator and the denominator are measured on the same
        # excess-of-financing series.  Dividing an excess mean by the *funded*
        # standard deviation would silently mix bases and break the identity
        # below, which is the whole point of the check.
        deviation = float(returns.std(ddof=0))
        if deviation <= 0:
            return float("nan")
        return float(returns.mean() / deviation * np.sqrt(annualization))

    rows = [
        {
            "Basis": EXCESS_BASIS_LABEL,
            **_path_metrics(excess, annualization),
            "Sharpe excess of financing": sharpe(excess),
            "Average financing rate": 0.0,
            "Annual collateral contribution": 0.0,
            "Limitations": "financing not recognized; positions earn no cash yield",
        },
        {
            "Basis": FUNDED_BASIS_LABEL,
            **_path_metrics(total, annualization),
            # total minus collateral is the excess return, by construction.
            "Sharpe excess of financing": sharpe(total - collateral),
            "Average financing rate": float(
                funded["collateral_rate_annual"].mean()
            ),
            "Annual collateral contribution": float(
                collateral.sum()
                / max(
                    (funded.index[-1] - funded.index[0]).days / 365.2425, 1e-9
                )
            ),
            "Limitations": FUNDED_LIMITATION,
        },
    ]
    return pd.DataFrame(rows)


def funded_regime_report(
    funded: pd.DataFrame,
    config: CollateralConfig = CollateralConfig(),
) -> pd.DataFrame:
    """Financing contribution by rate regime, so no single uplift is implied.

    Regimes are cut on the level of the financing rate itself, not on dates or
    on returns, so the partition carries no information about strategy
    performance and involves no selection.
    """

    edges = [(-np.inf, 0.01, "below 1%"), (0.01, 0.04, "1% to 4%"), (0.04, np.inf, "above 4%")]
    rows: list[dict[str, object]] = []
    rate = funded["collateral_rate_annual"]
    for low, high, label in edges:
        mask = (rate > low) & (rate <= high)
        if not mask.any():
            continue
        segment = funded.loc[mask]
        years = max(float(segment["accrual_days"].sum()) / 365.2425, 1e-9)
        rows.append(
            {
                "Rate regime": label,
                "Sessions": int(mask.sum()),
                "Share of sessions": float(mask.mean()),
                "Average financing rate": float(segment["collateral_rate_annual"].mean()),
                "Annualized collateral contribution": float(
                    segment["collateral_return"].sum() / years
                ),
                "First session": segment.index[0].date().isoformat(),
                "Last session": segment.index[-1].date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def collateral_source_report(
    rate: pd.DataFrame,
    config: CollateralConfig = CollateralConfig(),
) -> pd.DataFrame:
    """Provenance of the financing series.

    ``docs/controls/funding-and-margin.md`` requires a real, dated, sourced
    rate rather than an assumed constant; this row is the evidence that the
    accrual came from a specific file with a specific hash.
    """

    implied = rate["implied_annual_rate"]
    return pd.DataFrame(
        [
            {
                "Instrument": config.rate_instrument,
                "Transform": config.rate_transform,
                "Daycount basis": config.daycount_basis,
                "Source path": rate.attrs.get("source_path", ""),
                "Source SHA-256": rate.attrs.get("source_sha256", ""),
                "First observation": implied.index[0].date().isoformat(),
                "Last observation": implied.index[-1].date().isoformat(),
                "Observations": int(implied.size),
                "Minimum rate": float(implied.min()),
                "Maximum rate": float(implied.max()),
                "Mean rate": float(implied.mean()),
                "Funded view status": FUNDED_STATUS,
                "Limitations": FUNDED_LIMITATION,
            }
        ]
    )


def collateral_reconciliation_report(
    funded: pd.DataFrame,
    daily: pd.DataFrame,
    config: CollateralConfig = CollateralConfig(),
    *,
    absolute_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Fail-closed checks on the funded view, in the ledger report's schema."""

    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    rows: list[dict[str, object]] = []

    def add_check(name: str, error: float, detail: str) -> None:
        finite = float(error) if np.isfinite(error) else np.inf
        rows.append(
            {
                "check": name,
                "status": "PASS" if finite <= absolute_tolerance else "BLOCKED",
                "maximum_absolute_error": finite,
                "tolerance": absolute_tolerance,
                "detail": detail,
            }
        )

    reference = daily.loc[funded.index, "net_return"].astype(float)
    add_check(
        "excess_return_passthrough",
        float((funded["excess_net_return"] - reference).abs().max()),
        "The research ledger's net returns are reproduced unaltered.",
    )
    expected_accrual = (
        funded["collateral_rate_annual"]
        * funded["accrual_days"]
        / float(config.daycount_basis)
    )
    add_check(
        "collateral_accrual_identity",
        float((funded["collateral_return"] - expected_accrual).abs().max()),
        "Collateral return equals rate times daycount fraction.",
    )
    add_check(
        "funded_return_identity",
        float(
            (
                funded["funded_net_return"]
                - funded["excess_net_return"]
                - funded["collateral_return"]
            ).abs().max()
        ),
        "Funded return is the excess return plus collateral only.",
    )
    recursion = funded["funded_nav"].shift(1).fillna(1.0) * (
        1.0 + funded["funded_net_return"]
    )
    add_check(
        "funded_nav_recursion",
        float((funded["funded_nav"] - recursion).abs().max()),
        "Funded NAV compounds the funded return with no exogenous cash events.",
    )
    span_days = float((funded.index[-1] - funded.index[0]).days)
    add_check(
        "accrual_daycount_completeness",
        abs(float(funded["accrual_days"].sum()) - span_days),
        "Every calendar day in the window accrues exactly once.",
    )
    observation = pd.to_datetime(funded["rate_observation_date"])
    late = observation.notna() & (observation.to_numpy() >= funded.index.to_numpy())
    add_check(
        "rate_observation_precedes_decision",
        float(late.sum()),
        "Each session accrues at a rate observed strictly earlier.",
    )
    filled_flag = funded["rate_forward_filled"].astype(bool)
    # Longest unbroken run of forward-filled sessions: a single stale print is
    # tolerable, a long outage is not, and only the run length distinguishes
    # them.
    run_lengths = filled_flag.groupby((~filled_flag).cumsum()).cumsum()
    longest_run = float(run_lengths.max()) if len(run_lengths) else 0.0
    add_check(
        "rate_staleness_bounded",
        max(0.0, longest_run - float(config.max_stale_rate_sessions)),
        (
            f"Longest forward-filled run is {int(longest_run)} sessions against "
            f"a {config.max_stale_rate_sessions}-session limit."
        ),
    )
    add_check(
        "financing_rate_within_bounds",
        float(
            max(
                0.0,
                float((-funded["implied_annual_rate"]).max()),
                float((funded["implied_annual_rate"] - 0.25).max()),
            )
        ),
        "Implied financing rates stay within a plausible 0% to 25% range.",
    )
    # Check the numbers that are actually published, not a second copy of the
    # formula, so the report and its guard cannot drift apart.
    published = funded_performance_report(funded, config).set_index("Basis")
    excess_sharpe = float(
        published.loc[EXCESS_BASIS_LABEL, "Sharpe excess of financing"]
    )
    funded_sharpe = float(
        published.loc[FUNDED_BASIS_LABEL, "Sharpe excess of financing"]
    )
    total = funded["funded_net_return"]
    add_check(
        "funded_sharpe_matches_excess_sharpe",
        float(abs(funded_sharpe - excess_sharpe)),
        (
            "Recognizing financing does not change risk-adjusted return; a "
            "funded Sharpe that differs indicates the rate was added to the "
            "numerator without being subtracted as the hurdle."
        ),
    )
    stressed = funded_ledger(
        daily.loc[funded.index],
        pd.DataFrame(
            {"implied_annual_rate": funded["implied_annual_rate"].shift(-1)}
        ).dropna(),
        replace_spread(config, config.cash_rate_spread_bps + 100.0),
    )
    stressed_cagr = _path_metrics(
        stressed["funded_net_return"], config.annualization
    )["CAGR"]
    base_cagr = _path_metrics(total, config.annualization)["CAGR"]
    add_check(
        "monotone_financing_stress",
        float(max(0.0, stressed_cagr - base_cagr)),
        "A wider financing spread can never improve the funded result.",
    )
    return pd.DataFrame(rows)


def replace_spread(config: CollateralConfig, spread_bps: float) -> CollateralConfig:
    """Return the same convention with a different broker spread."""

    from dataclasses import replace

    return replace(config, cash_rate_spread_bps=spread_bps)


__all__ = [
    "EXCESS_BASIS_LABEL",
    "FUNDED_BASIS_LABEL",
    "FUNDED_LIMITATION",
    "FUNDED_STATUS",
    "REQUIRED_DAILY_COLUMNS",
    "CollateralConfig",
    "collateral_reconciliation_report",
    "collateral_source_report",
    "funded_ledger",
    "funded_performance_report",
    "funded_regime_report",
    "load_financing_rate",
    "replace_spread",
]
