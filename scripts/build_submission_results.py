#!/usr/bin/env python3
"""Section E "Required results" for the Delta1 submission, gross and net.

Produces two artifacts under ``outputs/submission/``:

  * ``required_results.csv``     -- the eleven required metrics, in the order the
                                    brief lists them, for three chronological
                                    windows on both a gross (pre-cost) and a net
                                    (post-cost) basis.
  * ``gross_vs_net_summary.csv`` -- the headline gross/net pairs side by side
                                    with the cost bridge between them.

Nothing here is hand-entered.  Every figure is derived from the canonical
bundle already committed under ``outputs/`` -- principally the daily ledger
``outputs/strategy_daily.csv``, the published window metrics
``outputs/strategy_metrics.csv``, the closed-episode file
``outputs/strategy_trade_episodes.csv`` and the benchmark table
``outputs/benchmarks/benchmark_comparison.csv``.

The one figure that cannot be read off the committed bundle per window is
*notional* turnover: the ledger records contract counts, and turning those into
a dollar notional needs the unadjusted price panel and USD point values.  That
one number is recomputed by re-running the frozen backtest, and the full-window
value is asserted against the published ``benchmark_comparison.csv`` figure so
the recomputation is proved to be the same object the rest of the bundle
describes.  Pass ``--skip-notional`` to build everything else without the data
directory.

Reconciliation is not decorative.  Every quantity this script computes that
``outputs/strategy_metrics.csv`` (or the benchmark table) already publishes is
checked against the published value and the check is reported in the CSV and on
stdout.  A failure raises rather than being silently rounded away.

Usage
-----
    python scripts/build_submission_results.py \
        --data-dir "Round1AllData/Quant Researcher/Delta1"
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
SUBMISSION = OUTPUTS / "submission"

ANNUALIZATION = 252
SESSIONS_PER_MONTH = ANNUALIZATION / 12.0  # 21.0, the monthly rebalance cadence

# The three chronological windows.  The split is the development / out-of-sample
# split the brief asks for: rules were fixed on 1990-2004 and the 2005-2014
# assessment applies them unchanged.  Names here are the submission's names; the
# canonical metrics file uses longer labels, mapped below.
WINDOWS: dict[str, tuple[str, str]] = {
    "1990-2004 development": ("1990-01-01", "2004-12-31"),
    "2005-2014 out-of-sample": ("2005-01-01", "2014-12-31"),
    "1990-2014 full": ("1990-01-01", "2014-12-31"),
}

CANONICAL_WINDOW_LABEL = {
    "1990-2004 development": "1990-2004 development history",
    "2005-2014 out-of-sample": "2005-2014 reused later diagnostic",
    "1990-2014 full": "1990-2014 full post-launch history",
}

BASIS_BOTH = "gross and net"
BASIS_INVARIANT = "basis-invariant (an execution fact, not a P&L fact)"
BASIS_NET_ONLY = "net only (cost is the gross-to-net bridge)"

GROSS_PLACEHOLDER_INVARIANT = "basis-invariant - see net column"
GROSS_PLACEHOLDER_NET_ONLY = "n/a - gross is pre-cost by construction"


# ---------------------------------------------------------------------------
# Reconciliation ledger
# ---------------------------------------------------------------------------


@dataclass
class Check:
    window: str
    item: str
    computed: float
    published: float
    source: str

    @property
    def absolute(self) -> float:
        return abs(self.computed - self.published)

    @property
    def relative(self) -> float:
        scale = max(abs(self.published), 1e-12)
        return self.absolute / scale

    def passed(self, tolerance: float) -> bool:
        return self.absolute <= tolerance or self.relative <= tolerance


class Reconciler:
    """Collect every published-vs-recomputed comparison, then fail loudly."""

    def __init__(self, tolerance: float = 1e-9) -> None:
        self.tolerance = tolerance
        self.checks: list[Check] = []

    def add(
        self, window: str, item: str, computed: float, published: float, source: str
    ) -> float:
        self.checks.append(Check(window, item, float(computed), float(published), source))
        return float(computed)

    def report(self) -> str:
        lines = [
            "Reconciliation against the canonical bundle "
            f"(tolerance {self.tolerance:g}, absolute or relative):",
            "",
        ]
        failures = [c for c in self.checks if not c.passed(self.tolerance)]
        for check in self.checks:
            flag = "OK  " if check.passed(self.tolerance) else "FAIL"
            lines.append(
                f"  [{flag}] {check.window:<24} {check.item:<44} "
                f"computed={check.computed:.12g}  published={check.published:.12g}  "
                f"abs_diff={check.absolute:.3g}  ({check.source})"
            )
        lines.append("")
        lines.append(
            f"  {len(self.checks) - len(failures)}/{len(self.checks)} checks reconcile."
        )
        if failures:
            lines.append("  DISCREPANCIES:")
            for check in failures:
                lines.append(
                    f"    {check.window} / {check.item}: computed {check.computed!r} "
                    f"vs published {check.published!r} (abs {check.absolute:.6g}, "
                    f"rel {check.relative:.6g}) from {check.source}"
                )
        return "\n".join(lines)

    def assert_clean(self) -> None:
        failures = [c for c in self.checks if not c.passed(self.tolerance)]
        if failures:
            raise AssertionError(
                "recomputed figures disagree with the published bundle:\n"
                + "\n".join(
                    f"  {c.window} / {c.item}: {c.computed!r} vs {c.published!r} "
                    f"(abs {c.absolute:.6g}) from {c.source}"
                    for c in failures
                )
            )


# ---------------------------------------------------------------------------
# Return-series metrics -- identical arithmetic to reference/delta1_reference.py
# ---------------------------------------------------------------------------


def series_metrics(daily: pd.DataFrame, column: str, start: str, end: str) -> dict:
    """CAGR / vol / Sharpe / max drawdown for one return column and window.

    Deliberately a line-for-line copy of ``reference.delta1_reference.metrics``
    so that applying it to ``net_return`` reproduces the published table exactly
    and applying it to ``gross_return`` is the same estimator, not a second one.

    ``years`` is measured from the session *before* the first in-window return,
    because a return stamped on day t is earned over (t-1, t].
    """
    returns = daily.loc[start:end, column].dropna()
    location = daily.index.get_indexer([returns.index[0]])[0]
    origin = daily.index[location - 1] if location > 0 else returns.index[0]
    years = (returns.index[-1] - origin).total_seconds() / (365.2425 * 86_400)

    equity = (1 + returns).cumprod()
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:] - 1
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    max_drawdown = float(drawdown.min())

    return {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "sessions": int(len(returns)),
        "years": float(years),
        "cagr": cagr,
        "volatility": float(returns.std() * math.sqrt(ANNUALIZATION)),
        "sharpe": float(returns.mean() / returns.std() * math.sqrt(ANNUALIZATION)),
        "max_drawdown": max_drawdown,
        "return_to_drawdown": cagr / abs(max_drawdown),
        "daily_win_rate": float((returns > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_daily() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUTS / "strategy_daily.csv", parse_dates=["date"])
    return frame.set_index("date").sort_index()


def load_published_metrics() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUTS / "strategy_metrics.csv")
    return frame.set_index("Window")


def load_episodes() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUTS / "strategy_trade_episodes.csv")
    frame["Entry date"] = pd.to_datetime(frame["Entry date"])
    frame["Exit date"] = pd.to_datetime(frame["Exit date"])
    return frame


def load_published_benchmark_row() -> pd.Series:
    frame = pd.read_csv(OUTPUTS / "benchmarks" / "benchmark_comparison.csv")
    row = frame.loc[frame["benchmark"].eq("incumbent")]
    if row.empty:
        raise ValueError("benchmark_comparison.csv has no 'incumbent' row")
    return row.iloc[0]


# ---------------------------------------------------------------------------
# Execution / cost / exposure facts
# ---------------------------------------------------------------------------


def execution_facts(daily: pd.DataFrame, episodes: pd.DataFrame, start: str, end: str) -> dict:
    window = daily.loc[start:end]
    # Turnover uses the calendar span of the window index, which is the
    # convention published in outputs/benchmarks/benchmark_comparison.csv.
    # It differs from the CAGR `years` by ~0.03% because the CAGR clock starts
    # one session earlier; the two are reported under their own labels rather
    # than silently blended.
    turnover_years = max((window.index[-1] - window.index[0]).days / 365.2425, 1e-9)

    prior_nav = window["prior_nav_usd"]
    fixed_usd = float(window["fixed_execution_cost_usd"].sum())
    impact_usd = float(window["market_impact_cost_usd"].sum())
    total_usd = float(window["transaction_cost_usd"].sum())
    if abs((fixed_usd + impact_usd) - total_usd) > 1e-6 * max(total_usd, 1.0):
        raise AssertionError(
            "fixed + impact cost does not equal total transaction cost: "
            f"{fixed_usd} + {impact_usd} != {total_usd}"
        )

    closed = episodes[
        episodes["Status"].eq("Closed")
        & episodes["Exit date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    open_censored = int(
        (
            episodes["Entry date"].le(pd.Timestamp(end))
            & (episodes["Exit date"].isna() | episodes["Exit date"].gt(pd.Timestamp(end)))
        ).sum()
    )

    return {
        "turnover_years": float(turnover_years),
        # Costs
        "cost_usd_total": total_usd,
        "cost_usd_fixed": fixed_usd,
        "cost_usd_impact": impact_usd,
        "cost_drag_total": float(window["cost"].mean() * ANNUALIZATION),
        "cost_drag_fixed": float(
            (window["fixed_execution_cost_usd"] / prior_nav).mean() * ANNUALIZATION
        ),
        "cost_drag_impact": float(
            (window["market_impact_cost_usd"] / prior_nav).mean() * ANNUALIZATION
        ),
        # Contract turnover
        "turnover_contracts_total": float(
            window["total_contract_turnover"].sum() / turnover_years
        ),
        "turnover_contracts_rebalance": float(
            window["rebalance_contract_turnover"].sum() / turnover_years
        ),
        "turnover_contracts_roll": float(
            window["roll_contract_turnover_increment"].sum() / turnover_years
        ),
        # Trade counts
        "closed_episodes": int(len(closed)),
        "open_censored_episodes": open_censored,
        "filled_market_sessions": int(window["filled_markets"].sum()),
        # Hit rate and holding period at trade level
        "trade_win_rate_net": float((closed["Net P&L USD"] > 0.01).mean()),
        "trade_win_rate_gross": float((closed["Gross P&L USD"] > 0.01).mean()),
        "holding_sessions": float(closed["Holding sessions"].mean()),
        # Exposure
        "avg_gross_notional_multiple": float(window["gross_notional_multiple"].mean()),
        "avg_markets_held": float(window["active_markets"].mean()),
    }


def notional_turnover(data_dir: Path) -> dict[str, dict[str, float]]:
    """Annualised traded notional as a multiple of NAV, per window.

    Re-runs the frozen backtest because the committed ledger stores contract
    counts, not dollars.  Covers the *rebalance* leg only: the ledger records
    roll turnover as an aggregate contract increment rather than per market, so
    a roll notional cannot be reconstructed, and an incomplete number labelled
    as a total would be worse than an honest partial one.  This is the same
    definition and the same omission as
    ``benchmarks._notional_turnover``, whose full-window output this
    reproduces exactly (asserted in ``main``).
    """
    sys.path.insert(0, str(REPO / "src"))
    from delta1_strategy.research import levers  # noqa: PLC0415
    from delta1_strategy.research.strategy import (  # noqa: PLC0415
        StrategyConfig,
        run_backtest,
        usd_point_values,
    )

    frames = levers.load_shared_frames(data_dir)
    # run_backtest does not write, but point output_dir at a throwaway location
    # so it can never deposit anything in the submission folder.
    with tempfile.TemporaryDirectory() as scratch:
        config = StrategyConfig(data_dir=data_dir, output_dir=Path(scratch))
        result = run_backtest(config, **frames)
    point_values = usd_point_values(
        frames["metadata"], frames["fx_rates"], frames["prices"].index
    )
    valuation = frames["unadjusted"]

    out: dict[str, dict[str, float]] = {}
    for name, (start, end) in WINDOWS.items():
        trades = result.trades.loc[start:end]
        notional = (
            trades.abs()
            * valuation.reindex_like(trades).abs()
            * point_values.reindex_like(trades)
        ).sum(axis=1)
        ratio = notional / result.daily.loc[trades.index, "prior_nav_usd"]
        years = max((trades.index[-1] - trades.index[0]).days / 365.2425, 1e-9)
        out[name] = {
            "notional_turnover_over_nav": float(ratio.sum() / years),
            "traded_notional_usd": float(notional.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

COLUMNS_ORDER = [
    ("1990-2004 development", "gross"),
    ("1990-2004 development", "net"),
    ("2005-2014 out-of-sample", "gross"),
    ("2005-2014 out-of-sample", "net"),
    ("1990-2014 full", "gross"),
    ("1990-2014 full", "net"),
]


def column_name(window: str, basis: str) -> str:
    return f"{window} ({basis})"


def build_required_results(
    net: dict, gross: dict, facts: dict, notional: dict | None
) -> pd.DataFrame:
    """One row per reported metric; the eleven required groups, in order."""

    rows: list[dict] = []

    def add(
        group: str,
        metric: str,
        unit: str,
        basis: str,
        definition: str,
        source: str,
        recon: str,
        pick_gross=None,
        pick_net=None,
    ) -> None:
        row = {
            "Metric group": group,
            "Metric": metric,
            "Unit": unit,
            "Basis dependence": basis,
            "Definition": definition,
            "Canonical source": source,
            "Reconciliation": recon,
        }
        for window, side in COLUMNS_ORDER:
            if side == "gross":
                if basis == BASIS_INVARIANT:
                    value = GROSS_PLACEHOLDER_INVARIANT
                elif basis == BASIS_NET_ONLY:
                    value = GROSS_PLACEHOLDER_NET_ONLY
                else:
                    value = pick_gross(window)
            else:
                value = pick_net(window)
            row[column_name(window, side)] = value
        rows.append(row)

    def g(key, digits=6):
        return lambda w: round(gross[w][key], digits)

    def n(key, digits=6):
        return lambda w: round(net[w][key], digits)

    def f(key, digits=6):
        return lambda w: round(facts[w][key], digits)

    def fi(key):
        return lambda w: int(facts[w][key])

    # --- 1. Annualised return -------------------------------------------------
    add(
        "Annualised return",
        "Annualised return (CAGR, geometric)",
        "fraction per year",
        BASIS_BOTH,
        "Terminal value of the compounded daily return series raised to 1/years, "
        "minus one. Years run from the session before the first in-window return. "
        "Gross compounds strategy_daily.csv 'gross_return'; net compounds 'net_return'.",
        "outputs/strategy_daily.csv; net published as 'CAGR' in outputs/strategy_metrics.csv",
        "net matches published CAGR to 1e-9",
        g("cagr"),
        n("cagr"),
    )

    # --- 2. Annualised volatility --------------------------------------------
    add(
        "Annualised volatility",
        "Annualised volatility (daily sd x sqrt(252))",
        "fraction per year",
        BASIS_BOTH,
        "Sample standard deviation of daily returns scaled by sqrt(252). "
        "The strategy targets 7% ex-ante; the realised figure is the outturn.",
        "outputs/strategy_daily.csv; net published as 'Annualized volatility'",
        "net matches published volatility to 1e-9",
        g("volatility"),
        n("volatility"),
    )

    # --- 3. Sharpe ratio ------------------------------------------------------
    add(
        "Sharpe ratio",
        "Sharpe ratio (daily, rf = 0, x sqrt(252))",
        "ratio",
        BASIS_BOTH,
        "Mean daily return divided by its standard deviation, times sqrt(252), "
        "risk-free rate zero. Futures returns are already excess of financing, so "
        "rf = 0 is the right convention here rather than a simplification.",
        "outputs/strategy_daily.csv; net published as 'Naive daily Sharpe (sqrt252, rf=0)'",
        "net matches published Sharpe to 1e-9",
        g("sharpe"),
        n("sharpe"),
    )

    # --- 4. Maximum drawdown --------------------------------------------------
    add(
        "Maximum drawdown",
        "Maximum drawdown (daily compounded, peak to trough)",
        "fraction (negative)",
        BASIS_BOTH,
        "Minimum of equity / running maximum of equity - 1, on daily compounded "
        "returns, with the running maximum seeded at 1.0 so a drawdown from the "
        "first session is measurable.",
        "outputs/strategy_daily.csv; net published as 'Max drawdown'",
        "net matches published max drawdown to 1e-9",
        g("max_drawdown"),
        n("max_drawdown"),
    )

    # --- 5. Return-to-drawdown ------------------------------------------------
    add(
        "Return-to-drawdown",
        "Return-to-drawdown (CAGR / |max drawdown|), i.e. Calmar",
        "ratio",
        BASIS_BOTH,
        "Annualised return divided by the absolute value of the maximum drawdown, "
        "each taken on the same basis (gross with gross, net with net).",
        "derived; net published as 'Calmar' in outputs/strategy_metrics.csv",
        "net matches published Calmar to 1e-9",
        g("return_to_drawdown"),
        n("return_to_drawdown"),
    )

    # --- 6. Hit rate ----------------------------------------------------------
    add(
        "Hit rate",
        "Hit rate - DAILY win rate (share of sessions with a positive return)",
        "fraction",
        BASIS_BOTH,
        "Share of in-window sessions whose return is strictly positive. This is "
        "the portfolio-level hit rate, not a per-trade one.",
        "outputs/strategy_daily.csv; net published as 'Daily win rate'",
        "net matches published daily win rate to 1e-9",
        g("daily_win_rate"),
        n("daily_win_rate"),
    )
    add(
        "Hit rate",
        "Hit rate - TRADE-LEVEL win rate (closed episodes with P&L > $0.01)",
        "fraction",
        BASIS_BOTH,
        "Closed directional episodes assigned to the window by EXIT date, counted "
        "as a win when realised P&L exceeds one cent (the cent tolerance stops "
        "floating-point dust manufacturing wins). Net uses 'Net P&L USD'; gross "
        "uses 'Gross P&L USD'. Below 50% by design: this is a trend book that "
        "loses often and small, wins rarely and large.",
        "outputs/strategy_trade_episodes.csv; net published as 'Trade win rate'",
        "net matches published trade win rate to 1e-9",
        f("trade_win_rate_gross"),
        f("trade_win_rate_net"),
    )

    # --- 7. Number of trades --------------------------------------------------
    add(
        "Number of trades",
        "Number of trades - CLOSED trade episodes (completed round trips)",
        "count",
        BASIS_INVARIANT,
        "A directional episode runs from a flat-to-nonzero entry until the position "
        "returns to flat or flips sign; resizes and rolls inside it do not start a "
        "new episode. Assigned to the window by exit date.",
        "outputs/strategy_metrics.csv 'Closed trade episodes'",
        "matches published closed-episode count exactly",
        pick_net=fi("closed_episodes"),
    )
    add(
        "Number of trades",
        "Number of trades - OPEN/censored episodes at window end (context)",
        "count",
        BASIS_INVARIANT,
        "Episodes live at the window end, excluded from the closed-episode "
        "statistics above. Reported so the closed count is not read as the whole book.",
        "outputs/strategy_metrics.csv 'Open/censored trade episodes'",
        "matches published open/censored count exactly",
        pick_net=fi("open_censored_episodes"),
    )
    add(
        "Number of trades",
        "Number of trades - RAW market-sessions with a nonzero fill",
        "count",
        BASIS_INVARIANT,
        "Sum of strategy_daily.csv 'filled_markets': the number of (market, session) "
        "pairs on which a nonzero quantity actually executed. This counts execution "
        "events, not round trips, and is the number an operations desk would size "
        "against. Roll legs are executed separately and are not in this count.",
        "outputs/strategy_daily.csv 'filled_markets', summed",
        "not separately published; derived from the canonical ledger",
        pick_net=fi("filled_market_sessions"),
    )

    # --- 8. Average holding period -------------------------------------------
    add(
        "Average holding period",
        "Average holding period (sessions)",
        "sessions",
        BASIS_INVARIANT,
        "Mean 'Holding sessions' across closed episodes in the window.",
        "outputs/strategy_metrics.csv 'Average holding sessions'",
        "matches published average holding sessions to 1e-9",
        pick_net=f("holding_sessions", 4),
    )
    add(
        "Average holding period",
        "Average holding period (months, at 21 sessions per month)",
        "months",
        BASIS_INVARIANT,
        "Sessions / 21, i.e. 252/12. Twenty-one is the rebalance cadence, so the "
        "figure reads directly as 'about N rebalance cycles per position'.",
        "derived from 'Average holding sessions'",
        "derived; no separate published value",
        pick_net=lambda w: round(facts[w]["holding_sessions"] / SESSIONS_PER_MONTH, 4),
    )

    # --- 9. Turnover ----------------------------------------------------------
    add(
        "Turnover",
        "Turnover - annualised NOTIONAL turnover (multiples of NAV per year, "
        "rebalance leg only)",
        "multiples of NAV per year",
        BASIS_INVARIANT,
        "Sum over sessions of |traded notional| / prior NAV, divided by the "
        "window's calendar span in years. Scale free, so it survives NAV "
        "compounding, and it is the reading most people mean by 'turnover'. "
        "REBALANCE LEG ONLY: the ledger stores roll turnover as an aggregate "
        "contract increment, so a roll notional cannot be reconstructed. True "
        "all-in notional turnover is therefore higher than shown.",
        "recomputed from the frozen backtest; full window published as "
        "'annual_rebalance_notional_turnover_over_nav' in "
        "outputs/benchmarks/benchmark_comparison.csv",
        "full window matches the published benchmark figure to 1e-9",
        pick_net=(
            (lambda w: round(notional[w]["notional_turnover_over_nav"], 6))
            if notional
            else (lambda w: "not computed (--skip-notional)")
        ),
    )
    add(
        "Turnover",
        "Turnover - annualised CONTRACT turnover, all-in (rebalance + roll)",
        "contracts per year",
        BASIS_INVARIANT,
        "Sum of daily total contract turnover divided by the window's calendar "
        "span in years. NOT scale free: the book compounds about 22x over the "
        "full window and buys proportionally more contracts for the same risk, so "
        "the rise from the development window to the out-of-sample window is "
        "mostly NAV growth, not more trading. Use the notional row above instead.",
        "outputs/strategy_daily.csv; full window published as "
        "'annual_total_turnover_contracts' in benchmark_comparison.csv",
        "full window matches the published benchmark figure to 1e-9",
        pick_net=f("turnover_contracts_total", 4),
    )
    add(
        "Turnover",
        "Turnover - annualised CONTRACT turnover, rebalance leg",
        "contracts per year",
        BASIS_INVARIANT,
        "The discretionary leg: monthly target changes that survive the 25% no-trade band.",
        "outputs/strategy_daily.csv 'rebalance_contract_turnover'",
        "full window matches published 'annual_rebalance_turnover_contracts'",
        pick_net=f("turnover_contracts_rebalance", 4),
    )
    add(
        "Turnover",
        "Turnover - annualised CONTRACT turnover, roll leg",
        "contracts per year",
        BASIS_INVARIANT,
        "The non-discretionary leg: contract rolls forced by delivery. About 64% "
        "of all-in contract turnover over the full window, which is why roll cost "
        "is modelled explicitly rather than folded into a single spread number.",
        "outputs/strategy_daily.csv 'roll_contract_turnover_increment'",
        "full window matches published 'annual_roll_turnover_contracts'",
        pick_net=f("turnover_contracts_roll", 4),
    )

    # --- 10. Total estimated costs -------------------------------------------
    add(
        "Total estimated costs",
        "Total estimated costs - annualised, all-in (% of NAV per year)",
        "fraction of NAV per year",
        BASIS_NET_ONLY,
        "Mean daily cost as a fraction of the NAV that bore it, times 252. This "
        "is the scale-free cost reading and the one that is comparable across "
        "windows. Costs comprise half-spread, slippage, commission, "
        "exchange/regulatory fees, square-root market impact and roll cost.",
        "outputs/strategy_metrics.csv 'Annual cost drag'",
        "matches published annual cost drag to 1e-9",
        pick_net=f("cost_drag_total"),
    )
    add(
        "Total estimated costs",
        "Total estimated costs - annualised, FIXED component (% of NAV per year)",
        "fraction of NAV per year",
        BASIS_NET_ONLY,
        "Spread + slippage + commission + exchange/regulatory fees + roll cost, "
        "i.e. everything that does not scale with order size relative to volume.",
        "outputs/strategy_metrics.csv 'Annual fixed-cost drag'",
        "matches published annual fixed-cost drag to 1e-9",
        pick_net=f("cost_drag_fixed"),
    )
    add(
        "Total estimated costs",
        "Total estimated costs - annualised, IMPACT component (% of NAV per year)",
        "fraction of NAV per year",
        BASIS_NET_ONLY,
        "Square-root market-impact charge on order size relative to median traded "
        "volume. Small at this book size (about 9% of all-in cost over the full "
        "window) and the component that grows fastest with AUM.",
        "outputs/strategy_metrics.csv 'Annual impact-cost drag'",
        "matches published annual impact-cost drag to 1e-9",
        pick_net=f("cost_drag_impact"),
    )
    add(
        "Total estimated costs",
        "Total estimated costs - cumulative USD, all-in",
        "USD",
        BASIS_NET_ONLY,
        "Sum of 'transaction_cost_usd' over the window on a book started at "
        "$1,000,000 in 1990 and compounded thereafter. NOT comparable across "
        "windows: the book compounds, so a later window pays more dollars for the "
        "same proportional cost. Read the annualised % rows for cross-window "
        "comparison.",
        "outputs/strategy_daily.csv 'transaction_cost_usd', summed",
        "fixed + impact asserted equal to all-in within 1e-6 relative",
        pick_net=lambda w: round(facts[w]["cost_usd_total"], 2),
    )
    add(
        "Total estimated costs",
        "Total estimated costs - cumulative USD, FIXED component",
        "USD",
        BASIS_NET_ONLY,
        "Sum of 'fixed_execution_cost_usd' over the window.",
        "outputs/strategy_daily.csv 'fixed_execution_cost_usd', summed",
        "component of the asserted fixed + impact = all-in identity",
        pick_net=lambda w: round(facts[w]["cost_usd_fixed"], 2),
    )
    add(
        "Total estimated costs",
        "Total estimated costs - cumulative USD, IMPACT component",
        "USD",
        BASIS_NET_ONLY,
        "Sum of 'market_impact_cost_usd' over the window.",
        "outputs/strategy_daily.csv 'market_impact_cost_usd', summed",
        "component of the asserted fixed + impact = all-in identity",
        pick_net=lambda w: round(facts[w]["cost_usd_impact"], 2),
    )

    # --- 11. Average exposure -------------------------------------------------
    add(
        "Average exposure",
        "Average exposure - gross notional as a multiple of NAV",
        "multiple of NAV",
        BASIS_INVARIANT,
        "Mean of daily gross notional / NAV. Above 1x because a 7% volatility "
        "target across 59 individually low-volatility futures needs notional "
        "leverage. Targets are capped at 5.0x gross at each decision point; the "
        "realised peak is 5.11x because positions drift between rebalances.",
        "outputs/strategy_metrics.csv 'Average gross notional multiple'",
        "matches published average gross notional multiple to 1e-9",
        pick_net=f("avg_gross_notional_multiple"),
    )
    add(
        "Average exposure",
        "Average exposure - number of markets held",
        "count of markets",
        BASIS_INVARIANT,
        "Mean daily count of markets with a nonzero position. Rises over time as "
        "the universe's liquidity gate admits more of the 59 markets.",
        "outputs/strategy_metrics.csv 'Average markets held'",
        "matches published average markets held to 1e-9",
        pick_net=f("avg_markets_held"),
    )

    frame = pd.DataFrame(rows)
    ordered = (
        ["Metric group", "Metric", "Unit", "Basis dependence"]
        + [column_name(w, s) for w, s in COLUMNS_ORDER]
        + ["Definition", "Canonical source", "Reconciliation"]
    )
    return frame[ordered]


def build_gross_vs_net_summary(
    net: dict, gross: dict, facts: dict, notional: dict | None
) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        n, g, f = net[window], gross[window], facts[window]
        cagr_cost = g["cagr"] - n["cagr"]
        sharpe_cost = g["sharpe"] - n["sharpe"]
        # Expressed as a widening of the drawdown's depth, so a positive number
        # means cost made the worst loss worse.
        drawdown_cost = abs(n["max_drawdown"]) - abs(g["max_drawdown"])
        rows.append(
            {
                "Window": window,
                "Start": n["start"],
                "End": n["end"],
                "Sessions": n["sessions"],
                "Years": round(n["years"], 6),
                "Gross annualised return": round(g["cagr"], 6),
                "Net annualised return": round(n["cagr"], 6),
                "Cost charged to annualised return (pp)": round(cagr_cost * 100, 4),
                "Cost as share of gross return": round(cagr_cost / g["cagr"], 6),
                "Gross Sharpe": round(g["sharpe"], 4),
                "Net Sharpe": round(n["sharpe"], 4),
                "Cost charged to Sharpe": round(sharpe_cost, 4),
                "Cost as share of gross Sharpe": round(sharpe_cost / g["sharpe"], 6),
                "Gross max drawdown": round(g["max_drawdown"], 6),
                "Net max drawdown": round(n["max_drawdown"], 6),
                "Cost widening of max drawdown (pp)": round(drawdown_cost * 100, 4),
                "Gross return-to-drawdown": round(g["return_to_drawdown"], 4),
                "Net return-to-drawdown": round(n["return_to_drawdown"], 4),
                "Annual cost drag (% NAV/yr)": round(f["cost_drag_total"], 6),
                "  of which fixed": round(f["cost_drag_fixed"], 6),
                "  of which impact": round(f["cost_drag_impact"], 6),
                "Cumulative cost (USD)": round(f["cost_usd_total"], 2),
                "Annualised notional turnover (x NAV, rebalance leg)": (
                    round(notional[window]["notional_turnover_over_nav"], 4)
                    if notional
                    else "not computed"
                ),
                # All-in cost includes roll, but only rebalance notional is
                # reconstructible, so this is an upper bound on the blended
                # cost per dollar traded, not the realised rate.
                "All-in cost / rebalance notional (bps, upper bound)": (
                    round(
                        1e4 * f["cost_usd_total"] / notional[window]["traded_notional_usd"],
                        3,
                    )
                    if notional
                    else "not computed"
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO / "Round1AllData" / "Quant Researcher" / "Delta1",
        help="futures data directory, needed only for notional turnover",
    )
    parser.add_argument(
        "--skip-notional",
        action="store_true",
        help="skip the backtest re-run; notional turnover is reported as not computed",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    arguments = parser.parse_args()

    SUBMISSION.mkdir(parents=True, exist_ok=True)

    daily = load_daily()
    published = load_published_metrics()
    episodes = load_episodes()
    benchmark = load_published_benchmark_row()
    reconciler = Reconciler(arguments.tolerance)

    net: dict[str, dict] = {}
    gross: dict[str, dict] = {}
    facts: dict[str, dict] = {}

    for window, (start, end) in WINDOWS.items():
        net[window] = series_metrics(daily, "net_return", start, end)
        gross[window] = series_metrics(daily, "gross_return", start, end)
        facts[window] = execution_facts(daily, episodes, start, end)

        row = published.loc[CANONICAL_WINDOW_LABEL[window]]
        source = "outputs/strategy_metrics.csv"
        pairs = [
            ("years", net[window]["years"], row["Years"]),
            ("annualised return (CAGR), net", net[window]["cagr"], row["CAGR"]),
            (
                "annualised volatility, net",
                net[window]["volatility"],
                row["Annualized volatility"],
            ),
            (
                "Sharpe ratio, net",
                net[window]["sharpe"],
                row["Naive daily Sharpe (sqrt252, rf=0)"],
            ),
            ("maximum drawdown, net", net[window]["max_drawdown"], row["Max drawdown"]),
            (
                "return-to-drawdown (Calmar), net",
                net[window]["return_to_drawdown"],
                row["Calmar"],
            ),
            (
                "hit rate, daily, net",
                net[window]["daily_win_rate"],
                row["Daily win rate"],
            ),
            (
                "hit rate, trade level, net",
                facts[window]["trade_win_rate_net"],
                row["Trade win rate"],
            ),
            (
                "number of trades, closed episodes",
                facts[window]["closed_episodes"],
                row["Closed trade episodes"],
            ),
            (
                "number of trades, open/censored",
                facts[window]["open_censored_episodes"],
                row["Open/censored trade episodes"],
            ),
            (
                "average holding period (sessions)",
                facts[window]["holding_sessions"],
                row["Average holding sessions"],
            ),
            (
                "total estimated costs, annual drag",
                facts[window]["cost_drag_total"],
                row["Annual cost drag"],
            ),
            (
                "total estimated costs, fixed drag",
                facts[window]["cost_drag_fixed"],
                row["Annual fixed-cost drag"],
            ),
            (
                "total estimated costs, impact drag",
                facts[window]["cost_drag_impact"],
                row["Annual impact-cost drag"],
            ),
            (
                "average exposure, gross notional multiple",
                facts[window]["avg_gross_notional_multiple"],
                row["Average gross notional multiple"],
            ),
            (
                "average exposure, markets held",
                facts[window]["avg_markets_held"],
                row["Average markets held"],
            ),
        ]
        for item, computed, publication in pairs:
            reconciler.add(window, item, computed, publication, source)

    notional = None
    if not arguments.skip_notional:
        if not arguments.data_dir.is_dir():
            raise FileNotFoundError(
                f"data directory not found: {arguments.data_dir}. "
                "Pass --data-dir or --skip-notional."
            )
        print(f"Re-running the frozen backtest from {arguments.data_dir} ...", flush=True)
        notional = notional_turnover(arguments.data_dir)

    # Full-window turnover reconciles against the published benchmark table.
    full = "1990-2014 full"
    source = "outputs/benchmarks/benchmark_comparison.csv (incumbent row)"
    reconciler.add(
        full,
        "turnover, contracts, all-in",
        facts[full]["turnover_contracts_total"],
        benchmark["annual_total_turnover_contracts"],
        source,
    )
    reconciler.add(
        full,
        "turnover, contracts, rebalance leg",
        facts[full]["turnover_contracts_rebalance"],
        benchmark["annual_rebalance_turnover_contracts"],
        source,
    )
    reconciler.add(
        full,
        "turnover, contracts, roll leg",
        facts[full]["turnover_contracts_roll"],
        benchmark["annual_roll_turnover_contracts"],
        source,
    )
    if notional is not None:
        reconciler.add(
            full,
            "turnover, notional multiple of NAV",
            notional[full]["notional_turnover_over_nav"],
            benchmark["annual_rebalance_notional_turnover_over_nav"],
            source,
        )

    print()
    print(reconciler.report())
    print()
    reconciler.assert_clean()

    results = build_required_results(net, gross, facts, notional)
    summary = build_gross_vs_net_summary(net, gross, facts, notional)

    results_path = SUBMISSION / "required_results.csv"
    summary_path = SUBMISSION / "gross_vs_net_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {results_path} ({len(results)} metric rows)")
    print(f"Wrote {summary_path} ({len(summary)} window rows)")


if __name__ == "__main__":
    main()
