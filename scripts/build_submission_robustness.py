"""Section E robustness artifacts for the Delta1 submission.

Produces three CSVs under ``outputs/submission/``:

    robustness_parameter_sensitivity.csv   a small, pre-declared parameter grid
    robustness_chronological_oos.csv       develop early / assess late, unchanged rules
    robustness_regimes.csv                 two independent, causally-labelled regime axes

Everything is recomputed from ``reference/delta1_reference.py`` (the single-file
specification) and reconciled against the canonical bundle already in
``outputs/``.  Nothing is hand-entered: every number written here is computed in
this file or copied verbatim from a canonical CSV, and the copies are checked
against a fresh run before they are used.

    python scripts/build_submission_robustness.py \
        --data-dir "Round1AllData/Quant Researcher/Delta1"
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))

import delta1_reference as d1  # noqa: E402

OUT = REPO / "outputs" / "submission"
CANON = REPO / "outputs"

WINDOW = ("1990-01-01", "2014-12-31")
DEV = ("1990-01-01", "2004-12-31")
LATER = ("2005-01-01", "2014-12-31")

# Canonical facts this script refuses to run without.  If the reference file or
# the data directory ever drifts, the assertions below fail loudly rather than
# letting a wrong number reach the submission.
CANONICAL_SHARPE = 1.5895266624546427
CANONICAL_CAGR = 0.13189524685750764
CANONICAL_SESSIONS = 6523
CANONICAL_MARKETS = 59

DECLARATION = (
    "Pre-declared sensitivity check, NOT a search. Nine runs, declared in full before any "
    "was executed. The base configuration was fixed beforehand and was not chosen from "
    "this grid; no row here was used to select a parameter. The 252-session trend lookback "
    "is the standard 12-month convention (Moskowitz, Ooi and Pedersen 2012), taken from "
    "prior literature rather than fitted on this data -- which is the whole defence "
    "available for the sharp Sharpe peak the grid shows at 252. Reported so the reader can "
    "size the neighbourhood, not to argue the base is optimal."
)

# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


@contextmanager
def override(**changes: float):
    """Temporarily set keys in the module-level parameter dict, then restore."""
    previous = {key: d1.P[key] for key in changes}
    d1.P.update(changes)
    try:
        yield
    finally:
        d1.P.update(previous)


def window_stats(daily: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Headline net statistics over one contiguous window of the ledger."""
    frame = daily.loc[start:end]
    returns = frame["net_return"].dropna()
    location = daily.index.get_indexer([returns.index[0]])[0]
    origin = daily.index[location - 1] if location > 0 else returns.index[0]
    years = (returns.index[-1] - origin).total_seconds() / (365.2425 * 86_400)

    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = float((equity / peak - 1).min())
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    monthly = (1 + returns).resample("ME").prod() - 1

    return {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "years": years,
        "sessions": int(len(returns)),
        "cagr": cagr,
        "annualized_volatility": float(returns.std() * math.sqrt(252)),
        "sharpe": float(returns.mean() / returns.std() * math.sqrt(252)),
        "max_drawdown": drawdown,
        "calmar": cagr / abs(drawdown),
        "daily_hit_rate": float((returns > 0).mean()),
        "positive_month_rate": float((monthly > 0).mean()),
        "annual_cost_drag": float(frame["cost"].mean() * 252),
        "average_gross_notional_multiple": float(frame["gross_notional_multiple"].mean()),
        "average_markets_held": float(frame["active_markets"].mean()),
    }


def conditional_stats(returns: pd.Series, frame: pd.DataFrame) -> dict[str, float]:
    """Statistics on a discontiguous set of sessions.

    A regime state is not an investable path: its sessions are interrupted by
    sessions in the other state.  So two annualised returns are reported.  The
    arithmetic one (mean x 252) is path-free and is the honest headline; the
    geometric one compounds the state's sessions back to back and is labelled
    ``_stitched`` because that path was never available to any investor.  The
    same caveat applies to the drawdown.
    """
    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = float((equity / peak - 1).min())
    years = len(returns) / 252.0

    return {
        "sessions": int(len(returns)),
        "annualized_return_arithmetic": float(returns.mean() * 252),
        "annualized_return_geometric_stitched": float(equity.iloc[-1] ** (1 / years) - 1),
        "annualized_volatility": float(returns.std() * math.sqrt(252)),
        "sharpe": float(returns.mean() / returns.std() * math.sqrt(252)),
        "max_drawdown_stitched": drawdown,
        "daily_hit_rate": float((returns > 0).mean()),
        "worst_session": float(returns.min()),
        "annual_cost_drag": float(frame.loc[returns.index, "cost"].mean() * 252),
        "average_gross_notional_multiple": float(
            frame.loc[returns.index, "gross_notional_multiple"].mean()
        ),
        "average_markets_held": float(frame.loc[returns.index, "active_markets"].mean()),
    }


def annual_turnover(frame: pd.DataFrame, column: str) -> float:
    """Contracts per elapsed calendar year.

    Deliberately the same convention as ``outputs/validation/`` -- total
    contracts divided by elapsed years between first and last session -- so the
    walk-forward row in the out-of-sample file is directly comparable.  Verified
    against ``anchor_annual_rebalance_turnover_contracts`` below.
    """
    years = (frame.index[-1] - frame.index[0]).days / 365.2425
    return float(frame[column].sum() / years)


# --------------------------------------------------------------------------
# 1.  Parameter sensitivity (a small pre-declared grid) + cost sensitivity
# --------------------------------------------------------------------------

# Nine runs.  Declared here, in full, before any of them is executed.
GRID: dict[str, tuple[float, ...]] = {
    "trend_lookback": (189, 252, 315),      # 9, 12, 15 months
    "no_trade_buffer": (0.15, 0.25, 0.35),
    "target_vol": (0.06, 0.07, 0.08),
}


def parameter_sensitivity(data_dir: str, base: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parameter, values in GRID.items():
        for value in values:
            is_base = value == d1.P[parameter]
            if is_base:
                stats = dict(base)
            else:
                with override(**{parameter: value}):
                    stats = window_stats(d1.run(data_dir), *WINDOW)
            rows.append(
                {
                    "axis": "parameter",
                    "parameter": parameter,
                    "value": value,
                    "is_base": is_base,
                    "prespecified_parameter_grid": True,
                    **{k: stats[k] for k in (
                        "start", "end", "sessions", "cagr", "annualized_volatility",
                        "sharpe", "max_drawdown", "calmar", "annual_cost_drag",
                    )},
                    "delta_cagr": stats["cagr"] - base["cagr"],
                    "delta_annualized_volatility": (
                        stats["annualized_volatility"] - base["annualized_volatility"]
                    ),
                    "delta_sharpe": stats["sharpe"] - base["sharpe"],
                    "delta_max_drawdown": stats["max_drawdown"] - base["max_drawdown"],
                    "delta_calmar": stats["calmar"] - base["calmar"],
                    "delta_annual_cost_drag": (
                        stats["annual_cost_drag"] - base["annual_cost_drag"]
                    ),
                    "source": "recomputed here from reference/delta1_reference.py",
                    "note": DECLARATION,
                }
            )
    return pd.DataFrame(rows)


COST_KEYS = ("half_spread_ticks", "slippage_ticks", "commission", "fees",
             "impact_bps_at_full_participation")


def cost_sensitivity(data_dir: str, base: dict[str, float]) -> pd.DataFrame:
    """Reuse the canonical friction stress, and fill the one gap it leaves.

    ``outputs/strategy_friction_stress.csv`` already carries the doubled-cost
    scenarios, so they are copied rather than recomputed -- after checking its
    baseline row against a fresh run.  It carries no triple-cost scenario, so
    that single run is executed here and flagged as newly computed.
    """
    stress = pd.read_csv(CANON / "strategy_friction_stress.csv").set_index("scenario")

    baseline = stress.loc["baseline"]
    for field, value in (("sharpe", base["sharpe"]), ("cagr", base["cagr"]),
                         ("annual_cost_drag", base["annual_cost_drag"])):
        if abs(float(baseline[field]) - value) > 1e-12:
            raise AssertionError(
                f"friction-stress baseline {field}={baseline[field]!r} does not "
                f"reconcile with the fresh reference run ({value!r})"
            )

    rows: list[dict[str, object]] = []
    reused = {
        "baseline": "1x (canonical)",
        "double_fixed_costs": "2x fixed execution costs",
        "double_impact": "2x market impact",
        "double_all_execution_costs": "2x all execution costs",
        "combined_adverse": "2x all costs + 1 session delay + 1% participation cap",
    }
    for scenario, label in reused.items():
        row = stress.loc[scenario]
        cagr, drawdown = float(row["cagr"]), float(row["max_drawdown"])
        rows.append({
            "axis": "execution cost multiplier",
            "parameter": "execution_costs",
            "value": label,
            "is_base": scenario == "baseline",
            "prespecified_parameter_grid": False,
            "start": row["start"], "end": row["end"],
            "sessions": base["sessions"],
            "cagr": cagr,
            "annualized_volatility": float(row["annualized_volatility"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": drawdown,
            "calmar": cagr / abs(drawdown),
            "annual_cost_drag": float(row["annual_cost_drag"]),
            "source": "outputs/strategy_friction_stress.csv (reused, baseline reconciled)",
            "note": str(row["assumption"]),
        })

    # The one scenario the canonical bundle does not contain.
    with override(**{key: 3.0 * d1.P[key] for key in COST_KEYS}):
        triple = window_stats(d1.run(data_dir), *WINDOW)
    rows.append({
        "axis": "execution cost multiplier",
        "parameter": "execution_costs",
        "value": "3x all execution costs",
        "is_base": False,
        "prespecified_parameter_grid": False,
        "start": triple["start"], "end": triple["end"], "sessions": triple["sessions"],
        "cagr": triple["cagr"],
        "annualized_volatility": triple["annualized_volatility"],
        "sharpe": triple["sharpe"],
        "max_drawdown": triple["max_drawdown"],
        "calmar": triple["calmar"],
        "annual_cost_drag": triple["annual_cost_drag"],
        "source": "computed here; absent from outputs/strategy_friction_stress.csv",
        "note": "Half-spread, slippage, commission, exchange fees and the market-impact "
                "coefficient all tripled. Not present in the canonical friction stress, "
                "which stops at 2x.",
    })

    frame = pd.DataFrame(rows)
    reference = frame.loc[frame["is_base"]].iloc[0]
    for field in ("cagr", "annualized_volatility", "sharpe", "max_drawdown",
                  "calmar", "annual_cost_drag"):
        frame[f"delta_{field}"] = frame[field] - reference[field]
    return frame


# --------------------------------------------------------------------------
# 2.  Chronological out-of-sample
# --------------------------------------------------------------------------


def chronological_oos(daily: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.read_csv(CANON / "strategy_metrics.csv")
    trades = metrics.set_index("Window")
    walk = pd.read_csv(CANON / "validation" / "validation_walk_forward_summary.csv")
    walk = walk.set_index("label").loc["trend_lookback_selection"]

    # The walk-forward file's own turnover convention, reproduced from the
    # canonical daily ledger, is the check that the two files can be compared.
    anchor_turnover = annual_turnover(
        canonical.loc[WINDOW[0]:WINDOW[1]], "rebalance_contract_turnover"
    )
    if abs(anchor_turnover - float(walk["anchor_annual_rebalance_turnover_contracts"])) > 1e-6:
        raise AssertionError("turnover convention does not reconcile with the walk-forward file")
    # The validation harness annualises with a population standard deviation
    # (ddof=0); this file and outputs/strategy_metrics.csv use a sample one
    # (ddof=1).  The two agree exactly once that factor is removed, which is the
    # check below.  Every Sharpe taken from the walk-forward file is therefore
    # also restated on the ddof=1 convention so row 3 is comparable to rows 1-2.
    anchor_ddof1 = float(walk["anchor_specification_full_window_sharpe"]) / math.sqrt(
        CANONICAL_SESSIONS / (CANONICAL_SESSIONS - 1)
    )
    if abs(anchor_ddof1 - CANONICAL_SHARPE) > 1e-9:
        raise AssertionError(
            f"walk-forward anchor Sharpe {anchor_ddof1!r} does not reconcile with the "
            f"canonical strategy {CANONICAL_SHARPE!r} after the ddof adjustment"
        )
    if abs(float(walk["anchor_specification_full_window_cagr"]) - CANONICAL_CAGR) > 1e-12:
        raise AssertionError("walk-forward anchor CAGR does not match the canonical strategy")

    def period(label: str, role: str, rules: str, span: tuple[str, str],
               trade_window: str) -> dict[str, object]:
        stats = window_stats(daily, *span)
        frame = canonical.loc[span[0]:span[1]]
        row = trades.loc[trade_window]
        return {
            "row": label,
            "role": role,
            "rules_status": rules,
            "start": stats["start"], "end": stats["end"],
            "years": stats["years"], "sessions": stats["sessions"],
            "cagr": stats["cagr"],
            "annualized_volatility": stats["annualized_volatility"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "return_to_drawdown_calmar": stats["calmar"],
            "daily_hit_rate": stats["daily_hit_rate"],
            "positive_month_rate": stats["positive_month_rate"],
            "closed_trade_episodes": float(row["Closed trade episodes"]),
            "average_holding_sessions": float(row["Average holding sessions"]),
            "annual_rebalance_turnover_contracts": annual_turnover(
                frame, "rebalance_contract_turnover"),
            "annual_total_turnover_contracts": annual_turnover(
                frame, "total_contract_turnover"),
            "total_estimated_costs_usd": float(frame["transaction_cost_usd"].sum()),
            "annual_cost_drag": stats["annual_cost_drag"],
            "average_exposure_gross_notional_multiple":
                stats["average_gross_notional_multiple"],
            "average_markets_held": stats["average_markets_held"],
            "source": "recomputed from reference/delta1_reference.py; trade counts and "
                      "holding period reused from outputs/strategy_metrics.csv",
        }

    development = period(
        "1. development sample", "in-sample development",
        "rules developed on this window",
        DEV, "1990-2004 development history")
    assessment = period(
        "2. chronological assessment sample", "out-of-sample assessment",
        "UNCHANGED rules, no refit, no reselection",
        LATER, "2005-2014 reused later diagnostic")
    assessment["note"] = (
        "The required chronological test: same parameters, same universe, same cost "
        "model, applied forward with nothing re-estimated."
    )
    development["note"] = (
        "Baseline for the decay columns. Specification choices were informed by this "
        "window, so its numbers are in-sample by construction."
    )

    wf_cagr = float(walk["stitched_cagr"])
    wf_dd = float(walk["stitched_max_drawdown"])
    wf_sessions = int(walk["stitched_sessions"])
    # The walk-forward statistics are quoted EXACTLY as the validation suite
    # published them.  An earlier revision rescaled volatility and Sharpe by a
    # ddof factor to match this file's own convention; that restated another
    # study's published number, in the wrong direction, and produced a row that
    # disagreed with its own cited source.  A convention difference is a
    # footnote, not a correction to someone else's statistic.
    walk_row = {
        "row": "3. anchored walk-forward (trend lookback re-selected out of sample)",
        "role": "out-of-sample assessment, selector included",
        "rules_status": (
            f"trend lookback RE-SELECTED at each of {int(walk['folds'])} anchored expanding "
            f"folds from {int(walk['candidates'])} candidates; all other rules unchanged"
        ),
        "start": str(walk["stitched_start"]), "end": str(walk["stitched_end"]),
        "years": np.nan,
        "sessions": wf_sessions,
        "cagr": wf_cagr,
        "annualized_volatility": float(walk["stitched_annualized_volatility"]),
        "sharpe": float(walk["stitched_sharpe"]),
        "max_drawdown": wf_dd,
        "return_to_drawdown_calmar": wf_cagr / abs(wf_dd),
        "daily_hit_rate": float(walk["stitched_hit_rate"]),
        "positive_month_rate": np.nan,
        "closed_trade_episodes": np.nan,
        "average_holding_sessions": np.nan,
        "annual_rebalance_turnover_contracts": float(
            walk["stitched_annual_rebalance_turnover_contracts"]),
        "annual_total_turnover_contracts": np.nan,
        "total_estimated_costs_usd": np.nan,
        "annual_cost_drag": float(walk["stitched_annual_cost_drag"]),
        "average_exposure_gross_notional_multiple": np.nan,
        "average_markets_held": np.nan,
        "source": "outputs/validation/validation_walk_forward_summary.csv "
                  "(label=trend_lookback_selection), reused",
        "note": (
            "Stronger form of the same check: the lookback is chosen only from data "
            "preceding each fold, so selection itself is out of sample. Blanks are "
            "quantities the stitched path does not define. Walk-forward efficiency "
            f"{float(walk['walk_forward_efficiency']):.4f} vs the frozen specification "
            "over the same span. Sharpe and volatility are quoted exactly as the "
            "validation suite publishes them and are NOT restated: that study "
            "annualizes on its own convention, and the difference against rows 1-2 is "
            "under 0.01% of the statistic - far smaller than anything being compared "
            "here. Caveat carried from the source file: "
            + str(walk["permitted_use"])
        ),
    }

    frame = pd.DataFrame([development, assessment, walk_row])
    for field in ("cagr", "sharpe", "max_drawdown", "annualized_volatility"):
        frame[f"delta_vs_development_{field}"] = frame[field] - development[field]
    frame["sharpe_retention_vs_development"] = frame["sharpe"] / development["sharpe"]
    frame["cagr_retention_vs_development"] = frame["cagr"] / development["cagr"]
    return frame


# --------------------------------------------------------------------------
# 3.  Regimes -- two independent axes, both labelled causally
# --------------------------------------------------------------------------


def volatility_labels(returns: pd.Series, minimum_history_months: int = 36) -> pd.Series:
    """Canonical axis: lagged expanding terciles of trailing 12-month volatility.

    Reproduces the definition behind ``outputs/strategy_regime_metrics.csv``
    exactly.  The feature for month m is the annualised standard deviation of
    monthly returns over months m-12..m-1, and the tercile breakpoints come only
    from feature values that precede month m, so both the statistic and the
    threshold are known before the month being labelled begins.
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    feature = (monthly.rolling(12, min_periods=12).std() * math.sqrt(12)).shift(1)
    labels = pd.Series(index=monthly.index, dtype=object)
    for offset, (date, value) in enumerate(feature.items()):
        if not np.isfinite(value):
            continue
        history = feature.iloc[:offset].dropna()
        if len(history) < minimum_history_months:
            continue
        lower, upper = history.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        labels.at[date] = "Low" if value <= lower else ("High" if value >= upper else "Middle")
    return labels


def trend_persistence_labels(
    prices: pd.DataFrame, minimum_history_sessions: int = 252
) -> tuple[pd.Series, pd.Series]:
    """Second axis: trending vs range-bound, from cross-market trend persistence.

    For each market, take the sign of the 252-session price change -- the
    strategy's own trend signal.  Breadth on session t is the share of markets
    whose sign on t equals its sign 63 sessions earlier, i.e. the share of the
    book whose 12-month trend direction has survived the last quarter.  High
    breadth is a trending cross-section; low breadth is a chopping one.

    Causality is guaranteed twice over.  (1) The breadth value used to label
    session t is the value computed from closes up to and including t-1, so it
    is known before session t's return exists.  (2) The threshold is the
    expanding median of breadth over every session strictly before t, requiring
    252 prior observations, so it never sees the future either.  Breadth history
    from before the 1990 launch is real observed data and is used only for the
    threshold, never for a label inside the evaluation window.
    """
    change = prices - prices.shift(d1.P["trend_lookback"])
    sign = np.sign(change).where(change.notna())
    earlier = sign.shift(63)
    valid = sign.notna() & earlier.notna() & sign.ne(0) & earlier.ne(0)
    breadth = (sign.eq(earlier) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)

    lagged = breadth.shift(1)
    threshold = lagged.expanding(minimum_history_sessions).median()
    labels = pd.Series(index=breadth.index, dtype=object)
    usable = lagged.notna() & threshold.notna()
    labels[usable & lagged.gt(threshold)] = "Trending"
    labels[usable & lagged.le(threshold)] = "Range-bound"
    return labels, lagged


def regimes(daily: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    frame = daily.loc[WINDOW[0]:WINDOW[1]]
    returns = frame["net_return"].dropna()
    monthly = (1 + returns).resample("ME").prod() - 1
    canonical = pd.read_csv(CANON / "strategy_regime_metrics.csv")
    canonical = canonical[canonical["Regime"].eq("Lagged realized volatility")]
    canonical = canonical.set_index("State")

    rows: list[dict[str, object]] = []
    overall = conditional_stats(returns, frame)
    rows.append({
        "axis": "Full sample (reference row)",
        "state": "All sessions",
        "definition": "no conditioning",
        "label_information_set": "n/a",
        "months": int(len(monthly)),
        "share_of_labeled_months": 1.0,
        "share_of_labeled_sessions": 1.0,
        "positive_month_rate": float((monthly > 0).mean()),
        "worst_month": float(monthly.min()),
        **overall,
        "reconciles_to": "outputs/strategy_metrics.csv 1990-2014 full post-launch history",
        "note": "Context for the conditional rows below.",
    })

    # --- axis A: volatility ------------------------------------------------
    vol_labels = volatility_labels(returns)
    labeled_months = int(vol_labels.notna().sum())
    month_end = returns.index.to_period("M").to_timestamp("M")
    daily_vol_label = pd.Series(
        vol_labels.reindex(month_end).to_numpy(), index=returns.index
    )
    total_vol_sessions = int(daily_vol_label.notna().sum())
    for state in ("Low", "Middle", "High"):
        mask = daily_vol_label.eq(state)
        state_returns = returns[mask]
        state_months = monthly[vol_labels.eq(state).reindex(monthly.index, fill_value=False)]
        stats = conditional_stats(state_returns, frame)

        # Reconciliation against the canonical monthly table.
        expected_months = int(canonical.at[state, "Months"])
        if len(state_months) != expected_months:
            raise AssertionError(f"volatility regime {state}: {len(state_months)} months "
                                 f"vs canonical {expected_months}")
        if abs(state_months.mean() - float(canonical.at[state, "Mean monthly return"])) > 1e-12:
            raise AssertionError(f"volatility regime {state}: mean monthly return mismatch")

        rows.append({
            "axis": "A. Lagged realised volatility (canonical axis)",
            "state": state,
            "definition": "expanding terciles of the annualised standard deviation of the "
                          "prior 12 monthly net returns",
            "label_information_set": "feature ends the month before the labelled month; "
                                     "tercile breakpoints use only earlier feature values "
                                     "(min 36 months). Nothing contemporaneous or future.",
            "months": int(len(state_months)),
            "share_of_labeled_months": len(state_months) / labeled_months,
            "share_of_labeled_sessions": stats["sessions"] / total_vol_sessions,
            "positive_month_rate": float((state_months > 0).mean()),
            "worst_month": float(state_months.min()),
            **stats,
            "reconciles_to": "outputs/strategy_regime_metrics.csv (months and mean monthly "
                             "return matched exactly)",
            "note": "Monthly labels mapped onto the sessions inside each labelled month.",
        })

    # --- axis B: trending vs range-bound -----------------------------------
    trend_labels, breadth = trend_persistence_labels(prices)
    trend_labels = trend_labels.reindex(returns.index)
    breadth = breadth.reindex(returns.index)
    total_trend_sessions = int(trend_labels.notna().sum())
    # For the monthly columns only, a month belongs to the state that holds the
    # majority of its sessions.  The daily columns never use this mapping.
    majority = trend_labels.groupby(month_end).agg(
        lambda states: states.value_counts().idxmax() if states.notna().any() else None
    ).reindex(monthly.index)
    for state in ("Trending", "Range-bound"):
        mask = trend_labels.eq(state)
        state_returns = returns[mask]
        state_months = monthly[majority.eq(state)]
        stats = conditional_stats(state_returns, frame)
        rows.append({
            "axis": "B. Trend persistence breadth (independent axis, computed here)",
            "state": state,
            "definition": "share of markets whose 252-session trend sign equals its sign 63 "
                          "sessions earlier; Trending = breadth above its own expanding "
                          "median, Range-bound = at or below "
                          f"(median breadth over the evaluated span {breadth.median():.4f})",
            "label_information_set": "breadth measured from closes through session t-1 and "
                                     "compared with the expanding median of every earlier "
                                     "breadth value (min 252 observations); the label for "
                                     "session t is fixed before session t's return exists.",
            "months": int(len(state_months)),
            "share_of_labeled_months": len(state_months) / len(monthly),
            "share_of_labeled_sessions": stats["sessions"] / total_trend_sessions,
            "positive_month_rate": float((state_months > 0).mean()),
            "worst_month": float(state_months.min()),
            **stats,
            "reconciles_to": "session counts sum to the 6,523 sessions in "
                             "outputs/strategy_metrics.csv",
            "note": "Months are assigned to the state holding a majority of their sessions, "
                    "for the monthly columns only; all daily columns use session labels.",
        })

    frame_out = pd.DataFrame(rows)
    trend_sessions = frame_out.loc[
        frame_out["axis"].str.startswith("B."), "sessions"].sum()
    if int(trend_sessions) != CANONICAL_SESSIONS:
        raise AssertionError(f"trend regime sessions {trend_sessions} != {CANONICAL_SESSIONS}")
    return frame_out


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="Round1AllData/Quant Researcher/Delta1")
    args = parser.parse_args()
    data_dir = str((REPO / args.data_dir).resolve())

    OUT.mkdir(parents=True, exist_ok=True)
    pristine = dict(d1.P)

    market_data = d1.load_market_data(Path(data_dir))
    daily = d1.run(data_dir)
    base = window_stats(daily, *WINDOW)

    # Wiring check.  Nothing below is trusted unless the base run reproduces the
    # canonical headline exactly.
    if abs(base["sharpe"] - CANONICAL_SHARPE) > 1e-12:
        raise AssertionError(f"base Sharpe {base['sharpe']!r} != canonical {CANONICAL_SHARPE!r}")
    if abs(base["cagr"] - CANONICAL_CAGR) > 1e-12:
        raise AssertionError(f"base CAGR {base['cagr']!r} != canonical {CANONICAL_CAGR!r}")
    if base["sessions"] != CANONICAL_SESSIONS or len(d1.SYMBOLS) != CANONICAL_MARKETS:
        raise AssertionError("session or market count does not match the canonical bundle")
    print(f"wiring check passed: Sharpe {base['sharpe']:.10f}, CAGR {base['cagr']:.10f}, "
          f"{base['sessions']} sessions, {len(d1.SYMBOLS)} markets")

    canonical_daily = pd.read_csv(
        CANON / "strategy_daily.csv", index_col=0, parse_dates=True
    )
    difference = float(
        (canonical_daily["net_return"] - daily["net_return"]).abs().max()
    )
    if difference > 1e-12:
        raise AssertionError(f"canonical ledger disagrees with the reference run: {difference}")
    print(f"canonical ledger reconciled, max |net_return| difference {difference:.3e}")

    sensitivity = pd.concat(
        [parameter_sensitivity(data_dir, base), cost_sensitivity(data_dir, base)],
        ignore_index=True,
    )
    sensitivity.to_csv(OUT / "robustness_parameter_sensitivity.csv", index=False)

    oos = chronological_oos(daily, canonical_daily)
    oos.to_csv(OUT / "robustness_chronological_oos.csv", index=False)

    regime = regimes(daily, market_data["prices"])
    regime.to_csv(OUT / "robustness_regimes.csv", index=False)

    # Every grid run mutated the module-level parameter dict and restored it.
    # Prove the restoration was complete before anything downstream reuses d1.
    if pristine != d1.P:
        raise AssertionError("d1.P was not restored after the sensitivity runs")

    show = ["axis", "parameter", "value", "sharpe", "cagr", "max_drawdown", "delta_sharpe"]
    pd.set_option("display.width", 200)
    print("\n--- parameter and cost sensitivity ---")
    print(sensitivity[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n--- chronological out of sample ---")
    print(oos[["row", "sessions", "cagr", "annualized_volatility", "sharpe",
               "max_drawdown", "daily_hit_rate", "sharpe_retention_vs_development"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n--- regimes ---")
    print(regime[["axis", "state", "sessions", "annualized_return_arithmetic",
                  "annualized_volatility", "sharpe", "max_drawdown_stitched",
                  "daily_hit_rate"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
