#!/usr/bin/env python3
"""Build, replay and audit the ETF dynamic regime allocation sleeve.

The hiring case asks for two things the incumbent futures CTA cannot both
supply: a dynamic regime allocation strategy for ETFs, and at least five years
of out-of-sample forecast under rolling walk-forward analysis.  The futures
panel ends 2014-12-31 and its only continuation ends 2016-12-30, so it cannot
carry five forward years.  The ETF panel ends 2018-12-31, and this script is
what turns it into a record a reviewer can check by arithmetic.

It produces two out-of-sample claims and never merges them into one sentence.
The stitched rolling record is out of sample with respect to the *selector*: the
annual choice among the three declared candidates never saw the year it governs,
nor the forty-four sessions before it.  The sealed block 2014-2018 is out of
sample with respect to everything sealed at registration time, and the custody
report proves by byte-identical replay that the development path could not have
read a sealed session.  That proof is narrower than "sealed before any fitting
decision", and the artifacts say so: the annual selector keeps fitting inside
the block, which is what the strategy does and is fully causal, and the
specification around it was written by someone who had already read the whole
panel.

It writes only into its own output directory.  The canonical futures bundle and
the frozen run manifest are never touched, and ``save_outputs`` is never called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from delta1_strategy.marketdata import etfs
from delta1_strategy.research import allocation


DEFAULT_OUTPUT_DIR = Path("outputs/etf")
DEFAULT_FIRST_BOUNDARY_YEAR = 2009
DEFAULT_LAST_BOUNDARY_YEAR = 2018
DEFAULT_SEALED_START = "2014-01-01"
DEFAULT_END = "2018-12-31"


def _print(title: str, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    print(f"\n=== {title} ===")
    view = frame if columns is None else frame.loc[:, columns]
    print(view.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="root of the vendor extract; defaults to DELTA1_DATA_DIR",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--first-boundary-year", type=int, default=DEFAULT_FIRST_BOUNDARY_YEAR
    )
    parser.add_argument(
        "--last-boundary-year", type=int, default=DEFAULT_LAST_BOUNDARY_YEAR
    )
    parser.add_argument("--sealed-start", default=DEFAULT_SEALED_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--volatility-budget",
        type=float,
        default=0.07,
        help="annualized risk budget; 0.07 matches the incumbent futures CTA",
    )
    parser.add_argument("--half-spread-bps", type=float, default=2.0)
    parser.add_argument("--commission-bps", type=float, default=0.5)
    parser.add_argument("--impact-coefficient-bps", type=float, default=10.0)
    parser.add_argument("--participation-limit", type=float, default=0.05)
    parser.add_argument("--no-trade-band", type=float, default=0.02)
    parser.add_argument("--initial-capital-usd", type=float, default=100_000_000.0)
    parser.add_argument(
        "--cost-multipliers",
        type=float,
        nargs="+",
        default=list(allocation.DEFAULT_COST_MULTIPLIERS),
        help="cost sensitivity dial; the promotion-relevant level is 2.0",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    costs = allocation.CostModel(
        half_spread_bps=args.half_spread_bps,
        commission_bps=args.commission_bps,
        impact_coefficient_bps=args.impact_coefficient_bps,
        participation_limit=args.participation_limit,
    )
    config = allocation.AllocationConfig(
        volatility_budget_annualized=args.volatility_budget,
        no_trade_band=args.no_trade_band,
        initial_capital_usd=args.initial_capital_usd,
        costs=costs,
    )

    inputs = allocation.load_allocation_inputs(end=args.end, data_dir=args.data_dir)
    sessions = inputs.panel.returns.index
    plan = allocation.WalkForwardPlan(
        anchor=sessions[0].date().isoformat(),
        boundaries=allocation.annual_boundaries(
            sessions,
            first_year=args.first_boundary_year,
            last_year=args.last_boundary_year,
        ),
        end=args.end,
        sealed_start=args.sealed_start,
    )

    result = allocation.run_walk_forward(inputs, config, plan)
    references = allocation.reference_ledgers(inputs, config)
    scales = {
        "etf_regime_allocation": result.spliced.volatility_scale,
        **{
            candidate.name: allocation.desired_weight_frame(
                inputs, candidate, config
            ).volatility_scale
            for candidate in allocation.DECLARED_CANDIDATES
        },
    }
    ledgers = {
        "etf_regime_allocation": result.stitched,
        **result.candidate_ledgers,
        **references,
    }

    first_boundary = plan.boundaries[0]
    accounting = allocation.out_of_sample_accounting(result)
    degradation = allocation.in_sample_out_of_sample_report(result, config)
    dispersion = allocation.fold_dispersion_report(result.folds)
    full_sample = allocation.comparison_report(
        ledgers,
        window=f"full replay {sessions[0].date()}..{args.end}",
        basis=allocation.IN_SAMPLE_BASIS,
        permitted_use=(
            "the whole replay, which contains the in-sample training span; a "
            "description of one path and not out-of-sample evidence"
        ),
    )
    rolling = allocation.comparison_report(
        ledgers,
        window=f"rolling out of sample {first_boundary}..{args.end}",
        basis=allocation.SELECTOR_BASIS,
        start=first_boundary,
        end=args.end,
    )
    sealed = allocation.comparison_report(
        ledgers,
        window=f"sealed block {args.sealed_start}..{args.end}",
        basis=allocation.SEALED_BASIS,
        start=args.sealed_start,
        end=args.end,
        permitted_use=allocation.SEALED_PERMITTED_USE,
    )
    # The only full equity bear market in the panel runs from mid-2007 into
    # mid-2009, so it STRADDLES the first boundary: part of it is training span
    # and part of it sits inside the first out-of-sample fold.  Labelling the
    # whole episode "in sample" would be false, and truncating it at the
    # boundary would cut away the trough, which is the part worth reading.
    crisis_start, crisis_end = "2007-07-01", "2009-06-30"
    crisis = allocation.straddling_episode_report(
        ledgers,
        plan,
        episode="the only full equity bear market in the panel",
        start=crisis_start,
        end=crisis_end,
    )
    risk_matched = allocation.risk_matched_comparison(
        ledgers,
        reference_label="etf_regime_allocation",
        window=f"rolling out of sample {first_boundary}..{args.end}",
        start=first_boundary,
        end=args.end,
        annualization=config.annualization,
    )
    budget = allocation.volatility_budget_diagnostic(
        ledgers,
        config,
        scales=scales,
        window=f"rolling out of sample {first_boundary}..{args.end}",
        start=first_boundary,
        end=args.end,
    )
    exposure = allocation.exposure_share_report(
        result.stitched,
        inputs,
        window=f"rolling out of sample {first_boundary}..{args.end}",
        start=first_boundary,
        end=args.end,
    )
    paired = allocation.paired_reference_inference(
        ledgers,
        subject_label="etf_regime_allocation",
        reference_labels=list(references),
        start=first_boundary,
        end=args.end,
        annualization=config.annualization,
        hac_lags=config.hac_lags,
    )
    sweep = allocation.cost_sensitivity_report(
        inputs, config, plan, multipliers=tuple(args.cost_multipliers)
    )
    custody = allocation.holdout_custody_report(
        inputs, config, plan, data_dir=args.data_dir
    )
    coverage = allocation.sealed_block_state_coverage(inputs, plan)

    daily = result.stitched.daily.copy()
    daily.index.name = "Date"
    weights = result.stitched.weights.copy()
    weights.index.name = "Date"

    artifacts: dict[str, pd.DataFrame] = {
        "etf_universe.csv": etfs.universe_table(),
        "etf_universe_turnover_audit.csv": etfs.universe_turnover_audit(args.data_dir),
        "etf_excluded_candidates.csv": etfs.excluded_candidate_table(),
        "etf_allocation_candidates.csv": allocation.candidate_table(),
        "etf_cost_model_assumptions.csv": costs.assumption_table(),
        "etf_walk_forward_folds.csv": result.folds,
        "etf_out_of_sample_accounting.csv": accounting,
        "etf_fold_dispersion.csv": dispersion,
        "etf_in_sample_versus_out_of_sample.csv": degradation,
        "etf_comparison_full_replay.csv": full_sample,
        "etf_comparison_rolling_out_of_sample.csv": rolling,
        "etf_comparison_sealed_block.csv": sealed,
        "etf_comparison_equity_bear_market.csv": crisis,
        "etf_risk_matched_comparison.csv": risk_matched,
        "etf_volatility_budget_diagnostic.csv": budget,
        "etf_exposure_shares.csv": exposure,
        "etf_paired_reference_inference.csv": paired,
        "etf_cost_sensitivity.csv": sweep,
        "etf_holdout_custody.csv": custody,
        "etf_walk_forward_summary.csv": result.summary.to_frame().T,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)
    # Every report frame above carries the limitation stack on every row.  The
    # two raw series below are written without one, following the house
    # convention for ``outputs/strategy_daily.csv`` and
    # ``outputs/benchmarks/benchmark_daily_net_returns.csv``: a per-row string
    # repeated across three thousand sessions is noise, not disclosure.  Their
    # disclosure travels in ``etf_run_manifest.json``, which sits beside them
    # and lists the same four limitations.
    daily.to_csv(output_dir / "etf_stitched_daily.csv")
    weights.to_csv(output_dir / "etf_stitched_weights.csv")

    manifest = {
        "lineage_id": plan.lineage_id,
        "anchor": plan.anchor,
        "boundaries": list(plan.boundaries),
        "sealed_start": plan.sealed_start,
        "end": plan.end,
        "purge_sessions": plan.purge_sessions,
        "embargo_sessions": plan.embargo_sessions,
        "stand_off_sessions": plan.stand_off_sessions,
        "universe": list(inputs.tickers),
        "cash_sleeve": config.cash_ticker,
        "volatility_budget_annualized": config.volatility_budget_annualized,
        "no_trade_band": config.no_trade_band,
        "initial_capital_usd": config.initial_capital_usd,
        "half_spread_bps": costs.half_spread_bps,
        "commission_bps": costs.commission_bps,
        "impact_coefficient_bps": costs.impact_coefficient_bps,
        "participation_limit": costs.participation_limit,
        "candidates": [c.name for c in allocation.DECLARED_CANDIDATES],
        "sealed_block_state_coverage_status": coverage.status,
        "sealed_block_state_coverage_reason": coverage.reason,
        "limitations": list(allocation.ALLOCATION_LIMITATIONS),
    }
    (output_dir / "etf_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    print("=== what this run is, and is not ===")
    print(
        "Two out-of-sample claims, never merged. The stitched rolling record is "
        "out of sample with respect to the SELECTOR only. The sealed block is "
        "out of sample with respect to everything sealed at registration, and "
        "it is one look."
    )
    print(f"universe: {', '.join(inputs.tickers)}  (cash sleeve {config.cash_ticker})")
    print(
        f"panel {inputs.panel.first_session.date()}..{inputs.panel.last_session.date()}, "
        f"{len(inputs.panel.sessions)} sessions; "
        f"full fund histories from {inputs.history_close.index.min().date()}"
    )

    _print(
        "out-of-sample accounting, checkable by arithmetic",
        accounting,
        [
            "Claim",
            "Basis",
            "First session",
            "Last session",
            "Complete calendar years",
            "Calendar years touched",
            "Distinct sessions",
            "Elapsed years (365.2425)",
            "Years at 252 sessions",
            "Folds",
            "Sessions double counted",
        ],
    )
    print(
        "  the five-year requirement is answered twice: "
        f"{int(accounting.iloc[-1]['Complete calendar years'])} complete calendar "
        f"years of causal out-of-sample record, of which "
        f"{int(accounting.iloc[2]['Complete calendar years'])} contiguous years "
        "are also custody-sealed."
    )

    _print(
        "fold ledger",
        result.folds,
        [
            "fold",
            "test_start",
            "test_end",
            "sessions",
            "universe_members",
            "selection_sessions",
            "selection_score_annualized_sharpe",
            "selection_standard_error_of_annualized_sharpe",
            "selected_candidate",
            "selection_switched",
            "fold_cagr",
            "fold_sharpe",
            "fold_max_drawdown",
            "fold_annual_one_way_turnover",
            "oos_basis",
        ],
    )
    _print(
        "per-fold dispersion",
        dispersion,
        [
            "Statistic",
            "Folds",
            "Mean",
            "Median",
            "Standard deviation",
            "Minimum",
            "Maximum",
            "Folds above zero",
        ],
    )

    print("\n=== in sample against out of sample ===")
    # The frame mixes the window-descriptor rows, whose values are strings, with
    # the statistic rows, whose values are floats, so ``round`` would leave the
    # floats at full width.  Format per cell instead.
    printable = degradation.drop(columns=["Limitation"]).map(
        lambda value: f"{value:.4f}" if isinstance(value, float) else value
    )
    print(printable.to_string(index=False))

    comparison_columns = [
        "Path",
        "Sessions",
        "Calendar years touched",
        "Complete calendar years",
        "CAGR",
        "Annualized volatility",
        "Sharpe (rf=0)",
        "HAC Sharpe",
        "Sortino",
        "Max drawdown",
        "Calmar",
        "Annual one-way turnover",
        "Annual cost drag",
        "Mean risk weight",
    ]
    _print(
        "rolling out-of-sample window, against doing nothing",
        rolling,
        comparison_columns,
    )
    _print("sealed block, one look", sealed, comparison_columns)
    _print(
        "full replay, INCLUDES the in-sample training span",
        full_sample,
        comparison_columns,
    )
    _print(
        "the only equity bear market in the panel, STRADDLING the first boundary",
        crisis,
        ["Path", "Basis", "Start", "End", *comparison_columns[1:]],
    )
    _print(
        "risk matched to the sleeve's own realized volatility",
        risk_matched,
        [
            "Path",
            "Realized volatility",
            "Exposure multiplier applied",
            "CAGR at matched volatility",
            "Max drawdown at matched volatility",
            "Calmar at matched volatility",
            "Drawdown per unit of volatility",
        ],
    )
    _print(
        "does the declared volatility budget bind?",
        budget,
        [
            "Path",
            "Volatility budget (annualized)",
            "Realized volatility (annualized)",
            "Realized over budget",
            "Mean volatility scale",
            "Share of sessions at the scale ceiling",
            "Mean risk weight",
            "Share of sessions with a deferred order",
            "Minimum participation fill fraction",
        ],
    )
    _print(
        "concentration by construction: weight share against risk share",
        exposure.loc[~exposure["Asset class"].str.startswith("sleeve")],
        ["Asset class", "Sleeves", "Mean weight share", "Maximum weight share", "Realized risk share"],
    )
    _print(
        "paired differential against each reference, out of sample",
        paired,
        [
            "Comparison",
            "Sessions",
            "Annualized mean differential",
            "Standard error (annualized)",
            "t statistic",
            "Confidence lower (95% one-sided)",
            "Confidence upper (95% one-sided)",
            "Tracking error (annualized)",
            "Return correlation",
            "Minimum detectable Sharpe effect",
        ],
    )
    _print(
        "cost sensitivity: one strategy priced under five assumptions",
        sweep,
        [
            "Cost multiplier",
            "One-way fixed cost (bps)",
            "Out-of-sample CAGR",
            "Out-of-sample Sharpe (rf=0)",
            "Out-of-sample HAC Sharpe",
            "Out-of-sample max drawdown",
            "Annual one-way turnover",
            "Annual cost drag",
            "Sealed-block Sharpe (rf=0)",
            "Mean sealed fold Sharpe (rf=0)",
            "Selection switches",
        ],
    )
    print(
        "  the two sealed columns are different objects: the first is the Sharpe "
        "of the sealed block's own return path, the second is the mean of the "
        "annual fold Sharpes, and the second is not the block statistic."
    )

    print("\n=== custody of the sealed block ===")
    print(custody.drop(columns=["Limitation"]).T.to_string(header=False))

    print("\n=== what the sealed block cannot test ===")
    print(f"  status: {coverage.status}")
    print(f"  reason: {coverage.reason}")

    print("\n=== limitations carried by every artifact ===")
    for limitation in allocation.ALLOCATION_LIMITATIONS:
        print(f"  - {limitation}")
    print(
        "  every report frame stamps all four on every row; the two raw series "
        "dumps carry them through etf_run_manifest.json in the same directory, "
        "following the convention of outputs/strategy_daily.csv."
    )

    print(f"\nartifacts written to {output_dir}")


if __name__ == "__main__":
    main()
