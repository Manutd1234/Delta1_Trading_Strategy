#!/usr/bin/env python3
"""Diagnose the incumbent's drawdowns before any drawdown lever is swept.

This script exists because of the order in which two questions must be asked.
"Does tightening a position bound reduce drawdown?" cannot be answered by a
sweep until "is that bound an active constraint, and is the loss it targets a
size failure?" has been answered first.  Run this before
``scripts/run_bounds_sweep.py`` and the sweep's numbers become interpretable;
run it after, and the sweep will have reported a real effect for an inert
parameter without either of them noticing.

It writes only into its own output directory.  The canonical bundle and the
frozen run manifest are never touched, and ``save_outputs`` is never called.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from delta1_strategy.research import levers
from delta1_strategy.research.attribution import (
    bound_activity_report,
    conditional_exposure_diagnostic,
    drawdown_anatomy,
    drawdown_episodes,
    episode_attribution,
    magnitude_bound_specs,
    trailing_mean_pairwise_correlation,
    uniform_rescale_frontier,
)
from delta1_strategy.research.strategy import (
    StrategyConfig,
    market_daily_ledger,
    run_backtest,
)


RETROSPECTIVE_START = "1990-01-01"
RETROSPECTIVE_END = "2014-12-31"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attribution"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/run_manifest.json")
    )
    parser.add_argument("--start", default=RETROSPECTIVE_START)
    parser.add_argument("--end", default=RETROSPECTIVE_END)
    parser.add_argument(
        "--drawdown-threshold",
        type=float,
        default=0.05,
        help="depth below which a session counts as 'in drawdown'",
    )
    parser.add_argument("--correlation-window", type=int, default=63)
    parser.add_argument("--correlation-quantile", type=float, default=0.8)
    parser.add_argument("--correlation-multiplier", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = levers.load_shared_frames(args.data_dir)
    config = StrategyConfig(data_dir=args.data_dir, output_dir=output_dir)
    result = run_backtest(config, **frames)

    # The same guard the lever sweep uses.  An attribution computed on a
    # mis-wired baseline would describe a strategy this repository does not run.
    if args.manifest.exists():
        fingerprint = levers.assert_baseline_fingerprint(result, args.manifest)
        print(f"baseline reconciles with the frozen manifest: {fingerprint}")
    else:
        print(f"manifest {args.manifest} absent; baseline fingerprint NOT verified")

    daily = result.daily.loc[args.start : args.end]
    returns = daily["net_return"].dropna()
    first_location = result.daily.index.get_indexer([returns.index[0]])[0]
    period_start = (
        result.daily.index[first_location - 1]
        if first_location > 0
        else returns.index[0]
    )

    activity = bound_activity_report(
        magnitude_bound_specs(result, config, start=args.start, end=args.end)
    )
    anatomy = drawdown_anatomy(
        daily,
        threshold=args.drawdown_threshold,
        annualization=config.annualization,
    )
    episodes = drawdown_episodes(returns, minimum_depth=args.drawdown_threshold)
    ledger = market_daily_ledger(result)
    summary, detail = episode_attribution(ledger, episodes)
    frontier = uniform_rescale_frontier(
        returns,
        annualization=config.annualization,
        period_start=period_start,
    )

    # The conditioner follows from the attribution rather than preceding it: the
    # episodes are broad, so cross-market agreement is the state the failure mode
    # is indexed by.  It is reported as a hypothesis, and the artifact says so on
    # every row.
    contributions = ledger.assign(date=pd.to_datetime(ledger["date"])).pivot_table(
        index="date", columns="symbol", values="net_return_contribution", aggfunc="sum"
    ).loc[args.start : args.end]
    correlation = trailing_mean_pairwise_correlation(
        contributions, window=args.correlation_window
    )
    conditional = conditional_exposure_diagnostic(
        returns,
        correlation,
        quantile=args.correlation_quantile,
        multiplier=args.correlation_multiplier,
        annualization=config.annualization,
        label=f"trailing {args.correlation_window}-session mean pairwise P&L correlation",
        period_start=period_start,
    )

    artifacts = {
        "attribution_bound_activity.csv": activity,
        "attribution_drawdown_anatomy.csv": anatomy,
        "attribution_drawdown_episodes.csv": episodes,
        "attribution_episode_summary.csv": summary,
        "attribution_episode_market_detail.csv": detail,
        "attribution_uniform_rescale_frontier.csv": frontier,
        "attribution_conditional_exposure.csv": conditional,
        "attribution_correlation_conditioner.csv": correlation.rename(
            "trailing_mean_pairwise_correlation"
        ).to_frame(),
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(f"\n=== bound activity, {args.start} to {args.end} ===")
    print(
        activity[
            [
                "Bound",
                "Side",
                "Limit",
                "Binding share",
                "Realized minimum",
                "Realized maximum",
                "Headroom to limit",
                "Activity",
            ]
        ].to_string(index=False)
    )
    print("\n=== drawdown anatomy: size failure or accuracy failure? ===")
    print(
        anatomy[
            [
                "Sample",
                "Sessions",
                "Annualized return",
                "Annualized volatility",
                "Daily hit rate",
                "Mean gross notional multiple",
                "Volatility ratio to all sessions",
                "Exposure ratio to all sessions",
            ]
        ].round(4).to_string(index=False)
    )
    print("\n=== worst episodes ===")
    print(episodes.head(8).to_string(index=False))
    print("\n=== breadth of loss inside each episode ===")
    if not summary.empty:
        print(summary.round(4).to_string(index=False))
    print("\n=== what a uniform de-lever already buys ===")
    print(frontier.round(4).to_string(index=False))
    print("\n=== conditioning exposure on cross-market agreement (HYPOTHESIS ONLY) ===")
    print(
        conditional[
            [
                "Configuration",
                "CAGR",
                "Annualized volatility",
                "Sharpe (rf=0)",
                "Max drawdown",
                "Calmar",
                "Drawdown per unit of volatility",
                "Sessions de-levered",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"    {conditional.iloc[-1]['Evidence note']}")
    print(f"\nartifacts written to {output_dir}")


if __name__ == "__main__":
    main()
