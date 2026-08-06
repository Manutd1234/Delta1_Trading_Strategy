#!/usr/bin/env python3
"""Replicate published trend-following rules on this panel and compare them.

The repository has no strategy benchmark, so "is a 1.59 Sharpe good?" is
currently unanswerable from inside it.  This script answers it by rebuilding
the academic rules on the identical 59-market panel and running them through
the identical execution ledger, then regressing the incumbent on them.

What this script refuses to do
------------------------------
It does not download SG Trend, BTOP50 or any other external index.  Those need
a licence and a network, and a published index carries a different universe,
cost base and management/performance fee load, so the comparison would not be
like-for-like even if it were available.  That remains a disclosed gap.

It does not rank the benchmarks, score them, or emit a recommendation.  Every
row is measured on the same reused 1990-2014 history the incumbent was
constructed against, and ``docs/research-methodology.md`` prohibits selecting a
configuration on reused history regardless of how the numbers come out.

It does not call ``strategy.save_outputs``, and it writes only inside
``outputs/benchmarks``.  The frozen canonical bundle in ``outputs/`` is left
untouched, and the incumbent run is checked against the frozen daily
fingerprint before any benchmark is allowed to run: a partially wired frame set
silently disables roll costs and would flatter every row by roughly the size of
the promotion gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from delta1_strategy.research import benchmarks, levers
from delta1_strategy.research.strategy import StrategyConfig, run_backtest


ARTIFACTS = (
    "seam_check",
    "comparison",
    "signal_diagnostics",
    "alpha",
    "spanning",
    "spanning_coefficients",
    "overlay_weights",
    "hop_scaling_diagnostics",
    "permutation_null",
    "permutation_draws",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/run_manifest.json"))
    parser.add_argument("--start", default=benchmarks.RETROSPECTIVE_START)
    parser.add_argument("--end", default=benchmarks.RETROSPECTIVE_END)
    parser.add_argument(
        "--permutations",
        type=int,
        default=benchmarks.DEFAULT_PERMUTATIONS,
        help=(
            "sign permutations per null construction. The reported p-value can "
            "never be finer than 1/(B+1), so a small B is reported as such "
            "rather than quoted at a resolution it does not have."
        ),
    )
    parser.add_argument("--seed", type=int, default=benchmarks.DEFAULT_PERMUTATION_SEED)
    parser.add_argument(
        "--skip-permutation-null",
        action="store_true",
        help="omit the sign-permutation null, which dominates the runtime",
    )
    arguments = parser.parse_args()

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = levers.load_shared_frames(arguments.data_dir)
    config = StrategyConfig(data_dir=arguments.data_dir, output_dir=output_dir)
    incumbent = run_backtest(config, **frames)
    fingerprint = levers.assert_baseline_fingerprint(incumbent, arguments.manifest)

    results = benchmarks.run_benchmark_suite(
        config,
        frames,
        incumbent_result=incumbent,
        start=arguments.start,
        end=arguments.end,
        permutations=arguments.permutations,
        seed=arguments.seed,
        include_permutation_null=not arguments.skip_permutation_null,
        progress=lambda message: print(f"  {message}", flush=True),
    )

    for name in ARTIFACTS:
        frame = results[name]
        frame.to_csv(output_dir / f"benchmark_{name}.csv", index=False)
    results["daily_net_returns"].to_csv(
        output_dir / "benchmark_daily_net_returns.csv", index_label="session"
    )
    results["monthly_net_returns"].to_csv(
        output_dir / "benchmark_monthly_net_returns.csv", index_label="month"
    )

    comparison = results["comparison"]
    spanning = results["spanning"]
    summary = {
        "window": {"start": arguments.start, "end": arguments.end},
        "incumbent_daily_fingerprint_sha256": fingerprint,
        "execution_seam_max_abs_net_return_deviation": float(
            results["seam_check"]["observed"].iloc[0]
        ),
        "benchmarks_declared": [
            specification.key for specification in benchmarks.BENCHMARK_FAMILY
        ],
        "benchmarks_completed": sorted(
            comparison.loc[
                comparison["run_status"] == benchmarks.RUN_COMPLETED, "benchmark"
            ]
        ),
        "benchmarks_errored": {
            str(row["benchmark"]): str(row["run_detail"])
            for _, row in comparison.iterrows()
            if row["run_status"] == benchmarks.RUN_ERRORED
        },
        "spanning_family": list(benchmarks.spanning_family_keys()),
        "spanning_alpha_annualized": (
            float(spanning["alpha_annualized"].iloc[0]) if len(spanning) else None
        ),
        "spanning_alpha_hac_t_statistic": (
            float(spanning["alpha_hac_t_statistic"].iloc[0]) if len(spanning) else None
        ),
        "permutations": int(arguments.permutations),
        "permutation_seed": int(arguments.seed),
        "selection_adjusted": False,
        "permitted_use": (
            "descriptive replication of published rules on reused history; no "
            "row supports promoting or demoting a configuration"
        ),
        "disclosed_gaps": [
            "no licensed external index (SG Trend, BTOP50) is used; a published "
            "index carries a different universe, cost base and fee load",
            "the benchmarks are run cold on this panel while the incumbent's "
            "parameters were chosen with this panel visible, so any alpha "
            "against them is an upper bound rather than an estimate",
            "results are gross of management and performance fees on both sides "
            "and net only of this repository's own cost model",
        ],
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(
            comparison[
                [
                    "benchmark",
                    "run_status",
                    "cagr",
                    "annualized_volatility",
                    "sharpe",
                    "max_drawdown",
                    "annual_cost_drag",
                ]
            ].to_string(index=False)
        )
    print(
        f"wrote {len(ARTIFACTS) + 2} artifacts to {output_dir}; "
        f"spanning alpha {summary['spanning_alpha_annualized']} "
        f"(t={summary['spanning_alpha_hac_t_statistic']})"
    )


if __name__ == "__main__":
    main()
