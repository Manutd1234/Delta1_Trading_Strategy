#!/usr/bin/env python3
"""Run the DELTA1 position-size bound sweep and write its artifacts.

The sweep never calls ``save_outputs``.  It writes only into ``outputs/bounds``
so the canonical bundle and the frozen run manifest are untouched while the
bounds are being measured.

It refuses to start until the baseline reproduces the frozen run manifest's
daily ledger fingerprint.  That check is not a formality here: omitting a
single input frame disables every roll cost and moves the result by more than
any bound in this grid does, which would turn a wiring mistake into an
apparent risk improvement.

Every row is reused 1990-2014 history.  Nothing is ranked and no row carries a
verdict.  Each bound setting appears twice: once as-is, and once with the risk
budget rescaled until realised volatility returns to the incumbent's, because a
leverage ceiling lowers volatility and drawdown together and only the matched
row says anything the rescale does not already say.

Two extra artifacts exist so the matched rows can be read at all.
``bound_match_jitter.csv`` re-runs the unchanged incumbent configuration at the
edges of the match tolerance and reports how far each loss-shape statistic
moves for no reason but where the solver stopped.  ``bound_resolution.csv``
reports the paired-bootstrap minimum detectable effect for the same statistics.
A matched delta that clears neither floor is written as ``not_estimable``
rather than as a number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from delta1_strategy.research import bounds, levers
from delta1_strategy.research.strategy import StrategyConfig, run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bounds"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/run_manifest.json")
    )
    parser.add_argument("--max-risk-scalars", type=float, nargs="*", default=None)
    parser.add_argument("--min-risk-scalars", type=float, nargs="*", default=None)
    parser.add_argument("--signal-caps", type=float, nargs="*", default=None)
    parser.add_argument("--risk-managed-caps", type=float, nargs="*", default=None)
    parser.add_argument("--shock-floors", type=float, nargs="*", default=None)
    parser.add_argument(
        "--max-gross-notional-multiples", type=float, nargs="*", default=None
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-horizon-sessions", type=int, default=2_520)
    parser.add_argument("--bootstrap-block-sessions", type=int, default=63)
    parser.add_argument("--paired-samples", type=int, default=2_000)
    parser.add_argument("--resolution-samples", type=int, default=2_000)
    parser.add_argument("--match-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--max-solver-iterations", type=int, default=3)
    parser.add_argument(
        "--match-control-fractions",
        type=float,
        nargs="*",
        default=None,
        help="positions inside the match tolerance, in units of the "
        "tolerance, at which the unchanged incumbent is re-run to measure "
        "solver jitter; pass none of them to skip the control and lose the "
        "measured floor the matched rows are read against",
    )
    parser.add_argument(
        "--skip-risk-match",
        action="store_true",
        help="report only the as-is rows; the sweep then answers nothing that "
        "a change of risk budget would not also answer",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = bounds.BoundGrid()
    grid = bounds.BoundGrid(
        max_risk_scalars=(
            tuple(args.max_risk_scalars)
            if args.max_risk_scalars
            else defaults.max_risk_scalars
        ),
        min_risk_scalars=(
            tuple(args.min_risk_scalars)
            if args.min_risk_scalars
            else defaults.min_risk_scalars
        ),
        signal_caps=(
            tuple(args.signal_caps) if args.signal_caps else defaults.signal_caps
        ),
        risk_managed_caps=(
            tuple(args.risk_managed_caps)
            if args.risk_managed_caps
            else defaults.risk_managed_caps
        ),
        shock_floors=(
            tuple(args.shock_floors) if args.shock_floors else defaults.shock_floors
        ),
        max_gross_notional_multiples=(
            tuple(args.max_gross_notional_multiples)
            if args.max_gross_notional_multiples
            else defaults.max_gross_notional_multiples
        ),
    )

    config = StrategyConfig(data_dir=args.data_dir, output_dir=output_dir)
    frames = levers.load_shared_frames(args.data_dir)

    # The baseline runs first, on its own, and the sweep proceeds only if its
    # daily ledger reproduces the frozen manifest.  Holding the result lets
    # every later pass reuse it instead of paying for it again.
    baseline_result = run_backtest(config, **frames)
    fingerprint = levers.assert_baseline_fingerprint(baseline_result, args.manifest)

    def engine(variant_config: StrategyConfig) -> object:
        if variant_config == config:
            return baseline_result
        return run_backtest(variant_config, **frames)

    artifacts = bounds.run_bounds_sweep(
        config,
        grid=grid,
        frames=frames,
        run_fn=engine,
        match=bounds.RiskMatchConfig(
            tolerance=args.match_tolerance,
            max_iterations=args.max_solver_iterations,
            control_probe_fractions=(
                tuple(args.match_control_fractions)
                if args.match_control_fractions is not None
                else bounds.RiskMatchConfig().control_probe_fractions
            ),
        ),
        risk=bounds.DrawdownRiskConfig(
            horizon_sessions=args.bootstrap_horizon_sessions,
            block_length_sessions=args.bootstrap_block_sessions,
            samples=args.bootstrap_samples,
        ),
        resolution=bounds.LossShapeResolutionConfig(
            samples=args.resolution_samples
        ),
        paired_samples=args.paired_samples,
        include_risk_matched=not args.skip_risk_match,
    )

    artifacts.comparison.to_csv(output_dir / "bound_comparison.csv", index=False)
    artifacts.shape.to_csv(output_dir / "bound_shape.csv", index=False)
    artifacts.binding.to_csv(output_dir / "bound_binding.csv", index=False)
    artifacts.drawdown_risk.to_csv(output_dir / "bound_drawdown_risk.csv", index=False)
    artifacts.risk_match.to_csv(output_dir / "bound_risk_match.csv", index=False)
    artifacts.match_jitter.to_csv(output_dir / "bound_match_jitter.csv", index=False)
    artifacts.resolution.to_csv(output_dir / "bound_resolution.csv", index=False)
    artifacts.paired.to_csv(
        output_dir / "bound_paired_differentials.csv", index=False
    )
    artifacts.power.to_csv(output_dir / "bound_power.csv", index=False)
    artifacts.monthly_net_returns.to_csv(
        output_dir / "bound_monthly_net_returns.csv", index_label="month"
    )

    variants = bounds.build_bound_variants(config, grid)
    (output_dir / "bound_variants.json").write_text(
        json.dumps(
            {
                "baseline_daily_fingerprint_sha256": fingerprint,
                "window": [levers.RETROSPECTIVE_START, levers.RETROSPECTIVE_END],
                "validation_status": bounds.BOUNDS_VALIDATION_STATUS,
                "grid": {
                    "max_risk_scalars": list(grid.max_risk_scalars),
                    "min_risk_scalars": list(grid.min_risk_scalars),
                    "signal_caps": list(grid.signal_caps),
                    "risk_managed_caps": list(grid.risk_managed_caps),
                    "shock_floors": list(grid.shock_floors),
                    "max_gross_notional_multiples": list(
                        grid.max_gross_notional_multiples
                    ),
                },
                "declared_paired_comparisons": list(
                    bounds.declared_paired_comparisons(config, grid)
                ),
                "variants": [
                    {
                        "name": variant.name,
                        "lever": variant.lever,
                        "evaluation_mode": variant.evaluation_mode,
                        "assumption": variant.assumption,
                        "overrides": dict(variant.overrides),
                    }
                    for variant in variants
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ratio = "max_drawdown_over_annualized_volatility"
    jitter = artifacts.match_jitter.set_index("statistic")
    band = (
        float(jitter.loc[ratio, "solver_jitter_band"])
        if ratio in jitter.index
        else float("nan")
    )
    resolved = int(
        artifacts.shape[f"delta_{ratio}_status"].eq(bounds.ESTIMATED).sum()
    )
    print(
        f"wrote {len(artifacts.shape)} bound rows "
        f"({len(artifacts.risk_match)} risk-match solver steps) to {output_dir}; "
        f"solver jitter band on maxDD/vol {band:.4f}; "
        f"{resolved} of {len(artifacts.shape)} maxDD/vol deltas clear their "
        "resolution floor"
    )


if __name__ == "__main__":
    main()
