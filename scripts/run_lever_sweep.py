#!/usr/bin/env python3
"""Run the DELTA1 lever comparison and write its artifacts.

The sweep never calls ``save_outputs``.  It writes only into its own output
directory, so the canonical bundle and the frozen run manifest are untouched
while levers are being measured.

Every configuration here is reused 1990-2014 history.  The report ranks
nothing and carries no verdict column: choosing a configuration on these
numbers is exactly the retrospective fitting the research methodology
prohibits.  The purpose is to measure what a change would do, and to attach an
honest uncertainty to that measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research import collateral as collateral_module
from delta1_strategy.research import inference
from delta1_strategy.research import levers
from delta1_strategy.research.strategy import StrategyConfig


def volatility_frontier_variants(
    targets: tuple[float, ...],
) -> tuple[levers.LeverVariant, ...]:
    """Baseline plus one variant per candidate risk budget.

    A risk-budget change is a capital-policy decision, not an alpha result, so
    these rows exist to price the decision rather than to argue for it.
    """

    variants = [
        levers.LeverVariant(
            name="baseline",
            lever="baseline",
            evaluation_mode="baseline",
            assumption="Canonical v3.2.1 configuration at a 7% risk budget.",
        )
    ]
    for target in targets:
        variants.append(
            levers.LeverVariant(
                name=f"risk_budget_{target:.4f}".rstrip("0").rstrip("."),
                lever="risk_budget_volatility",
                evaluation_mode="marginal",
                assumption=(
                    f"Portfolio risk budget raised to {target:.2%} with every "
                    "other control unchanged."
                ),
                overrides=(("target_vol", target),),
            )
        )
    return tuple(variants)


def bootstrap_breach_probability(
    returns: pd.Series,
    *,
    label: str,
    threshold: float = 0.15,
    horizon: int = 2_520,
    block_length: int = 63,
    samples: int = 2_000,
    seed: int = inference.DEFAULT_PAIRED_SEED,
) -> dict[str, float]:
    """Probability a ten-year path breaches the drawdown policy.

    The historical maximum drawdown is a single realized path and says little
    about how much drawdown risk a configuration actually carries.  Holding
    *this* number constant, rather than the historical one, is what makes
    "without sacrificing drawdown" a real constraint.
    """

    values = returns.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty return series")
    entropy = np.random.SeedSequence(
        [seed, abs(hash(label)) % (2**32), int(block_length), int(horizon)]
    )
    rng = np.random.default_rng(entropy)
    indices = inference._stationary_indices(
        rng, values.size, samples, horizon, block_length
    )
    paths = np.cumprod(1.0 + values[indices], axis=1)
    peaks = np.maximum.accumulate(np.maximum(paths, 1.0), axis=1)
    drawdowns = (paths / peaks - 1.0).min(axis=1)
    return {
        "bootstrap_p_drawdown_breach_15pct": float((drawdowns < -threshold).mean()),
        "bootstrap_p_drawdown_breach_20pct": float((drawdowns < -0.20).mean()),
        "bootstrap_p01_max_drawdown": float(np.percentile(drawdowns, 1)),
        "bootstrap_median_max_drawdown": float(np.median(drawdowns)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/levers"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/run_manifest.json")
    )
    parser.add_argument(
        "--risk-budgets",
        type=float,
        nargs="*",
        default=[0.075, 0.080, 0.085],
        help="candidate portfolio risk budgets to price",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--paired-samples", type=int, default=5_000)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = StrategyConfig(data_dir=args.data_dir, output_dir=output_dir)
    frames = levers.load_shared_frames(args.data_dir)
    variants = volatility_frontier_variants(tuple(args.risk_budgets))

    comparison, monthly = levers.run_lever_comparison(
        config,
        variants=variants,
        frames=frames,
        manifest_path=args.manifest,
    )

    # Drawdown risk, measured on the bootstrap rather than the single realized
    # path, plus the funded basis for every variant.  Both are computed from
    # the daily paths the sweep already produced.
    from delta1_strategy.research.strategy import run_backtest

    rate = collateral_module.load_financing_rate(args.data_dir)
    risk_rows: list[dict[str, object]] = []
    daily_paths: dict[str, pd.Series] = {}
    for variant in variants:
        if not variant.is_supported(config):
            continue
        result = run_backtest(variant.apply(config), **frames)
        daily = result.daily.loc[
            levers.RETROSPECTIVE_START : levers.RETROSPECTIVE_END
        ]
        returns = daily["net_return"].dropna()
        daily_paths[variant.name] = returns
        funded = collateral_module.funded_ledger(
            result.daily, rate, period_start=levers.RETROSPECTIVE_START
        )
        funded_report = collateral_module.funded_performance_report(funded).set_index(
            "Basis"
        )
        row: dict[str, object] = {
            "variant": variant.name,
            "lever": variant.lever,
            "validation_status": levers.VALIDATION_STATUS,
            "excess_cagr": float(
                funded_report.loc[collateral_module.EXCESS_BASIS_LABEL, "CAGR"]
            ),
            "funded_cagr": float(
                funded_report.loc[collateral_module.FUNDED_BASIS_LABEL, "CAGR"]
            ),
            "sharpe_excess_of_financing": float(
                funded_report.loc[
                    collateral_module.FUNDED_BASIS_LABEL,
                    "Sharpe excess of financing",
                ]
            ),
            "historical_max_drawdown": float(
                funded_report.loc[
                    collateral_module.EXCESS_BASIS_LABEL, "Max drawdown"
                ]
            ),
            "funded_historical_max_drawdown": float(
                funded_report.loc[
                    collateral_module.FUNDED_BASIS_LABEL, "Max drawdown"
                ]
            ),
        }
        row.update(
            bootstrap_breach_probability(
                returns, label=variant.name, samples=args.bootstrap_samples
            )
        )
        funded_risk = bootstrap_breach_probability(
            funded["funded_net_return"],
            label=f"{variant.name}_funded",
            samples=args.bootstrap_samples,
        )
        row["funded_bootstrap_p_drawdown_breach_15pct"] = funded_risk[
            "bootstrap_p_drawdown_breach_15pct"
        ]
        risk_rows.append(row)

    risk = pd.DataFrame(risk_rows)

    # Paired differentials against the baseline, so each variant's effect
    # carries a sampling interval instead of a bare point estimate.
    baseline_path = daily_paths[variants[0].name]
    paired_frames = []
    for variant in variants[1:]:
        path = daily_paths.get(variant.name)
        if path is None or not path.index.equals(baseline_path.index):
            continue
        paired_frames.append(
            inference.paired_block_bootstrap_differential(
                baseline_path,
                path,
                comparison=f"{variant.name}_vs_baseline",
                samples=args.paired_samples,
            )
        )
    paired = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame(columns=inference.PAIRED_REPORT_COLUMNS)
    )

    comparison.to_csv(output_dir / "lever_comparison.csv", index=False)
    monthly.to_csv(output_dir / "lever_monthly_net_returns.csv", index_label="month")
    risk.to_csv(output_dir / "lever_drawdown_risk.csv", index=False)
    paired.to_csv(output_dir / "lever_paired_differentials.csv", index=False)
    (output_dir / "lever_variants.json").write_text(
        json.dumps(
            [
                {
                    "name": variant.name,
                    "lever": variant.lever,
                    "evaluation_mode": variant.evaluation_mode,
                    "assumption": variant.assumption,
                    "overrides": dict(variant.overrides),
                }
                for variant in variants
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(comparison)} comparison rows to {output_dir}")


if __name__ == "__main__":
    main()
