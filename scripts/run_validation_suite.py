#!/usr/bin/env python3
"""Run the family-wise validation estimators over a pre-registered family.

``docs/research-methodology.md`` names an anchored expanding walk-forward, a
family-wise max-statistic procedure and CSCV/PBO as required, and until now
none of them existed in code.  This script executes all three on the real
1990-2014 history and writes the results to ``outputs/validation`` only.

What the script refuses to do.

It does not select a configuration.  Every artifact it writes is descriptive:
no row carries a verdict, a ranking or a recommendation, and the rows are
emitted in the order the family was declared rather than in order of
performance.  Selecting on these numbers is the retrospective fitting the
methodology document prohibits, and the estimators here exist to measure how
badly such a selection would generalise, not to perform one.

It does not extend the family after seeing a result.  ``DECLARED_FAMILY``
below is the manifest, it is written into the artifacts, and the assembler
raises if a path arrives that the manifest does not contain.  A max statistic
over a family that grew after the fact is not a max statistic.

It does not write outside its own subdirectory and never calls
``strategy.save_outputs``, which would overwrite the frozen canonical bundle
the rest of the repository is checked against.

It also reports the estimators' refusals rather than routing around them.  The
existing four-variant lever sweep is deliberately fed to CSCV so the artifact
records what a family too small and too coarse to support PBO looks like, and
the deflated Sharpe is written out as NOT ESTIMABLE - which it is, because the
declared family is a lower bound on this lineage's search rather than its trial
count - with the illustrative arithmetic under a separate name.

It reports PBO on the MONTHLY matrix, which is the only input the methodology
document permits, and emits the daily estimate beside it as a labelled
secondary rather than instead of it.  The two disagree materially on this
family and the disagreement is part of the result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from delta1_strategy.research import levers, validation
from delta1_strategy.research.strategy import (
    StrategyConfig,
    performance_metrics,
    run_backtest,
)
from delta1_strategy.research.validation import (
    RETROSPECTIVE_END,
    RETROSPECTIVE_START,
    WalkForwardCandidate,
    WalkForwardConfig,
)

BENCHMARK = "baseline"

# The family manifest.  Registered here, before any estimator runs, and written
# verbatim into the artifacts.  Every entry is a configuration a researcher
# might plausibly have tried on this history; that is what makes the
# multiplicity correction meaningful.
DECLARED_FAMILY: tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...] = (
    (BENCHMARK, "Canonical v3.2.1 configuration at a 7% risk budget.", ()),
    ("trend_126", "Six-month trend lookback.", (("trend_lookback", 126),)),
    ("trend_189", "Nine-month trend lookback.", (("trend_lookback", 189),)),
    ("trend_315", "Fifteen-month trend lookback.", (("trend_lookback", 315),)),
    ("basis_030", "Basis sleeve weighted 30%.", (("basis_weight", 0.30),)),
    ("basis_040", "Basis sleeve weighted 40%.", (("basis_weight", 0.40),)),
    ("basis_060", "Basis sleeve weighted 60%.", (("basis_weight", 0.60),)),
    ("basis_070", "Basis sleeve weighted 70%.", (("basis_weight", 0.70),)),
    ("buffer_015", "No-trade band narrowed to 15%.", (("no_trade_buffer", 0.15),)),
    ("buffer_020", "No-trade band narrowed to 20%.", (("no_trade_buffer", 0.20),)),
    ("buffer_030", "No-trade band widened to 30%.", (("no_trade_buffer", 0.30),)),
    ("risk_060", "Portfolio risk budget at 6.0%.", (("target_vol", 0.060),)),
    ("risk_065", "Portfolio risk budget at 6.5%.", (("target_vol", 0.065),)),
    ("risk_075", "Portfolio risk budget at 7.5%.", (("target_vol", 0.075),)),
    ("risk_080", "Portfolio risk budget at 8.0%.", (("target_vol", 0.080),)),
    ("volspan_040", "Faster 40-session sizing volatility.", (("vol_span", 40),)),
    ("volspan_090", "Slower 90-session sizing volatility.", (("vol_span", 90),)),
)

# The selection dimension the walk-forward re-fits inside each training span.
# One dimension, four levels, declared before the first fold runs.
WALK_FORWARD_CANDIDATES = ("baseline", "trend_126", "trend_189", "trend_315")


def _monthly(daily: pd.Series) -> pd.Series:
    """Compound inside the month, matching ``levers`` and ``diagnostics``."""

    return (1.0 + daily).resample("ME").prod() - 1.0


def _lever_family(levers_dir: Path) -> validation.StrategyFamily | None:
    """The existing four-variant lever sweep, if it has been run.

    Included so the artifacts record a real refusal rather than only a
    constructed one: four monthly paths cannot support a PBO, and the estimator
    should say so in the output the reader is looking at.
    """

    paths_file = levers_dir / "lever_monthly_net_returns.csv"
    manifest_file = levers_dir / "lever_variants.json"
    if not paths_file.exists() or not manifest_file.exists():
        return None
    frame = pd.read_csv(paths_file, index_col="month", parse_dates=True)
    declared = [
        str(entry["name"])
        for entry in json.loads(manifest_file.read_text(encoding="utf-8"))
    ]
    return validation.assemble_family(
        frame.dropna(how="any"),
        label="lever_sweep_risk_budget",
        benchmark=BENCHMARK,
        declared_members=declared,
        provenance=f"outputs/levers/{paths_file.name}",
        frequency=validation.MONTHLY_FREQUENCY,
        annualization=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/run_manifest.json"))
    parser.add_argument("--levers-dir", type=Path, default=Path("outputs/levers"))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--submatrices", type=int, default=validation.DEFAULT_SUBMATRICES)
    parser.add_argument("--seed", type=int, default=validation.DEFAULT_PAIRED_SEED)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    frames = levers.load_shared_frames(args.data_dir)
    base_config = StrategyConfig(data_dir=args.data_dir, output_dir=output_dir)

    # ---- 1. Run the declared family once, and prove the baseline is real. ---
    runs: dict[str, Any] = {}
    configs: dict[str, StrategyConfig] = {}
    manifest_rows: list[dict[str, Any]] = []
    for name, assumption, overrides in DECLARED_FAMILY:
        candidate = WalkForwardCandidate(name=name, assumption=assumption, overrides=overrides)
        config = candidate.apply(base_config)
        configs[name] = config
        result = run_backtest(config, **frames)
        runs[name] = result
        if name == BENCHMARK:
            fingerprint = levers.assert_baseline_fingerprint(result, args.manifest)
            print(f"baseline fingerprint verified {fingerprint[:16]}...")
        metrics = performance_metrics(
            result, RETROSPECTIVE_START, RETROSPECTIVE_END, config.annualization
        )
        manifest_rows.append(
            {
                "variant": name,
                "assumption": assumption,
                "overrides": json.dumps(dict(overrides), sort_keys=True),
                "cagr": float(metrics["CAGR"]),
                "annualized_volatility": float(metrics["Annualized volatility"]),
                "sharpe": float(metrics["Naive daily Sharpe (sqrt252, rf=0)"]),
                "max_drawdown": float(metrics["Max drawdown"]),
                "annual_cost_drag": float(metrics["Annual cost drag"]),
                "validation_status": "declared_family_member_on_reused_history",
                "selection_adjusted": False,
            }
        )
        print(
            f"  {name:<12} CAGR {float(metrics['CAGR']):+.4%}  "
            f"Sharpe {float(metrics['Naive daily Sharpe (sqrt252, rf=0)']):.4f}"
        )
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "validation_family_manifest.csv", index=False
    )

    daily_paths = {
        name: result.daily.loc[RETROSPECTIVE_START:RETROSPECTIVE_END, "net_return"]
        for name, result in runs.items()
    }
    declared_names = tuple(name for name, _, _ in DECLARED_FAMILY)
    family = validation.assemble_family(
        daily_paths,
        label="delta1_configuration_family_1990_2014",
        benchmark=BENCHMARK,
        declared_members=declared_names,
        provenance="run_backtest over the declared configuration manifest",
        frequency=validation.DAILY_FREQUENCY,
        annualization=base_config.annualization,
    )
    monthly = pd.DataFrame({name: _monthly(path) for name, path in daily_paths.items()})
    monthly.to_csv(
        output_dir / "validation_family_monthly_net_returns.csv", index_label="month"
    )
    monthly_family = validation.assemble_family(
        monthly.dropna(how="any"),
        label="delta1_configuration_family_1990_2014_monthly",
        benchmark=BENCHMARK,
        declared_members=declared_names,
        provenance="monthly compounding of the declared configuration manifest",
        frequency=validation.MONTHLY_FREQUENCY,
        annualization=12,
    )

    # ---- 2. Family-wise max statistics. -------------------------------------
    family_wise = validation.family_wise_report(
        family, samples=args.samples, seed=args.seed
    )
    family_wise.to_csv(output_dir / "validation_family_wise.csv", index=False)

    deflated = pd.DataFrame(
        [validation.family_deflated_sharpe(family, member=name) for name in declared_names]
    )
    deflated.to_csv(output_dir / "validation_deflated_sharpe.csv", index=False)

    # ---- 3. CSCV, on the declared family and on the lever sweep. ------------
    # research-methodology.md permits CSCV/PBO "only as a secondary
    # selection-instability diagnostic on the complete matrix of synchronized,
    # fully causal MONTHLY paths for every registered variant".  The monthly
    # matrix is therefore the primary artifact, and the daily one is emitted
    # beside it because the two disagree materially and hiding the disagreement
    # would be the more comfortable of the two options.  A daily submatrix
    # spans the same calendar window as a monthly one at the same S, so the gap
    # is not a window-length artefact: it is what happens when the ordering of
    # seventeen highly correlated configurations is ranked on 407 noisy daily
    # observations rather than on 18 monthly ones.
    cscv = validation.cscv_pbo(monthly_family, submatrices=args.submatrices)
    cscv.to_frame().to_csv(output_dir / "validation_cscv.csv", index=False)
    cscv.splits.to_csv(output_dir / "validation_cscv_splits.csv", index=False)
    cscv.diagnostics.to_frame("value").to_csv(
        output_dir / "validation_cscv_diagnostics.csv", index_label="field"
    )

    cscv_daily = validation.cscv_pbo(family, submatrices=args.submatrices)
    cscv_daily.to_frame().to_csv(
        output_dir / "validation_cscv_daily_secondary.csv", index=False
    )
    cscv_daily.diagnostics.to_frame("value").to_csv(
        output_dir / "validation_cscv_daily_secondary_diagnostics.csv",
        index_label="field",
    )

    lever_family = _lever_family(args.levers_dir)
    if lever_family is not None:
        lever_cscv = validation.cscv_pbo(lever_family, submatrices=args.submatrices)
        lever_cscv.to_frame().to_csv(
            output_dir / "validation_cscv_lever_sweep.csv", index=False
        )
        validation.family_wise_report(
            lever_family, samples=args.samples, seed=args.seed
        ).to_csv(output_dir / "validation_family_wise_lever_sweep.csv", index=False)

    # ---- 4. Anchored expanding walk-forward, both modes. --------------------
    boundaries = validation.calendar_boundaries("1994-12-31", RETROSPECTIVE_END)
    schedule = WalkForwardConfig(
        boundaries=boundaries,
        anchor=RETROSPECTIVE_START,
        end=RETROSPECTIVE_END,
        annualization=base_config.annualization,
    )

    def cached_runner(candidate_config: StrategyConfig) -> Any:
        for name, config in configs.items():
            if config == candidate_config:
                return runs[name]
        raise KeyError("the walk-forward asked for an unregistered configuration")

    candidates = tuple(
        WalkForwardCandidate(
            name=name,
            assumption=dict((entry[0], entry[1]) for entry in DECLARED_FAMILY)[name],
            overrides=dict((entry[0], entry[2]) for entry in DECLARED_FAMILY)[name],
        )
        for name in WALK_FORWARD_CANDIDATES
    )

    selecting = validation.anchored_walk_forward(
        base_config,
        frames=frames,
        walk_forward=schedule,
        candidates=candidates,
        label="trend_lookback_selection",
        run_fn=cached_runner,
    )
    selecting.folds.to_csv(
        output_dir / "validation_walk_forward_folds.csv", index=False
    )
    _monthly(selecting.stitched_path).to_csv(
        output_dir / "validation_walk_forward_monthly_path.csv",
        index_label="month",
        header=["net_return"],
    )

    stability = validation.anchored_walk_forward(
        base_config,
        frames=frames,
        walk_forward=schedule,
        label="frozen_specification",
        run_fn=cached_runner,
    )
    stability.folds.to_csv(
        output_dir / "validation_stability_folds.csv", index=False
    )
    pd.DataFrame([selecting.summary, stability.summary]).to_csv(
        output_dir / "validation_walk_forward_summary.csv", index=False
    )

    # ---- 5. Provenance sidecar. ---------------------------------------------
    sidecar = {
        "window": [RETROSPECTIVE_START, RETROSPECTIVE_END],
        "declared_family": [
            {"name": name, "assumption": assumption, "overrides": dict(overrides)}
            for name, assumption, overrides in DECLARED_FAMILY
        ],
        "family_size": family.size,
        "family_completeness": family.completeness,
        "observations": family.observations,
        "monthly_observations": monthly_family.observations,
        "cscv_primary_frequency": validation.MONTHLY_FREQUENCY,
        "cscv_primary_reason": (
            "research-methodology.md permits CSCV/PBO only on the complete "
            "matrix of synchronized monthly paths; the daily estimate is "
            "emitted as a labelled secondary because the two disagree"
        ),
        "bootstrap_samples": int(args.samples),
        "p_value_resolution": 1.0 / (args.samples + 1.0),
        "block_lengths": list(validation.DEFAULT_BLOCK_LENGTHS),
        "seed": int(args.seed),
        "submatrices": int(args.submatrices),
        "walk_forward_boundaries": list(boundaries),
        "walk_forward_candidates": list(WALK_FORWARD_CANDIDATES),
        "selection_adjusted_reports": ["validation_family_wise.csv", "validation_cscv.csv"],
        "limitations": [
            "every path is the reused 1990-2014 history; nothing here is out of sample",
            "the multiplicity correction covers the declared family only",
            "the walk-forward selector is naive by design and is not a proposal",
            (
                "the annualized-mean differential is not risk matched: a "
                "configuration that simply takes more risk raises it "
                "mechanically, so the Sharpe differential is the statistic to "
                "read for an improvement claim"
            ),
            (
                "the deflated Sharpe is NOT ESTIMABLE for this lineage and is "
                f"reported as such: the declared family of {len(DECLARED_FAMILY)} "
                "is a lower bound on the search, not the unrecoverable trial "
                "count, so the probability computed from it cannot support the "
                "promotion gate.  The illustrative lower bound is emitted "
                "beside the Gumbel threshold it deflates by, which is far "
                "below every member, so it does not discriminate inside the "
                "family either"
            ),
            (
                "PBO is reported primarily on the monthly matrix because that "
                "is the only input research-methodology.md permits; the daily "
                "estimate is a labelled secondary and the two disagree"
            ),
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output_dir / "validation_run.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- 6. One-screen summary. ---------------------------------------------
    headline = family_wise.loc[family_wise["Expected block sessions"] == 21]
    print("\nfamily-wise max statistics at a 21-session expected block")
    print(
        headline[["Statistic", "Procedure", "Status", "p value"]].to_string(index=False)
    )
    print("\nCSCV on the monthly matrix (the frequency the methodology permits)")
    print(
        cscv.to_frame()[["Statistic", "Status", "Value", "Reason"]].to_string(index=False)
    )
    print("\nCSCV on the daily matrix (secondary; not the permitted input)")
    print(
        cscv_daily.to_frame()[["Statistic", "Status", "Value"]].to_string(index=False)
    )
    if lever_family is not None:
        print("\nCSCV on the existing four-variant lever sweep")
        print(
            lever_cscv.to_frame()[["Statistic", "Status", "Value"]].to_string(index=False)
        )
    print("\nwalk-forward")
    for report in (selecting, stability):
        summary = report.summary
        print(
            f"  {summary['label']:<28} {summary['mode']:<34} "
            f"stitched Sharpe {float(summary['stitched_sharpe']):.4f}  "
            f"CAGR {float(summary['stitched_cagr']):+.4%}  "
            f"efficiency {float(summary['walk_forward_efficiency']):.4f}  "
            f"switches {int(summary['selection_switches'])}"
        )
    print(f"\nwrote {len(list(output_dir.glob('validation_*')))} artifacts to {output_dir}")


if __name__ == "__main__":
    main()
