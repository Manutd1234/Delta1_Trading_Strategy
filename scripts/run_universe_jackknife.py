#!/usr/bin/env python3
"""Leave-one-sector-out replay of the frozen DELTA1 configuration.

The repository already asserts, in prose, that no single asset class carries
the result.  This script substantiates that sentence: it reruns the frozen
v3.2.1 configuration on the reused 1990-2014 history six times, each time with
one sector of the reference universe removed from the traded panel, and writes
what remains of the net result beside the zero-exclusion baseline.

What the script refuses to do.

It does not rank.  Rows are emitted in the manifest order declared below,
never in order of performance, and no row carries a verdict: choosing a
universe on these numbers would be the retrospective selection
``docs/research-methodology.md`` prohibits.

It does not extend or reorder the manifest after seeing a result.
``EXCLUSION_MANIFEST`` is declared before any file is read, and the script
raises if the manifest disagrees with the engine's own catalogue.

It does not emit anything until the zero-exclusion replay has reproduced both
the frozen run manifest's daily fingerprint and the published full-history
Sharpe.  A jackknife around a baseline the script cannot reproduce would be
measuring its own wiring, not the universe.

It does not write outside ``outputs/validation`` and never calls
``strategy.save_outputs``, which would overwrite the frozen canonical bundle.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from delta1_strategy.research import levers
from delta1_strategy.research.strategy import (
    StrategyConfig,
    performance_metrics,
    run_backtest,
    strategy_symbols,
)

# The exclusion manifest: the six sectors of the reference UNIVERSE dict
# (reference/delta1_reference.py), in its declaration order.  YXT and YYT do
# not appear because they sit outside the supported research book everywhere,
# so no exclusion can remove them.
EXCLUSION_MANIFEST: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Equity indices", ("EMD", "ES", "FCE", "FDAX", "FESX", "FSMI", "HTW", "LFT",
                        "NKD", "NQ", "RTY", "SNK", "SXF", "YAP", "YM")),
    ("Government bonds", ("CGB", "FGBL", "FGBM", "FGBS", "FGBX", "LLG", "SJB",
                          "ZT", "ZF", "ZN", "ZB")),
    ("FX", ("6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S")),
    ("Energy", ("BRN", "CL", "GAS", "HO", "NG", "RB")),
    ("Metals", ("GC", "HG", "PA", "PL", "SI")),
    ("Agriculture & livestock", ("CC", "CT", "GF", "HE", "KC", "KE", "LE", "RS",
                                 "SB", "ZC", "ZL", "ZM", "ZS", "ZW")),
)

BASELINE_LABEL = "none"

# outputs/strategy_metrics.csv, "1990-2014 full post-launch history",
# "Naive daily Sharpe (sqrt252, rf=0)".  The zero-exclusion replay must
# reproduce this figure before any jackknife row is written.
PUBLISHED_BASELINE_SHARPE = 1.5895266624546427

VALIDATION_STATUS = (
    "retrospective_reused_history; leave-one-class-out replay of the frozen "
    "specification; no selection"
)

REPORT_COLUMNS = [
    "excluded_class",
    "markets_remaining",
    "sharpe",
    "cagr",
    "annualized_volatility",
    "max_drawdown",
    "delta_sharpe_vs_baseline",
    "delta_cagr_vs_baseline",
    "validation_status",
]

# Frames whose columns are the traded markets.  ``metadata`` (indexed by
# symbol) and ``fx_rates`` (indexed by currency) are handled separately.
_MARKET_COLUMN_FRAMES = (
    "prices",
    "observed_opens",
    "observed_closes",
    "unadjusted",
    "volumes",
    "delivery_months",
)


def _check_manifest_against_catalogue() -> None:
    """Raise unless the declared manifest is exactly the engine's book.

    A manifest that silently drifted from ``strategy_symbols()`` would make
    ``markets_remaining`` a lie and could leave a market in no sector, where
    no exclusion ever touches it.
    """

    declared = [symbol for _, members in EXCLUSION_MANIFEST for symbol in members]
    if len(declared) != len(set(declared)):
        raise ValueError("a symbol appears in more than one manifest sector")
    if set(declared) != set(strategy_symbols()):
        raise ValueError(
            "EXCLUSION_MANIFEST does not partition the engine's research book"
        )


def _excluded_frames(
    frames: dict[str, pd.DataFrame], excluded: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    """The shared input frames with ``excluded`` markets removed.

    Column order is preserved so the only difference a variant can see is the
    membership itself.
    """

    removed = set(excluded)
    keep = [column for column in frames["prices"].columns if column not in removed]
    subset: dict[str, pd.DataFrame] = {
        key: frames[key].loc[:, keep] for key in _MARKET_COLUMN_FRAMES
    }
    subset["metadata"] = frames["metadata"].loc[frames["metadata"].index.isin(keep)]
    # FX valuation is independent of the traded panel: conversion rates come
    # from the currency-futures files, not from the traded book, so excluding
    # the FX sector removes its positions while every remaining
    # foreign-currency contract is still valued.  The full frame is passed
    # unconditionally.
    subset["fx_rates"] = frames["fx_rates"]
    return subset


def _replay_row(
    config: StrategyConfig,
    frames: dict[str, pd.DataFrame],
    excluded_class: str,
    excluded: tuple[str, ...],
) -> dict[str, object]:
    """Run one exclusion and report the frozen configuration's net result."""

    subset = _excluded_frames(frames, excluded)
    result = run_backtest(config, **subset)
    metrics = performance_metrics(
        result, levers.RETROSPECTIVE_START, levers.RETROSPECTIVE_END, config.annualization
    )
    return {
        "excluded_class": excluded_class,
        "markets_remaining": len(subset["prices"].columns),
        "sharpe": float(metrics["Naive daily Sharpe (sqrt252, rf=0)"]),
        "cagr": float(metrics["CAGR"]),
        "annualized_volatility": float(metrics["Annualized volatility"]),
        "max_drawdown": float(metrics["Max drawdown"]),
        "validation_status": VALIDATION_STATUS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/run_manifest.json")
    )
    args = parser.parse_args()

    _check_manifest_against_catalogue()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = StrategyConfig(data_dir=args.data_dir, output_dir=output_dir)
    frames = levers.load_shared_frames(args.data_dir)

    # The zero-exclusion replay, proven against the frozen bundle twice: the
    # daily fingerprint catches a wiring difference anywhere in the ledger,
    # and the published Sharpe pins the exact window and column this report
    # summarizes.
    baseline_result = run_backtest(config, **frames)
    fingerprint = levers.assert_baseline_fingerprint(baseline_result, args.manifest)
    print(f"baseline fingerprint verified {fingerprint[:16]}...")
    baseline_metrics = performance_metrics(
        baseline_result,
        levers.RETROSPECTIVE_START,
        levers.RETROSPECTIVE_END,
        config.annualization,
    )
    baseline_sharpe = float(baseline_metrics["Naive daily Sharpe (sqrt252, rf=0)"])
    if abs(baseline_sharpe - PUBLISHED_BASELINE_SHARPE) > 1e-12:
        raise ValueError(
            "zero-exclusion replay does not reproduce the published baseline "
            f"Sharpe: expected {PUBLISHED_BASELINE_SHARPE}, observed {baseline_sharpe}"
        )
    rows = [
        {
            "excluded_class": BASELINE_LABEL,
            "markets_remaining": len(frames["prices"].columns),
            "sharpe": baseline_sharpe,
            "cagr": float(baseline_metrics["CAGR"]),
            "annualized_volatility": float(baseline_metrics["Annualized volatility"]),
            "max_drawdown": float(baseline_metrics["Max drawdown"]),
            "validation_status": VALIDATION_STATUS,
        }
    ]
    print(
        f"  {BASELINE_LABEL:<24} markets {rows[0]['markets_remaining']:>2}  "
        f"Sharpe {baseline_sharpe:.4f}  CAGR {rows[0]['cagr']:+.4%}"
    )

    for excluded_class, members in EXCLUSION_MANIFEST:
        row = _replay_row(config, frames, excluded_class, members)
        rows.append(row)
        print(
            f"  {excluded_class:<24} markets {row['markets_remaining']:>2}  "
            f"Sharpe {row['sharpe']:.4f}  CAGR {row['cagr']:+.4%}"
        )

    report = pd.DataFrame(rows)
    report["delta_sharpe_vs_baseline"] = report["sharpe"] - baseline_sharpe
    report["delta_cagr_vs_baseline"] = report["cagr"] - rows[0]["cagr"]
    report = report.loc[:, REPORT_COLUMNS]
    report.to_csv(output_dir / "validation_universe_jackknife.csv", index=False)
    print(
        f"wrote {len(report)} rows to "
        f"{output_dir / 'validation_universe_jackknife.csv'} "
        f"in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
