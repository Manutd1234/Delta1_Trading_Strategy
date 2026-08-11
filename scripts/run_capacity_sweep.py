#!/usr/bin/env python3
"""Run the DELTA1 capacity sweep and write its artifact.

The sweep characterizes the cost model and participation bounds of the FROZEN
v3.2.1 configuration at scale.  Every level runs the identical configuration
with only ``initial_capital`` changed, so a difference between rows can only
come from how the impact model, the participation caps and the roll-backlog
guard respond to larger orders.  Capital is not a tuning dimension: the rows
are emitted in the order the grid was declared and no row carries a verdict.

What the script refuses to do.

It does not emit anything before the 1x level reproduces the published
full-history naive daily Sharpe from ``outputs/strategy_metrics.csv``.  A sweep
whose base level does not match the frozen baseline is not measuring the frozen
configuration.

It does not route around the roll-backlog guard.  A level that aborts on
``max_roll_backlog_sessions`` is recorded as that row's outcome in place of
metrics, because the abort is the capacity result at that scale, not a failure
to obtain one.

It does not write outside its own output directory and never calls
``strategy.save_outputs``, which would overwrite the frozen canonical bundle
the rest of the repository is checked against.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research import levers
from delta1_strategy.research.strategy import (
    StrategyConfig,
    performance_metrics,
    run_backtest,
)

# The capital grid.  Declared here, before any file is read, and written
# verbatim into the artifact.  1x is the frozen initial capital of the
# published baseline; every other level is a pure multiple of it.
FROZEN_INITIAL_CAPITAL = 1_000_000.0
CAPITAL_MULTIPLES: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)

# Sharpe-erosion levels the summary rows locate on the grid, in Sharpe units
# relative to the 1x row.
SHARPE_EROSION_THRESHOLDS: tuple[float, ...] = (0.10, 0.25)
EROSION_CONVENTION = (
    "linear interpolation of Sharpe erosion (1x naive daily Sharpe minus the "
    "level's) against log10(initial capital) between the first pair of "
    "completed grid levels that bracket the threshold; a threshold beyond the "
    "largest completed level extends the final segment's slope and is "
    "labelled an extrapolation"
)

PUBLISHED_WINDOW = "1990-2014 full post-launch history"
SHARPE_COLUMN = "Naive daily Sharpe (sqrt252, rf=0)"
BACKLOG_GUARD_PREFIX = "Delivery roll cannot be completed within"

VALIDATION_STATUS = "retrospective_capacity_characterization"
PERMITTED_USE = (
    "descriptive cost-model and participation-bound characterization of the "
    "frozen configuration at scale on reused 1990-2014 history; capital is "
    "not a tuning dimension and no row licenses a configuration change"
)

COLUMNS = (
    "row_type",
    "capital_multiple",
    "initial_capital_usd",
    "run_outcome",
    "sharpe",
    "sharpe_erosion_vs_1x",
    "cagr",
    "max_drawdown",
    "annual_cost_drag",
    "annual_fixed_cost_drag",
    "annual_impact_cost_drag",
    "mean_nonzero_order_participation",
    "peak_order_participation",
    "sessions_nonzero_deferral_backlog",
    "sessions_nonzero_capacity_deferral",
    "sessions_nonzero_roll_backlog",
    "peak_pending_markets",
    "portfolio_limit_binding_decision_days",
    "backlog_guard_raised",
    "backlog_guard_detail",
    "erosion_threshold",
    "estimation_method",
    "estimation_convention",
    "validation_status",
    "permitted_use",
)


def published_full_history_sharpe(metrics_path: Path) -> float:
    """The frozen baseline's full-history naive daily Sharpe, read exactly."""

    frame = pd.read_csv(metrics_path, float_precision="round_trip")
    rows = frame.loc[frame["Window"] == PUBLISHED_WINDOW]
    if len(rows) != 1:
        raise ValueError(
            f"{metrics_path} carries {len(rows)} rows for {PUBLISHED_WINDOW!r}; expected one"
        )
    return float(rows.iloc[0][SHARPE_COLUMN])


def erosion_crossing_capital(
    capitals: tuple[float, ...] | list[float],
    erosions: tuple[float, ...] | list[float],
    threshold: float,
) -> tuple[float, str]:
    """Capital where Sharpe erosion first reaches ``threshold``.

    Implements ``EROSION_CONVENTION``: the first grid segment whose right edge
    reaches the threshold is interpolated linearly in log10(capital); beyond
    the grid the final segment's slope is extended and the result labelled an
    extrapolation.  A flat or negative final slope leaves the crossing
    undefined rather than inventing one.
    """

    if len(capitals) != len(erosions):
        raise ValueError("capitals and erosions must be the same length")
    if len(capitals) < 2:
        return float("nan"), "not_estimable_insufficient_completed_levels"
    logs = np.log10(np.asarray(capitals, dtype=float))
    values = np.asarray(erosions, dtype=float)
    if values[0] >= threshold:
        return float(capitals[0]), "interpolated_within_grid"
    for right in range(1, len(values)):
        if values[right] >= threshold:
            left = right - 1
            fraction = (threshold - values[left]) / (values[right] - values[left])
            crossing = logs[left] + fraction * (logs[right] - logs[left])
            return float(10.0**crossing), "interpolated_within_grid"
    slope = (values[-1] - values[-2]) / (logs[-1] - logs[-2])
    if slope <= 0:
        return float("nan"), "not_estimable_final_slope_nonpositive"
    crossing = logs[-1] + (threshold - values[-1]) / slope
    return float(10.0**crossing), "extrapolated_beyond_final_grid_level"


def _level_row(multiple: int, config: StrategyConfig, result) -> dict[str, object]:
    """One completed level, measured over the full published window."""

    metrics = performance_metrics(
        result, levers.RETROSPECTIVE_START, levers.RETROSPECTIVE_END, config.annualization
    )
    daily = result.daily.loc[levers.RETROSPECTIVE_START : levers.RETROSPECTIVE_END]
    participation = daily["max_order_participation"]
    nonzero = participation[participation > 0]
    deferred = daily["capacity_deferred_contracts"] > 0
    backlog = daily["roll_backlog_contracts"] > 0
    return {
        "row_type": "capacity_level",
        "capital_multiple": float(multiple),
        "initial_capital_usd": float(config.initial_capital),
        "run_outcome": "completed",
        "sharpe": float(metrics[SHARPE_COLUMN]),
        "cagr": float(metrics["CAGR"]),
        "max_drawdown": float(metrics["Max drawdown"]),
        "annual_cost_drag": float(metrics["Annual cost drag"]),
        "annual_fixed_cost_drag": float(metrics["Annual fixed-cost drag"]),
        "annual_impact_cost_drag": float(metrics["Annual impact-cost drag"]),
        "mean_nonzero_order_participation": (
            float(nonzero.mean()) if not nonzero.empty else 0.0
        ),
        "peak_order_participation": float(metrics["Peak order participation"]),
        "sessions_nonzero_deferral_backlog": int((deferred | backlog).sum()),
        "sessions_nonzero_capacity_deferral": int(deferred.sum()),
        "sessions_nonzero_roll_backlog": int(backlog.sum()),
        "peak_pending_markets": int(metrics["Peak pending markets"]),
        "portfolio_limit_binding_decision_days": int(
            metrics["Portfolio-limit binding decision days"]
        ),
        "backlog_guard_raised": False,
        "backlog_guard_detail": "",
        "validation_status": VALIDATION_STATUS,
        "permitted_use": PERMITTED_USE,
    }


def _aborted_row(multiple: int, config: StrategyConfig, detail: str) -> dict[str, object]:
    """A level the roll-backlog guard refused: the abort is the result."""

    return {
        "row_type": "capacity_level",
        "capital_multiple": float(multiple),
        "initial_capital_usd": float(config.initial_capital),
        "run_outcome": "aborted_roll_backlog_guard",
        "backlog_guard_raised": True,
        "backlog_guard_detail": detail,
        "validation_status": VALIDATION_STATUS,
        "permitted_use": PERMITTED_USE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument(
        "--metrics", type=Path, default=Path("outputs/strategy_metrics.csv")
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    published = published_full_history_sharpe(args.metrics)
    frames = levers.load_shared_frames(args.data_dir)
    base_config = StrategyConfig(data_dir=args.data_dir, output_dir=output_dir)

    rows: list[dict[str, object]] = []
    for multiple in CAPITAL_MULTIPLES:
        config = replace(
            base_config, initial_capital=multiple * FROZEN_INITIAL_CAPITAL
        )
        try:
            result = run_backtest(config, **frames)
        except ValueError as error:
            if not str(error).startswith(BACKLOG_GUARD_PREFIX):
                raise
            rows.append(_aborted_row(multiple, config, str(error)))
            print(f"  {multiple:>4}x  aborted: {error}")
            continue
        row = _level_row(multiple, config, result)
        if multiple == 1 and not np.isclose(
            row["sharpe"], published, atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                "1x level does not reproduce the published baseline Sharpe: "
                f"expected {published!r}, observed {row['sharpe']!r}. Nothing "
                "was emitted."
            )
        rows.append(row)
        print(
            f"  {multiple:>4}x  Sharpe {row['sharpe']:.4f}  "
            f"cost drag {row['annual_cost_drag']:.4%}  "
            f"peak participation {row['peak_order_participation']:.4f}"
        )

    base_row = rows[0]
    if base_row["run_outcome"] != "completed":
        # The 1x assertion above never ran, so nothing here is checked against
        # the published baseline and the sweep has no base for erosion.
        raise ValueError(
            "1x level did not complete, so the sweep cannot be anchored to the "
            "published baseline. Nothing was emitted."
        )
    sharpe_1x = base_row["sharpe"]
    for row in rows:
        if row["run_outcome"] == "completed":
            row["sharpe_erosion_vs_1x"] = sharpe_1x - row["sharpe"]

    completed = [row for row in rows if row["run_outcome"] == "completed"]
    capitals = [row["initial_capital_usd"] for row in completed]
    erosions = [row["sharpe_erosion_vs_1x"] for row in completed]
    for threshold in SHARPE_EROSION_THRESHOLDS:
        capital, method = erosion_crossing_capital(capitals, erosions, threshold)
        rows.append(
            {
                "row_type": "sharpe_erosion_threshold",
                "capital_multiple": (
                    capital / FROZEN_INITIAL_CAPITAL
                    if math.isfinite(capital)
                    else float("nan")
                ),
                "initial_capital_usd": capital,
                "erosion_threshold": threshold,
                "estimation_method": method,
                "estimation_convention": EROSION_CONVENTION,
                "validation_status": VALIDATION_STATUS,
                "permitted_use": PERMITTED_USE,
            }
        )

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table.to_csv(output_dir / "validation_capacity.csv", index=False)
    print(f"\nwrote {len(table)} rows to {output_dir / 'validation_capacity.csv'}")


if __name__ == "__main__":
    main()
