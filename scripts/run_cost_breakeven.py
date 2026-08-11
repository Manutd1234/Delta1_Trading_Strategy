"""Price the frozen configuration under a ladder of execution-cost multiples.

One full reference run per multiplier in ``MULTIPLIER_GRID``, each with all
five execution-cost inputs scaled together and every other parameter frozen,
written to ``outputs/validation/validation_cost_breakeven.csv``: one row per
multiplier, plus two summary rows locating the multiplier at which net CAGR
and net Sharpe cross zero.

What the script refuses to do.

It does not evaluate a family.  Every run prices the SAME frozen strategy
under a different cost assumption; no run is a candidate, so no row here may
be fed to selection, and the rows carry no ranking and no verdict.

It does not claim to measure the breakeven.  The crossing multiplier is
linear interpolation between adjacent grid points -- extrapolation from the
last segment if the metric never crosses inside the grid, and the row says
which -- so it is an estimate of where the crossing lies, not a measurement.

It does not write before reconciling.  The 1.0x run must reproduce the
published full-history row of ``outputs/strategy_metrics.csv`` to 12 decimal
places, and the 2.0x run must reproduce the jointly-doubled scenario in
``outputs/strategy_friction_stress.csv``, before any row is emitted.

    python scripts/run_cost_breakeven.py \
        --data-dir "Round1AllData/Quant Researcher/Delta1"
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))

import delta1_reference as d1  # noqa: E402

# The manifest.  Declared here, in full, before any file is read; the artifact
# is checked against it, and the test suite pins the two as equal.
MULTIPLIER_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)

COST_KEYS = ("half_spread_ticks", "slippage_ticks", "commission", "fees",
             "impact_bps_at_full_participation")

OUT = REPO / "outputs" / "validation"
CANON = REPO / "outputs"

WINDOW = ("1990-01-01", "2014-12-31")

VALIDATION_STATUS = "retrospective_sensitivity"
PERMITTED_USE = (
    "prices one frozen strategy under scaled execution-cost assumptions on reused "
    "1990-2014 history; not a family and must not be fed to selection; breakeven rows "
    "are interpolation estimates, not measurements"
)


@contextmanager
def override(**changes: float):
    """Temporarily set keys in the module-level parameter dict, then restore."""
    previous = {key: d1.P[key] for key in changes}
    d1.P.update(changes)
    try:
        yield
    finally:
        d1.P.update(previous)


def sweep(data_dir: str, base_costs: dict[str, float]) -> pd.DataFrame:
    """One reference run per multiplier; every non-cost parameter frozen."""
    rows: list[dict[str, object]] = []
    for multiplier in MULTIPLIER_GRID:
        with override(**{key: multiplier * base_costs[key] for key in COST_KEYS}):
            stats = d1.metrics(d1.run(data_dir), *WINDOW)
        rows.append({
            "row_type": "grid",
            "cost_multiplier": multiplier,
            "crossing_metric": "",
            "convention": "measured: full reference run with all five execution-cost "
                          "inputs scaled by the multiplier",
            "start": stats["Start"],
            "end": stats["End"],
            "net_cagr": float(stats["CAGR"]),
            "net_sharpe": float(stats["Sharpe (rf=0)"]),
            "max_drawdown": float(stats["Max drawdown"]),
            "annual_cost_drag": float(stats["Annual cost drag"]),
            "within_declared_grid": True,
            "validation_status": VALIDATION_STATUS,
            "permitted_use": PERMITTED_USE,
        })
        print(f"{multiplier:5.1f}x  sharpe {rows[-1]['net_sharpe']:+.4f}  "
              f"cagr {rows[-1]['net_cagr']:+.4f}  drag {rows[-1]['annual_cost_drag']:.4f}")
    return pd.DataFrame(rows)


def zero_crossing(multipliers: list[float], values: list[float]) -> tuple[float, bool]:
    """Multiplier where the piecewise-linear metric crosses zero.

    Returns ``(multiplier, within_grid)``.  Interpolates inside the first grid
    segment whose endpoints bracket zero; if no segment does, extrapolates the
    last segment, and the caller labels the row an extrapolation.
    """
    pairs = list(zip(multipliers, values))
    for (m0, v0), (m1, v1) in zip(pairs, pairs[1:]):
        if v0 == 0.0:
            return m0, True
        if (v0 > 0.0) != (v1 > 0.0):
            return m0 - v0 * (m1 - m0) / (v1 - v0), True
    (m0, v0), (m1, v1) = pairs[-2], pairs[-1]
    if v1 == 0.0:
        return m1, True
    if v1 == v0:
        return float("nan"), False
    return m0 - v0 * (m1 - m0) / (v1 - v0), False


def breakeven_row(grid: pd.DataFrame, metric: str) -> dict[str, object]:
    multiplier, within = zero_crossing(
        grid["cost_multiplier"].tolist(), grid[metric].tolist()
    )
    if within:
        convention = (f"linear interpolation of {metric} in the cost multiplier between "
                      "the adjacent grid runs that bracket the sign change; an estimate, "
                      "not a measurement")
    else:
        convention = (f"linear EXTRAPOLATION of {metric} from the last grid segment "
                      f"({grid['cost_multiplier'].iloc[-2]:g}x-"
                      f"{grid['cost_multiplier'].iloc[-1]:g}x): the metric never crosses "
                      "zero inside the declared grid, so this lies outside the measured "
                      "range; an estimate, not a measurement")
    return {
        "row_type": "breakeven",
        "cost_multiplier": multiplier,
        "crossing_metric": metric,
        "convention": convention,
        "start": "",
        "end": "",
        "net_cagr": 0.0 if metric == "net_cagr" else np.nan,
        "net_sharpe": 0.0 if metric == "net_sharpe" else np.nan,
        "max_drawdown": np.nan,
        "annual_cost_drag": np.nan,
        "within_declared_grid": within,
        "validation_status": VALIDATION_STATUS,
        "permitted_use": PERMITTED_USE,
    }


def reconcile(grid: pd.DataFrame) -> None:
    """Refuse to write unless the sweep reproduces the canonical bundle.

    The 1.0x run is checked against the published full-history metrics, and
    the 2.0x run against ``double_all_execution_costs`` in the friction
    stress, which doubles the same five inputs jointly and so is the same
    construction as multiplier 2.0.  The friction stress has no comparable
    3.0x scenario -- it stops at 2x, and its remaining rows stress fixed costs
    and impact separately or change non-cost assumptions -- so the 3.0x run
    has nothing canonical to reconcile against and is not forced to.
    """
    grid = grid.set_index("cost_multiplier")

    published = pd.read_csv(CANON / "strategy_metrics.csv")
    full = published.loc[
        published["Window"].eq("1990-2014 full post-launch history")
    ].iloc[0]
    for field, column in (("net_sharpe", "Naive daily Sharpe (sqrt252, rf=0)"),
                          ("net_cagr", "CAGR"),
                          ("max_drawdown", "Max drawdown"),
                          ("annual_cost_drag", "Annual cost drag")):
        if abs(grid.at[1.0, field] - float(full[column])) > 1e-12:
            raise AssertionError(
                f"1.0x {field}={grid.at[1.0, field]!r} does not reconcile with "
                f"strategy_metrics.csv {column}={full[column]!r}"
            )

    stress = pd.read_csv(CANON / "strategy_friction_stress.csv").set_index("scenario")
    doubled = stress.loc["double_all_execution_costs"]
    for field, column in (("net_sharpe", "sharpe"), ("net_cagr", "cagr"),
                          ("max_drawdown", "max_drawdown"),
                          ("annual_cost_drag", "annual_cost_drag")):
        if abs(grid.at[2.0, field] - float(doubled[column])) > 1e-12:
            raise AssertionError(
                f"2.0x {field}={grid.at[2.0, field]!r} does not reconcile with "
                f"strategy_friction_stress.csv {column}={doubled[column]!r}"
            )
    print("reconciled: 1.0x against strategy_metrics.csv, "
          "2.0x against strategy_friction_stress.csv (double_all_execution_costs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="Round1AllData/Quant Researcher/Delta1")
    args = parser.parse_args()
    data_dir = str((REPO / args.data_dir).resolve())

    pristine = dict(d1.P)
    base_costs = {key: d1.P[key] for key in COST_KEYS}

    grid = sweep(data_dir, base_costs)
    reconcile(grid)

    frame = pd.concat(
        [grid, pd.DataFrame([breakeven_row(grid, "net_cagr"),
                             breakeven_row(grid, "net_sharpe")])],
        ignore_index=True,
    )

    # Every run mutated the module-level parameter dict and restored it.  Prove
    # the restoration was complete before anything downstream reuses d1.
    if pristine != d1.P:
        raise AssertionError("d1.P was not restored after the sweep")

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "validation_cost_breakeven.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["row_type", "cost_multiplier", "crossing_metric", "net_cagr", "net_sharpe",
            "max_drawdown", "annual_cost_drag", "within_declared_grid"]
    print(frame[show].to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    print(f"\nwritten to {OUT / 'validation_cost_breakeven.csv'}")


if __name__ == "__main__":
    main()
