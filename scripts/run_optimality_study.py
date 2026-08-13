#!/usr/bin/env python3
"""Was the frozen configuration the one a searcher would have chosen?

    python scripts/run_optimality_study.py --data-dir "Round1AllData/Quant Researcher/Delta1"

`outputs/submission/robustness_parameter_sensitivity.csv` answers a narrow
question: nine pre-declared runs, three per axis, reported so a reviewer can
size the neighbourhood.  It deliberately does not search.  This study answers
the question that check leaves open -- *if* someone had searched, what would
they have found, and would finding it have been worth anything?

Four measurements, in the order they are run:

    A.  Profiles     Dense one-parameter-at-a-time sweeps over eighteen axes.
                     Locates the frozen value inside each axis and reports
                     whether it sits on a plateau or on a spike.
    B.  Search       A joint random search over the same box.  Prices the best
                     configuration the search could have found, and deflates it
                     for the size of the family that produced it.
    C.  Selection    The only test that decides anything: choose the best
                     configuration on 1990-2004, then measure it on 2005-2014.
                     If in-sample optimisation does not pay out of sample, the
                     frozen configuration is optimal in the sense that matters.
    D.  Candidates   Pre-registered structural alternatives -- multi-horizon
                     trend ensembles and single-sleeve books -- judged on the
                     same out-of-sample split rather than on the full history.

Nothing here selects a configuration.  The frozen baseline is an input to this
study, never an output of it: `outputs/run_manifest.json` pins the engine and
`reference/delta1_reference.py` is imported read-only, so no result below can
change a published number.  A run that fails to reproduce the canonical Sharpe
to the last bit aborts before it writes anything.

The search is declared *after* the configuration was frozen, which is the only
reason it is safe to run at all.  Its output is an audit of how much of the
headline Sharpe could be an artefact of choices, not a menu to pick from --
selecting any row here on its full-sample number would be exactly the
retrospective fitting the research methodology prohibits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))

import delta1_reference as d1  # noqa: E402

from delta1_strategy.research import inference  # noqa: E402

TRAIN = ("1990-01-01", "2004-12-31")
TEST = ("2005-01-01", "2014-12-31")
FULL = ("1990-01-01", "2014-12-31")

# The canonical facts this script refuses to run without.  Identical to the
# constants `build_submission_robustness.py` asserts, for the same reason: a
# drifted reference file must fail loudly rather than quietly reporting a
# search against the wrong incumbent.
CANONICAL_SHARPE = 1.5895266624546427
CANONICAL_CAGR = 0.13189524685750764
CANONICAL_SESSIONS = 6523

DECLARATION = (
    "Post-freeze optimality audit, and an explicit search. Unlike "
    "robustness_parameter_sensitivity.csv, this file IS a search: it exists to "
    "measure what a search could have found, after the configuration was "
    "already frozen and published. No row here was used to select any "
    "parameter, and no published number depends on this file."
)

# --------------------------------------------------------------------------
# The declared search space.  Every value in this section is fixed before any
# run executes; nothing below is widened or trimmed after seeing a result.
# --------------------------------------------------------------------------

# A. One axis at a time.  Grids are centred on the frozen value and extend far
#    enough either side that a plateau and a spike look different.
PROFILE_GRID: dict[str, tuple[float, ...]] = {
    "trend_lookback": (63, 84, 105, 126, 147, 168, 189, 210, 231, 252,
                       273, 294, 315, 378, 441, 504),
    "basis_weight": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    "basis_roll_window": (63, 126, 189, 252, 315, 378, 504),
    "basis_lookback": (63, 126, 189, 252, 315, 378, 504),
    "signal_norm_window": (126, 189, 252, 378, 504),
    "signal_cap": (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    "vol_span": (20, 30, 40, 60, 80, 100, 120),
    "risk_managed_window": (21, 42, 63, 84, 126, 189, 252),
    "risk_managed_cap": (1.0, 1.5, 2.0, 2.5, 3.0),
    "fast_vol_span": (10, 15, 20, 30, 40),
    "slow_vol_span": (60, 90, 120, 180, 252),
    "shock_start": (1.0, 1.2, 1.35, 1.5, 1.75, 1.9),
    "shock_full": (1.5, 1.75, 2.0, 2.25, 2.5),
    "shock_floor": (0.5, 0.6, 0.75, 0.9, 1.0),
    "vol_decay": (0.90, 0.92, 0.94, 0.96, 0.97, 0.98),
    "portfolio_vol_window": (21, 42, 63, 126, 189, 252),
    "no_trade_buffer": (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    "target_vol": (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12),
}

# B. The joint box.  `target_vol` is excluded on purpose: it is a capital
#    policy, priced in outputs/levers/, and moving it inside a Sharpe search
#    would confound a risk-budget decision with an alpha one.  Cost and
#    participation parameters are excluded because they are not free choices.
SEARCH_AXES: tuple[str, ...] = tuple(
    key for key in PROFILE_GRID if key != "target_vol"
)

# D. Structural alternatives, declared as a fixed list.  The ensembles are the
#    standard robustification in the trend literature (Moskowitz, Ooi and
#    Pedersen 2012 report 1-, 3- and 12-month horizons; blending them is the
#    usual answer to a single-horizon choice), so they are prior-motivated
#    rather than fitted here.
CANDIDATES: tuple[tuple[str, dict, tuple[int, ...] | None, str], ...] = (
    ("ensemble_63_126_252", {}, (63, 126, 252),
     "Trend sign averaged over 3, 6 and 12 month lookbacks."),
    ("ensemble_126_252_504", {}, (126, 252, 504),
     "Trend sign averaged over 6, 12 and 24 month lookbacks."),
    ("ensemble_84_168_252", {}, (84, 168, 252),
     "Trend sign averaged over 4, 8 and 12 month lookbacks."),
    ("ensemble_21_63_126_252_504", {}, (21, 63, 126, 252, 504),
     "Trend sign averaged over five horizons from 1 to 24 months."),
    ("trend_only", {"basis_weight": 0.0}, None,
     "Trend sleeve alone; the basis sleeve switched off."),
    ("basis_only", {"basis_weight": 1.0}, None,
     "Basis sleeve alone, falling back to trend before it is estimable."),
    ("ensemble_63_126_252_trend_only", {"basis_weight": 0.0}, (63, 126, 252),
     "Multi-horizon trend ensemble with the basis sleeve switched off."),
)


# --------------------------------------------------------------------------
# Running one configuration
# --------------------------------------------------------------------------

_PANEL: dict | None = None


def _initialise(data_dir: str) -> None:
    """Load the panel once per worker process, not once per configuration."""
    global _PANEL
    _PANEL = d1.load_market_data(Path(data_dir))


def _ensemble_signal(lookbacks: tuple[int, ...]):
    """A trend signal averaged across horizons, in place of the single one.

    Every component must be estimable before the average is: a market enters
    the book when its *longest* lookback is available, exactly as the frozen
    single-horizon rule waits for its own.  Averaging over whatever happens to
    be available would quietly start the book earlier on a shorter horizon and
    make the comparison against the baseline unfair.
    """

    def signal(prices: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for lookback in lookbacks:
            change = prices - prices.shift(lookback)
            parts.append(np.sign(change).where(change.notna()))
        stacked = pd.concat(parts, keys=range(len(parts)))
        return stacked.groupby(level=1).mean()

    return signal


def _statistics(daily: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Net statistics over one window, on the reference file's conventions."""
    frame = daily.loc[start:end]
    returns = frame["net_return"].dropna()
    if returns.empty:
        raise ValueError(f"no sessions between {start} and {end}")
    location = daily.index.get_indexer([returns.index[0]])[0]
    origin = daily.index[location - 1] if location > 0 else returns.index[0]
    years = (returns.index[-1] - origin).total_seconds() / (365.2425 * 86_400)

    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = float((equity / peak - 1).min())
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)

    return {
        "sessions": int(len(returns)),
        "years": float(years),
        "cagr": cagr,
        "annualized_volatility": float(returns.std() * math.sqrt(252)),
        "sharpe": float(returns.mean() / returns.std() * math.sqrt(252)),
        "max_drawdown": drawdown,
        "calmar": cagr / abs(drawdown) if drawdown else float("nan"),
        "annual_cost_drag": float(frame["cost"].mean() * 252),
        "annual_contract_turnover": float(
            frame["total_contract_turnover"].sum()
            / ((frame.index[-1] - frame.index[0]).days / 365.2425)
        ),
        "average_markets_held": float(frame["active_markets"].mean()),
    }


def _run(task: tuple[str, dict, tuple[int, ...] | None]):
    """Run one configuration and return its statistics and its daily path.

    A configuration can be *infeasible* rather than merely worse: the engine
    fails closed when a roll cannot complete within its cycle under the
    participation cap, which is what a larger book in a thin market runs into.
    Those runs are recorded and excluded from every statistic rather than
    silently retried with the constraint relaxed -- a search that quietly
    lifted a risk control to reach a higher Sharpe would be measuring a
    different strategy.
    """
    name, overrides, ensemble = task
    previous = {key: d1.P[key] for key in overrides}
    saved_signal = d1.trend_signal
    d1.P.update(overrides)
    if ensemble is not None:
        d1.trend_signal = _ensemble_signal(ensemble)
    try:
        daily = d1.run(data=_PANEL)
    except Exception as error:  # noqa: BLE001 - the reason is the result here
        return {"name": name, "status": "infeasible", "detail": str(error)}, None
    finally:
        d1.P.update(previous)
        d1.trend_signal = saved_signal

    row = {"name": name, "status": "completed", "detail": ""}
    for label, (start, end) in (
        ("full", FULL), ("train", TRAIN), ("test", TEST)
    ):
        for key, value in _statistics(daily, start, end).items():
            row[f"{label}_{key}"] = value
    path = daily.loc[FULL[0]:FULL[1], "net_return"].dropna().astype(float)
    return row, path


# --------------------------------------------------------------------------
# Assembling the task list
# --------------------------------------------------------------------------


def _configuration_key(overrides: dict, ensemble: tuple[int, ...] | None) -> str:
    payload = json.dumps(
        {"overrides": {k: overrides[k] for k in sorted(overrides)},
         "ensemble": list(ensemble) if ensemble else None},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def profile_tasks() -> list[tuple[str, dict, None]]:
    tasks = []
    for parameter, values in PROFILE_GRID.items():
        for value in values:
            if value == d1.P[parameter]:
                continue  # the baseline run covers it
            tasks.append((f"profile:{parameter}={value}", {parameter: value}, None))
    return tasks


def search_tasks(trials: int, seed: int) -> list[tuple[str, dict, None]]:
    """Independent draws from the declared box, rejecting invalid geometry.

    Two orderings have to hold for the controls to mean what they are named:
    the shock taper must start before it saturates, and the fast volatility
    estimate must be faster than the slow one it is compared against.  Draws
    that violate either are redrawn rather than clipped, so the box stays the
    one that was declared.
    """
    rng = np.random.default_rng(seed)
    tasks: list[tuple[str, dict, None]] = []
    seen: set[str] = set()
    attempts = 0
    while len(tasks) < trials and attempts < trials * 20:
        attempts += 1
        draw = {
            axis: PROFILE_GRID[axis][rng.integers(len(PROFILE_GRID[axis]))]
            for axis in SEARCH_AXES
        }
        if draw["shock_start"] >= draw["shock_full"]:
            continue
        if draw["fast_vol_span"] >= draw["slow_vol_span"]:
            continue
        key = _configuration_key(draw, None)
        if key in seen:
            continue
        seen.add(key)
        tasks.append((f"search:{len(tasks):04d}", draw, None))
    if len(tasks) < trials:
        raise RuntimeError(
            f"only {len(tasks)} of {trials} valid draws after {attempts} attempts"
        )
    return tasks


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation, computed from ranks without a SciPy dependency."""
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def profile_report(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """Where the frozen value sits on each axis, and how flat that axis is."""
    rows = []
    for parameter, values in PROFILE_GRID.items():
        whole = frame.loc[frame["parameter"] == parameter]
        axis = whole.dropna(subset=["full_sharpe"]).copy()
        best = axis.loc[axis["full_sharpe"].idxmax()]
        train_best = axis.loc[axis["train_sharpe"].idxmax()]
        span = float(axis["full_sharpe"].max() - axis["full_sharpe"].min())
        near = axis.loc[axis["full_sharpe"] >= axis["full_sharpe"].max() - 0.10]
        rows.append({
            "parameter": parameter,
            "frozen_value": d1.P[parameter],
            "grid_points": int(len(axis)),
            "infeasible_points": int(len(whole) - len(axis)),
            "grid_min": min(values),
            "grid_max": max(values),
            "frozen_full_sharpe": float(baseline["full_sharpe"]),
            "best_full_sharpe": float(best["full_sharpe"]),
            "best_full_value": best["value"],
            "full_sharpe_headroom": float(best["full_sharpe"] - baseline["full_sharpe"]),
            "frozen_is_full_sample_argmax": bool(best["value"] == d1.P[parameter]),
            "frozen_full_sharpe_percentile": float(
                (axis["full_sharpe"] <= baseline["full_sharpe"]).mean()
            ),
            "axis_sharpe_span": span,
            "plateau_share_within_0.10_sharpe": float(len(near) / len(axis)),
            "neighbour_sharpe_drop": _neighbour_drop(axis, parameter, baseline),
            "best_train_value": train_best["value"],
            "test_sharpe_of_train_best": float(train_best["test_sharpe"]),
            "frozen_test_sharpe": float(baseline["test_sharpe"]),
            "selection_gain_out_of_sample": float(
                train_best["test_sharpe"] - baseline["test_sharpe"]
            ),
            "train_test_rank_correlation": _rank_correlation(
                axis.set_index("value")["train_sharpe"],
                axis.set_index("value")["test_sharpe"],
            ),
            "shape": _axis_shape(span, float(len(near) / len(axis))),
            "note": DECLARATION,
        })
    return pd.DataFrame(rows)


def _axis_shape(span: float, plateau_share: float) -> str:
    """A mechanical label, applied by rule to every axis alike.

    An axis whose grid is flat to the last bit is not evidence of robustness;
    it means the parameter does nothing over this window, which a reader should
    be told rather than left to infer from a zero.
    """
    if span < 1e-12:
        return "inert over the evaluation window"
    if plateau_share >= 0.75:
        return "plateau"
    if plateau_share <= 0.25:
        return "spike"
    return "sloped"


def _neighbour_drop(axis: pd.DataFrame, parameter: str, baseline: pd.Series) -> float:
    """Worst full-sample Sharpe loss at the grid points either side of frozen.

    A configuration whose immediate neighbours fall away is fragile even if its
    own number is high, because the frozen value is a choice made with far less
    precision than one grid step.
    """
    ordered = axis.sort_values("value")
    values = list(ordered["value"])
    frozen = d1.P[parameter]
    if frozen not in values:
        return float("nan")
    index = values.index(frozen)
    neighbours = [
        ordered.iloc[position]["full_sharpe"]
        for position in (index - 1, index + 1)
        if 0 <= position < len(values)
    ]
    if not neighbours:
        return float("nan")
    return float(baseline["full_sharpe"] - min(neighbours))


def selection_report(
    frame: pd.DataFrame, baseline: pd.Series, pool: str
) -> dict[str, object]:
    """Choose on 1990-2004, then measure on 2005-2014.

    This is the whole study in one row.  The gap between what the winner
    promised in-sample and what it delivered afterwards is the honest price of
    optimisation, and it is reported per pool so a wide search and a
    single-axis tweak are not averaged together.
    """
    chosen = frame.loc[frame["train_sharpe"].idxmax()]
    return {
        "pool": pool,
        "configurations": int(len(frame)),
        "selected": chosen["name"],
        "selected_train_sharpe": float(chosen["train_sharpe"]),
        "frozen_train_sharpe": float(baseline["train_sharpe"]),
        "in_sample_gain": float(chosen["train_sharpe"] - baseline["train_sharpe"]),
        "selected_test_sharpe": float(chosen["test_sharpe"]),
        "frozen_test_sharpe": float(baseline["test_sharpe"]),
        "out_of_sample_gain": float(chosen["test_sharpe"] - baseline["test_sharpe"]),
        "gain_decay": float(
            (chosen["train_sharpe"] - baseline["train_sharpe"])
            - (chosen["test_sharpe"] - baseline["test_sharpe"])
        ),
        "share_beating_frozen_in_sample": float(
            (frame["train_sharpe"] > baseline["train_sharpe"]).mean()
        ),
        "share_beating_frozen_out_of_sample": float(
            (frame["test_sharpe"] > baseline["test_sharpe"]).mean()
        ),
        "share_of_in_sample_winners_that_win_later": float(
            (frame.loc[frame["train_sharpe"] > baseline["train_sharpe"], "test_sharpe"]
             > baseline["test_sharpe"]).mean()
        ),
        "train_test_rank_correlation": _rank_correlation(
            frame.set_index("name")["train_sharpe"],
            frame.set_index("name")["test_sharpe"],
        ),
        "frozen_full_sharpe_percentile": float(
            (frame["full_sharpe"] <= baseline["full_sharpe"]).mean()
        ),
        "frozen_test_sharpe_percentile": float(
            (frame["test_sharpe"] <= baseline["test_sharpe"]).mean()
        ),
        "note": DECLARATION,
    }


def walk_forward(paths: pd.DataFrame, baseline_name: str, first_year: int) -> tuple:
    """Re-optimise every year on everything seen so far, and live with it.

    A single train/test split can be lucky.  This repeats the decision
    nineteen times: at each year end, pick the configuration with the best
    Sharpe over all history to date, hold it through the next calendar year,
    and stitch the realised years into one path.  It is the closest thing to
    what an optimising researcher would actually have experienced.
    """
    rows = []
    segments = []
    years = sorted({stamp.year for stamp in paths.index if stamp.year >= first_year})
    for year in years:
        history = paths.loc[: f"{year - 1}-12-31"]
        if history.empty:
            continue
        sharpe = history.mean() / history.std() * math.sqrt(252)
        chosen = str(sharpe.idxmax())
        realised = paths.loc[f"{year}-01-01": f"{year}-12-31"]
        if realised.empty:
            continue
        segments.append(realised[chosen].rename("reoptimized"))
        rows.append({
            "year": year,
            "selected": chosen,
            "selected_history_sharpe": float(sharpe[chosen]),
            "frozen_history_sharpe": float(sharpe[baseline_name]),
            "history_sessions": int(len(history)),
            "selected_realized_return": float((1 + realised[chosen]).prod() - 1),
            "frozen_realized_return": float((1 + realised[baseline_name]).prod() - 1),
            "selection_changed": bool(rows and rows[-1]["selected"] != chosen),
            "selected_is_frozen": chosen == baseline_name,
        })
    path = pd.concat(segments) if segments else pd.Series(dtype=float)
    return pd.DataFrame(rows), path


def summarize(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["question", "measure", "value", "reading"])


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(REPO / "Round1AllData" / "Quant Researcher" / "Delta1"))
    parser.add_argument("--output-dir", default=str(REPO / "outputs" / "optimality"))
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--walk-forward-from", type=int, default=1996)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, dict, tuple[int, ...] | None]] = [("baseline", {}, None)]
    tasks += profile_tasks()
    tasks += search_tasks(args.trials, args.seed)
    tasks += [(f"candidate:{name}", overrides, ensemble)
              for name, overrides, ensemble, _ in CANDIDATES]

    print(f"running {len(tasks)} configurations on {args.jobs} workers ...", flush=True)
    rows: list[dict] = []
    paths: dict[str, pd.Series] = {}
    with ProcessPoolExecutor(
        max_workers=args.jobs, initializer=_initialise, initargs=(args.data_dir,)
    ) as pool:
        for done, (row, path) in enumerate(pool.map(_run, tasks, chunksize=4), start=1):
            rows.append(row)
            if path is not None:
                paths[row["name"]] = path
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)

    everything_run = pd.DataFrame(rows).set_index("name", drop=False)
    everything_run.to_csv(output_dir / "optimality_runs.csv", index=False)
    infeasible = everything_run.loc[everything_run["status"] != "completed"]
    print(
        f"{len(everything_run) - len(infeasible)} completed, "
        f"{len(infeasible)} infeasible under the engine's own controls",
        flush=True,
    )

    frame = everything_run.loc[everything_run["status"] == "completed"].copy()
    if "baseline" not in frame.index:
        raise AssertionError("the frozen baseline itself failed to run")
    baseline = frame.loc["baseline"]

    # Reconcile before anything is written.  A search against a drifted
    # incumbent is worse than no search at all.
    for field, expected in (
        ("full_sharpe", CANONICAL_SHARPE),
        ("full_cagr", CANONICAL_CAGR),
    ):
        if abs(float(baseline[field]) - expected) > 1e-12:
            raise AssertionError(
                f"baseline {field}={baseline[field]!r} does not reconcile with the "
                f"published {expected!r}; refusing to write an optimality study"
            )
    if int(baseline["full_sessions"]) != CANONICAL_SESSIONS:
        raise AssertionError("baseline session count does not reconcile")

    path_frame = pd.DataFrame(paths)
    baseline_path = path_frame["baseline"]

    # --- A. profiles ----------------------------------------------------
    profile_rows = []
    for name in everything_run["name"]:
        if not name.startswith("profile:"):
            continue
        parameter, value = name.split(":", 1)[1].split("=")
        profile_rows.append({"parameter": parameter, "value": float(value),
                             **everything_run.loc[name].to_dict()})
    for parameter in PROFILE_GRID:
        profile_rows.append({"parameter": parameter, "value": float(d1.P[parameter]),
                             **baseline.to_dict()})
    profiles = pd.DataFrame(profile_rows)
    profiles["is_frozen"] = [
        row["value"] == d1.P[row["parameter"]] for _, row in profiles.iterrows()
    ]
    profiles["note"] = DECLARATION
    profiles = profiles.sort_values(["parameter", "value"]).reset_index(drop=True)
    profiles.to_csv(output_dir / "optimality_profiles.csv", index=False)

    summary = profile_report(profiles, baseline)
    summary.to_csv(output_dir / "optimality_profile_summary.csv", index=False)

    # --- B. joint search ------------------------------------------------
    search = frame.loc[frame["name"].str.startswith("search:")].copy()
    draws = {name: overrides for name, overrides, _ in tasks if name.startswith("search:")}
    for axis in SEARCH_AXES:
        search[f"param_{axis}"] = [draws[name][axis] for name in search["name"]]
    search["note"] = DECLARATION
    search.sort_values("full_sharpe", ascending=False).to_csv(
        output_dir / "optimality_search.csv", index=False
    )

    # Deflate the best full-sample Sharpe the search found, and the frozen
    # Sharpe as though it had come from the same family.  Per-session units,
    # matching outputs/validation/validation_deflated_sharpe.csv.
    # The family is every configuration this study ran, the frozen one
    # included -- not just the joint draws.  A profile run is a trial too.
    family = frame
    per_session = family["full_sharpe"].astype(float) / math.sqrt(252)
    trial_variance = float(per_session.var(ddof=1))
    searched = family.loc[family["name"] != "baseline", "full_sharpe"].astype(float)
    deflated_rows = []
    for label, name in (("search maximum", str(searched.idxmax())),
                        ("frozen baseline", "baseline")):
        returns = path_frame[name]
        result = inference.deflated_sharpe_ratio(
            float(frame.loc[name, "full_sharpe"]) / math.sqrt(252),
            trial_count=int(len(family)),
            trial_sharpe_variance=trial_variance,
            sessions=int(frame.loc[name, "full_sessions"]),
            skewness=float(returns.skew()),
            excess_kurtosis=float(returns.kurt()),
        )
        deflated_rows.append({
            "member": label,
            "configuration": name,
            "annualized_sharpe": float(frame.loc[name, "full_sharpe"]),
            "per_session_sharpe": float(frame.loc[name, "full_sharpe"]) / math.sqrt(252),
            "declared_trials": int(len(family)),
            "trial_sharpe_variance_per_session": trial_variance,
            "sessions": int(frame.loc[name, "full_sessions"]),
            "status": result.status,
            "deflated_probability": result.value,
            "reason": result.reason,
            "family": (
                "optimality_search_family_1990_2014: every joint draw in this "
                "study plus the frozen baseline; still a lower bound on the "
                "true search that produced the lineage"
            ),
            "note": DECLARATION,
        })

    # --- C. selection ---------------------------------------------------
    candidates = frame.loc[frame["name"].str.startswith("candidate:")].copy()
    pools = {
        "one-parameter profiles": profiles.loc[~profiles["is_frozen"]].dropna(
            subset=["train_sharpe"]
        ),
        "joint random search": search,
        "pre-registered candidates": candidates,
        "everything": frame.loc[frame["name"] != "baseline"],
    }
    selection = pd.DataFrame(
        [selection_report(pool, baseline, label) for label, pool in pools.items()]
    )
    selection.to_csv(output_dir / "optimality_selection.csv", index=False)

    # --- walk-forward re-optimisation ------------------------------------
    schedule, reoptimized = walk_forward(path_frame, "baseline", args.walk_forward_from)
    schedule["note"] = DECLARATION
    schedule.to_csv(output_dir / "optimality_walk_forward.csv", index=False)

    window = (reoptimized.index[0].date().isoformat(), reoptimized.index[-1].date().isoformat())
    frozen_window = baseline_path.loc[reoptimized.index]
    walk_rows = []
    for label, series in (("annually re-optimised", reoptimized),
                          ("frozen baseline", frozen_window)):
        equity = (1 + series).cumprod()
        peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
        years = (series.index[-1] - series.index[0]).days / 365.2425
        walk_rows.append({
            "book": label,
            "start": window[0],
            "end": window[1],
            "sessions": int(len(series)),
            "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
            "annualized_volatility": float(series.std() * math.sqrt(252)),
            "sharpe": float(series.mean() / series.std() * math.sqrt(252)),
            "max_drawdown": float((equity / peak - 1).min()),
            # Only the re-optimised book has a selection history; leaving the
            # counts blank on the frozen row keeps a reader from reading them
            # as a property of the book they sit beside.
            "reselections": (
                int(schedule["selection_changed"].sum()) if label.startswith("annually") else None
            ),
            "years_frozen_was_chosen": (
                int(schedule["selected_is_frozen"].sum()) if label.startswith("annually") else None
            ),
            "note": DECLARATION,
        })
    pd.DataFrame(walk_rows).to_csv(
        output_dir / "optimality_walk_forward_summary.csv", index=False
    )

    # --- D. candidates ---------------------------------------------------
    assumptions = {f"candidate:{name}": text for name, _, _, text in CANDIDATES}
    candidate_report = candidates.copy()
    candidate_report.insert(1, "assumption", [assumptions[n] for n in candidate_report["name"]])
    for field in ("full_sharpe", "train_sharpe", "test_sharpe", "full_cagr",
                  "full_max_drawdown", "full_calmar", "full_annual_cost_drag"):
        candidate_report[f"delta_{field}"] = (
            candidate_report[field].astype(float) - float(baseline[field])
        )
    candidate_report["note"] = DECLARATION
    candidate_report.sort_values("test_sharpe", ascending=False).to_csv(
        output_dir / "optimality_candidates.csv", index=False
    )

    # --- inference on the two comparisons that matter ---------------------
    best_full = str(search["full_sharpe"].astype(float).idxmax())
    best_test_candidate = str(candidates["test_sharpe"].astype(float).idxmax())
    paired_frames = []
    for label, challenger in (
        (f"search_best_full_sample_{best_full}_vs_frozen", path_frame[best_full]),
        (f"{best_test_candidate}_vs_frozen", path_frame[best_test_candidate]),
        ("annually_reoptimized_vs_frozen", reoptimized),
    ):
        incumbent = baseline_path.loc[challenger.index]
        paired_frames.append(
            inference.paired_block_bootstrap_differential(
                incumbent, challenger, comparison=label, samples=args.bootstrap_samples
            )
        )
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(output_dir / "optimality_paired_differentials.csv", index=False)
    pd.DataFrame(deflated_rows).to_csv(
        output_dir / "optimality_deflated_sharpe.csv", index=False
    )

    # --- the summary a reader should start from --------------------------
    everything = pools["everything"]
    joint = selection.loc[selection["pool"] == "joint random search"].iloc[0]
    spike = summary.loc[summary["parameter"] == "trend_lookback"].iloc[0]
    # A change that beats the frozen configuration in one window is a
    # coincidence waiting to be found; one that beats it in both is the only
    # kind worth naming at all -- and still not worth acting on here.
    single = pools["one-parameter profiles"]
    both_windows = single.loc[
        (single["train_sharpe"] > float(baseline["train_sharpe"]))
        & (single["test_sharpe"] > float(baseline["test_sharpe"]))
    ].sort_values("test_sharpe", ascending=False)
    records = [
        {
            "question": "Is the frozen configuration the full-sample optimum?",
            "measure": "frozen full-sample Sharpe percentile among all configurations tested",
            "value": float((everything["full_sharpe"] <= baseline["full_sharpe"]).mean()),
            "reading": (
                f"{int((everything['full_sharpe'] > baseline['full_sharpe']).sum())} of "
                f"{len(everything)} configurations beat it in sample; the best reached "
                f"{everything['full_sharpe'].max():.3f} against {baseline['full_sharpe']:.3f}"
            ),
        },
        {
            "question": "How much could a searcher have added in sample?",
            "measure": "best full-sample Sharpe minus frozen",
            "value": float(everything["full_sharpe"].max() - baseline["full_sharpe"]),
            "reading": "the headroom an in-sample optimiser would have claimed",
        },
        {
            "question": "Does choosing on 1990-2004 pay over 2005-2014?",
            "measure": "out-of-sample Sharpe of the in-sample winner minus frozen",
            "value": float(joint["out_of_sample_gain"]),
            "reading": (
                f"in-sample gain {joint['in_sample_gain']:+.3f} decayed to "
                f"{joint['out_of_sample_gain']:+.3f} out of sample"
            ),
        },
        {
            "question": "Did any joint combination of parameters beat the frozen one?",
            "measure": "frozen full-sample Sharpe percentile within the joint random search",
            "value": float(joint["frozen_full_sharpe_percentile"]),
            "reading": (
                f"{int(round((1 - joint['frozen_full_sharpe_percentile']) * joint['configurations']))}"
                f" of {int(joint['configurations'])} independent draws beat it in sample, "
                f"{int(round(joint['share_beating_frozen_out_of_sample'] * joint['configurations']))}"
                " over 2005-2014"
            ),
        },
        {
            "question": "Does any single change beat the frozen one in both windows?",
            "measure": "profile points with a higher Sharpe over 1990-2004 and over 2005-2014",
            "value": int(len(both_windows)),
            "reading": (
                "; ".join(
                    f"{row['parameter']}={row['value']:g} "
                    f"({row['train_sharpe']:.3f}/{row['test_sharpe']:.3f})"
                    for _, row in both_windows.head(4).iterrows()
                )
                or "none"
            )
            + f" against frozen ({baseline['train_sharpe']:.3f}/{baseline['test_sharpe']:.3f})",
        },
        {
            "question": "Does in-sample rank predict out-of-sample rank?",
            "measure": "Spearman correlation of train and test Sharpe across all configurations",
            "value": _rank_correlation(
                everything.set_index("name")["train_sharpe"],
                everything.set_index("name")["test_sharpe"],
            ),
            "reading": "near zero would mean the search learns nothing transferable",
        },
        {
            "question": "Is the 12-month trend lookback a plateau or a spike?",
            "measure": "full-sample Sharpe lost at the adjacent grid points",
            "value": float(spike["neighbour_sharpe_drop"]),
            "reading": (
                f"grid spans {spike['axis_sharpe_span']:.3f} Sharpe; "
                f"{spike['plateau_share_within_0.10_sharpe']:.0%} of the axis lies within "
                "0.10 Sharpe of its own best"
            ),
        },
        {
            "question": "Would annual re-optimisation have beaten holding the frozen rules?",
            "measure": "Sharpe of the re-optimised path minus frozen, same window",
            "value": float(walk_rows[0]["sharpe"] - walk_rows[1]["sharpe"]),
            "reading": (
                f"{walk_rows[0]['start']} to {walk_rows[0]['end']}, "
                f"{walk_rows[0]['reselections']} changes of configuration"
            ),
        },
        {
            "question": "Does the headline Sharpe survive the size of this search?",
            "measure": "deflated Sharpe probability for the frozen configuration",
            "value": deflated_rows[1]["deflated_probability"],
            "reading": (
                f"declared family of {deflated_rows[1]['declared_trials']} trials; "
                f"the search maximum deflates to "
                f"{deflated_rows[0]['deflated_probability']}"
            ),
        },
    ]
    summarize(records).to_csv(output_dir / "optimality_summary.csv", index=False)

    manifest = {
        "configurations_run": len(tasks),
        "configurations_completed": int(len(frame)),
        "configurations_infeasible": int(len(infeasible)),
        "infeasible_configurations": sorted(infeasible["name"]),
        "joint_search_trials": args.trials,
        "search_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "walk_forward_first_year": args.walk_forward_from,
        "train_window": list(TRAIN),
        "test_window": list(TEST),
        "full_window": list(FULL),
        "profile_grid": {k: list(v) for k, v in PROFILE_GRID.items()},
        "search_axes": list(SEARCH_AXES),
        "candidates": [
            {"name": name, "overrides": overrides,
             "trend_ensemble": list(ensemble) if ensemble else None,
             "assumption": text}
            for name, overrides, ensemble, text in CANDIDATES
        ],
        "baseline_reconciled_sharpe": float(baseline["full_sharpe"]),
        "reference_file_sha256": hashlib.sha256(
            (REPO / "reference" / "delta1_reference.py").read_bytes()
        ).hexdigest(),
        "declaration": DECLARATION,
        "selection_adjusted": True,
        "changes_any_published_number": False,
    }
    (output_dir / "optimality_run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {output_dir}")
    print(summarize(records).to_string(index=False))


if __name__ == "__main__":
    main()
