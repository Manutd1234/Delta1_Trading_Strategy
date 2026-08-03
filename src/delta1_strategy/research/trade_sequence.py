"""Trade-sequence Monte Carlo diagnostics for DELTA1 research.

This module deliberately does not optimize a signal, choose a parameter, or
alter a position.  It answers a narrower question: how sensitive would a
stylized sequence of the observed, closed directional episodes be to the
ordering or re-sampling of those episode outcomes?

The result is not a synthetic portfolio backtest.  Directional episodes can
overlap in calendar time, share the same portfolio NAV, and close together.
Consequently, shuffling them breaks contemporaneous correlation, volatility
clustering, position-sizing feedback, and margin interactions.  The existing
stationary monthly bootstrap in :mod:`diagnostics` is the preferred account-
path diagnostic because it resamples the realized portfolio return series in
blocks.  Trade-sequence Monte Carlo is supplementary evidence about episode
order and outcome-sampling risk only.

"Risk of ruin" is reported as an empirical probability of crossing a stated
capital floor under the stylized additive-contribution accounting below.  It
is not literal insolvency probability and cannot model gaps, liquidation,
margin calls, rejected orders, or losses outside the reused historical sample.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


DEFAULT_TRADE_SEQUENCE_SEED = 20_260_803
DEFAULT_DRAWDOWN_THRESHOLDS = (0.10, 0.15, 0.20, 0.30, 0.40)
TRADE_SEQUENCE_QUANTILES: tuple[tuple[str, float], ...] = (
    ("P01", 0.01),
    ("P05", 0.05),
    ("P25", 0.25),
    ("Median", 0.50),
    ("P75", 0.75),
    ("P95", 0.95),
    ("P99", 0.99),
)

TRADE_SEQUENCE_COLUMNS = [
    "Window",
    "Source start",
    "Source end",
    "Closed episodes",
    "Peak concurrent selected episodes",
    "Maximum episodes sharing an exit date",
    "Method",
    "Replacement",
    "Accounting",
    "Horizon episodes",
    "Samples",
    "Seed",
    "Return scale",
    "Capital floor fraction",
    "Metric",
    "Statistic",
    "Value",
    "Monte Carlo standard error",
    "Confidence lower",
    "Confidence upper",
    "Confidence method",
    "Selection adjusted",
    "Dependence preserved",
    "Permitted use",
    "Limitations",
]


_COMMON_LIMITATIONS = (
    "Closed directional episodes overlap and share portfolio NAV. Resampling "
    "does not preserve simultaneous cross-market losses, sizing feedback, "
    "margin, volatility clustering, regime shifts, or unseen tail events. "
    "History is reused and results are not selection-adjusted or prospective."
)


@dataclass(frozen=True)
class TradeSequenceMonteCarloConfig:
    """Configuration for a deterministic, diagnostic-only simulation.

    ``additive contribution`` accounting starts every path at one unit of
    capital and cumulatively adds the selected episode's net return
    contribution multiplied by ``return_scale``.  It is used because episode
    contributions are slices of a common portfolio return, not independent
    fully-invested trade returns.  Compounding them would falsely treat every
    episode as a sequential whole-account investment.
    """

    samples: int = 5_000
    seed: int = DEFAULT_TRADE_SEQUENCE_SEED
    methods: tuple[Literal["permutation", "iid_bootstrap"], ...] = (
        "permutation",
        "iid_bootstrap",
    )
    return_scale: float = 1.0
    capital_floor_fraction: float = 0.50
    drawdown_thresholds: tuple[float, ...] = DEFAULT_DRAWDOWN_THRESHOLDS

    def __post_init__(self) -> None:
        if (
            isinstance(self.samples, (bool, np.bool_))
            or not isinstance(self.samples, (int, np.integer))
            or self.samples <= 0
        ):
            raise ValueError("samples must be a positive integer")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        if not self.methods:
            raise ValueError("methods must not be empty")
        allowed = {"permutation", "iid_bootstrap"}
        if any(method not in allowed for method in self.methods):
            raise ValueError(f"methods must be drawn from {sorted(allowed)}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")
        if not np.isfinite(self.return_scale) or self.return_scale <= 0:
            raise ValueError("return_scale must be finite and positive")
        if not np.isfinite(self.capital_floor_fraction) or not (
            0 <= self.capital_floor_fraction < 1
        ):
            raise ValueError("capital_floor_fraction must satisfy 0 <= floor < 1")
        thresholds = tuple(float(value) for value in self.drawdown_thresholds)
        if not thresholds or any(
            not np.isfinite(value) or not 0 < value < 1 for value in thresholds
        ):
            raise ValueError(
                "drawdown_thresholds must contain finite fractions between 0 and 1"
            )
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("drawdown_thresholds must not contain duplicates")
        object.__setattr__(self, "samples", int(self.samples))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "drawdown_thresholds", thresholds)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_SEQUENCE_COLUMNS)


def _window_items(
    windows: Mapping[str, tuple[str, str | None]],
) -> list[tuple[str, tuple[str, str | None]]]:
    if not isinstance(windows, Mapping):
        raise TypeError("windows must map labels to (start, end) pairs")
    items: list[tuple[str, tuple[str, str | None]]] = []
    for raw_label, bounds in windows.items():
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
            raise ValueError(f"Window {raw_label!r} must contain start and end")
        if len(bounds) != 2:
            raise ValueError(f"Window {raw_label!r} must contain start and end")
        start, end = bounds
        if start is None:
            raise ValueError(f"Window {raw_label!r} must have a start")
        items.append((str(raw_label), (str(start), None if end is None else str(end))))
    return items


def _prepare_episodes(result: Any) -> pd.DataFrame:
    episodes = getattr(result, "trade_episodes", None)
    if not isinstance(episodes, pd.DataFrame) or episodes.empty:
        return pd.DataFrame()
    episodes = episodes.copy()
    aliases = {
        "Symbol": "symbol",
        "Entry date": "entry_date",
        "Exit date": "exit_date",
        "Status": "status",
        "Net contribution": "net_return_contribution",
    }
    episodes = episodes.rename(
        columns={source: target for source, target in aliases.items() if source in episodes}
    )
    required = {"entry_date", "exit_date", "net_return_contribution"}
    missing = sorted(required.difference(episodes.columns))
    if missing:
        raise ValueError(f"trade_episodes missing required columns: {missing}")
    episodes["entry_date"] = pd.to_datetime(episodes["entry_date"], errors="coerce")
    episodes["exit_date"] = pd.to_datetime(episodes["exit_date"], errors="coerce")
    episodes["net_return_contribution"] = pd.to_numeric(
        episodes["net_return_contribution"], errors="coerce"
    )
    if "status" not in episodes:
        episodes["status"] = np.where(episodes["exit_date"].notna(), "closed", "open")
    return episodes


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    """Return each row's longest consecutive true run without Python row loops."""
    current = np.zeros(mask.shape[0], dtype=np.int64)
    longest = np.zeros(mask.shape[0], dtype=np.int64)
    for offset in range(mask.shape[1]):
        current = np.where(mask[:, offset], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _path_statistics(sequences: np.ndarray) -> dict[str, np.ndarray]:
    wealth = 1.0 + np.cumsum(sequences, axis=1)
    peaks = np.maximum.accumulate(
        np.concatenate([np.ones((len(sequences), 1)), wealth], axis=1), axis=1
    )[:, 1:]
    drawdown = wealth / peaks - 1.0
    return {
        "Terminal contribution return": wealth[:, -1] - 1.0,
        "Maximum drawdown": np.min(drawdown, axis=1),
        "Minimum capital fraction": np.min(wealth, axis=1),
        "Longest losing streak (episodes)": _longest_true_run(sequences < 0),
        "Longest underwater run (episodes)": _longest_true_run(wealth < peaks),
    }


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial estimate."""
    if trials <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1 + z * z / trials
    centre = (probability + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        probability * (1 - probability) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _simulate_statistics(
    contributions: np.ndarray,
    *,
    method: str,
    config: TradeSequenceMonteCarloConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Simulate in bounded chunks and return one value per path and metric."""
    observations = len(contributions)
    # Bound working arrays to roughly a few tens of MB even for long histories.
    chunk_size = max(1, min(config.samples, 1_000_000 // observations))
    parts: dict[str, list[np.ndarray]] = {}
    completed = 0
    while completed < config.samples:
        size = min(chunk_size, config.samples - completed)
        if method == "permutation":
            indices = np.empty((size, observations), dtype=np.int64)
            for row in range(size):
                indices[row] = rng.permutation(observations)
        else:
            indices = rng.integers(0, observations, size=(size, observations))
        sequences = contributions[indices] * config.return_scale
        statistics = _path_statistics(sequences)
        for metric, values in statistics.items():
            parts.setdefault(metric, []).append(values)
        completed += size
    return {metric: np.concatenate(values) for metric, values in parts.items()}


def _concurrency_diagnostics(selected: pd.DataFrame) -> tuple[int, int]:
    valid_intervals = selected.loc[
        selected["entry_date"].notna() & selected["exit_date"].notna(),
        ["entry_date", "exit_date"],
    ]
    if valid_intervals.empty:
        peak_concurrent = 0
    else:
        # Both endpoint dates are treated as active. Starts therefore precede
        # exits on a shared date. This is conservative for date-only episodes.
        starts = pd.DataFrame(
            {"date": valid_intervals["entry_date"], "change": 1, "priority": 0}
        )
        ends = pd.DataFrame(
            {"date": valid_intervals["exit_date"], "change": -1, "priority": 1}
        )
        events = pd.concat([starts, ends], ignore_index=True).sort_values(
            ["date", "priority"], kind="stable"
        )
        peak_concurrent = int(events["change"].cumsum().clip(lower=0).max())
    shared_exit = int(selected.groupby("exit_date", dropna=True).size().max())
    return peak_concurrent, shared_exit


def _common_row(
    *,
    label: str,
    selected: pd.DataFrame,
    method: str,
    config: TradeSequenceMonteCarloConfig,
    peak_concurrent: int,
    shared_exit: int,
) -> dict[str, object]:
    if method == "permutation":
        method_label = "random permutation without replacement"
        replacement = False
        limitation = (
            _COMMON_LIMITATIONS
            + " A permutation conditions on the exact realized set of outcomes; "
            "its terminal contribution is fixed and it measures ordering only."
        )
    else:
        method_label = "iid episode bootstrap with replacement"
        replacement = True
        limitation = (
            _COMMON_LIMITATIONS
            + " The iid bootstrap additionally assumes episodes are exchangeable, "
            "which can materially understate clustered or regime-dependent risk."
        )
    return {
        "Window": label,
        "Source start": selected["exit_date"].min().date().isoformat(),
        "Source end": selected["exit_date"].max().date().isoformat(),
        "Closed episodes": len(selected),
        "Peak concurrent selected episodes": peak_concurrent,
        "Maximum episodes sharing an exit date": shared_exit,
        "Method": method_label,
        "Replacement": replacement,
        "Accounting": "additive net return contribution; initial capital = 1",
        "Horizon episodes": len(selected),
        "Samples": config.samples,
        "Seed": config.seed,
        "Return scale": config.return_scale,
        "Capital floor fraction": config.capital_floor_fraction,
        "Selection adjusted": False,
        "Dependence preserved": False,
        "Permitted use": "supplementary retrospective risk diagnostic only",
        "Limitations": limitation,
    }


def _append_probability_row(
    rows: list[dict[str, object]],
    common: dict[str, object],
    *,
    metric: str,
    events: np.ndarray,
) -> None:
    successes = int(np.count_nonzero(events))
    trials = int(len(events))
    probability = successes / trials
    lower, upper = _wilson_interval(successes, trials)
    rows.append(
        {
            **common,
            "Metric": metric,
            "Statistic": "Probability",
            "Value": probability,
            "Monte Carlo standard error": math.sqrt(
                probability * (1 - probability) / trials
            ),
            "Confidence lower": lower,
            "Confidence upper": upper,
            "Confidence method": "95% Wilson score interval",
        }
    )


def trade_sequence_monte_carlo_summary(
    result: Any,
    windows: Mapping[str, tuple[str, str | None]],
    *,
    config: TradeSequenceMonteCarloConfig | None = None,
) -> pd.DataFrame:
    """Return long-form episode-sequence risk diagnostics.

    Closed episodes are selected by exit date.  For every reporting window,
    each path contains exactly as many outcomes as the selected history.
    ``permutation`` reshuffles that set without replacement; ``iid_bootstrap``
    samples it with replacement.  Random streams are stable by window and
    method, so adding a different reporting window does not change an existing
    window's results.

    The function is intentionally not a strategy-selection tool.  Results are
    selection-unadjusted reused-history diagnostics and must not be presented
    as an independent probability forecast or a deployment gate.
    """
    active_config = config or TradeSequenceMonteCarloConfig()
    episodes = _prepare_episodes(result)
    if episodes.empty:
        return _empty_summary()

    rows: list[dict[str, object]] = []
    status = episodes["status"].astype(str).str.strip().str.lower()
    valid_closed = (
        status.eq("closed")
        & episodes["exit_date"].notna()
        & episodes["net_return_contribution"].notna()
        & np.isfinite(episodes["net_return_contribution"])
    )
    closed = episodes.loc[valid_closed].sort_values(
        ["exit_date", "entry_date"], kind="stable"
    )

    for label, (raw_start, raw_end) in _window_items(windows):
        start = pd.Timestamp(raw_start)
        end = pd.Timestamp(raw_end) if raw_end is not None else pd.Timestamp.max.normalize()
        if end < start:
            raise ValueError(f"Window {label!r} end precedes start")
        selected = closed.loc[closed["exit_date"].between(start, end)].copy()
        if selected.empty:
            continue
        contributions = selected["net_return_contribution"].to_numpy(dtype=float)
        peak_concurrent, shared_exit = _concurrency_diagnostics(selected)
        label_seed = zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF

        for method in active_config.methods:
            method_seed = zlib.crc32(method.encode("utf-8")) & 0xFFFFFFFF
            seed_sequence = np.random.SeedSequence(
                [int(active_config.seed) & 0xFFFFFFFF, label_seed, method_seed]
            )
            rng = np.random.default_rng(seed_sequence)
            statistics = _simulate_statistics(
                contributions,
                method=method,
                config=active_config,
                rng=rng,
            )
            common = _common_row(
                label=label,
                selected=selected,
                method=method,
                config=active_config,
                peak_concurrent=peak_concurrent,
                shared_exit=shared_exit,
            )
            for metric, values in statistics.items():
                for statistic, quantile in TRADE_SEQUENCE_QUANTILES:
                    rows.append(
                        {
                            **common,
                            "Metric": metric,
                            "Statistic": statistic,
                            "Value": float(np.quantile(values, quantile)),
                            "Monte Carlo standard error": np.nan,
                            "Confidence lower": np.nan,
                            "Confidence upper": np.nan,
                            "Confidence method": "not estimated for quantiles",
                        }
                    )

            minimum_capital = statistics["Minimum capital fraction"]
            _append_probability_row(
                rows,
                common,
                metric="Capital-floor breach probability (risk-of-ruin proxy)",
                events=minimum_capital <= active_config.capital_floor_fraction,
            )
            maximum_drawdown = statistics["Maximum drawdown"]
            for threshold in sorted(set(active_config.drawdown_thresholds)):
                _append_probability_row(
                    rows,
                    common,
                    metric=(
                        "Maximum-drawdown breach probability "
                        f"({threshold:.0%} threshold)"
                    ),
                    events=maximum_drawdown <= -threshold,
                )

            fifth_percentile = float(np.quantile(maximum_drawdown, 0.05))
            tail = maximum_drawdown[maximum_drawdown <= fifth_percentile]
            rows.append(
                {
                    **common,
                    "Metric": "Mean maximum drawdown in worst 5% of paths",
                    "Statistic": "Expected shortfall",
                    "Value": float(tail.mean()),
                    "Monte Carlo standard error": float(
                        tail.std(ddof=1) / math.sqrt(len(tail))
                    )
                    if len(tail) > 1
                    else np.nan,
                    "Confidence lower": np.nan,
                    "Confidence upper": np.nan,
                    "Confidence method": "tail mean; no interval estimated",
                }
            )

    return pd.DataFrame(rows, columns=TRADE_SEQUENCE_COLUMNS)
