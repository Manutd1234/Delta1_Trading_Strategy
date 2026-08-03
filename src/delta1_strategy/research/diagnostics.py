"""Deterministic production diagnostics for the DELTA1 strategy.

The functions in this module are deliberately independent of ``strategy`` so
that they can be called from ``strategy.save_outputs`` without a circular
import.  They report descriptive diagnostics only: no function creates a
preselected return hurdle, verdict, or claim of external validation.

``trade_metrics_report`` expects ``result.trade_episodes`` to use the
directional-episode convention.  The preferred columns are ``symbol``,
``asset_class``, ``entry_date``, ``exit_date``, ``status``, ``net_pnl_usd``,
``net_return_contribution``, and ``holding_sessions``.  Empty or missing
episode data are handled without raising.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAP_SEED = 20_260_803
DEFAULT_BLOCK_LENGTHS = (6, 12, 24)
DEFAULT_HORIZONS_MONTHS = (12, 36, 60, 120, 300)
DEFAULT_DAILY_BLOCK_LENGTHS = (21, 63, 126)
DEFAULT_DAILY_HORIZONS_SESSIONS = (2_520, 6_300)
BOOTSTRAP_QUANTILES: tuple[tuple[str, float], ...] = (
    ("P01", 0.01),
    ("P05", 0.05),
    ("P25", 0.25),
    ("Median", 0.50),
    ("P75", 0.75),
    ("P95", 0.95),
    ("P99", 0.99),
)

BOOTSTRAP_COLUMNS = [
    "Window",
    "Source start",
    "Source end",
    "Source months",
    "Method",
    "Drawdown resolution",
    "Expected block months",
    "Horizon months",
    "Samples",
    "Seed",
    "Metric",
    "Statistic",
    "Value",
    "Selection adjusted",
]

DAILY_DRAWDOWN_BOOTSTRAP_COLUMNS = [
    "Window",
    "Source start",
    "Source end",
    "Source sessions",
    "Method",
    "Drawdown resolution",
    "Expected block sessions",
    "Approximate block months",
    "Horizon sessions",
    "Approximate horizon years",
    "Samples",
    "Seed",
    "Metric",
    "Statistic",
    "Value",
    "Selection adjusted",
]

REGIME_COLUMNS = [
    "Regime",
    "State",
    "Months",
    "Fraction of labeled months",
    "Mean monthly return",
    "Median monthly return",
    "Monthly volatility",
    "Annualized conditional mean",
    "Positive month rate",
    "Worst month",
    "Average monthly cost",
    "Average gross notional multiple",
    "Average static margin fraction",
    "Average maximum order participation",
    "First labeled month",
    "Last labeled month",
    "Threshold method",
    "Minimum threshold history months",
]

TRADE_COLUMNS = [
    "Window",
    "Cohort definition",
    "Scope",
    "Group",
    "Closed episodes",
    "Open or censored episodes",
    "Entries before window",
    "Wins",
    "Losses",
    "Breakevens",
    "Win rate",
    "Loss rate",
    "Non-breakeven win rate",
    "Profit factor contribution",
    "Profit factor USD",
    "Profit factor denominator zero",
    "Expectancy bps",
    "Expectancy USD",
    "Average win bps",
    "Average loss bps",
    "Payoff ratio contribution",
    "Average win USD",
    "Average loss USD",
    "Payoff ratio USD",
    "Median outcome bps",
    "Mean holding sessions",
    "Median holding sessions",
    "Low sample",
    "Definition version",
]


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _validate_datetime_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be sorted")
    if frame.index.has_duplicates:
        raise ValueError(f"{label} index must not contain duplicates")


def _window_items(
    windows: Mapping[str, tuple[str, str | None]],
) -> list[tuple[str, tuple[str, str | None]]]:
    if not isinstance(windows, Mapping):
        raise TypeError("windows must map labels to (start, end) pairs")
    items: list[tuple[str, tuple[str, str | None]]] = []
    for label, bounds in windows.items():
        if len(bounds) != 2:
            raise ValueError(f"Window {label!r} must contain start and end")
        items.append((str(label), (bounds[0], bounds[1])))
    return items


def _stationary_indices(
    rng: np.random.Generator,
    source_length: int,
    samples: int,
    horizon: int,
    expected_block_length: int,
) -> np.ndarray:
    """Return stationary-bootstrap indices with circular continuation."""
    indices = np.empty((samples, horizon), dtype=np.int64)
    indices[:, 0] = rng.integers(0, source_length, size=samples)
    restart_probability = 1.0 / expected_block_length
    for offset in range(1, horizon):
        restart = rng.random(samples) < restart_probability
        continuation = (indices[:, offset - 1] + 1) % source_length
        fresh = rng.integers(0, source_length, size=samples)
        indices[:, offset] = np.where(restart, fresh, continuation)
    return indices


def _path_drawdown_statistics(paths: np.ndarray) -> dict[str, np.ndarray]:
    """Return close-to-close path risk statistics, including initial capital.

    Treating the initial capital of 1.0 as the first high-water mark is
    important: a loss in the first sampled return is a drawdown.  Without the
    explicit initial point, that first loss would incorrectly establish its
    own high-water mark and be reported as zero drawdown.
    """
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("paths must be a non-empty two-dimensional array")

    valid = np.all(np.isfinite(values) & (values > -1.0), axis=1)
    maximum_drawdown = np.full(values.shape[0], np.nan)
    minimum_capital_fraction = np.full(values.shape[0], np.nan)
    terminal_return = np.full(values.shape[0], np.nan)
    if np.any(valid):
        wealth = np.cumprod(1.0 + values[valid], axis=1)
        wealth_with_initial = np.concatenate(
            [np.ones((wealth.shape[0], 1)), wealth],
            axis=1,
        )
        peaks = np.maximum.accumulate(wealth_with_initial, axis=1)
        drawdowns = wealth_with_initial / peaks - 1.0
        maximum_drawdown[valid] = np.min(drawdowns, axis=1)
        minimum_capital_fraction[valid] = np.min(wealth_with_initial, axis=1)
        terminal_return[valid] = wealth[:, -1] - 1.0

    return {
        "Maximum drawdown": maximum_drawdown,
        "Minimum capital fraction": minimum_capital_fraction,
        "Terminal return": terminal_return,
    }


def _bootstrap_path_statistics(paths: np.ndarray) -> dict[str, np.ndarray]:
    horizon = paths.shape[1]
    valid = np.all(np.isfinite(paths) & (paths > -1.0), axis=1)
    log_returns = np.full_like(paths, np.nan, dtype=float)
    log_returns[valid] = np.log1p(paths[valid])

    cagr = np.full(paths.shape[0], np.nan)
    cagr[valid] = np.expm1(log_returns[valid].sum(axis=1) * 12 / horizon)

    standard_deviation = paths.std(axis=1, ddof=1)
    monthly_sharpe = np.divide(
        paths.mean(axis=1) * math.sqrt(12),
        standard_deviation,
        out=np.full(paths.shape[0], np.nan),
        where=standard_deviation > 0,
    )

    path_risk = _path_drawdown_statistics(paths)

    if horizon >= 12:
        cumulative_log = np.cumsum(log_returns, axis=1)
        leading_zero = np.zeros((paths.shape[0], 1))
        padded = np.concatenate([leading_zero, cumulative_log], axis=1)
        twelve_month_log = padded[:, 12:] - padded[:, :-12]
        worst_twelve_month = np.nanmin(np.expm1(twelve_month_log), axis=1)
    else:
        worst_twelve_month = np.full(paths.shape[0], np.nan)

    return {
        "CAGR": cagr,
        "Monthly Sharpe": monthly_sharpe,
        "Maximum drawdown": path_risk["Maximum drawdown"],
        "Minimum capital fraction": path_risk["Minimum capital fraction"],
        "Worst 12-month return": worst_twelve_month,
        "Terminal return": path_risk["Terminal return"],
    }


def monthly_stationary_bootstrap_summary(
    result: Any,
    windows: Mapping[str, tuple[str, str | None]],
    *,
    samples: int = 2_000,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    horizons_months: Sequence[int] = DEFAULT_HORIZONS_MONTHS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Return long-form stationary-bootstrap path diagnostics.

    Paths resample compounded monthly portfolio returns.  Only requested
    horizons no longer than the source history are produced.  CAGR and Sharpe
    are the primary outputs.  Drawdown statistics are observed at month ends
    only, so they are informational and can understate intramonth losses.  The
    reported uncertainty is explicitly not adjusted for strategy selection.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    blocks = tuple(int(value) for value in block_lengths)
    horizons = tuple(int(value) for value in horizons_months)
    if not blocks or any(value <= 0 for value in blocks):
        raise ValueError("block_lengths must contain positive integers")
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_months must contain positive integers")
    if not hasattr(result, "daily") or not isinstance(result.daily, pd.DataFrame):
        return _empty_frame(BOOTSTRAP_COLUMNS)
    if "net_return" not in result.daily:
        return _empty_frame(BOOTSTRAP_COLUMNS)
    _validate_datetime_index(result.daily, "result.daily")

    rows: list[dict[str, object]] = []
    for label, (start, end) in _window_items(windows):
        daily = result.daily.loc[start:end, "net_return"].dropna()
        if daily.empty:
            continue
        monthly = ((1.0 + daily).resample("ME").prod() - 1.0).dropna()
        source_length = len(monthly)
        valid_horizons = sorted({value for value in horizons if 12 <= value <= source_length})
        valid_blocks = sorted({value for value in blocks if value <= source_length})
        if not valid_horizons or not valid_blocks:
            continue

        label_seed = zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF
        source = monthly.to_numpy(dtype=float)
        for block_length in valid_blocks:
            for horizon in valid_horizons:
                combination_seed = np.random.SeedSequence(
                    [int(seed) & 0xFFFFFFFF, label_seed, block_length, horizon]
                )
                rng = np.random.default_rng(combination_seed)
                indices = _stationary_indices(
                    rng,
                    source_length,
                    samples,
                    horizon,
                    block_length,
                )
                statistics = _bootstrap_path_statistics(source[indices])
                common = {
                    "Window": label,
                    "Source start": monthly.index.min().date().isoformat(),
                    "Source end": monthly.index.max().date().isoformat(),
                    "Source months": source_length,
                    "Method": "stationary monthly bootstrap",
                    "Drawdown resolution": (
                        "month-end only; informational; may understate "
                        "intramonth drawdown"
                    ),
                    "Expected block months": block_length,
                    "Horizon months": horizon,
                    "Samples": samples,
                    "Seed": seed,
                    "Selection adjusted": False,
                }
                for metric in (
                    "CAGR",
                    "Monthly Sharpe",
                    "Maximum drawdown",
                    "Worst 12-month return",
                ):
                    values = statistics[metric]
                    for statistic, quantile in BOOTSTRAP_QUANTILES:
                        rows.append(
                            {
                                **common,
                                "Metric": metric,
                                "Statistic": statistic,
                                "Value": float(np.nanquantile(values, quantile)),
                            }
                        )

                probability_values = {
                    "Probability of terminal loss": np.mean(
                        statistics["Terminal return"] < 0
                    ),
                    "Probability capital falls below 50% of initial (diagnostic)": np.mean(
                        statistics["Minimum capital fraction"] <= 0.50
                    ),
                    "Probability maximum drawdown exceeds 15%": np.mean(
                        statistics["Maximum drawdown"] <= -0.15
                    ),
                    "Probability maximum drawdown exceeds 20%": np.mean(
                        statistics["Maximum drawdown"] <= -0.20
                    ),
                    "Probability maximum drawdown exceeds 30%": np.mean(
                        statistics["Maximum drawdown"] <= -0.30
                    ),
                    "Probability maximum drawdown exceeds 40%": np.mean(
                        statistics["Maximum drawdown"] <= -0.40
                    ),
                }
                for metric, value in probability_values.items():
                    rows.append(
                        {
                            **common,
                            "Metric": metric,
                            "Statistic": "Probability",
                            "Value": float(value),
                        }
                    )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def _stationary_daily_path_risk_samples(
    source: np.ndarray,
    *,
    samples: int,
    horizon: int,
    expected_block_length: int,
    rng: np.random.Generator,
    batch_size: int = 250,
) -> dict[str, np.ndarray]:
    """Generate daily path-risk samples without materializing every path."""
    output = {
        "Maximum drawdown": np.empty(samples, dtype=float),
        "Minimum capital fraction": np.empty(samples, dtype=float),
    }
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = _stationary_indices(
            rng,
            len(source),
            stop - start,
            horizon,
            expected_block_length,
        )
        path_risk = _path_drawdown_statistics(source[indices])
        for metric in output:
            output[metric][start:stop] = path_risk[metric]
    return output


def daily_stationary_bootstrap_drawdown_summary(
    result: Any,
    windows: Mapping[str, tuple[str, str | None]],
    *,
    samples: int = 2_000,
    block_lengths_sessions: Sequence[int] = DEFAULT_DAILY_BLOCK_LENGTHS,
    horizons_sessions: Sequence[int] = DEFAULT_DAILY_HORIZONS_SESSIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Return daily-close stationary-bootstrap drawdown diagnostics.

    The predeclared production diagnostic uses expected block lengths of
    21/63/126 trading sessions (roughly 1/3/6 months) and horizons of
    2,520/6,300 sessions (roughly 10/25 years).  Every requested combination
    is reported; the grid is not searched to select a favorable result.

    Returns are read directly from canonical ``result.daily["net_return"]``.
    Drawdowns include initial capital as the first high-water mark and retain
    daily-close resolution, so losses recovered before month end are visible.
    Only horizons no longer than the source history are produced.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    blocks = tuple(int(value) for value in block_lengths_sessions)
    horizons = tuple(int(value) for value in horizons_sessions)
    if not blocks or any(value <= 0 for value in blocks):
        raise ValueError("block_lengths_sessions must contain positive integers")
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_sessions must contain positive integers")
    if not hasattr(result, "daily") or not isinstance(result.daily, pd.DataFrame):
        return _empty_frame(DAILY_DRAWDOWN_BOOTSTRAP_COLUMNS)
    if "net_return" not in result.daily:
        return _empty_frame(DAILY_DRAWDOWN_BOOTSTRAP_COLUMNS)
    _validate_datetime_index(result.daily, "result.daily")

    rows: list[dict[str, object]] = []
    method_seed = zlib.crc32(b"stationary daily drawdown bootstrap") & 0xFFFFFFFF
    for label, (start, end) in _window_items(windows):
        daily = result.daily.loc[start:end, "net_return"].dropna()
        if daily.empty:
            continue
        source = daily.to_numpy(dtype=float)
        if not np.all(np.isfinite(source) & (source > -1.0)):
            raise ValueError("daily net returns must be finite and greater than -1")
        source_length = len(source)
        valid_horizons = sorted(
            {value for value in horizons if value <= source_length}
        )
        valid_blocks = sorted({value for value in blocks if value <= source_length})
        if not valid_horizons or not valid_blocks:
            continue

        label_seed = zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF
        for block_length in valid_blocks:
            for horizon in valid_horizons:
                combination_seed = np.random.SeedSequence(
                    [
                        int(seed) & 0xFFFFFFFF,
                        method_seed,
                        label_seed,
                        block_length,
                        horizon,
                    ]
                )
                rng = np.random.default_rng(combination_seed)
                statistics = _stationary_daily_path_risk_samples(
                    source,
                    samples=samples,
                    horizon=horizon,
                    expected_block_length=block_length,
                    rng=rng,
                )
                common = {
                    "Window": label,
                    "Source start": daily.index.min().date().isoformat(),
                    "Source end": daily.index.max().date().isoformat(),
                    "Source sessions": source_length,
                    "Method": "stationary daily bootstrap",
                    "Drawdown resolution": "daily close; includes initial capital",
                    "Expected block sessions": block_length,
                    "Approximate block months": block_length / 21.0,
                    "Horizon sessions": horizon,
                    "Approximate horizon years": horizon / 252.0,
                    "Samples": samples,
                    "Seed": seed,
                    "Selection adjusted": False,
                }
                drawdowns = statistics["Maximum drawdown"]
                for statistic, quantile in BOOTSTRAP_QUANTILES:
                    rows.append(
                        {
                            **common,
                            "Metric": "Maximum drawdown",
                            "Statistic": statistic,
                            "Value": float(np.quantile(drawdowns, quantile)),
                        }
                    )

                probability_values = {
                    "Probability capital falls below 50% of initial (diagnostic)": np.mean(
                        statistics["Minimum capital fraction"] <= 0.50
                    ),
                    "Probability maximum drawdown exceeds 15%": np.mean(
                        drawdowns <= -0.15
                    ),
                    "Probability maximum drawdown exceeds 20%": np.mean(
                        drawdowns <= -0.20
                    ),
                    "Probability maximum drawdown exceeds 30%": np.mean(
                        drawdowns <= -0.30
                    ),
                    "Probability maximum drawdown exceeds 40%": np.mean(
                        drawdowns <= -0.40
                    ),
                }
                for metric, value in probability_values.items():
                    rows.append(
                        {
                            **common,
                            "Metric": metric,
                            "Statistic": "Probability",
                            "Value": float(value),
                        }
                    )
    return pd.DataFrame(rows, columns=DAILY_DRAWDOWN_BOOTSTRAP_COLUMNS)


def _monthly_diagnostics(result: Any, start: str, end: str) -> pd.DataFrame:
    daily = result.daily.loc[start:end].copy()
    if daily.empty or "net_return" not in daily:
        return pd.DataFrame()
    monthly = pd.DataFrame(
        {"Monthly return": (1.0 + daily["net_return"]).resample("ME").prod() - 1.0}
    )
    optional_aggregations = {
        "cost": ("Average monthly cost", "sum"),
        "gross_notional_multiple": ("Average gross notional multiple", "mean"),
        "static_margin_fraction": ("Average static margin fraction", "mean"),
        "max_order_participation": (
            "Average maximum order participation",
            "max",
        ),
    }
    for source, (destination, aggregation) in optional_aggregations.items():
        if source not in daily:
            monthly[destination] = np.nan
        else:
            resampler = daily[source].resample("ME")
            monthly[destination] = getattr(resampler, aggregation)()
    return monthly


def _causal_tercile_labels(
    feature: pd.Series,
    minimum_history_months: int,
) -> pd.Series:
    labels = pd.Series(pd.NA, index=feature.index, dtype="string")
    for offset, (date, value) in enumerate(feature.items()):
        if not np.isfinite(value):
            continue
        history = feature.iloc[:offset].dropna()
        if len(history) < minimum_history_months:
            continue
        lower, upper = history.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        if value <= lower:
            labels.at[date] = "Low"
        elif value >= upper:
            labels.at[date] = "High"
        else:
            labels.at[date] = "Middle"
    return labels


def causal_regime_report(
    result: Any,
    start: str = "1990-01-01",
    end: str = "2014-12-31",
    *,
    minimum_history_months: int = 36,
) -> pd.DataFrame:
    """Report conditional monthly outcomes under strictly lagged regimes.

    Regime thresholds are expanding terciles computed only from feature values
    preceding the labeled month.  Discontiguous regimes deliberately omit
    CAGR and maximum drawdown because concatenating them would create a false
    path.
    """
    if minimum_history_months <= 0:
        raise ValueError("minimum_history_months must be positive")
    if not hasattr(result, "daily") or not isinstance(result.daily, pd.DataFrame):
        return _empty_frame(REGIME_COLUMNS)
    if "net_return" not in result.daily:
        return _empty_frame(REGIME_COLUMNS)
    _validate_datetime_index(result.daily, "result.daily")

    monthly = _monthly_diagnostics(result, start, end)
    if monthly.empty:
        return _empty_frame(REGIME_COLUMNS)

    features: dict[str, pd.Series] = {
        "Lagged realized volatility": (
            monthly["Monthly return"].rolling(12, min_periods=12).std()
            * math.sqrt(12)
        ).shift(1),
    }
    if hasattr(result, "signals") and isinstance(result.signals, pd.DataFrame):
        signals = result.signals.loc[start:end]
        if not signals.empty:
            _validate_datetime_index(signals, "result.signals")
            features["Lagged forecast magnitude"] = (
                signals.abs().mean(axis=1).resample("ME").last().shift(1)
            ).reindex(monthly.index)
    if "max_order_participation" in result.daily:
        features["Lagged liquidity pressure"] = (
            result.daily.loc[start:end, "max_order_participation"]
            .resample("ME")
            .max()
            .shift(1)
            .reindex(monthly.index)
        )
    gross_by_market = getattr(result, "gross_pnl_by_market", None)
    if isinstance(gross_by_market, pd.DataFrame) and not gross_by_market.empty:
        _validate_datetime_index(gross_by_market, "result.gross_pnl_by_market")
        if "prior_nav_usd" in result.daily:
            contributions = gross_by_market.loc[start:end].div(
                result.daily.loc[start:end, "prior_nav_usd"],
                axis=0,
            )
            correlation_values: list[float] = []
            for date in monthly.index:
                trailing = contributions.loc[:date].tail(63)
                correlation = trailing.corr(min_periods=20).to_numpy(dtype=float)
                upper = correlation[np.triu_indices_from(correlation, k=1)]
                finite = upper[np.isfinite(upper)]
                correlation_values.append(
                    float(finite.mean()) if len(finite) else np.nan
                )
            features["Lagged cross-market correlation"] = pd.Series(
                correlation_values,
                index=monthly.index,
            ).shift(1)

    rows: list[dict[str, object]] = []
    for regime, feature in features.items():
        labels = _causal_tercile_labels(feature.reindex(monthly.index), minimum_history_months)
        labeled = labels.notna()
        denominator = int(labeled.sum())
        if denominator == 0:
            continue
        for state in ("Low", "Middle", "High"):
            mask = labels.eq(state)
            observations = monthly.loc[mask]
            if observations.empty:
                continue
            returns = observations["Monthly return"].dropna()
            rows.append(
                {
                    "Regime": regime,
                    "State": state,
                    "Months": len(returns),
                    "Fraction of labeled months": len(returns) / denominator,
                    "Mean monthly return": returns.mean(),
                    "Median monthly return": returns.median(),
                    "Monthly volatility": returns.std(),
                    "Annualized conditional mean": returns.mean() * 12,
                    "Positive month rate": (returns > 0).mean(),
                    "Worst month": returns.min(),
                    "Average monthly cost": observations[
                        "Average monthly cost"
                    ].mean(),
                    "Average gross notional multiple": observations[
                        "Average gross notional multiple"
                    ].mean(),
                    "Average static margin fraction": observations[
                        "Average static margin fraction"
                    ].mean(),
                    "Average maximum order participation": observations[
                        "Average maximum order participation"
                    ].mean(),
                    "First labeled month": returns.index.min().date().isoformat(),
                    "Last labeled month": returns.index.max().date().isoformat(),
                    "Threshold method": "lagged expanding terciles",
                    "Minimum threshold history months": minimum_history_months,
                }
            )
    return pd.DataFrame(rows, columns=REGIME_COLUMNS)


def _prepare_trade_episodes(result: Any) -> pd.DataFrame:
    if not hasattr(result, "trade_episodes"):
        return pd.DataFrame()
    episodes = result.trade_episodes
    if not isinstance(episodes, pd.DataFrame) or episodes.empty:
        return pd.DataFrame()
    episodes = episodes.copy()
    aliases = {
        "Symbol": "symbol",
        "Asset class": "asset_class",
        "Entry date": "entry_date",
        "Exit date": "exit_date",
        "Status": "status",
        "Net P&L USD": "net_pnl_usd",
        "Net contribution": "net_return_contribution",
        "Entry NAV USD": "entry_nav_usd",
        "Holding sessions": "holding_sessions",
    }
    episodes = episodes.rename(
        columns={source: target for source, target in aliases.items() if source in episodes}
    )
    if "symbol" not in episodes:
        raise ValueError("trade_episodes must contain a symbol column")
    for column in ("entry_date", "exit_date"):
        if column in episodes:
            episodes[column] = pd.to_datetime(episodes[column], errors="coerce")
    if "entry_date" not in episodes:
        episodes["entry_date"] = pd.NaT
    if "exit_date" not in episodes:
        episodes["exit_date"] = pd.NaT
    if "status" not in episodes:
        episodes["status"] = np.where(episodes["exit_date"].notna(), "closed", "open")
    if "asset_class" not in episodes:
        asset_class = pd.Series("Unknown", index=episodes.index, dtype="object")
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, pd.DataFrame) and "Class" in metadata:
            asset_class = episodes["symbol"].map(metadata["Class"]).fillna("Unknown")
        episodes["asset_class"] = asset_class
    if "net_pnl_usd" not in episodes:
        episodes["net_pnl_usd"] = np.nan
    if "net_return_contribution" not in episodes:
        if "entry_nav_usd" in episodes:
            episodes["net_return_contribution"] = (
                episodes["net_pnl_usd"]
                / episodes["entry_nav_usd"].replace(0, np.nan)
            )
        else:
            episodes["net_return_contribution"] = np.nan
    if "holding_sessions" not in episodes:
        episodes["holding_sessions"] = np.nan
    return episodes


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _trade_metric_row(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scope: str,
    group: str,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    status = episodes["status"].astype(str).str.lower()
    closed = status.eq("closed") & episodes["exit_date"].notna()
    closed_in_window = closed & episodes["exit_date"].between(start, end)
    selected = episodes.loc[closed_in_window]
    active_at_end = (
        episodes["entry_date"].notna()
        & episodes["entry_date"].le(end)
        & (episodes["exit_date"].isna() | episodes["exit_date"].gt(end))
    )

    usd = pd.to_numeric(selected["net_pnl_usd"], errors="coerce")
    contribution = pd.to_numeric(
        selected["net_return_contribution"], errors="coerce"
    )
    classification = usd.where(usd.notna(), contribution)
    usd_wins = (usd.notna() & usd.gt(0.01)) | (
        usd.isna() & contribution.gt(1e-12)
    )
    usd_losses = (usd.notna() & usd.lt(-0.01)) | (
        usd.isna() & contribution.lt(-1e-12)
    )
    breakevens = classification.notna() & ~(usd_wins | usd_losses)
    closed_count = int(classification.notna().sum())
    win_count = int(usd_wins.sum())
    loss_count = int(usd_losses.sum())
    breakeven_count = int(breakevens.sum())

    # Keep each numeraire's economics internally coherent.  A contribution can
    # have a different sign from aggregate USD P&L when NAV changes during a
    # multi-day episode, so contribution PF/payoff must use contribution signs.
    contribution_wins = contribution.gt(1e-12)
    contribution_losses = contribution.lt(-1e-12)
    winning_contribution = contribution[contribution_wins]
    losing_contribution = contribution[contribution_losses]
    winning_usd = usd[usd.gt(0.01)]
    losing_usd = usd[usd.lt(-0.01)]

    contribution_gross_profit = float(winning_contribution.sum())
    contribution_gross_loss = float(-losing_contribution.sum())
    usd_gross_profit = float(winning_usd.sum())
    usd_gross_loss = float(-losing_usd.sum())
    average_win_contribution = winning_contribution.mean()
    average_loss_contribution = -losing_contribution.mean()
    average_win_usd = winning_usd.mean()
    average_loss_usd = -losing_usd.mean()

    return {
        "Window": label,
        "Cohort definition": "closed episodes grouped by exit date",
        "Scope": scope,
        "Group": group,
        "Closed episodes": closed_count,
        "Open or censored episodes": int(active_at_end.sum()),
        "Entries before window": int(
            (selected["entry_date"].notna() & selected["entry_date"].lt(start)).sum()
        ),
        "Wins": win_count,
        "Losses": loss_count,
        "Breakevens": breakeven_count,
        "Win rate": win_count / closed_count if closed_count else np.nan,
        "Loss rate": loss_count / closed_count if closed_count else np.nan,
        "Non-breakeven win rate": (
            win_count / (win_count + loss_count)
            if win_count + loss_count
            else np.nan
        ),
        "Profit factor contribution": _safe_ratio(
            contribution_gross_profit, contribution_gross_loss
        ),
        "Profit factor USD": _safe_ratio(usd_gross_profit, usd_gross_loss),
        "Profit factor denominator zero": bool(
            contribution.notna().any() and contribution_gross_loss <= 0
        ),
        "Expectancy bps": contribution.mean() * 10_000,
        "Expectancy USD": usd.mean(),
        "Average win bps": average_win_contribution * 10_000,
        "Average loss bps": average_loss_contribution * 10_000,
        "Payoff ratio contribution": _safe_ratio(
            average_win_contribution, average_loss_contribution
        ),
        "Average win USD": average_win_usd,
        "Average loss USD": average_loss_usd,
        "Payoff ratio USD": _safe_ratio(average_win_usd, average_loss_usd),
        "Median outcome bps": contribution.median() * 10_000,
        "Mean holding sessions": pd.to_numeric(
            selected["holding_sessions"], errors="coerce"
        ).mean(),
        "Median holding sessions": pd.to_numeric(
            selected["holding_sessions"], errors="coerce"
        ).median(),
        "Low sample": closed_count < 20,
        "Definition version": "directional episode v1",
    }


def trade_metrics_report(
    result: Any,
    reporting_windows: Mapping[str, tuple[str, str | None]],
) -> pd.DataFrame:
    """Summarize net directional trade episodes by exit-date cohort.

    The report contains portfolio, asset-class, and symbol rows.  Rolls and
    same-sign resizes must already remain within the same episode in
    ``result.trade_episodes``.
    """
    episodes = _prepare_trade_episodes(result)
    rows: list[dict[str, object]] = []
    for label, (raw_start, raw_end) in _window_items(reporting_windows):
        start = pd.Timestamp(raw_start)
        end = pd.Timestamp(raw_end) if raw_end is not None else pd.Timestamp.max.normalize()
        if episodes.empty:
            placeholder = pd.DataFrame(
                columns=[
                    "status",
                    "exit_date",
                    "entry_date",
                    "net_pnl_usd",
                    "net_return_contribution",
                    "holding_sessions",
                ]
            )
            rows.append(_trade_metric_row(label, start, end, "Overall", "All", placeholder))
            continue

        rows.append(_trade_metric_row(label, start, end, "Overall", "All", episodes))
        for asset_class, group_frame in episodes.groupby("asset_class", dropna=False):
            rows.append(
                _trade_metric_row(
                    label,
                    start,
                    end,
                    "Asset class",
                    str(asset_class),
                    group_frame,
                )
            )
        for symbol, group_frame in episodes.groupby("symbol", dropna=False):
            rows.append(
                _trade_metric_row(
                    label,
                    start,
                    end,
                    "Symbol",
                    str(symbol),
                    group_frame,
                )
            )
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)
