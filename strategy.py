"""Optimized DELTA1 strategy: 61-market trend plus basis momentum.

This module contains only the adopted v2.6 strategy.  It is intentionally
self-contained: data loading, causal forecasts, risk sizing, execution,
costs, reporting, and the command-line entrypoint live here without imports
from any earlier research strategy.

The default book combines a 12-month sign trend with basis momentum at equal
risk weight, applies causal per-market risk management, assigns equal nominal
pre-forecast volatility budgets to available instruments, and targets 10%
annualized portfolio volatility.
Month-end targets become active on the next business day and all reported
returns are net of the configured costs.

The construction was selected retrospectively for the 1990-2004 research
window.  Its 2.0+ Sharpe is a selected-period result, not a forward guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STRATEGY_NAME = "Optimized Global TSMOM + Basis Momentum (61 markets)"
CURRENT_OPTIMIZATION_TRIALS = 50
PRIOR_TARGET_SEARCH_TRIALS = 72

GLOBAL_ASSET_CLASSES: dict[str, tuple[str, ...]] = {
    "Equity indices": (
        "EMD", "ES", "FCE", "FDAX", "FESX", "FSMI", "HTW", "LFT",
        "NKD", "NQ", "RTY", "SNK", "SXF", "YAP", "YM",
    ),
    "Government bonds": (
        "CGB", "FGBL", "FGBM", "FGBS", "FGBX", "LLG", "SJB",
        "YXT", "YYT", "ZT", "ZF", "ZN", "ZB",
    ),
    "FX": ("6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S"),
    "Energy": ("BRN", "CL", "GAS", "HO", "NG", "RB"),
    "Metals": ("GC", "HG", "PA", "PL", "SI"),
    "Agriculture & livestock": (
        "CC", "CT", "GF", "HE", "KC", "KE", "LE", "RS",
        "SB", "ZC", "ZL", "ZM", "ZS", "ZW",
    ),
}

# USD per unit of foreign currency.  The 6J file is USD per 100 yen.
FX_SOURCE: dict[str, tuple[str, float]] = {
    "EUR": ("6E", 1.0),
    "GBP": ("6B", 1.0),
    "JPY": ("6J", 0.01),
    "CHF": ("6S", 1.0),
    "CAD": ("6C", 1.0),
    "AUD": ("6A", 1.0),
}

REPORTING_WINDOWS: dict[str, tuple[str, str]] = {
    "1980-1989 backward check": ("1980-01-01", "1989-12-31"),
    "1990-1997 discovery": ("1990-01-01", "1997-12-31"),
    "1998-2004 confirmation": ("1998-01-01", "2004-12-31"),
    "1990-2004 optimized window": ("1990-01-01", "2004-12-31"),
    "2005-2014 later stress": ("2005-01-01", "2014-12-31"),
    "1980-2014 full history": ("1980-01-01", "2014-12-31"),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the retrospectively optimized v2.6 strategy."""

    data_dir: Path
    output_dir: Path = Path("outputs")

    trend_lookback: int = 252
    vol_span: int = 60
    basis_roll_window: int = 252
    basis_lookback: int = 252
    signal_normalization_window: int = 252
    signal_cap: float = 2.0
    basis_weight: float = 0.50

    volume_gate_window: int = 60
    min_median_contracts: float = 1000.0
    price_ffill_limit: int = 10

    fast_vol_span: int = 20
    slow_vol_span: int = 120
    shock_start: float = 1.35
    shock_full: float = 2.00
    shock_floor: float = 0.75

    target_vol: float = 0.10
    vol_decay: float = 0.94
    vol_estimator_min_periods: int = 20
    portfolio_vol_window: int = 63
    min_leverage: float = 0.25
    max_leverage: float = 2.00
    risk_budget: str = "flat"
    risk_managed_window: int | None = 63
    risk_managed_cap: float = 2.00
    no_trade_buffer: float = 0.25

    half_spread_ticks: float = 0.50
    commission_per_contract: float = 2.50
    annualization: int = 252

    target_cagr: float = 0.20
    target_sharpe: float = 2.00

    def validate(self) -> None:
        if self.trend_lookback <= 0 or self.vol_span <= 1:
            raise ValueError("lookbacks and volatility spans must be positive")
        if min(self.basis_roll_window, self.basis_lookback) <= 0:
            raise ValueError("basis windows must be positive")
        if self.signal_normalization_window <= 1 or self.signal_cap <= 0:
            raise ValueError("signal normalization settings must be positive")
        if not 0 <= self.basis_weight <= 1:
            raise ValueError("basis_weight must be in [0, 1]")
        if self.volume_gate_window <= 1 or self.min_median_contracts < 0:
            raise ValueError("invalid volume gate")
        if self.fast_vol_span <= 1 or self.slow_vol_span <= self.fast_vol_span:
            raise ValueError("slow_vol_span must exceed fast_vol_span")
        if self.shock_full <= self.shock_start:
            raise ValueError("shock_full must exceed shock_start")
        if not 0 <= self.shock_floor <= 1:
            raise ValueError("shock_floor must be in [0, 1]")
        if self.target_vol <= 0 or not 0 < self.vol_decay < 1:
            raise ValueError("invalid volatility target settings")
        if not 0 <= self.min_leverage <= self.max_leverage:
            raise ValueError("invalid leverage bounds")
        if self.risk_budget not in {"asset_classes", "flat"}:
            raise ValueError("risk_budget must be 'asset_classes' or 'flat'")
        if self.risk_managed_window is not None and self.risk_managed_window <= 1:
            raise ValueError("risk_managed_window must exceed one day")
        if self.risk_managed_cap <= 0:
            raise ValueError("risk_managed_cap must be positive")
        if not 0 <= self.no_trade_buffer < 1:
            raise ValueError("no_trade_buffer must be in [0, 1)")
        if self.half_spread_ticks < 0 or self.commission_per_contract < 0:
            raise ValueError("costs cannot be negative")


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    positions: pd.DataFrame
    target_positions: pd.DataFrame
    signals: pd.DataFrame
    prices: pd.DataFrame
    metadata: pd.DataFrame


def strategy_symbols() -> list[str]:
    return [symbol for members in GLOBAL_ASSET_CLASSES.values() for symbol in members]


def _load_column(path: Path, column: str, name: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["Date", column], parse_dates=["Date"])
    frame = frame.drop_duplicates("Date", keep="last").sort_values("Date")
    return frame.set_index("Date")[column].rename(name)


def load_prices(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
    ffill_limit: int = 10,
) -> pd.DataFrame:
    """Load back-adjusted closes on a common business-day calendar."""
    columns = [
        _load_column(Path(data_dir) / "Futures Data" / f"&{symbol}_CCB.csv", "Close", symbol)
        for symbol in (symbols or strategy_symbols())
    ]
    prices = pd.concat(columns, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(prices.index.min(), prices.index.max())
    return prices.reindex(calendar).ffill(limit=ffill_limit)


def load_unadjusted_prices(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
    ffill_limit: int = 10,
) -> pd.DataFrame:
    """Load unadjusted closes used to recover realized roll gaps."""
    columns = [
        _load_column(Path(data_dir) / "Futures Data" / f"&{symbol}.csv", "Close", symbol)
        for symbol in (symbols or strategy_symbols())
    ]
    prices = pd.concat(columns, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(prices.index.min(), prices.index.max())
    return prices.reindex(calendar).ffill(limit=ffill_limit)


def load_volumes(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    columns = [
        _load_column(Path(data_dir) / "Futures Data" / f"&{symbol}_CCB.csv", "Volume", symbol)
        for symbol in (symbols or strategy_symbols())
    ]
    volumes = pd.concat(columns, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(volumes.index.min(), volumes.index.max())
    return volumes.reindex(calendar)


def load_metadata(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load contract terms and validate that every currency is convertible."""
    wanted = set(symbols or strategy_symbols())
    catalogue = pd.read_csv(Path(data_dir) / "CATALOGUE_Delta1_Futures.csv")
    catalogue["clean_symbol"] = (
        catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    )
    catalogue = catalogue[
        catalogue["symbol"].str.endswith("_CCB")
        & catalogue["clean_symbol"].isin(wanted)
    ].drop_duplicates("clean_symbol").set_index("clean_symbol")
    for column in ("tick_size", "point_value"):
        catalogue[column] = pd.to_numeric(catalogue[column], errors="coerce")
    missing = wanted.difference(catalogue.index)
    if missing:
        raise ValueError(f"Missing catalogue rows for: {sorted(missing)}")
    unconvertible = set(catalogue["currency"]) - {"USD"} - set(FX_SOURCE)
    if unconvertible:
        raise ValueError(f"No FX conversion source for: {sorted(unconvertible)}")
    return catalogue.sort_index()


def load_fx_rates(data_dir: Path, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Load point-in-time USD conversion rates from currency futures."""
    plausible = {
        "EUR": (0.5, 2.5), "GBP": (0.8, 3.0), "JPY": (0.003, 0.02),
        "CHF": (0.3, 2.0), "CAD": (0.4, 1.5), "AUD": (0.3, 1.5),
    }
    rates: dict[str, pd.Series] = {}
    for currency, (symbol, scale) in FX_SOURCE.items():
        rate = _load_column(
            Path(data_dir) / "Futures Data" / f"&{symbol}.csv", "Close", currency
        ) * scale
        low, high = plausible[currency]
        observed = rate.dropna()
        if not ((observed > low) & (observed < high)).all():
            raise ValueError(f"{currency} rate outside plausible bounds; check scaling")
        rates[currency] = rate.reindex(calendar).ffill(limit=10)
    frame = pd.DataFrame(rates, index=calendar)
    frame["USD"] = 1.0
    return frame


def usd_point_values(
    metadata: pd.DataFrame,
    fx_rates: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            symbol: fx_rates[row["currency"]].reindex(calendar) * row["point_value"]
            for symbol, row in metadata.iterrows()
        },
        index=calendar,
    )


def trend_signal(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Causal 12-month sign trend; no price level is used as a denominator."""
    difference = prices - prices.shift(lookback)
    return np.sign(difference).where(difference.notna())


def basis_momentum(
    prices: pd.DataFrame,
    unadjusted: pd.DataFrame,
    vol_span: int = 60,
    cap: float = 2.0,
    roll_window: int = 252,
    lookback: int = 252,
    normalization_window: int = 252,
) -> pd.DataFrame:
    """Causal change in trailing roll yield, normalized and clipped to [-1, 1]."""
    roll_gap = unadjusted.reindex_like(prices).diff() - prices.diff()
    daily_vol = prices.diff().ewm(
        span=vol_span, min_periods=vol_span, adjust=False
    ).std()
    roll_sum = -roll_gap.rolling(roll_window, min_periods=roll_window).sum()
    raw = (roll_sum - roll_sum.shift(lookback)) / (
        daily_vol.replace(0, np.nan) * math.sqrt(252)
    )
    scale = raw.rolling(
        normalization_window, min_periods=normalization_window
    ).std().replace(0, np.nan)
    return (raw / scale).clip(-cap, cap) / cap


def blend_signals(
    trend: pd.DataFrame,
    basis: pd.DataFrame,
    basis_weight: float = 0.50,
) -> pd.DataFrame:
    """Blend fixed sleeves, falling back to trend before basis is estimable."""
    if not 0 <= basis_weight <= 1:
        raise ValueError("basis_weight must be in [0, 1]")
    combined = trend * (1 - basis_weight) + basis.fillna(trend) * basis_weight
    return combined.clip(-1, 1).where(trend.notna())


def tradeable_mask(
    volumes: pd.DataFrame,
    window: int = 60,
    min_median_contracts: float = 1000.0,
) -> pd.DataFrame:
    trailing = volumes.fillna(0.0).rolling(window, min_periods=window).median()
    return trailing > min_median_contracts


def shock_multiplier(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Causal 20d/120d volatility-shock taper, bounded by shock_floor."""
    changes = prices.diff()
    fast = changes.ewm(
        span=config.fast_vol_span,
        min_periods=config.fast_vol_span,
        adjust=False,
    ).std()
    slow = changes.ewm(
        span=config.slow_vol_span,
        min_periods=config.slow_vol_span,
        adjust=False,
    ).std()
    ratio = fast / slow.replace(0, np.nan)
    progress = (
        (ratio - config.shock_start) / (config.shock_full - config.shock_start)
    ).clip(0, 1)
    return (1 - progress * (1 - config.shock_floor)).where(ratio.notna())


def risk_managed_forecast(
    forecast: pd.DataFrame,
    prices: pd.DataFrame,
    vol_span: int,
    window: int,
    cap: float = 2.0,
) -> pd.DataFrame:
    """Scale forecasts by trailing volatility of their own market-level P&L.

    The return proxy uses yesterday's forecast and volatility, so the scale
    applied to a forecast at date ``t`` contains no information after ``t``.
    """
    daily_vol = prices.diff().ewm(
        span=vol_span, min_periods=vol_span, adjust=False
    ).std()
    strategy_return = (
        forecast.shift(1)
        * prices.diff()
        / daily_vol.shift(1).replace(0, np.nan)
    )
    realized = strategy_return.pow(2).rolling(
        window, min_periods=window
    ).mean().pow(0.5)
    weight = (1.0 / realized.replace(0, np.nan)).clip(upper=cap)
    return (forecast * weight).clip(-1, 1).where(forecast.notna())


def _month_end_rows(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame.groupby(frame.index.to_period("M")).tail(1)


def _base_target_positions(
    prices: pd.DataFrame,
    forecast: pd.DataFrame,
    point_values: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    daily_price_vol = prices.diff().ewm(
        span=config.vol_span,
        min_periods=config.vol_span,
        adjust=False,
    ).std()
    annual_dollar_vol = daily_price_vol * point_values * math.sqrt(config.annualization)
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    risk_groups = (
        {"All markets": tuple(prices.columns)}
        if config.risk_budget == "flat"
        else GLOBAL_ASSET_CLASSES
    )
    class_weight = 1.0 / len(risk_groups)
    for members in risk_groups.values():
        symbols = list(members)
        available = forecast[symbols].notna() & annual_dollar_vol[symbols].gt(0)
        count = available.sum(axis=1).replace(0, np.nan)
        risk_budget = config.target_vol * np.sqrt(class_weight / count)
        positions[symbols] = forecast[symbols].mul(risk_budget, axis=0).div(
            annual_dollar_vol[symbols]
        )
    return positions.replace([np.inf, -np.inf], np.nan)


def _held_positions(
    monthly_targets: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    # A target formed at month-end close is active on the next business day.
    return monthly_targets.reindex(calendar).ffill().shift(1).fillna(0.0)


def _gross_returns(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    point_values: pd.DataFrame,
) -> pd.Series:
    contract_pnl = prices.diff() * point_values
    return (positions * contract_pnl).sum(axis=1, min_count=1).fillna(0.0)


def ewma_portfolio_leverage(
    base_gross_returns: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    """RiskMetrics EWMA(lambda=0.94 by default) portfolio volatility target."""
    realized_vol = base_gross_returns.ewm(
        alpha=1 - config.vol_decay,
        min_periods=config.vol_estimator_min_periods,
    ).std() * math.sqrt(config.annualization)
    ratio = config.target_vol / realized_vol.replace(0, np.nan)
    return ratio.clip(config.min_leverage, config.max_leverage).fillna(1.0)


def apply_no_trade_buffer(
    desired: pd.DataFrame,
    buffer_fraction: float = 0.25,
) -> pd.DataFrame:
    """Keep the previous target when a month-end adjustment is too small."""
    if not 0 <= buffer_fraction < 1:
        raise ValueError("buffer_fraction must be in [0, 1)")
    executable = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
    previous = pd.Series(0.0, index=desired.columns)
    for date, raw_target in desired.iterrows():
        target = raw_target.fillna(0.0)
        change = target - previous
        reference = pd.concat([target.abs(), previous.abs()], axis=1).max(axis=1)
        should_trade = change.abs() > buffer_fraction * reference
        previous = previous.where(~should_trade, target)
        executable.loc[date] = previous
    return executable


def run_backtest(
    config: StrategyConfig,
    *,
    prices: pd.DataFrame | None = None,
    unadjusted: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    fx_rates: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the canonical strategy through forecast, risk, execution, and costs."""
    config.validate()
    symbols = strategy_symbols()
    prices = prices if prices is not None else load_prices(
        config.data_dir, symbols, config.price_ffill_limit
    )
    unadjusted = unadjusted if unadjusted is not None else load_unadjusted_prices(
        config.data_dir, symbols, config.price_ffill_limit
    )
    metadata = metadata if metadata is not None else load_metadata(config.data_dir, symbols)
    volumes = volumes if volumes is not None else load_volumes(config.data_dir, symbols)
    fx_rates = fx_rates if fx_rates is not None else load_fx_rates(
        config.data_dir, prices.index
    )
    point_values = usd_point_values(metadata, fx_rates, prices.index)

    trend = trend_signal(prices, config.trend_lookback)
    basis = basis_momentum(
        prices,
        unadjusted,
        vol_span=config.vol_span,
        cap=config.signal_cap,
        roll_window=config.basis_roll_window,
        lookback=config.basis_lookback,
        normalization_window=config.signal_normalization_window,
    )
    signal = blend_signals(trend, basis, config.basis_weight)
    forecast = (signal * shock_multiplier(prices, config)).clip(-1, 1)
    if config.risk_managed_window is not None:
        forecast = risk_managed_forecast(
            forecast,
            prices,
            config.vol_span,
            config.risk_managed_window,
            config.risk_managed_cap,
        )
    tradeable = tradeable_mask(
        volumes,
        config.volume_gate_window,
        config.min_median_contracts,
    ).reindex(prices.index).fillna(False)
    forecast = forecast.where(tradeable.reindex_like(forecast), np.nan)

    base_target = _base_target_positions(prices, forecast, point_values, config)
    base_monthly = _month_end_rows(base_target)
    base_positions = _held_positions(base_monthly, prices.index)
    base_gross = _gross_returns(base_positions, prices, point_values)

    leverage = ewma_portfolio_leverage(base_gross, config)
    live = base_positions.abs().sum(axis=1).gt(0).cummax()
    warmed_up = live & live.shift(config.portfolio_vol_window).fillna(False)
    leverage = leverage.where(warmed_up, 1.0)
    monthly_leverage = _month_end_rows(leverage)
    desired = base_monthly.mul(monthly_leverage.reindex(base_monthly.index), axis=0)

    buffered = apply_no_trade_buffer(desired, config.no_trade_buffer)
    positions = _held_positions(buffered, prices.index)
    gross = _gross_returns(positions, prices, point_values)
    turnover = positions.diff().abs().fillna(positions.abs())
    one_way_cost = (
        config.half_spread_ticks
        * point_values.mul(metadata["tick_size"], axis=1)
        + config.commission_per_contract
    )
    costs = (turnover * one_way_cost).sum(axis=1)
    net = gross - costs

    daily = pd.DataFrame(
        {
            "gross_return": gross,
            "cost": costs,
            "net_return": net,
            "leverage": monthly_leverage.reindex(prices.index)
            .ffill()
            .shift(1)
            .fillna(1.0),
            "contract_turnover_per_dollar": turnover.sum(axis=1),
        }
    )
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    daily["gross_equity"] = (1 + daily["gross_return"]).cumprod()
    return BacktestResult(
        STRATEGY_NAME,
        daily,
        positions,
        buffered,
        forecast,
        prices,
        metadata,
    )


def performance_metrics(
    result: BacktestResult,
    start: str,
    end: str | None = None,
    annualization: int = 252,
) -> pd.Series:
    """Return net performance metrics, with CAGR based on elapsed calendar time."""
    daily = result.daily.loc[start:end].copy()
    returns = daily["net_return"].dropna()
    if returns.empty:
        raise ValueError(f"No returns in evaluation window for {result.name}")

    elapsed_seconds = (returns.index[-1] - returns.index[0]).total_seconds()
    years = max(elapsed_seconds / (365.2425 * 24 * 60 * 60), 1 / 365.2425)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    volatility = returns.std() * math.sqrt(annualization)
    downside = returns.clip(upper=0).std() * math.sqrt(annualization)
    monthly = (1 + returns).resample("ME").prod() - 1

    hac_lags = 21
    hac_variance = np.nan
    if len(returns) > hac_lags:
        centered = returns.to_numpy() - returns.mean()
        hac_variance = float(np.dot(centered, centered) / (len(centered) - 1))
        for lag in range(1, hac_lags + 1):
            bartlett_weight = 1 - lag / (hac_lags + 1)
            autocovariance = float(
                np.dot(centered[lag:], centered[:-lag]) / (len(centered) - 1)
            )
            hac_variance += 2 * bartlett_weight * autocovariance

    return pd.Series(
        {
            "Start": returns.index.min().date().isoformat(),
            "End": returns.index.max().date().isoformat(),
            "Years": years,
            "CAGR": cagr,
            "Annualized volatility": volatility,
            "Sharpe (rf=0)": returns.mean() / returns.std() * math.sqrt(annualization),
            "Monthly Sharpe (rf=0)": (
                monthly.mean() / monthly.std() * math.sqrt(12)
                if monthly.std() > 0
                else np.nan
            ),
            "HAC Sharpe (21 lags, rf=0)": (
                returns.mean() / math.sqrt(hac_variance) * math.sqrt(annualization)
                if np.isfinite(hac_variance) and hac_variance > 0
                else np.nan
            ),
            "Sortino (rf=0)": (
                returns.mean() / downside * annualization if downside > 0 else np.nan
            ),
            "Max drawdown": drawdown.min(),
            "Calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
            "Positive months": (monthly > 0).mean(),
            "Best month": monthly.max(),
            "Worst month": monthly.min(),
            "Annual cost drag": daily["cost"].mean() * annualization,
            "Average leverage": daily["leverage"].mean(),
            "Average markets held": result.positions.loc[start:end].abs().gt(0).sum(axis=1).mean(),
        },
        name=result.name,
    )


def run_pipeline(config: StrategyConfig) -> tuple[BacktestResult, pd.DataFrame]:
    """Run the book once and report all six fixed research windows."""
    result = run_backtest(config)
    rows: list[dict[str, object]] = []
    for label, (start, end) in REPORTING_WINDOWS.items():
        metrics = performance_metrics(result, start, end, config.annualization)
        row: dict[str, object] = {
            "Strategy": result.name,
            "Window": label,
            **metrics.to_dict(),
        }
        row["CAGR >= 20%"] = bool(metrics["CAGR"] >= config.target_cagr)
        row["Sharpe >= 2.0"] = bool(metrics["Sharpe (rf=0)"] >= config.target_sharpe)
        row["Both targets met"] = bool(
            row["CAGR >= 20%"] and row["Sharpe >= 2.0"]
        )
        rows.append(row)
    return result, pd.DataFrame(rows)


def save_outputs(
    result: BacktestResult,
    metrics: pd.DataFrame,
    config: StrategyConfig,
) -> None:
    """Save the compact production audit trail for the canonical strategy."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "strategy_metrics.csv", index=False)
    result.daily.to_csv(output_dir / "strategy_daily.csv", index_label="Date")
    result.target_positions.to_csv(
        output_dir / "strategy_monthly_targets.csv", index_label="Date"
    )
    serializable = asdict(config)
    serializable["data_dir"] = "${DELTA1_DATA_DIR}"
    serializable["output_dir"] = str(config.output_dir)
    serializable["optimization"] = {
        "current_round_unique_trials": CURRENT_OPTIMIZATION_TRIALS,
        "prior_archived_target_search_configurations": PRIOR_TARGET_SEARCH_TRIALS,
        "earlier_archived_variant_minimum": 81,
        "cross_round_overlap_not_deduplicated": True,
        "discovery_window": "1990-01-01/1997-12-31",
        "confirmation_window": "1998-01-01/2004-12-31",
        "target_estimator": "daily net mean/std, rf=0, annualized by sqrt(252)",
        "selection": "simplicity plateau among configurations clearing both targets",
        "selected_trial_id": "OPT025",
        "selected_combined_sharpe_rank": 1,
        "selected_minimum_subperiod_sharpe_rank": 5,
        "externally_validated": False,
    }
    with (output_dir / "strategy_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    result, metrics = run_pipeline(config)
    save_outputs(result, metrics, config)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
