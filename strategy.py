"""Audited DELTA1 strategy: 61-market trend plus basis momentum.

This module contains only the adopted v2.7 strategy.  It is intentionally
self-contained: data loading, causal forecasts, risk sizing, execution,
costs, reporting, and the command-line entrypoint live here without imports
from any earlier research strategy.

The default book combines a 12-month sign trend with basis momentum at equal
risk weight, applies causal per-market risk management, assigns equal nominal
pre-forecast volatility budgets to available instruments, and targets 10%
annualized portfolio volatility.
The canonical ledger launches with $1 million and zero positions on
1990-01-01, using earlier observations only to warm causal estimates.
Month-end targets are queued for each market's next observed, positive-volume
session.  The model fills at that session's close, carries actual contract
quantities in a self-financing USD NAV ledger, and charges both strategy
turnover and approximate two-leg continuous-contract roll turnover.

The construction was selected retrospectively for the 1990-2004 research
window.  Under the corrected canonical engine its naive daily Sharpe is just
below 2.0; no historical point estimate is a forward guarantee.
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


STRATEGY_NAME = "Audited Global TSMOM + Basis Momentum (61 markets)"
ENGINE_VERSION = "2.7.0"
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
    "1990-1997 selected subperiod A": ("1990-01-01", "1997-12-31"),
    "1998-2004 selected subperiod B": ("1998-01-01", "2004-12-31"),
    "1990-2004 selected window": ("1990-01-01", "2004-12-31"),
    "2005-2014 reused later diagnostic": ("2005-01-01", "2014-12-31"),
    "1990-2014 full post-launch history": ("1990-01-01", "2014-12-31"),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the retrospectively selected, audited v2.7 strategy."""

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
    min_risk_scalar: float = 0.25
    max_risk_scalar: float = 2.00
    risk_budget: str = "flat"
    risk_managed_window: int | None = 63
    risk_managed_cap: float = 2.00
    no_trade_buffer: float = 0.25

    half_spread_ticks: float = 0.50
    commission_per_contract: float = 2.50
    initial_capital: float = 1_000_000.0
    launch_date: str | None = "1990-01-01"
    integer_contracts: bool = True
    execution_timing: str = "next_close"
    charge_roll_costs: bool = True
    max_rebalance_participation: float | None = 0.05
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
        if not 0 <= self.min_risk_scalar <= self.max_risk_scalar:
            raise ValueError("invalid risk-scalar bounds")
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
        if self.initial_capital <= 0 or not np.isfinite(self.initial_capital):
            raise ValueError("initial_capital must be finite and positive")
        if self.launch_date is not None:
            try:
                pd.Timestamp(self.launch_date)
            except (TypeError, ValueError) as error:
                raise ValueError("launch_date must be an ISO-like date or None") from error
        if self.execution_timing not in {"next_open", "next_close"}:
            raise ValueError("execution_timing must be 'next_open' or 'next_close'")
        if (
            self.max_rebalance_participation is not None
            and not 0 < self.max_rebalance_participation <= 1
        ):
            raise ValueError("max_rebalance_participation must be in (0, 1]")


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    positions: pd.DataFrame
    target_positions: pd.DataFrame
    signals: pd.DataFrame
    prices: pd.DataFrame
    metadata: pd.DataFrame
    trades: pd.DataFrame
    roll_events: pd.DataFrame
    executed_rolls: pd.DataFrame


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


def load_observed_prices(
    data_dir: Path,
    column: str,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load unfilled back-adjusted OHLC observations for execution checks."""
    if column not in {"Open", "Close"}:
        raise ValueError("column must be 'Open' or 'Close'")
    columns = [
        _load_column(
            Path(data_dir) / "Futures Data" / f"&{symbol}_CCB.csv",
            column,
            symbol,
        )
        for symbol in (symbols or strategy_symbols())
    ]
    frame = pd.concat(columns, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(frame.index.min(), frame.index.max())
    return frame.reindex(calendar)


def load_unadjusted_prices(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
    ffill_limit: int = 10,
) -> pd.DataFrame:
    """Load raw active-contract closes for roll gaps and economic valuation."""
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


def load_delivery_months(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load raw vendor delivery-month labels without forward filling."""
    columns = [
        _load_column(
            Path(data_dir) / "Futures Data" / f"&{symbol}.csv",
            "Delivery Month",
            symbol,
        )
        for symbol in (symbols or strategy_symbols())
    ]
    frame = pd.concat(columns, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(frame.index.min(), frame.index.max())
    return frame.reindex(calendar)


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
    for column in ("tick_size", "point_value", "margin"):
        catalogue[column] = pd.to_numeric(catalogue[column], errors="coerce")
    missing = wanted.difference(catalogue.index)
    if missing:
        raise ValueError(f"Missing catalogue rows for: {sorted(missing)}")
    unconvertible = set(catalogue["currency"]) - {"USD"} - set(FX_SOURCE)
    if unconvertible:
        raise ValueError(f"No FX conversion source for: {sorted(unconvertible)}")
    terms = catalogue[["tick_size", "point_value", "margin"]]
    if not np.isfinite(terms.to_numpy(dtype=float)).all() or not terms.gt(0).all().all():
        raise ValueError("Contract tick size, point value, and margin must be positive")
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


def usd_margin_values(
    metadata: pd.DataFrame,
    fx_rates: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert the catalogue's static margin snapshot to USD."""
    return pd.DataFrame(
        {
            symbol: fx_rates[row["currency"]].reindex(calendar) * row["margin"]
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
    open_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Scale forecasts by trailing volatility of their own market-level P&L.

    With observed opens, yesterday's forecast earns only today's open-to-close
    move while the older forecast earns the intervening overnight move.  The
    fallback uses a conservative full-bar delay.  Either representation is
    causal for a forecast calculated after the close.
    """
    daily_vol = prices.diff().ewm(
        span=vol_span, min_periods=vol_span, adjust=False
    ).std()
    if open_prices is None:
        strategy_return = (
            forecast.shift(2)
            * prices.diff()
            / daily_vol.shift(2).replace(0, np.nan)
        )
    else:
        opens = open_prices.reindex_like(prices)
        overnight = opens - prices.shift(1)
        intraday = prices - opens
        split_return = (
            forecast.shift(2)
            * overnight
            / daily_vol.shift(2).replace(0, np.nan)
            + forecast.shift(1)
            * intraday
            / daily_vol.shift(1).replace(0, np.nan)
        )
        closed_session_return = (
            forecast.shift(2)
            * prices.diff()
            / daily_vol.shift(2).replace(0, np.nan)
        )
        strategy_return = split_return.where(opens.notna(), closed_session_return)
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


@dataclass
class _ExecutionLedger:
    daily: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    executed_rolls: pd.DataFrame


def roll_event_mask(
    delivery_months: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every vendor contract switch plus a data-quality diagnostic.

    The continuous price path follows the vendor's active-contract labels, so
    every observed label change is charged as a roll, including a reversal.
    The second mask flags suspicious short reversals/backward switches for
    review only; it never suppresses transaction costs.
    """
    rolls = pd.DataFrame(False, index=delivery_months.index, columns=delivery_months.columns)
    anomalies = rolls.copy()
    for symbol in delivery_months:
        observed = delivery_months[symbol].dropna().astype(int)
        if observed.empty:
            continue
        raw_changes = observed.ne(observed.shift())
        raw_changes.iloc[0] = False
        rolls.loc[observed.index, symbol] = raw_changes.to_numpy()

        # Diagnose likely vendor-label glitches without changing the execution
        # path or using future information to avoid their trading costs.
        cleaned = observed.copy()
        while True:
            run_id = cleaned.ne(cleaned.shift()).cumsum()
            runs = [group for _, group in cleaned.groupby(run_id)]
            repaired = False
            for run_number in range(1, len(runs) - 1):
                run = runs[run_number]
                if (
                    len(run) <= 5
                    and runs[run_number - 1].iloc[-1]
                    == runs[run_number + 1].iloc[0]
                ):
                    cleaned.loc[run.index] = runs[run_number - 1].iloc[-1]
                    anomalies.at[run.index[0], symbol] = True
                    repaired = True
                    break
            if not repaired:
                break

        values = cleaned.to_numpy()
        accepted = values[0]
        for offset in range(1, len(values)):
            date = observed.index[offset]
            value = values[offset]
            if value == values[offset - 1]:
                continue
            if value > accepted:
                accepted = value
            elif value < accepted:
                anomalies.at[date, symbol] = True
    return rolls, anomalies


def _simulate_execution(
    monthly_targets_per_dollar: pd.DataFrame,
    prices: pd.DataFrame,
    valuation_prices: pd.DataFrame,
    observed_opens: pd.DataFrame,
    observed_closes: pd.DataFrame,
    volumes: pd.DataFrame,
    point_values: pd.DataFrame,
    one_way_costs: pd.DataFrame,
    margin_per_contract: pd.DataFrame,
    rolls: pd.DataFrame,
    config: StrategyConfig,
    *,
    initial_capital: float,
    integer_contracts: bool,
    charge_costs: bool,
    launch_date: str | None = None,
) -> _ExecutionLedger:
    """Execute queued targets in a self-financing USD futures NAV ledger."""
    index = prices.index
    columns = prices.columns
    aligned = [
        valuation_prices.reindex(index=index, columns=columns),
        observed_opens.reindex(index=index, columns=columns),
        observed_closes.reindex(index=index, columns=columns),
        volumes.reindex(index=index, columns=columns),
        point_values.reindex(index=index, columns=columns),
        one_way_costs.reindex(index=index, columns=columns),
        margin_per_contract.reindex(index=index, columns=columns),
        rolls.reindex(index=index, columns=columns).fillna(False),
    ]
    (
        valuation_close,
        opens,
        raw_closes,
        volume,
        point_value,
        one_way_cost,
        margin,
        roll_frame,
    ) = aligned
    target_rows = monthly_targets_per_dollar.reindex(columns=columns)

    close_values = prices.to_numpy(dtype=float)
    valuation_close_values = valuation_close.to_numpy(dtype=float)
    open_values = opens.to_numpy(dtype=float)
    raw_close_values = raw_closes.to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    point_value_values = point_value.to_numpy(dtype=float)
    one_way_values = one_way_cost.to_numpy(dtype=float)
    margin_values = margin.to_numpy(dtype=float)
    roll_values = roll_frame.to_numpy(dtype=bool)

    n_dates, n_markets = close_values.shape
    positions = np.zeros((n_dates, n_markets), dtype=float)
    trades = np.zeros_like(positions)
    executed_rolls = np.zeros((n_dates, n_markets), dtype=bool)
    gross_pnl = np.zeros(n_dates)
    costs = np.zeros(n_dates)
    nav_path = np.zeros(n_dates)
    rebalance_turnover = np.zeros(n_dates)
    roll_turnover_increment = np.zeros(n_dates)
    total_turnover = np.zeros(n_dates)
    gross_notional = np.zeros(n_dates)
    static_margin = np.zeros(n_dates)
    max_participation = np.zeros(n_dates)
    filled_markets = np.zeros(n_dates, dtype=int)
    pending_markets = np.zeros(n_dates, dtype=int)

    quantity = np.zeros(n_markets, dtype=float)
    pending_target = np.full(n_markets, np.nan)
    pending_quantity = np.full(n_markets, np.nan)
    pending_roll = np.zeros(n_markets, dtype=bool)
    nav = float(initial_capital)
    launch_timestamp = pd.Timestamp(launch_date) if launch_date is not None else None
    targets_by_date = {
        date: row.fillna(0.0).to_numpy(dtype=float)
        for date, row in target_rows.iterrows()
    }
    if config.max_rebalance_participation is None:
        rebalance_capacity = np.full((n_dates, n_markets), np.inf)
    else:
        trailing_volume = volume.fillna(0.0).rolling(
            config.volume_gate_window,
            min_periods=config.volume_gate_window,
        ).median().shift(1)
        capacity_volume = trailing_volume
        if config.execution_timing == "next_close":
            capacity_volume = trailing_volume.clip(upper=volume.fillna(0.0))
        rebalance_capacity = (
            capacity_volume.to_numpy(dtype=float)
            * config.max_rebalance_participation
        )
        if integer_contracts:
            rebalance_capacity = np.floor(rebalance_capacity)

    for i, date in enumerate(index):
        if launch_timestamp is not None and date < launch_timestamp:
            nav_path[i] = nav
            positions[i] = quantity
            continue
        first_live_row = (
            launch_timestamp is not None
            and (i == 0 or index[i - 1] < launch_timestamp)
        )
        decision_date = None
        if i > 0 and index[i - 1] in targets_by_date:
            decision_date = index[i - 1]
        elif first_live_row:
            earlier_decisions = [
                target_date for target_date in targets_by_date if target_date < date
            ]
            if earlier_decisions:
                decision_date = max(earlier_decisions)
        if decision_date is not None:
            pending_target = targets_by_date[decision_date].copy()
            pending_quantity = pending_target * nav
            if integer_contracts:
                pending_quantity = np.rint(pending_quantity)
            no_change = np.isclose(
                pending_quantity,
                quantity,
                atol=1e-12,
                rtol=0.0,
            )
            pending_target[no_change] = np.nan
            pending_quantity[no_change] = np.nan
        pending_roll |= roll_values[i]

        starting_nav = nav
        old_quantity = quantity.copy()
        actual_session = (
            np.isfinite(raw_close_values[i])
            & (volume_values[i] > 0)
        )
        if config.execution_timing == "next_open":
            actual_session &= np.isfinite(open_values[i])

        if i == 0:
            close_change = np.zeros(n_markets)
        else:
            close_change = close_values[i] - close_values[i - 1]

        held = old_quantity != 0
        if held.any():
            required = close_change[held]
            required_pv = point_value_values[i, held]
            if not np.isfinite(required).all() or not np.isfinite(required_pv).all():
                raise ValueError(f"Missing held settlement or FX value on {date.date()}")

        if config.execution_timing == "next_open":
            overnight_change = close_change.copy()
            if i > 0:
                overnight_change[actual_session] = (
                    open_values[i, actual_session]
                    - close_values[i - 1, actual_session]
                )
            overnight_pnl = np.where(
                held,
                old_quantity * overnight_change * point_value_values[i],
                0.0,
            )
            pre_trade_nav = nav + float(np.sum(overnight_pnl))
        else:
            full_pnl = np.where(
                held,
                old_quantity * close_change * point_value_values[i],
                0.0,
            )
            pre_trade_nav = nav + float(np.sum(full_pnl))

        if not np.isfinite(pre_trade_nav) or pre_trade_nav <= 0:
            raise ValueError(f"Portfolio NAV is non-positive or non-finite on {date.date()}")

        fill = actual_session & np.isfinite(pending_quantity)
        desired = pending_quantity
        desired_change = desired - old_quantity
        executable_change = np.clip(
            desired_change,
            -rebalance_capacity[i],
            rebalance_capacity[i],
        )
        executable_change = np.where(np.isfinite(executable_change), executable_change, 0.0)
        quantity[fill] = old_quantity[fill] + executable_change[fill]
        completed = fill & np.isclose(quantity, desired, atol=1e-12, rtol=0.0)
        pending_target[completed] = np.nan
        pending_quantity[completed] = np.nan
        trade = quantity - old_quantity
        trades[i] = trade
        filled_markets[i] = int(np.count_nonzero(trade))
        pending_markets[i] = int(np.isfinite(pending_target).sum())

        if config.execution_timing == "next_open":
            intraday_change = np.zeros(n_markets)
            intraday_change[actual_session] = (
                close_values[i, actual_session] - open_values[i, actual_session]
            )
            new_held = quantity != 0
            if new_held.any():
                if (
                    not np.isfinite(intraday_change[new_held]).all()
                    or not np.isfinite(point_value_values[i, new_held]).all()
                ):
                    raise ValueError(f"Missing held intraday or FX value on {date.date()}")
            intraday_pnl = np.where(
                new_held,
                quantity * intraday_change * point_value_values[i],
                0.0,
            )
            day_gross_pnl = float(np.sum(overnight_pnl) + np.sum(intraday_pnl))
        else:
            day_gross_pnl = float(np.sum(full_pnl))

        regular_turnover = np.abs(trade)
        processed_roll = pending_roll & actual_session
        roll_today = processed_roll & ((old_quantity != 0) | (quantity != 0))
        executed_rolls[i] = roll_today
        roll_adjusted_turnover = np.where(
            roll_today,
            np.abs(old_quantity) + np.abs(quantity),
            regular_turnover,
        )
        incremental_roll = roll_adjusted_turnover - regular_turnover
        pending_roll[processed_roll] = False
        charged_turnover = (
            roll_adjusted_turnover
            if config.charge_roll_costs
            else regular_turnover
        )
        if np.any((charged_turnover > 0) & ~np.isfinite(one_way_values[i])):
            raise ValueError(f"Missing transaction-cost input on {date.date()}")
        day_cost = (
            float(
                np.sum(
                    np.where(
                        charged_turnover > 0,
                        charged_turnover * one_way_values[i],
                        0.0,
                    )
                )
            )
            if charge_costs
            else 0.0
        )
        nav = starting_nav + day_gross_pnl - day_cost
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"Portfolio NAV is non-positive or non-finite on {date.date()}")

        end_held = quantity != 0
        if end_held.any():
            close_now = valuation_close_values[i, end_held]
            pv_now = point_value_values[i, end_held]
            margin_now = margin_values[i, end_held]
            if (
                not np.isfinite(close_now).all()
                or not np.isfinite(pv_now).all()
                or not np.isfinite(margin_now).all()
            ):
                raise ValueError(f"Missing held valuation input on {date.date()}")
            gross_notional[i] = float(
                np.sum(np.abs(quantity[end_held] * close_now * pv_now))
            )
            static_margin[i] = float(
                np.sum(np.abs(quantity[end_held]) * margin_now)
            )

        participation = np.divide(
            charged_turnover,
            volume_values[i],
            out=np.zeros(n_markets),
            where=(charged_turnover > 0) & (volume_values[i] > 0),
        )
        max_participation[i] = float(np.max(participation))
        gross_pnl[i] = day_gross_pnl
        costs[i] = day_cost
        nav_path[i] = nav
        rebalance_turnover[i] = float(np.sum(regular_turnover))
        roll_turnover_increment[i] = float(np.sum(incremental_roll))
        total_turnover[i] = float(np.sum(charged_turnover))
        positions[i] = quantity

    prior_nav = np.r_[initial_capital, nav_path[:-1]]
    daily = pd.DataFrame(
        {
            "gross_pnl_usd": gross_pnl,
            "transaction_cost_usd": costs,
            "net_pnl_usd": gross_pnl - costs,
            "gross_return": gross_pnl / prior_nav,
            "cost": costs / prior_nav,
            "net_return": (gross_pnl - costs) / prior_nav,
            "nav": nav_path,
            "equity": nav_path / initial_capital,
            "rebalance_contract_turnover": rebalance_turnover,
            "roll_contract_turnover_increment": roll_turnover_increment,
            "total_contract_turnover": total_turnover,
            "gross_notional_usd": gross_notional,
            "gross_notional_multiple": gross_notional / nav_path,
            "static_margin_requirement_usd": static_margin,
            "static_margin_fraction": static_margin / nav_path,
            "max_order_participation": max_participation,
            "filled_markets": filled_markets,
            "pending_markets": pending_markets,
        },
        index=index,
    )
    daily["gross_equity"] = (1 + daily["gross_return"]).cumprod()
    return _ExecutionLedger(
        daily=daily,
        positions=pd.DataFrame(positions, index=index, columns=columns),
        trades=pd.DataFrame(trades, index=index, columns=columns),
        executed_rolls=pd.DataFrame(executed_rolls, index=index, columns=columns),
    )


def ewma_portfolio_risk_scalar(
    base_gross_returns: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    """RiskMetrics EWMA volatility-target multiplier, not gross leverage."""
    realized_vol = base_gross_returns.ewm(
        alpha=1 - config.vol_decay,
        min_periods=config.vol_estimator_min_periods,
    ).std() * math.sqrt(config.annualization)
    ratio = config.target_vol / realized_vol.replace(0, np.nan)
    return ratio.clip(config.min_risk_scalar, config.max_risk_scalar).fillna(1.0)


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
    observed_opens: pd.DataFrame | None = None,
    observed_closes: pd.DataFrame | None = None,
    unadjusted: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    delivery_months: pd.DataFrame | None = None,
    fx_rates: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the canonical strategy through forecast, risk, and audited execution."""
    config.validate()
    symbols = strategy_symbols()
    supplied_prices = prices is not None
    prices = prices if prices is not None else load_prices(
        config.data_dir, symbols, config.price_ffill_limit
    )
    observed_opens = (
        observed_opens
        if observed_opens is not None
        else (
            prices.copy()
            if supplied_prices
            else load_observed_prices(config.data_dir, "Open", symbols)
        )
    )
    observed_closes = (
        observed_closes
        if observed_closes is not None
        else (
            prices.copy()
            if supplied_prices
            else load_observed_prices(config.data_dir, "Close", symbols)
        )
    )
    unadjusted = unadjusted if unadjusted is not None else load_unadjusted_prices(
        config.data_dir, symbols, config.price_ffill_limit
    )
    metadata = metadata if metadata is not None else load_metadata(config.data_dir, symbols)
    volumes = volumes if volumes is not None else load_volumes(config.data_dir, symbols)
    delivery_months = (
        delivery_months
        if delivery_months is not None
        else (
            pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
            if supplied_prices
            else load_delivery_months(config.data_dir, symbols)
        )
    )
    fx_rates = fx_rates if fx_rates is not None else load_fx_rates(
        config.data_dir, prices.index
    )
    point_values = usd_point_values(metadata, fx_rates, prices.index)
    margin_values = usd_margin_values(metadata, fx_rates, prices.index)
    one_way_costs = (
        config.half_spread_ticks
        * point_values.mul(metadata["tick_size"], axis=1)
        + config.commission_per_contract
    )
    rolls, delivery_anomalies = roll_event_mask(delivery_months)

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
        execution_opens = observed_opens.where(
            observed_closes.notna() & volumes.gt(0)
        )
        forecast = risk_managed_forecast(
            forecast,
            prices,
            config.vol_span,
            config.risk_managed_window,
            config.risk_managed_cap,
            execution_opens if config.execution_timing == "next_open" else None,
        )
    tradeable = tradeable_mask(
        volumes,
        config.volume_gate_window,
        config.min_median_contracts,
    ).reindex(prices.index).fillna(False)
    forecast = forecast.where(tradeable.reindex_like(forecast), np.nan)

    base_target = _base_target_positions(prices, forecast, point_values, config)
    base_monthly = _month_end_rows(base_target)
    base_execution = _simulate_execution(
        base_monthly,
        prices,
        unadjusted,
        observed_opens,
        observed_closes,
        volumes,
        point_values,
        one_way_costs,
        margin_values,
        rolls,
        config,
        initial_capital=1.0,
        integer_contracts=False,
        charge_costs=False,
    )

    risk_scalar = ewma_portfolio_risk_scalar(
        base_execution.daily["gross_return"], config
    )
    live = base_execution.positions.abs().sum(axis=1).gt(0).cummax()
    warmed_up = live & live.shift(config.portfolio_vol_window).fillna(False)
    risk_scalar = risk_scalar.where(warmed_up, 1.0)
    monthly_risk_scalar = _month_end_rows(risk_scalar)
    desired = base_monthly.mul(
        monthly_risk_scalar.reindex(base_monthly.index), axis=0
    )

    buffered = apply_no_trade_buffer(desired, config.no_trade_buffer)
    if config.launch_date is not None and not desired.empty:
        launch_timestamp = pd.Timestamp(config.launch_date)
        prior_decisions = desired.index[desired.index < launch_timestamp]
        live_decisions = desired.index[desired.index >= launch_timestamp]
        if len(prior_decisions) or len(live_decisions):
            first_live_decision = (
                prior_decisions[-1] if len(prior_decisions) else live_decisions[0]
            )
            buffered = apply_no_trade_buffer(
                desired.loc[first_live_decision:],
                config.no_trade_buffer,
            )
        else:
            buffered = desired.iloc[0:0].copy()
    execution = _simulate_execution(
        buffered,
        prices,
        unadjusted,
        observed_opens,
        observed_closes,
        volumes,
        point_values,
        one_way_costs,
        margin_values,
        rolls,
        config,
        initial_capital=config.initial_capital,
        integer_contracts=config.integer_contracts,
        charge_costs=True,
        launch_date=config.launch_date,
    )
    daily = execution.daily.copy()
    daily["risk_scalar"] = (
        monthly_risk_scalar.reindex(prices.index).ffill().shift(1).fillna(1.0)
    )
    daily["active_markets"] = execution.positions.ne(0).sum(axis=1)
    daily["delivery_label_anomalies"] = delivery_anomalies.sum(axis=1)
    return BacktestResult(
        STRATEGY_NAME,
        daily,
        execution.positions,
        buffered,
        forecast,
        prices,
        metadata,
        execution.trades,
        rolls,
        execution.executed_rolls,
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

    first_location = result.daily.index.get_indexer([returns.index[0]])[0]
    period_start = (
        result.daily.index[first_location - 1]
        if first_location > 0
        else returns.index[0]
    )
    elapsed_seconds = (returns.index[-1] - period_start).total_seconds()
    years = max(elapsed_seconds / (365.2425 * 24 * 60 * 60), 1 / 365.2425)
    equity = (1 + returns).cumprod()
    running_peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = equity / running_peak - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    volatility = returns.std() * math.sqrt(annualization)
    downside = math.sqrt(float(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2)))
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
            "Naive daily Sharpe (sqrt252, rf=0)": (
                returns.mean() / returns.std() * math.sqrt(annualization)
            ),
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
                returns.mean() * annualization
                / (downside * math.sqrt(annualization))
                if downside > 0
                else np.nan
            ),
            "Daily return autocorrelation (lag 1)": (
                returns.autocorr(1) if len(returns) >= 3 else np.nan
            ),
            "Max drawdown": drawdown.min(),
            "Calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
            "Positive months": (monthly > 0).mean(),
            "Best month": monthly.max(),
            "Worst month": monthly.min(),
            "Annual cost drag": daily["cost"].mean() * annualization,
            "Average risk scalar": daily["risk_scalar"].mean(),
            "Average gross notional multiple": daily["gross_notional_multiple"].mean(),
            "Peak gross notional multiple": daily["gross_notional_multiple"].max(),
            "Average static margin fraction": daily["static_margin_fraction"].mean(),
            "Peak static margin fraction": daily["static_margin_fraction"].max(),
            "Peak order participation": daily["max_order_participation"].max(),
            "Average markets held": result.positions.loc[start:end].ne(0).sum(axis=1).mean(),
        },
        name=result.name,
    )


def monthly_block_bootstrap_intervals(
    result: BacktestResult,
    start: str,
    end: str | None = None,
    *,
    samples: int = 20_000,
    block_months: int = 6,
    seed: int = 20_260_803,
) -> pd.Series:
    """Circular monthly-block intervals, explicitly unadjusted for selection."""
    if samples <= 0 or block_months <= 0:
        raise ValueError("samples and block_months must be positive")
    returns = result.daily.loc[start:end, "net_return"].dropna()
    monthly = ((1 + returns).resample("ME").prod() - 1).to_numpy(dtype=float)
    if len(monthly) < max(12, block_months):
        raise ValueError("At least 12 monthly returns are required")
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(monthly) / block_months)
    starts = rng.integers(0, len(monthly), size=(samples, blocks_needed))
    offsets = np.arange(block_months)
    indices = (starts[:, :, None] + offsets) % len(monthly)
    sampled = monthly[indices.reshape(samples, -1)[:, : len(monthly)]]
    standard_deviation = sampled.std(axis=1, ddof=1)
    sharpe = np.divide(
        sampled.mean(axis=1) * math.sqrt(12),
        standard_deviation,
        out=np.full(samples, np.nan),
        where=standard_deviation > 0,
    )
    cagr = np.exp(np.log1p(sampled).sum(axis=1) * 12 / len(monthly)) - 1
    return pd.Series(
        {
            "Monthly Sharpe 95% lower": np.nanquantile(sharpe, 0.025),
            "Monthly Sharpe 95% upper": np.nanquantile(sharpe, 0.975),
            "CAGR 95% lower": np.nanquantile(cagr, 0.025),
            "CAGR 95% upper": np.nanquantile(cagr, 0.975),
            "Bootstrap samples": samples,
            "Block months": block_months,
            "Selection adjusted": False,
        },
        name="Non-selection-adjusted circular block bootstrap",
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
        if label == "1990-2004 selected window":
            interval = monthly_block_bootstrap_intervals(result, start, end)
            row.update(interval.to_dict())
        else:
            row.update(
                {
                    "Monthly Sharpe 95% lower": np.nan,
                    "Monthly Sharpe 95% upper": np.nan,
                    "CAGR 95% lower": np.nan,
                    "CAGR 95% upper": np.nan,
                    "Bootstrap samples": np.nan,
                    "Block months": np.nan,
                    "Selection adjusted": False,
                }
            )
        row["CAGR point estimate >= 20%"] = bool(
            metrics["CAGR"] >= config.target_cagr
        )
        row["Naive daily Sharpe point estimate >= 2.0"] = bool(
            metrics["Naive daily Sharpe (sqrt252, rf=0)"] >= config.target_sharpe
        )
        row["Both point-estimate targets met"] = bool(
            row["CAGR point estimate >= 20%"]
            and row["Naive daily Sharpe point estimate >= 2.0"]
        )
        row["Monthly Sharpe >= 2.0"] = bool(
            metrics["Monthly Sharpe (rf=0)"] >= config.target_sharpe
        )
        row["HAC Sharpe >= 2.0"] = bool(
            metrics["HAC Sharpe (21 lags, rf=0)"] >= config.target_sharpe
        )
        row["Externally validated"] = False
        row["Multiplicity adjusted"] = False
        row["Peak modeled daily participation <= 100%"] = bool(
            metrics["Peak order participation"] <= 1.0
        )
        row["Durable 20% CAGR / 2.0 Sharpe claim validated"] = False
        rows.append(row)
    return result, pd.DataFrame(rows)


def save_outputs(
    result: BacktestResult,
    metrics: pd.DataFrame,
    config: StrategyConfig,
) -> None:
    """Save the compact research audit trail for the canonical strategy."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "strategy_metrics.csv", index=False)
    result.daily.to_csv(output_dir / "strategy_daily.csv", index_label="Date")
    result.target_positions.to_csv(
        output_dir / "strategy_monthly_targets_per_dollar.csv", index_label="Date"
    )
    held_around_roll = result.positions.ne(0) | result.positions.shift().fillna(0).ne(0)
    event_mask = result.trades.ne(0) | (
        result.roll_events.reindex_like(result.trades).fillna(False)
        & held_around_roll
    ) | result.executed_rolls
    events: list[pd.DataFrame] = []
    for date in event_mask.index[event_mask.any(axis=1)]:
        symbols = event_mask.columns[event_mask.loc[date]]
        old_contracts = result.positions.shift().fillna(0).loc[date, symbols]
        end_contracts = result.positions.loc[date, symbols]
        executed_roll = result.executed_rolls.loc[date, symbols]
        physical_turnover = pd.Series(
            np.where(
                executed_roll,
                old_contracts.abs() + end_contracts.abs(),
                result.trades.loc[date, symbols].abs(),
            ),
            index=symbols,
        )
        charged_turnover = physical_turnover.where(
            executed_roll & config.charge_roll_costs,
            result.trades.loc[date, symbols].abs(),
        )
        event = pd.DataFrame(
            {
                "Date": date,
                "Symbol": symbols,
                "Trade contracts": result.trades.loc[date, symbols].to_numpy(),
                "End contracts": end_contracts.to_numpy(),
                "Vendor roll label date": result.roll_events.loc[date, symbols].to_numpy(),
                "Executed roll": executed_roll.to_numpy(),
                "Physical contract turnover": physical_turnover.to_numpy(),
                "Charged contract turnover": charged_turnover.to_numpy(),
            }
        )
        events.append(event)
    fills = pd.concat(events, ignore_index=True) if events else pd.DataFrame(
        columns=[
            "Date",
            "Symbol",
            "Trade contracts",
            "End contracts",
            "Vendor roll label date",
            "Executed roll",
            "Physical contract turnover",
            "Charged contract turnover",
        ]
    )
    fills.to_csv(output_dir / "strategy_execution_events.csv", index=False)
    serializable = asdict(config)
    serializable["engine_version"] = ENGINE_VERSION
    serializable["data_dir"] = "${DELTA1_DATA_DIR}"
    serializable["output_dir"] = str(config.output_dir)
    serializable["optimization"] = {
        "historical_v2_6_unique_trials": CURRENT_OPTIMIZATION_TRIALS,
        "prior_archived_target_search_configurations": PRIOR_TARGET_SEARCH_TRIALS,
        "earlier_archived_variant_minimum": 81,
        "cross_round_overlap_not_deduplicated": True,
        "selected_subperiod_a": "1990-01-01/1997-12-31",
        "selected_subperiod_b": "1998-01-01/2004-12-31",
        "target_estimator": "naive daily net mean/std, rf=0, annualized by sqrt(252)",
        "selection": (
            "historical v2.6 simplicity-plateau selection; "
            "not re-optimized after v2.7 accounting fixes"
        ),
        "selected_trial_id": "OPT025",
        "historical_ledgers_recomputed_after_v2_7_engine_fix": False,
        "historical_point_metric_ledgers_retained": False,
        "externally_validated": False,
        "multiplicity_adjusted": False,
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
