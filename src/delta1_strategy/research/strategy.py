"""Diversified global-futures trend and basis-momentum research engine.

This module contains only the selected v3.2.1 research specification.  It is
intentionally self-contained: data loading, causal forecasts, risk sizing,
execution, costs, reporting, and the command-line entrypoint live here without
imports from any earlier research strategy.

The default book combines a 12-month sign trend with basis momentum at equal
risk weight, applies causal per-market risk management, assigns equal nominal
pre-forecast volatility budgets to available instruments, and targets 7%
annualized portfolio volatility.
The canonical ledger launches with $1 million and zero positions on
1990-01-01, using earlier observations only to warm causal estimates.
Month-end targets are queued for each market's next observed, positive-volume
session.  The model fills at that session's close, carries actual contract
quantities in a self-financing USD NAV ledger, and charges both strategy
turnover and approximate two-leg continuous-contract roll turnover.

The construction was selected retrospectively and every available period has
been reused.  Reports therefore emphasize uncertainty, trade quality,
friction, capacity, and operational readiness rather than return targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STRATEGY_DESIGNATION = "Best Available Hardened Strategy"
STRATEGY_TECHNICAL_NAME = (
    "Diversified Global-Futures Time-Series Momentum Plus Basis-Momentum Portfolio"
)
STRATEGY_NAME = STRATEGY_TECHNICAL_NAME
ENGINE_VERSION = "3.2.1"

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

# These yield-quoted ASX bond contracts require rate-dependent valuation and
# explicit face/notional functions.  The supplied undated scalar point values
# cannot represent their exact P&L or gross exposure, so they remain outside
# the research book until effective-dated contract logic is available.  This
# is a model-correctness exclusion, not a return-based universe selection.
UNSUPPORTED_RESEARCH_CONTRACTS = frozenset({"YXT", "YYT"})

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
    "1990-2004 development history": ("1990-01-01", "2004-12-31"),
    "2005-2014 reused later diagnostic": ("2005-01-01", "2014-12-31"),
    "1990-2014 full post-launch history": ("1990-01-01", "2014-12-31"),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen research parameters and conservative execution assumptions."""

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

    # A 7% portfolio risk budget leaves operational headroom below the
    # independent 15% drawdown halt.  It is a risk-policy choice, not a return
    # target, and changing it invalidates any prior prospective evidence.
    target_vol: float = 0.07
    vol_decay: float = 0.94
    vol_estimator_min_periods: int = 20
    portfolio_vol_window: int = 63
    min_risk_scalar: float = 0.25
    max_risk_scalar: float = 2.00
    risk_budget: str = "flat"
    risk_managed_window: int | None = 63
    risk_managed_cap: float = 2.00
    no_trade_buffer: float = 0.25
    # Both default to the historical behaviour so the canonical run is
    # unchanged until a lever is switched on deliberately.
    cost_aware_buffer: bool = False
    cost_aware_buffer_floor: float = 0.05
    cost_aware_buffer_ceiling: float = 0.60
    buffer_suppresses_risk_scalar: bool = True

    half_spread_ticks: float = 0.50
    slippage_ticks: float = 0.25
    commission_per_contract: float = 2.50
    exchange_and_regulatory_fees_per_contract: float = 1.50
    impact_bps_at_full_participation: float = 10.00
    initial_capital: float = 1_000_000.0
    launch_date: str | None = "1990-01-01"
    integer_contracts: bool = True
    execution_timing: str = "next_close"
    execution_delay_sessions: int = 0
    charge_roll_costs: bool = True
    max_rebalance_participation: float | None = 0.02
    # The 5x position-intent buffer sits inside the independent 6x live gate in
    # the deployment-control layer.  A static margin snapshot must not resize
    # 1990-2014
    # history: doing so would inject today's margin schedule into past
    # decisions.  Historical margin remains an informational proxy, while live
    # sizing uses an effective-dated broker/clearing snapshot and the 35% gate.
    max_gross_notional_multiple: float | None = 5.0
    max_static_margin_fraction: float | None = None
    annualization: int = 252

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
        if not isinstance(self.cost_aware_buffer, bool):
            raise ValueError("cost_aware_buffer must be a boolean")
        if not isinstance(self.buffer_suppresses_risk_scalar, bool):
            raise ValueError("buffer_suppresses_risk_scalar must be a boolean")
        if not 0 <= self.cost_aware_buffer_floor < self.cost_aware_buffer_ceiling < 1:
            raise ValueError(
                "cost-aware buffer bounds must satisfy "
                "0 <= floor < ceiling < 1"
            )
        if min(
            self.half_spread_ticks,
            self.slippage_ticks,
            self.commission_per_contract,
            self.exchange_and_regulatory_fees_per_contract,
            self.impact_bps_at_full_participation,
        ) < 0:
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
        if self.execution_delay_sessions < 0:
            raise ValueError("execution_delay_sessions cannot be negative")
        if (
            self.max_rebalance_participation is not None
            and not 0 < self.max_rebalance_participation <= 1
        ):
            raise ValueError("max_rebalance_participation must be in (0, 1]")
        if (
            self.max_gross_notional_multiple is not None
            and self.max_gross_notional_multiple <= 0
        ):
            raise ValueError("max_gross_notional_multiple must be positive")
        if (
            self.max_static_margin_fraction is not None
            and not 0 < self.max_static_margin_fraction <= 1
        ):
            raise ValueError("max_static_margin_fraction must be in (0, 1]")


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
    gross_pnl_by_market: pd.DataFrame
    regular_cost_by_market: pd.DataFrame
    roll_cost_by_market: pd.DataFrame
    trade_episodes: pd.DataFrame
    execution_timing: str


def strategy_symbols() -> list[str]:
    return [
        symbol
        for members in GLOBAL_ASSET_CLASSES.values()
        for symbol in members
        if symbol not in UNSUPPORTED_RESEARCH_CONTRACTS
    ]


def _load_column(path: Path, column: str, name: str) -> pd.Series:
    """Load one vendor column and reject ambiguous or malformed observations."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input file: {path}")
    frame = pd.read_csv(path, usecols=["Date", column])
    parsed_dates = pd.to_datetime(frame["Date"], errors="coerce")
    if parsed_dates.isna().any():
        bad_rows = frame.index[parsed_dates.isna()].tolist()[:5]
        raise ValueError(f"Invalid Date values in {path} at rows {bad_rows}")
    if parsed_dates.duplicated().any():
        duplicates = parsed_dates[parsed_dates.duplicated(keep=False)]
        examples = sorted({date.date().isoformat() for date in duplicates})[:5]
        raise ValueError(f"Duplicate Date values in {path}: {examples}")
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & values.isna()
    if invalid.any():
        bad_rows = frame.index[invalid].tolist()[:5]
        raise ValueError(f"Non-numeric {column} values in {path} at rows {bad_rows}")
    non_finite = values.notna() & ~np.isfinite(values.to_numpy(dtype=float))
    if non_finite.any():
        bad_rows = frame.index[non_finite].tolist()[:5]
        raise ValueError(f"Non-finite {column} values in {path} at rows {bad_rows}")
    if column == "Volume":
        negative = values.notna() & values.lt(0)
        if negative.any():
            bad_rows = frame.index[negative].tolist()[:5]
            raise ValueError(f"Negative Volume values in {path} at rows {bad_rows}")
    return pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex(parsed_dates),
        name=name,
    ).sort_index()


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
    if volumes.lt(0).any().any():
        locations = np.argwhere(volumes.to_numpy(dtype=float) < 0)
        row, column = locations[0]
        raise ValueError(
            "Negative volume in "
            f"{volumes.columns[column]} on {volumes.index[row].date()}"
        )
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
    ]
    duplicates = catalogue["clean_symbol"].duplicated(keep=False)
    if duplicates.any():
        symbols_with_duplicates = sorted(catalogue.loc[duplicates, "clean_symbol"].unique())
        raise ValueError(
            f"Duplicate catalogue rows for: {symbols_with_duplicates}"
        )
    catalogue = catalogue.set_index("clean_symbol")
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
    """Select completed monthly decision rows without using a partial month.

    A later month proves that every preceding observed month has completed, so
    its final available row is retained.  The terminal month is retained only
    when the data end on a standard Monday-Friday business-month-end.  Venue
    holiday calendars are intentionally external to this research scheduler;
    production scheduling must supply an exchange-specific completed-session
    calendar rather than infer it from a truncated data file.
    """
    if frame.empty:
        return frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("monthly scheduling requires a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("monthly scheduling index must be sorted")

    rows = frame.groupby(frame.index.to_period("M")).tail(1)
    terminal = frame.index[-1].normalize()
    if not pd.offsets.BMonthEnd().is_on_offset(terminal):
        rows = rows.iloc[:-1]
    return rows


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
    if config.risk_budget == "flat":
        risk_groups = {"All markets": tuple(prices.columns)}
    else:
        # The configured catalogue contains contracts that may be deliberately
        # excluded from the supported research universe.  Intersect every
        # group with the actual frame before indexing; an optional allocator
        # must not reintroduce unsupported contracts or fail on a subset book.
        available_columns = set(prices.columns)
        risk_groups = {
            name: tuple(symbol for symbol in members if symbol in available_columns)
            for name, members in GLOBAL_ASSET_CLASSES.items()
        }
        risk_groups = {
            name: members for name, members in risk_groups.items() if members
        }
    if not risk_groups:
        return positions
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
    gross_pnl_by_market: pd.DataFrame
    regular_cost_by_market: pd.DataFrame
    roll_cost_by_market: pd.DataFrame


def _limit_target_contracts(
    target_per_dollar: np.ndarray,
    decision_nav: float,
    valuation_prices: np.ndarray,
    point_values: np.ndarray,
    margins: np.ndarray,
    config: StrategyConfig,
    integer_contracts: bool,
) -> tuple[np.ndarray, float]:
    """Convert a target to contracts and apply portfolio-level hard limits."""
    desired = target_per_dollar * decision_nav
    nonzero = desired != 0
    if nonzero.any() and (
        not np.isfinite(valuation_prices[nonzero]).all()
        or not np.isfinite(point_values[nonzero]).all()
        or not np.isfinite(margins[nonzero]).all()
    ):
        raise ValueError("Missing decision-date valuation input for a nonzero target")

    gross_notional = float(
        np.sum(np.abs(desired[nonzero] * valuation_prices[nonzero] * point_values[nonzero]))
    )
    margin_required = float(np.sum(np.abs(desired[nonzero]) * margins[nonzero]))
    scale = 1.0
    if config.max_gross_notional_multiple is not None and gross_notional > 0:
        scale = min(
            scale,
            config.max_gross_notional_multiple * decision_nav / gross_notional,
        )
    if config.max_static_margin_fraction is not None and margin_required > 0:
        scale = min(
            scale,
            config.max_static_margin_fraction * decision_nav / margin_required,
        )
    desired = desired * scale
    if integer_contracts:
        desired = np.rint(desired)

        nonzero = desired != 0
        rounded_gross = float(
            np.sum(
                np.abs(
                    desired[nonzero]
                    * valuation_prices[nonzero]
                    * point_values[nonzero]
                )
            )
        )
        rounded_margin = float(np.sum(np.abs(desired[nonzero]) * margins[nonzero]))
        post_round_scale = 1.0
        if (
            config.max_gross_notional_multiple is not None
            and rounded_gross > config.max_gross_notional_multiple * decision_nav
        ):
            post_round_scale = min(
                post_round_scale,
                config.max_gross_notional_multiple * decision_nav / rounded_gross,
            )
        if (
            config.max_static_margin_fraction is not None
            and rounded_margin > config.max_static_margin_fraction * decision_nav
        ):
            post_round_scale = min(
                post_round_scale,
                config.max_static_margin_fraction * decision_nav / rounded_margin,
            )
        if post_round_scale < 1.0:
            desired = np.trunc(desired * post_round_scale)
            scale *= post_round_scale
    return desired, scale


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
    market_gross_pnl = np.zeros_like(positions)
    market_regular_cost = np.zeros_like(positions)
    market_roll_cost = np.zeros_like(positions)
    gross_pnl = np.zeros(n_dates)
    costs = np.zeros(n_dates)
    fixed_costs = np.zeros(n_dates)
    impact_costs = np.zeros(n_dates)
    nav_path = np.zeros(n_dates)
    rebalance_turnover = np.zeros(n_dates)
    roll_turnover_increment = np.zeros(n_dates)
    total_turnover = np.zeros(n_dates)
    gross_notional = np.zeros(n_dates)
    static_margin = np.zeros(n_dates)
    max_participation = np.zeros(n_dates)
    max_rebalance_participation = np.zeros(n_dates)
    max_roll_participation = np.zeros(n_dates)
    filled_markets = np.zeros(n_dates, dtype=int)
    pending_markets = np.zeros(n_dates, dtype=int)
    target_limit_scale = np.ones(n_dates)

    quantity = np.zeros(n_markets, dtype=float)
    pending_target = np.full(n_markets, np.nan)
    pending_quantity = np.full(n_markets, np.nan)
    pending_delay = np.zeros(n_markets, dtype=int)
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
        # Capacity is fixed from information available before the execution
        # session.  Using the completed session's volume to size an order that
        # is then priced at that session's close is an ex-post liquidity
        # look-ahead.  Realized participation is still reported against actual
        # volume and can therefore exceed the intended ADV fraction.
        trailing_volume = volume.fillna(0.0).rolling(
            config.volume_gate_window,
            min_periods=config.volume_gate_window,
        ).median().shift(1)
        rebalance_capacity = (
            trailing_volume.to_numpy(dtype=float)
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
            decision_location = index.get_loc(decision_date)
            pending_quantity, target_limit_scale[i] = _limit_target_contracts(
                pending_target,
                nav,
                valuation_close_values[decision_location],
                point_value_values[decision_location],
                margin_values[decision_location],
                config,
                integer_contracts,
            )
            pending_delay.fill(config.execution_delay_sessions)
            no_change = np.isclose(
                pending_quantity,
                quantity,
                atol=1e-12,
                rtol=0.0,
            )
            pending_target[no_change] = np.nan
            pending_quantity[no_change] = np.nan
            pending_delay[no_change] = 0
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

        eligible = actual_session & np.isfinite(pending_quantity)
        fill = eligible & (pending_delay == 0)
        pending_delay[eligible & (pending_delay > 0)] -= 1
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
        pending_delay[completed] = 0
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
            day_market_gross_pnl = overnight_pnl + intraday_pnl
            day_gross_pnl = float(np.sum(day_market_gross_pnl))
        else:
            day_market_gross_pnl = full_pnl
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
        traded = charged_turnover > 0
        if np.any(traded & ~np.isfinite(one_way_values[i])):
            raise ValueError(f"Missing transaction-cost input on {date.date()}")
        if np.any(
            traded
            & (
                ~np.isfinite(valuation_close_values[i])
                | ~np.isfinite(point_value_values[i])
            )
        ):
            raise ValueError(f"Missing market-impact input on {date.date()}")
        cost_participation = np.divide(
            charged_turnover,
            volume_values[i],
            out=np.zeros(n_markets),
            where=traded & (volume_values[i] > 0),
        )
        physical_order_turnover = roll_adjusted_turnover
        physical_order_participation = np.divide(
            physical_order_turnover,
            volume_values[i],
            out=np.zeros(n_markets),
            where=(physical_order_turnover > 0) & (volume_values[i] > 0),
        )
        rebalance_participation = np.divide(
            regular_turnover,
            volume_values[i],
            out=np.zeros(n_markets),
            where=(regular_turnover > 0) & (volume_values[i] > 0),
        )
        roll_participation = np.divide(
            incremental_roll,
            volume_values[i],
            out=np.zeros(n_markets),
            where=(incremental_roll > 0) & (volume_values[i] > 0),
        )
        fixed_market_cost = np.where(
            traded,
            charged_turnover * one_way_values[i],
            0.0,
        )
        impact_market_cost = np.where(
            traded,
            charged_turnover
            * np.abs(valuation_close_values[i])
            * point_value_values[i]
            * (config.impact_bps_at_full_participation / 10_000)
            * np.sqrt(cost_participation),
            0.0,
        )
        total_market_cost = (
            fixed_market_cost + impact_market_cost
            if charge_costs
            else np.zeros(n_markets)
        )
        effective_unit_cost = np.divide(
            total_market_cost,
            charged_turnover,
            out=np.zeros(n_markets),
            where=traded,
        )
        day_regular_cost = regular_turnover * effective_unit_cost
        day_roll_cost = (
            incremental_roll * effective_unit_cost
            if config.charge_roll_costs
            else np.zeros(n_markets)
        )
        day_cost = float(np.sum(total_market_cost))
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

        max_participation[i] = float(np.max(physical_order_participation))
        max_rebalance_participation[i] = float(np.max(rebalance_participation))
        max_roll_participation[i] = float(np.max(roll_participation))
        market_gross_pnl[i] = day_market_gross_pnl
        market_regular_cost[i] = day_regular_cost
        market_roll_cost[i] = day_roll_cost
        gross_pnl[i] = day_gross_pnl
        costs[i] = day_cost
        fixed_costs[i] = float(np.sum(fixed_market_cost)) if charge_costs else 0.0
        impact_costs[i] = float(np.sum(impact_market_cost)) if charge_costs else 0.0
        nav_path[i] = nav
        rebalance_turnover[i] = float(np.sum(regular_turnover))
        roll_turnover_increment[i] = float(np.sum(incremental_roll))
        total_turnover[i] = float(np.sum(charged_turnover))
        positions[i] = quantity

    prior_nav = np.r_[initial_capital, nav_path[:-1]]
    daily = pd.DataFrame(
        {
            "prior_nav_usd": prior_nav,
            "gross_pnl_usd": gross_pnl,
            "transaction_cost_usd": costs,
            "fixed_execution_cost_usd": fixed_costs,
            "market_impact_cost_usd": impact_costs,
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
            "max_rebalance_participation": max_rebalance_participation,
            "max_roll_participation_proxy": max_roll_participation,
            "filled_markets": filled_markets,
            "pending_markets": pending_markets,
            "target_portfolio_limit_scale": target_limit_scale,
        },
        index=index,
    )
    return _ExecutionLedger(
        daily=daily,
        positions=pd.DataFrame(positions, index=index, columns=columns),
        trades=pd.DataFrame(trades, index=index, columns=columns),
        executed_rolls=pd.DataFrame(executed_rolls, index=index, columns=columns),
        gross_pnl_by_market=pd.DataFrame(
            market_gross_pnl,
            index=index,
            columns=columns,
        ),
        regular_cost_by_market=pd.DataFrame(
            market_regular_cost,
            index=index,
            columns=columns,
        ),
        roll_cost_by_market=pd.DataFrame(
            market_roll_cost,
            index=index,
            columns=columns,
        ),
    )


def _asset_class_lookup() -> dict[str, str]:
    return {
        symbol: asset_class
        for asset_class, symbols in GLOBAL_ASSET_CLASSES.items()
        for symbol in symbols
    }


def build_trade_episodes(result: BacktestResult) -> pd.DataFrame:
    """Build net directional position episodes for canonical next-close fills.

    A same-sign resize stays inside the episode, a sign flip closes one episode
    and opens another, and a contract roll contributes cost without creating a
    new alpha trade.  Open episodes are retained as censored observations.
    """
    columns = [
        "Episode ID",
        "Symbol",
        "Asset class",
        "Direction",
        "Entry date",
        "Exit date",
        "Status",
        "Entry contracts",
        "Maximum absolute contracts",
        "Entry NAV USD",
        "Holding sessions",
        "Holding calendar days",
        "Gross P&L USD",
        "Regular cost USD",
        "Roll cost USD",
        "Net P&L USD",
        "Gross contribution",
        "Cost contribution",
        "Net contribution",
        "MFE contribution",
        "MAE contribution",
        "Resize count",
        "Roll count",
    ]
    if result.execution_timing != "next_close":
        return pd.DataFrame(columns=columns)

    prior_positions = result.positions.shift().fillna(0.0)
    prior_nav = result.daily["prior_nav_usd"]
    asset_classes = _asset_class_lookup()
    episodes: list[dict[str, object]] = []

    for symbol in result.positions.columns:
        state: dict[str, object] | None = None
        episode_number = 0

        def start_episode(date: pd.Timestamp, quantity: float) -> dict[str, object]:
            nonlocal episode_number
            episode_number += 1
            return {
                "Episode ID": f"{symbol}-{episode_number:04d}",
                "Symbol": symbol,
                "Asset class": asset_classes[symbol],
                "Direction": "Long" if quantity > 0 else "Short",
                "Entry date": date,
                "Exit date": pd.NaT,
                "Status": "Open",
                "Entry contracts": abs(quantity),
                "Maximum absolute contracts": abs(quantity),
                "Entry NAV USD": float(prior_nav.loc[date]),
                "Holding sessions": 0,
                "Holding calendar days": 0,
                "Gross P&L USD": 0.0,
                "Regular cost USD": 0.0,
                "Roll cost USD": 0.0,
                "Gross contribution": 0.0,
                "Cost contribution": 0.0,
                "Net contribution": 0.0,
                "MFE contribution": 0.0,
                "MAE contribution": 0.0,
                "Resize count": 0,
                "Roll count": 0,
            }

        def add_outcome(
            current: dict[str, object],
            date: pd.Timestamp,
            gross_pnl: float = 0.0,
            regular_cost: float = 0.0,
            roll_cost: float = 0.0,
        ) -> None:
            denominator = float(prior_nav.loc[date])
            current["Gross P&L USD"] = float(current["Gross P&L USD"]) + gross_pnl
            current["Regular cost USD"] = (
                float(current["Regular cost USD"]) + regular_cost
            )
            current["Roll cost USD"] = float(current["Roll cost USD"]) + roll_cost
            gross_contribution = gross_pnl / denominator
            cost_contribution = (regular_cost + roll_cost) / denominator
            current["Gross contribution"] = (
                float(current["Gross contribution"]) + gross_contribution
            )
            current["Cost contribution"] = (
                float(current["Cost contribution"]) + cost_contribution
            )
            current["Net contribution"] = (
                float(current["Net contribution"])
                + gross_contribution
                - cost_contribution
            )
            current["MFE contribution"] = max(
                float(current["MFE contribution"]),
                float(current["Net contribution"]),
            )
            current["MAE contribution"] = min(
                float(current["MAE contribution"]),
                float(current["Net contribution"]),
            )

        def close_episode(current: dict[str, object], date: pd.Timestamp) -> None:
            current["Exit date"] = date
            current["Status"] = "Closed"
            current["Holding calendar days"] = (
                date - pd.Timestamp(current["Entry date"])
            ).days
            current["Net P&L USD"] = (
                float(current["Gross P&L USD"])
                - float(current["Regular cost USD"])
                - float(current["Roll cost USD"])
            )
            episodes.append(current)

        for date in result.positions.index:
            old_quantity = float(prior_positions.at[date, symbol])
            end_quantity = float(result.positions.at[date, symbol])
            trade_quantity = float(result.trades.at[date, symbol])
            gross_pnl = float(result.gross_pnl_by_market.at[date, symbol])
            regular_cost = float(result.regular_cost_by_market.at[date, symbol])
            roll_cost = float(result.roll_cost_by_market.at[date, symbol])

            if old_quantity != 0:
                if state is None:
                    raise ValueError(f"Missing open episode for {symbol} on {date.date()}")
                add_outcome(state, date, gross_pnl=gross_pnl, roll_cost=roll_cost)
                state["Holding sessions"] = int(state["Holding sessions"]) + 1
                if bool(result.executed_rolls.at[date, symbol]):
                    state["Roll count"] = int(state["Roll count"]) + 1

            old_sign = np.sign(old_quantity)
            end_sign = np.sign(end_quantity)
            if old_sign == 0 and end_sign != 0:
                state = start_episode(date, end_quantity)
                add_outcome(state, date, regular_cost=regular_cost, roll_cost=roll_cost)
            elif old_sign != 0 and end_sign == 0:
                if state is None:
                    raise ValueError(f"Missing closing episode for {symbol} on {date.date()}")
                add_outcome(state, date, regular_cost=regular_cost)
                close_episode(state, date)
                state = None
            elif old_sign != 0 and end_sign == old_sign:
                if state is None:
                    raise ValueError(f"Missing resized episode for {symbol} on {date.date()}")
                if trade_quantity != 0:
                    add_outcome(state, date, regular_cost=regular_cost)
                    state["Resize count"] = int(state["Resize count"]) + 1
                state["Maximum absolute contracts"] = max(
                    float(state["Maximum absolute contracts"]),
                    abs(end_quantity),
                )
            elif old_sign != 0 and end_sign == -old_sign:
                if state is None:
                    raise ValueError(f"Missing flipped episode for {symbol} on {date.date()}")
                unit_cost = regular_cost / abs(trade_quantity)
                closing_cost = abs(old_quantity) * unit_cost
                opening_cost = abs(end_quantity) * unit_cost
                add_outcome(state, date, regular_cost=closing_cost)
                close_episode(state, date)
                state = start_episode(date, end_quantity)
                add_outcome(state, date, regular_cost=opening_cost)

        if state is not None:
            final_date = result.positions.index[-1]
            state["Holding calendar days"] = (
                final_date - pd.Timestamp(state["Entry date"])
            ).days
            state["Net P&L USD"] = (
                float(state["Gross P&L USD"])
                - float(state["Regular cost USD"])
                - float(state["Roll cost USD"])
            )
            episodes.append(state)

    return pd.DataFrame(episodes, columns=columns).sort_values(
        ["Entry date", "Symbol", "Episode ID"]
    ).reset_index(drop=True)


def trade_episode_metrics(
    episodes: pd.DataFrame,
    start: str,
    end: str | None = None,
) -> pd.Series:
    """Summarize closed episodes by exit date; open episodes are censored."""
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    if episodes.empty:
        closed = episodes
        open_count = 0
    else:
        exit_dates = pd.to_datetime(episodes["Exit date"])
        entry_dates = pd.to_datetime(episodes["Entry date"])
        closed = episodes[
            episodes["Status"].eq("Closed")
            & exit_dates.between(start_date, end_date)
        ]
        open_count = int(
            (
                entry_dates.le(end_date)
                & (exit_dates.isna() | exit_dates.gt(end_date))
            ).sum()
        )

    contribution = closed["Net contribution"].astype(float)
    pnl_usd = closed["Net P&L USD"].astype(float)
    # Classify the economically realised episode after all costs.  A one-cent
    # tolerance prevents floating-point dust from manufacturing wins/losses.
    # USD and NAV-normalised contribution can occasionally have opposite signs
    # because each day's contribution uses that day's prior NAV.  Classify each
    # numeraire independently so its profit-factor and payoff identities remain
    # internally coherent.  Trade counts retain the economically realised USD
    # convention, including the one-cent dust tolerance.
    usd_wins = pnl_usd > 0.01
    usd_losses = pnl_usd < -0.01
    breakevens = ~(usd_wins | usd_losses)
    contribution_wins = contribution > 1e-12
    contribution_losses = contribution < -1e-12
    win_values = contribution[contribution_wins]
    loss_values = -contribution[contribution_losses]
    win_usd = pnl_usd[usd_wins]
    loss_usd = -pnl_usd[usd_losses]

    def safe_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else np.nan

    average_win = float(win_values.mean()) if len(win_values) else np.nan
    average_loss = float(loss_values.mean()) if len(loss_values) else np.nan
    return pd.Series(
        {
            "Closed trade episodes": len(closed),
            "Open/censored trade episodes": open_count,
            "Episodes entered before window": int(
                (pd.to_datetime(closed["Entry date"]) < start_date).sum()
            ),
            "Winning trade episodes": int(usd_wins.sum()),
            "Losing trade episodes": int(usd_losses.sum()),
            "Breakeven trade episodes": int(breakevens.sum()),
            "Trade win rate": usd_wins.mean() if len(closed) else np.nan,
            "Non-breakeven trade win rate": safe_ratio(
                float(usd_wins.sum()),
                float(usd_wins.sum() + usd_losses.sum()),
            ),
            "Trade profit factor (contribution)": safe_ratio(
                float(win_values.sum()),
                float(loss_values.sum()),
            ),
            "Trade profit factor (USD)": safe_ratio(
                float(win_usd.sum()),
                float(loss_usd.sum()),
            ),
            "Trade expectancy (bps NAV)": (
                float(contribution.mean()) * 10_000 if len(closed) else np.nan
            ),
            "Trade expectancy (USD)": float(pnl_usd.mean()) if len(closed) else np.nan,
            "Average winning trade (bps NAV)": average_win * 10_000,
            "Average losing trade (bps NAV)": average_loss * 10_000,
            "Average win / average loss": safe_ratio(average_win, average_loss),
            "Win / loss count ratio": safe_ratio(
                float(usd_wins.sum()),
                float(usd_losses.sum()),
            ),
            "Median trade outcome (bps NAV)": (
                float(contribution.median()) * 10_000 if len(closed) else np.nan
            ),
            "Average holding sessions": (
                float(closed["Holding sessions"].mean()) if len(closed) else np.nan
            ),
        }
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
    buffer_fraction: float | pd.Series = 0.25,
) -> pd.DataFrame:
    """Keep the previous target when a month-end adjustment is too small.

    ``buffer_fraction`` may be a scalar, which is the historical behaviour, or
    a per-market Series.  A per-market width lets the band scale with what a
    trade actually costs: trading is many times more expensive per unit of risk
    in a thin bond future than in an equity index, so one uniform band
    necessarily overtrades the expensive markets and undertrades the cheap
    ones.
    """

    if isinstance(buffer_fraction, pd.Series):
        widths = buffer_fraction.reindex(desired.columns)
        if widths.isna().any():
            missing = widths.index[widths.isna()].tolist()[:5]
            raise ValueError(f"buffer_fraction is missing markets: {missing}")
        if not ((widths >= 0) & (widths < 1)).all():
            raise ValueError("buffer_fraction values must be in [0, 1)")
    else:
        if not 0 <= buffer_fraction < 1:
            raise ValueError("buffer_fraction must be in [0, 1)")
        widths = pd.Series(float(buffer_fraction), index=desired.columns)

    executable = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
    previous = pd.Series(0.0, index=desired.columns)
    for date, raw_target in desired.iterrows():
        target = raw_target.fillna(0.0)
        change = target - previous
        reference = pd.concat([target.abs(), previous.abs()], axis=1).max(axis=1)
        should_trade = change.abs() > widths * reference
        previous = previous.where(~should_trade, target)
        executable.loc[date] = previous
    return executable


def cost_aware_buffer_widths(
    prices: pd.DataFrame,
    point_values: pd.DataFrame,
    one_way_costs: pd.DataFrame,
    config: StrategyConfig,
) -> pd.Series:
    """Per-market no-trade widths scaled by round-trip cost per unit of risk.

    The width is proportional to what a round trip costs measured against the
    market's own daily risk, so a band is wide exactly where trading destroys
    the most value.  Every input is a configured cost parameter or a realized
    price, and the proportionality constant is fixed so the book's
    risk-weighted average width reproduces ``config.no_trade_buffer``.  Nothing
    here is fitted to returns.
    """

    daily_price_vol = prices.diff().ewm(
        span=config.vol_span,
        min_periods=config.vol_span,
        adjust=False,
    ).std()
    risk_per_contract = (daily_price_vol * point_values).replace(
        [np.inf, -np.inf], np.nan
    )
    round_trip = 2.0 * one_way_costs
    ratio = (round_trip / risk_per_contract).replace([np.inf, -np.inf], np.nan)
    typical = ratio.median(axis=0, skipna=True)
    if typical.notna().sum() == 0:
        return pd.Series(config.no_trade_buffer, index=prices.columns)
    # Normalize so the cross-sectional mean width equals the configured
    # buffer: this varies *where* the band is wide without changing how much
    # buffering the book does in aggregate, which keeps the change attributable.
    scale = config.no_trade_buffer / float(typical.mean(skipna=True))
    widths = (typical * scale).fillna(config.no_trade_buffer)
    return widths.clip(
        lower=config.cost_aware_buffer_floor,
        upper=config.cost_aware_buffer_ceiling,
    )


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
        (config.half_spread_ticks + config.slippage_ticks)
        * point_values.mul(metadata["tick_size"], axis=1)
        + config.commission_per_contract
        + config.exchange_and_regulatory_fees_per_contract
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
    scalar_rows = monthly_risk_scalar.reindex(base_monthly.index)
    desired = base_monthly.mul(scalar_rows, axis=0)

    buffer_widths: float | pd.Series = config.no_trade_buffer
    if config.cost_aware_buffer:
        buffer_widths = cost_aware_buffer_widths(
            prices, point_values, one_way_costs, config
        )

    # The buffer normally filters the scalar-multiplied target, so a portfolio
    # de-risking smaller than the band is suppressed along with the market's
    # own drift.  Applying it to the pre-scalar target instead lets every
    # volatility-target move reach the book while still buffering signal churn.
    buffer_input = desired if config.buffer_suppresses_risk_scalar else base_monthly

    def _buffered_targets(frame: pd.DataFrame) -> pd.DataFrame:
        executable = apply_no_trade_buffer(frame, buffer_widths)
        if config.buffer_suppresses_risk_scalar:
            return executable
        return executable.mul(scalar_rows.reindex(executable.index), axis=0)

    buffered = _buffered_targets(buffer_input)
    if config.launch_date is not None and not desired.empty:
        launch_timestamp = pd.Timestamp(config.launch_date)
        prior_decisions = desired.index[desired.index < launch_timestamp]
        live_decisions = desired.index[desired.index >= launch_timestamp]
        if len(prior_decisions) or len(live_decisions):
            first_live_decision = (
                prior_decisions[-1] if len(prior_decisions) else live_decisions[0]
            )
            buffered = _buffered_targets(buffer_input.loc[first_live_decision:])
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
    result = BacktestResult(
        name=STRATEGY_NAME,
        daily=daily,
        positions=execution.positions,
        target_positions=buffered,
        signals=forecast,
        prices=prices,
        metadata=metadata,
        trades=execution.trades,
        roll_events=rolls,
        executed_rolls=execution.executed_rolls,
        gross_pnl_by_market=execution.gross_pnl_by_market,
        regular_cost_by_market=execution.regular_cost_by_market,
        roll_cost_by_market=execution.roll_cost_by_market,
        trade_episodes=pd.DataFrame(),
        execution_timing=config.execution_timing,
    )
    result.trade_episodes = build_trade_episodes(result)
    return result


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

    underwater = pd.Series(drawdown < 0, index=returns.index)
    drawdown_streak = underwater.groupby((~underwater).cumsum()).cumsum()
    positive_daily = returns[returns > 0].sum()
    negative_daily = -returns[returns < 0].sum()
    rolling_five_day = (1 + returns).rolling(5).apply(np.prod, raw=True) - 1
    rolling_month = (1 + returns).rolling(21).apply(np.prod, raw=True) - 1
    historical_var = -float(returns.quantile(0.05))
    tail = returns[returns <= returns.quantile(0.05)]
    metrics = pd.Series(
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
            "Annualized downside deviation": downside * math.sqrt(annualization),
            "Daily net profit factor": (
                positive_daily / negative_daily if negative_daily > 0 else np.nan
            ),
            "Daily win rate": (returns > 0).mean(),
            "Daily return skew": returns.skew(),
            "Daily return excess kurtosis": returns.kurt(),
            "Historical daily VaR 95%": historical_var,
            "Historical daily CVaR 95%": -float(tail.mean()),
            "Daily return autocorrelation (lag 1)": (
                returns.autocorr(1) if len(returns) >= 3 else np.nan
            ),
            "Max drawdown": drawdown.min(),
            "Max drawdown duration (sessions)": int(drawdown_streak.max()),
            "Calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
            "Positive months": (monthly > 0).mean(),
            "Best day": returns.max(),
            "Worst day": returns.min(),
            "Worst rolling 5 sessions": rolling_five_day.min(),
            "Worst rolling 21 sessions": rolling_month.min(),
            "Best month": monthly.max(),
            "Worst month": monthly.min(),
            "Annual cost drag": daily["cost"].mean() * annualization,
            "Annual fixed-cost drag": (
                daily["fixed_execution_cost_usd"]
                .div(daily["prior_nav_usd"])
                .mean()
                * annualization
            ),
            "Annual impact-cost drag": (
                daily["market_impact_cost_usd"]
                .div(daily["prior_nav_usd"])
                .mean()
                * annualization
            ),
            "Average risk scalar": daily["risk_scalar"].mean(),
            "Peak rolling 63-session realized volatility": (
                returns.rolling(63).std().max() * math.sqrt(annualization)
            ),
            "Average gross notional multiple": daily["gross_notional_multiple"].mean(),
            "Peak gross notional multiple": daily["gross_notional_multiple"].max(),
            "Average static margin fraction": daily["static_margin_fraction"].mean(),
            "Peak static margin fraction": daily["static_margin_fraction"].max(),
            "Peak order participation": daily["max_order_participation"].max(),
            "Peak rebalance participation": (
                daily["max_rebalance_participation"].max()
            ),
            "Peak roll participation proxy": (
                daily["max_roll_participation_proxy"].max()
            ),
            "Peak pending markets": daily["pending_markets"].max(),
            "Portfolio-limit binding decision days": (
                daily["target_portfolio_limit_scale"] < 1 - 1e-12
            ).sum(),
            "Average markets held": (
                result.positions.loc[start:end].ne(0).sum(axis=1).mean()
            ),
        },
        name=result.name,
    )
    return pd.concat(
        [metrics, trade_episode_metrics(result.trade_episodes, start, end)]
    )


def ledger_reconciliation_report(
    result: BacktestResult,
    *,
    absolute_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Reconcile market ledgers, positions, costs, returns, NAV, and episodes."""
    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")

    daily = result.daily
    rows: list[dict[str, object]] = []

    def add_check(name: str, error: float, detail: str) -> None:
        finite_error = float(error) if np.isfinite(error) else np.inf
        rows.append(
            {
                "check": name,
                "status": "PASS" if finite_error <= absolute_tolerance else "BLOCKED",
                "maximum_absolute_error": finite_error,
                "tolerance": absolute_tolerance,
                "detail": detail,
            }
        )

    add_check(
        "market_gross_pnl_to_portfolio",
        float(
            (
                result.gross_pnl_by_market.sum(axis=1)
                - daily["gross_pnl_usd"]
            ).abs().max()
        ),
        "Sum of market gross P&L equals the portfolio gross P&L.",
    )
    add_check(
        "market_costs_to_portfolio",
        float(
            (
                (result.regular_cost_by_market + result.roll_cost_by_market).sum(axis=1)
                - daily["transaction_cost_usd"]
            ).abs().max()
        ),
        "Regular plus roll costs reconcile to the portfolio cost ledger.",
    )
    add_check(
        "fixed_plus_impact_costs",
        float(
            (
                daily["fixed_execution_cost_usd"]
                + daily["market_impact_cost_usd"]
                - daily["transaction_cost_usd"]
            ).abs().max()
        ),
        "Parameterized fixed and impact costs reconcile to total costs.",
    )
    add_check(
        "position_change_to_trades",
        float(
            (
                result.positions.diff().fillna(result.positions)
                - result.trades
            ).abs().to_numpy().max()
        ),
        "Recorded trades equal the change in contract positions.",
    )
    add_check(
        "net_pnl_identity",
        float(
            (
                daily["gross_pnl_usd"]
                - daily["transaction_cost_usd"]
                - daily["net_pnl_usd"]
            ).abs().max()
        ),
        "Net P&L equals gross P&L less all modeled costs.",
    )
    add_check(
        "nav_recursion",
        float(
            (
                daily["prior_nav_usd"]
                + daily["net_pnl_usd"]
                - daily["nav"]
            ).abs().max()
        ),
        "Closing NAV equals prior NAV plus net P&L.",
    )
    add_check(
        "net_return_identity",
        float(
            (
                daily["net_pnl_usd"].div(daily["prior_nav_usd"])
                - daily["net_return"]
            ).abs().max()
        ),
        "Net return is calculated on prior closing NAV.",
    )

    if result.execution_timing == "next_close" and not result.trade_episodes.empty:
        episodes = result.trade_episodes
        add_check(
            "episode_gross_pnl_to_market_ledger",
            abs(
                float(episodes["Gross P&L USD"].sum())
                - float(result.gross_pnl_by_market.to_numpy().sum())
            ),
            "Closed and censored directional episodes include all gross market P&L.",
        )
        add_check(
            "episode_costs_to_market_ledger",
            abs(
                float(
                    episodes["Regular cost USD"].sum()
                    + episodes["Roll cost USD"].sum()
                )
                - float(daily["transaction_cost_usd"].sum())
            ),
            "Closed and censored directional episodes include all modeled costs.",
        )
        add_check(
            "episode_contribution_to_daily_return",
            abs(
                float(episodes["Net contribution"].sum())
                - float(daily["net_return"].sum())
            ),
            "Episode net contributions reconcile to additive daily net returns.",
        )
    return pd.DataFrame(rows)


def validate_backtest_result(result: BacktestResult) -> pd.DataFrame:
    """Fail closed when a core numerical accounting invariant is violated."""
    report = ledger_reconciliation_report(result)
    blocked = report[report["status"].ne("PASS")]
    if not blocked.empty:
        names = ", ".join(blocked["check"].astype(str))
        raise ValueError(f"Backtest reconciliation failed: {names}")
    numeric = result.daily.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Backtest daily ledger contains non-finite numeric values")
    if not result.daily["nav"].gt(0).all():
        raise ValueError("Backtest NAV must remain positive")
    if not result.daily["transaction_cost_usd"].ge(0).all():
        raise ValueError("Backtest transaction costs cannot be negative")
    return report


def data_quality_report(result: BacktestResult) -> pd.DataFrame:
    """Summarize model-ready coverage and execution activity by market."""
    lookup = _asset_class_lookup()
    rows: list[dict[str, object]] = []
    for symbol in result.prices.columns:
        price = result.prices[symbol]
        observed = price.dropna()
        rows.append(
            {
                "symbol": symbol,
                "asset_class": lookup.get(symbol, "Unknown"),
                "first_model_price": (
                    observed.index.min().date().isoformat() if not observed.empty else None
                ),
                "last_model_price": (
                    observed.index.max().date().isoformat() if not observed.empty else None
                ),
                "calendar_rows": len(price),
                "model_price_rows": int(price.notna().sum()),
                "model_price_missing_fraction": float(price.isna().mean()),
                "forecast_rows": int(result.signals[symbol].notna().sum()),
                "position_sessions": int(result.positions[symbol].ne(0).sum()),
                "rebalance_sessions": int(result.trades[symbol].ne(0).sum()),
                "executed_rolls": int(result.executed_rolls[symbol].sum()),
                "gross_pnl_usd": float(result.gross_pnl_by_market[symbol].sum()),
                "modeled_cost_usd": float(
                    result.regular_cost_by_market[symbol].sum()
                    + result.roll_cost_by_market[symbol].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def source_manifest(config: StrategyConfig) -> pd.DataFrame:
    """Hash every source file consumed by the canonical run."""
    data_dir = Path(config.data_dir)
    paths = [data_dir / "CATALOGUE_Delta1_Futures.csv"]
    for symbol in strategy_symbols():
        paths.extend(
            [
                data_dir / "Futures Data" / f"&{symbol}_CCB.csv",
                data_dir / "Futures Data" / f"&{symbol}.csv",
            ]
        )
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file while hashing inputs: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "relative_path": path.relative_to(data_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


def run_pipeline(config: StrategyConfig) -> tuple[BacktestResult, pd.DataFrame]:
    """Run the book once and report fixed, non-independent research windows."""
    result = run_backtest(config)
    validate_backtest_result(result)
    rows: list[dict[str, object]] = []
    for label, (start, end) in REPORTING_WINDOWS.items():
        metrics = performance_metrics(result, start, end, config.annualization)
        rows.append(
            {
                "Strategy": result.name,
                "Strategy designation": STRATEGY_DESIGNATION,
                "Window": label,
                "Validation status": "retrospective_reused_history",
                **metrics.to_dict(),
            }
        )
    return result, pd.DataFrame(rows)


def market_daily_ledger(result: BacktestResult) -> pd.DataFrame:
    """Return a long-form market ledger that reconciles to the portfolio."""
    start_positions = result.positions.shift().fillna(0.0)

    def stack(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame.rename_axis(index="date", columns="symbol").stack().rename(name)

    ledger = pd.concat(
        [
            stack(start_positions, "start_contracts"),
            stack(result.positions, "end_contracts"),
            stack(result.trades, "trade_contracts"),
            stack(result.gross_pnl_by_market, "gross_pnl_usd"),
            stack(result.regular_cost_by_market, "regular_cost_usd"),
            stack(result.roll_cost_by_market, "roll_cost_usd"),
            stack(result.executed_rolls.astype(bool), "executed_roll"),
            stack(result.signals.reindex_like(result.positions), "forecast"),
        ],
        axis=1,
    ).reset_index()
    prior_nav = result.daily["prior_nav_usd"].rename_axis("date")
    ledger = ledger.join(prior_nav, on="date")
    ledger["net_pnl_usd"] = (
        ledger["gross_pnl_usd"]
        - ledger["regular_cost_usd"]
        - ledger["roll_cost_usd"]
    )
    ledger["net_return_contribution"] = (
        ledger["net_pnl_usd"] / ledger["prior_nav_usd"]
    )
    ledger.insert(
        2,
        "asset_class",
        ledger["symbol"].map(_asset_class_lookup()).fillna("Unknown"),
    )
    return ledger


def save_outputs(
    result: BacktestResult,
    metrics: pd.DataFrame,
    config: StrategyConfig,
    *,
    bootstrap_samples: int = 2_000,
    trade_sequence_samples: int = 5_000,
    include_stress: bool = True,
) -> dict[str, object]:
    """Save the reproducible research package and fail-closed readiness audit."""
    from .diagnostics import (
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_DAILY_BLOCK_LENGTHS,
        DEFAULT_DAILY_HORIZONS_SESSIONS,
        causal_regime_report,
        daily_stationary_bootstrap_drawdown_summary,
        monthly_stationary_bootstrap_summary,
        trade_metrics_report,
    )
    from ..controls.production import (
        ProductionLimits,
        daily_frame_fingerprint,
        overall_readiness_status,
        production_readiness_report,
    )
    from .trade_sequence import (
        TradeSequenceMonteCarloConfig,
        trade_sequence_monte_carlo_summary,
    )
    from .friction import cost_model_assumptions, run_friction_stress_suite
    from .drawdown import (
        DrawdownOverlayConfig,
        overlay_performance_report,
        simulate_drawdown_overlay,
    )

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if trade_sequence_samples <= 0:
        raise ValueError("trade_sequence_samples must be positive")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_checks = validate_backtest_result(result)
    market_ledger = market_daily_ledger(result)
    trade_metrics = trade_metrics_report(result, REPORTING_WINDOWS)
    regime_metrics = causal_regime_report(result)
    bootstrap_windows = {
        "2005-2014 reused later diagnostic": REPORTING_WINDOWS[
            "2005-2014 reused later diagnostic"
        ],
        "1990-2014 full post-launch history": REPORTING_WINDOWS[
            "1990-2014 full post-launch history"
        ],
    }
    bootstrap = monthly_stationary_bootstrap_summary(
        result,
        bootstrap_windows,
        samples=bootstrap_samples,
    )
    daily_drawdown_bootstrap = daily_stationary_bootstrap_drawdown_summary(
        result,
        bootstrap_windows,
        samples=bootstrap_samples,
    )
    trade_sequence = trade_sequence_monte_carlo_summary(
        result,
        REPORTING_WINDOWS,
        config=TradeSequenceMonteCarloConfig(samples=trade_sequence_samples),
    )
    limits = ProductionLimits()
    readiness = production_readiness_report(
        result,
        limits=limits,
    )
    readiness_status = overall_readiness_status(readiness)
    quality = data_quality_report(result)
    sources = source_manifest(config)
    cost_assumptions = cost_model_assumptions(config)
    overlay_config = DrawdownOverlayConfig(annualization=config.annualization)
    overlay_start = config.launch_date or result.daily.index.min()
    overlay_source = result.daily.loc[str(overlay_start):]
    first_overlay_location = result.daily.index.get_indexer([overlay_source.index[0]])[0]
    overlay_period_start = (
        result.daily.index[first_overlay_location - 1]
        if first_overlay_location > 0
        else overlay_source.index[0]
    )
    drawdown_overlay = simulate_drawdown_overlay(overlay_source, overlay_config)
    drawdown_overlay_summary = overlay_performance_report(
        drawdown_overlay,
        overlay_config,
        period_start=overlay_period_start,
    )

    metrics.to_csv(output_dir / "strategy_metrics.csv", index=False)
    result.daily.to_csv(output_dir / "strategy_daily.csv", index_label="date")
    result.target_positions.to_csv(
        output_dir / "strategy_monthly_position_intents_per_dollar.csv",
        index_label="decision_date",
    )
    result.trade_episodes.to_csv(output_dir / "strategy_trade_episodes.csv", index=False)
    trade_metrics.to_csv(output_dir / "strategy_trade_metrics.csv", index=False)
    regime_metrics.to_csv(output_dir / "strategy_regime_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "strategy_monte_carlo_summary.csv", index=False)
    daily_drawdown_bootstrap.to_csv(
        output_dir / "strategy_daily_drawdown_monte_carlo.csv", index=False
    )
    trade_sequence.to_csv(
        output_dir / "strategy_trade_sequence_monte_carlo.csv", index=False
    )
    readiness.to_csv(output_dir / "production_readiness.csv", index=False)
    ledger_checks.to_csv(output_dir / "strategy_ledger_checks.csv", index=False)
    quality.to_csv(output_dir / "strategy_data_quality.csv", index=False)
    sources.to_csv(output_dir / "source_manifest.csv", index=False)
    cost_assumptions.to_csv(output_dir / "cost_model_assumptions.csv", index=False)
    drawdown_overlay.to_csv(
        output_dir / "strategy_drawdown_overlay_diagnostic.csv",
        index_label="date",
    )
    drawdown_overlay_summary.to_csv(
        output_dir / "strategy_drawdown_overlay_summary.csv", index=False
    )
    active_market_rows = (
        market_ledger["start_contracts"].ne(0)
        | market_ledger["end_contracts"].ne(0)
        | market_ledger["trade_contracts"].ne(0)
        | market_ledger["gross_pnl_usd"].ne(0)
        | market_ledger["regular_cost_usd"].ne(0)
        | market_ledger["roll_cost_usd"].ne(0)
    )
    persisted_market_ledger = market_ledger.loc[active_market_rows]
    persisted_market_ledger.to_csv(
        output_dir / "strategy_market_daily.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    event_mask = market_ledger["trade_contracts"].ne(0) | market_ledger[
        "executed_roll"
    ].eq(True)
    market_ledger.loc[event_mask].to_csv(
        output_dir / "strategy_execution_events.csv", index=False
    )

    stress = pd.DataFrame()
    stress_path = output_dir / "strategy_friction_stress.csv"
    if include_stress:
        def stress_runner(scenario_config: StrategyConfig) -> BacktestResult:
            return result if scenario_config == config else run_backtest(scenario_config)

        stress = run_friction_stress_suite(config, run_fn=stress_runner)
        stress.to_csv(stress_path, index=False)
    else:
        # A fast diagnostic run must not leave an older stress report looking
        # as though it belongs to the new manifest.
        stress_path.unlink(missing_ok=True)

    serializable = asdict(config)
    serializable["engine_version"] = ENGINE_VERSION
    serializable["strategy_name"] = STRATEGY_NAME
    serializable["strategy_designation"] = STRATEGY_DESIGNATION
    serializable["strategy_technical_name"] = STRATEGY_TECHNICAL_NAME
    serializable["data_dir"] = "${DELTA1_DATA_DIR}"
    serializable["output_dir"] = str(config.output_dir)
    serializable["research_status"] = {
        "designation": STRATEGY_DESIGNATION,
        "history": "retrospective_reused_history",
        "selection_history_complete": False,
        "selection_adjusted_statistics": False,
        "independent_holdout": False,
        "paper_or_live_track_record": False,
        "production_readiness": readiness_status,
    }
    config_text = json.dumps(serializable, indent=2, sort_keys=True, default=str)
    (output_dir / "strategy_config.json").write_text(config_text + "\n", encoding="utf-8")

    source_fingerprint = hashlib.sha256(
        "\n".join(
            sources["relative_path"].astype(str) + ":" + sources["sha256"].astype(str)
        ).encode("utf-8")
    ).hexdigest()
    repository_root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parents[1]
    implementation_hashes = {}
    for module_path in sorted(package_root.rglob("*.py")):
        relative_name = module_path.relative_to(repository_root).as_posix()
        implementation_hashes[relative_name] = hashlib.sha256(
            module_path.read_bytes()
        ).hexdigest()
    implementation_fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}:{digest}"
            for name, digest in sorted(implementation_hashes.items())
        ).encode("utf-8")
    ).hexdigest()
    output_names = [
        "cost_model_assumptions.csv",
        "production_readiness.csv",
        "source_manifest.csv",
        "strategy_config.json",
        "strategy_daily.csv",
        "strategy_daily_drawdown_monte_carlo.csv",
        "strategy_data_quality.csv",
        "strategy_drawdown_overlay_diagnostic.csv",
        "strategy_drawdown_overlay_summary.csv",
        "strategy_execution_events.csv",
        "strategy_ledger_checks.csv",
        "strategy_market_daily.csv.gz",
        "strategy_metrics.csv",
        "strategy_monte_carlo_summary.csv",
        "strategy_monthly_position_intents_per_dollar.csv",
        "strategy_regime_metrics.csv",
        "strategy_trade_episodes.csv",
        "strategy_trade_metrics.csv",
        "strategy_trade_sequence_monte_carlo.csv",
    ]
    if include_stress:
        output_names.append("strategy_friction_stress.csv")
    output_hashes: dict[str, str] = {}
    for output_name in sorted(output_names):
        output_path = output_dir / output_name
        if not output_path.is_file():
            raise RuntimeError(f"canonical output was not written: {output_name}")
        output_hashes[output_name] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    run_manifest = {
        "strategy": STRATEGY_NAME,
        "strategy_designation": STRATEGY_DESIGNATION,
        "strategy_technical_name": STRATEGY_TECHNICAL_NAME,
        "engine_version": ENGINE_VERSION,
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "source_fingerprint_sha256": source_fingerprint,
        "daily_fingerprint_sha256": daily_frame_fingerprint(result.daily),
        "implementation_fingerprint_sha256": implementation_fingerprint,
        "implementation_files": implementation_hashes,
        "output_files": output_hashes,
        "data_start": result.daily.index.min().date().isoformat(),
        "data_end": result.daily.index.max().date().isoformat(),
        "launch_date": config.launch_date,
        "reporting_status": "retrospective_reused_history",
        "selection_adjusted": False,
        "externally_validated": False,
        "production_readiness": readiness_status,
        "bootstrap_samples": bootstrap_samples,
        "daily_drawdown_bootstrap": {
            "block_lengths_sessions": list(DEFAULT_DAILY_BLOCK_LENGTHS),
            "horizons_sessions": list(DEFAULT_DAILY_HORIZONS_SESSIONS),
            "samples": bootstrap_samples,
            "seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "trade_sequence_samples": trade_sequence_samples,
        "friction_stress_included": include_stress,
        "drawdown_overlay_status": "diagnostic_not_execution_integrated",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "readiness_status": readiness_status,
        "readiness": readiness,
        "ledger_checks": ledger_checks,
        "trade_metrics": trade_metrics,
        "regime_metrics": regime_metrics,
        "bootstrap": bootstrap,
        "daily_drawdown_bootstrap": daily_drawdown_bootstrap,
        "trade_sequence": trade_sequence,
        "stress": stress,
        "drawdown_overlay": drawdown_overlay,
        "drawdown_overlay_summary": drawdown_overlay_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2_000,
        help="stationary-bootstrap paths per block/horizon combination",
    )
    parser.add_argument(
        "--trade-sequence-samples",
        type=int,
        default=5_000,
        help="diagnostic episode-sequence paths per resampling method",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="skip the six additional full backtest friction reruns",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    result, metrics = run_pipeline(config)
    artifacts = save_outputs(
        result,
        metrics,
        config,
        bootstrap_samples=args.bootstrap_samples,
        trade_sequence_samples=args.trade_sequence_samples,
        include_stress=not args.skip_stress,
    )
    print(metrics.to_string(index=False))
    print(f"\nProduction readiness: {artifacts['readiness_status']}")


if __name__ == "__main__":
    main()
