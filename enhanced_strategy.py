"""Multi-sleeve trend CTA: breadth, basis momentum, and leakage-safe training.

Changes to the adaptive TSMOM strategy, in the order they were adopted:

1. breadth: trade every investable USD contract in the supplied catalogue
   (43 markets across six asset classes) instead of 22, because diversification
   across weakly correlated trends is the most reliable Sharpe improvement
   available (Moskowitz-Ooi-Pedersen 2012 use 58 markets; Hurst-Ooi-Pedersen
   2017 use 67);
2. volatility targeting: a RiskMetrics EWMA estimator in place of the 63-day
   rolling window, so portfolio risk tracks its 10% target through regime
   changes instead of lagging them;
3. a second return source: basis momentum (Boons-Prado 2019), the change in
   roll yield, blended with trend at equal risk weight. It survived a
   pre-declared adoption rule that rejected seven other candidate sleeves —
   value, seasonality, skewness, cross-sectional momentum, hedging pressure,
   breakout, and volatility term structure — most of them because they merely
   restate the trend signal.

Two experiments are kept because their negative results are informative: a
rolling walk-forward selector over five pre-declared forecast models (genuine
out-of-sample training that loses to the simplest model it contains), and a
roll-yield carry sleeve (subsumed by basis momentum).

Everything downstream of the forecast (risk budgeting, volatility targeting,
shock de-risking, no-trade buffer, cost accounting) is shared and identical
for every sleeve and every candidate.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_cta import (
    BacktestConfig,
    BacktestResult,
    _base_target_positions,
    _gross_returns,
    _held_positions,
    _month_end_rows,
    _portfolio_leverage,
    contribution_by_class,
    load_metadata,
    load_prices,
    performance_metrics,
    run_backtest,
)
from institutional_strategy import (
    InstitutionalConfig,
    apply_no_trade_buffer,
    run_institutional_backtest,
)

# Every USD-denominated continuous contract in the supplied catalogue is
# included unless listed in EXCLUDED_CONTRACTS below: 56 USD contracts split
# into 43 members and 13 named exclusions. Thin markets are excluded by a
# numeric rule — median daily dollar volume below roughly $100M over the
# sample — rather than by adjective. Late-starting or thinly marked markets
# (6N, RB) additionally enter point-in-time only once the volume gate and the
# signal lookbacks are simultaneously valid.
EXPANDED_ASSET_CLASSES: dict[str, tuple[str, ...]] = {
    "Equity indices": ("EMD", "ES", "HTW", "NKD", "NQ", "RTY", "YM"),
    "Government bonds": ("ZT", "ZF", "ZN", "ZB"),
    "FX": ("6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S"),
    "Energy": ("BRN", "CL", "GAS", "HO", "NG", "RB"),
    "Metals": ("GC", "HG", "PA", "PL", "SI"),
    "Agriculture & livestock": (
        "CC", "CT", "GF", "HE", "KC", "KE", "LE",
        "SB", "ZC", "ZL", "ZM", "ZS", "ZW",
    ),
}

EXCLUDED_CONTRACTS: dict[str, str] = {
    "WBS": "ICE WTI duplicates CL",
    "LSU": "white sugar duplicates SB",
    "LRC": "robusta coffee duplicates KC",
    "MWE": "third wheat contract; ZW and KE retained",
    "GD": "S&P GSCI future is a redundant commodity basket",
    "DX": "the dollar index is a basket of currencies already in the universe",
    "VX": "VIX futures are not delta-one; roll decay belongs to a vol strategy",
    "ZQ": "fed funds price vol collapses at the zero lower bound; 1/vol sizing breaks",
    "LBS": "lumber: median daily dollar volume far below the $100M liquidity rule",
    "ZR": "rough rice: below the $100M liquidity rule",
    "DC": "milk: below the $100M liquidity rule",
    "OJ": "orange juice: ~$45M/day, below the $100M liquidity rule",
    "ZO": "oats: below the $100M liquidity rule",
}

# A market is tradeable on day t only if its trailing 60-day median reported
# volume exceeds this many contracts. This keeps positions out of markets with
# untradeable marks (6N shows zero reported volume on every 2005 session) and
# is fully point-in-time because it reads only past volumes.
MIN_MEDIAN_CONTRACTS: int = 1000
VOLUME_GATE_WINDOW: int = 60

# Alternative risk budget reported in the stress table: equal risk across the
# four traditional super-classes, with the commodity quarter split equally
# across energy, metals, and agriculture (Hurst-Ooi-Pedersen construction).
HIERARCHICAL_CLASS_WEIGHTS: dict[str, float] = {
    "Equity indices": 1 / 4,
    "Government bonds": 1 / 4,
    "FX": 1 / 4,
    "Energy": 1 / 12,
    "Metals": 1 / 12,
    "Agriculture & livestock": 1 / 12,
}


@dataclass(frozen=True)
class ForecastCandidate:
    """One pre-declared forecast model competing in the walk-forward selection."""

    name: str
    kind: str  # "sign" or "strength"
    horizons: tuple[int, ...]

    def validate(self) -> None:
        if self.kind not in ("sign", "strength"):
            raise ValueError(f"Unknown forecast kind: {self.kind}")
        if not self.horizons or min(self.horizons) < 5:
            raise ValueError("horizons must be at least a week of business days")


# The candidate set is fixed before evaluation: the pre-existing baseline, the
# literature's standard horizon blend, and their continuous-signal versions.
CANDIDATES: tuple[ForecastCandidate, ...] = (
    ForecastCandidate("Sign 12m", "sign", (252,)),
    ForecastCandidate("Sign 3/6/12m", "sign", (63, 126, 252)),
    ForecastCandidate("Strength 12m", "strength", (252,)),
    ForecastCandidate("Strength 3/6/12m", "strength", (63, 126, 252)),
    ForecastCandidate("Strength 1/3/6/12m", "strength", (21, 63, 126, 252)),
)


@dataclass(frozen=True)
class EnhancedConfig:
    """Walk-forward and signal settings; risk/execution reuse InstitutionalConfig.

    ``carry_weight`` blends the roll-gap carry signal with trend (0.5 = equal
    risk to the two return sources, the no-information prior). ``vol_decay``
    is the RiskMetrics EWMA decay for portfolio volatility targeting; None
    falls back to the v1 63-day rolling estimator.
    """

    selection_window_months: int = 60
    first_selection_year: int = 1990
    signal_vol_span: int = 60
    strength_cap: float = 2.0
    carry_weight: float = 0.5
    basis_weight: float = 0.5
    vol_decay: float | None = 0.94
    # Tested and reported, not adopted: per-market risk-managed sizing improved
    # both search windows but did not replicate on the reporting window.
    risk_managed_window: int | None = None
    candidates: tuple[ForecastCandidate, ...] = CANDIDATES

    def validate(self) -> None:
        if self.selection_window_months < 12:
            raise ValueError("selection_window_months must be at least a year")
        if self.strength_cap <= 0:
            raise ValueError("strength_cap must be positive")
        if not 0 <= self.carry_weight <= 1:
            raise ValueError("carry_weight must be in [0, 1]")
        if not 0 <= self.basis_weight <= 1:
            raise ValueError("basis_weight must be in [0, 1]")
        if self.vol_decay is not None and not 0 < self.vol_decay < 1:
            raise ValueError("vol_decay must be in (0, 1)")
        if len({candidate.name for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate names must be unique")
        for candidate in self.candidates:
            candidate.validate()


@dataclass
class WalkForwardResult:
    """Composite walk-forward strategy plus everything needed to audit it."""

    backtest: BacktestResult
    selections: pd.Series
    selection_scores: pd.DataFrame
    candidate_results: dict[str, BacktestResult]
    candidate_signals: dict[str, pd.DataFrame]
    ensemble: BacktestResult
    composite_signal: pd.DataFrame


def trend_strength(
    prices: pd.DataFrame,
    horizons: tuple[int, ...],
    vol_span: int,
    cap: float,
    normalization_window: int = 252,
) -> pd.DataFrame:
    """Average volatility-scaled trend strength across horizons, in [-1, 1].

    Following Baz et al. (2015), each horizon is normalized twice. The h-day
    price move is first divided by sigma_daily * sqrt(h) so trends are
    comparable across markets; that ratio is then divided by its own trailing
    one-year standard deviation so the clip at +/-cap is a genuine cap-sigma
    event, and so position size does not fall mechanically with volatility a
    second time (sizing already divides by dollar volatility once). All
    windows are trailing; back-adjusted price levels can be negative, so
    nothing here ever divides by a price level.
    """
    daily_vol = prices.diff().ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    scores = []
    for horizon in horizons:
        raw = (prices - prices.shift(horizon)) / (
            daily_vol.replace(0, np.nan) * math.sqrt(horizon)
        )
        z = raw / raw.rolling(
            normalization_window, min_periods=normalization_window
        ).std().replace(0, np.nan)
        scores.append(z.clip(-cap, cap) / cap)
    signal = sum(scores) / len(scores)
    valid = pd.concat(
        [score.notna() for score in scores], axis=1, keys=range(len(scores))
    ).T.groupby(level=1).all().T
    return signal.where(valid)


def candidate_signal(
    prices: pd.DataFrame,
    candidate: ForecastCandidate,
    config: EnhancedConfig,
) -> pd.DataFrame:
    candidate.validate()
    if candidate.kind == "sign":
        votes = [np.sign(prices - prices.shift(h)) for h in candidate.horizons]
        signal = sum(votes) / len(votes)
        valid = pd.concat(
            [(prices - prices.shift(h)).notna() for h in candidate.horizons],
            axis=1,
            keys=range(len(candidate.horizons)),
        ).T.groupby(level=1).all().T
        return signal.where(valid)
    return trend_strength(prices, candidate.horizons, config.signal_vol_span, config.strength_cap)


# The global extension (v2.5): every institutional-grade contract in the
# catalogue whose currency can be converted point-in-time using the FX futures
# already in the universe. STIRs stay excluded (zero-lower-bound sizing, the
# ZQ rule); NIY duplicates SNK, MHI duplicates HSI, FDAX9/FESX9/YAP4/YAP10 are
# venue duplicates; HSI/KOS/SSG have no FX conversion series in this dataset;
# AFB/AWM/LWB/LCC/GWM are thin or late agricultural contracts; EUA and FTDX
# are late and niche.
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

# USD per unit of foreign currency, proxied by the unadjusted currency-futures
# close. The 6J file quotes USD per 100 yen, hence the scale.
FX_SOURCE: dict[str, tuple[str, float]] = {
    "EUR": ("6E", 1.0),
    "GBP": ("6B", 1.0),
    "JPY": ("6J", 0.01),
    "CHF": ("6S", 1.0),
    "CAD": ("6C", 1.0),
    "AUD": ("6A", 1.0),
}


def load_global_metadata(data_dir: Path, symbols: list[str]) -> pd.DataFrame:
    """Catalogue rows for a multi-currency universe.

    Same catalogue parsing as ``load_metadata`` but without the USD-only
    check; every non-USD currency must instead have an FX conversion source.
    """
    catalogue = pd.read_csv(data_dir / "CATALOGUE_Delta1_Futures.csv")
    catalogue["clean_symbol"] = (
        catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    )
    wanted = set(symbols)
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
    """USD per foreign-currency unit for each convertible currency.

    The unadjusted currency-futures close stands in for spot; the futures
    basis (an interest differential of a percent or two a year) is a benign
    scaling error, and the series is observable at each close, so nothing
    here is forward-looking. Sanity bounds catch a mis-scaled file loudly.
    """
    plausible = {
        "EUR": (0.5, 2.5), "GBP": (0.8, 3.0), "JPY": (0.003, 0.02),
        "CHF": (0.3, 2.0), "CAD": (0.4, 1.5), "AUD": (0.3, 1.5),
    }
    rates = {}
    for currency, (symbol, scale) in FX_SOURCE.items():
        path = data_dir / "Futures Data" / f"&{symbol}.csv"
        series = pd.read_csv(path, usecols=["Date", "Close"], parse_dates=["Date"])
        series = series.drop_duplicates("Date", keep="last").sort_values("Date")
        rate = series.set_index("Date")["Close"] * scale
        low, high = plausible[currency]
        observed = rate.dropna()
        if not ((observed > low) & (observed < high)).all():
            raise ValueError(f"{currency} rate outside plausible bounds — check scaling")
        rates[currency] = rate.reindex(calendar).ffill(limit=10)
    frame = pd.DataFrame(rates, index=calendar)
    frame["USD"] = 1.0
    return frame


def usd_point_values(
    metadata: pd.DataFrame,
    fx: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Date x symbol frame of USD point values: native value times FX rate."""
    columns = {}
    for symbol, row in metadata.iterrows():
        columns[symbol] = fx[row["currency"]].reindex(calendar) * row["point_value"]
    return pd.DataFrame(columns, index=calendar)


def load_unadjusted_prices(
    data_dir: Path,
    symbols: list[str],
    ffill_limit: int = 10,
) -> pd.DataFrame:
    """Load the unadjusted continuous closes (&SYM, no back-adjustment).

    The difference between unadjusted and back-adjusted price changes isolates
    the roll gaps, which is what the carry signal is built from.
    """
    series = []
    for symbol in symbols:
        path = data_dir / "Futures Data" / f"&{symbol}.csv"
        frame = pd.read_csv(path, usecols=["Date", "Close"], parse_dates=["Date"])
        frame = frame.drop_duplicates("Date", keep="last").sort_values("Date")
        series.append(frame.set_index("Date")["Close"].rename(symbol))
    prices = pd.concat(series, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(prices.index.min(), prices.index.max())
    return prices.reindex(calendar).ffill(limit=ffill_limit)


def carry_strength(
    prices: pd.DataFrame,
    unadjusted: pd.DataFrame,
    vol_span: int,
    cap: float,
    normalization_window: int = 252,
) -> pd.DataFrame:
    """Roll-yield carry in [-1, 1], from roll gaps between the two series.

    At each contract roll the unadjusted series jumps by the calendar spread
    while the back-adjusted series does not, so (unadjusted change − adjusted
    change) summed over a trailing year measures the curve slope actually paid
    or earned: rolls that jump up (contango) are negative carry for a long.
    The sum is scaled by annualized dollar volatility to a unitless score,
    then normalized and clipped exactly like the trend-strength signal.
    Everything reads only past closes.
    """
    roll_gap = unadjusted.reindex_like(prices).diff() - prices.diff()
    daily_vol = prices.diff().ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    raw = -roll_gap.rolling(252, min_periods=126).sum() / (
        daily_vol.replace(0, np.nan) * math.sqrt(252)
    )
    z = raw / raw.rolling(
        normalization_window, min_periods=normalization_window
    ).std().replace(0, np.nan)
    return z.clip(-cap, cap) / cap


def basis_momentum(
    prices: pd.DataFrame,
    unadjusted: pd.DataFrame,
    vol_span: int,
    cap: float,
    roll_window: int = 252,
    lookback: int = 252,
    normalization_window: int = 252,
) -> pd.DataFrame:
    """Change in roll yield over a year — Boons & Prado (2019) basis momentum.

    Carry says where the curve *is*; basis momentum says where it is *going*.
    The roll yield earned over the trailing year is differenced against the
    year before it — two non-overlapping years of roll history — so a market
    whose curve is moving toward backwardation scores positive. This predicts
    futures returns beyond both price momentum and the carry level, and it is
    the one sleeve of eight tested here that earned its place.

    Same house conventions as every other signal: roll gaps come from the
    unadjusted-minus-back-adjusted price change, scaling is by annualized
    dollar volatility, every window is trailing, and no price level is ever a
    denominator.

    The two legs are differenced in price units and scaled by a single common
    volatility. Scaling each leg by the volatility of its own era instead
    would leave a carry-level-times-volatility-drift term in the signal, which
    is a different effect wearing this one's name.
    """
    roll_gap = unadjusted.reindex_like(prices).diff() - prices.diff()
    daily_vol = prices.diff().ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    roll_sum = -roll_gap.rolling(roll_window, min_periods=roll_window).sum()
    raw = (roll_sum - roll_sum.shift(lookback)) / (
        daily_vol.replace(0, np.nan) * math.sqrt(252)
    )
    z = raw / raw.rolling(
        normalization_window, min_periods=normalization_window
    ).std().replace(0, np.nan)
    return z.clip(-cap, cap) / cap


def blend_sleeves(
    anchor: pd.DataFrame,
    sleeves: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Combine forecast sleeves at fixed risk weights.

    ``anchor`` defines coverage: a market runs on the anchor alone until the
    other sleeves become estimable, and never trades while the anchor itself
    is unestimable. Weights are pre-declared, never fitted.
    """
    total = sum(weights.values())
    combined = sum(
        sleeves[name].fillna(anchor) * weight for name, weight in weights.items()
    ) / total
    return combined.clip(-1, 1).where(anchor.notna())


def blend_trend_and_carry(
    trend: pd.DataFrame,
    carry: pd.DataFrame,
    carry_weight: float,
) -> pd.DataFrame:
    """Risk-blend trend with carry; a market runs on trend alone until its
    carry becomes estimable (about 1.5 years of roll history)."""
    return blend_sleeves(
        trend,
        {"trend": trend, "carry": carry},
        {"trend": 1 - carry_weight, "carry": carry_weight},
    )


def load_volumes(data_dir: Path, symbols: list[str]) -> pd.DataFrame:
    """Load reported contract volumes aligned to the price calendar."""
    series = []
    for symbol in symbols:
        path = data_dir / "Futures Data" / f"&{symbol}_CCB.csv"
        frame = pd.read_csv(path, usecols=["Date", "Volume"], parse_dates=["Date"])
        frame = frame.drop_duplicates("Date", keep="last").sort_values("Date")
        series.append(frame.set_index("Date")["Volume"].rename(symbol))
    volumes = pd.concat(series, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(volumes.index.min(), volumes.index.max())
    return volumes.reindex(calendar)


def tradeable_mask(volumes: pd.DataFrame) -> pd.DataFrame:
    """True where the trailing median reported volume clears the gate.

    The median over 60 trailing sessions ignores isolated holiday closures, so
    a market is not ejected by a one-week exchange break, but a market whose
    marks carry no reported volume (6N through 2005-06) stays out until real
    trading appears.
    """
    trailing = volumes.fillna(0.0).rolling(
        VOLUME_GATE_WINDOW, min_periods=VOLUME_GATE_WINDOW
    ).median()
    return trailing > MIN_MEDIAN_CONTRACTS


def _shock_multiplier(prices: pd.DataFrame, config: InstitutionalConfig) -> pd.DataFrame:
    """Fast/slow volatility-ratio de-risking, identical to the institutional layer."""
    price_change = prices.diff()
    fast = price_change.ewm(
        span=config.fast_vol_span, min_periods=config.fast_vol_span, adjust=False
    ).std()
    slow = price_change.ewm(
        span=config.slow_vol_span, min_periods=config.slow_vol_span, adjust=False
    ).std()
    ratio = fast / slow.replace(0, np.nan)
    progress = ((ratio - config.shock_start) / (config.shock_full - config.shock_start)).clip(0, 1)
    return (1 - progress * (1 - config.shock_floor)).where(ratio.notna())


def risk_managed_forecast(
    forecast: pd.DataFrame,
    prices: pd.DataFrame,
    vol_span: int,
    window: int = 126,
    cap: float = 2.0,
) -> pd.DataFrame:
    """Scale each market's forecast by the inverse volatility of its own P&L.

    Barroso & Santa-Clara (2015) manage momentum's crashes with the strategy's
    own realized volatility rather than the asset's. Applied per market, this
    cuts risk in a market whose trend book has turned erratic even when its
    price volatility alone would not say so — a distinction the portfolio-level
    volatility target cannot see, because it only observes the aggregate.

    ``u`` is the market's strategy return in risk units: yesterday's forecast
    times today's price change over yesterday's volatility. Both inputs are
    lagged, so the scale applied on day t is known at the close of t-1.
    """
    daily_vol = prices.diff().ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    strategy_return = (
        forecast.shift(1) * prices.diff() / daily_vol.shift(1).replace(0, np.nan)
    )
    realized = strategy_return.pow(2).rolling(window, min_periods=window).mean() ** 0.5
    weight = (1.0 / realized.replace(0, np.nan)).clip(upper=cap)
    return (forecast * weight).clip(-1, 1).where(forecast.notna())


def _ewma_portfolio_leverage(
    base_gross_returns: pd.Series,
    config: BacktestConfig,
    decay: float,
) -> pd.Series:
    """RiskMetrics EWMA volatility targeting (lambda = decay, e.g. 0.94).

    The exponential estimator reacts to volatility shifts within weeks instead
    of dragging a 63-day equal-weight window, which is the entire content of
    the upgrade; the target, clipping, and monthly activation are unchanged.
    """
    realized_vol = base_gross_returns.ewm(
        alpha=1 - decay, min_periods=20
    ).std() * math.sqrt(config.annualization)
    ratio = config.target_vol / realized_vol.replace(0.0, np.nan)
    return ratio.clip(config.min_leverage, config.max_leverage).fillna(1.0)


def run_signal_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    base_config: BacktestConfig,
    inst_config: InstitutionalConfig,
    *,
    name: str,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    class_weights: dict[str, float] | None = None,
    tradeable: pd.DataFrame | None = None,
    cost_multiplier: pd.Series | None = None,
    vol_decay: float | None = None,
    risk_managed_window: int | None = None,
    point_values: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run any forecast through the shared risk-execution-accounting pipeline."""
    asset_classes = asset_classes or EXPANDED_ASSET_CLASSES
    shock = _shock_multiplier(prices, inst_config)
    forecast = (signal * shock).clip(-1, 1)
    if risk_managed_window is not None:
        forecast = risk_managed_forecast(
            forecast, prices, base_config.vol_span, window=risk_managed_window
        )
    if tradeable is not None:
        forecast = forecast.where(tradeable.reindex_like(forecast), np.nan)

    base_target = _base_target_positions(
        prices, forecast, metadata, base_config, asset_classes, class_weights,
        point_values=point_values,
    )
    base_monthly = _month_end_rows(base_target)
    base_positions = _held_positions(base_monthly, prices.index)
    base_gross = _gross_returns(base_positions, prices, metadata, point_values=point_values)
    if vol_decay is not None:
        leverage = _ewma_portfolio_leverage(base_gross, base_config, vol_decay)
    else:
        leverage = _portfolio_leverage(base_gross, base_config)
    # Until the trailing vol window holds only live history, realized vol is
    # diluted by structural zeros and would overstate the safe leverage; stay
    # at neutral 1.0 through the strategy's first vol window instead.
    live = base_positions.abs().sum(axis=1).gt(0).cummax()
    warmed_up = live & live.shift(base_config.portfolio_vol_window).fillna(False)
    leverage = leverage.where(warmed_up, 1.0)
    monthly_leverage = _month_end_rows(leverage)
    desired = base_monthly.mul(monthly_leverage.reindex(base_monthly.index), axis=0)

    buffered = apply_no_trade_buffer(desired, inst_config.no_trade_buffer)
    positions = _held_positions(buffered, prices.index)
    gross = _gross_returns(positions, prices, metadata, point_values=point_values)
    turnover = positions.diff().abs().fillna(positions.abs())
    if point_values is not None:
        # the half-tick spread is paid in native currency, so its USD cost
        # moves with the FX rate exactly as the point value does.
        one_way_cost = (
            base_config.half_spread_ticks
            * point_values.mul(metadata["tick_size"], axis=1)
            + base_config.commission_per_contract
        )
        if cost_multiplier is not None:
            one_way_cost = one_way_cost.mul(cost_multiplier, axis=1).fillna(one_way_cost)
        costs = (turnover * one_way_cost).sum(axis=1)
    else:
        one_way_cost = (
            base_config.half_spread_ticks * metadata["tick_size"] * metadata["point_value"]
            + base_config.commission_per_contract
        )
        if cost_multiplier is not None:
            one_way_cost = one_way_cost * cost_multiplier.reindex(one_way_cost.index).fillna(1.0)
        costs = turnover.mul(one_way_cost, axis=1).sum(axis=1)
    net = gross - costs

    daily = pd.DataFrame(
        {
            "gross_return": gross,
            "cost": costs,
            "net_return": net,
            "leverage": monthly_leverage.reindex(prices.index).ffill().shift(1).fillna(1.0),
            "contract_turnover_per_dollar": turnover.sum(axis=1),
        }
    )
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    daily["gross_equity"] = (1 + daily["gross_return"]).cumprod()
    return BacktestResult(name, daily, positions, buffered, forecast, prices, metadata)


def _trailing_sharpe(returns: pd.Series, end: pd.Timestamp, months: int) -> float:
    monthly = (1 + returns.loc[:end]).resample("ME").prod() - 1
    window = monthly.iloc[-months:]
    if len(window) < months or window.std(ddof=1) == 0:
        return np.nan
    return float(window.mean() / window.std(ddof=1) * math.sqrt(12))


def walk_forward_selection(
    candidate_returns: dict[str, pd.Series],
    config: EnhancedConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Choose next year's forecast model from trailing net Sharpe, once a year.

    The selection at the end of year Y uses only candidate returns up to and
    including December of year Y; the chosen model is applied to all of year
    Y+1. Returns the year -> candidate mapping and the score table behind it.
    """
    config.validate()
    calendar = next(iter(candidate_returns.values())).index
    years = range(config.first_selection_year, calendar[-1].year + 1)
    selections, score_rows = {}, []
    for year in years:
        cutoff = pd.Timestamp(year=year, month=12, day=31)
        scores = {
            name: _trailing_sharpe(returns, cutoff, config.selection_window_months)
            for name, returns in candidate_returns.items()
        }
        score_rows.append({"Selection year-end": year, **scores})
        valid = {name: score for name, score in scores.items() if not np.isnan(score)}
        if not valid:
            continue
        selections[year + 1] = max(valid, key=lambda name: (valid[name], name))
    return (
        pd.Series(selections, name="Selected model"),
        pd.DataFrame(score_rows).set_index("Selection year-end"),
    )


def stitch_signals(
    signals: dict[str, pd.DataFrame],
    selections: pd.Series,
) -> pd.DataFrame:
    """Assemble the composite forecast: each year uses last year's selection."""
    calendar = next(iter(signals.values()))
    composite = pd.DataFrame(np.nan, index=calendar.index, columns=calendar.columns)
    for year, name in selections.items():
        rows = composite.index.year == year
        composite.loc[rows] = signals[name].loc[rows]
    return composite


def ensemble_signal(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight forecast combination across all candidates."""
    return sum(signals.values()) / len(signals)


def run_walk_forward(
    base_config: BacktestConfig,
    enhanced_config: EnhancedConfig | None = None,
    inst_config: InstitutionalConfig | None = None,
    *,
    prices: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    tradeable: pd.DataFrame | None = None,
    point_values: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run candidates, select annually out-of-sample, and account the composite."""
    config = enhanced_config or EnhancedConfig()
    config.validate()
    inst = inst_config or InstitutionalConfig()
    classes = asset_classes or EXPANDED_ASSET_CLASSES
    symbols = [symbol for members in classes.values() for symbol in members]
    metadata = metadata if metadata is not None else load_metadata(base_config.data_dir, symbols)
    if prices is None:
        prices = load_prices(base_config.data_dir, symbols, ffill_limit=10)
    if tradeable is None:
        tradeable = tradeable_mask(
            load_volumes(base_config.data_dir, symbols)
        ).reindex(prices.index).fillna(False)

    signals = {c.name: candidate_signal(prices, c, config) for c in config.candidates}
    candidate_results = {
        name: run_signal_backtest(
            signal, prices, metadata, base_config, inst,
            name=name, asset_classes=classes, tradeable=tradeable,
            vol_decay=config.vol_decay, point_values=point_values,
        )
        for name, signal in signals.items()
    }
    candidate_returns = {
        name: result.daily["net_return"] for name, result in candidate_results.items()
    }

    selections, scores = walk_forward_selection(candidate_returns, config)
    composite_signal = stitch_signals(signals, selections)
    n_markets = len(symbols)
    composite = run_signal_backtest(
        composite_signal, prices, metadata, base_config, inst,
        name=f"Walk-forward TSMOM ({n_markets} markets)",
        asset_classes=classes, tradeable=tradeable, vol_decay=config.vol_decay,
        point_values=point_values,
    )
    ensemble = run_signal_backtest(
        ensemble_signal(signals), prices, metadata, base_config, inst,
        name=f"Ensemble TSMOM ({n_markets} markets)",
        asset_classes=classes, tradeable=tradeable, vol_decay=config.vol_decay,
        point_values=point_values,
    )
    return WalkForwardResult(
        backtest=composite,
        selections=selections,
        selection_scores=scores,
        candidate_results=candidate_results,
        candidate_signals=signals,
        ensemble=ensemble,
        composite_signal=composite_signal,
    )


# Evaluation windows. 2005-2014 was already reported for the v1 strategy, so it
# is a second look rather than pristine out-of-sample data; 1990-2004 is the
# primary evidence window for every design choice made in this iteration, and a
# result is only claimed when both windows agree.
PRIMARY_WINDOW: tuple[str, str] = ("1990-01-01", "2004-12-31")
HOLDOUT_WINDOW: tuple[str, str] = ("2005-01-01", "2014-12-31")


def _monthly_returns(returns: pd.Series, start: str, end: str) -> pd.Series:
    daily = returns.loc[start:end]
    return ((1 + daily).resample("ME").prod() - 1).dropna()


def _monthly_sharpe(returns: pd.Series, start: str, end: str) -> float:
    monthly = _monthly_returns(returns, start, end)
    return float(monthly.mean() / monthly.std(ddof=1) * math.sqrt(12))


def paired_sharpe_difference(
    strategy: pd.Series,
    baseline: pd.Series,
    start: str,
    end: str,
    *,
    samples: int = 2000,
    block_months: int = 6,
    random_state: int = 7,
) -> pd.Series:
    """Block-bootstrap the Sharpe difference, resampling both series together.

    Sampling the same month blocks from both strategies preserves their
    correlation, which is what makes the difference test meaningful. The
    interval is on annualized monthly Sharpe.
    """
    joint = pd.concat(
        {
            "strategy": _monthly_returns(strategy, start, end),
            "baseline": _monthly_returns(baseline, start, end),
        },
        axis=1,
    ).dropna()
    matrix = joint.to_numpy()
    n = len(matrix)
    rng = np.random.default_rng(random_state)
    starts = np.arange(max(1, n - block_months + 1))

    def sharpe(values: np.ndarray) -> float:
        return values.mean() / values.std(ddof=1) * math.sqrt(12)

    differences = np.empty(samples)
    for sample in range(samples):
        picks: list[int] = []
        while len(picks) < n:
            block_start = int(rng.choice(starts))
            picks.extend(range(block_start, min(block_start + block_months, n)))
        draw = matrix[picks[:n]]
        differences[sample] = sharpe(draw[:, 0]) - sharpe(draw[:, 1])
    observed = sharpe(matrix[:, 0]) - sharpe(matrix[:, 1])
    return pd.Series(
        {
            "Observed Sharpe difference": observed,
            "Bootstrap 5%": float(np.quantile(differences, 0.05)),
            "Bootstrap 50%": float(np.quantile(differences, 0.50)),
            "Bootstrap 95%": float(np.quantile(differences, 0.95)),
            "P(difference > 0)": float((differences > 0).mean()),
            "Months": n,
        }
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    start: str,
    end: str,
    benchmark_sharpe: float,
) -> float:
    """Bailey & Lopez de Prado PSR: P(true Sharpe > benchmark), non-normal safe.

    Computed on monthly returns against the annualized benchmark Sharpe, using
    the skewness/kurtosis-adjusted standard error of the Sharpe estimator.
    """
    monthly = _monthly_returns(returns, start, end)
    n = len(monthly)
    sharpe_monthly = monthly.mean() / monthly.std(ddof=1)
    benchmark_monthly = benchmark_sharpe / math.sqrt(12)
    skew = float(monthly.skew())
    kurt = float(monthly.kurt()) + 3
    standard_error = math.sqrt(
        (1 - skew * sharpe_monthly + (kurt - 1) / 4 * sharpe_monthly**2) / (n - 1)
    )
    z = (sharpe_monthly - benchmark_monthly) / standard_error
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def deflated_sharpe_ratio(
    returns: pd.Series,
    start: str,
    end: str,
    *,
    n_trials: int,
    trial_sharpe_std: float = 0.25,
) -> float:
    """Bailey & Lopez de Prado DSR: PSR against the max-of-N-trials benchmark.

    The benchmark Sharpe is the expected maximum of ``n_trials`` zero-skill
    strategies whose annualized Sharpes scatter with ``trial_sharpe_std``; a
    DSR near one means the observed Sharpe is unlikely to be the lucky best of
    the search. ``n_trials`` must honestly count every variant evaluated on
    the window, including the v1 robustness grid.
    """
    euler_gamma = 0.5772156649015329
    max_z = (1 - euler_gamma) * _normal_ppf(1 - 1 / n_trials) + euler_gamma * _normal_ppf(
        1 - 1 / (n_trials * math.e)
    )
    benchmark_annual = trial_sharpe_std * max_z
    return probabilistic_sharpe_ratio(returns, start, end, benchmark_annual)


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF via bisection; avoids a scipy dependency."""
    low, high = -10.0, 10.0
    for _ in range(80):
        mid = (low + high) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cscv_pbo(
    candidate_returns: dict[str, pd.Series],
    start: str,
    end: str,
    n_partitions: int = 16,
) -> float:
    """Probability of backtest overfitting via combinatorially symmetric CV.

    Bailey et al. (2017): split the monthly return matrix into ``n_partitions``
    blocks; for every half/half combination, pick the best candidate in-sample
    and record its out-of-sample rank. PBO is the fraction of combinations in
    which the in-sample winner lands in the bottom half out-of-sample.
    """
    from itertools import combinations

    monthly = pd.DataFrame(
        {name: _monthly_returns(series, start, end) for name, series in candidate_returns.items()}
    ).dropna()
    blocks = np.array_split(np.arange(len(monthly)), n_partitions)
    below_median = 0
    total = 0
    for in_sample_ids in combinations(range(n_partitions), n_partitions // 2):
        in_rows = np.concatenate([blocks[i] for i in in_sample_ids])
        out_rows = np.concatenate(
            [blocks[i] for i in range(n_partitions) if i not in in_sample_ids]
        )
        in_sharpe = monthly.iloc[in_rows].mean() / monthly.iloc[in_rows].std(ddof=1)
        out_sharpe = monthly.iloc[out_rows].mean() / monthly.iloc[out_rows].std(ddof=1)
        winner = in_sharpe.idxmax()
        rank = (out_sharpe < out_sharpe[winner]).sum() / (len(out_sharpe) - 1)
        # the conservative convention counts an exactly-median rank as overfit
        below_median += rank <= 0.5
        total += 1
    return below_median / total


def yearly_comparison(
    strategy: pd.Series,
    baseline: pd.Series,
    start: str,
    end: str,
    labels: tuple[str, str],
) -> pd.DataFrame:
    """Calendar-year net returns side by side with a win flag."""
    table = pd.concat(
        {
            labels[0]: (1 + strategy.loc[start:end]).resample("YE").prod() - 1,
            labels[1]: (1 + baseline.loc[start:end]).resample("YE").prod() - 1,
        },
        axis=1,
    ).dropna()
    table.index = table.index.year
    table["Winner"] = np.where(table[labels[0]] > table[labels[1]], labels[0], labels[1])
    return table


def headline_comparison_table(
    results: list[BacktestResult],
    windows: dict[str, tuple[str, str]],
    start_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Key metrics for every strategy in every evaluation window.

    ``start_overrides`` maps a strategy name to its earliest meaningful date —
    the walk-forward composite has no exposure before its seeding rule allows
    a first selection, so measuring it from 1990 would count dead years.
    The "From" column shows the start each row was actually measured from.
    """
    rows = []
    for window_name, (start, end) in windows.items():
        for result in results:
            row_start = max(start, (start_overrides or {}).get(result.name, start))
            metrics = performance_metrics(result, row_start, end)
            rows.append(
                {
                    "Window": window_name,
                    "Strategy": result.name,
                    "From": metrics["Start"],
                    "CAGR": metrics["CAGR"],
                    "Volatility": metrics["Annualized volatility"],
                    "Sharpe": metrics["Sharpe (rf=0)"],
                    "Max drawdown": metrics["Max drawdown"],
                    "Calmar": metrics["Calmar"],
                    "Annual cost drag": metrics["Annual cost drag"],
                }
            )
    return pd.DataFrame(rows).set_index(["Window", "Strategy"])


# Markets whose half-tick spread assumption is least reliable; the stress
# table charges them five times the estimated cost.
THIN_MARKETS: tuple[str, ...] = (
    "PA", "EMD", "NKD", "GF", "6N", "HTW",
    "FGBX", "SJB", "CGB", "SXF", "RS",
)

BOTH_WINDOWS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("1990-2004", PRIMARY_WINDOW),
    ("2005-2014", HOLDOUT_WINDOW),
)


def _window_rows(result: BacktestResult, label: str) -> list[dict]:
    rows = []
    for window_name, (start, end) in BOTH_WINDOWS:
        metrics = performance_metrics(result, start, end)
        rows.append(
            {
                "Scenario": label,
                "Window": window_name,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe (rf=0)"],
                "Max drawdown": metrics["Max drawdown"],
                "Annual cost drag": metrics["Annual cost drag"],
            }
        )
    return rows


def absorption_ratio(
    prices: pd.DataFrame,
    vol_span: int = 60,
    window: int = 252,
    min_live: int = 200,
) -> pd.Series:
    """Share of variance explained by the top two principal components.

    Kritzman et al. (2011) systemic-risk measure, computed each month-end from
    the trailing year of volatility-normalized price changes. Values near one
    mean the markets have collapsed onto a couple of common factors.
    """
    daily_vol = prices.diff().ewm(span=vol_span, min_periods=vol_span, adjust=False).std()
    normalized = (prices.diff() / daily_vol.replace(0, np.nan)).clip(-5, 5)
    month_ends = _month_end_rows(pd.Series(1.0, index=prices.index)).index
    values = {}
    for date in month_ends:
        trailing = normalized.loc[:date].tail(window)
        live = trailing.columns[trailing.notna().sum() > min_live]
        if len(live) < 10:
            continue
        eigenvalues = np.linalg.eigvalsh(trailing[live].corr().to_numpy())
        values[date] = float(eigenvalues[-2:].sum() / eigenvalues.sum())
    return pd.Series(values, name="Absorption ratio")


def absorption_overlay(
    signal: pd.DataFrame,
    ratio: pd.Series,
    quantile: float = 0.90,
    cut: float = 0.5,
) -> pd.DataFrame:
    """Halve exposure when the absorption ratio exceeds its trailing quantile.

    The trigger compares each month-end value only with its own past, so the
    overlay is point-in-time; it exists as a tested (and rejected) exhibit.
    """
    triggered = ratio.expanding(60).apply(
        lambda history: float(history.iloc[-1] > np.quantile(history[:-1], quantile))
    ).fillna(0.0)
    multiplier = pd.Series(
        np.where(triggered > 0, cut, 1.0), index=ratio.index
    ).reindex(signal.index).ffill().fillna(1.0)
    return signal.mul(multiplier, axis=0)


def enhanced_stress_tests(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    base_config: BacktestConfig,
    signal: pd.DataFrame,
    tradeable: pd.DataFrame,
    *,
    vol_decay: float | None = None,
    signal_variants: dict[str, pd.DataFrame] | None = None,
    class_variants: dict[str, dict[str, tuple[str, ...]]] | None = None,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    point_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Perturb costs, buffers, shock control, sizing, budget, and signal mix.

    Every scenario is evaluated in both windows so a perturbation that only
    works in the reused 2005-2014 window is visible as such.
    """
    def costs_times(multiple: float) -> BacktestConfig:
        return replace(
            base_config,
            half_spread_ticks=base_config.half_spread_ticks * multiple,
            commission_per_contract=base_config.commission_per_contract * multiple,
        )

    thin_multiplier = pd.Series(
        {symbol: 5.0 if symbol in THIN_MARKETS else 1.0 for symbol in prices.columns}
    )
    scenarios: list[tuple[str, dict]] = [
        ("Base", {}),
        ("63-day rolling vol targeting (v2.0)", {"vol_decay": None}),
        ("No trade buffer", {"inst": InstitutionalConfig(no_trade_buffer=0.0)}),
        ("Wide 40% buffer", {"inst": InstitutionalConfig(no_trade_buffer=0.40)}),
        ("No volatility shock control", {"inst": InstitutionalConfig(shock_floor=1.0)}),
        ("Hierarchical 4-super-class budget", {"class_weights": HIERARCHICAL_CLASS_WEIGHTS}),
        ("Sizing vol span 30", {"base": replace(base_config, vol_span=30)}),
        ("Sizing vol span 90", {"base": replace(base_config, vol_span=90)}),
        ("2x trading costs", {"base": costs_times(2)}),
        ("3x trading costs", {"base": costs_times(3)}),
        ("5x costs on thin markets", {"cost_multiplier": thin_multiplier}),
    ]
    for label, variant_signal in (signal_variants or {}).items():
        scenarios.append((label, {"signal": variant_signal}))
    for label, classes in (class_variants or {}).items():
        scenarios.append((label, {"asset_classes": classes}))
    rows = []
    for label, overrides in scenarios:
        scenario_base = overrides.get("base", base_config)
        result = run_signal_backtest(
            overrides.get("signal", signal),
            prices,
            metadata,
            scenario_base,
            overrides.get("inst", InstitutionalConfig()),
            name=label,
            asset_classes=overrides.get("asset_classes", asset_classes),
            class_weights=overrides.get("class_weights"),
            tradeable=tradeable,
            cost_multiplier=overrides.get("cost_multiplier"),
            vol_decay=overrides.get("vol_decay", vol_decay),
            point_values=point_values,
        )
        rows.extend(_window_rows(result, label))
    return pd.DataFrame(rows).set_index(["Scenario", "Window"])


def no_judgment_universe_row(
    base_config: BacktestConfig,
    enhanced_config: EnhancedConfig,
) -> pd.DataFrame:
    """The screen-free variant: every USD contract except the five duplicates.

    Shows how much work the liquidity/basket screens actually do. The
    point-in-time volume gate still applies because it is part of the strategy
    definition, not of universe membership.
    """
    extra = {
        "Equity indices": ("VX",),
        "Government bonds": ("ZQ",),
        "FX": ("DX",),
        "Agriculture & livestock": ("LBS", "ZR", "DC", "OJ", "ZO"),
    }
    classes = {
        name: members + extra.get(name, ())
        for name, members in EXPANDED_ASSET_CLASSES.items()
    }
    symbols = [s for members in classes.values() for s in members]
    prices = load_prices(base_config.data_dir, symbols, ffill_limit=10)
    metadata = load_metadata(base_config.data_dir, symbols)
    tradeable = tradeable_mask(load_volumes(base_config.data_dir, symbols)).reindex(
        prices.index
    ).fillna(False)
    # Same trend-only forecast as the Base stress row, so the delta between
    # the two rows isolates the universe screens and nothing else.
    sign12 = next(c for c in enhanced_config.candidates if c.name == "Sign 12m")
    signal = candidate_signal(prices, sign12, enhanced_config)
    result = run_signal_backtest(
        signal, prices, metadata, base_config, InstitutionalConfig(),
        name="No-judgment universe", asset_classes=classes, tradeable=tradeable,
        vol_decay=enhanced_config.vol_decay,
    )
    return pd.DataFrame(
        _window_rows(result, "No-judgment universe (51 markets)")
    ).set_index(["Scenario", "Window"])


def leave_one_class_out(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    base_config: BacktestConfig,
    signal: pd.DataFrame,
    tradeable: pd.DataFrame,
    *,
    vol_decay: float | None = None,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    point_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Headline Sharpe when each asset class is removed entirely.

    If the improvement over the 22-market baseline came from one lucky class,
    dropping that class would erase it; roughly stable Sharpes across rows
    mean the gain is diversification, not a single bet.
    """
    universe = asset_classes or EXPANDED_ASSET_CLASSES
    rows = []
    for dropped in universe:
        classes = {
            name: members
            for name, members in universe.items()
            if name != dropped
        }
        result = run_signal_backtest(
            signal, prices, metadata, base_config, InstitutionalConfig(),
            name=f"Without {dropped}", asset_classes=classes, tradeable=tradeable,
            vol_decay=vol_decay, point_values=point_values,
        )
        for window_name, (start, end) in BOTH_WINDOWS:
            metrics = performance_metrics(result, start, end)
            rows.append(
                {
                    "Dropped class": dropped,
                    "Window": window_name,
                    "Sharpe": metrics["Sharpe (rf=0)"],
                    "CAGR": metrics["CAGR"],
                }
            )
    return pd.DataFrame(rows).set_index(["Dropped class", "Window"])


def effective_breadth(result: BacktestResult, start: str, end: str) -> pd.Series:
    """Effective number of independent bets from P&L correlations.

    N_eff = N / (1 + (N - 1) * rho_bar), where rho_bar is the average pairwise
    correlation of per-market P&L streams. Five correlated equity indices are
    fewer than five bets; this reports how many they actually are.
    """
    pnl = result.prices.diff().mul(result.metadata["point_value"], axis=1)
    contributions = (result.positions * pnl).loc[start:end]
    active = contributions.columns[contributions.abs().sum() > 0]
    correlation = contributions[active].corr()
    n = len(active)
    off_diagonal = correlation.to_numpy()[~np.eye(n, dtype=bool)]
    rho_bar = float(np.nanmean(off_diagonal))
    return pd.Series(
        {
            "Markets traded": n,
            "Average pairwise P&L correlation": rho_bar,
            "Effective independent bets": n / (1 + (n - 1) * rho_bar),
        }
    )


def factorial_attribution(
    base_config: BacktestConfig,
    enhanced_config: EnhancedConfig,
    prices43: pd.DataFrame,
    metadata43: pd.DataFrame,
    tradeable43: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose the gain: universe (22 vs 43) x forecast (sign 12m vs blend).

    The 22-market cells use the v1 universe and its original 4-class budget so
    the universe dimension isolates exactly what changed.
    """
    from delta1_cta import ASSET_CLASSES

    symbols22 = [s for members in ASSET_CLASSES.values() for s in members]
    prices22 = load_prices(base_config.data_dir, symbols22)
    metadata22 = load_metadata(base_config.data_dir, symbols22)

    sign12 = enhanced_config.candidates[0]
    blend = next(c for c in enhanced_config.candidates if c.name == "Strength 3/6/12m")
    cells = [
        ("22 markets", "Sign 12m", prices22, metadata22, ASSET_CLASSES, None),
        ("22 markets", "Strength 3/6/12m", prices22, metadata22, ASSET_CLASSES, None),
        ("43 markets", "Sign 12m", prices43, metadata43, EXPANDED_ASSET_CLASSES, tradeable43),
        ("43 markets", "Strength 3/6/12m", prices43, metadata43, EXPANDED_ASSET_CLASSES, tradeable43),
    ]
    rows = []
    for universe, forecast_name, prices, metadata, classes, tradeable in cells:
        candidate = sign12 if forecast_name == "Sign 12m" else blend
        signal = candidate_signal(prices, candidate, enhanced_config)
        result = run_signal_backtest(
            signal, prices, metadata, base_config, InstitutionalConfig(),
            name=f"{universe} / {forecast_name}", asset_classes=classes,
            tradeable=tradeable, vol_decay=enhanced_config.vol_decay,
        )
        for window_name, (start, end) in BOTH_WINDOWS:
            metrics = performance_metrics(result, start, end)
            rows.append(
                {
                    "Universe": universe,
                    "Forecast": forecast_name,
                    "Window": window_name,
                    "Sharpe": metrics["Sharpe (rf=0)"],
                    "CAGR": metrics["CAGR"],
                }
            )
    return pd.DataFrame(rows).set_index(["Universe", "Forecast", "Window"])


def selection_window_sensitivity(
    base_config: BacktestConfig,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    tradeable: pd.DataFrame,
    windows_months: tuple[int, ...] = (36, 60, 120),
    *,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    point_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk-forward results under different pre-declared selection windows."""
    rows = []
    for months in windows_months:
        result = run_walk_forward(
            base_config,
            EnhancedConfig(selection_window_months=months),
            prices=prices,
            metadata=metadata,
            tradeable=tradeable,
            asset_classes=asset_classes,
            point_values=point_values,
        )
        for window_name, (start, end) in (
            ("1992-2004", ("1992-01-01", "2004-12-31")),
            ("2005-2014", HOLDOUT_WINDOW),
        ):
            metrics = performance_metrics(result.backtest, start, end)
            rows.append(
                {
                    "Selection window": f"{months}m",
                    "Window": window_name,
                    "CAGR": metrics["CAGR"],
                    "Sharpe": metrics["Sharpe (rf=0)"],
                    "Max drawdown": metrics["Max drawdown"],
                }
            )
    return pd.DataFrame(rows).set_index(["Selection window", "Window"])


@dataclass
class EnhancedPipelineOutput:
    """Everything the notebook and README report, in one place."""

    headline: BacktestResult
    walk_forward: WalkForwardResult
    baseline_adaptive: BacktestResult
    baseline_tsmom: BacktestResult
    long_only: BacktestResult
    comparison: pd.DataFrame
    bootstrap: pd.DataFrame
    psr: pd.Series
    dsr: pd.Series
    pbo: float
    yearly: pd.DataFrame
    stress: pd.DataFrame
    selection_sensitivity: pd.DataFrame
    class_contributions: pd.DataFrame
    crisis_robustness: pd.DataFrame
    leave_one_out: pd.DataFrame
    breadth: pd.DataFrame
    factorial: pd.DataFrame
    claim: pd.DataFrame
    carry_variant: BacktestResult
    trend_only: BacktestResult
    sleeve_increment: pd.DataFrame


def claim_rule_evaluation(
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    headline_name: str,
) -> pd.Series:
    """Evaluate the pre-registered claim rule and return its verdict.

    Leg (a): the 90% paired-bootstrap interval of the 1990-2004 Sharpe
    difference excludes zero. Leg (b): the 2005-2014 point estimate is
    positive and the headline wins at least 6 of 10 calendar years.
    """
    leg_a = bool(bootstrap.loc["1990-2004", "Bootstrap 5%"] > 0)
    eval_years = yearly.loc[2005:2014]
    wins = int((eval_years["Winner"] == headline_name).sum())
    leg_b = bool(bootstrap.loc["2005-2014", "Observed Sharpe difference"] > 0) and wins >= 6
    return pd.Series(
        {
            "Leg (a): primary-window 90% CI excludes zero": leg_a,
            "Leg (b): second-use difference positive with >= 6/10 wins": leg_b,
            "Winning years 2005-2014": wins,
            "Improvement claimed": leg_a and leg_b,
        },
        name="Claim rule",
    )


def crisis_excluded_sharpe(
    results: list[BacktestResult],
    exclude: tuple[str, str] = ("2008-01-01", "2009-12-31"),
) -> pd.DataFrame:
    """Holdout Sharpe with and without the financial-crisis years.

    Trend strategies famously feasted on 2008; this shows how much of the
    holdout result depends on that single episode.
    """
    rows = []
    for result in results:
        returns = result.daily.loc[HOLDOUT_WINDOW[0] : HOLDOUT_WINDOW[1], "net_return"]
        outside = returns.loc[(returns.index < exclude[0]) | (returns.index > exclude[1])]
        rows.append(
            {
                "Strategy": result.name,
                "Sharpe 2005-2014": returns.mean() / returns.std() * math.sqrt(252),
                "Sharpe excl. 2008-09": outside.mean() / outside.std() * math.sqrt(252),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")


def run_enhanced_pipeline(
    base_config: BacktestConfig,
    enhanced_config: EnhancedConfig | None = None,
) -> EnhancedPipelineOutput:
    """Run the headline strategy, the training experiment, and all diagnostics."""
    config = enhanced_config or EnhancedConfig()
    config.validate()
    symbols = [s for members in GLOBAL_ASSET_CLASSES.values() for s in members]
    prices = load_prices(base_config.data_dir, symbols, ffill_limit=10)
    metadata = load_global_metadata(base_config.data_dir, symbols)
    tradeable = tradeable_mask(load_volumes(base_config.data_dir, symbols)).reindex(
        prices.index
    ).fillna(False)
    fx = load_fx_rates(base_config.data_dir, prices.index)
    point_values = usd_point_values(metadata, fx, prices.index)

    walk_forward = run_walk_forward(
        base_config, config, prices=prices, metadata=metadata, tradeable=tradeable,
        asset_classes=GLOBAL_ASSET_CLASSES, point_values=point_values,
    )
    # The headline is two sleeves at equal risk weight: the pre-specified
    # 12-month sign trend and basis momentum, the one candidate of eight that
    # passed the v2.2 adoption rule and improved both evaluation windows.
    trend_signal_frame = walk_forward.candidate_signals["Sign 12m"]
    unadjusted = load_unadjusted_prices(base_config.data_dir, symbols)
    basis = basis_momentum(
        prices, unadjusted, config.signal_vol_span, config.strength_cap
    )
    headline_signal = blend_sleeves(
        trend_signal_frame,
        {"trend": trend_signal_frame, "basis": basis},
        {"trend": 1 - config.basis_weight, "basis": config.basis_weight},
    )
    headline = run_signal_backtest(
        headline_signal, prices, metadata, base_config, InstitutionalConfig(),
        name="Global TSMOM + Basis Momentum (61 markets)",
        asset_classes=GLOBAL_ASSET_CLASSES, tradeable=tradeable,
        vol_decay=config.vol_decay, risk_managed_window=config.risk_managed_window,
        point_values=point_values,
    )
    # The 43-market book the global universe was gated against, same sleeves.
    incumbent_43 = run_signal_backtest(
        headline_signal, prices, metadata, base_config, InstitutionalConfig(),
        name="Two sleeves, 43 USD markets (v2.2)", tradeable=tradeable,
        vol_decay=config.vol_decay, point_values=point_values,
    )
    risk_managed = run_signal_backtest(
        headline_signal, prices, metadata, base_config, InstitutionalConfig(),
        name="+ Risk-managed sizing (tested, not adopted)",
        asset_classes=GLOBAL_ASSET_CLASSES, tradeable=tradeable,
        vol_decay=config.vol_decay, risk_managed_window=126,
        point_values=point_values,
    )
    trend_only = replace(
        walk_forward.candidate_results["Sign 12m"],
        name="Trend sleeve only (61 markets)",
    )
    # Carry remains reported but unadopted: its increment over trend is
    # indecisive in both windows, and it is built from the same roll-gap data
    # that basis momentum uses to better effect.
    carry = carry_strength(
        prices, unadjusted, config.signal_vol_span, config.strength_cap
    )
    carry_signal = blend_trend_and_carry(
        trend_signal_frame, carry, config.carry_weight
    )
    carry_variant = run_signal_backtest(
        carry_signal, prices, metadata, base_config, InstitutionalConfig(),
        name="+ Carry sleeve (tested, not adopted)",
        asset_classes=GLOBAL_ASSET_CLASSES, tradeable=tradeable,
        vol_decay=config.vol_decay, point_values=point_values,
    )
    # The v2.0 spec (63-day rolling volatility targeting) is re-run so the
    # original claim of record stays reproducible next to the current spec.
    v2_spec = run_signal_backtest(
        walk_forward.candidate_signals["Sign 12m"], prices, metadata, base_config,
        InstitutionalConfig(), name="Breadth TSMOM (v2.0 spec, 43 USD markets)",
        tradeable=tradeable, vol_decay=None, point_values=point_values,
    )

    baseline_adaptive = replace(
        run_institutional_backtest(base_config).backtest,
        name="Adaptive TSMOM (22 markets)",
    )
    baseline_tsmom = run_backtest(base_config)
    long_only_signal = pd.DataFrame(
        1.0, index=prices.index, columns=prices.columns
    ).where(prices.notna())
    long_only = run_signal_backtest(
        long_only_signal, prices, metadata, base_config, InstitutionalConfig(),
        name="Long-only risk-balanced (61 markets)",
        asset_classes=GLOBAL_ASSET_CLASSES, tradeable=tradeable,
        vol_decay=config.vol_decay, point_values=point_values,
    )

    lineup = [
        headline,
        incumbent_43,
        risk_managed,
        trend_only,
        carry_variant,
        walk_forward.backtest,
        walk_forward.ensemble,
        baseline_adaptive,
        baseline_tsmom,
        long_only,
    ]
    comparison = headline_comparison_table(
        lineup,
        {
            "1990-2004 (primary)": PRIMARY_WINDOW,
            "2005-2014 (second use)": HOLDOUT_WINDOW,
        },
        # per the pre-registered seeding rule, the walk-forward composite is
        # measured from 1992, after its first selection has taken effect.
        start_overrides={walk_forward.backtest.name: "1992-01-01"},
    )

    bootstrap_rows = {}
    for window_name, (start, end) in (
        ("1990-2004", PRIMARY_WINDOW),
        ("2005-2014", HOLDOUT_WINDOW),
    ):
        bootstrap_rows[window_name] = paired_sharpe_difference(
            headline.daily["net_return"],
            baseline_adaptive.daily["net_return"],
            start,
            end,
        )
    bootstrap = pd.DataFrame(bootstrap_rows).T

    psr = pd.Series(
        {
            # the benchmark Sharpe is computed from monthly returns, the same
            # estimator the PSR itself uses, so no daily/monthly units mix.
            window_name: probabilistic_sharpe_ratio(
                headline.daily["net_return"],
                start,
                end,
                benchmark_sharpe=_monthly_sharpe(
                    baseline_adaptive.daily["net_return"], start, end
                ),
            )
            for window_name, (start, end) in (
                ("1990-2004", PRIMARY_WINDOW),
                ("2005-2014", HOLDOUT_WINDOW),
            )
        },
        name="PSR vs adaptive baseline",
    )

    # Honest trial count for the deflated Sharpe: 5 candidates + ensemble +
    # walk-forward + 3 selection windows + 2 class schemes + 3 buffers +
    # 2 sizing vol spans + 3 cost multiples + thin-cost + no-judgment
    # + the v1 robustness grid of 12 (4 horizon sets x 3 cost multiples)
    # + the v2.1 blueprint experiments (2 vol estimators x carry weights
    # {0, 25, 50, 75, 100} minus overlaps, plus the absorption overlay) = 42
    # + the v2.2 alpha search: 8 candidate sleeves, 2 blend weights, the
    # 3-sleeve variant, 5 sizing/class schemes = 58
    # + the v2.3 search: 10 further candidate sleeves, and the v2.4
    # risk-managed sizing sweep of 6 configurations = 74.
    n_trials = 74
    dsr = pd.Series(
        {
            window_name: deflated_sharpe_ratio(
                headline.daily["net_return"], start, end, n_trials=n_trials
            )
            for window_name, (start, end) in BOTH_WINDOWS
        },
        name="Deflated Sharpe ratio",
    )
    pbo = cscv_pbo(
        {n: r.daily["net_return"] for n, r in walk_forward.candidate_results.items()},
        PRIMARY_WINDOW[0],
        HOLDOUT_WINDOW[1],
    )

    yearly = yearly_comparison(
        headline.daily["net_return"],
        baseline_adaptive.daily["net_return"],
        PRIMARY_WINDOW[0],
        HOLDOUT_WINDOW[1],
        (headline.name, baseline_adaptive.name),
    )
    # The claim rule is evaluated for the v2.0 spec of record and re-evaluated
    # for the current spec, side by side; neither replaces the other.
    claim_columns = {}
    for label, result in (
        ("v2.0 spec (claim of record)", v2_spec),
        ("v2.1 spec (current)", headline),
    ):
        spec_bootstrap = pd.DataFrame(
            {
                window_name: paired_sharpe_difference(
                    result.daily["net_return"],
                    baseline_adaptive.daily["net_return"],
                    start,
                    end,
                )
                for window_name, (start, end) in BOTH_WINDOWS
            }
        ).T
        spec_yearly = yearly_comparison(
            result.daily["net_return"],
            baseline_adaptive.daily["net_return"],
            PRIMARY_WINDOW[0],
            HOLDOUT_WINDOW[1],
            (result.name, baseline_adaptive.name),
        )
        claim_columns[label] = claim_rule_evaluation(
            spec_bootstrap, spec_yearly, result.name
        )
    claim = pd.DataFrame(claim_columns)

    # Both candidate second sleeves are measured the same way: the paired
    # Sharpe increment of the blend over the trend sleeve alone. Basis
    # momentum is positive in both windows and was adopted; carry is
    # indecisive in both and was not.
    sleeve_increment = pd.concat(
        {
            sleeve.name: pd.DataFrame(
                {
                    window_name: paired_sharpe_difference(
                        sleeve.daily["net_return"],
                        trend_only.daily["net_return"],
                        start,
                        end,
                    )
                    for window_name, (start, end) in BOTH_WINDOWS
                }
            ).T
            for sleeve in (headline, carry_variant)
        }
    )

    three_sleeves = blend_sleeves(
        trend_signal_frame,
        {"trend": trend_signal_frame, "carry": carry, "basis": basis},
        {"trend": 1.0, "carry": 1.0, "basis": 1.0},
    )
    signal_variants = {
        "Trend sleeve alone (no basis)": trend_signal_frame,
        "Basis momentum sleeve alone": basis,
        "Trend + basis 67/33": blend_sleeves(
            trend_signal_frame, {"trend": trend_signal_frame, "basis": basis},
            {"trend": 2.0, "basis": 1.0},
        ),
        "Trend + carry + basis (equal)": three_sleeves,
        "Trend + carry 50/50 (tested)": carry_signal,
        "Absorption-ratio overlay (rejected)": absorption_overlay(
            trend_signal_frame, absorption_ratio(prices, config.signal_vol_span)
        ),
    }
    stress = enhanced_stress_tests(
        prices, metadata, base_config, headline_signal, tradeable,
        vol_decay=config.vol_decay, signal_variants=signal_variants,
        class_variants={"Flat risk budget (no asset classes)":
                        {"All markets": tuple(prices.columns)}},
        asset_classes=GLOBAL_ASSET_CLASSES, point_values=point_values,
    )
    stress = pd.concat([stress, no_judgment_universe_row(base_config, config)])
    selection_sensitivity = selection_window_sensitivity(
        base_config, prices, metadata, tradeable,
        asset_classes=GLOBAL_ASSET_CLASSES, point_values=point_values,
    )
    class_contributions = contribution_by_class(
        headline, base_config, GLOBAL_ASSET_CLASSES, point_values=point_values
    )
    crisis_robustness = crisis_excluded_sharpe([headline, baseline_adaptive])
    leave_one_out = leave_one_class_out(
        prices, metadata, base_config, trend_signal_frame, tradeable,
        vol_decay=config.vol_decay,
        asset_classes=GLOBAL_ASSET_CLASSES, point_values=point_values,
    )
    breadth = pd.DataFrame(
        {
            headline.name: effective_breadth(headline, *HOLDOUT_WINDOW),
            baseline_adaptive.name: effective_breadth(
                baseline_adaptive, *HOLDOUT_WINDOW
            ),
        }
    ).T
    factorial = factorial_attribution(base_config, config, prices, metadata, tradeable)

    return EnhancedPipelineOutput(
        headline=headline,
        walk_forward=walk_forward,
        baseline_adaptive=baseline_adaptive,
        baseline_tsmom=baseline_tsmom,
        long_only=long_only,
        comparison=comparison,
        bootstrap=bootstrap,
        psr=psr,
        dsr=dsr,
        pbo=pbo,
        yearly=yearly,
        stress=stress,
        selection_sensitivity=selection_sensitivity,
        class_contributions=class_contributions,
        crisis_robustness=crisis_robustness,
        leave_one_out=leave_one_out,
        breadth=breadth,
        factorial=factorial,
        claim=claim,
        carry_variant=carry_variant,
        trend_only=trend_only,
        sleeve_increment=sleeve_increment,
    )


def enhanced_invariants(
    output: EnhancedPipelineOutput,
    base_config: BacktestConfig,
) -> pd.Series:
    """Production-style checks for the headline and walk-forward strategies."""
    checks = {}
    for label, result in (
        ("headline", output.headline),
        ("walk_forward", output.walk_forward.backtest),
    ):
        daily = result.daily
        active = result.positions.diff().abs().sum(axis=1).gt(1e-15)
        month = pd.Series(daily.index.to_period("M"), index=daily.index)
        transition = month.ne(month.shift(1))
        checks[f"{label}_finite_net_returns"] = bool(np.isfinite(daily["net_return"]).all())
        checks[f"{label}_non_negative_costs"] = bool((daily["cost"] >= -1e-15).all())
        checks[f"{label}_forecast_bounded"] = (
            float(result.signals.abs().max().max()) <= 1 + 1e-12
        )
        checks[f"{label}_positions_change_only_at_month_boundary"] = bool(
            (~active.iloc[1:] | transition[1:]).all()
        )
    holdout_days = output.headline.daily.loc[
        HOLDOUT_WINDOW[0] : HOLDOUT_WINDOW[1]
    ].shape[0]
    checks["holdout_exceeds_five_years"] = holdout_days / base_config.annualization >= 5
    # every application year Y must trace back to a score row computed at the
    # end of year Y-1, i.e. selections never use same-year information.
    checks["selection_uses_only_prior_years"] = set(
        output.walk_forward.selections.index - 1
    ).issubset(set(output.walk_forward.selection_scores.index))
    return pd.Series(checks, name="Pass")


def save_enhanced_outputs(
    output: EnhancedPipelineOutput,
    invariants: pd.Series,
    base_config: BacktestConfig,
) -> None:
    directory = base_config.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    output.comparison.to_csv(directory / "enhanced_metrics.csv")
    output.bootstrap.to_csv(directory / "enhanced_bootstrap.csv")
    statistics = pd.concat(
        {
            "PSR vs adaptive baseline": output.psr,
            "Deflated Sharpe ratio": output.dsr,
        },
        axis=1,
    )
    statistics["CSCV PBO (candidate set)"] = output.pbo
    statistics.to_csv(directory / "enhanced_statistics.csv")
    output.yearly.to_csv(directory / "enhanced_yearly.csv", index_label="Year")
    output.stress.to_csv(directory / "enhanced_stress_tests.csv")
    output.selection_sensitivity.to_csv(directory / "enhanced_selection_sensitivity.csv")
    output.class_contributions.to_csv(
        directory / "enhanced_class_contributions.csv", index_label="Asset class"
    )
    output.crisis_robustness.to_csv(directory / "enhanced_crisis_robustness.csv")
    output.claim.to_csv(directory / "enhanced_claim_rule.csv")
    output.sleeve_increment.to_csv(directory / "enhanced_sleeve_increment.csv")
    output.leave_one_out.to_csv(directory / "enhanced_leave_one_class_out.csv")
    output.breadth.to_csv(directory / "enhanced_effective_breadth.csv")
    output.factorial.to_csv(directory / "enhanced_factorial.csv")
    output.walk_forward.selections.to_csv(
        directory / "enhanced_selections.csv", index_label="Year", header=True
    )
    output.walk_forward.selection_scores.to_csv(
        directory / "enhanced_selection_scores.csv"
    )
    invariants.to_csv(directory / "enhanced_invariants.csv", header=True)
    _save_enhanced_plot(output, base_config)


def _save_enhanced_plot(
    output: EnhancedPipelineOutput,
    base_config: BacktestConfig,
) -> None:
    # Draw on an explicit Figure so no global backend is touched; the pipeline
    # can run headless and inside a notebook without breaking inline plots.
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig = Figure(figsize=(12, 11))
        axes = fig.subplots(3, 1, sharex=True, height_ratios=[2.2, 1, 1])
    lineup = [
        (output.headline, "#005F73"),
        (output.walk_forward.backtest, "#9B2226"),
        (output.baseline_adaptive, "#6C757D"),
    ]
    start = "1992-01-01"
    for result, color in lineup:
        returns = result.daily.loc[start:, "net_return"]
        equity = (1 + returns).cumprod()
        axes[0].plot(equity.index, equity, label=result.name, color=color, linewidth=1.6)
        drawdown = equity / equity.cummax() - 1
        axes[1].plot(drawdown.index, drawdown, color=color, linewidth=1)
        rolling = (
            returns.rolling(756).mean() / returns.rolling(756).std() * math.sqrt(252)
        )
        axes[2].plot(rolling.index, rolling, color=color, linewidth=1.2)
    holdout_start = pd.Timestamp(HOLDOUT_WINDOW[0])
    for axis in axes:
        axis.axvline(holdout_start, color="#333333", linewidth=1, linestyle="--")
    axes[0].annotate(
        "2005-2014 evaluation (second use)", xy=(holdout_start, axes[0].get_ylim()[1]),
        xytext=(8, -12), textcoords="offset points", fontsize=9, color="#333333",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1, net of costs (log)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title(
        "Breadth TSMOM vs walk-forward training and the 22-market baseline"
    )
    axes[1].set_ylabel("Drawdown")
    axes[2].set_ylabel("Rolling 3y Sharpe")
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(
        base_config.output_dir / "enhanced_performance.png", dpi=180, bbox_inches="tight"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = BacktestConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    output = run_enhanced_pipeline(base_config)
    invariants = enhanced_invariants(output, base_config)
    save_enhanced_outputs(output, invariants, base_config)
    print(output.comparison.to_string())
    print()
    print(output.bootstrap.to_string())
    if not invariants.all():
        raise SystemExit(f"Invariant failures:\n{invariants[~invariants]}")


if __name__ == "__main__":
    main()
