"""Delta1 case answer: the whole strategy in one readable file.

Run it and it prints the headline table.  Read it top to bottom and you have
the complete specification -- there is no hidden state, no configuration
inheritance, and no import from the hardened package.

    python reference/delta1_reference.py --data-dir "Round1AllData/Quant Researcher/Delta1"

The strategy is a diversified time-series momentum book on 59 global futures,
blended with a basis-momentum sleeve, sized to a 7% annualized volatility
target, rebalanced monthly, and charged spread, slippage, commission,
exchange fees, square-root market impact, and continuous-contract roll costs.

Six steps, in the order the code runs them:

    1.  Signals      12-month sign trend, plus year-on-year change in trailing
                     roll yield (basis momentum).  Equal risk weight.
    2.  Conditioning A 20d/120d volatility-shock taper cuts the forecast when
                     short-horizon risk spikes; a 63-day risk-managed scaler
                     divides each market's forecast by the realized volatility
                     of its own forecast-weighted P&L.
    3.  Sizing       Each market gets an equal share of the risk budget,
                     converted to contracts through its own annualized dollar
                     volatility.  A portfolio EWMA scaler steers realized
                     volatility to the 7% target.
    4.  Trading      Monthly decisions, a 25% no-trade band, integer contracts,
                     and a 2% participation cap on both order size and fill.
    5.  Costs        Fixed per-contract costs plus impact proportional to
                     sqrt(participation), charged on rebalance and roll
                     turnover alike.
    6.  Ledger       Self-financing USD NAV, marked daily.

Everything is causal.  Every quantity used in a decision on date t is computed
from data observed strictly before the fill.  `tests/test_reference.py` proves
this file reproduces the hardened engine in `src/delta1_strategy/` to within
floating-point noise, which is what makes a short file safe to read instead of
a long one.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Specification.  Every number the strategy uses appears once, here.
# --------------------------------------------------------------------------

UNIVERSE: dict[str, tuple[str, ...]] = {
    "Equity indices": ("EMD", "ES", "FCE", "FDAX", "FESX", "FSMI", "HTW", "LFT",
                       "NKD", "NQ", "RTY", "SNK", "SXF", "YAP", "YM"),
    "Government bonds": ("CGB", "FGBL", "FGBM", "FGBS", "FGBX", "LLG", "SJB",
                         "ZT", "ZF", "ZN", "ZB"),
    "FX": ("6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S"),
    "Energy": ("BRN", "CL", "GAS", "HO", "NG", "RB"),
    "Metals": ("GC", "HG", "PA", "PL", "SI"),
    "Agriculture & livestock": ("CC", "CT", "GF", "HE", "KC", "KE", "LE", "RS",
                                "SB", "ZC", "ZL", "ZM", "ZS", "ZW"),
}
# YXT and YYT are excluded: they are yield-quoted, so `price x multiplier`
# cannot value them.  A correctness exclusion, not a return-based choice.

FX_SOURCE = {"EUR": ("6E", 1.0), "GBP": ("6B", 1.0), "JPY": ("6J", 0.01),
             "CHF": ("6S", 1.0), "CAD": ("6C", 1.0), "AUD": ("6A", 1.0)}

P = {
    # Signals
    "trend_lookback": 252,          # 12-month sign trend
    "basis_roll_window": 252,       # trailing roll-yield accumulation
    "basis_lookback": 252,          # year-on-year change in that roll yield
    "signal_norm_window": 252,      # rolling std used to standardise basis
    "signal_cap": 2.0,              # basis clip, in normalised units
    "basis_weight": 0.50,           # equal risk weight with trend
    # Volatility-shock taper
    "fast_vol_span": 20, "slow_vol_span": 120,
    "shock_start": 1.35, "shock_full": 2.00, "shock_floor": 0.75,
    # Risk sizing
    "vol_span": 60,                 # per-market EWMA price volatility
    "risk_managed_window": 63, "risk_managed_cap": 2.00,
    "target_vol": 0.07,             # portfolio annualized volatility budget
    "vol_decay": 0.94,              # RiskMetrics lambda for the portfolio scaler
    "vol_min_periods": 20, "portfolio_vol_window": 63,
    "min_risk_scalar": 0.25, "max_risk_scalar": 2.00,
    "max_gross_notional_multiple": 5.0,
    # Trading
    "no_trade_buffer": 0.25,        # skip adjustments smaller than 25%
    "volume_gate_window": 60, "min_median_contracts": 1000.0,
    "max_participation": 0.02,      # of median (sizing) and realized (fill) volume
    "max_roll_backlog_sessions": 21,
    # Costs, per contract per side
    "half_spread_ticks": 0.50, "slippage_ticks": 0.25,
    "commission": 2.50, "fees": 1.50,
    "impact_bps_at_full_participation": 10.0,
    # Ledger
    "initial_capital": 1_000_000.0, "launch": "1990-01-01",
    "price_ffill_limit": 10, "annualization": 252,
}

SYMBOLS = [s for members in UNIVERSE.values() for s in members]


# --------------------------------------------------------------------------
# 1.  Data
# --------------------------------------------------------------------------

def _column(path: Path, column: str, name: str) -> pd.Series:
    """One vendor column, date-indexed, with malformed rows rejected loudly."""
    frame = pd.read_csv(path, usecols=["Date", column])
    dates = pd.to_datetime(frame["Date"], errors="raise")
    values = pd.to_numeric(frame[column], errors="raise")
    if dates.duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    observed = values.notna()
    if not np.isfinite(values.to_numpy(dtype=float)[observed]).all():
        raise ValueError(f"Non-finite {column} values in {path}")
    if column == "Volume" and values[observed].lt(0).any():
        raise ValueError(f"Negative Volume values in {path}")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=name).sort_index()


def _panel(data_dir: Path, filename: str, column: str, ffill: int | None) -> pd.DataFrame:
    """Stack one column across every market onto a common business-day calendar."""
    frame = pd.concat(
        [_column(data_dir / "Futures Data" / filename.format(s=s), column, s) for s in SYMBOLS],
        axis=1,
        sort=False,
    ).sort_index()
    frame = frame.reindex(pd.bdate_range(frame.index.min(), frame.index.max()))
    return frame.ffill(limit=ffill) if ffill else frame


def load_market_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Back-adjusted closes drive P&L; unadjusted closes drive roll yield and notional."""
    data_dir = Path(data_dir)
    limit = P["price_ffill_limit"]
    prices = _panel(data_dir, "&{s}_CCB.csv", "Close", limit)
    calendar = prices.index

    catalogue = pd.read_csv(data_dir / "CATALOGUE_Delta1_Futures.csv")
    catalogue["sym"] = catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    terms = (
        catalogue[catalogue["symbol"].str.endswith("_CCB") & catalogue["sym"].isin(SYMBOLS)]
        .set_index("sym")
        .reindex(SYMBOLS)
    )

    # Point-in-time USD conversion, taken from the currency futures themselves
    # so no external FX series enters the backtest.
    fx = pd.DataFrame({"USD": 1.0}, index=calendar)
    for currency, (symbol, scale) in FX_SOURCE.items():
        rate = _column(data_dir / "Futures Data" / f"&{symbol}.csv", "Close", currency) * scale
        fx[currency] = rate.reindex(calendar).ffill(limit=limit)

    def _usd(field: str) -> pd.DataFrame:
        return pd.DataFrame(
            {s: fx[terms.at[s, "currency"]] * float(terms.at[s, field]) for s in SYMBOLS},
            index=calendar,
        )

    return {
        "prices": prices,
        "unadjusted": _panel(data_dir, "&{s}.csv", "Close", limit),
        "closes": _panel(data_dir, "&{s}_CCB.csv", "Close", None),   # unfilled: was it a real session?
        "volumes": _panel(data_dir, "&{s}_CCB.csv", "Volume", None),
        "delivery": _panel(data_dir, "&{s}.csv", "Delivery Month", None),
        "point_values": _usd("point_value"),
        "margins": _usd("margin"),
        "tick_size": terms["tick_size"].astype(float),
    }


# --------------------------------------------------------------------------
# 2.  Signals
# --------------------------------------------------------------------------

def trend_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Sign of the 12-month price change (Moskowitz-Ooi-Pedersen 2012).

    A sign, not a t-statistic: the level of a back-adjusted series is not a
    meaningful denominator, so nothing here divides by price.
    """
    change = prices - prices.shift(P["trend_lookback"])
    return np.sign(change).where(change.notna())


def basis_momentum(prices: pd.DataFrame, unadjusted: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year change in trailing roll yield -- momentum in carry.

    The gap between the unadjusted and back-adjusted daily change *is* the roll
    return; accumulating it over a year gives the carry the term structure paid,
    and its year-on-year change is the momentum in that carry.

    Motivated by Boons & Porras Prado, "Basis-Momentum", Journal of Finance
    74(1), 2019.  Their signal is a spread between first- and second-nearby
    contract returns; the supplied panel carries only a front-month series, so
    this is the closest available proxy rather than a replication of that
    construction, and it is named accordingly.
    """
    roll_gap = unadjusted.reindex_like(prices).diff() - prices.diff()
    roll_yield = -roll_gap.rolling(P["basis_roll_window"], min_periods=P["basis_roll_window"]).sum()
    daily_vol = prices.diff().ewm(span=P["vol_span"], min_periods=P["vol_span"], adjust=False).std()

    raw = (roll_yield - roll_yield.shift(P["basis_lookback"])) / (
        daily_vol.replace(0, np.nan) * math.sqrt(P["annualization"])
    )
    scale = raw.rolling(P["signal_norm_window"], min_periods=P["signal_norm_window"]).std()
    return (raw / scale.replace(0, np.nan)).clip(-P["signal_cap"], P["signal_cap"]) / P["signal_cap"]


def shock_multiplier(prices: pd.DataFrame) -> pd.DataFrame:
    """Taper exposure when 20-day risk runs hot against the 120-day baseline.

    Linear from 1.0 at a ratio of 1.35 down to the 0.75 floor at 2.00.  It is a
    risk control, so it can only ever cut.
    """
    change = prices.diff()
    fast = change.ewm(span=P["fast_vol_span"], min_periods=P["fast_vol_span"], adjust=False).std()
    slow = change.ewm(span=P["slow_vol_span"], min_periods=P["slow_vol_span"], adjust=False).std()
    ratio = fast / slow.replace(0, np.nan)
    progress = ((ratio - P["shock_start"]) / (P["shock_full"] - P["shock_start"])).clip(0, 1)
    return (1 - progress * (1 - P["shock_floor"])).where(ratio.notna())


def risk_managed_forecast(forecast: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Divide each forecast by the realized volatility of its own strategy P&L.

    A forecast decided after close t can only earn the move from t+1 onward,
    hence `.shift(2)` against a differenced price: shift(1) aligns the decision
    to the fill, and the second shift is the difference itself.  That two-bar
    delay is the conservative choice and it is what the causality test pins.
    """
    daily_vol = prices.diff().ewm(span=P["vol_span"], min_periods=P["vol_span"], adjust=False).std()
    strategy_return = forecast.shift(2) * prices.diff() / daily_vol.shift(2).replace(0, np.nan)
    realized = (
        strategy_return.pow(2)
        .rolling(P["risk_managed_window"], min_periods=P["risk_managed_window"])
        .mean()
        .pow(0.5)
    )
    weight = (1.0 / realized.replace(0, np.nan)).clip(upper=P["risk_managed_cap"])
    return (forecast * weight).clip(-1, 1).where(forecast.notna())


def build_forecast(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Signals -> blend -> shock taper -> risk management -> liquidity gate."""
    prices = data["prices"]
    trend = trend_signal(prices)
    basis = basis_momentum(prices, data["unadjusted"])

    # Basis falls back to trend before it is estimable, so the book is never
    # under-invested purely because one sleeve needs a longer warm-up.
    blended = (trend * (1 - P["basis_weight"]) + basis.fillna(trend) * P["basis_weight"])
    blended = blended.clip(-1, 1).where(trend.notna())

    forecast = risk_managed_forecast((blended * shock_multiplier(prices)).clip(-1, 1), prices)

    tradeable = (
        data["volumes"].fillna(0.0)
        .rolling(P["volume_gate_window"], min_periods=P["volume_gate_window"])
        .median()
        .gt(P["min_median_contracts"])
        .reindex(prices.index)
        .fillna(False)
    )
    return forecast.where(tradeable, np.nan)


# --------------------------------------------------------------------------
# 3.  Sizing
# --------------------------------------------------------------------------

def target_positions_per_dollar(
    prices: pd.DataFrame, forecast: pd.DataFrame, point_values: pd.DataFrame
) -> pd.DataFrame:
    """Contracts per dollar of NAV, before the portfolio scaler.

    Every available market gets an equal *pre-forecast* risk budget of
    target_vol / sqrt(N).  Dividing by the market's own annualized dollar
    volatility per contract turns a risk budget into a contract count, which is
    what makes a bond future and a crude future comparable.
    """
    annual_dollar_vol = (
        prices.diff().ewm(span=P["vol_span"], min_periods=P["vol_span"], adjust=False).std()
        * point_values
        * math.sqrt(P["annualization"])
    )
    available = forecast.notna() & annual_dollar_vol.gt(0)
    count = available.sum(axis=1).replace(0, np.nan)
    budget = P["target_vol"] / np.sqrt(count)
    positions = forecast.mul(budget, axis=0).div(annual_dollar_vol)
    return positions.replace([np.inf, -np.inf], np.nan)


def month_end_rows(frame: pd.DataFrame | pd.Series):
    """Last observation of each completed month.

    A later month proves the previous one closed.  The final month survives only
    if the data end on a business month end, so a truncated file never produces
    a decision from a partial month.
    """
    rows = frame.groupby(frame.index.to_period("M")).tail(1)
    if not pd.offsets.BMonthEnd().is_on_offset(frame.index[-1].normalize()):
        rows = rows.iloc[:-1]
    return rows


def portfolio_risk_scalar(gross_returns: pd.Series, positions: pd.DataFrame) -> pd.Series:
    """EWMA multiplier steering realized volatility to the target.

    The decay is the RiskMetrics lambda = 0.94, but under pandas' default
    adjusted weighting rather than the classic adjust=False recursion the
    per-market estimators use -- a deliberate difference the hardened engine
    mirrors exactly.  Held to 1.0 until the book has been live for a full
    63-session window, so the scaler never extrapolates from a nearly empty
    portfolio.
    """
    realized = gross_returns.ewm(
        alpha=1 - P["vol_decay"], min_periods=P["vol_min_periods"]
    ).std() * math.sqrt(P["annualization"])
    scalar = (P["target_vol"] / realized.replace(0, np.nan)).clip(
        P["min_risk_scalar"], P["max_risk_scalar"]
    ).fillna(1.0)

    live = positions.abs().sum(axis=1).gt(0).cummax()
    warm = live & live.shift(P["portfolio_vol_window"]).fillna(False)
    return scalar.where(warm, 1.0)


def apply_no_trade_buffer(desired: pd.DataFrame, width: float) -> pd.DataFrame:
    """Hold the previous target unless the adjustment exceeds `width` of it.

    The single most valuable cost control in the book: most month-to-month
    forecast changes are noise, and this suppresses the turnover they would
    otherwise generate without changing the signal.
    """
    executable = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
    previous = pd.Series(0.0, index=desired.columns)
    for date, row in desired.iterrows():
        target = row.fillna(0.0)
        reference = pd.concat([target.abs(), previous.abs()], axis=1).max(axis=1)
        trade = (target - previous).abs() > width * reference
        previous = previous.where(~trade, target)
        executable.loc[date] = previous
    return executable


# --------------------------------------------------------------------------
# 4.  Execution and the NAV ledger
# --------------------------------------------------------------------------

def roll_events(delivery: pd.DataFrame) -> pd.DataFrame:
    """True on every vendor delivery-label change.

    The continuous price path follows those labels, so each change is a real
    transfer the book has to pay for -- including a reversal.
    """
    rolls = pd.DataFrame(False, index=delivery.index, columns=delivery.columns)
    for symbol in delivery:
        observed = delivery[symbol].dropna().astype(int)
        if observed.empty:
            continue
        changed = observed.ne(observed.shift())
        changed.iloc[0] = False
        rolls.loc[observed.index, symbol] = changed.to_numpy()
    return rolls


def _limit_gross(desired: np.ndarray, nav: float, price: np.ndarray,
                 pv: np.ndarray, margin: np.ndarray, integer: bool) -> np.ndarray:
    """Scale a target book down to the 5x gross-notional ceiling.

    Fails closed when a nonzero target cannot be valued: a NaN price, point
    value, or margin on the decision date must abort the run, not silently
    drop that market out of the cap arithmetic.
    """
    nonzero = desired != 0
    if nonzero.any() and (
        not np.isfinite(price[nonzero]).all()
        or not np.isfinite(pv[nonzero]).all()
        or not np.isfinite(margin[nonzero]).all()
    ):
        raise ValueError("Missing decision-date valuation input for a nonzero target")
    gross = float(np.nansum(np.abs(desired * price * pv)))
    cap = P["max_gross_notional_multiple"] * nav
    if gross > cap > 0:
        desired = desired * (cap / gross)
    if not integer:
        return desired
    desired = np.rint(desired)
    gross = float(np.nansum(np.abs(desired * price * pv)))
    if gross > cap > 0:                     # rounding can push back over the cap
        desired = np.trunc(desired * (cap / gross))
    return desired


def simulate(
    monthly_targets: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    one_way_cost: pd.DataFrame,
    *,
    capital: float,
    integer: bool,
    charge_costs: bool,
    launch: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill monthly targets session by session and mark a self-financing ledger.

    Two participation bounds, deliberately on different information sets:

      * order **size** uses lagged median volume, because sizing from the
        volume a session turned out to trade would be a look-ahead that trades
        bigger exactly when liquidity was good;
      * the **fill** is capped at 2% of what the session actually traded, with
        the residual carried to the next session.  Capping a fill can only ever
        truncate it, so it is safe to condition on.

    Roll turnover obeys the same budget.  A contract held through a delivery
    change owes two contracts of transfer turnover; that obligation is tracked
    as a quantity and worked off in slices priced against the parent roll's
    participation, so splitting an order earns no square-root discount.
    """
    prices, columns = data["prices"], data["prices"].columns
    index = prices.index
    n, m = len(index), len(columns)

    close = prices.to_numpy(float)
    unadj = data["unadjusted"].reindex(index=index, columns=columns).to_numpy(float)
    raw_close = data["closes"].reindex(index=index, columns=columns).to_numpy(float)
    volume = data["volumes"].reindex(index=index, columns=columns).to_numpy(float)
    pv = data["point_values"].reindex(index=index, columns=columns).to_numpy(float)
    margin = data["margins"].reindex(index=index, columns=columns).to_numpy(float)
    cost_per_contract = one_way_cost.reindex(index=index, columns=columns).to_numpy(float)
    rolls = roll_events(data["delivery"]).reindex(index=index, columns=columns).fillna(False).to_numpy(bool)

    # Sizing capacity from lagged median volume; fill capacity from this session.
    lagged_volume = (
        data["volumes"].reindex(index=index, columns=columns).fillna(0.0)
        .rolling(P["volume_gate_window"], min_periods=P["volume_gate_window"])
        .median().shift(1).to_numpy(float)
    )
    size_cap = lagged_volume * P["max_participation"]
    fill_cap = np.where(np.isfinite(volume), volume * P["max_participation"], 0.0)
    if integer:
        size_cap, fill_cap = np.floor(size_cap), np.floor(fill_cap)

    targets = {d: r.fillna(0.0).to_numpy(float) for d, r in monthly_targets.iterrows()}
    launch_ts = pd.Timestamp(launch) if launch else None

    quantity = np.zeros(m)
    pending = np.full(m, np.nan)
    pending_roll = np.zeros(m, bool)
    backlog = np.zeros(m)
    backlog_age = np.zeros(m, int)
    parent_participation = np.zeros(m)
    nav = float(capital)

    rows = np.zeros((n, 6))          # nav, gross pnl, cost, gross notional, margin, turnover
    positions = np.zeros((n, m))

    for i, date in enumerate(index):
        if launch_ts is not None and date < launch_ts:
            rows[i] = (nav, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        first_live = launch_ts is not None and (i == 0 or index[i - 1] < launch_ts)

        # A decision made at the previous close becomes today's pending order.
        decision = index[i - 1] if (i > 0 and index[i - 1] in targets) else None
        if decision is None and first_live:
            earlier = [d for d in targets if d < date]
            decision = max(earlier) if earlier else None
        if decision is not None:
            loc = index.get_loc(decision)
            pending = _limit_gross(
                targets[decision] * nav, nav, unadj[loc], pv[loc], margin[loc], integer
            )
            pending[np.isclose(pending, quantity, atol=1e-12, rtol=0.0)] = np.nan
        pending_roll |= rolls[i]

        start_nav, old = nav, quantity.copy()
        tradeable_now = np.isfinite(raw_close[i]) & (volume[i] > 0)

        change = close[i] - close[i - 1] if i > 0 else np.zeros(m)
        held = old != 0
        if held.any() and (
            not np.isfinite(change[held]).all() or not np.isfinite(pv[i, held]).all()
        ):
            raise ValueError(f"Missing held settlement or FX value on {date.date()}")
        gross_pnl = float(np.sum(np.where(held, old * change * pv[i], 0.0)))

        # Fill, bounded by both caps; whatever is refused stays pending.
        fill = tradeable_now & np.isfinite(pending)
        step = np.clip(pending - old, -np.minimum(size_cap[i], fill_cap[i]),
                       np.minimum(size_cap[i], fill_cap[i]))
        step = np.where(np.isfinite(step), step, 0.0)
        quantity[fill] = old[fill] + step[fill]
        pending[fill & np.isclose(quantity, pending, atol=1e-12, rtol=0.0)] = np.nan
        turnover = np.abs(quantity - old)

        # Delivery transfers.  A reducing trade comes out of the expiring leg
        # first and so costs nothing extra; the rest must be transferred.
        processed = pending_roll & tradeable_now
        pending_roll[processed] = False
        reduction = np.where(
            np.sign(old) * np.sign(quantity) < 0,
            np.abs(old),
            np.maximum(np.abs(old) - np.abs(quantity), 0.0),
        )
        stale = np.where(processed, np.abs(old), backlog)
        backlog = np.minimum(np.maximum(stale - reduction, 0.0), np.abs(quantity))

        parent_participation = np.where(
            processed,
            np.divide(turnover + 2.0 * backlog, volume[i],
                      out=np.zeros(m), where=tradeable_now & (volume[i] > 0)),
            parent_participation,
        )
        headroom = np.maximum(fill_cap[i] - turnover, 0.0) / 2.0
        slice_ = np.where(headroom >= backlog, backlog, np.floor(headroom)) if integer \
            else np.minimum(backlog, headroom)
        slice_ = np.where(tradeable_now, slice_, 0.0)
        backlog -= slice_
        cleared = backlog <= 1e-12

        backlog_age = np.where(cleared, 0, backlog_age + 1)
        if (backlog_age > P["max_roll_backlog_sessions"]).any():
            stuck = ", ".join(str(columns[k]) for k in np.flatnonzero(backlog_age > P["max_roll_backlog_sessions"]))
            raise ValueError(f"Roll cannot complete within one cycle on {date.date()}: {stuck}")

        charged = turnover + 2.0 * slice_
        traded = charged > 0
        participation = np.divide(charged, volume[i], out=np.zeros(m),
                                  where=traded & (volume[i] > 0))
        # Price a slice against the unsliced roll's participation, never less.
        impact_reference = np.where((slice_ > 0) | ~cleared, parent_participation, 0.0)
        parent_participation = np.where(cleared, 0.0, parent_participation)

        if charge_costs:
            fixed = np.where(traded, charged * cost_per_contract[i], 0.0)
            impact = np.where(
                traded,
                charged * np.abs(unadj[i]) * pv[i]
                * (P["impact_bps_at_full_participation"] / 10_000)
                * np.sqrt(np.maximum(participation, impact_reference)),
                0.0,
            )
            day_cost = float(np.sum(fixed + impact))
        else:
            day_cost = 0.0

        nav = start_nav + gross_pnl - day_cost
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"Portfolio NAV is non-positive or non-finite on {date.date()}")

        held = quantity != 0
        rows[i] = (
            nav, gross_pnl, day_cost,
            float(np.sum(np.abs(quantity[held] * unadj[i, held] * pv[i, held]))),
            float(np.sum(np.abs(quantity[held]) * margin[i, held])),
            float(np.sum(charged)),
        )
        positions[i] = quantity

    prior_nav = np.r_[capital, rows[:-1, 0]]
    return pd.DataFrame(
        {
            "nav": rows[:, 0],
            "prior_nav_usd": prior_nav,
            "gross_pnl_usd": rows[:, 1],
            "transaction_cost_usd": rows[:, 2],
            "gross_return": rows[:, 1] / prior_nav,
            "cost": rows[:, 2] / prior_nav,
            "net_return": (rows[:, 1] - rows[:, 2]) / prior_nav,
            "gross_notional_multiple": rows[:, 3] / rows[:, 0],
            "static_margin_fraction": rows[:, 4] / rows[:, 0],
            "total_contract_turnover": rows[:, 5],
            "active_markets": (positions != 0).sum(axis=1),
        },
        index=index,
    ), pd.DataFrame(positions, index=index, columns=columns)


# --------------------------------------------------------------------------
# 5.  The strategy, end to end
# --------------------------------------------------------------------------

def run(data_dir: str | Path | None = None, *, data: dict | None = None) -> pd.DataFrame:
    """Load, forecast, size, trade, and return the daily ledger.

    Pass `data` to run against an already-loaded panel -- the same dict
    `load_market_data` returns.  That is what lets the submission bundle
    reproduce these results offline from the cleaned parquet panel, with no
    access to the licensed vendor CSVs.
    """
    if (data is None) == (data_dir is None):
        raise ValueError("pass exactly one of data_dir or data")
    if data is None:
        data = load_market_data(Path(data_dir))
    prices, pv = data["prices"], data["point_values"]

    one_way_cost = (
        (P["half_spread_ticks"] + P["slippage_ticks"]) * pv.mul(data["tick_size"], axis=1)
        + P["commission"] + P["fees"]
    )

    forecast = build_forecast(data)
    base = month_end_rows(target_positions_per_dollar(prices, forecast, pv))

    # Pass one: an unlevered, costless, fractional replay whose realized
    # volatility is the only thing the portfolio scaler is allowed to see.
    base_daily, base_positions = simulate(
        base, data, one_way_cost, capital=1.0, integer=False, charge_costs=False, launch=None
    )
    scalar = month_end_rows(portfolio_risk_scalar(base_daily["gross_return"], base_positions))

    # Pass two: the traded book.  The buffer filters the scaled target, so a
    # de-risking smaller than the band is suppressed along with signal churn.
    desired = base.mul(scalar.reindex(base.index), axis=0)
    launch_ts = pd.Timestamp(P["launch"])
    prior = desired.index[desired.index < launch_ts]
    start_at = prior[-1] if len(prior) else desired.index[0]
    buffered = apply_no_trade_buffer(desired.loc[start_at:], P["no_trade_buffer"])

    daily, _ = simulate(
        buffered, data, one_way_cost,
        capital=P["initial_capital"], integer=True, charge_costs=True, launch=P["launch"],
    )
    return daily


def metrics(daily: pd.DataFrame, start: str, end: str | None = None) -> pd.Series:
    """Headline net performance over one window."""
    returns = daily.loc[start:end, "net_return"].dropna()
    location = daily.index.get_indexer([returns.index[0]])[0]
    origin = daily.index[location - 1] if location > 0 else returns.index[0]
    years = (returns.index[-1] - origin).total_seconds() / (365.2425 * 86_400)

    equity = (1 + returns).cumprod()
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:] - 1
    monthly = (1 + returns).resample("ME").prod() - 1
    downside = math.sqrt(float(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2)))
    a = P["annualization"]

    return pd.Series({
        "Start": returns.index[0].date().isoformat(),
        "End": returns.index[-1].date().isoformat(),
        "Years": years,
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "Annualized volatility": returns.std() * math.sqrt(a),
        "Sharpe (rf=0)": returns.mean() / returns.std() * math.sqrt(a),
        "Sortino (rf=0)": returns.mean() * math.sqrt(a) / downside if downside else np.nan,
        "Max drawdown": drawdown.min(),
        "Calmar": (equity.iloc[-1] ** (1 / years) - 1) / abs(drawdown.min()),
        "Positive months": (monthly > 0).mean(),
        # Mean daily cost as a fraction of the NAV that bore it, annualized --
        # the same convention the package reports, so the two agree exactly.
        # Sigma(cost) / mean(NAV) understates it on a compounding book, because
        # late costs are divided by a NAV far larger than the one they hit.
        "Annual cost drag": daily.loc[start:end, "cost"].mean() * a,
        "Average gross notional multiple": daily.loc[start:end, "gross_notional_multiple"].mean(),
        "Average markets held": daily.loc[start:end, "active_markets"].mean(),
    })


WINDOWS = {
    "1990-2004 development": ("1990-01-01", "2004-12-31"),
    "2005-2014 later window": ("2005-01-01", "2014-12-31"),
    "1990-2014 full history": ("1990-01-01", "2014-12-31"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="Directory holding 'Futures Data/'")
    parser.add_argument("--output", default=None, help="Optional CSV path for the daily ledger")
    args = parser.parse_args()

    daily = run(args.data_dir)
    table = pd.DataFrame({name: metrics(daily, *window) for name, window in WINDOWS.items()})

    pd.set_option("display.width", 140)
    print(f"\nDelta1 reference implementation — {len(SYMBOLS)} futures markets\n")
    print(table.to_string(float_format=lambda v: f"{v:,.4f}"))
    if args.output:
        daily.to_csv(args.output, index_label="date")
        print(f"\nDaily ledger written to {args.output}")


if __name__ == "__main__":
    main()
