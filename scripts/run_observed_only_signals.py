"""Rebuild every signal from observed prices only, and measure what the fill was worth.

    python scripts/run_observed_only_signals.py

The brief says not to fill artificially across holidays or missing values. The
canonical panel forward-fills with a 10-session limit, because 59 markets on
four continents share no holiday calendar and a held position must be marked
every session. This runner separates the two jobs that fill was doing:

  * **Marking** a held position through a session the market did not trade.
    Carrying the previous settlement is the correct economics -- a closed market
    genuinely did not move -- and it is retained. It contributes exactly zero
    P&L on the closed session and books the move in full on the next observed
    one, so no return is invented.

  * **Forming a signal.** Here the fill is a real concession, because a stale
    price entering a trend or volatility estimate changes the estimate. This
    runner removes it: every indicator is computed on each market's OWN observed
    trading sessions, with non-trading days absent from its history rather than
    imputed. The resulting decision is then carried across the sessions the
    market is shut, which holds a *decision* constant rather than inventing a
    price.

Naively masking filled cells inside the canonical frames does not test this: a
single gap inside a 252-session window voids the whole window, so the defined
forecast collapses by 96% and the comparison measures pandas' NaN propagation
rather than the fill. Compressing each market to its own calendar is what an
analyst who refused to fill would actually do.

Writes outputs/submission/robustness_no_fill.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "delta1_reference", ROOT / "reference" / "delta1_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observed_only_forecast(d1, data: dict) -> pd.DataFrame:
    """The forecast, with every indicator computed on observed sessions only."""
    P = d1.P
    calendar = data["prices"].index
    forecasts: dict[str, pd.Series] = {}

    for symbol in data["prices"].columns:
        # This market's own trading calendar: sessions the vendor actually
        # printed a close for. Holidays are absent, not imputed.
        observed = data["closes"][symbol].notna()
        index = calendar[observed]
        if len(index) < P["trend_lookback"] * 2:
            continue

        price = data["prices"].loc[index, symbol]
        unadjusted = data["unadjusted"].loc[index, symbol]
        volume = data["volumes"].loc[index, symbol]

        change = price.diff()
        daily_vol = change.ewm(
            span=P["vol_span"], min_periods=P["vol_span"], adjust=False
        ).std()

        trend = np.sign(price - price.shift(P["trend_lookback"])).where(
            (price - price.shift(P["trend_lookback"])).notna()
        )

        roll_gap = unadjusted.diff() - change
        roll_yield = (-roll_gap).rolling(
            P["basis_roll_window"], min_periods=P["basis_roll_window"]
        ).sum()
        raw = (roll_yield - roll_yield.shift(P["basis_lookback"])) / (
            daily_vol.replace(0, np.nan) * math.sqrt(P["annualization"])
        )
        scale = raw.rolling(
            P["signal_norm_window"], min_periods=P["signal_norm_window"]
        ).std().replace(0, np.nan)
        basis = (raw / scale).clip(-P["signal_cap"], P["signal_cap"]) / P["signal_cap"]

        blended = (
            trend * (1 - P["basis_weight"]) + basis.fillna(trend) * P["basis_weight"]
        ).clip(-1, 1).where(trend.notna())

        fast = change.ewm(
            span=P["fast_vol_span"], min_periods=P["fast_vol_span"], adjust=False
        ).std()
        slow = change.ewm(
            span=P["slow_vol_span"], min_periods=P["slow_vol_span"], adjust=False
        ).std()
        ratio = fast / slow.replace(0, np.nan)
        progress = (
            (ratio - P["shock_start"]) / (P["shock_full"] - P["shock_start"])
        ).clip(0, 1)
        shock = (1 - progress * (1 - P["shock_floor"])).where(ratio.notna())

        forecast = (blended * shock).clip(-1, 1)

        strategy_return = (
            forecast.shift(2) * change / daily_vol.shift(2).replace(0, np.nan)
        )
        realized = strategy_return.pow(2).rolling(
            P["risk_managed_window"], min_periods=P["risk_managed_window"]
        ).mean().pow(0.5)
        weight = (1.0 / realized.replace(0, np.nan)).clip(upper=P["risk_managed_cap"])
        forecast = (forecast * weight).clip(-1, 1).where(forecast.notna())

        tradeable = volume.fillna(0.0).rolling(
            P["volume_gate_window"], min_periods=P["volume_gate_window"]
        ).median().gt(P["min_median_contracts"])
        forecast = forecast.where(tradeable, np.nan)

        # Carry the DECISION across sessions this market is shut. No price is
        # invented; the book simply keeps yesterday's view until the market
        # reopens and tells it something new.
        forecasts[symbol] = forecast.reindex(calendar).ffill()

    return pd.DataFrame(forecasts, index=calendar).reindex(
        columns=data["prices"].columns
    )


def run_variant(d1, data: dict, forecast: pd.DataFrame) -> pd.DataFrame:
    """Replay the canonical sizing and execution against a supplied forecast."""
    P = d1.P
    prices, pv = data["prices"], data["point_values"]
    one_way_cost = (
        (P["half_spread_ticks"] + P["slippage_ticks"]) * pv.mul(data["tick_size"], axis=1)
        + P["commission"] + P["fees"]
    )
    base = d1.month_end_rows(d1.target_positions_per_dollar(prices, forecast, pv))
    base_daily, base_positions = d1.simulate(
        base, data, one_way_cost, capital=1.0, integer=False,
        charge_costs=False, launch=None,
    )
    scalar = d1.month_end_rows(
        d1.portfolio_risk_scalar(base_daily["gross_return"], base_positions)
    )
    desired = base.mul(scalar.reindex(base.index), axis=0)
    launch = pd.Timestamp(P["launch"])
    prior = desired.index[desired.index < launch]
    start_at = prior[-1] if len(prior) else desired.index[0]
    buffered = d1.apply_no_trade_buffer(desired.loc[start_at:], P["no_trade_buffer"])
    daily, _ = d1.simulate(
        buffered, data, one_way_cost, capital=P["initial_capital"],
        integer=True, charge_costs=True, launch=P["launch"],
    )
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="Round1AllData/Quant Researcher/Delta1")
    parser.add_argument("--output-dir", default="outputs/submission")
    args = parser.parse_args()

    d1 = load_reference()
    data = d1.load_market_data(Path(args.data_dir))

    filled = data["prices"].notna() & data["closes"].isna()
    filled_cells = int(filled.to_numpy().sum())
    populated = int(data["prices"].notna().to_numpy().sum())

    baseline = d1.run(data=data)
    observed = run_variant(d1, data, observed_only_forecast(d1, data))

    published = float(
        pd.read_csv(ROOT / "outputs/strategy_metrics.csv")
        .pipe(lambda f: f.loc[f["Window"].str.contains("full post-launch")])
        .iloc[0]["Naive daily Sharpe (sqrt252, rf=0)"]
    )
    measured = float(d1.metrics(baseline, "1990-01-01", "2014-12-31")["Sharpe (rf=0)"])
    if abs(measured - published) > 1e-12:
        raise SystemExit(f"baseline wiring check failed: {measured} vs {published}")

    windows = {
        "1990-2004 development": ("1990-01-01", "2004-12-31"),
        "2005-2014 out-of-sample": ("2005-01-01", "2014-12-31"),
        "1990-2014 full": ("1990-01-01", "2014-12-31"),
    }
    fields = ("CAGR", "Annualized volatility", "Sharpe (rf=0)", "Max drawdown",
              "Calmar", "Annual cost drag", "Average markets held")

    rows = []
    for window, (start, end) in windows.items():
        base_m = d1.metrics(baseline, start, end)
        obs_m = d1.metrics(observed, start, end)
        row = {"window": window,
               "sessions": int(baseline.loc[start:end, "net_return"].notna().sum())}
        for field in fields:
            row[f"canonical_{field}"] = float(base_m[field])
            row[f"observed_only_{field}"] = float(obs_m[field])
            row[f"delta_{field}"] = float(obs_m[field]) - float(base_m[field])
        rows.append(row)

    common = baseline.index.intersection(observed.index)
    correlation = float(
        baseline.loc[common, "net_return"].corr(observed.loc[common, "net_return"])
    )
    frame = pd.DataFrame(rows)
    frame["net_return_correlation"] = correlation
    frame["filled_cells"] = filled_cells
    frame["filled_share_of_populated"] = filled_cells / populated
    frame["variant"] = (
        "Signals rebuilt on each market's own observed sessions; non-trading days "
        "absent from its history rather than imputed. The decision is carried "
        "across closed sessions; no price is invented. Marking of held positions "
        "is unchanged, because a position must be marked every session and a "
        "closed market's settlement genuinely did not move."
    )
    frame["permitted_use"] = (
        "Sensitivity of the published result to the forward fill. Both paths reuse "
        "1990-2014 and are retrospective; neither is out of sample."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "robustness_no_fill.csv", index=False)

    full = frame.loc[frame["window"] == "1990-2014 full"].iloc[0]
    print(f"\nForward fill: {filled_cells:,} cells "
          f"({filled_cells / populated:.4%} of populated)\n")
    print(f"{'window':26s} {'canonical':>10s} {'observed':>10s} {'delta':>9s}")
    for _, row in frame.iterrows():
        print(f"{row['window']:26s} {row['canonical_Sharpe (rf=0)']:10.4f} "
              f"{row['observed_only_Sharpe (rf=0)']:10.4f} "
              f"{row['delta_Sharpe (rf=0)']:+9.4f}")
    print(f"\ndaily net-return correlation {correlation:.6f}")
    print(f"full-window Sharpe {full['canonical_Sharpe (rf=0)']:.4f} -> "
          f"{full['observed_only_Sharpe (rf=0)']:.4f} "
          f"({full['delta_Sharpe (rf=0)']:+.4f})")
    print(f"wrote {output_dir}/robustness_no_fill.csv")


if __name__ == "__main__":
    main()
