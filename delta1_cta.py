"""Cross-asset time-series momentum backtest for the DELTA1 case.

The implementation is deliberately compact and auditable:

* additive back-adjusted futures are handled with price changes, not percentage returns;
* signals, volatility estimates, and leverage are known before the return they trade;
* the portfolio rebalances monthly and includes explicit spread/commission estimates;
* asset classes receive equal ex-ante risk, then contracts share risk within each class.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ASSET_CLASSES: dict[str, tuple[str, ...]] = {
    "Equity indices": ("ES", "NQ", "RTY", "NKD"),
    "Government bonds": ("ZT", "ZF", "ZN", "ZB"),
    "G10 FX": ("6A", "6B", "6C", "6E", "6J", "6S"),
    "Commodities": ("CL", "NG", "GC", "SI", "HG", "ZC", "ZW", "ZS"),
}

SYMBOL_TO_CLASS = {
    symbol: asset_class
    for asset_class, symbols in ASSET_CLASSES.items()
    for symbol in symbols
}


@dataclass(frozen=True)
class BacktestConfig:
    data_dir: Path
    output_dir: Path
    oos_start: str = "2005-01-01"
    oos_end: str | None = None
    horizons: tuple[int, ...] = (252,)
    vol_span: int = 60
    portfolio_vol_window: int = 63
    target_vol: float = 0.10
    max_leverage: float = 2.0
    min_leverage: float = 0.25
    half_spread_ticks: float = 0.50
    commission_per_contract: float = 2.50
    annualization: int = 252


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    positions: pd.DataFrame
    target_positions: pd.DataFrame
    signals: pd.DataFrame
    prices: pd.DataFrame
    metadata: pd.DataFrame


def _symbols() -> list[str]:
    return [symbol for symbols in ASSET_CLASSES.values() for symbol in symbols]


def load_metadata(data_dir: Path, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load contract specifications from the supplied Norgate catalogue."""
    wanted = set(symbols or _symbols())
    catalogue = pd.read_csv(data_dir / "CATALOGUE_Delta1_Futures.csv")
    catalogue["clean_symbol"] = catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    catalogue = catalogue[
        catalogue["symbol"].str.endswith("_CCB")
        & catalogue["clean_symbol"].isin(wanted)
    ].copy()
    catalogue = catalogue.drop_duplicates("clean_symbol").set_index("clean_symbol")
    for column in ("tick_size", "point_value"):
        catalogue[column] = pd.to_numeric(catalogue[column], errors="coerce")
    missing = wanted.difference(catalogue.index)
    if missing:
        raise ValueError(f"Missing catalogue rows for: {sorted(missing)}")
    if not (catalogue["currency"] == "USD").all():
        bad = catalogue.loc[catalogue["currency"] != "USD", "currency"].to_dict()
        raise ValueError(f"Universe must be USD-denominated; found {bad}")
    catalogue["asset_class"] = catalogue.index.map(SYMBOL_TO_CLASS)
    return catalogue.sort_index()


def load_prices(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Load close prices and align them to a common business-day calendar."""
    series = []
    for symbol in symbols or _symbols():
        path = data_dir / "Futures Data" / f"&{symbol}_CCB.csv"
        frame = pd.read_csv(path, usecols=["Date", "Close"], parse_dates=["Date"])
        frame = frame.drop_duplicates("Date", keep="last").sort_values("Date")
        series.append(frame.set_index("Date")["Close"].rename(symbol))
    prices = pd.concat(series, axis=1, sort=False).sort_index()
    calendar = pd.bdate_range(prices.index.min(), prices.index.max())
    # Short exchange holidays are zero-P&L days; do not bridge long data outages.
    prices = prices.reindex(calendar).ffill(limit=ffill_limit)
    return prices


def trend_signal(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Average the direction of trends across several pre-declared horizons."""
    votes = [np.sign(prices - prices.shift(horizon)) for horizon in horizons]
    signal = sum(votes) / len(votes)
    valid = pd.concat(
        [(prices - prices.shift(horizon)).notna() for horizon in horizons],
        axis=1,
        keys=range(len(horizons)),
    ).T.groupby(level=1).all().T
    return signal.where(valid)


def _month_end_rows(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame.groupby(frame.index.to_period("M")).tail(1)


def _base_target_positions(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    metadata: pd.DataFrame,
    config: BacktestConfig,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    class_weights: dict[str, float] | None = None,
    point_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    asset_classes = asset_classes or ASSET_CLASSES
    price_change = prices.diff()
    daily_price_vol = price_change.ewm(
        span=config.vol_span,
        min_periods=config.vol_span,
        adjust=False,
    ).std()
    if point_values is not None:
        annual_dollar_vol = daily_price_vol * point_values * math.sqrt(config.annualization)
    else:
        annual_dollar_vol = daily_price_vol.mul(
            metadata["point_value"], axis=1
        ) * math.sqrt(config.annualization)

    positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for asset_class, symbols in asset_classes.items():
        weight = (
            class_weights[asset_class] if class_weights else 1.0 / len(asset_classes)
        )
        available = signals.loc[:, symbols].notna() & annual_dollar_vol.loc[:, symbols].gt(0)
        n_available = available.sum(axis=1).replace(0, np.nan)
        risk_budget = config.target_vol * np.sqrt(weight / n_available)
        positions.loc[:, symbols] = (
            signals.loc[:, symbols]
            .mul(risk_budget, axis=0)
            .div(annual_dollar_vol.loc[:, symbols])
        )
    return positions.replace([np.inf, -np.inf], np.nan)


def _held_positions(monthly_targets: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    # A target formed at month-end close becomes active on the next business day.
    return monthly_targets.reindex(index).ffill().shift(1).fillna(0.0)


def _gross_returns(
    held_positions: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    point_values: pd.DataFrame | None = None,
) -> pd.Series:
    # point_values, when given, is a date x symbol frame of USD point values
    # (native point value times the day's FX rate) for non-USD contracts.
    if point_values is not None:
        contract_pnl = prices.diff() * point_values
    else:
        contract_pnl = prices.diff().mul(metadata["point_value"], axis=1)
    return (held_positions * contract_pnl).sum(axis=1, min_count=1).fillna(0.0)


def _portfolio_leverage(
    base_gross_returns: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    realized_vol = (
        base_gross_returns.rolling(
            config.portfolio_vol_window,
            min_periods=config.portfolio_vol_window,
        ).std()
        * math.sqrt(config.annualization)
    )
    # Zero volatility means a structurally flat book, not infinite conviction:
    # replace it before clipping so it falls back to neutral leverage instead
    # of being pinned at the cap.
    ratio = config.target_vol / realized_vol.replace(0.0, np.nan)
    return ratio.clip(config.min_leverage, config.max_leverage).fillna(1.0)


def run_backtest(
    config: BacktestConfig,
    *,
    name: str = "12-month TSMOM CTA",
    signal_override: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the strategy with strict signal-to-return lags and trading costs."""
    symbols = _symbols()
    metadata = metadata if metadata is not None else load_metadata(config.data_dir, symbols)
    prices = prices if prices is not None else load_prices(config.data_dir, symbols)
    signals = (
        signal_override.reindex_like(prices)
        if signal_override is not None
        else trend_signal(prices, config.horizons)
    )

    base_target = _base_target_positions(prices, signals, metadata, config)
    base_monthly = _month_end_rows(base_target)
    base_held = _held_positions(base_monthly, prices.index)
    base_gross = _gross_returns(base_held, prices, metadata)

    leverage = _portfolio_leverage(base_gross, config)
    monthly_leverage = _month_end_rows(leverage)
    monthly_target = base_monthly.mul(monthly_leverage.reindex(base_monthly.index), axis=0)
    positions = _held_positions(monthly_target, prices.index)

    gross = _gross_returns(positions, prices, metadata)
    turnover = positions.diff().abs().fillna(positions.abs())
    one_way_cost = (
        config.half_spread_ticks
        * metadata["tick_size"]
        * metadata["point_value"]
        + config.commission_per_contract
    )
    costs = turnover.mul(one_way_cost, axis=1).sum(axis=1)
    net = gross - costs

    daily = pd.DataFrame(
        {
            "gross_return": gross,
            "cost": costs,
            "net_return": net,
            "leverage": monthly_leverage.reindex(prices.index).ffill().shift(1).fillna(1.0),
        }
    )
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["gross_equity"] = (1.0 + daily["gross_return"]).cumprod()
    return BacktestResult(name, daily, positions, monthly_target, signals, prices, metadata)


def performance_metrics(
    result: BacktestResult,
    start: str,
    end: str | None = None,
    annualization: int = 252,
) -> pd.Series:
    daily = result.daily.loc[start:end].copy()
    returns = daily["net_return"].dropna()
    if returns.empty:
        raise ValueError(f"No returns in evaluation window for {result.name}")
    years = len(returns) / annualization
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    ann_return = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = returns.std() * math.sqrt(annualization)
    downside = returns.clip(upper=0).std() * math.sqrt(annualization)
    monthly = (1 + returns).resample("ME").prod() - 1
    return pd.Series(
        {
            "Start": returns.index.min().date().isoformat(),
            "End": returns.index.max().date().isoformat(),
            "Years": years,
            "CAGR": ann_return,
            "Annualized volatility": ann_vol,
            "Sharpe (rf=0)": returns.mean() / returns.std() * math.sqrt(annualization),
            "Sortino (rf=0)": returns.mean() / downside * annualization if downside > 0 else np.nan,
            "Max drawdown": drawdown.min(),
            "Calmar": ann_return / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
            "Positive months": (monthly > 0).mean(),
            "Best month": monthly.max(),
            "Worst month": monthly.min(),
            "Annual cost drag": daily["cost"].mean() * annualization,
            "Average leverage": daily["leverage"].mean(),
        },
        name=result.name,
    )


def contribution_by_class(
    result: BacktestResult,
    config: BacktestConfig,
    asset_classes: dict[str, tuple[str, ...]] | None = None,
    point_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    asset_classes = asset_classes or ASSET_CLASSES
    if point_values is not None:
        contract_pnl = result.prices.diff() * point_values
    else:
        contract_pnl = result.prices.diff().mul(result.metadata["point_value"], axis=1)
    contribution = result.positions * contract_pnl
    contribution = contribution.loc[config.oos_start : config.oos_end]
    out = {}
    for asset_class, symbols in asset_classes.items():
        class_return = contribution.loc[:, symbols].sum(axis=1)
        out[asset_class] = {
            "Annualized mean contribution": class_return.mean() * config.annualization,
            "Annualized standalone volatility": class_return.std()
            * math.sqrt(config.annualization),
        }
    return pd.DataFrame(out).T


def benchmark_results(
    config: BacktestConfig,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> list[BacktestResult]:
    long_only = pd.DataFrame(1.0, index=prices.index, columns=prices.columns).where(prices.notna())
    multi_horizon = trend_signal(prices, (21, 63, 252))
    return [
        run_backtest(
            config,
            name="Long-only risk-balanced",
            signal_override=long_only,
            prices=prices,
            metadata=metadata,
        ),
        run_backtest(
            config,
            name="Multi-horizon trend alternative",
            signal_override=multi_horizon,
            prices=prices,
            metadata=metadata,
        ),
    ]


def robustness_table(
    config: BacktestConfig,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    variants = {
        "Short/medium (1m, 3m)": (21, 63),
        "Balanced (1m, 3m, 12m)": (21, 63, 252),
        "Medium/long (3m, 6m, 12m)": (63, 126, 252),
        "Classic (12m)": (252,),
    }
    rows = []
    for label, horizons in variants.items():
        for cost_multiple in (0.0, 1.0, 2.0):
            variant_config = replace(
                config,
                horizons=horizons,
                half_spread_ticks=config.half_spread_ticks * cost_multiple,
                commission_per_contract=config.commission_per_contract * cost_multiple,
            )
            result = run_backtest(
                variant_config,
                name=label,
                prices=prices,
                metadata=metadata,
            )
            metrics = performance_metrics(result, config.oos_start, config.oos_end)
            rows.append(
                {
                    "Signal": label,
                    "Cost multiple": cost_multiple,
                    "CAGR": metrics["CAGR"],
                    "Volatility": metrics["Annualized volatility"],
                    "Sharpe": metrics["Sharpe (rf=0)"],
                    "Max drawdown": metrics["Max drawdown"],
                }
            )
    return pd.DataFrame(rows)


def _save_performance_plot(results: list[BacktestResult], config: BacktestConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2.2, 1])
    colors = ["#0B6E4F", "#5B6770", "#7A5195"]
    for result, color in zip(results, colors, strict=True):
        returns = result.daily.loc[config.oos_start : config.oos_end, "net_return"]
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        axes[0].plot(equity.index, equity, label=result.name, color=color, linewidth=1.8)
        if result is results[0]:
            axes[1].fill_between(drawdown.index, drawdown, 0, color=color, alpha=0.35)
            axes[1].plot(drawdown.index, drawdown, color=color, linewidth=1)
    axes[0].set_title("DELTA1 cross-asset trend strategy — out-of-sample growth of $1")
    axes[0].set_ylabel("Growth of $1 (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_ylabel("CTA drawdown")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(config.output_dir / "performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    strategy: BacktestResult,
    benchmarks: list[BacktestResult],
    robustness: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = [strategy, *benchmarks]
    metrics = pd.concat(
        [performance_metrics(result, config.oos_start, config.oos_end) for result in results],
        axis=1,
    ).T
    metrics.to_csv(config.output_dir / "performance_metrics.csv")

    curve = pd.concat(
        {
            result.name: (1 + result.daily.loc[config.oos_start : config.oos_end, "net_return"]).cumprod()
            for result in results
        },
        axis=1,
    )
    curve.to_csv(config.output_dir / "equity_curves.csv", index_label="Date")

    yearly = pd.concat(
        {
            result.name: (
                (1 + result.daily.loc[config.oos_start : config.oos_end, "net_return"])
                .resample("YE")
                .prod()
                - 1
            )
            for result in results
        },
        axis=1,
    )
    yearly.index = yearly.index.year
    yearly.to_csv(config.output_dir / "yearly_returns.csv", index_label="Year")
    contribution_by_class(strategy, config).to_csv(
        config.output_dir / "class_contributions.csv", index_label="Asset class"
    )
    robustness.to_csv(config.output_dir / "robustness.csv", index=False)
    _save_performance_plot(results, config)

    serializable = asdict(config)
    # Keep committed artifacts portable and avoid publishing a workstation path.
    serializable["data_dir"] = "${DELTA1_DATA_DIR}"
    serializable["output_dir"] = "outputs"
    (config.output_dir / "run_config.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    return {
        "metrics": config.output_dir / "performance_metrics.csv",
        "equity": config.output_dir / "equity_curves.csv",
        "yearly": config.output_dir / "yearly_returns.csv",
        "robustness": config.output_dir / "robustness.csv",
        "plot": config.output_dir / "performance.png",
    }


def run_pipeline(config: BacktestConfig) -> tuple[BacktestResult, list[BacktestResult], pd.DataFrame]:
    metadata = load_metadata(config.data_dir)
    prices = load_prices(config.data_dir)
    strategy = run_backtest(config, prices=prices, metadata=metadata)
    benchmarks = benchmark_results(config, prices, metadata)
    robustness = robustness_table(config, prices, metadata)
    save_outputs(strategy, benchmarks, robustness, config)
    return strategy, benchmarks, robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("delta1_strategy/outputs"))
    parser.add_argument("--oos-start", default="2005-01-01")
    parser.add_argument("--oos-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )
    strategy, benchmarks, _ = run_pipeline(config)
    metrics = pd.concat(
        [performance_metrics(result, config.oos_start, config.oos_end) for result in [strategy, *benchmarks]],
        axis=1,
    ).T
    print(metrics.to_string())


if __name__ == "__main__":
    main()
