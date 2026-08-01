"""Institutional-style, cost-aware extension of the DELTA1 CTA.

This module does **not** claim to reproduce any proprietary strategy used by
Jane Street, Optiver, Citadel, or another trading firm.  It translates public,
widely understood quant-engineering principles into a daily-futures research
setting:

1. keep the alpha model simple and falsifiable;
2. separate forecast, risk, execution, and accounting layers;
3. react to volatility shocks without using future information;
4. trade only when the desired change is large enough to justify its cost;
5. make every position and P&L term auditable from contract specifications.

The supplied data are daily continuous futures, so this is a medium-frequency
portfolio strategy.  It is not an intraday market-making model; the dataset has
no order book, quotes, queue position, or fill data required for one.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
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
    load_metadata,
    load_prices,
    performance_metrics,
    run_backtest,
)


@dataclass(frozen=True)
class InstitutionalConfig:
    """Controls for the forecast and execution overlays.

    Attributes
    ----------
    slow_horizon:
        Core trend lookback in business days. 252 approximates one year and is
        pre-specified from the time-series momentum literature.
    confirmation_horizon:
        Faster horizon used only to scale conviction. It never reverses the
        core forecast by itself.
    disagreement_scale:
        Fraction of the core position retained when medium- and slow-horizon
        directions disagree.
    fast_vol_span / slow_vol_span:
        EWM spans used to detect a market-specific volatility shock.
    shock_start / shock_full:
        Fast/slow volatility ratios at which de-risking starts and reaches its
        maximum. Exposure is linearly tapered between the two thresholds.
    shock_floor:
        Minimum fraction of forecast retained during a full volatility shock.
    no_trade_buffer:
        A target change smaller than this fraction of the larger of the old and
        new positions is not traded. This is a transparent turnover control.
    """

    slow_horizon: int = 252
    confirmation_horizon: int = 63
    disagreement_scale: float = 1.00
    fast_vol_span: int = 20
    slow_vol_span: int = 120
    shock_start: float = 1.35
    shock_full: float = 2.00
    shock_floor: float = 0.75
    no_trade_buffer: float = 0.25

    def validate(self) -> None:
        if self.confirmation_horizon >= self.slow_horizon:
            raise ValueError("confirmation_horizon must be shorter than slow_horizon")
        if not 0 <= self.disagreement_scale <= 1:
            raise ValueError("disagreement_scale must be in [0, 1]")
        if not 0 <= self.shock_floor <= 1:
            raise ValueError("shock_floor must be in [0, 1]")
        if self.shock_full <= self.shock_start:
            raise ValueError("shock_full must exceed shock_start")
        if not 0 <= self.no_trade_buffer < 1:
            raise ValueError("no_trade_buffer must be in [0, 1)")


@dataclass
class InstitutionalResult:
    """Strategy result plus layer-level diagnostics used for audit and review."""

    backtest: BacktestResult
    core_direction: pd.DataFrame
    confirmation: pd.DataFrame
    volatility_ratio: pd.DataFrame
    shock_multiplier: pd.DataFrame
    desired_monthly_positions: pd.DataFrame
    buffered_monthly_positions: pd.DataFrame


def build_institutional_forecast(
    prices: pd.DataFrame,
    config: InstitutionalConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a slow trend forecast with confirmation and shock-aware sizing.

    Every value at date ``t`` depends only on closes at or before ``t``. The
    backtest activates month-end forecasts on the next business day.
    """
    config.validate()
    price_change = prices.diff()

    core_direction = np.sign(prices - prices.shift(config.slow_horizon))
    confirmation = np.sign(prices - prices.shift(config.confirmation_horizon))
    agreement = confirmation.eq(core_direction)
    conviction = agreement.astype(float).where(agreement, config.disagreement_scale)

    fast_vol = price_change.ewm(
        span=config.fast_vol_span,
        min_periods=config.fast_vol_span,
        adjust=False,
    ).std()
    slow_vol = price_change.ewm(
        span=config.slow_vol_span,
        min_periods=config.slow_vol_span,
        adjust=False,
    ).std()
    volatility_ratio = fast_vol / slow_vol.replace(0, np.nan)

    progress = (
        (volatility_ratio - config.shock_start)
        / (config.shock_full - config.shock_start)
    ).clip(0, 1)
    shock_multiplier = 1 - progress * (1 - config.shock_floor)

    forecast = (core_direction * conviction * shock_multiplier).clip(-1, 1)
    valid = (
        prices.shift(config.slow_horizon).notna()
        & slow_vol.notna()
        & volatility_ratio.notna()
    )
    forecast = forecast.where(valid)
    diagnostics = {
        "core_direction": core_direction.where(valid),
        "confirmation": confirmation.where(valid),
        "volatility_ratio": volatility_ratio.where(valid),
        "shock_multiplier": shock_multiplier.where(valid),
    }
    return forecast, diagnostics


def apply_no_trade_buffer(
    desired: pd.DataFrame,
    buffer_fraction: float,
) -> pd.DataFrame:
    """Convert desired month-end positions into cost-aware executable targets.

    The buffer is stateful: a small target move retains the prior target. A
    sign change is always traded because its distance exceeds the buffer.
    Missing desired positions are flattened rather than silently carried.
    """
    if not 0 <= buffer_fraction < 1:
        raise ValueError("buffer_fraction must be in [0, 1)")

    executable = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
    previous = pd.Series(0.0, index=desired.columns)
    for date, raw_target in desired.iterrows():
        target = raw_target.fillna(0.0)
        change = target - previous
        reference = pd.concat([target.abs(), previous.abs()], axis=1).max(axis=1)
        should_trade = change.abs() > buffer_fraction * reference
        current = previous.where(~should_trade, target)
        executable.loc[date] = current
        previous = current
    return executable


def _cost_series(
    positions: pd.DataFrame,
    metadata: pd.DataFrame,
    base_config: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    turnover = positions.diff().abs().fillna(positions.abs())
    one_way_cost = (
        base_config.half_spread_ticks
        * metadata["tick_size"]
        * metadata["point_value"]
        + base_config.commission_per_contract
    )
    return turnover.mul(one_way_cost, axis=1).sum(axis=1), turnover.sum(axis=1)


def run_institutional_backtest(
    base_config: BacktestConfig,
    institutional_config: InstitutionalConfig | None = None,
    *,
    prices: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
) -> InstitutionalResult:
    """Run the layered forecast-risk-execution-accounting pipeline."""
    config = institutional_config or InstitutionalConfig()
    config.validate()
    metadata = metadata if metadata is not None else load_metadata(base_config.data_dir)
    prices = prices if prices is not None else load_prices(base_config.data_dir)

    forecast, diagnostics = build_institutional_forecast(prices, config)
    base_target = _base_target_positions(prices, forecast, metadata, base_config)
    base_monthly = _month_end_rows(base_target)

    # First pass estimates the risk of the unlevered forecast portfolio. The
    # month-end multiplier is known before it becomes active next business day.
    base_positions = _held_positions(base_monthly, prices.index)
    base_gross = _gross_returns(base_positions, prices, metadata)
    leverage = _portfolio_leverage(base_gross, base_config)
    monthly_leverage = _month_end_rows(leverage)
    desired = base_monthly.mul(monthly_leverage.reindex(base_monthly.index), axis=0)

    buffered = apply_no_trade_buffer(desired, config.no_trade_buffer)
    positions = _held_positions(buffered, prices.index)
    gross = _gross_returns(positions, prices, metadata)
    costs, turnover = _cost_series(positions, metadata, base_config)
    net = gross - costs

    daily = pd.DataFrame(
        {
            "gross_return": gross,
            "cost": costs,
            "net_return": net,
            "leverage": monthly_leverage.reindex(prices.index).ffill().shift(1).fillna(1.0),
            "contract_turnover_per_dollar": turnover,
        }
    )
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    daily["gross_equity"] = (1 + daily["gross_return"]).cumprod()

    result = BacktestResult(
        name="Robust Adaptive TSMOM",
        daily=daily,
        positions=positions,
        target_positions=buffered,
        signals=forecast,
        prices=prices,
        metadata=metadata,
    )
    return InstitutionalResult(
        backtest=result,
        core_direction=diagnostics["core_direction"],
        confirmation=diagnostics["confirmation"],
        volatility_ratio=diagnostics["volatility_ratio"],
        shock_multiplier=diagnostics["shock_multiplier"],
        desired_monthly_positions=desired,
        buffered_monthly_positions=buffered,
    )


def strategy_invariants(
    result: InstitutionalResult,
    base_config: BacktestConfig,
    institutional_config: InstitutionalConfig,
) -> pd.Series:
    """Return production-style invariant checks as explicit booleans."""
    daily = result.backtest.daily
    active_changes = result.backtest.positions.diff().abs().sum(axis=1).gt(1e-15)
    month_period = pd.Series(daily.index.to_period("M"), index=daily.index)
    month_transition = month_period.ne(month_period.shift(1))
    # positions formed at a month-end can change only on the first business day
    # of a new month; the first row is ignored as an initialization boundary.
    rebalance_timing_ok = bool((~active_changes.iloc[1:] | month_transition[1:]).all())
    forecast_bound = float(result.backtest.signals.abs().max().max()) <= 1 + 1e-12
    shock_bound = (
        float(result.shock_multiplier.min().min()) >= institutional_config.shock_floor - 1e-12
        and float(result.shock_multiplier.max().max()) <= 1 + 1e-12
    )
    return pd.Series(
        {
            "finite_net_returns": bool(np.isfinite(daily["net_return"]).all()),
            "non_negative_costs": bool((daily["cost"] >= -1e-15).all()),
            "forecast_in_minus_one_plus_one": forecast_bound,
            "shock_multiplier_bounded": shock_bound,
            "positions_change_only_at_month_boundary": rebalance_timing_ok,
            "oos_exceeds_five_years": len(daily.loc[base_config.oos_start : base_config.oos_end])
            / base_config.annualization
            >= 5,
        },
        name="Pass",
    )


def stress_test_table(
    base_config: BacktestConfig,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate economically motivated execution/risk perturbations OOS."""
    scenarios = [
        ("Base", InstitutionalConfig(), 1.0),
        ("No trade buffer", InstitutionalConfig(no_trade_buffer=0.0), 1.0),
        ("Wide 40% buffer", InstitutionalConfig(no_trade_buffer=0.40), 1.0),
        ("No volatility shock control", InstitutionalConfig(shock_floor=1.0), 1.0),
        (
            "Aggressive 3m confirmation",
            InstitutionalConfig(disagreement_scale=0.50),
            1.0,
        ),
        ("2x trading costs", InstitutionalConfig(), 2.0),
        ("3x trading costs", InstitutionalConfig(), 3.0),
    ]
    rows = []
    for label, strategy_config, cost_multiple in scenarios:
        scenario_base = BacktestConfig(
            **{
                **asdict(base_config),
                "half_spread_ticks": base_config.half_spread_ticks * cost_multiple,
                "commission_per_contract": base_config.commission_per_contract * cost_multiple,
            }
        )
        result = run_institutional_backtest(
            scenario_base,
            strategy_config,
            prices=prices,
            metadata=metadata,
        ).backtest
        metrics = performance_metrics(result, base_config.oos_start, base_config.oos_end)
        rows.append(
            {
                "Scenario": label,
                "CAGR": metrics["CAGR"],
                "Volatility": metrics["Annualized volatility"],
                "Sharpe": metrics["Sharpe (rf=0)"],
                "Max drawdown": metrics["Max drawdown"],
                "Annual cost drag": metrics["Annual cost drag"],
            }
        )
    return pd.DataFrame(rows).set_index("Scenario")


def save_institutional_outputs(
    result: InstitutionalResult,
    baseline: BacktestResult,
    stress: pd.DataFrame,
    invariants: pd.Series,
    base_config: BacktestConfig,
    institutional_config: InstitutionalConfig,
) -> None:
    output_dir = base_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.concat(
        [
            performance_metrics(result.backtest, base_config.oos_start, base_config.oos_end),
            performance_metrics(baseline, base_config.oos_start, base_config.oos_end),
        ],
        axis=1,
    ).T
    metrics.to_csv(output_dir / "institutional_metrics.csv")
    stress.to_csv(output_dir / "institutional_stress_tests.csv")
    invariants.to_csv(output_dir / "institutional_invariants.csv", header=True)

    curves = pd.concat(
        {
            result.backtest.name: (
                1 + result.backtest.daily.loc[base_config.oos_start :, "net_return"]
            ).cumprod(),
            baseline.name: (
                1 + baseline.daily.loc[base_config.oos_start :, "net_return"]
            ).cumprod(),
        },
        axis=1,
    )
    curves.to_csv(output_dir / "institutional_equity_curves.csv", index_label="Date")

    config_json = {
        "backtest": {
            **asdict(base_config),
            "data_dir": "${DELTA1_DATA_DIR}",
            "output_dir": "outputs",
        },
        "institutional": asdict(institutional_config),
    }
    (output_dir / "institutional_run_config.json").write_text(
        json.dumps(config_json, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    ax = curves.plot(figsize=(12, 5), color=["#005F73", "#6C757D"], linewidth=1.8)
    ax.set_yscale("log")
    ax.set_title("Robust Adaptive TSMOM vs. pre-specified 12-month baseline")
    ax.set_ylabel("Growth of $1, net of estimated costs (log scale)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "institutional_performance.png", dpi=180, bbox_inches="tight")
    plt.close()


def run_pipeline(
    base_config: BacktestConfig,
    institutional_config: InstitutionalConfig | None = None,
) -> InstitutionalResult:
    strategy_config = institutional_config or InstitutionalConfig()
    prices = load_prices(base_config.data_dir)
    metadata = load_metadata(base_config.data_dir)
    result = run_institutional_backtest(
        base_config,
        strategy_config,
        prices=prices,
        metadata=metadata,
    )
    baseline = run_backtest(base_config, prices=prices, metadata=metadata)
    stress = stress_test_table(base_config, prices, metadata)
    invariants = strategy_invariants(result, base_config, strategy_config)
    save_institutional_outputs(
        result,
        baseline,
        stress,
        invariants,
        base_config,
        strategy_config,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--oos-start", default="2005-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = BacktestConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        oos_start=args.oos_start,
    )
    result = run_pipeline(base_config)
    metrics = performance_metrics(result.backtest, base_config.oos_start)
    print(metrics.to_string())


if __name__ == "__main__":
    main()
