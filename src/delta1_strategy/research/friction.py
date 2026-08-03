"""Retrospective execution-friction sensitivity analysis.

The scenarios in this module rerun the same historically selected strategy
under less favourable implementation assumptions.  They are diagnostics, not
independent validation and not production-readiness tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import pandas as pd

from .strategy import performance_metrics, run_backtest


RETROSPECTIVE_START = "1990-01-01"
RETROSPECTIVE_END = "2014-12-31"
VALIDATION_STATUS = "retrospective_sensitivity"


REPORT_COLUMNS = [
    "scenario",
    "assumption",
    "validation_status",
    "start",
    "end",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "episode_profit_factor",
    "episode_profit_factor_usd",
    "episode_expectancy_bps_nav",
    "episode_expectancy_usd",
    "annual_cost_drag",
    "annual_fixed_cost_drag",
    "annual_impact_cost_drag",
    "peak_gross_notional_multiple",
    "peak_static_margin_fraction",
    "peak_order_participation",
]


def cost_model_assumptions(config: Any) -> pd.DataFrame:
    """Describe the execution-cost and scheduling inputs on ``config``.

    Values are intentionally reported as assumptions rather than estimates of
    real-world accuracy.  The helper accepts any object exposing the canonical
    configuration attributes, which keeps it useful across config refactors.
    """

    specifications = (
        (
            "half_spread_ticks",
            "fixed",
            "ticks per contract per side",
            "Half of the quoted bid/ask spread charged on each execution.",
        ),
        (
            "slippage_ticks",
            "fixed",
            "ticks per contract per side",
            "Additional adverse movement charged on each execution.",
        ),
        (
            "commission_per_contract",
            "fixed",
            "USD per contract per side",
            "Broker commission charged on each execution.",
        ),
        (
            "exchange_and_regulatory_fees_per_contract",
            "fixed",
            "USD per contract per side",
            "Exchange and regulatory fees charged on each execution.",
        ),
        (
            "impact_bps_at_full_participation",
            "impact",
            "bps of traded notional",
            "Square-root market-impact coefficient at 100% realised participation.",
        ),
        (
            "charge_roll_costs",
            "rolls",
            "boolean",
            "Whether approximate two-leg continuous-contract roll costs are charged.",
        ),
        (
            "execution_timing",
            "scheduling",
            "execution convention",
            "Observed price field used for delayed execution.",
        ),
        (
            "execution_delay_sessions",
            "scheduling",
            "sessions",
            "Additional executable sessions waited after the normal next-session fill.",
        ),
        (
            "max_rebalance_participation",
            "capacity",
            "fraction of session volume",
            "Maximum fraction of observed session volume available to an order.",
        ),
    )
    rows: list[dict[str, Any]] = []
    for parameter, category, unit, description in specifications:
        if not hasattr(config, parameter):
            raise AttributeError(
                f"config does not expose required cost assumption {parameter!r}"
            )
        rows.append(
            {
                "parameter": parameter,
                "value": getattr(config, parameter),
                "category": category,
                "unit": unit,
                "description": description,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["parameter", "value", "category", "unit", "description"],
    )


def _scenario_configs(config: Any) -> list[tuple[str, str, Any]]:
    """Return ordered scenario names, descriptions, and replaced configs."""

    fixed_costs_2x = {
        "half_spread_ticks": config.half_spread_ticks * 2,
        "slippage_ticks": config.slippage_ticks * 2,
        "commission_per_contract": config.commission_per_contract * 2,
        "exchange_and_regulatory_fees_per_contract": (
            config.exchange_and_regulatory_fees_per_contract * 2
        ),
    }
    impact_2x = {
        "impact_bps_at_full_participation": (
            config.impact_bps_at_full_participation * 2
        )
    }
    delay_plus_one = config.execution_delay_sessions + 1
    participation_one_percent = 0.01

    return [
        (
            "baseline",
            "Canonical configuration; no stress applied.",
            replace(config),
        ),
        (
            "double_fixed_costs",
            "Half-spread, slippage, commission, and exchange/regulatory fees doubled.",
            replace(config, **fixed_costs_2x),
        ),
        (
            "double_impact",
            "Market-impact coefficient doubled; fixed costs unchanged.",
            replace(config, **impact_2x),
        ),
        (
            "double_all_execution_costs",
            "All fixed execution-cost inputs and the market-impact coefficient doubled.",
            replace(config, **fixed_costs_2x, **impact_2x),
        ),
        (
            "additional_execution_delay_1_session",
            "One executable session added to the configured execution delay.",
            replace(config, execution_delay_sessions=delay_plus_one),
        ),
        (
            "tighter_participation_1pct",
            "Maximum rebalance participation tightened to 1% of session volume.",
            replace(
                config,
                max_rebalance_participation=participation_one_percent,
            ),
        ),
        (
            "combined_adverse",
            "All execution costs doubled, one session added, and participation capped at 1%.",
            replace(
                config,
                **fixed_costs_2x,
                **impact_2x,
                execution_delay_sessions=delay_plus_one,
                max_rebalance_participation=participation_one_percent,
            ),
        ),
    ]


def _coerce_metrics(run_output: Any, annualization: int) -> pd.Series:
    """Measure a result, while permitting cheap metric fixtures in tests."""

    if isinstance(run_output, pd.Series):
        return run_output
    if isinstance(run_output, Mapping):
        return pd.Series(dict(run_output), dtype=object)
    if (
        isinstance(run_output, tuple)
        and len(run_output) == 2
        and isinstance(run_output[1], (pd.Series, Mapping))
    ):
        supplied = run_output[1]
        return supplied if isinstance(supplied, pd.Series) else pd.Series(dict(supplied))
    return performance_metrics(
        run_output,
        RETROSPECTIVE_START,
        RETROSPECTIVE_END,
        annualization,
    )


def _metric(metrics: pd.Series, name: str) -> Any:
    if name not in metrics.index:
        raise KeyError(f"stress result is missing required metric {name!r}")
    return metrics[name]


def run_friction_stress_suite(
    config: Any,
    run_fn: Callable[[Any], Any] | None = None,
) -> pd.DataFrame:
    """Run seven execution stresses over the full reused 1990--2014 history.

    By default each replaced configuration is sent to the canonical
    :func:`strategy.run_backtest`, then measured by
    :func:`strategy.performance_metrics`.  An injected ``run_fn`` may return a
    normal backtest result or, for lightweight tests, a metric Series/mapping.
    No row is labelled as a performance pass or fail because every scenario
    reuses the same retrospectively selected history.
    """

    runner = run_backtest if run_fn is None else run_fn
    annualization = getattr(config, "annualization", 252)
    rows: list[dict[str, Any]] = []
    for scenario, assumption, scenario_config in _scenario_configs(config):
        metrics = _coerce_metrics(runner(scenario_config), annualization)
        rows.append(
            {
                "scenario": scenario,
                "assumption": assumption,
                "validation_status": VALIDATION_STATUS,
                "start": _metric(metrics, "Start"),
                "end": _metric(metrics, "End"),
                "cagr": _metric(metrics, "CAGR"),
                "annualized_volatility": _metric(
                    metrics, "Annualized volatility"
                ),
                "sharpe": _metric(
                    metrics, "Naive daily Sharpe (sqrt252, rf=0)"
                ),
                "sortino": _metric(metrics, "Sortino (rf=0)"),
                "max_drawdown": _metric(metrics, "Max drawdown"),
                "episode_profit_factor": _metric(
                    metrics, "Trade profit factor (contribution)"
                ),
                "episode_profit_factor_usd": _metric(
                    metrics, "Trade profit factor (USD)"
                ),
                "episode_expectancy_bps_nav": _metric(
                    metrics, "Trade expectancy (bps NAV)"
                ),
                "episode_expectancy_usd": _metric(
                    metrics, "Trade expectancy (USD)"
                ),
                "annual_cost_drag": _metric(metrics, "Annual cost drag"),
                "annual_fixed_cost_drag": _metric(
                    metrics, "Annual fixed-cost drag"
                ),
                "annual_impact_cost_drag": _metric(
                    metrics, "Annual impact-cost drag"
                ),
                "peak_gross_notional_multiple": _metric(
                    metrics, "Peak gross notional multiple"
                ),
                "peak_static_margin_fraction": _metric(
                    metrics, "Peak static margin fraction"
                ),
                "peak_order_participation": _metric(
                    metrics, "Peak order participation"
                ),
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


__all__ = ["cost_model_assumptions", "run_friction_stress_suite"]
