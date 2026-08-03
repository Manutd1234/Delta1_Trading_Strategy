from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd

import delta1_strategy.research.friction as stress
from delta1_strategy.research.friction import cost_model_assumptions, run_friction_stress_suite


@dataclass(frozen=True)
class LightweightConfig:
    half_spread_ticks: float = 0.5
    slippage_ticks: float = 0.25
    commission_per_contract: float = 2.5
    exchange_and_regulatory_fees_per_contract: float = 1.5
    impact_bps_at_full_participation: float = 1.0
    charge_roll_costs: bool = True
    execution_timing: str = "next_close"
    execution_delay_sessions: int = 2
    max_rebalance_participation: float | None = 0.02
    annualization: int = 252


def metrics_for(config: LightweightConfig) -> dict[str, object]:
    cost_scale = (
        config.half_spread_ticks
        + config.slippage_ticks
        + config.commission_per_contract
        + config.exchange_and_regulatory_fees_per_contract
        + config.impact_bps_at_full_participation
    )
    return {
        "Start": "1990-01-01",
        "End": "2014-12-31",
        "CAGR": 0.20 - cost_scale / 10_000,
        "Annualized volatility": 0.10,
        "Naive daily Sharpe (sqrt252, rf=0)": 2.0 - config.execution_delay_sessions / 100,
        "Sortino (rf=0)": 2.5,
        "Max drawdown": -0.12,
        "Trade profit factor (contribution)": 1.4,
        "Trade profit factor (USD)": 1.35,
        "Trade expectancy (bps NAV)": 3.2,
        "Trade expectancy (USD)": 120.0,
        "Annual cost drag": cost_scale / 1_000,
        "Annual fixed-cost drag": (cost_scale - config.impact_bps_at_full_participation)
        / 1_000,
        "Annual impact-cost drag": config.impact_bps_at_full_participation / 1_000,
        "Peak gross notional multiple": 4.0,
        "Peak static margin fraction": 0.25,
        "Peak order participation": config.max_rebalance_participation,
    }


class TestCostModelAssumptions(unittest.TestCase):
    def test_helper_reports_all_execution_assumptions(self) -> None:
        config = LightweightConfig()
        report = cost_model_assumptions(config)
        self.assertEqual(
            report.columns.tolist(),
            ["parameter", "value", "category", "unit", "description"],
        )
        self.assertEqual(len(report), 9)
        values = report.set_index("parameter")["value"]
        self.assertEqual(values["half_spread_ticks"], 0.5)
        self.assertEqual(values["impact_bps_at_full_participation"], 1.0)
        self.assertEqual(values["execution_timing"], "next_close")
        self.assertEqual(values["max_rebalance_participation"], 0.02)

    def test_helper_rejects_incomplete_config(self) -> None:
        with self.assertRaisesRegex(AttributeError, "half_spread_ticks"):
            cost_model_assumptions(object())


class TestFrictionStressSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LightweightConfig()
        self.seen: list[LightweightConfig] = []

    def run_metrics(self, config: LightweightConfig) -> dict[str, object]:
        self.seen.append(config)
        return metrics_for(config)

    def test_suite_has_all_ordered_scenarios_and_descriptive_columns(self) -> None:
        report = run_friction_stress_suite(self.config, run_fn=self.run_metrics)
        self.assertEqual(
            report["scenario"].tolist(),
            [
                "baseline",
                "double_fixed_costs",
                "double_impact",
                "double_all_execution_costs",
                "additional_execution_delay_1_session",
                "tighter_participation_1pct",
                "combined_adverse",
            ],
        )
        self.assertTrue(
            report["validation_status"].eq("retrospective_sensitivity").all()
        )
        self.assertFalse(any("target" in column for column in report.columns))
        self.assertFalse(any("pass" in column or "fail" in column for column in report.columns))
        self.assertEqual(len(report), 7)
        self.assertEqual(set(report["start"]), {"1990-01-01"})
        self.assertEqual(set(report["end"]), {"2014-12-31"})

    def test_each_replacement_changes_only_its_stated_assumptions(self) -> None:
        run_friction_stress_suite(self.config, run_fn=self.run_metrics)
        baseline, fixed, impact, all_costs, delay, participation, combined = self.seen

        self.assertIsNot(baseline, self.config)
        self.assertEqual(baseline, self.config)

        self.assertEqual(fixed.half_spread_ticks, 1.0)
        self.assertEqual(fixed.slippage_ticks, 0.5)
        self.assertEqual(fixed.commission_per_contract, 5.0)
        self.assertEqual(fixed.exchange_and_regulatory_fees_per_contract, 3.0)
        self.assertEqual(fixed.impact_bps_at_full_participation, 1.0)

        self.assertEqual(impact.half_spread_ticks, 0.5)
        self.assertEqual(impact.impact_bps_at_full_participation, 2.0)

        self.assertEqual(all_costs.half_spread_ticks, 1.0)
        self.assertEqual(all_costs.impact_bps_at_full_participation, 2.0)
        self.assertEqual(delay.execution_delay_sessions, 3)
        self.assertEqual(participation.max_rebalance_participation, 0.01)
        self.assertEqual(combined.half_spread_ticks, 1.0)
        self.assertEqual(combined.impact_bps_at_full_participation, 2.0)
        self.assertEqual(combined.execution_delay_sessions, 3)
        self.assertEqual(combined.max_rebalance_participation, 0.01)
        self.assertEqual(self.config.execution_delay_sessions, 2)
        self.assertEqual(self.config.max_rebalance_participation, 0.02)

    def test_report_maps_the_requested_metrics(self) -> None:
        report = run_friction_stress_suite(self.config, run_fn=self.run_metrics)
        baseline = report.set_index("scenario").loc["baseline"]
        expected = metrics_for(self.config)
        self.assertEqual(baseline["cagr"], expected["CAGR"])
        self.assertEqual(
            baseline["annualized_volatility"], expected["Annualized volatility"]
        )
        self.assertEqual(
            baseline["episode_profit_factor"],
            expected["Trade profit factor (contribution)"],
        )
        self.assertEqual(
            baseline["episode_expectancy_bps_nav"],
            expected["Trade expectancy (bps NAV)"],
        )
        self.assertEqual(baseline["annual_cost_drag"], expected["Annual cost drag"])
        self.assertEqual(
            baseline["peak_order_participation"],
            expected["Peak order participation"],
        )

    def test_missing_required_metric_fails_loudly(self) -> None:
        incomplete = metrics_for(self.config)
        del incomplete["CAGR"]
        with self.assertRaisesRegex(KeyError, "CAGR"):
            run_friction_stress_suite(self.config, run_fn=lambda _: incomplete)

    def test_default_runner_uses_canonical_functions_and_full_window(self) -> None:
        marker = object()
        with (
            patch.object(stress, "run_backtest", return_value=marker) as run,
            patch.object(
                stress,
                "performance_metrics",
                side_effect=lambda result, start, end, annualization: pd.Series(
                    metrics_for(self.config)
                ),
            ) as measure,
        ):
            report = run_friction_stress_suite(self.config)
        self.assertEqual(len(report), 7)
        self.assertEqual(run.call_count, 7)
        self.assertEqual(measure.call_count, 7)
        for call in measure.call_args_list:
            self.assertIs(call.args[0], marker)
            self.assertEqual(call.args[1:3], ("1990-01-01", "2014-12-31"))
            self.assertEqual(call.args[3], 252)


if __name__ == "__main__":
    unittest.main()
