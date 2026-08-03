from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from delta1_strategy.research.drawdown import (
    DrawdownOverlayConfig,
    overlay_performance_report,
    simulate_drawdown_overlay,
)


def daily_frame(
    gross_returns: list[float],
    costs: list[float] | None = None,
    gross_notional: list[float] | None = None,
) -> pd.DataFrame:
    count = len(gross_returns)
    return pd.DataFrame(
        {
            "gross_return": gross_returns,
            "cost": costs if costs is not None else [0.0] * count,
            "gross_notional_multiple": (
                gross_notional if gross_notional is not None else [2.0] * count
            ),
        },
        index=pd.date_range("2020-01-01", periods=count, freq="D"),
    )


class DrawdownOverlayConfigTests(unittest.TestCase):
    def test_invalid_thresholds_and_execution_lag_fail(self) -> None:
        with self.assertRaises(ValueError):
            DrawdownOverlayConfig(
                soft_drawdown_fraction=0.12,
                hard_drawdown_fraction=0.10,
            )
        with self.assertRaises(ValueError):
            DrawdownOverlayConfig(execution_lag_sessions=0)


class SimulateDrawdownOverlayTests(unittest.TestCase):
    def test_risk_reduction_is_delayed_and_uses_only_known_drawdown(self) -> None:
        config = DrawdownOverlayConfig(
            soft_drawdown_fraction=0.05,
            hard_drawdown_fraction=0.10,
            minimum_exposure_multiplier=0.25,
            execution_lag_sessions=2,
            overlay_turnover_cost_bps=0.0,
        )
        result = simulate_drawdown_overlay(
            daily_frame([-0.10, 0.0, 0.0, 0.0]),
            config,
        )

        np.testing.assert_allclose(
            result["exposure_multiplier"],
            [1.0, 1.0, 0.25, 0.25],
        )
        self.assertAlmostEqual(result.iloc[0]["overlay_drawdown_fraction"], 0.10)

    def test_future_return_cannot_change_earlier_exposure_or_returns(self) -> None:
        config = DrawdownOverlayConfig(execution_lag_sessions=2)
        first = simulate_drawdown_overlay(
            daily_frame([0.01, -0.08, 0.02, 0.03, 0.01]),
            config,
        )
        changed_future = simulate_drawdown_overlay(
            daily_frame([0.01, -0.08, 0.02, -0.40, 0.01]),
            config,
        )
        pd.testing.assert_frame_equal(first.iloc[:3], changed_future.iloc[:3])

    def test_risk_on_changes_are_capped_but_risk_off_is_immediate(self) -> None:
        config = DrawdownOverlayConfig(
            soft_drawdown_fraction=0.05,
            hard_drawdown_fraction=0.10,
            minimum_exposure_multiplier=0.25,
            max_risk_on_step=0.05,
            execution_lag_sessions=1,
            overlay_turnover_cost_bps=0.0,
        )
        result = simulate_drawdown_overlay(
            daily_frame([-0.10, 0.50, 0.50, 0.50]),
            config,
        )
        decisions = result["decision_multiplier_after_close"]
        self.assertAlmostEqual(decisions.iloc[0], 0.25)
        self.assertLessEqual(decisions.iloc[1] - decisions.iloc[0], 0.05 + 1e-12)

    def test_adjustment_cost_uses_prior_notional_and_multiplier_change(self) -> None:
        config = DrawdownOverlayConfig(
            soft_drawdown_fraction=0.05,
            hard_drawdown_fraction=0.10,
            minimum_exposure_multiplier=0.25,
            execution_lag_sessions=1,
            overlay_turnover_cost_bps=2.0,
        )
        result = simulate_drawdown_overlay(
            daily_frame([-0.10, 0.0], gross_notional=[4.0, 4.0]),
            config,
        )
        expected_turnover = (1.0 - 0.25) * 4.0
        self.assertAlmostEqual(
            result.iloc[1]["overlay_adjustment_turnover_multiple"],
            expected_turnover,
        )
        self.assertAlmostEqual(
            result.iloc[1]["overlay_adjustment_cost_return"],
            expected_turnover * 2.0 / 10_000.0,
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        frame = daily_frame([0.0, 0.0]).drop(columns="cost")
        with self.assertRaises(ValueError):
            simulate_drawdown_overlay(frame)
        frame = daily_frame([0.0, np.nan])
        with self.assertRaises(ValueError):
            simulate_drawdown_overlay(frame)

    def test_performance_report_compares_base_and_overlay(self) -> None:
        frame = daily_frame([0.01, -0.08, 0.01, 0.01, 0.01])
        overlay = simulate_drawdown_overlay(frame)
        report = overlay_performance_report(overlay)
        self.assertEqual(
            report["Path"].tolist(),
            ["base", "drawdown_overlay_diagnostic"],
        )
        self.assertTrue(np.isfinite(report["Max drawdown"]).all())

    def test_performance_report_includes_initial_capital_high_water_mark(self) -> None:
        frame = daily_frame([-0.10, 1.0 / 9.0])
        report = overlay_performance_report(simulate_drawdown_overlay(frame))
        base = report.set_index("Path").loc["base"]
        self.assertAlmostEqual(base["Max drawdown"], -0.10)


if __name__ == "__main__":
    unittest.main()
