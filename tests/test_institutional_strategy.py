from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_cta import BacktestConfig, load_metadata, load_prices
from institutional_strategy import (
    InstitutionalConfig,
    apply_no_trade_buffer,
    build_institutional_forecast,
    run_institutional_backtest,
    strategy_invariants,
)


DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "data/Delta1"))


class TestInstitutionalForecast(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2000-01-03", periods=500)
        trend = np.linspace(100.0, 140.0, len(self.index))
        cycle = np.sin(np.arange(len(self.index)) / 11) * 2
        self.prices = pd.DataFrame({"X": trend + cycle}, index=self.index)
        self.config = InstitutionalConfig()

    def test_future_mutation_cannot_change_past_forecast(self) -> None:
        baseline, _ = build_institutional_forecast(self.prices, self.config)
        altered = self.prices.copy()
        altered.loc[self.index[400] :, "X"] *= -20
        changed, _ = build_institutional_forecast(altered, self.config)
        pd.testing.assert_frame_equal(
            baseline.loc[: self.index[399]],
            changed.loc[: self.index[399]],
        )

    def test_forecast_and_shock_controls_are_bounded(self) -> None:
        forecast, diagnostics = build_institutional_forecast(self.prices, self.config)
        self.assertLessEqual(float(forecast.abs().max().max()), 1.0)
        shock = diagnostics["shock_multiplier"].stack()
        self.assertGreaterEqual(float(shock.min()), self.config.shock_floor)
        self.assertLessEqual(float(shock.max()), 1.0)

    def test_invalid_configuration_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            InstitutionalConfig(shock_start=2.0, shock_full=1.0).validate()
        with self.assertRaises(ValueError):
            InstitutionalConfig(no_trade_buffer=1.0).validate()


class TestNoTradeBuffer(unittest.TestCase):
    def test_small_adjustment_is_suppressed_but_sign_flip_trades(self) -> None:
        dates = pd.to_datetime(["2005-01-31", "2005-02-28", "2005-03-31"])
        desired = pd.DataFrame({"X": [1.0, 1.10, -1.0]}, index=dates)
        actual = apply_no_trade_buffer(desired, 0.25)
        self.assertEqual(actual.iloc[0, 0], 1.0)
        self.assertEqual(actual.iloc[1, 0], 1.0)
        self.assertEqual(actual.iloc[2, 0], -1.0)

    def test_buffer_does_not_increase_target_change_count(self) -> None:
        dates = pd.date_range("2005-01-31", periods=24, freq="ME")
        desired = pd.DataFrame(
            {"X": 1 + np.sin(np.arange(len(dates)) / 3) * 0.15},
            index=dates,
        )
        actual = apply_no_trade_buffer(desired, 0.25)
        desired_changes = desired.diff().abs().gt(1e-15).sum().sum()
        actual_changes = actual.diff().abs().gt(1e-15).sum().sum()
        self.assertLessEqual(actual_changes, desired_changes)


@unittest.skipUnless(DATA_DIR.exists(), "Supplied DELTA1 data directory is not available")
class TestInstitutionalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = BacktestConfig(
            DATA_DIR, Path(tempfile.gettempdir()) / "delta1_test_outputs"
        )
        cls.strategy = InstitutionalConfig()
        prices = load_prices(DATA_DIR)
        metadata = load_metadata(DATA_DIR)
        cls.result = run_institutional_backtest(
            cls.base,
            cls.strategy,
            prices=prices,
            metadata=metadata,
        )

    def test_all_production_invariants_pass(self) -> None:
        checks = strategy_invariants(self.result, self.base, self.strategy)
        self.assertTrue(bool(checks.all()), checks.to_dict())

    def test_buffer_reduces_realized_cost_against_unbuffered(self) -> None:
        unbuffered = run_institutional_backtest(
            self.base,
            InstitutionalConfig(no_trade_buffer=0.0),
            prices=self.result.backtest.prices,
            metadata=self.result.backtest.metadata,
        )
        base_cost = self.result.backtest.daily.loc[self.base.oos_start :, "cost"].sum()
        unbuffered_cost = unbuffered.backtest.daily.loc[self.base.oos_start :, "cost"].sum()
        self.assertLess(base_cost, unbuffered_cost)


if __name__ == "__main__":
    unittest.main()
