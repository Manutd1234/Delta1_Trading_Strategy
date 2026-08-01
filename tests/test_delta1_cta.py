from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_cta import (
    BacktestConfig,
    _held_positions,
    load_metadata,
    load_prices,
    performance_metrics,
    run_backtest,
    trend_signal,
)


DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "data/Delta1"))


class TestSignalTiming(unittest.TestCase):
    def test_future_prices_do_not_change_past_signals(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=400)
        prices = pd.DataFrame({"X": np.linspace(100.0, 130.0, len(index))}, index=index)
        baseline = trend_signal(prices, (21, 63, 252))
        altered = prices.copy()
        altered.loc[index[300]:, "X"] *= -10
        changed = trend_signal(altered, (21, 63, 252))
        pd.testing.assert_frame_equal(baseline.loc[: index[299]], changed.loc[: index[299]])

    def test_month_end_target_is_held_from_next_business_day(self) -> None:
        index = pd.bdate_range("2005-01-28", "2005-02-03")
        targets = pd.DataFrame({"X": [1.0]}, index=[pd.Timestamp("2005-01-31")])
        held = _held_positions(targets, index)
        self.assertEqual(held.loc["2005-01-31", "X"], 0.0)
        self.assertEqual(held.loc["2005-02-01", "X"], 1.0)


@unittest.skipUnless(DATA_DIR.exists(), "Supplied DELTA1 data directory is not available")
class TestSuppliedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = load_metadata(DATA_DIR)
        cls.prices = load_prices(DATA_DIR)

    def test_universe_is_usd_and_complete(self) -> None:
        self.assertEqual(len(self.metadata), 22)
        self.assertTrue((self.metadata["currency"] == "USD").all())
        self.assertEqual(self.prices.shape[1], 22)

    def test_oos_backtest_is_finite(self) -> None:
        config = BacktestConfig(
            DATA_DIR, Path(tempfile.gettempdir()) / "delta1_test_outputs"
        )
        result = run_backtest(config, prices=self.prices, metadata=self.metadata)
        metrics = performance_metrics(result, config.oos_start)
        for field in ("CAGR", "Annualized volatility", "Sharpe (rf=0)", "Max drawdown"):
            self.assertTrue(np.isfinite(float(metrics[field])))
        self.assertGreater(float(metrics["Years"]), 5.0)


if __name__ == "__main__":
    unittest.main()
