from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import (
    BacktestResult,
    CURRENT_OPTIMIZATION_TRIALS,
    GLOBAL_ASSET_CLASSES,
    PRIOR_TARGET_SEARCH_TRIALS,
    StrategyConfig,
    _held_positions,
    apply_no_trade_buffer,
    basis_momentum,
    blend_signals,
    load_fx_rates,
    load_unadjusted_prices,
    load_volumes,
    performance_metrics,
    risk_managed_forecast,
    run_backtest,
    run_pipeline,
    tradeable_mask,
    trend_signal,
    usd_point_values,
)


DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "data/Delta1"))
ROOT_DIR = Path(__file__).resolve().parents[1]


class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.base = StrategyConfig(Path("unused"))

    def test_defaults_are_the_optimized_specification(self) -> None:
        self.assertEqual(self.base.trend_lookback, 252)
        self.assertEqual(self.base.basis_weight, 0.5)
        self.assertEqual(self.base.target_vol, 0.10)
        self.assertEqual(self.base.max_leverage, 2.0)
        self.assertEqual(self.base.risk_budget, "flat")
        self.assertEqual(self.base.risk_managed_window, 63)
        self.assertEqual(self.base.risk_managed_cap, 2.0)
        self.assertEqual(self.base.no_trade_buffer, 0.25)
        self.assertEqual(CURRENT_OPTIMIZATION_TRIALS, 50)
        self.assertEqual(PRIOR_TARGET_SEARCH_TRIALS, 72)
        self.base.validate()

    def test_invalid_risk_and_signal_parameters_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            StrategyConfig(Path("unused"), basis_weight=1.1).validate()
        with self.assertRaises(ValueError):
            StrategyConfig(Path("unused"), shock_start=2.0, shock_full=1.0).validate()
        with self.assertRaises(ValueError):
            StrategyConfig(Path("unused"), no_trade_buffer=1.0).validate()
        with self.assertRaises(ValueError):
            StrategyConfig(Path("unused"), risk_budget="optimized").validate()
        with self.assertRaises(ValueError):
            StrategyConfig(Path("unused"), risk_managed_window=1).validate()


class TestOptimizationLedger(unittest.TestCase):
    def test_current_round_ledger_is_complete_and_marks_the_adopted_plateau(self) -> None:
        ledger = pd.read_csv(ROOT_DIR / "outputs" / "optimization_trials.csv")
        varied = [
            "trend_spec",
            "trend_horizons",
            "basis_weight",
            "risk_managed_window",
            "risk_managed_cap",
            "risk_budget",
            "no_trade_buffer",
        ]
        self.assertEqual(len(ledger), CURRENT_OPTIMIZATION_TRIALS)
        self.assertEqual(len(ledger[varied].drop_duplicates()), len(ledger))
        self.assertEqual(ledger["stage"].value_counts().to_dict(), {
            "risk_execution": 41,
            "alpha_screen": 9,
        })
        self.assertEqual(int(ledger["both_subperiods_target_pass"].sum()), 15)

        selected = ledger.loc[ledger["selection_status"] == "selected_plateau"]
        self.assertEqual(len(selected), 1)
        selected = selected.iloc[0]
        self.assertEqual(selected["trial_id"], "OPT025")
        self.assertEqual(selected["trend_spec"], "T12")
        self.assertEqual(float(selected["basis_weight"]), 0.5)
        self.assertEqual(float(selected["risk_managed_window"]), 63.0)
        self.assertEqual(selected["risk_budget"], "flat")
        self.assertEqual(float(selected["no_trade_buffer"]), 0.25)
        self.assertAlmostEqual(float(selected["combined_cagr"]), 0.2610900536, places=9)
        self.assertAlmostEqual(float(selected["combined_sharpe"]), 2.1088424570, places=9)

    def test_robustness_ledger_records_plateau_and_breadth_dependency(self) -> None:
        checks = pd.read_csv(ROOT_DIR / "outputs" / "strategy_robustness.csv")
        self.assertEqual(len(checks), 19)
        self.assertEqual(checks["variant"].nunique(), 19)
        self.assertFalse(bool(checks["later_stress_pass"].any()))

        local = checks[checks["category"].isin(["parameter_neighbor", "cost_stress"])]
        self.assertEqual(len(local), 9)
        self.assertTrue(bool(local["all_windows_pass"].all()))

        ablations = checks[checks["category"] == "construction_ablation"]
        self.assertEqual(len(ablations), 3)
        self.assertFalse(bool(ablations["all_windows_pass"].any()))

        leave_one_out = checks[checks["category"] == "leave_one_class_out"]
        self.assertEqual(len(leave_one_out), len(GLOBAL_ASSET_CLASSES))
        self.assertFalse(bool(leave_one_out["all_windows_pass"].any()))

    def test_retained_metrics_limit_the_target_claim_to_selected_windows(self) -> None:
        metrics = pd.read_csv(ROOT_DIR / "outputs" / "strategy_metrics.csv").set_index(
            "Window"
        )
        primary = metrics.loc["1990-2004 optimized window"]
        self.assertAlmostEqual(float(primary["CAGR"]), 0.2610900536, places=9)
        self.assertAlmostEqual(float(primary["Sharpe (rf=0)"]), 2.1088424570, places=9)
        self.assertTrue(bool(primary["Both targets met"]))
        self.assertFalse(
            bool(metrics.loc["2005-2014 later stress", "Both targets met"])
        )


class TestNotebookArtifact(unittest.TestCase):
    def test_notebook_is_executed_and_imports_the_canonical_strategy(self) -> None:
        notebook = json.loads((ROOT_DIR / "DELTA1_Strategy.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 15)

        cell_ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(
            [cell["execution_count"] for cell in code_cells],
            list(range(1, len(code_cells) + 1)),
        )
        self.assertIn(
            "from strategy import",
            "".join("".join(cell["source"]) for cell in code_cells),
        )
        outputs = [output for cell in code_cells for output in cell["outputs"]]
        self.assertFalse(any(output["output_type"] == "error" for output in outputs))
        self.assertEqual(
            sum("image/png" in output.get("data", {}) for output in outputs),
            4,
        )


class TestSignals(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2000-01-03", periods=1_500)
        rng = np.random.default_rng(31)
        self.prices = pd.DataFrame(
            {"X": 100 + rng.normal(0, 0.9, len(self.index)).cumsum()},
            index=self.index,
        )
        gaps = np.zeros(len(self.index))
        roll_jitter = rng.uniform(0.6, 1.8, len(gaps[::21]))
        roll_sign = np.where(np.arange(len(roll_jitter)) * 21 < 756, 1.0, -1.0)
        gaps[::21] = roll_jitter * roll_sign
        self.unadjusted = self.prices + pd.DataFrame(
            {"X": gaps.cumsum()}, index=self.index
        )

    def test_trend_is_causal_bounded_and_level_invariant(self) -> None:
        baseline = trend_signal(self.prices)
        shifted = trend_signal(self.prices - 500)
        pd.testing.assert_frame_equal(baseline, shifted)
        self.assertLessEqual(float(baseline.abs().max().max()), 1.0)

        changed = self.prices.copy()
        changed.iloc[1_300:] *= -20
        pd.testing.assert_frame_equal(
            baseline.iloc[:1_300], trend_signal(changed).iloc[:1_300]
        )

    def test_basis_is_causal_bounded_and_level_invariant(self) -> None:
        baseline = basis_momentum(self.prices, self.unadjusted)
        translated = basis_momentum(self.prices + 200, self.unadjusted + 200)
        pd.testing.assert_frame_equal(baseline, translated)
        self.assertLessEqual(float(baseline.abs().max().max()), 1.0)

        changed_prices = self.prices.copy()
        changed_unadjusted = self.unadjusted.copy()
        changed_prices.iloc[1_300:] += 40
        changed_unadjusted.iloc[1_300:] -= 30
        changed = basis_momentum(changed_prices, changed_unadjusted)
        pd.testing.assert_frame_equal(baseline.iloc[:1_300], changed.iloc[:1_300])

    def test_curve_moving_toward_backwardation_scores_positive(self) -> None:
        signal = basis_momentum(self.prices, self.unadjusted)
        self.assertGreater(float(signal["X"].loc["2004":].dropna().mean()), 0.0)

    def test_blend_falls_back_to_trend_before_basis_is_live(self) -> None:
        index = self.index[:3]
        trend = pd.DataFrame({"X": [0.8, 0.8, 0.8]}, index=index)
        basis = pd.DataFrame({"X": [np.nan, -0.4, np.nan]}, index=index)
        blended = blend_signals(trend, basis, basis_weight=0.5)
        self.assertAlmostEqual(blended.iloc[0, 0], 0.8)
        self.assertAlmostEqual(blended.iloc[1, 0], 0.2)
        self.assertAlmostEqual(blended.iloc[2, 0], 0.8)

    def test_risk_management_is_causal_bounded_and_sign_preserving(self) -> None:
        forecast = trend_signal(self.prices)
        baseline = risk_managed_forecast(
            forecast, self.prices, vol_span=60, window=126, cap=2.0
        )
        self.assertLessEqual(float(baseline.abs().max().max()), 1.0)
        live = baseline.notna() & forecast.notna() & baseline.ne(0) & forecast.ne(0)
        same_sign_product = (baseline * forecast).to_numpy()[live.to_numpy()]
        self.assertTrue(bool((same_sign_product > 0).all()))

        changed = self.prices.copy()
        changed.iloc[1_300:] += 100
        changed_forecast = trend_signal(changed)
        altered = risk_managed_forecast(
            changed_forecast, changed, vol_span=60, window=126, cap=2.0
        )
        pd.testing.assert_frame_equal(baseline.iloc[:1_300], altered.iloc[:1_300])


class TestRiskAndExecution(unittest.TestCase):
    def test_volume_gate_uses_only_trailing_observations(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=80)
        volumes = pd.DataFrame({"X": [0.0] * 60 + [2_000.0] * 20}, index=index)
        mask = tradeable_mask(volumes, window=10, min_median_contracts=1_000)
        self.assertFalse(bool(mask.iloc[60, 0]))
        self.assertTrue(bool(mask.iloc[-1, 0]))

    def test_buffer_suppresses_small_move_but_trades_sign_flip(self) -> None:
        dates = pd.to_datetime(["2005-01-31", "2005-02-28", "2005-03-31"])
        desired = pd.DataFrame({"X": [1.0, 1.1, -1.0]}, index=dates)
        actual = apply_no_trade_buffer(desired, 0.25)
        self.assertEqual(actual.iloc[:, 0].tolist(), [1.0, 1.0, -1.0])

    def test_month_end_target_activates_next_business_day(self) -> None:
        index = pd.bdate_range("2005-01-28", "2005-02-03")
        target = pd.DataFrame({"X": [1.0]}, index=[pd.Timestamp("2005-01-31")])
        held = _held_positions(target, index)
        self.assertEqual(held.loc["2005-01-31", "X"], 0.0)
        self.assertEqual(held.loc["2005-02-01", "X"], 1.0)

    def test_point_values_convert_foreign_currency_daily(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=2)
        metadata = pd.DataFrame(
            {"currency": ["USD", "EUR"], "point_value": [50.0, 10.0]},
            index=["US", "EU"],
        )
        fx = pd.DataFrame({"USD": [1.0, 1.0], "EUR": [1.1, 1.2]}, index=index)
        values = usd_point_values(metadata, fx, index)
        self.assertEqual(values["US"].tolist(), [50.0, 50.0])
        self.assertEqual(values["EU"].tolist(), [11.0, 12.0])


class TestMetrics(unittest.TestCase):
    def test_cagr_uses_elapsed_calendar_time(self) -> None:
        index = pd.to_datetime(["2020-01-01", "2021-01-01"])
        daily = pd.DataFrame(
            {
                "net_return": [0.0, 0.10],
                "cost": [0.0, 0.0],
                "leverage": [1.0, 1.0],
            },
            index=index,
        )
        empty = pd.DataFrame(index=index)
        result = BacktestResult("test", daily, empty, empty, empty, empty, empty)
        metrics = performance_metrics(result, "2020-01-01", "2021-01-01")
        expected_years = 366 / 365.2425
        self.assertAlmostEqual(float(metrics["Years"]), expected_years)
        self.assertAlmostEqual(
            float(metrics["CAGR"]), 1.10 ** (1 / expected_years) - 1
        )


@unittest.skipUnless(DATA_DIR.exists(), "Supplied DELTA1 data directory is not available")
class TestSuppliedDataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = StrategyConfig(DATA_DIR, Path(cls.temp.name))
        cls.result, cls.report = run_pipeline(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_production_invariants(self) -> None:
        daily = self.result.daily
        self.assertTrue(np.isfinite(daily["net_return"]).all())
        self.assertTrue((daily["cost"] >= 0).all())
        self.assertLessEqual(float(self.result.signals.abs().max().max()), 1.0)

        changed = self.result.positions.diff().abs().sum(axis=1).gt(1e-15)
        month = pd.Series(daily.index.to_period("M"), index=daily.index)
        transitions = month.ne(month.shift(1))
        self.assertTrue(bool((~changed.iloc[1:] | transitions.iloc[1:]).all()))

    def test_full_pipeline_is_truncation_invariant(self) -> None:
        cutoff = "2004-12-31"
        prices = self.result.prices.loc[:cutoff]
        truncated = run_backtest(
            self.config,
            prices=prices,
            unadjusted=load_unadjusted_prices(DATA_DIR).loc[:cutoff],
            metadata=self.result.metadata,
            volumes=load_volumes(DATA_DIR).loc[:cutoff],
            fx_rates=load_fx_rates(DATA_DIR, prices.index),
        )
        pd.testing.assert_series_equal(
            self.result.daily.loc[:cutoff, "net_return"],
            truncated.daily["net_return"],
        )
        pd.testing.assert_frame_equal(
            self.result.positions.loc[:cutoff], truncated.positions
        )
        pd.testing.assert_frame_equal(
            self.result.signals.loc[:cutoff], truncated.signals
        )

    def test_optimized_window_clears_both_targets(self) -> None:
        row = self.report.set_index("Window").loc["1990-2004 optimized window"]
        self.assertAlmostEqual(float(row["Sharpe (rf=0)"]), 2.1088424570, places=9)
        self.assertAlmostEqual(
            float(row["Monthly Sharpe (rf=0)"]), 1.9928462420, places=9
        )
        self.assertAlmostEqual(
            float(row["HAC Sharpe (21 lags, rf=0)"]), 1.9317675887, places=9
        )
        self.assertAlmostEqual(float(row["CAGR"]), 0.2610900536, places=9)
        self.assertAlmostEqual(float(row["Max drawdown"]), -0.1142467407, places=9)
        self.assertTrue(bool(row["Both targets met"]))

    def test_full_span_metrics_are_recorded(self) -> None:
        row = self.report.set_index("Window").loc["1980-2014 full history"]
        self.assertAlmostEqual(float(row["Sharpe (rf=0)"]), 1.8706342469, places=9)
        self.assertAlmostEqual(float(row["CAGR"]), 0.2320235237, places=9)
        self.assertAlmostEqual(float(row["Max drawdown"]), -0.1575740620, places=9)

    def test_target_claim_is_limited_to_selected_periods(self) -> None:
        report = self.report.set_index("Window")
        self.assertTrue(bool(report.loc["1990-1997 discovery", "Both targets met"]))
        self.assertTrue(bool(report.loc["1998-2004 confirmation", "Both targets met"]))
        reporting = report.loc["2005-2014 later stress"]
        self.assertFalse(bool(reporting["CAGR >= 20%"]))
        self.assertFalse(bool(reporting["Sharpe >= 2.0"]))


if __name__ == "__main__":
    unittest.main()
