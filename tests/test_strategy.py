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
    PRIOR_TARGET_SEARCH_TRIALS,
    StrategyConfig,
    _simulate_execution,
    apply_no_trade_buffer,
    basis_momentum,
    blend_signals,
    load_delivery_months,
    load_fx_rates,
    load_observed_prices,
    load_unadjusted_prices,
    load_volumes,
    monthly_block_bootstrap_intervals,
    performance_metrics,
    risk_managed_forecast,
    roll_event_mask,
    run_backtest,
    run_pipeline,
    tradeable_mask,
    trend_signal,
    usd_margin_values,
    usd_point_values,
)


DATA_DIR = Path(
    os.environ.get(
        "DELTA1_DATA_DIR",
        "Round1AllData/Quant Researcher/Delta1",
    )
)
ROOT_DIR = Path(__file__).resolve().parents[1]


def execution_inputs(
    closes: list[float],
    opens: list[float],
    volumes: list[float],
    *,
    valuation_closes: list[float] | None = None,
    target: float = 0.01,
    target_date: int = 0,
    target_schedule: dict[int, float] | None = None,
    roll_dates: tuple[int, ...] = (),
    one_way_cost: float = 0.0,
    initial_capital: float = 100.0,
    integer_contracts: bool = False,
    max_participation: float | None = None,
    execution_timing: str = "next_open",
    charge_roll_costs: bool = True,
    launch_date: str | None = None,
) -> tuple[object, pd.DatetimeIndex]:
    index = pd.bdate_range("2020-01-30", periods=len(closes))
    columns = ["X"]
    frame = lambda values: pd.DataFrame({"X": values}, index=index)
    schedule = target_schedule or {target_date: target}
    targets = pd.DataFrame(
        {"X": list(schedule.values())},
        index=[index[offset] for offset in schedule],
    )
    roll = pd.DataFrame(False, index=index, columns=columns)
    for offset in roll_dates:
        roll.iloc[offset, 0] = True
    config = StrategyConfig(
        Path("unused"),
        initial_capital=initial_capital,
        integer_contracts=integer_contracts,
        max_rebalance_participation=max_participation,
        volume_gate_window=2,
        execution_timing=execution_timing,
        charge_roll_costs=charge_roll_costs,
    )
    ledger = _simulate_execution(
        targets,
        frame(closes),
        frame(valuation_closes if valuation_closes is not None else closes),
        frame(opens),
        frame(closes),
        frame(volumes),
        frame([1.0] * len(index)),
        frame([one_way_cost] * len(index)),
        frame([10.0] * len(index)),
        roll,
        config,
        initial_capital=initial_capital,
        integer_contracts=integer_contracts,
        charge_costs=True,
        launch_date=launch_date,
    )
    return ledger, index


class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.base = StrategyConfig(Path("unused"))

    def test_defaults_are_the_audited_specification(self) -> None:
        self.assertEqual(self.base.trend_lookback, 252)
        self.assertEqual(self.base.basis_weight, 0.5)
        self.assertEqual(self.base.target_vol, 0.10)
        self.assertEqual(self.base.max_risk_scalar, 2.0)
        self.assertEqual(self.base.risk_budget, "flat")
        self.assertEqual(self.base.risk_managed_window, 63)
        self.assertEqual(self.base.no_trade_buffer, 0.25)
        self.assertEqual(self.base.initial_capital, 1_000_000.0)
        self.assertEqual(self.base.launch_date, "1990-01-01")
        self.assertTrue(self.base.integer_contracts)
        self.assertEqual(self.base.execution_timing, "next_close")
        self.assertTrue(self.base.charge_roll_costs)
        self.assertEqual(self.base.max_rebalance_participation, 0.05)
        self.assertEqual(CURRENT_OPTIMIZATION_TRIALS, 50)
        self.assertEqual(PRIOR_TARGET_SEARCH_TRIALS, 72)
        self.base.validate()

    def test_invalid_parameters_fail_fast(self) -> None:
        invalid = [
            {"basis_weight": 1.1},
            {"shock_start": 2.0, "shock_full": 1.0},
            {"no_trade_buffer": 1.0},
            {"risk_budget": "optimized"},
            {"risk_managed_window": 1},
            {"initial_capital": 0.0},
            {"launch_date": "not-a-date"},
            {"execution_timing": "same_close"},
            {"max_rebalance_participation": 0.0},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                StrategyConfig(Path("unused"), **values).validate()

    def test_superseded_point_metric_ledgers_are_not_shipped(self) -> None:
        self.assertFalse((ROOT_DIR / "outputs" / "optimization_trials.csv").exists())
        self.assertFalse((ROOT_DIR / "outputs" / "strategy_robustness.csv").exists())


class TestNotebookArtifact(unittest.TestCase):
    def test_notebook_is_executed_and_imports_the_canonical_strategy(self) -> None:
        notebook = json.loads((ROOT_DIR / "DELTA1_Strategy.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 15)
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
        pd.testing.assert_frame_equal(baseline, trend_signal(self.prices - 500))
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

    def test_blend_and_risk_management_are_bounded_and_causal(self) -> None:
        trend = trend_signal(self.prices)
        basis = basis_momentum(self.prices, self.unadjusted)
        blended = blend_signals(trend, basis, 0.5)
        opens = self.prices.shift(1).add(0.1)
        baseline = risk_managed_forecast(blended, self.prices, 60, 126, 2.0, opens)
        self.assertLessEqual(float(baseline.abs().max().max()), 1.0)
        changed = self.prices.copy()
        changed.iloc[1_300:] += 100
        changed_signal = blend_signals(
            trend_signal(changed), basis_momentum(changed, self.unadjusted), 0.5
        )
        altered = risk_managed_forecast(
            changed_signal, changed, 60, 126, 2.0, opens
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

    def test_next_open_fill_does_not_receive_the_overnight_gap(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 120.0, 130.0], [100.0, 110.0, 125.0], [100.0] * 3
        )
        self.assertEqual(ledger.positions.loc[index[0], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[1], "X"], 1.0)
        self.assertEqual(ledger.daily.loc[index[1], "gross_pnl_usd"], 10.0)
        self.assertEqual(ledger.daily.loc[index[2], "gross_pnl_usd"], 10.0)

    def test_zero_volume_session_defers_the_order(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 120.0, 130.0], [100.0, 110.0, 125.0], [100.0, 0.0, 100.0]
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 1.0)
        self.assertEqual(ledger.daily.loc[index[2], "gross_pnl_usd"], 5.0)

    def test_unchanged_position_pays_two_roll_legs(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0] * 3,
            roll_dates=(2,),
            one_way_cost=3.0,
        )
        self.assertEqual(ledger.daily.loc[index[1], "transaction_cost_usd"], 3.0)
        self.assertEqual(ledger.daily.loc[index[2], "transaction_cost_usd"], 6.0)
        self.assertEqual(
            ledger.daily.loc[index[2], "roll_contract_turnover_increment"], 2.0
        )

    def test_roll_on_closed_row_is_charged_on_next_session(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 0.0, 100.0],
            roll_dates=(2,),
            one_way_cost=3.0,
        )
        self.assertEqual(ledger.daily.loc[index[2], "transaction_cost_usd"], 0.0)
        self.assertEqual(ledger.daily.loc[index[3], "transaction_cost_usd"], 6.0)
        self.assertEqual(
            ledger.daily.loc[index[3], "roll_contract_turnover_increment"], 2.0
        )

    def test_roll_label_while_flat_is_not_an_executed_roll(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            roll_dates=(0,),
            one_way_cost=3.0,
        )
        self.assertFalse(bool(ledger.executed_rolls.loc[index[0], "X"]))
        self.assertEqual(ledger.daily.loc[index[0], "total_contract_turnover"], 0.0)

    def test_roll_cost_switch_preserves_physical_roll_diagnostic(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0] * 3,
            roll_dates=(2,),
            one_way_cost=3.0,
            charge_roll_costs=False,
        )
        self.assertTrue(bool(ledger.executed_rolls.loc[index[2], "X"]))
        self.assertEqual(ledger.daily.loc[index[2], "transaction_cost_usd"], 0.0)
        self.assertEqual(ledger.daily.loc[index[2], "total_contract_turnover"], 0.0)
        self.assertEqual(
            ledger.daily.loc[index[2], "roll_contract_turnover_increment"], 2.0
        )

    def test_contracts_and_nav_are_stateful_and_self_financing(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 110.0, 121.0], [100.0, 100.0, 110.0], [100.0] * 3
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 1.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 1.0)
        self.assertAlmostEqual(ledger.daily.loc[index[1], "nav"], 110.0)
        self.assertAlmostEqual(ledger.daily.loc[index[2], "nav"], 121.0)
        np.testing.assert_allclose(
            ledger.daily["equity"], ledger.daily["nav"] / 100.0
        )

    def test_integer_rounding_depends_on_capital(self) -> None:
        low, low_index = execution_inputs(
            [100.0, 100.0], [100.0, 100.0], [100.0] * 2,
            target=0.006, initial_capital=50.0, integer_contracts=True,
        )
        high, high_index = execution_inputs(
            [100.0, 100.0], [100.0, 100.0], [100.0] * 2,
            target=0.006, initial_capital=100.0, integer_contracts=True,
        )
        self.assertEqual(low.positions.loc[low_index[1], "X"], 0.0)
        self.assertEqual(high.positions.loc[high_index[1], "X"], 1.0)

    def test_participation_cap_carries_a_partial_order(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 5,
            [100.0] * 5,
            [10.0] * 5,
            target=0.03,
            target_date=1,
            initial_capital=100.0,
            integer_contracts=True,
            max_participation=0.10,
        )
        self.assertEqual(ledger.positions.loc[index[2], "X"], 1.0)
        self.assertEqual(ledger.positions.loc[index[3], "X"], 2.0)
        self.assertEqual(ledger.positions.loc[index[4], "X"], 3.0)
        self.assertGreater(ledger.daily.loc[index[2], "pending_markets"], 0)
        self.assertEqual(ledger.daily.loc[index[4], "pending_markets"], 0)

    def test_partial_order_quantity_is_frozen_at_first_fill(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0, 200.0, 300.0, 400.0, 500.0],
            [100.0, 100.0, 100.0, 200.0, 300.0, 400.0],
            [10.0] * 6,
            target=0.03,
            target_date=1,
            initial_capital=100.0,
            integer_contracts=True,
            max_participation=0.10,
        )
        self.assertEqual(ledger.positions.loc[index[4], "X"], 3.0)
        self.assertEqual(ledger.positions.loc[index[5], "X"], 3.0)
        self.assertEqual(ledger.daily.loc[index[4], "pending_markets"], 0)

    def test_order_quantity_is_frozen_at_decision_nav(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0, 200.0, 200.0],
            [100.0, 100.0, 100.0, 200.0],
            [100.0] * 4,
            target_schedule={0: 0.01, 1: 0.02},
            execution_timing="next_close",
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 1.0)
        self.assertEqual(ledger.daily.loc[index[2], "nav"], 200.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 2.0)

    def test_midmonth_launch_queues_latest_prior_decision(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 4,
            [100.0] * 4,
            [100.0] * 4,
            launch_date="2020-02-03",
        )
        self.assertEqual(index[2], pd.Timestamp("2020-02-03"))
        self.assertEqual(ledger.positions.loc[index[1], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 1.0)

    def test_missing_held_settlement_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing held settlement"):
            execution_inputs(
                [100.0, 100.0, np.nan],
                [100.0, 100.0, np.nan],
                [100.0] * 3,
            )

    def test_every_vendor_label_change_is_charged_and_anomalies_are_flagged(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=7)
        delivery = pd.DataFrame(
            {"X": [202003, 202003, 202006, 202003, 202006, 202005, 202009]},
            index=index,
        )
        rolls, anomalies = roll_event_mask(delivery)
        self.assertTrue(bool(rolls.iloc[2, 0]))
        self.assertTrue(bool(rolls.iloc[3, 0]))
        self.assertTrue(bool(anomalies.iloc[2, 0]))
        self.assertTrue(bool(rolls.iloc[4, 0]))
        self.assertTrue(bool(rolls.iloc[5, 0]))
        self.assertTrue(bool(anomalies.iloc[5, 0]))
        self.assertTrue(bool(rolls.iloc[6, 0]))

    def test_notional_uses_raw_active_contract_level_not_adjusted_level(self) -> None:
        low, low_index = execution_inputs(
            [10.0, 11.0, 12.0],
            [10.0, 10.0, 11.0],
            [100.0] * 3,
            valuation_closes=[100.0, 101.0, 102.0],
        )
        high, high_index = execution_inputs(
            [10.0, 11.0, 12.0],
            [10.0, 10.0, 11.0],
            [100.0] * 3,
            valuation_closes=[200.0, 201.0, 202.0],
        )
        self.assertEqual(
            low.daily.loc[low_index[2], "gross_pnl_usd"],
            high.daily.loc[high_index[2], "gross_pnl_usd"],
        )
        self.assertEqual(low.daily.loc[low_index[2], "gross_notional_usd"], 102.0)
        self.assertEqual(high.daily.loc[high_index[2], "gross_notional_usd"], 202.0)

    def test_point_values_and_static_margin_convert_foreign_currency(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=2)
        metadata = pd.DataFrame(
            {
                "currency": ["USD", "EUR"],
                "point_value": [50.0, 10.0],
                "margin": [1_000.0, 2_000.0],
            },
            index=["US", "EU"],
        )
        fx = pd.DataFrame({"USD": [1.0, 1.0], "EUR": [1.1, 1.2]}, index=index)
        values = usd_point_values(metadata, fx, index)
        margins = usd_margin_values(metadata, fx, index)
        self.assertEqual(values["EU"].tolist(), [11.0, 12.0])
        self.assertEqual(margins["EU"].tolist(), [2_200.0, 2_400.0])


class TestMetrics(unittest.TestCase):
    @staticmethod
    def result(returns: list[float], index: pd.DatetimeIndex) -> BacktestResult:
        daily = pd.DataFrame(
            {
                "net_return": returns,
                "cost": 0.0,
                "risk_scalar": 1.0,
                "gross_notional_multiple": 0.0,
                "static_margin_fraction": 0.0,
                "max_order_participation": 0.0,
            },
            index=index,
        )
        empty = pd.DataFrame(index=index)
        return BacktestResult(
            "test", daily, empty, empty, empty, empty, empty, empty, empty, empty
        )

    def test_cagr_includes_the_first_return_interval(self) -> None:
        index = pd.to_datetime(["2019-12-31", "2020-01-01", "2021-01-01"])
        result = self.result([0.0, 0.0, 0.10], index)
        metrics = performance_metrics(result, "2020-01-01", "2021-01-01")
        expected_years = 367 / 365.2425
        self.assertAlmostEqual(float(metrics["Years"]), expected_years)
        self.assertAlmostEqual(
            float(metrics["CAGR"]), 1.10 ** (1 / expected_years) - 1
        )

    def test_sortino_uses_downside_root_mean_square(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=4)
        returns = [0.02, -0.01, 0.01, -0.02]
        metrics = performance_metrics(self.result(returns, index), str(index[0].date()))
        downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
        expected = np.mean(returns) * 252 / (downside * np.sqrt(252))
        self.assertAlmostEqual(float(metrics["Sortino (rf=0)"]), expected)

    def test_drawdown_includes_starting_wealth(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=2)
        metrics = performance_metrics(self.result([-0.10, 0.0], index), "2020")
        self.assertAlmostEqual(float(metrics["Max drawdown"]), -0.10)

    def test_block_bootstrap_is_deterministic_and_unadjusted(self) -> None:
        index = pd.bdate_range("2018-01-01", periods=756)
        returns = (0.0003 + np.sin(np.arange(len(index))) * 0.001).tolist()
        result = self.result(returns, index)
        first = monthly_block_bootstrap_intervals(result, "2018", samples=500)
        second = monthly_block_bootstrap_intervals(result, "2018", samples=500)
        pd.testing.assert_series_equal(first, second)
        self.assertFalse(bool(first["Selection adjusted"]))


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
        prelaunch = daily.index < pd.Timestamp(self.config.launch_date)
        self.assertTrue((self.result.positions.loc[prelaunch] == 0).all().all())
        self.assertTrue(
            (daily.loc[prelaunch, "nav"] == self.config.initial_capital).all()
        )
        self.assertTrue(np.isfinite(daily["net_return"]).all())
        self.assertTrue((daily["cost"] >= 0).all())
        self.assertTrue((daily["nav"] > 0).all())
        self.assertLessEqual(float(self.result.signals.abs().max().max()), 1.0)
        self.assertTrue(
            np.allclose(
                self.result.positions.to_numpy(),
                np.rint(self.result.positions.to_numpy()),
            )
        )
        np.testing.assert_allclose(
            daily["equity"], daily["nav"] / self.config.initial_capital
        )
        changed = self.result.trades.ne(0)
        volumes = load_volumes(DATA_DIR).reindex_like(self.result.positions)
        self.assertTrue(bool((~changed | volumes.gt(0)).all().all()))
        self.assertGreater(float(daily["roll_contract_turnover_increment"].sum()), 0)
        self.assertEqual(int(daily["delivery_label_anomalies"].sum()), 4)
        self.assertGreater(int(self.result.executed_rolls.to_numpy().sum()), 0)

        prior_positions = self.result.positions.shift().fillna(0.0)
        np.testing.assert_allclose(
            self.result.trades,
            self.result.positions - prior_positions,
        )
        charged_by_market = self.result.trades.abs().where(
            ~self.result.executed_rolls,
            prior_positions.abs() + self.result.positions.abs(),
        )
        np.testing.assert_allclose(
            daily["total_contract_turnover"],
            charged_by_market.sum(axis=1),
        )
        np.testing.assert_allclose(
            daily["roll_contract_turnover_increment"],
            (charged_by_market - self.result.trades.abs()).sum(axis=1),
        )

    def test_full_pipeline_is_truncation_invariant(self) -> None:
        cutoff = "2004-12-31"
        prices = self.result.prices.loc[:cutoff]
        truncated = run_backtest(
            self.config,
            prices=prices,
            observed_opens=load_observed_prices(DATA_DIR, "Open").loc[:cutoff],
            observed_closes=load_observed_prices(DATA_DIR, "Close").loc[:cutoff],
            unadjusted=load_unadjusted_prices(DATA_DIR).loc[:cutoff],
            metadata=self.result.metadata,
            volumes=load_volumes(DATA_DIR).loc[:cutoff],
            delivery_months=load_delivery_months(DATA_DIR).loc[:cutoff],
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

    def test_selected_window_fails_the_sharpe_target_and_validation(self) -> None:
        row = self.report.set_index("Window").loc["1990-2004 selected window"]
        self.assertAlmostEqual(
            float(row["Naive daily Sharpe (sqrt252, rf=0)"]), 1.984864, places=5
        )
        self.assertAlmostEqual(float(row["CAGR"]), 0.245232, places=5)
        self.assertTrue(bool(row["CAGR point estimate >= 20%"]))
        self.assertFalse(bool(row["Naive daily Sharpe point estimate >= 2.0"]))
        self.assertFalse(bool(row["Both point-estimate targets met"]))
        self.assertFalse(bool(row["Monthly Sharpe >= 2.0"]))
        self.assertFalse(bool(row["HAC Sharpe >= 2.0"]))
        self.assertFalse(bool(row["Peak modeled daily participation <= 100%"]))
        self.assertFalse(bool(row["Externally validated"]))
        self.assertFalse(
            bool(row["Durable 20% CAGR / 2.0 Sharpe claim validated"])
        )

    def test_reused_later_period_does_not_clear_targets(self) -> None:
        row = self.report.set_index("Window").loc[
            "2005-2014 reused later diagnostic"
        ]
        self.assertFalse(bool(row["CAGR point estimate >= 20%"]))
        self.assertFalse(bool(row["Naive daily Sharpe point estimate >= 2.0"]))


if __name__ == "__main__":
    unittest.main()
