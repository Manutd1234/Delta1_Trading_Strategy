from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.controls.production import (
    overall_readiness_status,
    production_readiness_report,
)
from delta1_strategy.research.strategy import (
    BacktestResult,
    ENGINE_VERSION,
    STRATEGY_NAME,
    STRATEGY_DESIGNATION,
    STRATEGY_TECHNICAL_NAME,
    StrategyConfig,
    _base_target_positions,
    _limit_target_contracts,
    _load_column,
    _month_end_rows,
    _simulate_execution,
    apply_no_trade_buffer,
    basis_momentum,
    blend_signals,
    build_trade_episodes,
    load_delivery_months,
    load_fx_rates,
    load_observed_prices,
    load_prices,
    load_unadjusted_prices,
    load_volumes,
    performance_metrics,
    risk_managed_forecast,
    roll_event_mask,
    run_backtest,
    run_pipeline,
    save_outputs,
    strategy_symbols,
    trade_episode_metrics,
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

REQUIRED_DAILY_COLUMNS = {
    "prior_nav_usd",
    "gross_pnl_usd",
    "transaction_cost_usd",
    "fixed_execution_cost_usd",
    "market_impact_cost_usd",
    "net_pnl_usd",
    "gross_return",
    "cost",
    "net_return",
    "nav",
    "equity",
    "rebalance_contract_turnover",
    "roll_contract_turnover_increment",
    "total_contract_turnover",
    "gross_notional_usd",
    "gross_notional_multiple",
    "static_margin_requirement_usd",
    "static_margin_fraction",
    "max_order_participation",
    "max_rebalance_participation",
    "max_roll_participation_proxy",
    "filled_markets",
    "pending_markets",
    "target_portfolio_limit_scale",
}

EMPTY_EPISODE_COLUMNS = [
    "Status",
    "Entry date",
    "Exit date",
    "Net contribution",
    "Net P&L USD",
    "Holding sessions",
]


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
    impact_bps: float = 0.0,
    point_value: float = 1.0,
    margin_per_contract: float = 10.0,
    initial_capital: float = 100.0,
    integer_contracts: bool = False,
    max_participation: float | None = None,
    max_gross_notional_multiple: float | None = 5.0,
    max_static_margin_fraction: float | None = None,
    execution_timing: str = "next_open",
    execution_delay_sessions: int = 0,
    charge_roll_costs: bool = True,
    launch_date: str | None = None,
    cap_fills_at_realized_capacity: bool = True,
    spread_roll_turnover: bool = True,
    max_roll_backlog_sessions: int | None = 5,
    roll_slice_impact_reference: str = "parent",
) -> tuple[object, pd.DatetimeIndex]:
    index = pd.bdate_range("2020-01-30", periods=len(closes))
    frame = lambda values: pd.DataFrame({"X": values}, index=index)
    schedule = target_schedule if target_schedule is not None else {target_date: target}
    targets = pd.DataFrame(
        {"X": list(schedule.values())},
        index=[index[offset] for offset in schedule],
    )
    roll = pd.DataFrame(False, index=index, columns=["X"])
    for offset in roll_dates:
        roll.iloc[offset, 0] = True
    config = StrategyConfig(
        Path("unused"),
        initial_capital=initial_capital,
        integer_contracts=integer_contracts,
        max_rebalance_participation=max_participation,
        max_gross_notional_multiple=max_gross_notional_multiple,
        max_static_margin_fraction=max_static_margin_fraction,
        volume_gate_window=2,
        execution_timing=execution_timing,
        execution_delay_sessions=execution_delay_sessions,
        charge_roll_costs=charge_roll_costs,
        impact_bps_at_full_participation=impact_bps,
        cap_fills_at_realized_capacity=cap_fills_at_realized_capacity,
        spread_roll_turnover=spread_roll_turnover,
        max_roll_backlog_sessions=max_roll_backlog_sessions,
        roll_slice_impact_reference=roll_slice_impact_reference,
    )
    ledger = _simulate_execution(
        targets,
        frame(closes),
        frame(valuation_closes if valuation_closes is not None else closes),
        frame(opens),
        frame(closes),
        frame(volumes),
        frame([point_value] * len(index)),
        frame([one_way_cost] * len(index)),
        frame([margin_per_contract] * len(index)),
        roll,
        config,
        initial_capital=initial_capital,
        integer_contracts=integer_contracts,
        charge_costs=True,
        launch_date=launch_date,
    )
    return ledger, index


def metric_result(returns: list[float], index: pd.DatetimeIndex) -> BacktestResult:
    values = np.asarray(returns, dtype=float)
    nav = 100.0 * np.cumprod(1.0 + values)
    prior_nav = np.r_[100.0, nav[:-1]]
    daily = pd.DataFrame(
        {
            "prior_nav_usd": prior_nav,
            "net_return": values,
            "cost": 0.0,
            "fixed_execution_cost_usd": 0.0,
            "market_impact_cost_usd": 0.0,
            "nav": nav,
            "risk_scalar": 1.0,
            "gross_notional_multiple": 0.0,
            "static_margin_fraction": 0.0,
            "max_order_participation": 0.0,
            "max_rebalance_participation": 0.0,
            "max_roll_participation_proxy": 0.0,
            "pending_markets": 0,
            "target_portfolio_limit_scale": 1.0,
        },
        index=index,
    )
    zeros = pd.DataFrame({"ES": np.zeros(len(index))}, index=index)
    falses = zeros.astype(bool)
    return BacktestResult(
        name="test",
        daily=daily,
        positions=zeros.copy(),
        target_positions=zeros.copy(),
        signals=zeros.copy(),
        prices=zeros.copy(),
        metadata=pd.DataFrame(),
        trades=zeros.copy(),
        roll_events=falses.copy(),
        executed_rolls=falses.copy(),
        gross_pnl_by_market=zeros.copy(),
        regular_cost_by_market=zeros.copy(),
        roll_cost_by_market=zeros.copy(),
        trade_episodes=pd.DataFrame(columns=EMPTY_EPISODE_COLUMNS),
        execution_timing="next_close",
    )


def episode_result() -> BacktestResult:
    index = pd.bdate_range("2024-01-02", periods=6)
    positions = pd.DataFrame({"ES": [1.0, 2.0, -1.0, -1.0, 0.0, 1.0]}, index=index)
    prior = positions.shift().fillna(0.0)
    trades = positions - prior
    gross = pd.DataFrame({"ES": [0.0, 10.0, -4.0, 3.0, 2.0, 0.0]}, index=index)
    regular = pd.DataFrame({"ES": [1.0, 1.0, 3.0, 0.0, 1.0, 1.0]}, index=index)
    roll_cost = pd.DataFrame({"ES": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]}, index=index)
    executed_rolls = pd.DataFrame(
        {"ES": [False, False, False, True, False, False]}, index=index
    )
    daily = pd.DataFrame({"prior_nav_usd": 100.0}, index=index)
    return BacktestResult(
        name="episode test",
        daily=daily,
        positions=positions,
        target_positions=positions.copy(),
        signals=positions.copy(),
        prices=pd.DataFrame({"ES": 100.0}, index=index),
        metadata=pd.DataFrame(),
        trades=trades,
        roll_events=executed_rolls.copy(),
        executed_rolls=executed_rolls,
        gross_pnl_by_market=gross,
        regular_cost_by_market=regular,
        roll_cost_by_market=roll_cost,
        trade_episodes=pd.DataFrame(columns=EMPTY_EPISODE_COLUMNS),
        execution_timing="next_close",
    )


class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.base = StrategyConfig(Path("unused"))

    def test_defaults_are_current_research_and_execution_controls(self) -> None:
        self.assertEqual(self.base.trend_lookback, 252)
        self.assertEqual(self.base.basis_weight, 0.5)
        self.assertEqual(self.base.target_vol, 0.07)
        self.assertEqual(self.base.risk_budget, "flat")
        self.assertEqual(self.base.initial_capital, 1_000_000.0)
        self.assertEqual(self.base.launch_date, "1990-01-01")
        self.assertTrue(self.base.integer_contracts)
        self.assertEqual(self.base.execution_timing, "next_close")
        self.assertEqual(self.base.execution_delay_sessions, 0)
        self.assertTrue(self.base.charge_roll_costs)
        self.assertEqual(self.base.max_rebalance_participation, 0.02)
        self.assertEqual(self.base.max_gross_notional_multiple, 5.0)
        self.assertIsNone(self.base.max_static_margin_fraction)
        self.assertFalse(hasattr(self.base, "mode"))
        self.base.validate()

    def test_strategy_identity_separates_methodology_and_best_available_label(
        self,
    ) -> None:
        self.assertEqual(ENGINE_VERSION, "3.2.1")
        self.assertEqual(STRATEGY_DESIGNATION, "Best Available Hardened Strategy")
        self.assertEqual(
            STRATEGY_TECHNICAL_NAME,
            "Diversified Global-Futures Time-Series Momentum Plus "
            "Basis-Momentum Portfolio",
        )
        self.assertEqual(STRATEGY_NAME, STRATEGY_TECHNICAL_NAME)
        self.assertNotEqual(STRATEGY_DESIGNATION, STRATEGY_NAME)

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
            {"execution_delay_sessions": -1},
            {"max_rebalance_participation": 0.0},
            {"max_gross_notional_multiple": 0.0},
            {"max_static_margin_fraction": 1.1},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                StrategyConfig(Path("unused"), **values).validate()

    def test_research_config_has_no_live_mode_switch(self) -> None:
        with self.assertRaises(TypeError):
            StrategyConfig(Path("unused"), mode="production")

    def test_research_universe_excludes_unsupported_yield_quoted_contracts(self) -> None:
        symbols = strategy_symbols()
        self.assertEqual(len(symbols), 59)
        self.assertTrue({"YXT", "YYT"}.isdisjoint(symbols))

    def test_asset_class_budget_uses_only_supported_frame_columns(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=120)
        rng = np.random.default_rng(7)
        prices = pd.DataFrame(
            {
                "ES": 100.0 + np.cumsum(rng.normal(0.0, 1.0, len(index))),
                "ZN": 120.0 + np.cumsum(rng.normal(0.0, 0.5, len(index))),
            },
            index=index,
        )
        forecast = pd.DataFrame(1.0, index=index, columns=prices.columns)
        point_values = pd.DataFrame(1.0, index=index, columns=prices.columns)
        positions = _base_target_positions(
            prices,
            forecast,
            point_values,
            StrategyConfig(Path("unused"), risk_budget="asset_classes", vol_span=20),
        )
        self.assertEqual(positions.columns.tolist(), ["ES", "ZN"])
        self.assertTrue(np.isfinite(positions.iloc[-1]).all())


class TestNotebookArtifact(unittest.TestCase):
    """The one notebook the repository ships must be executed and honest.

    It replaces an identical guard on a second, committee-facing notebook that
    was removed on 2026-08-13. The assertions are the same in kind: the
    notebook must read the canonical artifacts rather than recompute them, must
    have been executed rather than shipped as source, must carry its charts
    inline so a reviewer with no kernel still sees them, and must not have
    printed a warning or raised on the way.
    """

    def test_notebook_is_executed_uses_canonical_artifacts_and_embeds_charts(self) -> None:
        notebook = json.loads(
            (ROOT_DIR / "notebooks" / "delta1_case_research.ipynb").read_text()
        )
        self.assertEqual(notebook["nbformat"], 4)
        self.assertIsInstance(notebook.get("cells"), list)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        source = "".join("".join(cell.get("source", [])) for cell in code_cells)
        all_source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        # Runs the readable reference file rather than the hardened package,
        # and takes every figure it does not compute from the canonical
        # artifacts rather than from a number typed into a cell.
        self.assertIn("delta1_reference", source)
        for artifact in (
            "outputs/universe/universe_audit.csv",
            "outputs/validation/validation_walk_forward_summary.csv",
            "outputs/benchmarks/benchmark_comparison.csv",
            "outputs/optimality/optimality_summary.csv",
        ):
            self.assertIn(artifact, source, f"notebook does not read {artifact}")
        # The sections the case brief names, in the notebook a reviewer opens.
        for heading in (
            "Introduction",
            "Methodology",
            "Optimizations",
            "Findings",
            "Limitations",
            "Key takeaways",
        ):
            self.assertIn(heading, all_source, f"missing section: {heading}")
        self.assertTrue(
            all(
                isinstance(cell.get("execution_count"), int)
                for cell in code_cells
            )
        )
        outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
        self.assertFalse(any(output.get("output_type") == "error" for output in outputs))
        embedded_images = [
            output
            for output in outputs
            if "image/png" in output.get("data", {})
        ]
        self.assertGreaterEqual(len(embedded_images), 4)
        stream_text = "".join(
            "".join(output.get("text", []))
            for output in outputs
            if output.get("output_type") == "stream"
        ).lower()
        self.assertNotIn("warning", stream_text)


class TestInputValidation(unittest.TestCase):
    def write_csv(self, directory: str, rows: list[dict[str, object]]) -> Path:
        path = Path(directory) / "vendor.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_load_column_rejects_duplicate_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                [
                    {"Date": "2020-01-02", "Close": 100},
                    {"Date": "2020-01-02", "Close": 101},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate Date"):
                _load_column(path, "Close", "X")

    def test_load_column_rejects_bad_dates_and_non_numeric_values(self) -> None:
        cases = (
            (
                [
                    {"Date": "2020-01-03", "Close": 101},
                    {"Date": "not-a-date", "Close": 100},
                ],
                "Invalid Date",
            ),
            (
                [
                    {"Date": "2020-01-02", "Close": "broken"},
                    {"Date": "2020-01-03", "Close": 101},
                ],
                "Non-numeric Close",
            ),
        )
        for rows, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = self.write_csv(directory, rows)
                with self.assertRaisesRegex(ValueError, message):
                    _load_column(path, "Close", "X")

    def test_load_column_sorts_unambiguous_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                [
                    {"Date": "2020-01-03", "Close": 101},
                    {"Date": "2020-01-02", "Close": 100},
                ],
            )
            loaded = _load_column(path, "Close", "X")
        self.assertEqual(loaded.name, "X")
        self.assertTrue(loaded.index.is_monotonic_increasing)
        self.assertEqual(loaded.tolist(), [100, 101])

    def test_canonical_price_loader_rejects_positive_and_negative_infinity(
        self,
    ) -> None:
        for value in ("inf", "-inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                futures = Path(directory) / "Futures Data"
                futures.mkdir()
                pd.DataFrame(
                    [
                        {"Date": "2020-01-02", "Close": 100.0},
                        {"Date": "2020-01-03", "Close": value},
                    ]
                ).to_csv(futures / "&X_CCB.csv", index=False)
                with self.assertRaisesRegex(ValueError, "Non-finite Close"):
                    load_prices(Path(directory), symbols=["X"])

    def test_canonical_volume_loader_rejects_non_finite_and_negative_values(
        self,
    ) -> None:
        for value, message in (
            ("inf", "Non-finite Volume"),
            ("-inf", "Non-finite Volume"),
            (-1, "Negative Volume"),
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                futures = Path(directory) / "Futures Data"
                futures.mkdir()
                pd.DataFrame(
                    [
                        {"Date": "2020-01-02", "Volume": 1_000},
                        {"Date": "2020-01-03", "Volume": value},
                    ]
                ).to_csv(futures / "&X_CCB.csv", index=False)
                with self.assertRaisesRegex(ValueError, message):
                    load_volumes(Path(directory), symbols=["X"])


class TestMonthlyScheduling(unittest.TestCase):
    def test_terminal_mid_month_row_is_not_a_completed_month(self) -> None:
        index = pd.to_datetime(
            ["2024-01-29", "2024-02-29", "2024-03-01", "2024-03-15"]
        )
        frame = pd.DataFrame({"X": range(len(index))}, index=index)
        selected = _month_end_rows(frame)
        self.assertEqual(
            selected.index.tolist(),
            [pd.Timestamp("2024-01-29"), pd.Timestamp("2024-02-29")],
        )

    def test_terminal_standard_business_month_end_is_retained(self) -> None:
        index = pd.to_datetime(["2024-01-29", "2024-02-01", "2024-02-29"])
        series = pd.Series([1.0, 2.0, 3.0], index=index, name="X")
        selected = _month_end_rows(series)
        self.assertEqual(
            selected.index.tolist(),
            [pd.Timestamp("2024-01-29"), pd.Timestamp("2024-02-29")],
        )
        self.assertEqual(selected.name, "X")


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
        changed_prices = self.prices.copy()
        changed_prices.iloc[1_300:] += 100
        changed_signal = blend_signals(
            trend_signal(changed_prices),
            basis_momentum(changed_prices, self.unadjusted),
            0.5,
        )
        altered = risk_managed_forecast(
            changed_signal, changed_prices, 60, 126, 2.0, opens
        )
        pd.testing.assert_frame_equal(baseline.iloc[:1_300], altered.iloc[:1_300])


class TestTargetLimits(unittest.TestCase):
    def test_gross_notional_limit_scales_target_contracts(self) -> None:
        config = StrategyConfig(
            Path("unused"),
            max_gross_notional_multiple=1.0,
            max_static_margin_fraction=None,
        )
        contracts, scale = _limit_target_contracts(
            np.array([0.10, -0.10]),
            100.0,
            np.array([10.0, 10.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            config,
            integer_contracts=False,
        )
        np.testing.assert_allclose(contracts, [5.0, -5.0])
        self.assertAlmostEqual(scale, 0.5)
        gross = np.sum(np.abs(contracts * np.array([10.0, 10.0])))
        self.assertLessEqual(gross, config.max_gross_notional_multiple * 100.0)

    def test_static_margin_limit_scales_target_contracts(self) -> None:
        config = StrategyConfig(
            Path("unused"),
            max_gross_notional_multiple=None,
            max_static_margin_fraction=0.20,
        )
        contracts, scale = _limit_target_contracts(
            np.array([0.10, -0.10]),
            100.0,
            np.array([10.0, 10.0]),
            np.array([1.0, 1.0]),
            np.array([5.0, 5.0]),
            config,
            integer_contracts=False,
        )
        np.testing.assert_allclose(contracts, [2.0, -2.0])
        self.assertAlmostEqual(scale, 0.2)
        margin = np.sum(np.abs(contracts) * np.array([5.0, 5.0]))
        self.assertLessEqual(margin, config.max_static_margin_fraction * 100.0)

    def test_nonzero_target_requires_complete_valuation_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing decision-date valuation"):
            _limit_target_contracts(
                np.array([0.01]),
                100.0,
                np.array([np.nan]),
                np.array([1.0]),
                np.array([10.0]),
                StrategyConfig(Path("unused")),
                integer_contracts=True,
            )

    def test_post_rounding_rescale_truncates_back_under_the_gross_limit(self) -> None:
        """np.rint can recross the cap, so the rescue rescale must truncate.

        The pre-round scale leaves [5/3, 5/3] contracts; rint gives [2, 2] and
        120 gross against the 100 cap, and re-rounding the rescaled [5/3, 5/3]
        would give [2, 2] right back.  Only truncation terminates under the cap.
        """

        config = StrategyConfig(
            Path("unused"),
            max_gross_notional_multiple=1.0,
            max_static_margin_fraction=None,
        )
        contracts, _ = _limit_target_contracts(
            np.array([0.018, 0.018]),
            100.0,
            np.array([30.0, 30.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            config,
            integer_contracts=True,
        )
        np.testing.assert_allclose(contracts, [1.0, 1.0])
        np.testing.assert_allclose(contracts, np.rint(contracts))
        gross = float(np.sum(np.abs(contracts * 30.0)))
        self.assertLessEqual(gross, config.max_gross_notional_multiple * 100.0)

    def test_integer_targets_respect_hard_limits_across_random_books(self) -> None:
        """Integrality and both hard limits survive rounding for any book."""

        rng = np.random.default_rng(20260811)
        for draw in range(200):
            n_markets = int(rng.integers(1, 6))
            targets = rng.normal(0.0, 0.05, n_markets)
            targets[rng.random(n_markets) < 0.2] = 0.0
            nav = float(rng.uniform(50.0, 1_000_000.0))
            prices = rng.uniform(0.5, 500.0, n_markets) * rng.choice(
                [-1.0, 1.0], n_markets
            )
            point_values = rng.uniform(0.5, 100.0, n_markets)
            margins = rng.uniform(0.1, 5_000.0, n_markets)
            gross_cap = float(rng.choice([0.5, 1.0, 2.0, 5.0]))
            margin_cap = (
                float(rng.uniform(0.05, 0.5)) if rng.random() < 0.5 else None
            )
            config = StrategyConfig(
                Path("unused"),
                max_gross_notional_multiple=gross_cap,
                max_static_margin_fraction=margin_cap,
            )
            contracts, _ = _limit_target_contracts(
                targets,
                nav,
                prices,
                point_values,
                margins,
                config,
                integer_contracts=True,
            )
            label = f"draw {draw}"
            self.assertTrue(np.allclose(contracts, np.rint(contracts)), label)
            gross = float(np.sum(np.abs(contracts * prices * point_values)))
            self.assertLessEqual(gross, gross_cap * nav * (1 + 1e-12), label)
            if margin_cap is not None:
                margin = float(np.sum(np.abs(contracts) * margins))
                self.assertLessEqual(margin, margin_cap * nav * (1 + 1e-12), label)


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

    def test_buffer_is_idempotent_for_scalar_and_per_market_widths(self) -> None:
        """A buffered path re-buffered is itself, bit for bit.

        The output only moves when a move cleared the band, so feeding it back
        must reproduce it exactly; anything else means the band depends on
        state the output does not carry.
        """

        rng = np.random.default_rng(20260811)
        index = pd.date_range("2005-01-31", periods=48, freq="BME")
        columns = ["A", "B", "C", "D"]
        # Small drifts from the scaled random walk, plus NaN holes, exact
        # zeros, and sign flips — every hazard the band must absorb.
        values = 0.1 * rng.normal(size=(48, 4)).cumsum(axis=0)
        values[rng.random((48, 4)) < 0.15] = np.nan
        values[rng.random((48, 4)) < 0.10] = 0.0
        values = np.where(rng.random((48, 4)) < 0.10, -values, values)
        desired = pd.DataFrame(values, index=index, columns=columns)
        for width in (0.25, pd.Series([0.0, 0.1, 0.25, 0.9], index=columns)):
            once = apply_no_trade_buffer(desired, width)
            pd.testing.assert_frame_equal(
                apply_no_trade_buffer(once, width), once, check_exact=True
            )

    def test_next_open_fill_does_not_receive_the_overnight_gap(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 120.0, 130.0], [100.0, 110.0, 125.0], [100.0] * 3
        )
        self.assertEqual(ledger.positions.loc[index[0], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[1], "X"], 1.0)
        self.assertEqual(ledger.daily.loc[index[1], "gross_pnl_usd"], 10.0)
        self.assertEqual(ledger.daily.loc[index[2], "gross_pnl_usd"], 10.0)

    def test_next_close_fill_receives_returns_only_after_the_fill(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 120.0, 130.0],
            [100.0, 110.0, 125.0],
            [100.0] * 3,
            execution_timing="next_close",
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 1.0)
        self.assertEqual(ledger.daily.loc[index[1], "gross_pnl_usd"], 0.0)
        self.assertEqual(ledger.daily.loc[index[2], "gross_pnl_usd"], 10.0)

    def test_additional_execution_delay_waits_one_eligible_session(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 4,
            [100.0] * 4,
            [100.0] * 4,
            execution_timing="next_close",
            execution_delay_sessions=1,
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 1.0)

    def test_zero_volume_session_defers_the_order_and_delay_clock(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 5,
            [100.0] * 5,
            [100.0, 0.0, 100.0, 100.0, 100.0],
            execution_timing="next_close",
            execution_delay_sessions=1,
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 0.0)
        self.assertEqual(ledger.positions.loc[index[3], "X"], 1.0)

    def test_fixed_and_impact_costs_decompose_total_cost(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 3,
            [100.0] * 3,
            [100.0] * 3,
            one_way_cost=3.0,
            impact_bps=10.0,
            execution_timing="next_close",
        )
        fill_date = index[1]
        self.assertAlmostEqual(
            ledger.daily.loc[fill_date, "fixed_execution_cost_usd"], 3.0
        )
        self.assertAlmostEqual(
            ledger.daily.loc[fill_date, "market_impact_cost_usd"], 0.01
        )
        self.assertAlmostEqual(
            ledger.daily.loc[fill_date, "transaction_cost_usd"], 3.01
        )
        np.testing.assert_allclose(
            ledger.daily["transaction_cost_usd"],
            ledger.daily["fixed_execution_cost_usd"]
            + ledger.daily["market_impact_cost_usd"],
        )

    def test_rolls_charge_two_legs_and_defer_on_closed_sessions(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 4,
            [100.0] * 4,
            [100.0, 100.0, 0.0, 100.0],
            roll_dates=(2,),
            one_way_cost=3.0,
        )
        self.assertEqual(ledger.daily.loc[index[1], "transaction_cost_usd"], 3.0)
        self.assertEqual(ledger.daily.loc[index[2], "transaction_cost_usd"], 0.0)
        self.assertEqual(ledger.daily.loc[index[3], "transaction_cost_usd"], 6.0)
        self.assertEqual(
            ledger.daily.loc[index[3], "roll_contract_turnover_increment"], 2.0
        )
        self.assertTrue(bool(ledger.executed_rolls.loc[index[3], "X"]))

    def test_roll_while_flat_is_not_executed(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            target=0.0,
            roll_dates=(0,),
            one_way_cost=3.0,
        )
        self.assertFalse(bool(ledger.executed_rolls.loc[index[0], "X"]))
        self.assertEqual(ledger.daily.loc[index[0], "total_contract_turnover"], 0.0)

    def test_roll_cost_switch_preserves_physical_roll_diagnostic(self) -> None:
        ledger, index = execution_inputs(
            [100.0] * 3,
            [100.0] * 3,
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

    def test_roll_turnover_is_spread_but_the_total_is_preserved(self) -> None:
        """A thin session defers roll turnover; it never discards it.

        The position transfers to the new delivery whether or not the turnover
        is charged, so dropping a slice would hand out an uncosted transfer.
        Spreading must therefore move the charge in time and leave the total
        alone.
        """

        # The position must be established before the thin session, or the
        # capacity cap changes the position path and the two runs would be
        # rolling different books.
        common = dict(
            closes=[100.0] * 7,
            opens=[100.0] * 7,
            target=0.04,
            target_date=0,
            initial_capital=100.0,
            integer_contracts=True,
            roll_dates=(4,),
            one_way_cost=1.0,
            max_participation=0.05,
            execution_timing="next_close",
        )
        unspread, _ = execution_inputs(volumes=[500.0] * 7, **common)
        spread, index = execution_inputs(
            volumes=[500.0, 500.0, 500.0, 500.0, 40.0, 500.0, 500.0], **common
        )
        # Same book rolled in both runs.
        self.assertEqual(
            spread.positions["X"].tolist(), unspread.positions["X"].tolist()
        )
        self.assertAlmostEqual(
            float(spread.daily["roll_contract_turnover_increment"].sum()),
            float(unspread.daily["roll_contract_turnover_increment"].sum()),
            places=9,
        )
        # Bounded per session, spread over more than one, reported once.
        self.assertLessEqual(
            float(spread.daily["max_order_participation"].max()), 0.05 + 1e-12
        )
        self.assertGreater(
            float(spread.daily.loc[index[4], "roll_backlog_contracts"]), 0.0
        )
        self.assertEqual(int(spread.executed_rolls["X"].sum()), 1)

    def test_roll_slice_impact_cost_is_priced_against_the_parent(self) -> None:
        """Splitting a roll must not earn a square-root impact discount.

        Impact is proportional to turnover times sqrt(participation), which is
        superadditive, so per-slice pricing would make an order cheaper simply
        by being cut up.  This model prices neither permanent impact nor delay
        cost, so that discount would be an arbitrage against the cost model
        rather than a modelled saving.
        """

        common = dict(
            closes=[100.0] * 7,
            opens=[100.0] * 7,
            target=0.04,
            target_date=0,
            initial_capital=100.0,
            integer_contracts=True,
            roll_dates=(4,),
            impact_bps=50.0,
            max_participation=0.05,
            execution_timing="next_close",
        )
        # The control must be the same market with spreading disabled, not a
        # different volume series: the parent reference is defined against the
        # roll session's own depth.
        thin = [500.0, 500.0, 500.0, 500.0, 40.0, 500.0, 500.0]
        unspread, _ = execution_inputs(
            volumes=thin, spread_roll_turnover=False, **common
        )
        spread, _ = execution_inputs(volumes=thin, **common)
        discounted, _ = execution_inputs(
            volumes=thin, roll_slice_impact_reference="session", **common
        )
        total = lambda ledger: float(ledger.daily["transaction_cost_usd"].sum())
        self.assertAlmostEqual(total(spread), total(unspread), places=6)
        # The permissive convention is measurably cheaper; that gap is the
        # size of the artifact the parent reference removes.
        self.assertLess(total(discounted), total(spread))

    def test_reducing_a_position_extinguishes_the_roll_backlog(self) -> None:
        """Closing out of the expiring leg costs nothing extra, and is not free.

        A reduction is filled from the expiring contract, so the new leg was
        never bought for those contracts.  The ordinary trade already paid for
        the exit, so no incremental roll turnover may be charged and no
        obligation may survive.
        """

        ledger, index = execution_inputs(
            closes=[100.0] * 5,
            opens=[100.0] * 5,
            volumes=[500.0, 500.0, 20.0, 500.0, 500.0],
            target_schedule={0: 0.04, 2: 0.0},
            initial_capital=100.0,
            integer_contracts=True,
            roll_dates=(2,),
            one_way_cost=1.0,
            max_participation=0.05,
            execution_timing="next_close",
        )
        self.assertEqual(float(ledger.positions.iloc[-1, 0]), 0.0)
        self.assertEqual(float(ledger.daily["roll_backlog_contracts"].iloc[-1]), 0.0)
        self.assertTrue(bool((ledger.daily["roll_backlog_contracts"] >= 0).all()))

    def test_roll_backlog_never_exceeds_the_position(self) -> None:
        ledger, _ = execution_inputs(
            closes=[100.0] * 6,
            opens=[100.0] * 6,
            volumes=[500.0, 500.0, 30.0, 30.0, 500.0, 500.0],
            target=0.05,
            target_date=0,
            initial_capital=100.0,
            integer_contracts=True,
            roll_dates=(2,),
            max_participation=0.05,
            execution_timing="next_close",
        )
        backlog = ledger.daily["roll_backlog_contracts"]
        held = ledger.positions["X"].abs()
        self.assertTrue(bool((backlog <= held + 1e-12).all()))

    def test_an_unrollable_position_fails_closed(self) -> None:
        """Delivery is a hard deadline, so a stuck transfer must not pass silently."""

        with self.assertRaisesRegex(ValueError, "Delivery roll cannot be completed"):
            execution_inputs(
                closes=[100.0] * 10,
                opens=[100.0] * 10,
                volumes=[500.0] * 4 + [2.0] * 6,
                target=0.04,
                target_date=0,
                initial_capital=100.0,
                integer_contracts=True,
                roll_dates=(4,),
                max_participation=0.05,
                max_roll_backlog_sessions=3,
                execution_timing="next_close",
            )

    def test_roll_spreading_is_inert_without_a_participation_limit(self) -> None:
        """With no limit the backlog model reduces to the two-leg expression."""

        ledger, index = execution_inputs(
            closes=[100.0] * 4,
            opens=[100.0] * 4,
            volumes=[100.0, 100.0, 0.0, 100.0],
            roll_dates=(2,),
            one_way_cost=3.0,
            target=0.01,
            target_date=0,
        )
        self.assertEqual(ledger.daily.loc[index[3], "transaction_cost_usd"], 6.0)
        self.assertEqual(
            ledger.daily.loc[index[3], "roll_contract_turnover_increment"], 2.0
        )
        self.assertEqual(float(ledger.daily["roll_backlog_contracts"].max()), 0.0)

    def test_contracts_and_nav_are_stateful_and_self_financing(self) -> None:
        ledger, index = execution_inputs(
            [100.0, 110.0, 121.0], [100.0, 100.0, 110.0], [100.0] * 3
        )
        self.assertEqual(ledger.positions.loc[index[1], "X"], 1.0)
        self.assertEqual(ledger.positions.loc[index[2], "X"], 1.0)
        self.assertAlmostEqual(ledger.daily.loc[index[1], "nav"], 110.0)
        self.assertAlmostEqual(ledger.daily.loc[index[2], "nav"], 121.0)
        np.testing.assert_allclose(ledger.daily["equity"], ledger.daily["nav"] / 100.0)

    def test_ledger_and_per_market_attribution_reconcile(self) -> None:
        ledger, _ = execution_inputs(
            [100.0, 110.0, 115.0, 117.0],
            [100.0, 102.0, 111.0, 115.0],
            [100.0] * 4,
            target_schedule={0: 0.01, 2: 0.0},
            one_way_cost=1.25,
            impact_bps=5.0,
            execution_timing="next_close",
        )
        daily = ledger.daily
        self.assertTrue(REQUIRED_DAILY_COLUMNS.issubset(daily.columns))
        np.testing.assert_allclose(
            daily["gross_pnl_usd"], ledger.gross_pnl_by_market.sum(axis=1)
        )
        np.testing.assert_allclose(
            daily["transaction_cost_usd"],
            (ledger.regular_cost_by_market + ledger.roll_cost_by_market).sum(axis=1),
        )
        np.testing.assert_allclose(
            daily["net_pnl_usd"],
            daily["gross_pnl_usd"] - daily["transaction_cost_usd"],
        )
        np.testing.assert_allclose(
            daily["nav"],
            daily["prior_nav_usd"] + daily["net_pnl_usd"],
        )
        np.testing.assert_allclose(
            daily["net_return"], daily["net_pnl_usd"] / daily["prior_nav_usd"]
        )
        np.testing.assert_allclose(
            ledger.trades,
            ledger.positions - ledger.positions.shift().fillna(0.0),
        )

    def test_integer_rounding_depends_on_capital(self) -> None:
        low, low_index = execution_inputs(
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0] * 2,
            target=0.006,
            initial_capital=50.0,
            integer_contracts=True,
        )
        high, high_index = execution_inputs(
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0] * 2,
            target=0.006,
            initial_capital=100.0,
            integer_contracts=True,
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
        self.assertLessEqual(
            float(ledger.daily["max_rebalance_participation"].max()), 0.10
        )

    def test_execution_day_volume_caps_a_fill_but_never_enlarges_it(self) -> None:
        """Realized depth may truncate a fill; it may never resize the order.

        The distinction is the whole basis for conditioning a fill on the
        executing session's volume.  Sizing an order that way would be an
        optimistic look-ahead — more would be traded precisely when liquidity
        turned out well.  Truncating a fill is a market-response model, and it
        is monotonically unfavourable.
        """

        common = dict(
            closes=[100.0] * 4,
            opens=[100.0] * 4,
            target=0.05,
            target_date=1,
            initial_capital=100.0,
            integer_contracts=True,
            max_participation=0.10,
            execution_timing="next_close",
        )
        liquid, index = execution_inputs(
            volumes=[100.0, 100.0, 100.0, 100.0], **common
        )
        thin, _ = execution_inputs(volumes=[100.0, 100.0, 1.0, 100.0], **common)

        # The order is sized identically from lagged information; only the
        # fill differs, and only downward.
        self.assertEqual(liquid.positions.loc[index[2], "X"], 5.0)
        self.assertEqual(thin.positions.loc[index[2], "X"], 0.0)
        self.assertTrue(
            bool(
                (thin.positions["X"].abs() <= liquid.positions["X"].abs()).all()
            )
        )
        # The residual is carried, not cancelled, and completes once depth
        # returns.
        self.assertGreater(thin.daily.loc[index[2], "pending_markets"], 0)
        self.assertEqual(thin.positions.loc[index[3], "X"], 5.0)
        self.assertLessEqual(
            float(thin.daily["max_rebalance_participation"].max()), 0.10 + 1e-12
        )

    def test_realized_volume_never_enlarges_a_fill(self) -> None:
        """Monotonicity: filled quantity is non-decreasing in session volume."""

        common = dict(
            closes=[100.0] * 4,
            opens=[100.0] * 4,
            target=0.05,
            target_date=1,
            initial_capital=100.0,
            integer_contracts=True,
            max_participation=0.10,
            execution_timing="next_close",
        )
        filled = []
        for session_volume in (1.0, 10.0, 30.0, 60.0, 100.0):
            ledger, index = execution_inputs(
                volumes=[100.0, 100.0, session_volume, 100.0], **common
            )
            filled.append(float(ledger.positions.loc[index[2], "X"]))
        self.assertEqual(filled, sorted(filled))
        self.assertLessEqual(max(filled), 5.0)

    def test_realized_capacity_bounds_reported_participation(self) -> None:
        """With integer contracts the reported participation cannot breach."""

        ledger, _ = execution_inputs(
            closes=[100.0] * 6,
            opens=[100.0] * 6,
            volumes=[500.0, 500.0, 40.0, 500.0, 500.0, 500.0],
            target=0.20,
            target_date=1,
            initial_capital=1_000.0,
            integer_contracts=True,
            max_participation=0.10,
            execution_timing="next_close",
        )
        for column in (
            "max_rebalance_participation",
            "max_order_participation",
            "max_roll_participation_proxy",
        ):
            self.assertLessEqual(
                float(ledger.daily[column].max()), 0.10 + 1e-12, column
            )

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

    def test_insolvent_pre_trade_nav_fails_closed(self) -> None:
        # The open crashes with the close, so the overnight mark wipes the NAV
        # before any trade or cost is booked: the pre-trade guard fires.
        with self.assertRaisesRegex(ValueError, "non-positive or non-finite"):
            execution_inputs(
                [100.0, 100.0, -50.0],
                [100.0, 100.0, -50.0],
                [100.0] * 3,
            )

    def test_insolvent_end_of_day_nav_fails_closed(self) -> None:
        # The flat open keeps the overnight mark solvent under next_open
        # timing, so only the end-of-day ledger can observe the loss.
        with self.assertRaisesRegex(ValueError, "non-positive or non-finite"):
            execution_inputs(
                [100.0, 100.0, -50.0],
                [100.0, 100.0, 100.0],
                [100.0] * 3,
            )

    def test_missing_held_valuation_fails_closed_without_a_trade(self) -> None:
        # Back-adjusted closes stay complete, so settlement P&L is computable;
        # only the held position's contract-level valuation is missing.
        with self.assertRaisesRegex(ValueError, "Missing held valuation"):
            execution_inputs(
                [100.0] * 3,
                [100.0] * 3,
                [100.0] * 3,
                valuation_closes=[100.0, 100.0, np.nan],
            )

    def test_roll_event_mask_charges_every_switch_and_flags_anomalies(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=7)
        delivery = pd.DataFrame(
            {"X": [202003, 202003, 202006, 202003, 202006, 202005, 202009]},
            index=index,
        )
        rolls, anomalies = roll_event_mask(delivery)
        self.assertEqual(int(rolls["X"].sum()), 5)
        self.assertTrue(bool(anomalies.iloc[2, 0]))
        self.assertTrue(bool(anomalies.iloc[5, 0]))

    def test_notional_uses_raw_contract_level_not_adjusted_level(self) -> None:
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

    def test_negative_prices_keep_costs_and_notional_nonnegative(self) -> None:
        """A contract price through zero flips neither a cost nor a notional.

        Impact and gross notional are both priced off the absolute contract
        level, so a WTI-April-2020-style excursion — including a roll worked
        on a negative-price session — changes magnitudes, never signs.
        """

        valuation = [5.0, 2.0, -3.0, -4.0, -2.0, 1.0]
        ledger, index = execution_inputs(
            [10.0] * 6,
            [10.0] * 6,
            [100.0] * 6,
            valuation_closes=valuation,
            roll_dates=(3,),
            one_way_cost=1.0,
            impact_bps=10.0,
        )
        daily = ledger.daily
        self.assertTrue(bool((daily["transaction_cost_usd"] >= 0).all()))
        self.assertTrue(bool(ledger.executed_rolls.loc[index[3], "X"]))
        self.assertGreater(float(daily.loc[index[3], "transaction_cost_usd"]), 0.0)
        np.testing.assert_allclose(
            daily["gross_notional_multiple"] * daily["nav"],
            (ledger.positions["X"] * pd.Series(valuation, index=index)).abs(),
        )
        np.testing.assert_allclose(
            daily["nav"],
            daily["prior_nav_usd"]
            + daily["gross_pnl_usd"]
            - daily["transaction_cost_usd"],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_point_values_and_margin_convert_foreign_currency(self) -> None:
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


class TestTradeEpisodes(unittest.TestCase):
    def test_sequence_resize_sign_flip_roll_close_and_censoring(self) -> None:
        result = episode_result()
        episodes = build_trade_episodes(result)
        self.assertEqual(episodes["Episode ID"].tolist(), [
            "ES-0001",
            "ES-0002",
            "ES-0003",
        ])
        self.assertEqual(episodes["Direction"].tolist(), ["Long", "Short", "Long"])
        self.assertEqual(episodes["Status"].tolist(), ["Closed", "Closed", "Open"])

        long_episode, short_episode, censored = [row for _, row in episodes.iterrows()]
        self.assertEqual(long_episode["Maximum absolute contracts"], 2.0)
        self.assertEqual(long_episode["Resize count"], 1)
        self.assertEqual(long_episode["Gross P&L USD"], 6.0)
        self.assertEqual(long_episode["Regular cost USD"], 4.0)
        self.assertEqual(long_episode["Net P&L USD"], 2.0)

        self.assertEqual(short_episode["Roll count"], 1)
        self.assertEqual(short_episode["Roll cost USD"], 2.0)
        self.assertEqual(short_episode["Gross P&L USD"], 5.0)
        self.assertEqual(short_episode["Net P&L USD"], 1.0)
        self.assertTrue(pd.isna(censored["Exit date"]))
        self.assertEqual(censored["Net P&L USD"], -1.0)

    def test_episode_totals_and_closed_metrics_reconcile(self) -> None:
        result = episode_result()
        result.trade_episodes = build_trade_episodes(result)
        episodes = result.trade_episodes
        self.assertAlmostEqual(
            float(episodes["Gross P&L USD"].sum()),
            float(result.gross_pnl_by_market.to_numpy().sum()),
        )
        self.assertAlmostEqual(
            float(episodes["Regular cost USD"].sum()),
            float(result.regular_cost_by_market.to_numpy().sum()),
        )
        self.assertAlmostEqual(
            float(episodes["Roll cost USD"].sum()),
            float(result.roll_cost_by_market.to_numpy().sum()),
        )
        np.testing.assert_allclose(
            episodes["Net P&L USD"],
            episodes["Gross P&L USD"]
            - episodes["Regular cost USD"]
            - episodes["Roll cost USD"],
        )
        metrics = trade_episode_metrics(episodes, "2024-01-01", "2024-12-31")
        self.assertEqual(metrics["Closed trade episodes"], 2)
        self.assertEqual(metrics["Open/censored trade episodes"], 1)

    def test_non_next_close_results_do_not_claim_episode_attribution(self) -> None:
        result = episode_result()
        result.execution_timing = "next_open"
        self.assertTrue(build_trade_episodes(result).empty)

    def test_trade_profit_factors_classify_each_numeraire_by_its_own_sign(self) -> None:
        episodes = pd.DataFrame(
            {
                "Status": ["Closed", "Closed"],
                "Entry date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "Exit date": pd.to_datetime(["2024-01-10", "2024-02-10"]),
                "Net contribution": [-0.02, 0.01],
                "Net P&L USD": [10.0, -5.0],
                "Holding sessions": [7, 7],
            }
        )
        metrics = trade_episode_metrics(episodes, "2024-01-01", "2024-12-31")
        self.assertAlmostEqual(metrics["Trade profit factor (contribution)"], 0.5)
        self.assertAlmostEqual(metrics["Trade profit factor (USD)"], 2.0)
        self.assertAlmostEqual(metrics["Trade win rate"], 0.5)


class TestMetrics(unittest.TestCase):
    def test_cagr_includes_first_return_interval(self) -> None:
        index = pd.to_datetime(["2019-12-31", "2020-01-01", "2021-01-01"])
        result = metric_result([0.0, 0.0, 0.10], index)
        metrics = performance_metrics(result, "2020-01-01", "2021-01-01")
        expected_years = 367 / 365.2425
        self.assertAlmostEqual(float(metrics["Years"]), expected_years)
        self.assertAlmostEqual(float(metrics["CAGR"]), 1.10 ** (1 / expected_years) - 1)

    def test_sortino_uses_downside_root_mean_square(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=4)
        returns = [0.02, -0.01, 0.01, -0.02]
        metrics = performance_metrics(metric_result(returns, index), str(index[0].date()))
        downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
        expected = np.mean(returns) * 252 / (downside * np.sqrt(252))
        self.assertAlmostEqual(float(metrics["Sortino (rf=0)"]), expected)

    def test_drawdown_includes_starting_wealth(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=2)
        metrics = performance_metrics(metric_result([-0.10, 0.0], index), "2020")
        self.assertAlmostEqual(float(metrics["Max drawdown"]), -0.10)


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

    def test_core_ledger_and_execution_invariants(self) -> None:
        daily = self.result.daily
        self.assertTrue(REQUIRED_DAILY_COLUMNS.issubset(daily.columns))
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
        self.assertGreater(int(self.result.executed_rolls.to_numpy().sum()), 0)

        np.testing.assert_allclose(
            self.result.trades,
            self.result.positions - self.result.positions.shift().fillna(0.0),
        )
        lagged_capacity = np.floor(
            volumes.fillna(0.0)
            .rolling(
                self.config.volume_gate_window,
                min_periods=self.config.volume_gate_window,
            )
            .median()
            .shift(1)
            * float(self.config.max_rebalance_participation)
        )
        capacity_breach = self.result.trades.abs().gt(lagged_capacity + 1e-12)
        self.assertFalse(bool(capacity_breach.fillna(False).any().any()))
        self.assertTrue((daily["max_roll_participation_proxy"] >= 0).all())
        self.assertTrue(
            (
                daily["max_order_participation"] + 1e-15
                >= daily["max_rebalance_participation"]
            ).all()
        )
        self.assertTrue(
            (
                daily["max_order_participation"] + 1e-15
                >= daily["max_roll_participation_proxy"]
            ).all()
        )
        # The roll competes for the rebalance's leftover session budget, so
        # every reported participation — combined order, rebalance leg, and
        # roll leg — sits under the one configured cap.
        participation_cap = float(self.config.max_rebalance_participation)
        for column in (
            "max_order_participation",
            "max_rebalance_participation",
            "max_roll_participation_proxy",
        ):
            self.assertLessEqual(
                float(daily[column].max()), participation_cap + 1e-12, column
            )

    def test_per_market_and_episode_attribution_reconcile(self) -> None:
        daily = self.result.daily
        np.testing.assert_allclose(
            daily["gross_pnl_usd"],
            self.result.gross_pnl_by_market.sum(axis=1),
            rtol=1e-10,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            daily["transaction_cost_usd"],
            (
                self.result.regular_cost_by_market
                + self.result.roll_cost_by_market
            ).sum(axis=1),
            rtol=1e-10,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            daily["nav"],
            daily["prior_nav_usd"] + daily["net_pnl_usd"],
            rtol=1e-10,
            atol=1e-8,
        )

        episodes = self.result.trade_episodes
        self.assertFalse(episodes.empty)
        self.assertAlmostEqual(
            float(episodes["Gross P&L USD"].sum()),
            float(self.result.gross_pnl_by_market.to_numpy().sum()),
            places=6,
        )
        self.assertAlmostEqual(
            float(episodes["Regular cost USD"].sum()),
            float(self.result.regular_cost_by_market.to_numpy().sum()),
            places=6,
        )
        self.assertAlmostEqual(
            float(episodes["Roll cost USD"].sum()),
            float(self.result.roll_cost_by_market.to_numpy().sum()),
            places=6,
        )
        np.testing.assert_allclose(
            episodes["Net P&L USD"],
            episodes["Gross P&L USD"]
            - episodes["Regular cost USD"]
            - episodes["Roll cost USD"],
        )
        self.assertAlmostEqual(
            float(episodes["Net contribution"].sum()),
            float(daily["net_return"].sum()),
            places=10,
        )

    def test_report_is_descriptive_without_preselected_performance_gates(self) -> None:
        columns = [str(column).lower() for column in self.report.columns]
        self.assertFalse(any("target" in column for column in columns))
        self.assertFalse(any("pass" in column or "fail" in column for column in columns))
        self.assertIn("validation status", columns)
        self.assertTrue(
            self.report["Validation status"].eq("retrospective_reused_history").all()
        )
        self.assertTrue(self.report["Strategy"].eq(STRATEGY_TECHNICAL_NAME).all())
        self.assertTrue(
            self.report["Strategy designation"].eq(STRATEGY_DESIGNATION).all()
        )

    def test_production_readiness_defaults_to_blocked(self) -> None:
        readiness = production_readiness_report(self.result)
        self.assertEqual(overall_readiness_status(readiness), "BLOCKED")
        evidence = readiness.loc[readiness["category"].eq("external evidence")]
        self.assertFalse(evidence.empty)
        self.assertTrue(evidence["status"].eq("BLOCKED").all())

    def test_canonical_outputs_include_daily_drawdown_bootstrap(self) -> None:
        artifacts = save_outputs(
            self.result,
            self.report,
            self.config,
            bootstrap_samples=4,
            trade_sequence_samples=4,
            include_stress=False,
        )
        output_path = (
            Path(self.config.output_dir)
            / "strategy_daily_drawdown_monte_carlo.csv"
        )
        self.assertTrue(output_path.is_file())
        persisted = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(
            persisted,
            artifacts["daily_drawdown_bootstrap"],
            check_dtype=False,
        )
        self.assertFalse(persisted.empty)
        self.assertTrue(
            persisted["Drawdown resolution"].eq(
                "daily close; includes initial capital"
            ).all()
        )

        manifest = json.loads(
            (Path(self.config.output_dir) / "run_manifest.json").read_text()
        )
        configuration = json.loads(
            (Path(self.config.output_dir) / "strategy_config.json").read_text()
        )
        self.assertEqual(manifest["strategy"], STRATEGY_TECHNICAL_NAME)
        self.assertEqual(
            manifest["strategy_designation"], STRATEGY_DESIGNATION
        )
        self.assertEqual(
            manifest["strategy_technical_name"], STRATEGY_TECHNICAL_NAME
        )
        self.assertEqual(
            configuration["strategy_designation"], STRATEGY_DESIGNATION
        )
        self.assertEqual(
            configuration["research_status"]["designation"],
            STRATEGY_DESIGNATION,
        )
        self.assertEqual(manifest["production_readiness"], "BLOCKED")
        self.assertFalse(manifest["externally_validated"])
        filename = "strategy_daily_drawdown_monte_carlo.csv"
        self.assertIn(filename, manifest["output_files"])
        self.assertEqual(
            manifest["daily_drawdown_bootstrap"],
            {
                "block_lengths_sessions": [21, 63, 126],
                "horizons_sessions": [2520, 6300],
                "samples": 4,
                "seed": 20_260_803,
            },
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
        pd.testing.assert_frame_equal(self.result.signals.loc[:cutoff], truncated.signals)


if __name__ == "__main__":
    unittest.main()
