from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from delta1_strategy.research import inference
from delta1_strategy.research.benchmarks import (
    ALPHA_REPORT_COLUMNS,
    BENCHMARK_COMPARISON_COLUMNS,
    BENCHMARK_FAMILY,
    BLOCK_SIGN_FLIP,
    IID_SIGN_FLIP,
    OVERLAY_WEIGHT_COLUMNS,
    PERMUTATION_NULL_COLUMNS,
    SIGNAL_DIAGNOSTIC_COLUMNS,
    SPANNING_COEFFICIENT_COLUMNS,
    SPANNING_REPORT_COLUMNS,
    BenchmarkConfig,
    _hac_ols,
    _median_run_length,
    _run_identifiers,
    apply_overlay_to_decisions,
    assert_seam_reproduces_incumbent,
    baltas_kosowski_signal,
    barroso_santa_clara_weights,
    baz_macd_signal,
    benchmark_alpha_report,
    comparison_frame,
    comparison_row,
    equal_notional_decision_frame,
    errored_row,
    ex_ante_volatility,
    excess_returns,
    hop_blend_signal,
    hop_decision_frame,
    inverse_volatility_weights,
    long_only_signal,
    monthly_excess_returns,
    mop_tsmom_signal,
    moreira_muir_weights,
    notional_weights_to_contracts_per_dollar,
    overlay_weight_report,
    prepare_engine_inputs,
    release_leverage_cap,
    sign_permutation_null,
    signal_decision_frame,
    signal_diagnostic_row,
    simulate_monthly_targets,
    solve_volatility_matched_budget,
    spanning_family_keys,
    spanning_report,
    time_trend_tstatistic,
)
from delta1_strategy.research.strategy import (
    StrategyConfig,
    _month_end_rows,
    _simulate_execution,
    tradeable_mask,
)


FORBIDDEN_COLUMN_WORDS = ("target", "pass", "fail", "verdict")
DECISION_WORDS = ("rank", "score", "winner", "recommend", "promote", "best")

# Real catalogue symbols: build_trade_episodes resolves an asset class per
# market, so a made-up ticker would fail inside the shared ledger.
SYMBOLS = ("ES", "ZN", "6E", "CL")


def business_index(sessions: int, start: str = "1995-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=sessions)


def synthetic_frames(
    sessions: int = 1400,
    symbols: tuple[str, ...] = SYMBOLS,
    seed: int = 11,
) -> dict[str, pd.DataFrame]:
    """A complete eight-frame panel with the shape the engine expects.

    One market is deliberately given a back-adjusted path that crosses zero, so
    every test that touches the return definition exercises the case that makes
    ``prices.pct_change()`` meaningless on the real panel.
    """

    rng = np.random.default_rng(seed)
    index = business_index(sessions)
    steps = rng.normal(0.05, 1.0, size=(sessions, len(symbols)))
    drift = np.linspace(0.0, 1.0, len(symbols)) - 0.5
    adjusted = pd.DataFrame(
        100.0 + np.cumsum(steps + drift * 0.02, axis=0),
        index=index,
        columns=list(symbols),
    )
    # Force the first market's back-adjusted level through zero.
    adjusted[symbols[0]] = adjusted[symbols[0]] - adjusted[symbols[0]].iloc[sessions // 2]
    unadjusted = adjusted - adjusted.min() + 50.0
    volumes = pd.DataFrame(50_000.0, index=index, columns=list(symbols))
    delivery = pd.DataFrame(
        np.repeat(
            (np.arange(sessions) // 63 + 1)[:, None],
            len(symbols),
            axis=1,
        ).astype(float),
        index=index,
        columns=list(symbols),
    )
    metadata = pd.DataFrame(
        {
            "currency": ["USD"] * len(symbols),
            "point_value": [50.0] * len(symbols),
            "tick_size": [0.25] * len(symbols),
            "margin": [5_000.0] * len(symbols),
        },
        index=list(symbols),
    )
    fx_rates = pd.DataFrame({"USD": 1.0}, index=index)
    return {
        "prices": adjusted,
        "observed_opens": adjusted.shift(1).bfill(),
        "observed_closes": adjusted,
        "unadjusted": unadjusted,
        "metadata": metadata,
        "volumes": volumes,
        "delivery_months": delivery,
        "fx_rates": fx_rates,
    }


def synthetic_config(frames: dict[str, pd.DataFrame]) -> StrategyConfig:
    launch = frames["prices"].index[400]
    return StrategyConfig(
        data_dir=".",
        output_dir=".",
        launch_date=launch.date().isoformat(),
        max_gross_notional_multiple=None,
    )


def small_benchmark_config() -> BenchmarkConfig:
    """Shorter windows so a 1,400-session synthetic panel warms up."""

    return BenchmarkConfig(
        volatility_center_of_mass=20,
        volatility_min_periods=20,
        trend_lookback_sessions=126,
        min_return_observations=100,
        hop_horizon_sessions=(21, 42, 126),
        covariance_months=12,
        macd_price_volatility_window=42,
        macd_signal_volatility_window=126,
        macd_warmup_sessions=200,
    )


def series(values: np.ndarray, start: str = "2000-01-03") -> pd.Series:
    return pd.Series(values, index=business_index(len(values), start))


class ExcessReturnTests(unittest.TestCase):
    def test_denominator_is_the_previous_unadjusted_price(self) -> None:
        index = business_index(3)
        adjusted = pd.DataFrame({"A": [10.0, 12.0, 11.0]}, index=index)
        unadjusted = pd.DataFrame({"A": [100.0, 104.0, 101.0]}, index=index)
        returns = excess_returns(adjusted, unadjusted)
        self.assertTrue(np.isnan(returns["A"].iloc[0]))
        self.assertAlmostEqual(float(returns["A"].iloc[1]), 2.0 / 100.0)
        self.assertAlmostEqual(float(returns["A"].iloc[2]), -1.0 / 104.0)

    def test_non_positive_back_adjusted_prices_stay_finite(self) -> None:
        index = business_index(4)
        adjusted = pd.DataFrame({"A": [2.0, -1.0, -3.0, 0.0]}, index=index)
        unadjusted = pd.DataFrame({"A": [90.0, 87.0, 85.0, 88.0]}, index=index)
        returns = excess_returns(adjusted, unadjusted)
        observed = returns["A"].to_numpy(dtype=float)[1:]
        self.assertTrue(np.isfinite(observed).all())
        # A percentage change on the back-adjusted panel would be nonsense here.
        naive = adjusted["A"].pct_change().to_numpy(dtype=float)[1:]
        self.assertFalse(np.allclose(observed, naive))

    def test_a_non_positive_unadjusted_denominator_is_masked(self) -> None:
        index = business_index(3)
        adjusted = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=index)
        unadjusted = pd.DataFrame({"A": [100.0, 0.0, 101.0]}, index=index)
        returns = excess_returns(adjusted, unadjusted)
        self.assertTrue(np.isnan(returns["A"].iloc[2]))


class ExAnteVolatilityTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.returns = pd.DataFrame(
            {"A": rng.normal(0.0, 0.01, 400)}, index=business_index(400)
        )

    def test_matches_the_paper_recursion_and_is_lagged_one_session(self) -> None:
        config = BenchmarkConfig()
        alpha = 1.0 / 61.0
        values = self.returns["A"].to_numpy(dtype=float)
        mean = 0.0
        second = 0.0
        expected: list[float] = []
        for position, value in enumerate(values):
            if position == 0:
                mean = value
                second = 0.0
            else:
                mean = (1 - alpha) * mean + alpha * value
                second = (1 - alpha) * second + alpha * (value - mean) ** 2
            expected.append(math.sqrt(261.0 * second))
        volatility = ex_ante_volatility(self.returns, config)["A"].to_numpy(dtype=float)
        # min_periods=60 blanks the warm-up, and the whole frame is shifted one
        # session, so entry t must equal the unshifted recursion at t-1.
        for position in range(80, 400):
            self.assertAlmostEqual(
                volatility[position], expected[position - 1], places=12
            )

    def test_uses_the_population_second_moment_not_a_corrected_variance(self) -> None:
        config = BenchmarkConfig()
        alpha = 1.0 / 61.0
        corrected = (
            self.returns.ewm(alpha=alpha, adjust=False).std().shift(1)
            * math.sqrt(261.0)
        )
        produced = ex_ante_volatility(self.returns, config)
        overlap = produced["A"].dropna().index
        self.assertGreater(len(overlap), 100)
        self.assertFalse(
            np.allclose(
                produced.loc[overlap, "A"].to_numpy(),
                corrected.loc[overlap, "A"].to_numpy(),
            )
        )

    def test_a_future_return_cannot_change_an_earlier_volatility(self) -> None:
        config = BenchmarkConfig()
        baseline = ex_ante_volatility(self.returns, config)
        shocked = self.returns.copy()
        shocked.iloc[-1, 0] = 5.0
        perturbed = ex_ante_volatility(shocked, config)
        pd.testing.assert_frame_equal(baseline.iloc[:-1], perturbed.iloc[:-1])

    def test_annualization_constant_is_261_not_252(self) -> None:
        produced = ex_ante_volatility(self.returns, BenchmarkConfig())
        alternative = ex_ante_volatility(
            self.returns, BenchmarkConfig(volatility_annualization=252)
        )
        ratio = (produced / alternative).dropna().to_numpy(dtype=float)
        self.assertTrue(np.allclose(ratio, math.sqrt(261.0 / 252.0)))


class SignalConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = synthetic_frames()
        self.config = small_benchmark_config()
        self.returns = excess_returns(
            self.frames["prices"], self.frames["unadjusted"]
        )

    def test_mop_uses_the_current_month_and_does_not_skip_one(self) -> None:
        index = business_index(300)
        returns = pd.DataFrame(0.0, index=index, columns=["A"])
        # Everything flat except a large gain in the final 21 sessions.
        returns.iloc[-21:, 0] = 0.01
        config = BenchmarkConfig(trend_lookback_sessions=252, min_return_observations=200)
        signal = mop_tsmom_signal(returns, config)
        self.assertEqual(float(signal["A"].iloc[-1]), 1.0)
        # A skip-month rule would see only the flat window and return 0.
        skipped = mop_tsmom_signal(returns.shift(21).fillna(0.0), config)
        self.assertEqual(float(skipped["A"].iloc[-1]), 0.0)

    def test_hop_blend_takes_only_the_declared_values(self) -> None:
        blend = hop_blend_signal(self.returns, self.config)
        observed = blend.to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        allowed = np.array([-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0])
        for value in np.unique(observed):
            self.assertTrue(
                np.isclose(value, allowed).any(), msg=f"unexpected blend value {value}"
            )

    def test_hop_drops_a_market_rather_than_renormalising_a_missing_sleeve(self) -> None:
        returns = self.returns.copy()
        returns.iloc[:, 0] = np.nan
        returns.iloc[-30:, 0] = 0.001
        blend = hop_blend_signal(returns, self.config)
        self.assertTrue(blend.iloc[:, 0].isna().all())

    def test_rolling_time_trend_matches_a_direct_least_squares_fit(self) -> None:
        config = BenchmarkConfig(trend_lookback_sessions=126, min_return_observations=100)
        statistic = time_trend_tstatistic(self.returns, config)
        cumulative = np.log1p(self.returns).fillna(0.0).cumsum()
        window = config.trend_lookback_sessions
        for position in (600, 900, 1200):
            for symbol in self.returns.columns[:2]:
                block = cumulative[symbol].to_numpy(dtype=float)[
                    position - window + 1 : position + 1
                ]
                design = np.column_stack(
                    [np.ones(window), np.arange(window, dtype=float)]
                )
                coefficients, *_ = np.linalg.lstsq(design, block, rcond=None)
                residuals = block - design @ coefficients
                sum_squares = window * (window**2 - 1) / 12.0
                error = math.sqrt(
                    (float(residuals @ residuals) / (window - 2)) / sum_squares
                )
                with self.subTest(position=position, symbol=symbol):
                    self.assertAlmostEqual(
                        float(statistic[symbol].iloc[position]),
                        coefficients[1] / error,
                        places=8,
                    )

    def test_clip_and_hard_threshold_are_different_strategies(self) -> None:
        clipped = baltas_kosowski_signal(self.returns, self.config)
        thresholded = baltas_kosowski_signal(
            self.returns, self.config, hard_threshold=True
        )
        self.assertLessEqual(float(np.nanmax(np.abs(clipped.to_numpy()))), 1.0)
        self.assertFalse(clipped.equals(thresholded))
        values = thresholded.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        self.assertTrue(np.isin(values, (-1.0, 0.0, 1.0)).all())

    def test_macd_uses_alpha_one_over_n_and_stays_inside_the_phi_bound(self) -> None:
        signal = baz_macd_signal(self.frames["prices"], self.config)
        peak = math.sqrt(2.0) * math.exp(-0.5) / 0.89
        observed = np.abs(signal.to_numpy(dtype=float))
        observed = observed[np.isfinite(observed)]
        self.assertGreater(observed.size, 0)
        self.assertLessEqual(float(observed.max()), peak + 1e-12)
        # A span=n reading would be a materially different filter.
        prices = self.frames["prices"]
        alpha_form = prices.ewm(alpha=1.0 / 8.0, adjust=False).mean()
        span_form = prices.ewm(span=8, adjust=False).mean()
        self.assertFalse(np.allclose(alpha_form.to_numpy(), span_form.to_numpy()))

    def test_every_signal_is_truncation_invariant(self) -> None:
        cutoff = self.frames["prices"].index[1000]
        truncated_returns = self.returns.loc[:cutoff]
        truncated_prices = self.frames["prices"].loc[:cutoff]
        builders = {
            "mop": (lambda r, p: mop_tsmom_signal(r, self.config)),
            "hop": (lambda r, p: hop_blend_signal(r, self.config)),
            "baltas_kosowski": (lambda r, p: baltas_kosowski_signal(r, self.config)),
            "baz_macd": (lambda r, p: baz_macd_signal(p, self.config)),
        }
        for name, build in builders.items():
            with self.subTest(signal=name):
                full = build(self.returns, self.frames["prices"]).loc[:cutoff]
                cut = build(truncated_returns, truncated_prices)
                pd.testing.assert_frame_equal(full, cut)

    def test_no_signal_reacts_to_a_future_observation(self) -> None:
        shocked_prices = self.frames["prices"].copy()
        shocked_prices.iloc[-1] = shocked_prices.iloc[-1] * 3.0
        shocked_returns = excess_returns(shocked_prices, self.frames["unadjusted"])
        for name, baseline, perturbed in (
            (
                "mop",
                mop_tsmom_signal(self.returns, self.config),
                mop_tsmom_signal(shocked_returns, self.config),
            ),
            (
                "baz_macd",
                baz_macd_signal(self.frames["prices"], self.config),
                baz_macd_signal(shocked_prices, self.config),
            ),
        ):
            with self.subTest(signal=name):
                pd.testing.assert_frame_equal(baseline.iloc[:-1], perturbed.iloc[:-1])


class SizingTests(unittest.TestCase):
    def test_weight_uses_one_over_count_not_one_over_root_count(self) -> None:
        index = business_index(3)
        columns = ["A", "B", "C", "D"]
        signal = pd.DataFrame(1.0, index=index, columns=columns)
        volatility = pd.DataFrame(0.20, index=index, columns=columns)
        eligible = pd.DataFrame(True, index=index, columns=columns)
        weights = inverse_volatility_weights(
            signal, volatility, eligible, volatility_budget=0.40
        )
        expected = 0.40 / 0.20 / 4.0
        self.assertAlmostEqual(float(weights.iloc[0, 0]), expected)
        self.assertAlmostEqual(float(weights.abs().sum(axis=1).iloc[0]), 4.0 * expected)
        root_form = 0.40 / 0.20 / math.sqrt(4.0)
        self.assertNotAlmostEqual(float(weights.iloc[0, 0]), root_form)

    def test_a_market_failing_the_volume_gate_is_excluded_from_the_count(self) -> None:
        index = business_index(2)
        columns = ["A", "B"]
        signal = pd.DataFrame(1.0, index=index, columns=columns)
        volatility = pd.DataFrame(0.20, index=index, columns=columns)
        eligible = pd.DataFrame(
            {"A": [True, True], "B": [False, False]}, index=index
        )
        weights = inverse_volatility_weights(
            signal, volatility, eligible, volatility_budget=0.40
        )
        self.assertAlmostEqual(float(weights.loc[index[0], "A"]), 0.40 / 0.20)
        self.assertEqual(float(weights.loc[index[0], "B"]), 0.0)

    def test_contracts_use_the_unadjusted_notional(self) -> None:
        index = business_index(1)
        weights = pd.DataFrame({"A": [0.5]}, index=index)
        valuation = pd.DataFrame({"A": [80.0]}, index=index)
        point_values = pd.DataFrame({"A": [50.0]}, index=index)
        contracts = notional_weights_to_contracts_per_dollar(
            weights, valuation, point_values
        )
        self.assertAlmostEqual(float(contracts.iloc[0, 0]), 0.5 / (80.0 * 50.0))

    def test_a_missing_valuation_zeroes_the_contract_intent(self) -> None:
        index = business_index(1)
        weights = pd.DataFrame({"A": [0.5]}, index=index)
        valuation = pd.DataFrame({"A": [np.nan]}, index=index)
        point_values = pd.DataFrame({"A": [50.0]}, index=index)
        contracts = notional_weights_to_contracts_per_dollar(
            weights, valuation, point_values
        )
        self.assertEqual(float(contracts.iloc[0, 0]), 0.0)


class EqualNotionalControlTests(unittest.TestCase):
    """The equal-notional row is the smallest book in the family, so the
    granularity of a whole contract bites it and nothing else.  Both tests here
    exist because the difference between it and the equal-risk row was once
    published as a clean read of the weighting rule, which it is not."""

    def coarse_frames(self) -> dict[str, pd.DataFrame]:
        """The same panel with a point value large against the capital base.

        This is the real panel's situation in miniature: ``w = 1/S_t`` asks for
        a fraction of a contract in most markets, while the inverse-volatility
        rule asks for a multiple of the same intent and survives the rounding.
        """

        frames = dict(synthetic_frames())
        metadata = frames["metadata"].copy()
        metadata["point_value"] = 7_000.0
        frames["metadata"] = metadata
        return frames

    def test_the_equal_notional_book_loses_markets_to_integer_contracts(self) -> None:
        frames = self.coarse_frames()
        config = synthetic_config(frames)
        benchmark_config = small_benchmark_config()
        inputs = prepare_engine_inputs(config, frames)
        notional_frame, _ = equal_notional_decision_frame(inputs, benchmark_config)
        risk_frame, _ = signal_decision_frame(
            long_only_signal(inputs), inputs, benchmark_config
        )
        capital = float(config.initial_capital)
        lost: dict[str, float] = {}
        for name, frame in (("notional", notional_frame), ("risk", risk_frame)):
            contracts = frame * capital
            intended = contracts.ne(0.0)
            self.assertGreater(float(intended.sum(axis=1).mean()), 3.0)
            lost[name] = float(
                (intended & (contracts.abs() < 0.5)).sum(axis=1).mean()
            )
        # The two rules see the same universe by construction, so any gap in
        # the delivered book is granularity and not the weighting rule.
        self.assertGreater(lost["notional"], 2.0)
        self.assertEqual(lost["risk"], 0.0)

    def test_the_published_row_discloses_the_granularity_shortfall(self) -> None:
        specification = next(
            item for item in BENCHMARK_FAMILY if item.key == "long_only_equal_notional"
        )
        text = specification.construction.lower()
        self.assertIn("integer", text)
        self.assertIn("not a clean read", text)
        # The row may not claim the gap isolates the weighting rule, because
        # the same gap also carries a universe and an investment-level change.
        self.assertNotIn("shows how much", text)


class HopPortfolioScalingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = synthetic_frames()
        self.config = small_benchmark_config()
        self.inputs = prepare_engine_inputs(
            synthetic_config(self.frames), self.frames
        )

    def test_ex_ante_volatility_of_the_scaled_book_is_the_declared_budget(self) -> None:
        _, weights, diagnostics = hop_decision_frame(self.inputs, self.config)
        self.assertFalse(diagnostics.empty)
        monthly = monthly_excess_returns(self.inputs.excess_returns)
        checked = 0
        for decision in weights.index[-5:]:
            row = weights.loc[decision]
            live = row[row != 0.0]
            if len(live) < 2:
                continue
            block = monthly.loc[: decision.to_period("M")].iloc[
                -self.config.covariance_months :
            ][live.index]
            covariance = np.cov(block.to_numpy(dtype=float), rowvar=False, ddof=1) * 12.0
            values = live.to_numpy(dtype=float)
            realized = math.sqrt(float(values @ covariance @ values))
            self.assertAlmostEqual(
                realized, self.config.portfolio_volatility_budget, places=10
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_the_covariance_is_monthly_and_never_inverted(self) -> None:
        monthly = monthly_excess_returns(self.inputs.excess_returns)
        self.assertIsInstance(monthly.index, pd.PeriodIndex)
        self.assertLess(len(monthly), len(self.inputs.excess_returns) / 15)
        _, weights, diagnostics = hop_decision_frame(self.inputs, self.config)
        self.assertTrue((diagnostics["ex_ante_portfolio_volatility"] > 0).all())
        self.assertTrue(np.isfinite(diagnostics["portfolio_scaling_constant"]).all())

    def test_hop_decisions_are_truncation_invariant(self) -> None:
        cutoff = self.frames["prices"].index[1100]
        truncated = {
            key: (frame.loc[:cutoff] if key != "metadata" else frame)
            for key, frame in self.frames.items()
        }
        truncated_inputs = prepare_engine_inputs(
            synthetic_config(self.frames), truncated
        )
        full_frame, _, _ = hop_decision_frame(self.inputs, self.config)
        cut_frame, _, _ = hop_decision_frame(truncated_inputs, self.config)
        shared = cut_frame.index
        self.assertGreater(len(shared), 10)
        pd.testing.assert_frame_equal(
            full_frame.loc[shared], cut_frame, check_freq=False
        )


class OverlayTests(unittest.TestCase):
    def test_barroso_weight_is_a_monthly_quantity(self) -> None:
        config = BenchmarkConfig()
        deviation = 0.004
        rng = np.random.default_rng(5)
        returns = series(rng.normal(0.0, deviation, 400))
        decisions = pd.DatetimeIndex([returns.index[-1]])
        weights = barroso_santa_clara_weights(returns, decisions, config)
        realized = math.sqrt(
            float(np.mean(returns.to_numpy()[-126:] ** 2)) * 21.0
        )
        expected = (0.12 / math.sqrt(12.0)) / realized
        self.assertAlmostEqual(float(weights.iloc[0]), expected, places=12)
        # The annual-target transcription error is off by sqrt(12).
        self.assertNotAlmostEqual(float(weights.iloc[0]), 0.12 / realized, places=3)

    def test_barroso_does_not_demean_but_moreira_does(self) -> None:
        config = BenchmarkConfig(moreira_min_history_months=2)
        index = business_index(300)
        values = pd.Series(0.01, index=index)  # constant, so demeaning matters
        decisions = pd.DatetimeIndex([index[-1]])
        barroso = barroso_santa_clara_weights(values, decisions, config)
        self.assertTrue(np.isfinite(float(barroso.iloc[0])))
        # A demeaned realised variance of a constant series is exactly zero, so
        # Moreira-Muir has nothing to scale by and must not produce a number.
        moreira = moreira_muir_weights(
            values, decisions, config, expanding_constant=True
        )
        self.assertTrue(np.isnan(float(moreira.iloc[0])))

    def test_a_moreira_decision_reads_its_own_month_not_the_month_before(
        self,
    ) -> None:
        """``RV`` of month ``m``, because the decision governs month ``m+1``.

        This is the assertion that pins the lag, and nothing else in the suite
        can: a weight taken from the wrong month is finite, causal, plausibly
        distributed and dimensionally correct.  It simply carries none of the
        rule's conditioning information, so only an exact mapping test sees it.
        """

        index = business_index(760, "2000-01-03")
        periods = index.to_period("M")
        rng = np.random.default_rng(3)
        # A per-month dispersion that cycles over seven distinct levels, so the
        # month a weight was read from is identifiable from the weight alone.
        scale = {
            period: 0.002 * (1.0 + 5.0 * (position % 7))
            for position, period in enumerate(periods.unique())
        }
        returns = pd.Series(
            [rng.normal(0.0, scale[period]) for period in periods], index=index
        )
        decisions = pd.DatetimeIndex(_month_end_rows(returns.to_frame("x")).index)
        weights = moreira_muir_weights(
            returns, decisions, BenchmarkConfig(), expanding_constant=False
        )
        demeaned = returns - returns.groupby(periods).transform("mean")
        realized = (demeaned**2).groupby(periods).sum()
        last = decisions[-1].to_period("M")
        constant = float(weights.loc[decisions[-1]]) * float(realized.loc[last])
        checked = 0
        for decision in decisions[-8:]:
            month = decision.to_period("M")
            with self.subTest(month=str(month)):
                self.assertAlmostEqual(
                    float(weights.loc[decision]),
                    constant / float(realized.loc[month]),
                    places=10,
                )
                self.assertNotAlmostEqual(
                    float(weights.loc[decision]),
                    constant / float(realized.loc[month - 1]),
                    places=3,
                )
                if (month + 1) in realized.index:
                    # ... and equally not the month it is about to trade, which
                    # is the look-ahead the same one-line slip could produce in
                    # the other direction.
                    self.assertNotAlmostEqual(
                        float(weights.loc[decision]),
                        constant / float(realized.loc[month + 1]),
                        places=3,
                    )
            checked += 1
        self.assertEqual(checked, 8)

    def test_moreira_full_sample_constant_matches_unmanaged_dispersion(self) -> None:
        rng = np.random.default_rng(9)
        # Deliberately heteroskedastic.  On a homoskedastic series every month's
        # realised variance is the same up to sampling noise, so the identity
        # below holds whichever month the weight is read from and the test
        # proves nothing; a persistent volatility cycle is what makes it bite.
        index = business_index(900, "2000-01-03")
        periods = index.to_period("M")
        cycle = {
            period: 0.004 * (1.0 + 0.8 * math.sin(2.0 * math.pi * position / 9.0))
            for position, period in enumerate(periods.unique())
        }
        returns = pd.Series(
            [rng.normal(0.0004, cycle[period]) for period in periods], index=index
        )
        decisions = pd.DatetimeIndex(_month_end_rows(returns.to_frame("x")).index)
        weights = moreira_muir_weights(
            returns, decisions, BenchmarkConfig(), expanding_constant=False
        )
        # The decision at the end of month m scales the sessions that follow it,
        # so the weight is stepped forward one session before it is applied.
        # Applying it on the decision session itself would blur the month
        # boundary the constant is calibrated against.
        applied = weights.reindex(returns.index).shift(1).ffill()
        managed = (applied * returns).dropna()
        unmanaged = returns.loc[managed.index]
        # The constant is calibrated on the proxy series, and the simulated
        # series must be that same series: matching the paper's defining
        # identity to a few percent, not merely to the same order of magnitude.
        # A weight read from the wrong month leaves this ratio visibly off.
        self.assertLess(
            abs(managed.std(ddof=1) / unmanaged.std(ddof=1) - 1.0), 0.02
        )

    def test_moreira_expanding_constant_ignores_the_future(self) -> None:
        rng = np.random.default_rng(13)
        returns = series(rng.normal(0.0004, 0.006, 900))
        decisions = pd.DatetimeIndex(_month_end_rows(returns.to_frame("x")).index)
        config = BenchmarkConfig(moreira_min_history_months=6)
        baseline = moreira_muir_weights(
            returns, decisions, config, expanding_constant=True
        )
        shocked = returns.copy()
        shocked.iloc[-40:] = shocked.iloc[-40:] * 12.0
        perturbed = moreira_muir_weights(
            shocked, decisions, config, expanding_constant=True
        )
        cut = decisions[decisions < shocked.index[-40]]
        pd.testing.assert_series_equal(baseline.loc[cut], perturbed.loc[cut])

    def test_full_sample_constant_does_react_to_the_future(self) -> None:
        rng = np.random.default_rng(17)
        returns = series(rng.normal(0.0004, 0.006, 900))
        decisions = pd.DatetimeIndex(_month_end_rows(returns.to_frame("x")).index)
        baseline = moreira_muir_weights(
            returns, decisions, BenchmarkConfig(), expanding_constant=False
        )
        shocked = returns.copy()
        shocked.iloc[-40:] = shocked.iloc[-40:] * 12.0
        perturbed = moreira_muir_weights(
            shocked, decisions, BenchmarkConfig(), expanding_constant=False
        )
        cut = decisions[decisions < shocked.index[-40]]
        # This is the look-ahead the non-causal variant exists to measure.
        self.assertFalse(np.allclose(baseline.loc[cut], perturbed.loc[cut]))

    def test_overlay_multiplies_decisions_and_leaves_warm_up_unmanaged(self) -> None:
        index = business_index(3)
        decisions = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [-1.0, -2.0, -3.0]}, index=index
        )
        weights = pd.Series([np.nan, 2.0, 0.5], index=index)
        scaled = apply_overlay_to_decisions(decisions, weights)
        self.assertEqual(float(scaled.loc[index[0], "A"]), 1.0)
        self.assertEqual(float(scaled.loc[index[1], "A"]), 4.0)
        self.assertEqual(float(scaled.loc[index[2], "B"]), -1.5)


class ExecutionSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = synthetic_frames()
        self.benchmark_config = small_benchmark_config()
        self.config = synthetic_config(self.frames)
        self.inputs = prepare_engine_inputs(self.config, self.frames)

    def test_partial_frames_are_refused(self) -> None:
        partial = {
            key: frame
            for key, frame in self.frames.items()
            if key != "delivery_months"
        }
        with self.assertRaises(ValueError):
            prepare_engine_inputs(self.config, partial)

    def test_the_eligible_universe_is_the_strategy_config_s_own_gate(self) -> None:
        """A benchmark may not trade a market the incumbent's gate excludes.

        The gate is applied here, before ``_simulate_execution``, so neither
        ``_require_matching_fees`` nor ``assert_seam_reproduces_incumbent`` can
        observe a divergence in it — both would accept a run in which every
        benchmark traded a different universe from the incumbent.  The defence
        is that there is only one place the gate can come from.
        """

        for window, minimum in ((60, 60_000.0), (120, 1_000.0)):
            with self.subTest(window=window, minimum=minimum):
                varied = replace(
                    self.config,
                    volume_gate_window=window,
                    min_median_contracts=minimum,
                )
                inputs = prepare_engine_inputs(varied, self.frames)
                expected = (
                    tradeable_mask(
                        self.frames["volumes"],
                        varied.volume_gate_window,
                        varied.min_median_contracts,
                    )
                    .reindex(self.frames["prices"].index)
                    .fillna(False)
                )
                pd.testing.assert_frame_equal(inputs.eligible, expected)
                self.assertFalse(inputs.eligible.equals(self.inputs.eligible))
        # The synthetic panel's volume is a constant 50,000 contracts, so a
        # 60,000 threshold empties the universe entirely; the default does not.
        empty = prepare_engine_inputs(
            replace(self.config, min_median_contracts=60_000.0), self.frames
        )
        self.assertFalse(bool(empty.eligible.to_numpy().any()))
        self.assertTrue(bool(self.inputs.eligible.to_numpy().any()))

    def test_a_benchmark_runs_through_the_shared_ledger(self) -> None:
        signal = mop_tsmom_signal(self.inputs.excess_returns, self.benchmark_config)
        frame, _ = signal_decision_frame(signal, self.inputs, self.benchmark_config)
        result = simulate_monthly_targets(
            frame, self.inputs, self.config, name="mop"
        )
        daily = result.daily
        self.assertIn("risk_scalar", daily)
        identity = (
            daily["net_pnl_usd"] - (daily["gross_pnl_usd"] - daily["transaction_cost_usd"])
        ).abs().max()
        self.assertLess(float(identity), 1e-9)
        recomputed = daily["net_pnl_usd"] / daily["prior_nav_usd"]
        self.assertLess(float((recomputed - daily["net_return"]).abs().max()), 1e-12)
        self.assertTrue((daily["nav"] > 0).all())

    def test_the_seam_reproduces_a_supplied_decision_frame_exactly(self) -> None:
        """The whole claim of the module: same frame in, same ledger out."""

        signal = mop_tsmom_signal(self.inputs.excess_returns, self.benchmark_config)
        frame, _ = signal_decision_frame(signal, self.inputs, self.benchmark_config)
        through_module = simulate_monthly_targets(
            frame, self.inputs, self.config, name="mop"
        ).daily["net_return"]
        direct = _simulate_execution(
            frame,
            self.inputs.prices,
            self.inputs.valuation_prices,
            self.inputs.observed_opens,
            self.inputs.observed_closes,
            self.inputs.volumes,
            self.inputs.point_values,
            self.inputs.one_way_costs,
            self.inputs.margin_values,
            self.inputs.rolls,
            self.config,
            initial_capital=self.config.initial_capital,
            integer_contracts=self.config.integer_contracts,
            charge_costs=True,
            launch_date=self.config.launch_date,
        ).daily["net_return"]
        pd.testing.assert_series_equal(through_module, direct)

    def test_the_seam_assertion_accepts_a_faithful_replay_and_rejects_a_drift(
        self,
    ) -> None:
        signal = mop_tsmom_signal(self.inputs.excess_returns, self.benchmark_config)
        frame, _ = signal_decision_frame(signal, self.inputs, self.benchmark_config)
        reference = simulate_monthly_targets(
            frame, self.inputs, self.config, name="reference"
        )
        self.assertEqual(
            assert_seam_reproduces_incumbent(reference, self.inputs, self.config), 0.0
        )
        # A cost change the pre-derived frame cannot see is refused outright.
        with self.assertRaises(ValueError):
            assert_seam_reproduces_incumbent(
                reference,
                self.inputs,
                replace(self.config, commission_per_contract=25.0),
            )
        # A change the engine does honour must show up as a ledger deviation.
        drifted = replace(self.config, impact_bps_at_full_participation=400.0)
        with self.assertRaises(ValueError):
            assert_seam_reproduces_incumbent(reference, self.inputs, drifted)

    def test_releasing_the_leverage_cap_changes_only_that_field(self) -> None:
        released = release_leverage_cap(self.config)
        self.assertIsNone(released.max_gross_notional_multiple)
        capped = StrategyConfig(
            data_dir=".", output_dir=".", max_gross_notional_multiple=5.0
        )
        self.assertEqual(
            release_leverage_cap(capped).max_rebalance_participation,
            capped.max_rebalance_participation,
        )

    def test_decision_frames_are_truncation_invariant(self) -> None:
        cutoff = self.frames["prices"].index[1100]
        truncated = {
            key: (frame.loc[:cutoff] if key != "metadata" else frame)
            for key, frame in self.frames.items()
        }
        truncated_inputs = prepare_engine_inputs(self.config, truncated)
        for name, build in (
            ("mop", lambda i: signal_decision_frame(
                mop_tsmom_signal(i.excess_returns, self.benchmark_config),
                i,
                self.benchmark_config,
            )[0]),
            ("long_only", lambda i: signal_decision_frame(
                long_only_signal(i), i, self.benchmark_config
            )[0]),
            ("equal_notional", lambda i: equal_notional_decision_frame(
                i, self.benchmark_config
            )[0]),
        ):
            with self.subTest(builder=name):
                full = build(self.inputs)
                cut = build(truncated_inputs)
                shared = cut.index
                self.assertGreater(len(shared), 5)
                pd.testing.assert_frame_equal(
                    full.loc[shared], cut, check_freq=False
                )

    def test_the_full_benchmark_pipeline_is_truncation_invariant(self) -> None:
        """The analogue of test_full_pipeline_is_truncation_invariant.

        Signal, sizing, decision scheduling and the execution ledger together
        must not use anything after a cutoff, so the truncated run has to be a
        byte-exact prefix of the full one.
        """

        cutoff = self.frames["prices"].index[1200]
        truncated = {
            key: (frame.loc[:cutoff] if key != "metadata" else frame)
            for key, frame in self.frames.items()
        }
        truncated_inputs = prepare_engine_inputs(self.config, truncated)

        def run(inputs) -> object:
            signal = mop_tsmom_signal(inputs.excess_returns, self.benchmark_config)
            frame, _ = signal_decision_frame(signal, inputs, self.benchmark_config)
            return simulate_monthly_targets(
                frame, inputs, self.config, name="mop"
            )

        full = run(self.inputs)
        short = run(truncated_inputs)
        pd.testing.assert_series_equal(
            full.daily.loc[:cutoff, "net_return"], short.daily["net_return"]
        )
        pd.testing.assert_frame_equal(
            full.positions.loc[:cutoff], short.positions
        )

    def test_volatility_matching_refuses_rather_than_inventing_a_budget(self) -> None:
        def build(_: float) -> pd.DataFrame:
            raise ValueError("engine refused")

        with self.assertRaises(ValueError):
            solve_volatility_matched_budget(
                build,
                self.inputs,
                self.config,
                volatility=0.07,
                max_iterations=3,
            )


class PermutationNullTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = synthetic_frames()
        self.benchmark_config = small_benchmark_config()
        self.config = synthetic_config(self.frames)
        self.inputs = prepare_engine_inputs(self.config, self.frames)
        signal = mop_tsmom_signal(self.inputs.excess_returns, self.benchmark_config)
        self.frame, _ = signal_decision_frame(
            signal, self.inputs, self.benchmark_config
        )
        self.result = simulate_monthly_targets(
            self.frame, self.inputs, self.config, name="underlying"
        )
        self.daily = self.result.daily["net_return"]
        self.window = (
            self.frames["prices"].index[420].date().isoformat(),
            self.frames["prices"].index[-1].date().isoformat(),
        )

    def test_run_identifiers_label_contiguous_constant_sign_blocks(self) -> None:
        values = np.array(
            [
                [1.0, -2.0],
                [3.0, -1.0],
                [0.0, 4.0],
                [2.0, 4.0],
                [-1.0, 0.0],
            ]
        )
        identifiers, total = _run_identifiers(values)
        self.assertEqual(total, 5)
        np.testing.assert_array_equal(identifiers[:, 0], [0, 0, -1, 1, 2])
        np.testing.assert_array_equal(identifiers[:, 1], [3, 3, 4, 4, -1])
        self.assertEqual(_median_run_length(identifiers, total), 2.0)

    def test_the_null_is_reproducible_from_its_seed(self) -> None:
        kwargs = dict(
            incumbent_daily=self.daily,
            permutations=4,
            start=self.window[0],
            end=self.window[1],
        )
        first, _ = sign_permutation_null(
            self.frame, self.inputs, self.config, seed=101, **kwargs
        )
        second, _ = sign_permutation_null(
            self.frame, self.inputs, self.config, seed=101, **kwargs
        )
        pd.testing.assert_frame_equal(first, second)
        different, _ = sign_permutation_null(
            self.frame, self.inputs, self.config, seed=102, **kwargs
        )
        self.assertFalse(
            np.allclose(
                first["sharpe"].to_numpy(dtype=float),
                different["sharpe"].to_numpy(dtype=float),
            )
        )

    def test_costs_are_charged_so_the_null_reference_is_negative(self) -> None:
        draws, summary = sign_permutation_null(
            self.frame,
            self.inputs,
            self.config,
            incumbent_daily=self.daily,
            permutations=5,
            start=self.window[0],
            end=self.window[1],
        )
        completed = draws[draws["run_status"] == "completed"]
        self.assertGreater(len(completed), 0)
        self.assertTrue((completed["annual_cost_drag"] > 0).all())
        self.assertLess(float(summary["cost_only_sharpe_reference"].iloc[0]), 0.0)

    def test_the_p_value_can_never_be_zero(self) -> None:
        _, summary = sign_permutation_null(
            self.frame,
            self.inputs,
            self.config,
            incumbent_daily=self.daily * 1_000.0,
            permutations=3,
            start=self.window[0],
            end=self.window[1],
        )
        value = float(summary["empirical_one_sided_p_value"].iloc[0])
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, 1.0 / 4.0)

    def test_iid_flips_shred_persistence_more_than_block_flips(self) -> None:
        common = dict(
            incumbent_daily=self.daily,
            permutations=3,
            start=self.window[0],
            end=self.window[1],
            seed=7,
        )
        _, block = sign_permutation_null(
            self.frame, self.inputs, self.config, mode=BLOCK_SIGN_FLIP, **common
        )
        _, iid = sign_permutation_null(
            self.frame, self.inputs, self.config, mode=IID_SIGN_FLIP, **common
        )
        self.assertLess(
            float(iid["null_median_run_decisions"].iloc[0]),
            float(block["null_median_run_decisions"].iloc[0]),
        )

    def test_an_unknown_mode_and_an_empty_frame_raise(self) -> None:
        with self.assertRaises(ValueError):
            sign_permutation_null(
                self.frame,
                self.inputs,
                self.config,
                incumbent_daily=self.daily,
                mode="shuffle_returns",
                permutations=1,
            )
        with self.assertRaises(ValueError):
            sign_permutation_null(
                self.frame.iloc[0:0],
                self.inputs,
                self.config,
                incumbent_daily=self.daily,
                permutations=1,
            )


class RegressionTests(unittest.TestCase):
    def test_intercept_only_hac_matches_the_repository_estimator(self) -> None:
        rng = np.random.default_rng(21)
        values = rng.normal(0.0005, 0.006, 800)
        design = np.ones((values.size, 1))
        fit = _hac_ols(values, design, 21)
        expected = inference.hac_standard_error(values, 21)
        self.assertAlmostEqual(float(fit["standard_errors"][0]), expected, places=15)
        self.assertAlmostEqual(float(fit["coefficients"][0]), float(values.mean()))

    def test_alpha_recovers_a_planted_value(self) -> None:
        rng = np.random.default_rng(23)
        sessions = 4000
        benchmark = series(rng.normal(0.0002, 0.005, sessions))
        planted_alpha = 0.0003
        planted_beta = 0.6
        incumbent = (
            planted_alpha
            + planted_beta * benchmark
            + pd.Series(rng.normal(0.0, 0.002, sessions), index=benchmark.index)
        )
        report = benchmark_alpha_report(incumbent, {"published": benchmark})
        row = report.iloc[0]
        # The planted intercept is recovered to within the regression's own
        # reported uncertainty; asserting a tighter band would only be asserting
        # that this particular seed was lucky.
        self.assertLess(
            abs(float(row["alpha_annualized"]) - planted_alpha * 252),
            3.0 * float(row["alpha_hac_standard_error_annualized"]),
        )
        self.assertAlmostEqual(float(row["beta"]), planted_beta, places=2)
        self.assertGreater(float(row["r_squared"]), 0.6)
        self.assertGreater(float(row["alpha_hac_t_statistic"]), 3.0)
        self.assertLess(
            abs(
                float(row["alpha_hac_t_statistic"])
                - float(row["alpha_hedged_mean_t_statistic"])
            ),
            0.10 * abs(float(row["alpha_hac_t_statistic"])),
        )

    def test_alpha_is_zero_when_there_is_none_to_find(self) -> None:
        rng = np.random.default_rng(29)
        benchmark = series(rng.normal(0.0003, 0.005, 4000))
        incumbent = 0.8 * benchmark
        report = benchmark_alpha_report(incumbent, {"published": benchmark})
        self.assertAlmostEqual(float(report.iloc[0]["alpha_annualized"]), 0.0, places=10)
        self.assertAlmostEqual(float(report.iloc[0]["r_squared"]), 1.0, places=12)

    def test_spanning_intercept_vanishes_for_a_combination_of_the_family(self) -> None:
        rng = np.random.default_rng(31)
        first = series(rng.normal(0.0003, 0.005, 2500))
        second = series(rng.normal(0.0001, 0.004, 2500))
        incumbent = 0.3 * first + 0.7 * second
        summary, coefficients = spanning_report(
            incumbent, {"one": first, "two": second}
        )
        self.assertAlmostEqual(
            float(summary.iloc[0]["alpha_annualized"]), 0.0, places=10
        )
        self.assertAlmostEqual(float(summary.iloc[0]["r_squared"]), 1.0, places=12)
        loadings = coefficients.set_index("explanatory_series")["coefficient"]
        self.assertAlmostEqual(float(loadings["one"]), 0.3, places=10)
        self.assertAlmostEqual(float(loadings["two"]), 0.7, places=10)

    def test_spanning_refuses_a_regressor_derived_from_the_incumbent(self) -> None:
        rng = np.random.default_rng(37)
        path = series(rng.normal(0.0003, 0.005, 500))
        with self.assertRaises(ValueError):
            spanning_report(path, {"overlay_on_incumbent": path * 1.5})

    def test_spanning_requires_a_declared_family(self) -> None:
        rng = np.random.default_rng(41)
        path = series(rng.normal(0.0003, 0.005, 500))
        with self.assertRaises(ValueError):
            spanning_report(path, {})

    def test_the_declared_spanning_family_excludes_incumbent_derived_rows(self) -> None:
        keys = spanning_family_keys()
        self.assertGreater(len(keys), 2)
        for key in keys:
            self.assertNotIn("incumbent", key)
        declared = {specification.key for specification in BENCHMARK_FAMILY}
        self.assertEqual(len(declared), len(BENCHMARK_FAMILY))


class ReportConventionTests(unittest.TestCase):
    def emitted_frames(self) -> list[pd.DataFrame]:
        rng = np.random.default_rng(43)
        benchmark = series(rng.normal(0.0002, 0.005, 800))
        incumbent = 0.0002 + 0.7 * benchmark
        summary, coefficients = spanning_report(
            incumbent, {"one": benchmark, "two": benchmark * 0.5 + 1e-6}
        )
        return [
            comparison_frame(
                [
                    errored_row(
                        benchmark="x",
                        family="y",
                        citation="z",
                        construction="c",
                        leverage_cap_applied="none",
                        detail="d",
                    )
                ]
            ),
            benchmark_alpha_report(incumbent, {"one": benchmark}),
            summary,
            coefficients,
            overlay_weight_report(
                pd.Series([1.0, 2.0], index=business_index(2)),
                overlay="o",
                underlying="u",
                causality="causal",
                maximum_risk_scalar_reference=2.0,
            ),
            pd.DataFrame(columns=PERMUTATION_NULL_COLUMNS),
            pd.DataFrame(columns=SIGNAL_DIAGNOSTIC_COLUMNS),
            # The HOP scaling diagnostics are written to CSV by the script, so
            # they are held to the same naming ban as every other artifact.
            hop_decision_frame(
                prepare_engine_inputs(
                    synthetic_config(synthetic_frames()),
                    synthetic_frames(),
                ),
                small_benchmark_config(),
            )[2],
        ]

    def test_no_emitted_frame_can_read_as_a_promotion_decision(self) -> None:
        for frame in self.emitted_frames():
            joined = " ".join(map(str, frame.columns)).lower()
            for word in FORBIDDEN_COLUMN_WORDS:
                self.assertNotIn(word, joined)
            for word in DECISION_WORDS:
                self.assertNotIn(word, joined)

    def test_declared_column_lists_have_no_banned_substrings(self) -> None:
        for columns in (
            BENCHMARK_COMPARISON_COLUMNS,
            ALPHA_REPORT_COLUMNS,
            SPANNING_REPORT_COLUMNS,
            SPANNING_COEFFICIENT_COLUMNS,
            PERMUTATION_NULL_COLUMNS,
            OVERLAY_WEIGHT_COLUMNS,
            SIGNAL_DIAGNOSTIC_COLUMNS,
        ):
            joined = " ".join(columns).lower()
            for word in FORBIDDEN_COLUMN_WORDS + DECISION_WORDS:
                self.assertNotIn(word, joined)

    def test_comparison_frame_preserves_declaration_order(self) -> None:
        rows = [
            errored_row(
                benchmark=name,
                family="f",
                citation="c",
                construction="k",
                leverage_cap_applied="none",
                detail="d",
            )
            for name in ("zzz", "aaa", "mmm")
        ]
        frame = comparison_frame(rows)
        self.assertEqual(list(frame["benchmark"]), ["zzz", "aaa", "mmm"])
        self.assertEqual(list(frame.columns), BENCHMARK_COMPARISON_COLUMNS)

    def test_every_report_states_it_is_not_selection_adjusted(self) -> None:
        rng = np.random.default_rng(47)
        benchmark = series(rng.normal(0.0002, 0.005, 500))
        report = benchmark_alpha_report(0.5 * benchmark, {"one": benchmark})
        self.assertFalse(bool(report["selection_adjusted"].any()))
        self.assertTrue(report["permitted_use"].str.contains("not selection").all())


class SignalDiagnosticTests(unittest.TestCase):
    def test_an_inert_clip_is_visible_as_a_unit_mean_exposure(self) -> None:
        index = business_index(4)
        columns = ["A", "B"]
        decisions = pd.DatetimeIndex(index[1:])
        signal = pd.DataFrame(1.0, index=index, columns=columns)
        weights = pd.DataFrame(0.5, index=index, columns=columns)
        row = signal_diagnostic_row(
            signal, weights, decisions, name="s", citation="c"
        )
        self.assertAlmostEqual(row["mean_absolute_exposure"], 1.0)
        self.assertAlmostEqual(row["average_gross_notional_multiple"], 1.0)
        self.assertEqual(row["decisions"], 3)


class ConfigurationTests(unittest.TestCase):
    def test_out_of_range_fields_raise(self) -> None:
        cases = {
            "instrument_volatility_budget": 0.0,
            "portfolio_volatility_budget": -0.1,
            "macd_normalizing_constant": 0.0,
            "hop_min_observation_fraction": 1.5,
            "volatility_center_of_mass": 0,
            "covariance_months": 1,
            "volatility_min_periods": 0,
        }
        for field, value in cases.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                BenchmarkConfig(**{field: value})

    def test_the_universe_gate_is_not_duplicated_onto_the_benchmark_config(
        self,
    ) -> None:
        """The gate belongs to the engine, so a second copy must not exist.

        A duplicate would be silently authoritative: the gate is applied
        upstream of ``_simulate_execution``, so neither the fee guard nor the
        seam identity can detect a benchmark trading a different universe from
        the incumbent it is compared against.
        """

        fields = set(BenchmarkConfig().__dataclass_fields__)
        self.assertNotIn("volume_gate_window", fields)
        self.assertNotIn("min_median_contracts", fields)
        with self.assertRaises(TypeError):
            BenchmarkConfig(min_median_contracts=5000.0)

    def test_integer_fields_reject_booleans(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(covariance_months=True)

    def test_structural_fields_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(hop_horizon_sessions=())
        with self.assertRaises(ValueError):
            BenchmarkConfig(macd_time_scales=((32, 8),))
        with self.assertRaises(ValueError):
            BenchmarkConfig(min_return_observations=400, trend_lookback_sessions=252)


class ComparisonRowTests(unittest.TestCase):
    def test_a_row_carries_a_scale_free_drawdown_reading(self) -> None:
        frames = synthetic_frames()
        benchmark_config = small_benchmark_config()
        config = synthetic_config(frames)
        inputs = prepare_engine_inputs(config, frames)
        signal = mop_tsmom_signal(inputs.excess_returns, benchmark_config)
        frame, _ = signal_decision_frame(signal, inputs, benchmark_config)
        result = simulate_monthly_targets(frame, inputs, config, name="mop")
        row = comparison_row(
            result,
            benchmark="mop",
            family="f",
            citation="c",
            construction="k",
            leverage_cap_applied="none",
            valuation_prices=inputs.valuation_prices,
            point_values=inputs.point_values,
            start=frames["prices"].index[420].date().isoformat(),
            end=frames["prices"].index[-1].date().isoformat(),
        )
        self.assertAlmostEqual(
            row["max_drawdown_over_volatility"],
            abs(row["max_drawdown"]) / row["annualized_volatility"],
        )
        self.assertGreaterEqual(row["annual_rebalance_notional_turnover_over_nav"], 0.0)
        self.assertEqual(set(row) - set(BENCHMARK_COMPARISON_COLUMNS), set())


if __name__ == "__main__":
    unittest.main()
