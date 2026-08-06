from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.marketdata.etfs import (
    SURVIVORSHIP_LIMITATION,
    UNIVERSE_TICKERS,
    load_panel,
)
from delta1_strategy.research.inference import ESTIMATED, NOT_ESTIMABLE
from delta1_strategy.research.regimes import (
    BEAR,
    HIGH,
    LOW,
    MIDDLE,
    NOT_BEAR,
    NOT_PANIC,
    PANIC,
    RISK_OFF,
    RISK_ON,
    STRESS,
    MarkovSwitchingFit,
    RegimeConfig,
    absorption_ratio,
    absorption_ratio_shift,
    barroso_santa_clara_scale,
    combined_regime_states,
    cooper_gutierrez_hameed_state,
    daniel_moskowitz_bear_state,
    ewma_volatility,
    faber_trend_state,
    fit_markov_switching,
    forbes_rigobon_adjusted_correlation,
    high_volatility_state,
    markov_filtered_probabilities,
    markov_switching_state,
    mean_pairwise_correlation,
    month_end_sessions,
    moreira_muir_scale,
    panic_state,
    realized_volatility,
    regime_state_report,
    regime_transition_matrix,
    regime_transition_report,
    systemic_stress_state,
    time_series_momentum_state,
    volatility_state,
    weekly_equal_weight_log_returns,
)


DATA_DIR = Path(
    os.environ.get("DELTA1_DATA_DIR", "Round1AllData/Quant Researcher/Delta1")
)
REAL_PRICE_DIR = DATA_DIR / "ETF Data"

BANNED_SUBSTRINGS = ("target", "pass", "fail", "verdict", "rank", "winner", "recommend")

SESSIONS = 2600
SLEEVES = 8


def synthetic_prices(periods: int = SESSIONS, seed: int = 17) -> pd.DataFrame:
    """A multi-asset panel with a persistent common factor and a stress episode.

    The point of the stress block is that it exercises the volatility,
    correlation and absorption states rather than leaving them constant, so a
    truncation test on those series is testing something.
    """

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2005-01-03", periods=periods)
    common = rng.normal(0.0003, 0.008, periods)
    stress = slice(int(periods * 0.45), int(periods * 0.60))
    common[stress] = rng.normal(-0.002, 0.028, common[stress].shape[0])
    columns = {}
    for sleeve in range(SLEEVES):
        beta = 0.3 + 0.2 * sleeve
        noise = rng.normal(0.0, 0.006 + 0.001 * sleeve, periods)
        noise[stress] *= 2.5
        returns = beta * common + noise
        columns[f"S{sleeve}"] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(columns, index=index)


def synthetic_returns(periods: int = SESSIONS, seed: int = 17) -> pd.DataFrame:
    prices = synthetic_prices(periods, seed)
    return prices.div(prices.shift(1)).sub(1.0).iloc[1:]


def assert_labels_match(
    case: unittest.TestCase, left: pd.Series, right: pd.Series
) -> None:
    """Equality that treats two missing labels as equal, which ``==`` does not."""

    case.assertTrue(left.index.equals(right.index))
    same = (left.isna() & right.isna()) | (left.astype(object) == right.astype(object))
    case.assertTrue(bool(same.all()), f"{int((~same).sum())} labels differ")


class MonthEndTests(unittest.TestCase):
    def test_the_decision_dates_are_the_last_session_of_each_month(self) -> None:
        index = pd.bdate_range("2020-01-01", "2020-04-30")
        ends = month_end_sessions(index)
        self.assertEqual(
            [stamp.date().isoformat() for stamp in ends],
            ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"],
        )

    def test_an_incomplete_final_month_still_yields_a_decision_date(self) -> None:
        index = pd.bdate_range("2020-01-01", "2020-02-12")
        ends = month_end_sessions(index)
        self.assertEqual(ends[-1], pd.Timestamp("2020-02-12"))

    def test_invalid_arguments_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            month_end_sessions(pd.DatetimeIndex([]))


class FaberTrendTests(unittest.TestCase):
    def test_the_state_matches_the_monthly_average_by_hand(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=500)
        prices = pd.DataFrame(
            {"A": np.linspace(100.0, 200.0, len(index))}, index=index
        )
        state = faber_trend_state(prices, months=10, lag=1)
        ends = month_end_sessions(index)
        monthly = prices["A"].loc[ends]
        # A monotonically rising series sits above its own trailing average on
        # every seasoned decision date, so every label after warm-up is risk on.
        for position in range(10, len(ends)):
            average = float(monthly.iloc[position - 9 : position + 1].mean())
            self.assertGreater(float(monthly.iloc[position]), average)
        labelled = state["A"].dropna()
        self.assertTrue((labelled == RISK_ON).all())

    def test_a_falling_series_is_labelled_risk_off(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=500)
        prices = pd.DataFrame({"A": np.linspace(200.0, 100.0, len(index))}, index=index)
        labelled = faber_trend_state(prices)["A"].dropna()
        self.assertTrue((labelled == RISK_OFF).all())

    def test_a_young_sleeve_is_unlabelled_rather_than_defaulted_to_risk_on(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=500)
        prices = pd.DataFrame(
            {
                "OLD": np.linspace(100.0, 200.0, len(index)),
                "NEW": np.linspace(100.0, 200.0, len(index)),
            },
            index=index,
        )
        prices.loc[prices.index[:300], "NEW"] = np.nan
        state = faber_trend_state(prices, months=10)
        # The sleeve needs ten complete monthly closes after its own inception,
        # so it is missing for longer than the panel-wide warm-up.
        self.assertGreater(
            int(state["NEW"].isna().sum()), int(state["OLD"].isna().sum())
        )
        self.assertTrue(state["NEW"].iloc[:300].isna().all())

    def test_the_series_is_truncation_invariant(self) -> None:
        # The causality proof that matters: values computed on a longer history
        # must be identical on the overlap.  Any forward-looking term --- a
        # centred window, a backward fill, a full-sample normalisation --- would
        # make the earlier values move when later data is appended.
        prices = synthetic_prices()
        full = faber_trend_state(prices)
        truncated = faber_trend_state(prices.iloc[:1800])
        overlap = truncated.index
        self.assertGreater(int(truncated.notna().to_numpy().sum()), 5000)
        for column in truncated.columns:
            assert_labels_match(self, full.loc[overlap, column], truncated[column])

    def test_a_shock_at_a_session_cannot_move_that_session_label(self) -> None:
        prices = synthetic_prices()
        base = faber_trend_state(prices)
        shocked = prices.copy()
        shocked.iloc[1500] = shocked.iloc[1500] * 3.0
        after = faber_trend_state(shocked)
        assert_labels_match(self, base.iloc[1500], after.iloc[1500])

    def test_invalid_arguments_are_refused(self) -> None:
        prices = synthetic_prices(periods=300)
        for kwargs in ({"months": 1}, {"months": True}, {"lag": 0}, {"lag": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                faber_trend_state(prices, **kwargs)
        with self.assertRaises(ValueError):
            faber_trend_state(prices.reset_index(drop=True))


class TimeSeriesMomentumTests(unittest.TestCase):
    def test_the_gate_follows_the_sign_of_the_trailing_return(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=800)
        rising = np.linspace(100.0, 300.0, len(index))
        prices = pd.DataFrame({"UP": rising, "DOWN": rising[::-1]}, index=index)
        state = time_series_momentum_state(prices, lookback=252)
        self.assertTrue((state["UP"].dropna() == RISK_ON).all())
        self.assertTrue((state["DOWN"].dropna() == RISK_OFF).all())

    def test_a_flat_series_is_labelled_risk_off(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=600)
        prices = pd.DataFrame({"FLAT": np.full(len(index), 50.0)}, index=index)
        self.assertTrue(
            (time_series_momentum_state(prices)["FLAT"].dropna() == RISK_OFF).all()
        )

    def test_the_series_is_truncation_invariant(self) -> None:
        prices = synthetic_prices()
        full = time_series_momentum_state(prices)
        truncated = time_series_momentum_state(prices.iloc[:1800])
        for column in truncated.columns:
            assert_labels_match(
                self, full.loc[truncated.index, column], truncated[column]
            )

    def test_a_shock_at_a_session_cannot_move_that_session_label(self) -> None:
        prices = synthetic_prices()
        base = time_series_momentum_state(prices)
        shocked = prices.copy()
        shocked.iloc[2000] = shocked.iloc[2000] * 4.0
        after = time_series_momentum_state(shocked)
        assert_labels_match(self, base.iloc[2000], after.iloc[2000])


class VolatilityTests(unittest.TestCase):
    def test_realized_volatility_is_analytic_on_a_constant_magnitude_series(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=400)
        alternating = pd.Series(
            np.where(np.arange(len(index)) % 2 == 0, 0.01, -0.01), index=index
        )
        estimate = realized_volatility(alternating, window=63, demean=False, lag=1)
        self.assertAlmostEqual(
            float(estimate.iloc[100]), 0.01 * math.sqrt(252), places=12
        )

    def test_the_window_excludes_the_session_being_labelled(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=300)
        quiet = pd.Series(np.full(len(index), 0.001), index=index)
        quiet.iloc[200] = 0.5
        estimate = realized_volatility(quiet, window=21, lag=1)
        before = float(estimate.iloc[200])
        after = float(estimate.iloc[201])
        self.assertLess(before, 0.05)
        self.assertGreater(after, 0.5)

    def test_the_ewma_estimate_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()
        full = ewma_volatility(returns)
        truncated = ewma_volatility(returns.iloc[:1800])
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)

    def test_the_realized_estimate_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()
        full = realized_volatility(returns)
        truncated = realized_volatility(returns.iloc[:1800])
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)

    def test_the_state_is_truncation_invariant(self) -> None:
        book = synthetic_returns().mean(axis=1)
        full = volatility_state(book, minimum_history=252, history=756)
        truncated = volatility_state(
            book.iloc[:1800], minimum_history=252, history=756
        )
        self.assertGreater(int(truncated.notna().sum()), 800)
        assert_labels_match(self, full.loc[truncated.index], truncated)

    def test_the_state_refuses_to_label_before_the_minimum_history(self) -> None:
        book = synthetic_returns().mean(axis=1)
        state = volatility_state(book, window=21, minimum_history=504, history=1260)
        first = state.dropna().index[0]
        position = state.index.get_loc(first)
        self.assertGreaterEqual(position, 21 + 504)
        self.assertEqual(set(state.dropna().unique()), {LOW, MIDDLE, HIGH})

    def test_the_thresholds_are_not_full_sample_percentiles(self) -> None:
        # A full-sample tercile would relabel history when the future arrives.
        book = synthetic_returns().mean(axis=1)
        early = volatility_state(book.iloc[:1500], minimum_history=252, history=756)
        late = volatility_state(book, minimum_history=252, history=756)
        assert_labels_match(self, late.loc[early.index], early)

    def test_invalid_arguments_are_refused(self) -> None:
        book = synthetic_returns(periods=400).mean(axis=1)
        for kwargs in (
            {"window": 1},
            {"lag": 0},
            {"minimum_history": 1},
            {"annualization": 0},
            {"demean": "yes"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                volatility_state(book, **kwargs)


class VolatilityScalingTests(unittest.TestCase):
    def test_barroso_santa_clara_has_no_fitted_constant(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=400)
        steady = pd.Series(
            np.where(np.arange(len(index)) % 2 == 0, 0.01, -0.01), index=index
        )
        scale = barroso_santa_clara_scale(
            steady, window=126, volatility_budget_annualized=0.12, max_scale=10.0
        )
        expected = 0.12 / (0.01 * math.sqrt(252))
        self.assertAlmostEqual(float(scale.iloc[200]), expected, places=12)

    def test_the_scale_is_capped_at_one_for_an_unlevered_book(self) -> None:
        book = synthetic_returns().mean(axis=1)
        scale = barroso_santa_clara_scale(book)
        self.assertLessEqual(float(scale.max()), 1.0)
        self.assertGreaterEqual(float(scale.min()), 0.0)

    def test_the_scale_is_truncation_invariant(self) -> None:
        book = synthetic_returns().mean(axis=1)
        full = barroso_santa_clara_scale(book)
        truncated = barroso_santa_clara_scale(book.iloc[:1800])
        pd.testing.assert_series_equal(full.loc[truncated.index], truncated)

    def test_the_moreira_muir_constant_is_fitted_inside_the_window_only(self) -> None:
        book = synthetic_returns().mean(axis=1)
        boundary = book.index[1500]
        first = moreira_muir_scale(book, fit_window=(book.index[0], boundary))
        # Appending history after the boundary must not move the constant.
        second = moreira_muir_scale(
            book.iloc[:2100], fit_window=(book.index[0], boundary)
        )
        self.assertEqual(first.constant.status, ESTIMATED)
        self.assertAlmostEqual(
            float(first.constant.value), float(second.constant.value), places=12
        )
        pd.testing.assert_series_equal(
            first.scale.loc[second.scale.index], second.scale
        )

    def test_a_full_sample_fit_is_refused_unless_asked_for_explicitly(self) -> None:
        book = synthetic_returns().mean(axis=1)
        with self.assertRaises(ValueError):
            moreira_muir_scale(book, fit_window=(book.index[0], book.index[-1]))
        descriptive = moreira_muir_scale(
            book,
            fit_window=(book.index[0], book.index[-1]),
            full_sample_descriptive_fit=True,
        )
        self.assertEqual(descriptive.constant.status, ESTIMATED)

    def test_a_short_fit_window_is_not_estimable(self) -> None:
        book = synthetic_returns().mean(axis=1)
        result = moreira_muir_scale(
            book, fit_window=(book.index[0], book.index[10]), window=21
        )
        self.assertEqual(result.constant.status, NOT_ESTIMABLE)
        self.assertIsNone(result.constant.value)
        self.assertTrue(result.scale.isna().all())

    def test_invalid_arguments_are_refused(self) -> None:
        book = synthetic_returns(periods=400).mean(axis=1)
        with self.assertRaises(ValueError):
            barroso_santa_clara_scale(book, volatility_budget_annualized=0.0)
        with self.assertRaises(ValueError):
            barroso_santa_clara_scale(book, min_scale=1.0, max_scale=0.5)
        with self.assertRaises(ValueError):
            moreira_muir_scale(book, fit_window=(book.index[10], book.index[0]))
        with self.assertRaises(ValueError):
            moreira_muir_scale(book, fit_window="2010-01-01")  # type: ignore[arg-type]


class DiversificationTests(unittest.TestCase):
    def test_the_absorption_ratio_of_a_rank_one_panel_is_one(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=400)
        rng = np.random.default_rng(4)
        factor = rng.normal(0.0, 0.01, len(index))
        frame = pd.DataFrame(
            {f"S{i}": factor * (1.0 + 0.1 * i) for i in range(6)}, index=index
        )
        ratio = absorption_ratio(frame, window=126, components=1)
        self.assertAlmostEqual(float(ratio.dropna().iloc[0]), 1.0, places=10)

    def test_independent_columns_absorb_roughly_their_share(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=4000)
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(
            rng.normal(0.0, 0.01, (len(index), 10)),
            index=index,
            columns=[f"S{i}" for i in range(10)],
        )
        ratio = absorption_ratio(frame, window=1000, components=2)
        self.assertLess(float(ratio.dropna().mean()), 0.45)
        self.assertGreater(float(ratio.dropna().mean()), 0.20)

    def test_the_ratio_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()
        full = absorption_ratio(returns, window=252)
        truncated = absorption_ratio(returns.iloc[:1800], window=252)
        pd.testing.assert_series_equal(full.loc[truncated.index], truncated)

    def test_the_standardized_shift_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()
        ratio = absorption_ratio(returns, window=252)
        full = absorption_ratio_shift(ratio)
        truncated = absorption_ratio_shift(ratio.iloc[:2200])
        pd.testing.assert_series_equal(full.loc[truncated.index], truncated)

    def test_the_stress_state_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()
        ratio = absorption_ratio(returns, window=252)
        full = systemic_stress_state(absorption_ratio_shift(ratio))
        truncated = systemic_stress_state(absorption_ratio_shift(ratio.iloc[:2200]))
        assert_labels_match(self, full.loc[truncated.index], truncated)
        self.assertIn(STRESS, set(full.dropna().unique()))

    def test_the_correlation_series_is_truncation_invariant_in_both_modes(self) -> None:
        returns = synthetic_returns()
        for drop in (True, False):
            with self.subTest(drop_zero_observations=drop):
                full = mean_pairwise_correlation(
                    returns, window=63, drop_zero_observations=drop
                )
                truncated = mean_pairwise_correlation(
                    returns.iloc[:1800], window=63, drop_zero_observations=drop
                )
                pd.testing.assert_series_equal(full.loc[truncated.index], truncated)

    def test_the_two_correlation_modes_differ_only_on_exact_zeros(self) -> None:
        # A zero asset return is a real observation.  The house function maps it
        # to a missing one, which is right for P&L and wrong for prices, so the
        # size of the effect is measured rather than assumed away.
        returns = synthetic_returns(periods=900)
        identical = mean_pairwise_correlation(returns, window=63)
        retained = mean_pairwise_correlation(
            returns, window=63, drop_zero_observations=False
        )
        pd.testing.assert_series_equal(identical, retained)
        quantized = returns.copy()
        quantized.iloc[::7, 0] = 0.0
        left = mean_pairwise_correlation(quantized, window=63)
        right = mean_pairwise_correlation(
            quantized, window=63, drop_zero_observations=False
        )
        self.assertGreater(float((left - right).abs().max()), 0.0)

    def test_the_forbes_rigobon_adjustment_is_analytic(self) -> None:
        unchanged = forbes_rigobon_adjusted_correlation(0.6, 1.0)
        self.assertEqual(unchanged.status, ESTIMATED)
        self.assertAlmostEqual(float(unchanged.value), 0.6, places=12)
        adjusted = forbes_rigobon_adjusted_correlation(0.6, 4.0)
        expected = 0.6 / math.sqrt(1.0 + 3.0 * (1.0 - 0.36))
        self.assertAlmostEqual(float(adjusted.value), expected, places=12)
        self.assertLess(float(adjusted.value), 0.6)
        for arguments in ((1.5, 2.0), (0.5, 0.0), (0.5, float("nan")), (True, 2.0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                forbes_rigobon_adjusted_correlation(*arguments)

    def test_invalid_arguments_are_refused(self) -> None:
        returns = synthetic_returns(periods=400)
        for kwargs in ({"window": 2}, {"components": 0}, {"components": 99}, {"lag": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                absorption_ratio(returns, **kwargs)
        with self.assertRaises(ValueError):
            absorption_ratio_shift(
                absorption_ratio(returns, window=63), fast=300, slow=252
            )


class CrisisStateTests(unittest.TestCase):
    def test_the_bear_indicator_excludes_the_decision_month(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=900)
        prices = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
        # A rising market is never a bear market, whatever the current month did.
        self.assertTrue(
            (daniel_moskowitz_bear_state(prices).dropna() == NOT_BEAR).all()
        )
        falling = pd.Series(np.linspace(200.0, 100.0, len(index)), index=index)
        self.assertTrue(
            (daniel_moskowitz_bear_state(falling).dropna() == BEAR).all()
        )

    def test_the_bear_state_is_truncation_invariant(self) -> None:
        prices = synthetic_prices()["S0"]
        full = daniel_moskowitz_bear_state(prices)
        truncated = daniel_moskowitz_bear_state(prices.iloc[:1800])
        assert_labels_match(self, full.loc[truncated.index], truncated)

    def test_the_market_state_is_truncation_invariant(self) -> None:
        prices = synthetic_prices()["S0"]
        full = cooper_gutierrez_hameed_state(prices)
        truncated = cooper_gutierrez_hameed_state(prices.iloc[:2000])
        assert_labels_match(self, full.loc[truncated.index], truncated)

    def test_the_high_volatility_state_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()["S0"]
        full = high_volatility_state(returns, minimum_history=252, history=756)
        truncated = high_volatility_state(
            returns.iloc[:1800], minimum_history=252, history=756
        )
        assert_labels_match(self, full.loc[truncated.index], truncated)

    def test_panic_is_the_conjunction_and_missing_where_either_is(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=6)
        bear = pd.Series(
            [BEAR, BEAR, NOT_BEAR, BEAR, pd.NA, NOT_BEAR], index=index, dtype="string"
        )
        volatility = pd.Series(
            [HIGH, LOW, HIGH, pd.NA, HIGH, MIDDLE], index=index, dtype="string"
        )
        state = panic_state(bear, volatility)
        self.assertEqual(
            [None if pd.isna(v) else v for v in state],
            [PANIC, NOT_PANIC, NOT_PANIC, None, None, NOT_PANIC],
        )

    def test_panic_requires_one_index(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=4)
        bear = pd.Series([BEAR] * 4, index=index, dtype="string")
        other = pd.Series([HIGH] * 4, index=pd.bdate_range("2011-01-03", periods=4),
                          dtype="string")
        with self.assertRaises(ValueError):
            panic_state(bear, other)


class MarkovSwitchingTests(unittest.TestCase):
    def generated(self, periods: int = 900, seed: int = 21) -> tuple[pd.Series, np.ndarray]:
        rng = np.random.default_rng(seed)
        index = pd.bdate_range("2005-01-03", periods=periods, freq="W-FRI")
        transition = np.array([[0.94, 0.06], [0.02, 0.98]])
        means = np.array([-0.010, 0.004])
        deviations = np.array([0.045, 0.012])
        states = np.empty(periods, dtype=int)
        states[0] = 1
        for position in range(1, periods):
            states[position] = rng.choice(
                2, p=transition[states[position - 1]]
            )
        values = rng.normal(means[states], deviations[states])
        return pd.Series(values, index=index), states

    def test_the_weekly_series_never_reports_an_open_week(self) -> None:
        returns = synthetic_returns(periods=600)
        weekly = weekly_equal_weight_log_returns(returns)
        self.assertGreater(len(weekly), 100)
        self.assertTrue(weekly.index.is_monotonic_increasing)
        self.assertLess(weekly.index[-1], returns.index[-1] + pd.Timedelta(days=7))

    def test_the_weekly_series_is_truncation_invariant(self) -> None:
        returns = synthetic_returns()
        full = weekly_equal_weight_log_returns(returns)
        truncated = weekly_equal_weight_log_returns(returns.iloc[:1803])
        pd.testing.assert_series_equal(full.loc[truncated.index], truncated)

    def test_the_fit_recovers_the_generating_parameters(self) -> None:
        observations, states = self.generated()
        boundary = observations.index[700]
        fit = fit_markov_switching(
            observations, fit_window=(observations.index[0], boundary)
        )
        self.assertEqual(fit.status, ESTIMATED)
        self.assertTrue(fit.converged)
        self.assertLess(fit.means[0], fit.means[1])
        self.assertAlmostEqual(fit.means[0], -0.010, delta=0.006)
        self.assertAlmostEqual(fit.means[1], 0.004, delta=0.004)
        self.assertAlmostEqual(math.sqrt(fit.variances[0]), 0.045, delta=0.015)
        self.assertAlmostEqual(math.sqrt(fit.variances[1]), 0.012, delta=0.006)
        self.assertGreater(fit.transition[0][0], 0.7)
        self.assertGreater(fit.transition[1][1], 0.9)
        labels = markov_switching_state(observations, fit)
        agreement = np.mean(
            (labels.shift(-1).dropna() == "Low mean").to_numpy()
            == (states[: len(labels) - 1] == 0)
        )
        self.assertGreater(float(agreement), 0.85)

    def test_the_filter_is_truncation_invariant_under_frozen_parameters(self) -> None:
        # T1.  This is what catches a smoothed probability: a series conditioned
        # on the whole sample changes when the sample is cut.
        observations, _ = self.generated()
        fit = fit_markov_switching(
            observations, fit_window=(observations.index[0], observations.index[600])
        )
        full = markov_filtered_probabilities(observations, fit)
        truncated = markov_filtered_probabilities(observations.iloc[:750], fit)
        pd.testing.assert_frame_equal(full.iloc[:750], truncated)
        state_full = markov_switching_state(observations, fit)
        state_cut = markov_switching_state(observations.iloc[:750], fit)
        assert_labels_match(self, state_full.iloc[:750], state_cut)

    def test_the_fit_is_truncation_invariant_past_the_boundary(self) -> None:
        # T2.  Appending data after the boundary must not move the parameters.
        # It is a separate property from T1 and neither implies the other.
        observations, _ = self.generated()
        boundary = observations.index[600]
        early = fit_markov_switching(
            observations.iloc[:700], fit_window=(observations.index[0], boundary)
        )
        late = fit_markov_switching(
            observations, fit_window=(observations.index[0], boundary)
        )
        self.assertEqual(early.means, late.means)
        self.assertEqual(early.variances, late.variances)
        self.assertEqual(early.transition, late.transition)
        self.assertEqual(early.log_likelihood, late.log_likelihood)

    def test_the_fit_is_deterministic_and_fold_indexed(self) -> None:
        observations, _ = self.generated()
        window = (observations.index[0], observations.index[600])
        first = fit_markov_switching(observations, fit_window=window, fold_index=3)
        again = fit_markov_switching(observations, fit_window=window, fold_index=3)
        self.assertEqual(first.means, again.means)
        self.assertEqual(first.transition, again.transition)

    def test_states_are_anchored_by_ascending_mean(self) -> None:
        observations, _ = self.generated()
        fit = fit_markov_switching(
            observations, fit_window=(observations.index[0], observations.index[600])
        )
        self.assertEqual(list(fit.means), sorted(fit.means))
        probabilities = markov_filtered_probabilities(observations, fit)
        self.assertEqual(list(probabilities.columns), ["Low mean", "High mean"])
        np.testing.assert_allclose(
            probabilities.sum(axis=1).to_numpy(), 1.0, rtol=1e-10
        )

    def test_a_full_sample_fit_is_refused_unless_asked_for_explicitly(self) -> None:
        observations, _ = self.generated()
        with self.assertRaises(ValueError):
            fit_markov_switching(
                observations,
                fit_window=(observations.index[0], observations.index[-1]),
            )
        descriptive = fit_markov_switching(
            observations,
            fit_window=(observations.index[0], observations.index[-1]),
            full_sample_descriptive_fit=True,
        )
        self.assertEqual(descriptive.status, ESTIMATED)

    def test_a_short_window_yields_a_not_estimable_fit(self) -> None:
        observations, _ = self.generated()
        fit = fit_markov_switching(
            observations, fit_window=(observations.index[0], observations.index[10])
        )
        self.assertEqual(fit.status, NOT_ESTIMABLE)
        self.assertIsNone(fit.log_likelihood)
        self.assertEqual(fit.means, ())
        self.assertEqual(fit.log_likelihood_result().status, NOT_ESTIMABLE)
        self.assertTrue(
            markov_filtered_probabilities(observations, fit).isna().to_numpy().all()
        )
        self.assertTrue(markov_switching_state(observations, fit).isna().all())

    def test_the_fit_record_refuses_an_inconsistent_state(self) -> None:
        base = {
            "status": ESTIMATED,
            "reason": "test",
            "states": 2,
            "observations": 100,
            "fit_start": "2010-01-01",
            "fit_end": "2012-01-01",
            "means": (-0.01, 0.01),
            "variances": (1e-4, 1e-4),
            "transition": ((0.9, 0.1), (0.1, 0.9)),
            "initial": (0.5, 0.5),
            "log_likelihood": 10.0,
            "iterations": 5,
            "converged": True,
        }
        MarkovSwitchingFit(**base)
        for override in (
            {"means": (0.01, -0.01)},
            {"variances": (0.0, 1e-4)},
            {"transition": ((0.9, 0.2), (0.1, 0.9))},
            {"initial": (0.4, 0.4)},
            {"status": "MAYBE"},
            {"states": 1},
            {"log_likelihood": None},
            {"converged": 1},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                MarkovSwitchingFit(**{**base, **override})
        with self.assertRaises(ValueError):
            MarkovSwitchingFit(
                **{**base, "status": NOT_ESTIMABLE, "log_likelihood": 1.0}
            )

    def test_invalid_arguments_are_refused(self) -> None:
        observations, _ = self.generated(periods=300)
        window = (observations.index[0], observations.index[200])
        for kwargs in (
            {"states": 1},
            {"states": True},
            {"max_iterations": 0},
            {"restarts": 0},
            {"fold_index": -1},
            {"tolerance": 0.0},
            {"variance_floor": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_markov_switching(observations, fit_window=window, **kwargs)


class CombinedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = synthetic_prices()
        cls.config = RegimeConfig(
            volatility_history=756,
            volatility_minimum_history=252,
            correlation_history=756,
        )
        cls.states = combined_regime_states(
            cls.prices, equity_ticker="S0", config=cls.config
        )

    def test_every_declared_state_is_present_and_labelled_somewhere(self) -> None:
        expected = {
            "Equity trend state",
            "Equity momentum state",
            "Book volatility state",
            "Equity volatility state",
            "Diversification state",
            "Absorption state",
            "Systemic stress state",
            "Equity bear state",
            "Long market state",
            "Panic state",
        }
        self.assertEqual(set(self.states.columns), expected)
        self.assertTrue(self.states.index.equals(self.prices.index))
        for column in self.states.columns:
            with self.subTest(column=column):
                self.assertGreater(int(self.states[column].notna().sum()), 200)

    def test_sleeve_states_are_available_on_request(self) -> None:
        detailed = combined_regime_states(
            self.prices, equity_ticker="S0", config=self.config,
            include_sleeve_states=True,
        )
        self.assertIn("Faber trend S3", detailed.columns)
        self.assertIn("Momentum S3", detailed.columns)
        self.assertEqual(len(detailed.columns), 10 + 2 * SLEEVES)

    def test_the_combined_frame_is_truncation_invariant(self) -> None:
        truncated = combined_regime_states(
            self.prices.iloc[:1800], equity_ticker="S0", config=self.config
        )
        for column in truncated.columns:
            with self.subTest(column=column):
                assert_labels_match(
                    self, self.states.loc[truncated.index, column], truncated[column]
                )

    def test_the_state_report_counts_episodes_and_shares(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=10)
        column = pd.Series(
            [pd.NA, LOW, LOW, HIGH, HIGH, HIGH, LOW, pd.NA, LOW, LOW],
            index=index,
            dtype="string",
        )
        report = regime_state_report(pd.DataFrame({"R": column}))
        low = report[report["State"] == LOW].iloc[0]
        high = report[report["State"] == HIGH].iloc[0]
        self.assertEqual(int(low["Sessions"]), 5)
        self.assertEqual(int(low["Episodes"]), 3)
        self.assertAlmostEqual(float(low["Share of labelled sessions"]), 5 / 8)
        self.assertAlmostEqual(float(low["Share of all sessions"]), 0.5)
        self.assertEqual(int(low["Longest episode (sessions)"]), 2)
        self.assertAlmostEqual(
            float(low["Mean episode duration (sessions)"]), 5 / 3, places=12
        )
        self.assertEqual(int(high["Episodes"]), 1)
        self.assertEqual(int(high["Longest episode (sessions)"]), 3)
        self.assertTrue((report["Limitation"] == SURVIVORSHIP_LIMITATION).all())

    def test_the_transition_matrix_counts_only_labelled_adjacent_pairs(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=7)
        column = pd.Series(
            [LOW, LOW, HIGH, pd.NA, HIGH, LOW, LOW], index=index, dtype="string"
        )
        counts = regime_transition_matrix(column, as_probabilities=False)
        self.assertEqual(int(counts.loc[LOW, LOW]), 2)
        self.assertEqual(int(counts.loc[LOW, HIGH]), 1)
        self.assertEqual(int(counts.loc[HIGH, LOW]), 1)
        self.assertEqual(int(counts.loc[HIGH, HIGH]), 0)
        self.assertEqual(int(counts.to_numpy().sum()), 4)
        probabilities = regime_transition_matrix(column)
        np.testing.assert_allclose(
            probabilities.sum(axis=1).to_numpy(), 1.0, rtol=1e-12
        )

    def test_the_transition_report_covers_every_regime(self) -> None:
        report = regime_transition_report(self.states)
        self.assertEqual(
            set(report["Regime"]), set(self.states.columns)
        )
        self.assertTrue((report["Transitions"] >= 0).all())
        self.assertTrue((report["Limitation"] == SURVIVORSHIP_LIMITATION).all())
        for regime, group in report.groupby("Regime"):
            with self.subTest(regime=regime):
                for _, rows in group.groupby("From state"):
                    self.assertAlmostEqual(
                        float(rows["Transition probability"].sum()), 1.0, places=12
                    )

    def test_reports_carry_no_banned_or_ranking_column_name(self) -> None:
        frames = [
            regime_state_report(self.states),
            regime_transition_report(self.states),
            regime_transition_matrix(self.states["Book volatility state"]),
            self.states,
        ]
        for position, frame in enumerate(frames):
            for column in frame.columns:
                lowered = str(column).lower()
                for banned in BANNED_SUBSTRINGS:
                    with self.subTest(frame=position, column=column, banned=banned):
                        self.assertNotIn(banned, lowered)

    def test_config_validation_refuses_out_of_range_fields(self) -> None:
        RegimeConfig()
        for override in (
            {"faber_months": 0},
            {"faber_months": True},
            {"lag": 0},
            {"correlation_minimum_markets": 1},
            {"correlation_minimum_coverage": 0.0},
            {"correlation_minimum_coverage": 1.5},
            {"absorption_fast": 300},
            {"volatility_history": 100},
            {"volatility_minimum_history": 1},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                RegimeConfig(**override)

    def test_the_builder_refuses_an_absent_equity_sleeve(self) -> None:
        with self.assertRaises(ValueError):
            combined_regime_states(self.prices, equity_ticker="NOPE")
        with self.assertRaises(ValueError):
            combined_regime_states(self.prices, config="not a config")  # type: ignore[arg-type]


@unittest.skipUnless(
    REAL_PRICE_DIR.is_dir(), "Supplied DELTA1 ETF data directory is not available"
)
class SuppliedEtfRegimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = load_panel(
            UNIVERSE_TICKERS, start="2006-02-03", end="2018-12-31", data_dir=DATA_DIR
        )
        cls.states = combined_regime_states(cls.panel.adjusted_close)

    def test_the_states_cover_the_panel_calendar(self) -> None:
        self.assertTrue(self.states.index.equals(self.panel.sessions))
        self.assertEqual(len(self.states), 3249)
        counts = self.states.notna().sum()
        self.assertEqual(int(counts["Equity trend state"]), 3020)
        for column in self.states.columns:
            with self.subTest(column=column):
                self.assertGreater(int(counts[column]), 2400)

    def test_the_only_bear_market_in_the_window_is_the_crisis(self) -> None:
        # The disclosure that must travel with any out-of-sample count: the
        # 2013-2018 block contains no full equity bear market, so the panic
        # state the design is partly built for is barely exercised there.
        bear = self.states["Equity bear state"]
        bear_sessions = bear.index[bear.eq(BEAR).fillna(False).to_numpy(dtype=bool)]
        self.assertEqual(bear_sessions.min(), pd.Timestamp("2008-11-03"))
        self.assertEqual(bear_sessions.max(), pd.Timestamp("2010-10-29"))
        self.assertEqual(len(bear_sessions), 502)
        sealed = bear.loc["2013-01-01":"2018-12-31"]
        self.assertEqual(int(sealed.eq(BEAR).sum()), 0)
        panic = self.states["Panic state"].loc["2013-01-01":"2018-12-31"]
        self.assertEqual(int(panic.eq(PANIC).sum()), 0)

    def test_the_absorption_ratio_is_high_and_coarse_on_eleven_sleeves(self) -> None:
        ratio = absorption_ratio(self.panel.returns).dropna()
        self.assertEqual(len(ratio), 2996)
        self.assertGreater(float(ratio.mean()), 0.70)
        self.assertLess(float(ratio.mean()), 0.85)
        self.assertGreaterEqual(float(ratio.min()), 0.0)
        self.assertLessEqual(float(ratio.max()), 1.0)

    def test_the_zero_return_handling_moves_the_correlation_series(self) -> None:
        dropped = mean_pairwise_correlation(self.panel.returns)
        retained = mean_pairwise_correlation(
            self.panel.returns, drop_zero_observations=False
        )
        difference = (dropped - retained).abs().dropna()
        self.assertGreater(float(difference.max()), 0.01)
        self.assertLess(float(difference.median()), 0.01)

    def test_the_markov_state_is_fitted_in_fold_and_filtered(self) -> None:
        weekly = weekly_equal_weight_log_returns(self.panel.returns)
        fit = fit_markov_switching(weekly, fit_window=("2006-02-06", "2012-12-31"))
        self.assertEqual(fit.status, ESTIMATED)
        self.assertTrue(fit.converged)
        self.assertLess(fit.means[0], 0.0)
        self.assertGreater(fit.means[1], 0.0)
        self.assertGreater(fit.variances[0], fit.variances[1])
        self.assertGreater(fit.transition[0][0], 0.8)
        self.assertGreater(fit.transition[1][1], 0.9)
        state = markov_switching_state(weekly, fit, sessions=self.panel.sessions)
        self.assertGreater(int(state.eq("Low mean").sum()), 100)
        self.assertGreater(int(state.eq("High mean").sum()), 2000)

    def test_the_real_states_are_truncation_invariant(self) -> None:
        early = load_panel(
            UNIVERSE_TICKERS, start="2006-02-03", end="2015-06-30", data_dir=DATA_DIR
        )
        truncated = combined_regime_states(early.adjusted_close)
        for column in truncated.columns:
            with self.subTest(column=column):
                assert_labels_match(
                    self, self.states.loc[truncated.index, column], truncated[column]
                )

    def test_the_reports_are_complete_and_disclosed(self) -> None:
        report = regime_state_report(self.states)
        self.assertEqual(set(report["Regime"]), set(self.states.columns))
        self.assertTrue((report["Limitation"] == SURVIVORSHIP_LIMITATION).all())
        for regime, group in report.groupby("Regime"):
            with self.subTest(regime=regime):
                self.assertAlmostEqual(
                    float(group["Share of labelled sessions"].sum()), 1.0, places=12
                )


if __name__ == "__main__":
    unittest.main()
