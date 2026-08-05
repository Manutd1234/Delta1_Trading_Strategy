"""Tests for the paired inference module."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from delta1_strategy.research.inference import (
    ESTIMATED,
    NOT_ESTIMABLE,
    InferenceResult,
    deflated_sharpe_ratio,
    hac_standard_error,
    minimum_detectable_effect,
    paired_block_bootstrap_differential,
    paired_hac_report,
    promotion_support,
)


def series(values: np.ndarray, start: str = "2000-01-03") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


class TestInferenceResult(unittest.TestCase):
    def test_not_estimable_cannot_carry_a_value(self) -> None:
        with self.assertRaises(ValueError):
            InferenceResult("s", NOT_ESTIMABLE, 0.99, "reason")

    def test_estimated_requires_a_finite_value(self) -> None:
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                InferenceResult("s", ESTIMATED, value, "reason")


class TestPairedBootstrap(unittest.TestCase):
    def test_identical_series_have_exactly_zero_differential(self) -> None:
        rng = np.random.default_rng(3)
        path = series(rng.normal(0.0004, 0.005, 900))
        report = paired_block_bootstrap_differential(
            path, path.copy(), comparison="identity", block_lengths=(63,), samples=200
        )
        np.testing.assert_allclose(report["Point estimate"].to_numpy(), 0.0)
        np.testing.assert_allclose(report["Bootstrap standard error"].to_numpy(), 0.0)

    def test_pairing_fails_closed_on_mismatched_indexes(self) -> None:
        rng = np.random.default_rng(4)
        path = series(rng.normal(0.0004, 0.005, 400))
        with self.assertRaises(ValueError):
            paired_block_bootstrap_differential(
                path, path.iloc[:-3], comparison="misaligned"
            )

    def test_known_constant_uplift_is_recovered(self) -> None:
        rng = np.random.default_rng(5)
        incumbent = series(rng.normal(0.0004, 0.005, 1200))
        challenger = incumbent + 0.02 / 252
        report = paired_block_bootstrap_differential(
            incumbent,
            challenger,
            comparison="uplift",
            block_lengths=(63,),
            samples=300,
        )
        mean_row = report.loc[
            report["Statistic"] == "Annualized mean differential"
        ].iloc[0]
        self.assertAlmostEqual(float(mean_row["Point estimate"]), 0.02, places=6)

    def test_pairing_beats_the_unpaired_standard_error(self) -> None:
        # The whole justification for the module: a shared index draw cancels
        # the common history, so the differential is far better determined
        # than either strategy's own statistic.
        rng = np.random.default_rng(6)
        incumbent = series(rng.normal(0.0004, 0.005, 2000))
        challenger = incumbent + series(
            rng.normal(0.0, 0.0005, 2000), start="2000-01-03"
        )
        report = paired_block_bootstrap_differential(
            incumbent,
            challenger,
            comparison="tight",
            block_lengths=(63,),
            samples=400,
        )
        paired = float(
            report.loc[
                report["Statistic"] == "Sharpe differential",
                "Bootstrap standard error",
            ].iloc[0]
        )
        self.assertLess(paired, 0.10)

    def test_report_carries_no_verdict_column(self) -> None:
        rng = np.random.default_rng(7)
        path = series(rng.normal(0.0004, 0.005, 400))
        report = paired_block_bootstrap_differential(
            path, path + 1e-5, comparison="labels", block_lengths=(63,), samples=100
        )
        for column in report.columns:
            lowered = column.lower()
            for forbidden in ("target", "pass", "fail", "verdict"):
                self.assertNotIn(forbidden, lowered)
        self.assertFalse(bool(report["Selection adjusted"].any()))


class TestClosedFormAgreement(unittest.TestCase):
    def test_closed_form_tracks_the_bootstrap(self) -> None:
        rng = np.random.default_rng(8)
        incumbent = series(rng.normal(0.0004, 0.005, 2500))
        challenger = incumbent + rng.normal(0.0, 0.0008, 2500)
        report = paired_block_bootstrap_differential(
            incumbent,
            challenger,
            comparison="agreement",
            block_lengths=(63,),
            samples=500,
        )
        row = report.loc[report["Statistic"] == "Sharpe differential"].iloc[0]
        preflight = minimum_detectable_effect(
            float(row["Tracking error (annualized)"]),
            float(incumbent.std(ddof=0) * np.sqrt(252)),
            len(incumbent) / 252,
        )
        self.assertAlmostEqual(
            float(row["Bootstrap standard error"]),
            float(preflight["Standard error of Sharpe differential"]),
            delta=0.02,
        )


class TestHac(unittest.TestCase):
    def test_positive_autocorrelation_widens_the_standard_error(self) -> None:
        rng = np.random.default_rng(9)
        white = rng.normal(0.0, 0.01, 3000)
        correlated = white.copy()
        for index in range(1, correlated.size):
            correlated[index] += 0.6 * correlated[index - 1]
        self.assertGreater(
            hac_standard_error(correlated), hac_standard_error(white)
        )

    def test_report_recovers_a_known_mean(self) -> None:
        rng = np.random.default_rng(10)
        incumbent = series(rng.normal(0.0004, 0.005, 800))
        report = paired_hac_report(
            incumbent, incumbent + 0.01 / 252, comparison="known"
        )
        self.assertAlmostEqual(float(report["Point estimate"]), 0.01, places=6)


class TestDeflatedSharpe(unittest.TestCase):
    def test_missing_trial_count_is_not_estimable(self) -> None:
        result = deflated_sharpe_ratio(
            1.59,
            trial_count=None,
            trial_sharpe_variance=None,
            sessions=6523,
            skewness=0.0,
            excess_kurtosis=2.3,
        )
        self.assertEqual(result.status, NOT_ESTIMABLE)
        self.assertIsNone(result.value)

    def test_declared_family_is_estimable(self) -> None:
        result = deflated_sharpe_ratio(
            1.59,
            trial_count=5,
            trial_sharpe_variance=0.04,
            sessions=6523,
            skewness=0.0,
            excess_kurtosis=2.3,
        )
        self.assertEqual(result.status, ESTIMATED)
        self.assertTrue(0.0 <= float(result.value) <= 1.0)

    def test_more_trials_never_increase_the_deflated_probability(self) -> None:
        def probability(trials: int) -> float:
            return float(
                deflated_sharpe_ratio(
                    1.59,
                    trial_count=trials,
                    trial_sharpe_variance=0.04,
                    sessions=6523,
                    skewness=0.0,
                    excess_kurtosis=2.3,
                ).value
            )

        self.assertGreaterEqual(probability(2), probability(200))


class TestPromotionSupport(unittest.TestCase):
    def test_reused_history_can_never_support_promotion(self) -> None:
        result = deflated_sharpe_ratio(
            1.59,
            trial_count=None,
            trial_sharpe_variance=None,
            sessions=6523,
            skewness=0.0,
            excess_kurtosis=2.3,
        )
        support = promotion_support(result, window_label="1990-2014")
        self.assertFalse(bool(support["Promotion supported"]))
        self.assertIn("not_estimable", str(support["Promotion reason"]))


if __name__ == "__main__":
    unittest.main()
