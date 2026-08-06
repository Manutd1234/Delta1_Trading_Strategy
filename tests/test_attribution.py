from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from delta1_strategy.research.attribution import (
    ACTIVE,
    INERT,
    BoundSpec,
    bound_activity_report,
    conditional_exposure_diagnostic,
    drawdown_anatomy,
    drawdown_episodes,
    episode_attribution,
    magnitude_bound_specs,
    trailing_mean_pairwise_correlation,
    uniform_rescale_frontier,
)


def series(values: list[float], start: str = "2000-01-03") -> pd.Series:
    return pd.Series(
        values, index=pd.bdate_range(start, periods=len(values)), dtype=float
    )


@dataclass
class StubResult:
    daily: pd.DataFrame


@dataclass
class StubConfig:
    max_risk_scalar: float = 2.0
    min_risk_scalar: float = 0.25
    max_gross_notional_multiple: float | None = 5.0


class BoundSpecTests(unittest.TestCase):
    def test_invalid_specifications_are_rejected(self) -> None:
        scalar = series([1.0, 1.1])
        with self.assertRaises(ValueError):
            BoundSpec(name="", limit=1.0, side="upper", series=scalar)
        with self.assertRaises(ValueError):
            BoundSpec(name="x", limit=1.0, side="sideways", series=scalar)
        with self.assertRaises(ValueError):
            BoundSpec(name="x", limit=float("inf"), side="upper", series=scalar)
        with self.assertRaises(ValueError):
            BoundSpec(name="x", limit=True, side="upper", series=scalar)
        with self.assertRaises(ValueError):
            BoundSpec(name="x", limit=1.0, side="upper", series=[1.0, 2.0])
        with self.assertRaises(ValueError):
            BoundSpec(name="x", limit=1.0, side="upper", series=scalar, tolerance=-1.0)


class BoundActivityTests(unittest.TestCase):
    def test_a_bound_outside_the_realized_range_is_reported_inert(self) -> None:
        scalar = series([0.30, 0.80, 1.20, 1.85])
        report = bound_activity_report(
            [
                BoundSpec(name="ceiling", limit=2.0, side="upper", series=scalar),
                BoundSpec(name="floor", limit=0.25, side="lower", series=scalar),
            ]
        )
        self.assertEqual(list(report["Activity"]), [INERT, INERT])
        self.assertEqual(list(report["Binding observations"]), [0, 0])
        # Headroom is signed away from the limit, so both are strictly positive
        # when the series never reaches either end.
        self.assertAlmostEqual(float(report.loc[0, "Headroom to limit"]), 0.15)
        self.assertAlmostEqual(float(report.loc[1, "Headroom to limit"]), 0.05)

    def test_a_binding_bound_is_counted_exactly(self) -> None:
        scalar = series([0.25, 0.50, 2.00, 2.00, 1.00])
        report = bound_activity_report(
            [
                BoundSpec(name="ceiling", limit=2.0, side="upper", series=scalar),
                BoundSpec(name="floor", limit=0.25, side="lower", series=scalar),
            ]
        )
        self.assertEqual(list(report["Activity"]), [ACTIVE, ACTIVE])
        self.assertEqual(int(report.loc[0, "Binding observations"]), 2)
        self.assertAlmostEqual(float(report.loc[0, "Binding share"]), 0.4)
        self.assertEqual(int(report.loc[1, "Binding observations"]), 1)
        self.assertAlmostEqual(float(report.loc[0, "Headroom to limit"]), 0.0)

    def test_empty_inputs_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            bound_activity_report([])
        with self.assertRaises(ValueError):
            bound_activity_report(
                [
                    BoundSpec(
                        name="x",
                        limit=1.0,
                        side="upper",
                        series=pd.Series(dtype=float),
                    )
                ]
            )

    def test_engine_bound_set_uses_the_recorded_gross_limit_intervention(self) -> None:
        # basis_momentum divides by `cap`, so signal_cap is the slope of the
        # sleeve map rather than a ceiling on it.  Tabulating it beside genuine
        # ceilings would invite the inference that lowering it cuts exposure,
        # when it raises exposure by saturating sooner.
        daily = pd.DataFrame(
            {
                "risk_scalar": [0.5, 1.0],
                "gross_notional_multiple": [2.0, 3.0],
                # Only the second session was clipped. Realized gross notional
                # is not itself a binding indicator because integer rounding
                # and market drift can leave it above the configured limit.
                "target_portfolio_limit_scale": [1.0, 0.8],
            },
            index=pd.bdate_range("2000-01-03", periods=2),
        )
        specs = magnitude_bound_specs(StubResult(daily), StubConfig())
        self.assertNotIn("signal_cap", {spec.name for spec in specs})
        self.assertEqual(
            {spec.name for spec in specs},
            {
                "max_risk_scalar",
                "min_risk_scalar",
                "max_gross_notional_multiple",
            },
        )
        report = bound_activity_report(specs)
        gross = report.set_index("Bound").loc["max_gross_notional_multiple"]
        self.assertEqual(int(gross["Binding observations"]), 1)
        self.assertAlmostEqual(float(gross["Binding share"]), 0.5)
        self.assertEqual(
            gross["Binding criterion"],
            "target_portfolio_limit_scale < 1 - 1e-12",
        )


    def test_the_window_restricts_the_observations_counted(self) -> None:
        # result.daily is indexed from the earliest warm-up session, where the
        # book is empty.  Counting those rows would dilute every binding share.
        index = pd.bdate_range("1990-01-01", periods=6)
        daily = pd.DataFrame(
            {
                "risk_scalar": [0.25, 0.25, 1.0, 1.0, 2.0, 2.0],
                "gross_notional_multiple": [0.0, 0.0, 2.0, 2.0, 3.0, 3.0],
            },
            index=index,
        )
        full = bound_activity_report(
            magnitude_bound_specs(StubResult(daily), StubConfig())
        )
        windowed = bound_activity_report(
            magnitude_bound_specs(
                StubResult(daily), StubConfig(), start=index[2], end=index[5]
            )
        )
        self.assertEqual(int(full.loc[0, "Observations"]), 6)
        self.assertEqual(int(windowed.loc[0, "Observations"]), 4)
        # The floor binds only in the excluded warm-up rows.
        self.assertEqual(full.loc[1, "Activity"], ACTIVE)
        self.assertEqual(windowed.loc[1, "Activity"], INERT)
        with self.assertRaises(ValueError):
            magnitude_bound_specs(
                StubResult(daily),
                StubConfig(),
                start="2050-01-01",
                end="2050-12-31",
            )

class DrawdownEpisodeTests(unittest.TestCase):
    def test_episode_depth_and_timing_match_a_hand_computed_path(self) -> None:
        # +10%, then three -10% sessions, then recovery above the old peak.
        returns = series([0.10, -0.10, -0.10, -0.10, 0.20, 0.20])
        episodes = drawdown_episodes(returns, minimum_depth=0.05)
        self.assertEqual(len(episodes), 1)
        row = episodes.iloc[0]
        self.assertAlmostEqual(float(row["Depth"]), 0.9**3 - 1.0, places=12)
        self.assertEqual(int(row["Sessions to trough"]), 3)
        self.assertTrue(bool(row["Recovered"]))

    def test_a_censored_final_episode_is_marked_unrecovered(self) -> None:
        returns = series([0.05, -0.10, -0.10])
        episodes = drawdown_episodes(returns, minimum_depth=0.05)
        self.assertEqual(len(episodes), 1)
        self.assertFalse(bool(episodes.iloc[0]["Recovered"]))
        self.assertEqual(int(episodes.iloc[0]["Sessions to recover"]), -1)

    def test_shallow_episodes_are_filtered_and_bad_thresholds_refused(self) -> None:
        returns = series([0.01, -0.01, 0.01, -0.01])
        self.assertTrue(drawdown_episodes(returns, minimum_depth=0.05).empty)
        with self.assertRaises(ValueError):
            drawdown_episodes(returns, minimum_depth=0.0)
        with self.assertRaises(ValueError):
            drawdown_episodes(returns, minimum_depth=1.0)

    def test_the_high_water_mark_starts_at_initial_capital(self) -> None:
        # A path that only ever falls is in drawdown from its first session,
        # rather than measuring against a peak it never reached.
        returns = series([-0.03] * 4)
        episodes = drawdown_episodes(returns, minimum_depth=0.05)
        self.assertEqual(len(episodes), 1)
        self.assertAlmostEqual(float(episodes.iloc[0]["Depth"]), 0.97**4 - 1.0)


class DrawdownAnatomyTests(unittest.TestCase):
    def test_ratios_separate_a_size_failure_from_an_accuracy_failure(self) -> None:
        rng = np.random.default_rng(20260806)
        calm = rng.normal(0.0008, 0.004, 300)
        # Same volatility, negative drift: an accuracy failure, not a size one.
        slump = rng.normal(-0.0035, 0.004, 300)
        recovery = rng.normal(0.0015, 0.004, 300)
        returns = np.concatenate([calm, slump, recovery])
        index = pd.bdate_range("2000-01-03", periods=returns.size)
        daily = pd.DataFrame(
            {
                "net_return": returns,
                "gross_notional_multiple": np.full(returns.size, 2.5),
                "risk_scalar": np.full(returns.size, 0.8),
            },
            index=index,
        )
        report = drawdown_anatomy(daily, threshold=0.05)
        inside = report.iloc[1]
        # Volatility is unchanged by construction; the hit rate is what moved.
        self.assertAlmostEqual(
            float(inside["Volatility ratio to all sessions"]), 1.0, delta=0.25
        )
        self.assertLess(float(inside["Hit-rate difference to all sessions"]), -0.05)
        self.assertAlmostEqual(
            float(inside["Exposure ratio to all sessions"]), 1.0, places=9
        )

    def test_missing_columns_and_empty_conditional_samples_are_refused(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=5)
        with self.assertRaises(ValueError):
            drawdown_anatomy(pd.DataFrame({"x": [0.0] * 5}, index=index))
        flat = pd.DataFrame({"net_return": [0.001] * 5}, index=index)
        with self.assertRaises(ValueError):
            drawdown_anatomy(flat, threshold=0.05)
        with self.assertRaises(ValueError):
            drawdown_anatomy(flat, threshold=1.5)


class EpisodeAttributionTests(unittest.TestCase):
    def ledger(self) -> pd.DataFrame:
        dates = pd.bdate_range("2000-01-03", periods=3)
        rows = []
        # AA carries the whole loss; BB and CC are flat.  A concentrated loss.
        for date in dates:
            rows.append((date, "AA", "Energy", -0.02))
            rows.append((date, "BB", "Metals", 0.0))
            rows.append((date, "CC", "FX", 0.001))
        return pd.DataFrame(
            rows, columns=["date", "symbol", "asset_class", "net_return_contribution"]
        )

    def test_concentration_is_measured_not_assumed(self) -> None:
        episodes = pd.DataFrame(
            [
                {
                    "Start": pd.Timestamp("2000-01-03"),
                    "Trough": pd.Timestamp("2000-01-05"),
                    "Depth": -0.06,
                    "Sessions to trough": 3,
                }
            ]
        )
        summary, detail = episode_attribution(self.ledger(), episodes)
        row = summary.iloc[0]
        self.assertEqual(int(row["Markets losing"]), 1)
        self.assertEqual(int(row["Markets with activity"]), 2)
        self.assertEqual(row["Worst market"], "AA")
        self.assertAlmostEqual(float(row["Worst market contribution"]), -0.06)
        self.assertAlmostEqual(float(row["Share of loss from worst 3"]), 1.0)
        self.assertEqual(len(detail), 3)
        self.assertEqual(detail.iloc[0]["Symbol"], "AA")

    def test_a_broad_loss_reports_low_concentration(self) -> None:
        dates = pd.bdate_range("2000-01-03", periods=2)
        rows = []
        for date in dates:
            for index in range(10):
                rows.append((date, f"S{index}", "Energy", -0.005))
        ledger = pd.DataFrame(
            rows, columns=["date", "symbol", "asset_class", "net_return_contribution"]
        )
        episodes = pd.DataFrame(
            [
                {
                    "Start": dates[0],
                    "Trough": dates[-1],
                    "Depth": -0.10,
                    "Sessions to trough": 2,
                }
            ]
        )
        summary, _ = episode_attribution(ledger, episodes)
        row = summary.iloc[0]
        self.assertEqual(int(row["Markets losing"]), 10)
        self.assertAlmostEqual(float(row["Breadth of loss"]), 1.0)
        self.assertAlmostEqual(float(row["Share of loss from worst 3"]), 0.3)

    def test_missing_columns_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            episode_attribution(pd.DataFrame({"date": []}), pd.DataFrame())


class UniformRescaleTests(unittest.TestCase):
    def test_drawdown_per_unit_of_volatility_is_the_rescale_invariant(self) -> None:
        rng = np.random.default_rng(7)
        returns = series(list(rng.normal(0.0004, 0.005, 2000)))
        frontier = uniform_rescale_frontier(returns, multipliers=(1.0, 0.75, 0.5))
        ratios = frontier["Drawdown per unit of volatility"].to_numpy()
        # Not exactly constant --- compounding is nonlinear --- but a constant
        # rescale must not move it materially, which is the whole point of
        # reporting it as the line a genuine lever has to beat.
        self.assertLess(float(np.ptp(ratios)) / float(ratios.mean()), 0.06)
        sharpes = frontier["Sharpe (rf=0)"].to_numpy()
        self.assertLess(float(np.ptp(sharpes)), 1e-9)

    def test_volatility_scales_linearly_with_the_multiplier(self) -> None:
        returns = series([0.01, -0.02, 0.015, -0.005, 0.008] * 40)
        frontier = uniform_rescale_frontier(returns, multipliers=(1.0, 0.5))
        full = float(frontier.loc[0, "Annualized volatility"])
        half = float(frontier.loc[1, "Annualized volatility"])
        self.assertAlmostEqual(half, full * 0.5, places=12)

    def test_invalid_multipliers_and_empty_returns_are_refused(self) -> None:
        returns = series([0.01, -0.01])
        with self.assertRaises(ValueError):
            uniform_rescale_frontier(returns, multipliers=())
        with self.assertRaises(ValueError):
            uniform_rescale_frontier(returns, multipliers=(0.0,))
        with self.assertRaises(ValueError):
            uniform_rescale_frontier(returns, multipliers=(float("nan"),))
        with self.assertRaises(ValueError):
            uniform_rescale_frontier(pd.Series(dtype=float))

    def test_cagr_matches_a_hand_computed_compounded_path(self) -> None:
        returns = series([0.01] * 252)
        period_start = returns.index[0] - pd.Timedelta(days=1)
        frontier = uniform_rescale_frontier(
            returns, multipliers=(1.0,), period_start=period_start
        )
        years = (returns.index[-1] - period_start).days / 365.2425
        self.assertAlmostEqual(
            float(frontier.loc[0, "CAGR"]),
            (1.01**252) ** (1.0 / years) - 1.0,
            places=9,
        )
        self.assertAlmostEqual(float(frontier.loc[0, "Elapsed years"]), years)
        self.assertEqual(
            frontier.loc[0, "CAGR convention"],
            "elapsed calendar time from period start",
        )
        self.assertTrue(math.isnan(float(frontier.loc[0, "Calmar"])))


class TrailingCorrelationTests(unittest.TestCase):
    def contributions(self, periods: int = 200, seed: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        common = rng.normal(0, 0.004, periods)
        return pd.DataFrame(
            {
                f"S{i}": 0.5 * common + rng.normal(0, 0.004, periods)
                for i in range(8)
            },
            index=pd.bdate_range("2000-01-03", periods=periods),
        )

    def test_the_series_is_truncation_invariant(self) -> None:
        # The causality proof that matters: values computed on a longer history
        # must be identical on the overlap.  Any forward-looking term --- a
        # centred window, a backward fill, a full-sample normalisation --- would
        # make the earlier values move when later data is appended.
        frame = self.contributions()
        full = trailing_mean_pairwise_correlation(frame, window=21)
        truncated = trailing_mean_pairwise_correlation(frame.iloc[:150], window=21)
        overlap = truncated.dropna().index
        self.assertGreater(len(overlap), 50)
        pd.testing.assert_series_equal(
            full.loc[overlap], truncated.loc[overlap], check_names=False
        )

    def test_the_window_excludes_the_labelled_session_and_lags(self) -> None:
        frame = self.contributions(periods=60)
        series = trailing_mean_pairwise_correlation(frame, window=21, lag=1)
        # window sessions consumed, then one more for the lag.
        self.assertTrue(series.iloc[:22].isna().all())
        self.assertTrue(np.isfinite(series.iloc[22]))
        # A shock inserted at position t must not move the value labelled t.
        shocked = frame.copy()
        shocked.iloc[30] = shocked.iloc[30] * 50.0
        after = trailing_mean_pairwise_correlation(shocked, window=21, lag=1)
        self.assertAlmostEqual(
            float(series.iloc[30]), float(after.iloc[30]), places=12
        )

    def test_thin_windows_yield_no_estimate_rather_than_a_bad_one(self) -> None:
        frame = self.contributions(periods=60).iloc[:, :3]
        series = trailing_mean_pairwise_correlation(
            frame, window=21, minimum_markets=5
        )
        self.assertTrue(series.isna().all())

    def test_invalid_arguments_are_refused(self) -> None:
        frame = self.contributions(periods=30)
        for kwargs in (
            {"window": 1},
            {"minimum_markets": 1},
            {"minimum_coverage": 0.0},
            {"minimum_coverage": 1.5},
            {"lag": 0},
            {"lag": True},
        ):
            with self.assertRaises(ValueError):
                trailing_mean_pairwise_correlation(frame, **kwargs)


class ConditionalExposureTests(unittest.TestCase):
    def test_the_risk_matched_row_isolates_shape_from_scale(self) -> None:
        rng = np.random.default_rng(11)
        periods = 1500
        index = pd.bdate_range("2000-01-03", periods=periods)
        state = pd.Series(rng.random(periods), index=index)
        # Returns are worse exactly where the conditioner is high, so a
        # conditional de-lever should improve the shape, not merely the scale.
        returns = pd.Series(
            np.where(state > 0.8, rng.normal(-0.002, 0.006, periods),
                     rng.normal(0.0009, 0.005, periods)),
            index=index,
        )
        report = conditional_exposure_diagnostic(returns, state, quantile=0.8)
        self.assertEqual(len(report), 3)
        matched = report.iloc[2]
        base = report.iloc[0]
        self.assertAlmostEqual(
            float(matched["Annualized volatility"]),
            float(base["Annualized volatility"]),
            places=9,
        )
        self.assertLess(
            float(matched["Drawdown per unit of volatility"]),
            float(base["Drawdown per unit of volatility"]),
        )

    def test_a_useless_conditioner_does_not_improve_the_shape(self) -> None:
        rng = np.random.default_rng(12)
        periods = 1500
        index = pd.bdate_range("2000-01-03", periods=periods)
        returns = pd.Series(rng.normal(0.0006, 0.005, periods), index=index)
        noise = pd.Series(rng.random(periods), index=index)
        report = conditional_exposure_diagnostic(returns, noise, quantile=0.8)
        base = float(report.iloc[0]["Drawdown per unit of volatility"])
        matched = float(report.iloc[2]["Drawdown per unit of volatility"])
        self.assertLess(abs(matched - base) / base, 0.25)

    def test_the_in_sample_threshold_is_disclosed_on_every_conditioned_row(
        self,
    ) -> None:
        index = pd.bdate_range("2000-01-03", periods=60)
        returns = pd.Series(np.linspace(-0.01, 0.01, 60), index=index)
        state = pd.Series(np.linspace(0, 1, 60), index=index)
        report = conditional_exposure_diagnostic(returns, state, quantile=0.8)
        notes = report["Evidence note"].tolist()
        self.assertNotIn("in sample", notes[0])
        for note in notes[1:]:
            self.assertIn("not promotable", note)

    def test_invalid_arguments_and_unalignable_conditioners_are_refused(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=10)
        returns = pd.Series(np.linspace(-0.01, 0.01, 10), index=index)
        state = pd.Series(np.linspace(0, 1, 10), index=index)
        with self.assertRaises(ValueError):
            conditional_exposure_diagnostic(returns, state, quantile=0.0)
        with self.assertRaises(ValueError):
            conditional_exposure_diagnostic(returns, state, multiplier=1.0)
        with self.assertRaises(ValueError):
            conditional_exposure_diagnostic(
                returns, pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            )


class ReportNamingTests(unittest.TestCase):
    BANNED = ("target", "pass", "fail")

    def test_no_report_column_reads_as_a_promotion_decision(self) -> None:
        scalar = series([0.5, 1.0, 2.0])
        returns = series([0.01, -0.06, 0.02, -0.03, 0.04])
        daily = pd.DataFrame(
            {
                "net_return": returns,
                "gross_notional_multiple": [2.0] * 5,
                "risk_scalar": [0.8] * 5,
            },
            index=returns.index,
        )
        frames = [
            bound_activity_report(
                [BoundSpec(name="c", limit=2.0, side="upper", series=scalar)]
            ),
            drawdown_episodes(returns, minimum_depth=0.02),
            drawdown_anatomy(daily, threshold=0.02),
            uniform_rescale_frontier(returns, multipliers=(1.0, 0.5)),
        ]
        for frame in frames:
            for column in frame.columns:
                for banned in self.BANNED:
                    self.assertNotIn(
                        banned,
                        str(column).lower(),
                        f"{column!r} contains the banned substring {banned!r}",
                    )


if __name__ == "__main__":
    unittest.main()
