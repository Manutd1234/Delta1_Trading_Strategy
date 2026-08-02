from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_cta import BacktestConfig
from enhanced_strategy import (
    EXPANDED_ASSET_CLASSES,
    EnhancedConfig,
    ForecastCandidate,
    ensemble_signal,
    paired_sharpe_difference,
    probabilistic_sharpe_ratio,
    stitch_signals,
    trend_strength,
    walk_forward_selection,
)

DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "data/Delta1"))


class TestTrendStrength(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2000-01-03", periods=600)
        rng = np.random.default_rng(11)
        drift = np.linspace(100.0, 160.0, len(self.index))
        noise = rng.normal(0, 1.5, len(self.index)).cumsum()
        self.prices = pd.DataFrame({"X": drift + noise}, index=self.index)

    def test_signal_is_bounded_and_positive_in_an_uptrend(self) -> None:
        signal = trend_strength(self.prices, (63, 252), vol_span=60, cap=2.0)
        self.assertLessEqual(float(signal.abs().max().max()), 1.0)
        self.assertGreater(float(signal.iloc[-100:].mean().iloc[0]), 0.0)

    def test_future_mutation_cannot_change_past_signal(self) -> None:
        baseline = trend_strength(self.prices, (63, 252), vol_span=60, cap=2.0)
        altered = self.prices.copy()
        altered.loc[self.index[500] :, "X"] = 5.0
        changed = trend_strength(altered, (63, 252), vol_span=60, cap=2.0)
        pd.testing.assert_frame_equal(
            baseline.loc[: self.index[499]], changed.loc[: self.index[499]]
        )

    def test_negative_price_levels_are_handled(self) -> None:
        shifted = self.prices - 200.0  # back-adjusted series can go negative
        original = trend_strength(self.prices, (252,), vol_span=60, cap=2.0)
        translated = trend_strength(shifted, (252,), vol_span=60, cap=2.0)
        pd.testing.assert_frame_equal(original, translated)


class TestCarryStrength(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2000-01-03", periods=900)
        rng = np.random.default_rng(19)
        adjusted = 100 + rng.normal(0, 0.8, len(self.index)).cumsum()
        self.prices = pd.DataFrame({"X": adjusted}, index=self.index)
        # contango: every ~21 days the unadjusted front jumps UP by the spread
        gaps = np.zeros(len(self.index))
        gaps[::21] = 1.5
        self.unadjusted = self.prices + pd.DataFrame(
            {"X": gaps.cumsum()}, index=self.index
        )

    def test_contango_produces_negative_carry(self) -> None:
        from enhanced_strategy import carry_strength

        carry = carry_strength(self.prices, self.unadjusted, vol_span=60, cap=2.0)
        self.assertLess(float(carry["X"].dropna().mean()), 0.0)
        self.assertLessEqual(float(carry.abs().max().max()), 1.0)

    def test_future_mutation_cannot_change_past_carry(self) -> None:
        from enhanced_strategy import carry_strength

        baseline = carry_strength(self.prices, self.unadjusted, vol_span=60, cap=2.0)
        altered_prices = self.prices.copy()
        altered_unadjusted = self.unadjusted.copy()
        altered_prices.loc[self.index[800] :, "X"] += 50
        altered_unadjusted.loc[self.index[800] :, "X"] -= 50
        changed = carry_strength(altered_prices, altered_unadjusted, vol_span=60, cap=2.0)
        pd.testing.assert_frame_equal(
            baseline.loc[: self.index[799]], changed.loc[: self.index[799]]
        )

    def test_blend_falls_back_to_trend_where_carry_is_missing(self) -> None:
        from enhanced_strategy import blend_trend_and_carry

        trend = pd.DataFrame({"X": [0.8, 0.8, 0.8]}, index=self.index[:3])
        carry = pd.DataFrame({"X": [np.nan, -0.4, np.nan]}, index=self.index[:3])
        blend = blend_trend_and_carry(trend, carry, carry_weight=0.5)
        self.assertAlmostEqual(blend.iloc[0, 0], 0.8)   # no carry -> pure trend
        self.assertAlmostEqual(blend.iloc[1, 0], 0.2)   # (0.8 - 0.4) / 2
        self.assertAlmostEqual(blend.iloc[2, 0], 0.8)


class TestEwmaLeverage(unittest.TestCase):
    def test_flat_history_stays_neutral_and_bounds_hold(self) -> None:
        from delta1_cta import BacktestConfig as Config
        from enhanced_strategy import _ewma_portfolio_leverage

        config = Config(Path("unused"), Path("unused"))
        flat_then_live = pd.Series(
            np.r_[np.zeros(150), np.random.default_rng(3).normal(0, 6e-3, 250)],
            index=pd.bdate_range("2000-01-03", periods=400),
        )
        leverage = _ewma_portfolio_leverage(flat_then_live, config, decay=0.94)
        self.assertTrue((leverage.iloc[:150] == 1.0).all())
        self.assertGreaterEqual(float(leverage.min()), config.min_leverage)
        self.assertLessEqual(float(leverage.max()), config.max_leverage)


class TestWalkForwardSelection(unittest.TestCase):
    def _candidate_returns(self) -> dict[str, pd.Series]:
        index = pd.bdate_range("1990-01-01", "2002-12-31")
        rng = np.random.default_rng(3)
        noise = pd.Series(rng.normal(0, 1e-4, len(index)), index=index)
        strong = pd.Series(8e-4, index=index) + noise
        weak = pd.Series(-2e-4, index=index) + noise
        # "A" dominates before 1997, "B" after.
        split = index < "1997-01-01"
        a = strong.where(split, weak)
        b = weak.where(split, strong)
        return {"A": a, "B": b}

    def test_selection_tracks_trailing_performance(self) -> None:
        config = EnhancedConfig(selection_window_months=36, first_selection_year=1994)
        selections, scores = walk_forward_selection(self._candidate_returns(), config)
        self.assertEqual(selections.loc[1995], "A")
        self.assertEqual(selections.loc[2002], "B")
        self.assertEqual(set(selections.index - 1) - set(scores.index), set())

    def test_future_mutation_cannot_change_past_selection(self) -> None:
        config = EnhancedConfig(selection_window_months=36, first_selection_year=1994)
        returns = self._candidate_returns()
        baseline, _ = walk_forward_selection(returns, config)
        mutated = {name: series.copy() for name, series in returns.items()}
        mutated["B"].loc["2001-01-01":] = 0.05
        changed, _ = walk_forward_selection(mutated, config)
        pd.testing.assert_series_equal(
            baseline.loc[:2001], changed.loc[:2001], check_names=False
        )

    def test_stitched_signal_uses_prior_year_selection(self) -> None:
        index = pd.bdate_range("1994-06-01", "1996-06-30")
        signals = {
            "A": pd.DataFrame({"X": 1.0}, index=index),
            "B": pd.DataFrame({"X": -1.0}, index=index),
        }
        selections = pd.Series({1995: "A", 1996: "B"})
        composite = stitch_signals(signals, selections)
        self.assertTrue(composite.loc["1994"].isna().all().all())
        self.assertTrue((composite.loc["1995"] == 1.0).all().all())
        self.assertTrue((composite.loc["1996"] == -1.0).all().all())

    def test_ensemble_is_average_and_bounded(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=10)
        signals = {
            "A": pd.DataFrame({"X": 1.0}, index=index),
            "B": pd.DataFrame({"X": -0.5}, index=index),
        }
        combined = ensemble_signal(signals)
        self.assertTrue((combined == 0.25).all().all())


class TestStatistics(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.bdate_range("1995-01-02", "2004-12-31")
        rng = np.random.default_rng(5)
        base = pd.Series(rng.normal(4e-4, 6e-3, len(index)), index=index)
        self.baseline = base
        self.better = base + 3e-4

    def test_identical_series_have_zero_sharpe_difference(self) -> None:
        stats = paired_sharpe_difference(
            self.baseline, self.baseline, "1995-01-01", "2004-12-31", samples=200
        )
        self.assertAlmostEqual(stats["Observed Sharpe difference"], 0.0)

    def test_uniform_improvement_is_detected(self) -> None:
        stats = paired_sharpe_difference(
            self.better, self.baseline, "1995-01-01", "2004-12-31", samples=500
        )
        self.assertGreater(stats["Observed Sharpe difference"], 0.0)
        self.assertGreater(stats["P(difference > 0)"], 0.95)

    def test_psr_is_one_half_against_own_sharpe(self) -> None:
        monthly = (1 + self.baseline).resample("ME").prod() - 1
        own_sharpe = monthly.mean() / monthly.std(ddof=1) * math.sqrt(12)
        psr = probabilistic_sharpe_ratio(
            self.baseline, "1995-01-01", "2004-12-31", own_sharpe
        )
        self.assertAlmostEqual(psr, 0.5, places=2)


def _synthetic_market(seed: int, periods: int = 1600) -> pd.DataFrame:
    index = pd.bdate_range("1998-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    x = 100 + np.linspace(0, 30, periods) + rng.normal(0, 1.2, periods).cumsum()
    y = 50 - np.linspace(0, 10, periods) + rng.normal(0, 0.8, periods).cumsum()
    return pd.DataFrame({"X": x, "Y": y}, index=index)


def _synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {"tick_size": [0.25, 0.1], "point_value": [50.0, 100.0]},
        index=["X", "Y"],
    )


class TestSharedPipeline(unittest.TestCase):
    """Panel-required checks on the shared signal-to-accounting pipeline."""

    def setUp(self) -> None:
        from delta1_cta import BacktestConfig as Config

        self.prices = _synthetic_market(23)
        self.metadata = _synthetic_metadata()
        self.classes = {"A": ("X",), "B": ("Y",)}
        self.base = Config(Path("unused"), Path("unused"))
        self.enhanced = EnhancedConfig()

    def _run(self, prices: pd.DataFrame):
        from enhanced_strategy import run_signal_backtest
        from institutional_strategy import InstitutionalConfig

        signal = trend_strength(prices, (63, 252), vol_span=60, cap=2.0)
        return run_signal_backtest(
            signal, prices, self.metadata, self.base, InstitutionalConfig(),
            name="synthetic", asset_classes=self.classes,
        )

    def test_truncated_history_reproduces_full_run_returns(self) -> None:
        full = self._run(self.prices)
        cutoff = self.prices.index[1200]  # deliberately mid-month
        truncated = self._run(self.prices.loc[:cutoff])
        pd.testing.assert_series_equal(
            full.daily.loc[:cutoff, "net_return"],
            truncated.daily.loc[:cutoff, "net_return"],
        )

    def test_flat_history_never_maps_to_maximum_leverage(self) -> None:
        from delta1_cta import _portfolio_leverage

        flat_then_live = pd.Series(
            np.r_[np.zeros(200), np.random.default_rng(7).normal(0, 6e-3, 200)],
            index=pd.bdate_range("2000-01-03", periods=400),
        )
        leverage = _portfolio_leverage(flat_then_live, self.base)
        self.assertTrue((leverage.iloc[:200] == 1.0).all())

    def test_leverage_stays_neutral_until_vol_window_is_live(self) -> None:
        # A strategy whose signal starts mid-history must not trade at the
        # leverage cap while its trailing vol window is diluted by flat days.
        signal = trend_strength(self.prices, (63, 252), vol_span=60, cap=2.0)
        signal.loc[: self.prices.index[800]] = np.nan  # flat first three years
        from enhanced_strategy import run_signal_backtest
        from institutional_strategy import InstitutionalConfig

        result = run_signal_backtest(
            signal, self.prices, self.metadata, self.base, InstitutionalConfig(),
            name="late-start", asset_classes=self.classes,
        )
        live_start = result.positions.abs().sum(axis=1).gt(0).idxmax()
        warmup = result.daily.loc[live_start:].iloc[: self.base.portfolio_vol_window]
        self.assertLessEqual(float(warmup["leverage"].max()), 1.0 + 1e-12)

    def test_short_closure_holds_positions_without_trading(self) -> None:
        prices = self.prices.copy()
        closure = self.prices.index[1000:1004]
        prices.loc[closure, "X"] = np.nan
        prices["X"] = prices["X"].ffill(limit=5)  # mimic the loader bridge
        result = self._run(prices)
        traded = result.positions.diff().abs().loc[closure, "X"]
        self.assertTrue((traded.fillna(0.0) == 0.0).all())


class TestConfigValidation(unittest.TestCase):
    def test_duplicate_candidate_names_fail_fast(self) -> None:
        candidates = (
            ForecastCandidate("Same", "sign", (252,)),
            ForecastCandidate("Same", "strength", (63,)),
        )
        with self.assertRaises(ValueError):
            EnhancedConfig(candidates=candidates).validate()

    def test_unknown_kind_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            ForecastCandidate("Bad", "carry", (252,)).validate()

    def test_universe_has_no_excluded_or_duplicate_symbols(self) -> None:
        from enhanced_strategy import EXCLUDED_CONTRACTS

        symbols = [s for members in EXPANDED_ASSET_CLASSES.values() for s in members]
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertFalse(set(symbols) & set(EXCLUDED_CONTRACTS))


@unittest.skipUnless(DATA_DIR.exists(), "Supplied DELTA1 data directory is not available")
class TestEnhancedIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from enhanced_strategy import run_walk_forward

        cls.base = BacktestConfig(
            DATA_DIR, Path(tempfile.gettempdir()) / "delta1_test_outputs"
        )
        cls.config = EnhancedConfig(
            candidates=(
                ForecastCandidate("Sign 12m", "sign", (252,)),
                ForecastCandidate("Strength 3/6/12m", "strength", (63, 126, 252)),
            )
        )
        cls.result = run_walk_forward(cls.base, cls.config)

    def test_positions_change_only_at_month_boundaries(self) -> None:
        for result in (
            self.result.backtest,
            *self.result.candidate_results.values(),
        ):
            active = result.positions.diff().abs().sum(axis=1).gt(1e-15)
            month = pd.Series(
                result.daily.index.to_period("M"), index=result.daily.index
            )
            transition = month.ne(month.shift(1))
            self.assertTrue(bool((~active.iloc[1:] | transition[1:]).all()), result.name)

    def test_forecasts_are_bounded_and_returns_finite(self) -> None:
        for result in (self.result.backtest, self.result.ensemble):
            self.assertLessEqual(float(result.signals.abs().max().max()), 1.0 + 1e-12)
            self.assertTrue(bool(np.isfinite(result.daily["net_return"]).all()))

    def test_selections_only_reference_declared_candidates(self) -> None:
        names = {candidate.name for candidate in self.config.candidates}
        self.assertTrue(set(self.result.selections).issubset(names))

    def test_universe_accounting_matches_catalogue(self) -> None:
        from enhanced_strategy import EXCLUDED_CONTRACTS

        catalogue = pd.read_csv(DATA_DIR / "CATALOGUE_Delta1_Futures.csv")
        ccb = catalogue[catalogue["symbol"].str.endswith("_CCB")].copy()
        ccb["clean"] = ccb["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
        usd = set(ccb.loc[ccb["currency"] == "USD", "clean"])
        members = {s for m in EXPANDED_ASSET_CLASSES.values() for s in m}
        self.assertEqual(len(usd), 56)
        self.assertEqual(usd, members | set(EXCLUDED_CONTRACTS))

    def test_volume_gate_delays_untradeable_marks(self) -> None:
        from enhanced_strategy import load_volumes, tradeable_mask

        gate = tradeable_mask(load_volumes(DATA_DIR, ["6N"]))
        # 6N carries zero reported volume through 2005-06; the gate must keep
        # it out until real trading appears.
        self.assertFalse(bool(gate.loc["2005":"2006", "6N"].any()))
        self.assertTrue(bool(gate.loc["2010":, "6N"].all()))


if __name__ == "__main__":
    unittest.main()
