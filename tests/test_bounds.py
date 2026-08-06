"""Tests for the position-size bound sensitivity module.

Nothing here touches the licensed dataset.  The engine is injected through the
``run_fn`` seam ``levers`` already provides, with a synthetic engine whose
realised volatility is a known, deliberately non-proportional function of the
risk budget so the risk-matching solver has something real to solve.

Two properties of that fixture are load-bearing rather than incidental.  Its
volatility response is superlinear in the budget, so a single first-order
rescale does not land and the solver's secant step is exercised.  And its
*path* is not a rescale of one fixed shape: session exposures are rounded to
integer contracts before scaling, so two budgets that realise the same
volatility trace different paths.  A pure-rescale fixture would make every
scale-free statistic invariant to ``target_vol`` by construction and would make
the module's solver-jitter control untestable — which is exactly the hole these
tests were written to close.
"""

from __future__ import annotations

import math
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from delta1_strategy.research import bounds, inference
from delta1_strategy.research.strategy import StrategyConfig, basis_momentum

FORBIDDEN_COLUMN_SUBSTRINGS = ("target", "pass", "fail", "verdict")


def config(**overrides: Any) -> StrategyConfig:
    return StrategyConfig(
        data_dir=Path("unused"),
        output_dir=Path("unused"),
        **overrides,
    )


def series(values: np.ndarray, start: str = "2000-01-03") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def synthetic_daily(
    returns: pd.Series,
    *,
    risk_scalar: pd.Series | None = None,
    limit_scale: pd.Series | None = None,
) -> pd.DataFrame:
    index = returns.index
    ones = pd.Series(1.0, index=index)
    return pd.DataFrame(
        {
            "net_return": returns,
            "cost": ones * 4.0e-5,
            "prior_nav_usd": ones * 1_000_000.0,
            "fixed_execution_cost_usd": ones * 30.0,
            "market_impact_cost_usd": ones * 10.0,
            "risk_scalar": ones * 0.8 if risk_scalar is None else risk_scalar,
            "gross_notional_multiple": ones * 2.5,
            "static_margin_fraction": ones * 0.2,
            "max_order_participation": ones * 0.01,
            "max_rebalance_participation": ones * 0.01,
            "max_roll_participation_proxy": ones * 0.01,
            "pending_markets": ones * 0.0,
            "target_portfolio_limit_scale": ones if limit_scale is None else limit_scale,
            "rebalance_contract_turnover": ones * 5.0,
            "roll_contract_turnover_increment": ones * 9.0,
        },
        index=index,
    )


def synthetic_metrics(returns: pd.Series, annualization: int = 252) -> pd.Series:
    """A metrics Series with the exact labels the report reads.

    Supplying it through the ``(result, metrics)`` seam keeps the tests off
    ``performance_metrics``, which needs a full engine ledger the synthetic
    result does not have.
    """

    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = float((equity.to_numpy() / peak - 1.0).min())
    years = len(returns) / annualization
    volatility = float(returns.std() * math.sqrt(annualization))
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    return pd.Series(
        {
            "Start": returns.index.min().date().isoformat(),
            "End": returns.index.max().date().isoformat(),
            "Years": years,
            "CAGR": cagr,
            "Annualized volatility": volatility,
            "Naive daily Sharpe (sqrt252, rf=0)": float(
                returns.mean() / returns.std() * math.sqrt(annualization)
            ),
            "HAC Sharpe (21 lags, rf=0)": 1.0,
            "Sortino (rf=0)": 2.0,
            "Annualized downside deviation": 0.05,
            "Daily return skew": 0.1,
            "Daily return excess kurtosis": 2.0,
            "Calmar": cagr / abs(drawdown) if drawdown < 0 else np.nan,
            "Max drawdown": drawdown,
            "Max drawdown duration (sessions)": 100.0,
            "Annual cost drag": 0.01,
            "Annual fixed-cost drag": 0.009,
            "Annual impact-cost drag": 0.001,
            "Average risk scalar": 0.8,
            "Average markets held": 40.0,
            "Average gross notional multiple": 2.5,
            "Peak gross notional multiple": 5.1,
            "Peak static margin fraction": 0.5,
            "Portfolio-limit binding decision days": 24.0,
            "Peak order participation": 0.02,
            "Peak rebalance participation": 0.02,
            "Trade profit factor (USD)": 1.8,
            "Trade expectancy (bps NAV)": 16.0,
        }
    )


class SyntheticEngine:
    """An engine whose realised volatility is a known function of the config.

    ``realised = reference * shrink(max_risk_scalar) * (budget / 0.07) ** 1.05``.
    The exponent is deliberately not one: a proportional response would let a
    single first-order rescale land exactly, which would leave the solver's
    secant step untested.

    The *path* is deliberately not a rescale of one fixed shape either.  Session
    exposures are rounded to integer contracts before the path is scaled, so
    two risk budgets that realise the same volatility trace genuinely different
    paths — which is what the real engine does through integer contracts, the
    no-trade buffer and the participation cap.  An engine that produced a pure
    rescale would make every scale-free statistic invariant to ``target_vol``
    by construction and would hide the solver jitter this module measures.
    """

    reference = 0.0772
    exponent = 1.05

    def __init__(self, sessions: int = 500, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        shape = rng.normal(0.0, 1.0, sessions)
        self._shape = shape - shape.mean()
        self._exposure = 1.0 + rng.random(sessions)
        self._index = pd.bdate_range("2000-01-03", periods=sessions)
        self.calls: list[StrategyConfig] = []

    def realised_volatility(self, variant_config: StrategyConfig) -> float:
        shrink = 0.55 + 0.45 * (variant_config.max_risk_scalar / 2.0)
        shrink *= 0.90 + 0.10 * (variant_config.signal_cap / 2.0)
        budget = variant_config.target_vol / 0.07
        return self.reference * shrink * budget**self.exponent

    def path(self, variant_config: StrategyConfig) -> pd.Series:
        wanted = self.realised_volatility(variant_config)
        contracts = np.floor(self._exposure * variant_config.target_vol * 1_000.0)
        raw = contracts * self._shape
        daily = raw * (wanted / math.sqrt(252) / raw.std(ddof=1))
        return pd.Series(daily + 0.0005, index=self._index)

    def __call__(self, variant_config: StrategyConfig) -> tuple[Any, pd.Series]:
        self.calls.append(variant_config)
        returns = self.path(variant_config)
        result = SimpleNamespace(
            name="synthetic",
            daily=synthetic_daily(returns),
            positions=pd.DataFrame(1.0, index=self._index, columns=["A"]),
            signals=None,
            trade_episodes=pd.DataFrame(),
        )
        return result, synthetic_metrics(returns)


class BoundGridTests(unittest.TestCase):
    def test_defaults_cover_the_swept_family(self) -> None:
        grid = bounds.BoundGrid()
        self.assertEqual(len(grid.max_risk_scalars), 5)
        self.assertEqual(len(grid.min_risk_scalars), 3)

    def test_invalid_grids_are_rejected(self) -> None:
        cases = {
            "empty": {"signal_caps": ()},
            "duplicate": {"shock_floors": (0.25, 0.25)},
            "not_a_tuple": {"risk_managed_caps": [1.0, 2.0]},
            "negative": {"max_gross_notional_multiples": (-1.0, 3.0)},
            "non_finite": {"signal_caps": (float("nan"),)},
            "floor_above_one": {"shock_floors": (1.5,)},
            "boolean": {"signal_caps": (True,)},
            "infeasible_pair": {
                "max_risk_scalars": (0.5,),
                "min_risk_scalars": (0.25, 0.75),
            },
        }
        for label, overrides in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                bounds.BoundGrid(**overrides)

    def test_risk_match_and_drawdown_configs_validate(self) -> None:
        with self.assertRaises(ValueError):
            bounds.RiskMatchConfig(tolerance=0.0)
        with self.assertRaises(ValueError):
            bounds.RiskMatchConfig(max_iterations=0)
        with self.assertRaises(ValueError):
            bounds.RiskMatchConfig(min_multiplier=1.5)
        with self.assertRaises(ValueError):
            bounds.RiskMatchConfig(reference_volatility=0.0)
        with self.assertRaises(ValueError):
            bounds.DrawdownRiskConfig(samples=0)
        with self.assertRaises(ValueError):
            bounds.DrawdownRiskConfig(breach_thresholds=(0.20, 0.15))

    def test_control_probe_and_resolution_configs_validate(self) -> None:
        cases: dict[str, dict[str, Any]] = {
            "zero_fraction": {"control_probe_fractions": (0.0,)},
            "outside_the_band": {"control_probe_fractions": (1.5,)},
            "duplicate": {"control_probe_fractions": (0.5, 0.5)},
            "not_a_tuple": {"control_probe_fractions": [0.5]},
            "boolean": {"control_probe_fractions": (True,)},
            "non_finite": {"control_probe_fractions": (float("nan"),)},
        }
        for label, overrides in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                bounds.RiskMatchConfig(**overrides)
        # The empty ladder is legal: it disables the control deliberately.
        self.assertEqual(
            bounds.RiskMatchConfig(control_probe_fractions=()).control_probe_fractions,
            (),
        )
        resolution_cases: dict[str, dict[str, Any]] = {
            "empty_blocks": {"block_lengths": ()},
            "duplicate_blocks": {"block_lengths": (21, 21)},
            "zero_block": {"block_lengths": (0,)},
            "samples": {"samples": 0},
            "batch": {"batch_size": 0},
            "confidence": {"confidence_z": 0.0},
        }
        for label, overrides in resolution_cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                bounds.LossShapeResolutionConfig(**overrides)


class BuildVariantTests(unittest.TestCase):
    def test_baseline_is_first_and_its_cell_is_not_repeated(self) -> None:
        variants = bounds.build_bound_variants(config())
        self.assertEqual(variants[0].evaluation_mode, bounds.BASELINE_MODE)
        self.assertEqual(variants[0].overrides, ())
        names = [variant.name for variant in variants]
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("risk_scalar_max2.00_min0.25", names)
        self.assertNotIn("signal_cap_2.00", names)
        # 5x3 risk-scalar cells less the baseline cell, plus two off-baseline
        # settings on each of the four remaining axes, plus the baseline row.
        self.assertEqual(len(variants), 1 + 14 + 8)

    def test_every_variant_is_supported_by_the_config(self) -> None:
        base = config()
        for variant in bounds.build_bound_variants(base):
            with self.subTest(variant=variant.name):
                self.assertTrue(variant.is_supported(base))

    def test_declared_panel_is_fixed_by_the_grid_alone(self) -> None:
        base = config()
        first = bounds.declared_paired_comparisons(base)
        second = bounds.declared_paired_comparisons(base)
        self.assertEqual(first, second)
        self.assertTrue(all(name.endswith("_risk_matched") for name in first))
        self.assertIn("signal_cap_1.00_risk_matched", first)
        self.assertIn("risk_scalar_max1.00_min0.25_risk_matched", first)

    def test_the_declared_panel_is_always_a_subset_of_the_built_variants(
        self,
    ) -> None:
        base = config()
        for grid in (
            bounds.BoundGrid(),
            bounds.BoundGrid(max_risk_scalars=(1.0, 2.0), min_risk_scalars=(0.25,)),
            bounds.BoundGrid(signal_caps=(1.0, 2.0)),
        ):
            with self.subTest(grid=grid.max_risk_scalars):
                built = {
                    variant.name
                    for variant in bounds.build_bound_variants(base, grid)
                }
                for name in bounds.declared_paired_comparisons(base, grid):
                    self.assertIn(name.removesuffix("_risk_matched"), built)

    def test_a_grid_missing_the_canonical_setting_is_refused(self) -> None:
        # The declared panel holds the canonical setting fixed on the other
        # axis, so a grid without it would declare comparisons the sweep never
        # builds.  Silently dropping those rows would leave the declared panel
        # and the reported panel free to differ with no trace in any artifact.
        base = config()
        cases = {
            "min_risk_scalar": bounds.BoundGrid(min_risk_scalars=(0.0, 0.10)),
            "max_risk_scalar": bounds.BoundGrid(max_risk_scalars=(1.0, 1.5)),
            "signal_cap": bounds.BoundGrid(signal_caps=(1.0, 1.5)),
            "shock_floor": bounds.BoundGrid(shock_floors=(0.25, 0.50)),
        }
        for label, grid in cases.items():
            with self.subTest(axis=label):
                with self.assertRaises(ValueError) as caught:
                    bounds.declared_paired_comparisons(base, grid)
                self.assertIn(label, str(caught.exception))

    def test_risk_matched_variant_replaces_rather_than_duplicates_the_budget(
        self,
    ) -> None:
        source = bounds.build_bound_variants(config())[1]
        matched = bounds.risk_matched_variant(source, risk_budget_volatility=0.08)
        keys = [key for key, _ in matched.overrides]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(dict(matched.overrides)["target_vol"], 0.08)
        again = bounds.risk_matched_variant(matched, risk_budget_volatility=0.09)
        self.assertEqual(dict(again.overrides)["target_vol"], 0.09)
        with self.assertRaises(ValueError):
            bounds.risk_matched_variant(source, risk_budget_volatility=0.0)


class RiskMatchSolverTests(unittest.TestCase):
    def test_first_order_step_is_the_plain_rescale(self) -> None:
        match = bounds.RiskMatchConfig()
        proposal = bounds.next_risk_budget_volatility(
            [(0.07, 0.0700)], 0.0772, match, anchor=0.07
        )
        self.assertAlmostEqual(proposal, 0.07 * 0.0772 / 0.0700, places=12)

    def test_secant_step_uses_the_observed_slope(self) -> None:
        match = bounds.RiskMatchConfig()
        history = [(0.07, 0.070), (0.08, 0.082)]
        proposal = bounds.next_risk_budget_volatility(
            history, 0.0772, match, anchor=0.07
        )
        slope = (0.082 - 0.070) / (0.08 - 0.07)
        self.assertAlmostEqual(proposal, 0.08 + (0.0772 - 0.082) / slope, places=12)

    def test_proposal_is_clamped_to_a_band_around_the_incumbent_budget(self) -> None:
        match = bounds.RiskMatchConfig()
        collapsed = bounds.next_risk_budget_volatility(
            [(0.07, 1e-6)], 0.0772, match, anchor=0.07
        )
        self.assertAlmostEqual(collapsed, 0.07 * match.max_multiplier, places=12)
        exploded = bounds.next_risk_budget_volatility(
            [(0.07, 100.0)], 0.0772, match, anchor=0.07
        )
        self.assertAlmostEqual(exploded, 0.07 * match.min_multiplier, places=12)

    def test_iterating_the_step_converges_on_a_non_proportional_engine(self) -> None:
        engine = SyntheticEngine()
        base = config(max_risk_scalar=1.0)
        reference = engine.realised_volatility(config())
        history = [(base.target_vol, engine.realised_volatility(base))]
        match = bounds.RiskMatchConfig()
        iterations = 0
        for _ in range(match.max_iterations):
            iterations += 1
            budget = bounds.next_risk_budget_volatility(
                history, reference, match, anchor=base.target_vol
            )
            achieved = engine.realised_volatility(
                config(max_risk_scalar=1.0, target_vol=budget)
            )
            history.append((budget, achieved))
            if abs(achieved - reference) <= match.tolerance:
                break
        self.assertLessEqual(abs(history[-1][1] - reference), match.tolerance)
        self.assertLessEqual(iterations, match.max_iterations)

    def test_solver_rejects_degenerate_inputs(self) -> None:
        match = bounds.RiskMatchConfig()
        with self.assertRaises(ValueError):
            bounds.next_risk_budget_volatility([], 0.0772, match, anchor=0.07)
        with self.assertRaises(ValueError):
            bounds.next_risk_budget_volatility(
                [(0.07, 0.0)], 0.0772, match, anchor=0.07
            )
        with self.assertRaises(ValueError):
            bounds.next_risk_budget_volatility(
                [(0.07, 0.07)], 0.0, match, anchor=0.07
            )


class MatchControlLadderTests(unittest.TestCase):
    """The measured noise floor beneath the matched rows."""

    def test_probe_budgets_sit_at_the_named_positions_in_the_band(self) -> None:
        match = bounds.RiskMatchConfig()
        probes = bounds.risk_match_control_budgets(
            anchor_budget=0.07,
            anchor_volatility=0.0772,
            reference=0.0772,
            match=match,
        )
        self.assertEqual(
            [label for label, _ in probes],
            ["minus1.00tol", "minus0.50tol", "plus0.50tol", "plus1.00tol"],
        )
        for (label, budget), fraction in zip(
            probes, match.control_probe_fractions
        ):
            with self.subTest(label=label):
                wanted = 0.0772 + fraction * match.tolerance
                self.assertAlmostEqual(budget, 0.07 * wanted / 0.0772, places=15)
                # A first-order rescale of the incumbent lands each probe on the
                # tolerance edge it is named for, which is the whole band the
                # solver would have accepted.
                self.assertLessEqual(
                    abs(budget * 0.0772 / 0.07 - 0.0772), match.tolerance + 1e-15
                )

    def test_disabling_the_ladder_yields_no_probes(self) -> None:
        probes = bounds.risk_match_control_budgets(
            anchor_budget=0.07,
            anchor_volatility=0.0772,
            reference=0.0772,
            match=bounds.RiskMatchConfig(control_probe_fractions=()),
        )
        self.assertEqual(probes, ())

    def test_probe_budgets_reject_degenerate_anchors(self) -> None:
        match = bounds.RiskMatchConfig()
        with self.assertRaises(ValueError):
            bounds.risk_match_control_budgets(
                anchor_budget=0.0,
                anchor_volatility=0.0772,
                reference=0.0772,
                match=match,
            )
        with self.assertRaises(ValueError):
            bounds.risk_match_control_budgets(
                anchor_budget=0.07,
                anchor_volatility=0.0,
                reference=0.0772,
                match=match,
            )

    def test_the_band_is_the_spread_across_in_tolerance_runs(self) -> None:
        shapes = [
            {"annualized_volatility": 0.0770, "max_drawdown": -0.10,
             "max_drawdown_over_annualized_volatility": 1.50, "ulcer_index": 0.030},
            {"annualized_volatility": 0.0774, "max_drawdown": -0.12,
             "max_drawdown_over_annualized_volatility": 1.58, "ulcer_index": 0.032},
            {"annualized_volatility": 0.0772, "max_drawdown": -0.11,
             "max_drawdown_over_annualized_volatility": 1.53, "ulcer_index": 0.031},
        ]
        report = bounds.match_jitter_report(
            [
                ("baseline", 0.0700, 0.0000, shapes[2]),
                ("low", 0.0697, -0.0002, shapes[0]),
                ("high", 0.0703, 0.0002, shapes[1]),
            ],
            tolerance=5.0e-4,
        )
        row = report.set_index("statistic").loc[
            "max_drawdown_over_annualized_volatility"
        ]
        self.assertEqual(row["band_status"], bounds.ESTIMATED)
        self.assertAlmostEqual(float(row["solver_jitter_band"]), 0.08, places=12)
        self.assertEqual(int(row["probes_inside_tolerance"]), 3)
        self.assertEqual(list(report.columns), bounds.BOUNDS_MATCH_JITTER_COLUMNS)

    def test_an_in_tolerance_budget_change_is_not_a_rescale_of_one_path(
        self,
    ) -> None:
        # The retracted justification for the matched comparison was that the
        # drawdown ratio is scale-free, so only a shape change could move it.
        # A budget change is not a rescale: it re-runs a discrete-execution
        # engine, and the resulting path is not an affine function of the
        # incumbent's.  A pure-rescale engine would give a correlation of
        # exactly one here, and would make the whole control ladder vacuous.
        engine = SyntheticEngine()
        base = config()
        anchor = engine.path(base)
        statistic = "max_drawdown_over_annualized_volatility"
        reference = bounds.drawdown_shape_metrics(anchor)["annualized_volatility"]
        probes = bounds.risk_match_control_budgets(
            anchor_budget=base.target_vol,
            anchor_volatility=reference,
            reference=reference,
            match=bounds.RiskMatchConfig(),
        )
        measured = [bounds.drawdown_shape_metrics(anchor)[statistic]]
        rescaled = list(measured)
        for _, budget in probes:
            probe = engine.path(config(target_vol=budget))
            correlation = float(np.corrcoef(anchor.to_numpy(), probe.to_numpy())[0, 1])
            self.assertLess(correlation, 1.0 - 1e-9)
            shape = bounds.drawdown_shape_metrics(probe)
            measured.append(shape[statistic])
            factor = shape["annualized_volatility"] / reference
            rescaled.append(bounds.drawdown_shape_metrics(anchor * factor)[statistic])
        # And the ladder moves the ratio much further than rescaling the
        # incumbent path to the same realised volatilities moves it.
        measured_band = max(measured) - min(measured)
        rescaled_band = max(rescaled) - min(rescaled)
        self.assertGreater(rescaled_band, 0.0)
        self.assertGreater(measured_band, 5.0 * rescaled_band)

    def test_out_of_tolerance_probes_are_excluded_not_averaged_in(self) -> None:
        inside = {
            "annualized_volatility": 0.0772,
            "max_drawdown": -0.11,
            "max_drawdown_over_annualized_volatility": 1.53,
            "ulcer_index": 0.031,
        }
        outside = {
            "annualized_volatility": 0.0900,
            "max_drawdown": -0.30,
            "max_drawdown_over_annualized_volatility": 3.00,
            "ulcer_index": 0.090,
        }
        report = bounds.match_jitter_report(
            [("baseline", 0.07, 0.0, inside), ("wild", 0.09, 0.013, outside)],
            tolerance=5.0e-4,
        )
        row = report.set_index("statistic").loc[
            "max_drawdown_over_annualized_volatility"
        ]
        self.assertEqual(row["band_status"], bounds.NOT_ESTIMABLE)
        self.assertTrue(pd.isna(row["solver_jitter_band"]))
        self.assertEqual(int(row["probes_inside_tolerance"]), 1)
        self.assertTrue(bool(row["band_basis"]))


class LossShapeResolutionTests(unittest.TestCase):
    settings = bounds.LossShapeResolutionConfig(
        block_lengths=(20,), samples=64, batch_size=16
    )

    def paths(self) -> tuple[pd.Series, dict[str, pd.Series]]:
        rng = np.random.default_rng(17)
        incumbent = series(rng.normal(0.0004, 0.006, 400))
        challenger = series(
            incumbent.to_numpy() + rng.normal(0.0, 0.0005, 400)
        )
        return incumbent, {"challenger": challenger}

    def test_the_realised_point_estimate_matches_drawdown_shape_metrics(
        self,
    ) -> None:
        incumbent, challengers = self.paths()
        report = bounds.loss_shape_resolution(
            incumbent, challengers, config=self.settings
        )
        base = bounds.drawdown_shape_metrics(incumbent)
        other = bounds.drawdown_shape_metrics(challengers["challenger"])
        for statistic in bounds.RESOLUTION_STATISTICS:
            row = report.loc[report["statistic"].eq(statistic)].iloc[0]
            with self.subTest(statistic=statistic):
                self.assertAlmostEqual(
                    float(row["point_estimate"]),
                    other[statistic] - base[statistic],
                    places=15,
                )

    def test_the_shared_draw_reproduces_by_hand_and_is_variant_independent(
        self,
    ) -> None:
        incumbent, challengers = self.paths()
        extra = dict(challengers)
        extra["second"] = challengers["challenger"] * 0.5
        alone = bounds.loss_shape_resolution(
            incumbent, challengers, config=self.settings
        )
        together = bounds.loss_shape_resolution(
            incumbent, extra, config=self.settings
        )
        # Adding a comparison must not move another comparison's floor: the
        # index draw is seeded from the block length and path length only.
        pd.testing.assert_frame_equal(
            alone,
            together.loc[together["comparison"].eq("challenger")].reset_index(
                drop=True
            ),
        )
        # And the draw is the repository's shared resampler, reproduced here.
        entropy = np.random.SeedSequence(
            [
                bounds.LossShapeResolutionConfig().seed & 0xFFFFFFFF,
                zlib.crc32(b"paired loss-shape stationary block bootstrap")
                & 0xFFFFFFFF,
                20,
                400,
            ]
        )
        indices = inference._stationary_indices(
            np.random.default_rng(entropy), 400, 64, 400, 20
        )
        base = incumbent.to_numpy(dtype=float)
        other = challengers["challenger"].to_numpy(dtype=float)

        def ratio(values: np.ndarray) -> np.ndarray:
            block = values[indices]
            equity = np.cumprod(1.0 + block, axis=1)
            peaks = np.maximum.accumulate(np.maximum(equity, 1.0), axis=1)
            drawdown = equity / peaks - 1.0
            volatility = block.std(axis=1, ddof=1) * math.sqrt(252)
            return np.abs(drawdown.min(axis=1)) / volatility

        expected = float((ratio(other) - ratio(base)).std(ddof=1))
        row = alone.loc[
            alone["statistic"].eq("max_drawdown_over_annualized_volatility")
        ].iloc[0]
        self.assertAlmostEqual(
            float(row["bootstrap_standard_error"]), expected, places=12
        )
        self.assertAlmostEqual(
            float(row["minimum_detectable_effect_95_one_sided"]),
            1.645 * expected,
            places=12,
        )

    def test_every_declared_block_length_is_reported(self) -> None:
        incumbent, challengers = self.paths()
        report = bounds.loss_shape_resolution(
            incumbent,
            challengers,
            config=bounds.LossShapeResolutionConfig(
                block_lengths=(10, 20), samples=32, batch_size=16
            ),
        )
        self.assertEqual(set(report["expected_block_sessions"]), {10, 20})
        self.assertEqual(len(report), 2 * len(bounds.RESOLUTION_STATISTICS))
        self.assertEqual(list(report.columns), bounds.BOUNDS_RESOLUTION_COLUMNS)

    def test_mismatched_indexes_fail_closed(self) -> None:
        incumbent, challengers = self.paths()
        challengers["challenger"] = challengers["challenger"].iloc[:-3]
        with self.assertRaises(ValueError):
            bounds.loss_shape_resolution(
                incumbent, challengers, config=self.settings
            )

    def test_an_inert_variant_refuses_a_floor_rather_than_reporting_zero(
        self,
    ) -> None:
        # A variant that leaves the path untouched really does have a zero
        # bootstrap standard error.  Published as a minimum detectable effect
        # that reads as infinite power, and used as a floor it would let a
        # delta of exactly zero count as resolved.
        incumbent, _ = self.paths()
        report = bounds.loss_shape_resolution(
            incumbent, {"inert": incumbent.copy()}, config=self.settings
        )
        self.assertTrue(bool(report["bootstrap_standard_error"].isna().all()))
        self.assertTrue(
            bool(report["minimum_detectable_effect_95_one_sided"].isna().all())
        )
        self.assertTrue(bool(report["point_estimate"].eq(0.0).all()))
        self.assertTrue(
            all("not estimable" in str(v) for v in report["confidence_method"])
        )
        floors = bounds.resolution_floors(
            report,
            pd.DataFrame(
                [
                    {"statistic": statistic, "solver_jitter_band": 0.01}
                    for statistic in bounds.RESOLUTION_STATISTICS
                ]
            ),
        )
        self.assertTrue(bool(floors["sampling_floor"].isna().all()))
        # As-is rows carry no solver exposure, so nothing rescues the floor.
        self.assertTrue(bool(floors["resolution_floor"].isna().all()))

    def test_a_partly_non_estimable_block_grid_refuses_the_sampling_floor(
        self,
    ) -> None:
        resolution = pd.DataFrame(
            [
                {
                    "comparison": "c",
                    "evaluation_mode": bounds.AS_IS_MODE,
                    "statistic": "ulcer_index",
                    "minimum_detectable_effect_95_one_sided": 0.003,
                },
                {
                    "comparison": "c",
                    "evaluation_mode": bounds.AS_IS_MODE,
                    "statistic": "ulcer_index",
                    "minimum_detectable_effect_95_one_sided": np.nan,
                },
            ]
        )
        floors = bounds.resolution_floors(resolution, pd.DataFrame())
        self.assertTrue(bool(floors["sampling_floor"].isna().all()))
        self.assertTrue(bool(floors["resolution_floor"].isna().all()))

    def test_the_floor_is_the_widest_block_and_only_matched_rows_pay_jitter(
        self,
    ) -> None:
        resolution = pd.DataFrame(
            [
                {
                    "comparison": "matched",
                    "evaluation_mode": bounds.RISK_MATCHED_MODE,
                    "statistic": "ulcer_index",
                    "minimum_detectable_effect_95_one_sided": 0.001,
                },
                {
                    "comparison": "matched",
                    "evaluation_mode": bounds.RISK_MATCHED_MODE,
                    "statistic": "ulcer_index",
                    "minimum_detectable_effect_95_one_sided": 0.004,
                },
                {
                    "comparison": "raw",
                    "evaluation_mode": bounds.AS_IS_MODE,
                    "statistic": "ulcer_index",
                    "minimum_detectable_effect_95_one_sided": 0.002,
                },
            ]
        )
        jitter = pd.DataFrame(
            [{"statistic": "ulcer_index", "solver_jitter_band": 0.010}]
        )
        floors = bounds.resolution_floors(resolution, jitter).set_index("comparison")
        self.assertAlmostEqual(float(floors.loc["matched", "sampling_floor"]), 0.004)
        self.assertAlmostEqual(
            float(floors.loc["matched", "resolution_floor"]), 0.010
        )
        self.assertAlmostEqual(float(floors.loc["raw", "sampling_floor"]), 0.002)
        self.assertTrue(pd.isna(floors.loc["raw", "solver_jitter_band"]))
        self.assertAlmostEqual(float(floors.loc["raw", "resolution_floor"]), 0.002)


class CommonRandomNumberTests(unittest.TestCase):
    settings = bounds.DrawdownRiskConfig(
        horizon_sessions=120,
        block_length_sessions=20,
        samples=64,
        batch_size=16,
    )

    def paths(self) -> dict[str, pd.Series]:
        rng = np.random.default_rng(5)
        first = series(rng.normal(0.0004, 0.006, 400))
        second = series(first.to_numpy() * 0.5 + rng.normal(0.0, 0.001, 400))
        return {"first": first, "second": second}

    def test_index_draw_is_deterministic(self) -> None:
        left = bounds.stationary_drawdown_indices(400, self.settings)
        right = bounds.stationary_drawdown_indices(400, self.settings)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(left.shape, (64, 120))

    def test_adding_a_variant_does_not_move_another_variants_numbers(self) -> None:
        paths = self.paths()
        alone = bounds.common_random_number_drawdown_risk(
            {"first": paths["first"]}, config=self.settings
        )
        together = bounds.common_random_number_drawdown_risk(
            paths, config=self.settings
        )
        pd.testing.assert_frame_equal(
            alone,
            together.loc[together["variant"].eq("first")].reset_index(drop=True),
        )

    def test_reported_quantiles_reproduce_the_shared_draw_by_hand(self) -> None:
        paths = self.paths()
        indices = bounds.stationary_drawdown_indices(400, self.settings)
        values = paths["second"].to_numpy(dtype=float)
        block = np.cumprod(1.0 + values[indices], axis=1)
        peaks = np.maximum.accumulate(np.maximum(block, 1.0), axis=1)
        worst = (block / peaks - 1.0).min(axis=1)
        report = bounds.common_random_number_drawdown_risk(paths, config=self.settings)
        row = report.loc[report["variant"].eq("second")].iloc[0]
        self.assertAlmostEqual(
            float(row["bootstrap_median_max_drawdown"]),
            float(np.median(worst)),
            places=12,
        )
        self.assertAlmostEqual(
            float(row["bootstrap_p_drawdown_breach_15pct"]),
            float((worst < -0.15).mean()),
            places=12,
        )

    def test_mismatched_indexes_fail_closed(self) -> None:
        paths = self.paths()
        paths["second"] = paths["second"].iloc[:-4]
        with self.assertRaises(ValueError):
            bounds.common_random_number_drawdown_risk(paths, config=self.settings)


class DocumentedMechanismTests(unittest.TestCase):
    """The module's stated premises, checked against the engine they describe."""

    def synthetic_panel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(23)
        index = pd.bdate_range("2000-01-03", periods=900)
        steps = rng.normal(0.0, 1.0, (len(index), 2))
        prices = pd.DataFrame(
            100.0 + np.cumsum(steps, axis=0), index=index, columns=["A", "B"]
        )
        roll = rng.normal(0.0, 0.15, (len(index), 2))
        unadjusted = prices + np.cumsum(roll, axis=0)
        return prices, unadjusted

    def test_signal_cap_renormalises_upward_rather_than_bounding_magnitude(
        self,
    ) -> None:
        # basis_momentum returns (raw / scale).clip(-cap, cap) / cap, so |signal|
        # is non-increasing in cap: lowering the cap makes every unsaturated
        # cell LARGER.  The module docstring claims exactly this, and the claim
        # is asserted against the engine rather than against another sentence.
        prices, unadjusted = self.synthetic_panel()
        wide = basis_momentum(prices, unadjusted, cap=2.0).to_numpy(dtype=float)
        tight = basis_momentum(prices, unadjusted, cap=1.0).to_numpy(dtype=float)
        live = np.isfinite(wide) & np.isfinite(tight)
        self.assertGreater(int(live.sum()), 0)
        self.assertTrue(
            bool(
                (np.abs(tight[live]) >= np.abs(wide[live]) - 1e-12).all()
            ),
            "a lower signal_cap must never shrink the delivered sleeve",
        )
        self.assertTrue(
            bool((np.abs(tight[live]) > np.abs(wide[live]) + 1e-12).any()),
            "a lower signal_cap must enlarge at least one unsaturated cell",
        )

    def test_the_docstring_does_not_claim_every_bound_de_levers(self) -> None:
        doc = bounds.__doc__ or ""
        self.assertNotIn("Every one of these", doc)
        self.assertNotIn("so tightening one lowers realised volatility", doc)
        self.assertIn("(raw / scale).clip(-cap, cap) / cap", doc)
        self.assertIn("max_gross_notional_multiple", doc)

    def test_the_docstring_does_not_rest_on_scale_invariance(self) -> None:
        # The retracted justification: the matched rows were said to be read
        # through statistics "a constant rescale cannot move", with a ~1%
        # compounding residual offered as the margin.  The risk match re-runs a
        # discrete-execution engine, so that margin was the wrong one.
        doc = bounds.__doc__ or ""
        self.assertNotIn("only through statistics a constant", doc)
        self.assertIn("NOT a constant rescale", doc)
        shape_doc = bounds.drawdown_shape_metrics.__doc__ or ""
        self.assertNotIn("the ratio by about 1%", shape_doc)
        self.assertIn("measured, not derived", shape_doc)


class ShapeMetricTests(unittest.TestCase):
    def test_the_drawdown_ratio_barely_moves_under_a_constant_rescale(self) -> None:
        # The counterfactual the module exists to exclude: halving the returns
        # halves the drawdown, which is exactly what a tighter bound does on
        # its own.  The ratio is what survives it.  Under compounding the
        # ratio is only approximately scale-free, so the assertion is that
        # halving the path moves the level by ~50% and the ratio by ~1%.
        rng = np.random.default_rng(3)
        base = series(rng.normal(0.0004, 0.005, 800))
        original = bounds.drawdown_shape_metrics(base)
        rescaled = bounds.drawdown_shape_metrics(base * 0.5)
        level_move = abs(
            rescaled["max_drawdown"] / original["max_drawdown"] - 1.0
        )
        ratio_move = abs(
            rescaled["max_drawdown_over_annualized_volatility"]
            / original["max_drawdown_over_annualized_volatility"]
            - 1.0
        )
        self.assertGreater(level_move, 0.45)
        self.assertLess(ratio_move, 0.02)
        self.assertLess(ratio_move * 20.0, level_move)

    def test_volatility_convention_matches_the_engine_report(self) -> None:
        rng = np.random.default_rng(4)
        base = series(rng.normal(0.0004, 0.005, 500))
        shape = bounds.drawdown_shape_metrics(base)
        self.assertAlmostEqual(
            shape["annualized_volatility"],
            float(base.std() * math.sqrt(252)),
            places=15,
        )
        equity = (1.0 + base).cumprod()
        peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
        self.assertAlmostEqual(
            shape["max_drawdown"],
            float((equity.to_numpy() / peak - 1.0).min()),
            places=15,
        )
        tail = base[base <= base.quantile(0.05)]
        self.assertAlmostEqual(
            shape["conditional_shortfall_95"], -float(tail.mean()), places=15
        )

    def test_appending_later_returns_cannot_change_an_earlier_window(self) -> None:
        rng = np.random.default_rng(9)
        full = series(rng.normal(0.0004, 0.005, 600))
        prefix = full.iloc[:400]
        self.assertEqual(
            bounds.drawdown_shape_metrics(prefix),
            bounds.drawdown_shape_metrics(full.iloc[:400]),
        )
        early = bounds.drawdown_shape_metrics(prefix)
        late = bounds.drawdown_shape_metrics(full)
        self.assertNotEqual(early["annualized_volatility"], late["annualized_volatility"])

    def test_degenerate_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            bounds.drawdown_shape_metrics([0.01, 0.02])
        with self.assertRaises(ValueError):
            bounds.drawdown_shape_metrics(pd.Series(dtype=float))
        with self.assertRaises(ValueError):
            bounds.drawdown_shape_metrics(series(np.array([0.01, np.inf])))


class BindingDiagnosticTests(unittest.TestCase):
    def hand_built(self) -> SimpleNamespace:
        index = pd.bdate_range("2000-01-03", "2000-06-30")
        scalar = pd.Series(0.9, index=index)
        # The last business session of each month is the decision row.
        scalar.loc["2000-01-31"] = 2.00
        scalar.loc["2000-02-29"] = 2.00
        scalar.loc["2000-03-31"] = 0.25
        scalar.loc["2000-04-27"] = 2.00  # not a decision row, must not count
        limit = pd.Series(1.0, index=index)
        limit.iloc[:7] = 0.8
        returns = pd.Series(0.0004, index=index)
        return SimpleNamespace(
            name="hand",
            daily=synthetic_daily(returns, risk_scalar=scalar, limit_scale=limit),
            signals=None,
        )

    def test_binding_shares_match_the_hand_computed_case(self) -> None:
        report = bounds.bound_binding_diagnostic(
            self.hand_built(),
            config(),
            frames=None,
            start="2000-01-01",
            end="2000-06-30",
        )
        rows = report.set_index("bound")
        upper = rows.loc["max_risk_scalar"]
        self.assertEqual(int(upper["observations"]), 6)
        self.assertEqual(int(upper["binding_observations"]), 2)
        self.assertAlmostEqual(float(upper["binding_share"]), 2 / 6, places=12)
        self.assertAlmostEqual(float(upper["extreme_observed"]), 2.00, places=12)
        lower = rows.loc["min_risk_scalar"]
        self.assertEqual(int(lower["binding_observations"]), 1)
        self.assertAlmostEqual(float(lower["binding_share"]), 1 / 6, places=12)
        gross = rows.loc["max_gross_notional_multiple"]
        self.assertEqual(int(gross["binding_observations"]), 7)
        self.assertEqual(int(gross["observations"]), len(self.hand_built().daily))

    def test_signal_stage_bounds_refuse_without_frames(self) -> None:
        report = bounds.bound_binding_diagnostic(
            self.hand_built(),
            config(),
            frames=None,
            start="2000-01-01",
            end="2000-06-30",
        )
        rows = report.set_index("bound")
        for bound in ("signal_cap", "risk_managed_cap", "shock_floor"):
            with self.subTest(bound=bound):
                self.assertEqual(rows.loc[bound, "binding_status"], "not_estimable")
                self.assertTrue(bool(rows.loc[bound, "binding_basis"]))
        self.assertEqual(list(report.columns), bounds.BOUNDS_BINDING_COLUMNS)

    def test_a_later_session_cannot_change_an_earlier_binding_count(self) -> None:
        result = self.hand_built()
        early = bounds.bound_binding_diagnostic(
            result, config(), start="2000-01-01", end="2000-03-31"
        )
        extended = bounds.bound_binding_diagnostic(
            result, config(), start="2000-01-01", end="2000-06-30"
        )
        early_upper = early.set_index("bound").loc["max_risk_scalar"]
        later_upper = extended.set_index("bound").loc["max_risk_scalar"]
        self.assertEqual(int(early_upper["binding_observations"]), 2)
        self.assertGreaterEqual(
            int(later_upper["binding_observations"]),
            int(early_upper["binding_observations"]),
        )
        self.assertEqual(int(early_upper["observations"]), 3)

    def test_an_empty_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bounds.bound_binding_diagnostic(
                self.hand_built(), config(), start="2010-01-01", end="2010-12-31"
            )


class SweepTests(unittest.TestCase):
    grid = bounds.BoundGrid(
        max_risk_scalars=(1.0, 2.0),
        min_risk_scalars=(0.25,),
        signal_caps=(1.0, 2.0),
        risk_managed_caps=(2.0,),
        shock_floors=(0.75,),
        max_gross_notional_multiples=(5.0,),
    )
    risk = bounds.DrawdownRiskConfig(
        horizon_sessions=120,
        block_length_sessions=20,
        samples=64,
        batch_size=32,
    )
    resolution = bounds.LossShapeResolutionConfig(
        block_lengths=(20,), samples=64, batch_size=32
    )

    def run_sweep(self, **kwargs: Any) -> tuple[Any, SyntheticEngine]:
        engine = SyntheticEngine()
        artifacts = bounds.run_bounds_sweep(
            config(),
            grid=self.grid,
            run_fn=engine,
            risk=self.risk,
            resolution=self.resolution,
            start="2000-01-01",
            end="2010-12-31",
            paired_samples=40,
            **kwargs,
        )
        return artifacts, engine

    def test_every_variant_appears_in_both_evaluation_modes(self) -> None:
        artifacts, _ = self.run_sweep()
        shape = artifacts.shape
        self.assertEqual(list(shape.columns), bounds.BOUNDS_SHAPE_COLUMNS)
        modes = shape.groupby("evaluation_mode").size().to_dict()
        self.assertEqual(modes[bounds.AS_IS_MODE], 2)
        self.assertEqual(modes[bounds.RISK_MATCHED_MODE], 3)
        self.assertEqual(modes[bounds.BASELINE_MODE], 1)
        self.assertEqual(
            modes[bounds.MATCH_CONTROL_MODE],
            len(bounds.RiskMatchConfig().control_probe_fractions),
        )
        self.assertEqual(shape.iloc[0]["variant"], "baseline")

    def test_matched_rows_land_inside_the_stated_tolerance(self) -> None:
        artifacts, _ = self.run_sweep()
        match = bounds.RiskMatchConfig()
        matched = artifacts.shape.loc[
            artifacts.shape["evaluation_mode"].eq(bounds.RISK_MATCHED_MODE)
        ]
        self.assertTrue(
            (matched["annualized_volatility_gap_to_reference"].abs() <= match.tolerance).all(),
            matched[["variant", "annualized_volatility_gap_to_reference"]].to_dict("records"),
        )
        final = artifacts.risk_match.groupby("source_variant").tail(1)
        self.assertTrue(bool(final["within_tolerance"].all()))
        self.assertTrue((artifacts.risk_match["iteration"] >= 1).all())

    def test_the_as_is_row_is_lower_volatility_than_the_matched_row(self) -> None:
        artifacts, _ = self.run_sweep()
        shape = artifacts.shape.set_index("variant")
        tightened = shape.loc["risk_scalar_max1.00_min0.25"]
        matched = shape.loc["risk_scalar_max1.00_min0.25_risk_matched"]
        self.assertLess(
            float(tightened["annualized_volatility"]),
            float(matched["annualized_volatility"]),
        )
        self.assertEqual(int(tightened["solver_engine_runs"]), 0)
        self.assertGreaterEqual(int(matched["solver_engine_runs"]), 1)

    def test_the_baseline_twin_is_an_arithmetic_identity_not_a_noise_control(
        self,
    ) -> None:
        # next_risk_budget_volatility proposes last_budget * reference /
        # last_volatility, and for the baseline those two are the same float, so
        # the proposal is exactly config.target_vol and the twin re-runs a
        # byte-identical configuration.  It is a plumbing check.  Reading it as
        # evidence that the matched panel carries no measurement noise is the
        # error the control ladder exists to prevent, so this test pins the
        # identity AND pins that it is not the noise control.
        artifacts, _ = self.run_sweep()
        shape = artifacts.shape.set_index("variant")
        twin = shape.loc["baseline_risk_matched"]
        base = shape.loc["baseline"]
        for column in (
            "annualized_volatility",
            "cagr",
            "sharpe",
            "max_drawdown",
            "max_drawdown_over_annualized_volatility",
            "ulcer_index",
        ):
            with self.subTest(column=column):
                self.assertAlmostEqual(
                    float(twin[column]), float(base[column]), places=15
                )
        self.assertAlmostEqual(float(twin["delta_cagr"]), 0.0, places=15)
        self.assertAlmostEqual(float(twin["delta_sharpe"]), 0.0, places=15)
        controls = artifacts.shape.loc[
            artifacts.shape["evaluation_mode"].eq(bounds.MATCH_CONTROL_MODE)
        ]
        self.assertGreaterEqual(len(controls), 2)
        spread = float(
            controls["max_drawdown_over_annualized_volatility"].max()
            - controls["max_drawdown_over_annualized_volatility"].min()
        )
        self.assertGreater(spread, 0.0)

    def test_the_control_ladder_measures_a_non_zero_solver_jitter_band(
        self,
    ) -> None:
        # The defect this pins: the only control row for the matched panel used
        # to be an arithmetically guaranteed exact null, so the sweep certified
        # nothing about how far the matched statistics move between budgets the
        # tolerance equally accepts.
        artifacts, _ = self.run_sweep()
        jitter = artifacts.match_jitter.set_index("statistic")
        self.assertEqual(
            set(jitter.index), set(bounds.RESOLUTION_STATISTICS)
        )
        for statistic in bounds.RESOLUTION_STATISTICS:
            row = jitter.loc[statistic]
            with self.subTest(statistic=statistic):
                self.assertEqual(row["band_status"], bounds.ESTIMATED)
                self.assertGreater(float(row["solver_jitter_band"]), 0.0)
                self.assertLessEqual(
                    float(row["widest_volatility_match_error"]),
                    float(row["tolerance"]),
                )
                self.assertGreaterEqual(int(row["probes_inside_tolerance"]), 2)

    def test_a_matched_delta_inside_the_measured_floor_is_refused(self) -> None:
        artifacts, _ = self.run_sweep()
        shape = artifacts.shape
        matched = shape.loc[shape["evaluation_mode"].eq(bounds.RISK_MATCHED_MODE)]
        self.assertGreater(len(matched), 0)
        for _, row in matched.iterrows():
            statistic = "max_drawdown_over_annualized_volatility"
            floor = float(row[f"{statistic}_resolution_floor"])
            status = row[f"delta_{statistic}_status"]
            with self.subTest(variant=row["variant"]):
                self.assertTrue(np.isfinite(floor))
                self.assertGreater(floor, 0.0)
                if status == bounds.NOT_ESTIMABLE:
                    self.assertTrue(pd.isna(row[f"delta_{statistic}"]))
                    # The refusal withholds a claim, not the evidence.
                    self.assertTrue(np.isfinite(float(row[statistic])))
                else:
                    self.assertGreater(abs(float(row[f"delta_{statistic}"])), floor)

    def test_the_matched_floor_is_at_least_the_solver_jitter_band(self) -> None:
        artifacts, _ = self.run_sweep()
        bands = artifacts.match_jitter.set_index("statistic")[
            "solver_jitter_band"
        ].to_dict()
        matched = artifacts.shape.loc[
            artifacts.shape["evaluation_mode"].eq(bounds.RISK_MATCHED_MODE)
        ]
        for statistic in bounds.RESOLUTION_STATISTICS:
            floors = matched[f"{statistic}_resolution_floor"].astype(float)
            with self.subTest(statistic=statistic):
                self.assertTrue(bool((floors >= float(bands[statistic])).all()))
        as_is = artifacts.shape.loc[
            artifacts.shape["evaluation_mode"].eq(bounds.AS_IS_MODE)
        ]
        self.assertTrue(
            bool(
                as_is["max_drawdown_over_annualized_volatility_resolution_floor"]
                .astype(float)
                .gt(0.0)
                .all()
            )
        )

    def test_no_emitted_frame_carries_a_decision_shaped_column(self) -> None:
        artifacts, _ = self.run_sweep()
        frames = {
            "comparison": artifacts.comparison,
            "shape": artifacts.shape,
            "binding": artifacts.binding,
            "drawdown_risk": artifacts.drawdown_risk,
            "risk_match": artifacts.risk_match,
            "match_jitter": artifacts.match_jitter,
            "resolution": artifacts.resolution,
            "paired": artifacts.paired,
            "power": artifacts.power,
        }
        for name, frame in frames.items():
            joined = " ".join(map(str, frame.columns)).lower()
            for forbidden in FORBIDDEN_COLUMN_SUBSTRINGS:
                with self.subTest(frame=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, joined)
        self.assertFalse(bool(artifacts.power["selection_adjusted"].any()))
        self.assertFalse(bool(artifacts.drawdown_risk["selection_adjusted"].any()))
        self.assertFalse(bool(artifacts.paired["Selection adjusted"].any()))

    def test_the_drawdown_report_covers_every_variant_on_one_draw(self) -> None:
        artifacts, _ = self.run_sweep()
        risk = artifacts.drawdown_risk
        self.assertEqual(len(risk), len(artifacts.shape))
        self.assertEqual(set(risk["variant"]), set(artifacts.shape["variant"]))
        self.assertTrue(bool(risk["common_random_numbers"].all()))
        self.assertEqual(set(risk["seed"]), {self.risk.seed})

    def test_the_declared_paired_panel_is_the_one_that_is_reported(self) -> None:
        artifacts, _ = self.run_sweep()
        declared = bounds.declared_paired_comparisons(config(), self.grid)
        reported = {
            str(value).removesuffix("_vs_baseline")
            for value in artifacts.paired["Comparison"].unique()
        }
        self.assertEqual(reported, set(declared))
        self.assertEqual(len(artifacts.power), len(declared))
        self.assertEqual(list(artifacts.power.columns), bounds.BOUNDS_POWER_COLUMNS)
        self.assertTrue(
            artifacts.power["power_status"].eq(bounds.POWER_ESTIMATED).all()
        )

    def test_skipping_the_match_leaves_only_as_is_rows(self) -> None:
        artifacts, engine = self.run_sweep(include_risk_matched=False)
        self.assertTrue(artifacts.risk_match.empty)
        self.assertEqual(
            set(artifacts.shape["evaluation_mode"]),
            {bounds.BASELINE_MODE, bounds.AS_IS_MODE},
        )
        self.assertEqual(len(engine.calls), 3)

    def test_skipping_the_match_still_accounts_for_the_declared_panel(
        self,
    ) -> None:
        # Every declared comparison names a risk-matched variant, so skipping
        # the match leaves none of them buildable.  Emptying both tables without
        # a word would hide a whole declared panel; each missing comparison is
        # reported as refused instead.
        artifacts, _ = self.run_sweep(include_risk_matched=False)
        declared = bounds.declared_paired_comparisons(config(), self.grid)
        self.assertEqual(len(artifacts.power), len(declared))
        self.assertTrue(
            bool(
                artifacts.power["power_status"]
                .eq(bounds.POWER_NOT_ESTIMABLE)
                .all()
            )
        )
        reported = {
            str(value).removesuffix("_vs_baseline")
            for value in artifacts.paired["Comparison"].unique()
        }
        self.assertEqual(reported, set(declared))
        self.assertTrue(
            bool(
                artifacts.paired["Minimum detectable effect (95% one-sided)"]
                .isna()
                .all()
            )
        )
        self.assertTrue(
            all("not estimable" in str(value) for value in artifacts.paired["Method"])
        )

    def test_an_inert_bound_refuses_in_the_power_and_the_paired_table_alike(
        self,
    ) -> None:
        # The synthetic engine ignores min_risk_scalar, so this variant leaves
        # the path byte-identical: the differential has no sampling variation.
        # The defect this pins is that the power table refused while the paired
        # table published a zero standard error, a zero-width interval and a
        # zero minimum detectable effect for the same comparison, which reads as
        # the best-powered row in the panel.
        inert_grid = bounds.BoundGrid(
            max_risk_scalars=(2.0,),
            min_risk_scalars=(0.10, 0.25),
            signal_caps=(2.0,),
            risk_managed_caps=(2.0,),
            shock_floors=(0.75,),
            max_gross_notional_multiples=(5.0,),
        )
        artifacts = bounds.run_bounds_sweep(
            config(),
            grid=inert_grid,
            run_fn=SyntheticEngine(),
            risk=self.risk,
            resolution=self.resolution,
            start="2000-01-01",
            end="2010-12-31",
            paired_samples=40,
        )
        declared = bounds.declared_paired_comparisons(config(), inert_grid)
        self.assertEqual(len(declared), 1)
        self.assertEqual(len(artifacts.power), 1)
        row = artifacts.power.iloc[0]
        self.assertEqual(row["power_status"], bounds.POWER_NOT_ESTIMABLE)
        self.assertEqual(float(row["tracking_error_annualized"]), 0.0)
        self.assertTrue(
            pd.isna(row["minimum_detectable_sharpe_effect_95_one_sided"])
        )
        self.assertTrue(bool(row["power_basis"]))

        paired = artifacts.paired
        self.assertEqual(
            set(paired["Comparison"]), {f"{declared[0]}_vs_baseline"}
        )
        self.assertEqual(
            len(paired), 3 * len(inference.DEFAULT_BLOCK_LENGTHS)
        )
        for column in (
            "Bootstrap standard error",
            "Confidence lower",
            "Confidence upper",
            "Minimum detectable effect (95% one-sided)",
        ):
            with self.subTest(column=column):
                self.assertTrue(bool(paired[column].isna().all()))
        # What is observable is still reported.
        self.assertTrue(bool(paired["Tracking error (annualized)"].eq(0.0).all()))
        self.assertTrue(
            bool(paired["Return correlation with incumbent"].eq(1.0).all())
        )
        self.assertTrue(
            all("not estimable" in str(value) for value in paired["Method"])
        )

    def test_the_sweep_rejects_a_mistyped_configuration_object(self) -> None:
        engine = SyntheticEngine()
        with self.assertRaises(TypeError):
            bounds.run_bounds_sweep(config(), grid=object(), run_fn=engine)
        with self.assertRaises(TypeError):
            bounds.run_bounds_sweep(config(), grid=self.grid, match=object(), run_fn=engine)
        with self.assertRaises(TypeError):
            bounds.run_bounds_sweep(
                config(), grid=self.grid, resolution=object(), run_fn=engine
            )

    def test_a_grid_that_cannot_support_the_declared_panel_fails_before_running(
        self,
    ) -> None:
        # The declared panel is the module's selection control.  A grid that
        # omits the canonical setting used to produce declared names no variant
        # matched, and the sweep dropped them without a trace; now it refuses,
        # and it refuses before spending a single engine run.
        engine = SyntheticEngine()
        broken = bounds.BoundGrid(
            max_risk_scalars=(1.0, 2.0),
            min_risk_scalars=(0.0, 0.10),
            signal_caps=(2.0,),
            risk_managed_caps=(2.0,),
            shock_floors=(0.75,),
            max_gross_notional_multiples=(5.0,),
        )
        with self.assertRaises(ValueError):
            bounds.run_bounds_sweep(
                config(),
                grid=broken,
                run_fn=engine,
                risk=self.risk,
                resolution=self.resolution,
                start="2000-01-01",
                end="2010-12-31",
                paired_samples=40,
            )
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":
    unittest.main()
