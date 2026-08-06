"""Tests for the family-wise validation estimators.

The assertions here are analytic wherever an analytic answer exists.  A smoke
test on a Reality Check or a PBO is close to worthless: both procedures return
a number in [0, 1] for almost any input, so only a fixture with a known exact
answer can distinguish a correct implementation from a plausible one.  Three
fixtures carry most of the weight - constant differentials pin White's
recentring to ``1 / (1 + B)`` exactly, unit studentisation collapses Hansen's
upper SPA onto the Reality Check exactly, and a dominant configuration pins PBO
to zero exactly.
"""

from __future__ import annotations

import dataclasses
import math
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from delta1_strategy.research import inference, validation
from delta1_strategy.research.inference import ESTIMATED, NOT_ESTIMABLE
from delta1_strategy.research.strategy import (
    StrategyConfig,
    performance_metrics,
    run_backtest,
)
from delta1_strategy.research.validation import (
    MEAN_STATISTIC,
    REALITY_CHECK,
    SHARPE_STATISTIC,
    SPA_CONSISTENT,
    SPA_LOWER,
    SPA_UPPER,
    WALK_FORWARD_SELECTION_MODE,
    WALK_FORWARD_STABILITY_MODE,
    StrategyFamily,
    WalkForwardCandidate,
    WalkForwardConfig,
    anchored_walk_forward,
    assemble_family,
    cscv_pbo,
    family_deflated_sharpe,
    family_wise_report,
    hansen_spa,
    reality_check,
    spa_threshold,
)

BANNED_SUBSTRINGS = ("target", "pass", "fail", "verdict")


def matrix_family(
    matrix: np.ndarray,
    *,
    label: str = "fixture",
    start: str = "1990-01-02",
    declared_extra: tuple[str, ...] = (),
    annualization: int = 252,
) -> StrategyFamily:
    """Wrap a ``T x N`` array as a family whose first column is the benchmark."""

    rows, columns = matrix.shape
    names = ["benchmark"] + [f"member_{index:03d}" for index in range(columns - 1)]
    index = pd.bdate_range(start, periods=rows)
    paths = {
        name: pd.Series(matrix[:, position], index=index)
        for position, name in enumerate(names)
    }
    return assemble_family(
        paths,
        label=label,
        benchmark="benchmark",
        declared_members=(*names, *declared_extra),
        provenance="synthetic fixture",
        annualization=annualization,
    )


def synthetic_frames(
    *,
    markets: int = 4,
    sessions: int = 2400,
    seed: int = 20_260_806,
    start: str = "1990-01-01",
) -> dict[str, pd.DataFrame]:
    """Eight engine frames small enough to run the real backtest in a test.

    Deliberately built with the same key set ``load_shared_frames`` returns: the
    engine substitutes permissive defaults for any omitted frame, and an omitted
    ``delivery_months`` alone removes every roll cost, so a partial fixture
    would test a different strategy from the one the module is for.
    """

    index = pd.bdate_range(start, periods=sessions)
    rng = np.random.default_rng(seed)
    # Real roots from four asset classes: the episode builder resolves an asset
    # class per symbol, so an invented ticker would fail inside the engine
    # rather than inside the estimator under test.
    symbols = ["ES", "ZN", "6E", "GC", "CL", "CC"][:markets]

    levels = {}
    unadjusted = {}
    deliveries = {}
    for position, symbol in enumerate(symbols):
        drift = 0.03 + 0.02 * position
        steps = rng.normal(drift, 1.0 + 0.25 * position, sessions)
        path = 100.0 + np.cumsum(steps)
        levels[symbol] = path
        # A quarterly roll gap makes the basis sleeve estimable without ever
        # letting the unadjusted series approach zero.
        quarter = index.year * 4 + (index.month - 1) // 3
        gap = np.cumsum(np.r_[0.0, np.diff(quarter)] * (0.10 + 0.02 * position))
        unadjusted[symbol] = path + gap
        deliveries[symbol] = index.year * 100 + ((index.month - 1) // 3) * 3 + 3

    prices = pd.DataFrame(levels, index=index)
    unadjusted_frame = pd.DataFrame(unadjusted, index=index)
    delivery_months = pd.DataFrame(deliveries, index=index, dtype=float)
    volumes = pd.DataFrame(50_000.0, index=index, columns=symbols)
    fx_rates = pd.DataFrame({"USD": 1.0}, index=index)
    metadata = pd.DataFrame(
        {
            "tick_size": [0.01] * markets,
            "point_value": [100.0] * markets,
            "margin": [3_000.0] * markets,
            "currency": ["USD"] * markets,
        },
        index=pd.Index(symbols, name="clean_symbol"),
    )
    return {
        "prices": prices,
        "observed_opens": prices,
        "observed_closes": prices,
        "unadjusted": unadjusted_frame,
        "metadata": metadata,
        "volumes": volumes,
        "delivery_months": delivery_months,
        "fx_rates": fx_rates,
    }


def synthetic_config(**overrides: Any) -> StrategyConfig:
    config = StrategyConfig(data_dir=Path("."), output_dir=Path("."))
    return replace(config, **overrides) if overrides else config


@dataclass
class StubResult:
    """Minimal duck-typed stand-in for a ``BacktestResult``."""

    daily: pd.DataFrame
    target_positions: pd.DataFrame


class TestFamilyAssembler(unittest.TestCase):
    def test_undeclared_paths_are_refused(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=40)
        paths = {name: pd.Series(0.001, index=index) for name in ("a", "b")}
        with self.assertRaises(ValueError):
            assemble_family(
                paths,
                label="f",
                benchmark="a",
                declared_members=("a",),
                provenance="test",
            )

    def test_unsynchronized_paths_fail_closed_rather_than_intersecting(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=40)
        paths = {
            "a": pd.Series(0.001, index=index),
            "b": pd.Series(0.001, index=index[:-1]),
        }
        with self.assertRaises(ValueError):
            assemble_family(
                paths,
                label="f",
                benchmark="a",
                declared_members=("a", "b"),
                provenance="test",
            )

    def test_missing_declared_member_is_recorded_not_dropped(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=40)
        family = assemble_family(
            {"a": pd.Series(0.001, index=index)},
            label="f",
            benchmark="a",
            declared_members=("a", "b", "c"),
            provenance="test",
        )
        self.assertFalse(family.is_complete)
        self.assertEqual(family.missing_members, ("b", "c"))
        self.assertEqual(family.completeness, "incomplete")

    def test_non_finite_returns_are_refused(self) -> None:
        index = pd.bdate_range("2000-01-03", periods=40)
        values = np.full(40, 0.001)
        values[7] = np.nan
        with self.assertRaises(ValueError):
            assemble_family(
                {"a": pd.Series(values, index=index)},
                label="f",
                benchmark="a",
                declared_members=("a",),
                provenance="test",
            )


class TestRealityCheck(unittest.TestCase):
    def test_constant_differentials_give_the_exact_minimum_p_value(self) -> None:
        # Every resample of a constant column reproduces that column's mean
        # exactly, so the recentred bootstrap maximum is identically zero while
        # the observed maximum is strictly positive.  The p-value must then be
        # exactly 1/(1+B).  Recentring on zero, or on the grand mean, fails.
        rows = 600
        constants = np.array([0.0, 0.001, 0.002, 0.003])
        matrix = np.tile(constants, (rows, 1))
        family = matrix_family(matrix, label="constant")
        samples = 199
        report = reality_check(
            family,
            statistics=(MEAN_STATISTIC,),
            block_lengths=(21,),
            samples=samples,
        )
        self.assertEqual(len(report), 1)
        self.assertEqual(report.loc[0, "Status"], ESTIMATED)
        self.assertEqual(report.loc[0, "p value"], 1.0 / (samples + 1.0))

    def test_pure_noise_family_does_not_reject_the_null(self) -> None:
        rng = np.random.default_rng(101)
        matrix = rng.normal(0.0002, 0.006, (2500, 21))
        family = matrix_family(matrix, label="noise")
        report = reality_check(
            family,
            statistics=(MEAN_STATISTIC, SHARPE_STATISTIC),
            block_lengths=(21,),
            samples=999,
        )
        for _, row in report.iterrows():
            with self.subTest(statistic=row["Statistic"]):
                self.assertGreater(float(row["p value"]), 0.20)

    def test_one_implanted_edge_rejects_at_the_resolution_floor(self) -> None:
        rng = np.random.default_rng(102)
        matrix = rng.normal(0.0002, 0.006, (3000, 21))
        matrix[:, 1] = matrix[:, 0] + 0.0025
        family = matrix_family(matrix, label="edge")
        report = reality_check(
            family,
            statistics=(MEAN_STATISTIC,),
            block_lengths=(21,),
            samples=999,
        )
        self.assertEqual(float(report.loc[0, "p value"]), 1.0 / 1000.0)

    def test_adding_inferior_models_weakly_increases_the_p_value(self) -> None:
        # White's documented weakness, and the reason Hansen's test exists.
        # The added columns are strictly worse than the benchmark, so the
        # observed maximum is unchanged while the bootstrap maximum can only
        # grow.  Index paths do not depend on family size, so this is exact.
        rng = np.random.default_rng(103)
        base = rng.normal(0.0, 0.005, (1500, 6))
        base[:, 1] = base[:, 0] + 0.0006
        poor = rng.normal(-0.004, 0.005, (1500, 30))
        small = matrix_family(base, label="growth")
        large = matrix_family(np.hstack([base, poor]), label="growth")
        kwargs = {
            "statistics": (MEAN_STATISTIC,),
            "block_lengths": (21,),
            "samples": 999,
        }
        p_small = float(reality_check(small, **kwargs).loc[0, "p value"])
        p_large = float(reality_check(large, **kwargs).loc[0, "p value"])
        self.assertGreaterEqual(p_large, p_small)

    def test_incomplete_family_is_refused(self) -> None:
        rng = np.random.default_rng(104)
        family = matrix_family(
            rng.normal(0.0, 0.005, (500, 4)),
            label="incomplete",
            declared_extra=("member_never_run",),
        )
        report = reality_check(family, block_lengths=(21,), samples=99)
        self.assertTrue((report["Status"] == NOT_ESTIMABLE).all())
        self.assertTrue(report["p value"].isna().all())

    def test_an_undefined_member_refuses_instead_of_flooring_the_p_value(self) -> None:
        # A flat configuration - no trades, no variance - is a plausible
        # declared family member and the assembler accepts it, because its
        # returns are finite.  Its Sharpe is not: 0/0.  Left in the family that
        # NaN propagates into the maximum, compares False against every
        # bootstrap draw, and drives the p-value onto 1/(1+B) - the strongest
        # rejection the procedure can emit - while the row reads ESTIMATED.
        # Both procedures must refuse instead, and the refusal must be specific
        # to the degeneracy rather than a blanket one: the same family without
        # the flat member is estimated, and nowhere near the floor.
        rng = np.random.default_rng(105)
        rows = 1500
        index = pd.bdate_range("2000-01-03", periods=rows)
        base = rng.normal(4e-4, 5e-3, rows)
        paths = {
            "benchmark": pd.Series(base, index=index),
            "challenger": pd.Series(base + rng.normal(1e-4, 2e-3, rows), index=index),
        }
        samples = 999
        kwargs = {
            "statistics": (SHARPE_STATISTIC,),
            "block_lengths": (21,),
            "samples": samples,
        }
        clean = family_wise_report(
            assemble_family(
                paths,
                label="clean",
                benchmark="benchmark",
                declared_members=("benchmark", "challenger"),
                provenance="test",
            ),
            **kwargs,
        )
        self.assertTrue((clean["Status"] == ESTIMATED).all())
        self.assertTrue((clean["p value"] > 10.0 / (samples + 1.0)).all())

        paths["flat"] = pd.Series(np.zeros(rows), index=index)
        degenerate = assemble_family(
            paths,
            label="degenerate",
            benchmark="benchmark",
            declared_members=("benchmark", "challenger", "flat"),
            provenance="test",
        )
        self.assertTrue(degenerate.is_complete)
        report = family_wise_report(degenerate, **kwargs)
        self.assertEqual(set(report["Procedure"]), set(validation.ALL_PROCEDURES))
        for _, row in report.iterrows():
            with self.subTest(procedure=row["Procedure"]):
                self.assertEqual(row["Status"], NOT_ESTIMABLE)
                self.assertIsNone(row["p value"])
                self.assertIn("flat", str(row["Reason"]))

    def test_an_exact_benchmark_duplicate_is_retained_and_counted(self) -> None:
        # The other degeneracy, and it must not be conflated with the one
        # above.  A member that reproduces the benchmark exactly has a finite
        # differential of zero; only Hansen's studentisation is undefined for
        # it.  The Reality Check keeps it - its recentred draws are identically
        # zero - and both procedures must report the count rather than a
        # hard-coded zero that denies the degeneracy the SPA rows describe.
        rng = np.random.default_rng(106)
        matrix = rng.normal(0.0, 0.005, (900, 5))
        matrix[:, 4] = matrix[:, 0]
        report = family_wise_report(
            matrix_family(matrix, label="duplicate"),
            statistics=(MEAN_STATISTIC,),
            block_lengths=(21,),
            samples=199,
        )
        self.assertTrue((report["Status"] == ESTIMATED).all())
        self.assertTrue((report["Members without bootstrap variance"] == 1).all())
        retained = report.loc[report["Procedure"] == REALITY_CHECK, "Reason"].iloc[0]
        self.assertIn("retained", retained)


class TestHansenSpa(unittest.TestCase):
    def test_iterated_logarithm_constant(self) -> None:
        self.assertAlmostEqual(spa_threshold(6289), 2.082623, places=6)
        self.assertAlmostEqual(
            spa_threshold(6289) / math.sqrt(6289),
            math.sqrt(2.0 * math.log(math.log(6289)) / 6289),
            places=12,
        )
        with self.assertRaises(ValueError):
            spa_threshold(10)

    def test_the_three_recentrings_are_ordered_on_every_input(self) -> None:
        # p_l <= p_c <= p_u is a theorem, not an approximation, so it is tested
        # property-style across randomly shaped families.
        for trial in range(50):
            rng = np.random.default_rng(200 + trial)
            width = int(rng.integers(2, 15))
            horizon = int(rng.integers(200, 4000))
            observed = rng.normal(0.0, 0.003, width) * rng.choice([1.0, 5.0])
            draws = observed[None, :] + rng.normal(0.0, 0.003, (400, width))
            omega = validation._spa_omega(observed, draws, horizon)
            p_values, _ = validation._spa_p_values(observed, draws, horizon, omega)
            with self.subTest(trial=trial):
                self.assertLessEqual(p_values[SPA_LOWER], p_values[SPA_CONSISTENT])
                self.assertLessEqual(p_values[SPA_CONSISTENT], p_values[SPA_UPPER])

    def test_unit_studentisation_reproduces_the_reality_check_exactly(self) -> None:
        # The strongest single check available: with omega forced to one and the
        # upper recentring, SPA and the Reality Check are the same statistic, so
        # sharing the bootstrap plumbing is provable rather than asserted.
        rng = np.random.default_rng(301)
        width = 9
        horizon = 1800
        observed = rng.normal(0.0008, 0.0004, width)
        draws = observed[None, :] + rng.normal(0.0, 0.0006, (800, width))
        p_rc, _ = validation._reality_check_p_value(observed, draws, horizon)
        p_spa, _ = validation._spa_p_values(
            observed, draws, horizon, np.ones(width)
        )
        self.assertEqual(p_rc, p_spa[SPA_UPPER])

    def test_consistent_recentring_ignores_models_below_the_threshold(self) -> None:
        # Adding models far worse than the benchmark must leave the consistent
        # SPA essentially unchanged while the upper SPA - which is White's
        # recentring - deteriorates.  That contrast is the whole point of the
        # test and the reason it belongs in the promotion gate.
        rng = np.random.default_rng(302)
        base = rng.normal(0.0, 0.005, (2000, 11))
        # A genuinely superior member, sized so its studentised statistic sits
        # near two.  A larger edge saturates every p-value at the resolution
        # floor and the contrast the test exists for becomes unobservable.
        base[:, 1] = base[:, 0] + rng.normal(0.0001, 0.003, 2000)
        poor = rng.normal(-0.010, 0.005, (2000, 190))
        kwargs = {
            "statistics": (MEAN_STATISTIC,),
            "block_lengths": (21,),
            "samples": 999,
        }
        small = hansen_spa(matrix_family(base, label="power"), **kwargs)
        large = hansen_spa(
            matrix_family(np.hstack([base, poor]), label="power"), **kwargs
        )

        def value(report: pd.DataFrame, procedure: str) -> float:
            return float(
                report.loc[report["Procedure"] == procedure, "p value"].iloc[0]
            )

        self.assertLess(
            abs(value(large, SPA_CONSISTENT) - value(small, SPA_CONSISTENT)), 0.02
        )
        self.assertGreater(
            value(large, SPA_UPPER) - value(small, SPA_UPPER), 0.05
        )

    def test_zero_variance_members_are_recorded_rather_than_floored(self) -> None:
        rng = np.random.default_rng(303)
        matrix = rng.normal(0.0, 0.005, (900, 5))
        matrix[:, 4] = matrix[:, 0]
        family = matrix_family(matrix, label="degenerate")
        report = hansen_spa(
            family,
            statistics=(MEAN_STATISTIC,),
            block_lengths=(21,),
            samples=199,
        )
        self.assertTrue((report["Members without bootstrap variance"] == 1).all())

    def test_a_family_of_one_has_nothing_to_maximise_over(self) -> None:
        rng = np.random.default_rng(305)
        index = pd.bdate_range("1990-01-02", periods=900)
        family = assemble_family(
            {"benchmark": pd.Series(rng.normal(0.0, 0.005, 900), index=index)},
            label="alone",
            benchmark="benchmark",
            declared_members=("benchmark",),
            provenance="test",
        )
        report = family_wise_report(family, block_lengths=(21,), samples=99)
        self.assertTrue((report["Status"] == NOT_ESTIMABLE).all())

    def test_short_histories_are_refused(self) -> None:
        rng = np.random.default_rng(304)
        family = matrix_family(rng.normal(0.0, 0.005, (25, 4)), label="short")
        report = family_wise_report(family, block_lengths=(21,), samples=99)
        self.assertTrue((report["Status"] == NOT_ESTIMABLE).all())


class TestCscvPbo(unittest.TestCase):
    def test_combination_counts_are_exact(self) -> None:
        rng = np.random.default_rng(400)
        family = matrix_family(rng.normal(0.0, 0.005, (6289, 12)), label="combos")
        for submatrices, expected in ((8, 70), (16, 12_870)):
            with self.subTest(submatrices=submatrices):
                result = cscv_pbo(
                    family,
                    submatrices=submatrices,
                    minimum_rows_per_submatrix=252,
                )
                self.assertEqual(
                    int(result.diagnostics["combinations"]),
                    math.comb(submatrices, submatrices // 2),
                )
                self.assertEqual(int(result.diagnostics["combinations"]), expected)
                self.assertEqual(len(result.splits), expected)

    def test_a_dominant_configuration_gives_pbo_exactly_zero(self) -> None:
        # Every column shares one deterministic within-submatrix pattern and
        # differs only by a constant mean shift, so the in-sample best is always
        # the same column and it is always the out-of-sample best too.
        rows, columns = 4_032, 10
        pattern = np.tile(np.array([1.0, -1.0, 0.5, -0.5]), rows // 4)
        matrix = np.column_stack(
            [pattern * 0.004 + 0.001 * position for position in range(columns)]
        )
        family = matrix_family(matrix, label="dominant")
        result = cscv_pbo(family, submatrices=16)
        self.assertEqual(result.probability_of_backtest_overfitting.status, ESTIMATED)
        self.assertEqual(result.probability_of_backtest_overfitting.value, 0.0)
        self.assertEqual(
            set(result.splits["in_sample_selection"]), {"member_008"}
        )

    def test_a_pure_noise_family_gives_pbo_near_one_half(self) -> None:
        rng = np.random.default_rng(401)
        family = matrix_family(rng.normal(0.0002, 0.006, (6289, 50)), label="noise")
        result = cscv_pbo(family, submatrices=16)
        value = result.probability_of_backtest_overfitting.value
        self.assertIsNotNone(value)
        self.assertGreaterEqual(float(value), 0.40)
        self.assertLessEqual(float(value), 0.60)
        self.assertEqual(result.probability_of_loss.status, ESTIMATED)

    def test_refusals_fire_before_any_number_is_produced(self) -> None:
        rng = np.random.default_rng(402)
        complete = matrix_family(rng.normal(0.0, 0.006, (6289, 12)), label="ok")
        cases = {
            "odd submatrices": {"submatrices": 15},
            "too few submatrices": {"submatrices": 6},
            "submatrix too short": {"submatrices": 32},
            "too few configurations": {"minimum_configurations": 40},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                result = cscv_pbo(complete, **kwargs)
                self.assertEqual(
                    result.probability_of_backtest_overfitting.status, NOT_ESTIMABLE
                )
                self.assertIsNone(result.probability_of_backtest_overfitting.value)
                self.assertTrue(result.splits.empty)
        incomplete = matrix_family(
            rng.normal(0.0, 0.006, (6289, 12)),
            label="incomplete",
            declared_extra=("never_run_a", "never_run_b"),
        )
        result = cscv_pbo(incomplete)
        self.assertEqual(
            result.probability_of_backtest_overfitting.status, NOT_ESTIMABLE
        )
        self.assertIn("incomplete", result.probability_of_backtest_overfitting.reason)

    def test_rows_are_dropped_from_the_front(self) -> None:
        rng = np.random.default_rng(403)
        family = matrix_family(rng.normal(0.0, 0.006, (6289, 12)), label="trim")
        result = cscv_pbo(family, submatrices=16)
        self.assertEqual(int(result.diagnostics["rows_dropped"]), 6289 - 16 * 393)
        self.assertEqual(int(result.diagnostics["rows_per_submatrix"]), 393)

    def test_the_row_floor_follows_the_family_frequency(self) -> None:
        # research-methodology.md permits CSCV/PBO only on monthly paths.  A
        # single hard-coded daily floor refuses every such family with a
        # message about needing 252 monthly observations, which both reads as
        # nonsense and refuses the only input the document allows.
        rng = np.random.default_rng(405)
        rows, columns = 300, 17
        index = pd.date_range("1990-01-31", periods=rows, freq="ME")
        matrix = rng.normal(0.006, 0.02, (rows, columns))
        names = ["benchmark"] + [f"member_{i:03d}" for i in range(columns - 1)]
        monthly = validation.assemble_family(
            {n: pd.Series(matrix[:, i], index=index) for i, n in enumerate(names)},
            label="monthly",
            benchmark="benchmark",
            declared_members=tuple(names),
            provenance="test",
            frequency=validation.MONTHLY_FREQUENCY,
            annualization=12,
        )
        self.assertEqual(
            validation.MINIMUM_ROWS_PER_SUBMATRIX_BY_FREQUENCY[
                validation.MONTHLY_FREQUENCY
            ],
            12,
        )
        result = cscv_pbo(monthly, submatrices=16)
        self.assertEqual(result.probability_of_backtest_overfitting.status, ESTIMATED)
        self.assertEqual(int(result.diagnostics["rows_per_submatrix"]), 18)
        self.assertEqual(int(result.diagnostics["minimum_rows_per_submatrix"]), 12)
        # A monthly family too short for even a year per submatrix still
        # refuses, and the refusal is worded in its own frequency.
        short = cscv_pbo(monthly, submatrices=32)
        self.assertEqual(
            short.probability_of_backtest_overfitting.status, NOT_ESTIMABLE
        )
        self.assertIn("monthly", short.probability_of_backtest_overfitting.reason)
        self.assertIn(
            "the floor is 12", short.probability_of_backtest_overfitting.reason
        )
        # The daily default is untouched.
        daily = matrix_family(rng.normal(0.0, 0.006, (6289, 12)), label="daily")
        self.assertEqual(
            int(cscv_pbo(daily, submatrices=16).diagnostics[
                "minimum_rows_per_submatrix"
            ]),
            252,
        )

    def test_repeated_runs_are_identical(self) -> None:
        rng = np.random.default_rng(404)
        family = matrix_family(rng.normal(0.0, 0.006, (6289, 12)), label="determinism")
        first = cscv_pbo(family, submatrices=16)
        second = cscv_pbo(family, submatrices=16)
        self.assertEqual(
            first.probability_of_backtest_overfitting.value,
            second.probability_of_backtest_overfitting.value,
        )
        pd.testing.assert_frame_equal(first.splits, second.splits)


class TestDeflatedSharpe(unittest.TestCase):
    def test_higher_moment_substitution_reproduces_the_published_denominator(
        self,
    ) -> None:
        # The published coefficient is (gamma_4 - 1)/4 on non-excess kurtosis.
        # inference.deflated_sharpe_ratio forms it from the excess figure as
        # (excess + 2)/4, so a Gaussian series is passed excess kurtosis 0.0
        # and the whole denominator collapses to sqrt(1 + SR^2/2).  Supplying a
        # pre-compensated gamma_4 - 1 here, as the caller once had to, would
        # now double-correct.
        sessions = 6289
        for sharpe in (0.05, 0.10, 0.50, 1.00):
            with self.subTest(sharpe=sharpe):
                result = inference.deflated_sharpe_ratio(
                    sharpe,
                    trial_count=1,
                    trial_sharpe_variance=1.0,
                    sessions=sessions,
                    skewness=0.0,
                    excess_kurtosis=0.0,
                )
                expected = NormalDist().cdf(
                    sharpe
                    * math.sqrt(sessions - 1.0)
                    / math.sqrt(1.0 + sharpe**2 / 2.0)
                )
                self.assertAlmostEqual(float(result.value), expected, places=12)

    def test_the_declared_family_cannot_stand_in_for_the_search(self) -> None:
        # The deflated Sharpe deflates by the number of trials SEARCHED.  A
        # declared family is the number REGISTERED, a lower bound.  Reporting
        # the resulting probability as ESTIMATED flips
        # inference.promotion_support to True, so a seventeen-configuration
        # family would clear the methodology document's deflated-Sharpe gate
        # for every one of its members - including its worst.
        rng = np.random.default_rng(500)
        family = matrix_family(rng.normal(0.0004, 0.006, (6289, 12)), label="dsr")
        row = family_deflated_sharpe(family, member="member_000")
        self.assertEqual(row["Status"], NOT_ESTIMABLE)
        self.assertIsNone(row["Value"])
        support = inference.promotion_support(
            inference.InferenceResult(
                statistic=str(row["Statistic"]),
                status=str(row["Status"]),
                value=row["Value"],
                reason=str(row["Reason"]),
            ),
            window_label="fixture",
        )
        self.assertFalse(bool(support["Promotion supported"]))

    def test_the_illustrative_value_uses_a_per_period_sharpe(self) -> None:
        rng = np.random.default_rng(500)
        family = matrix_family(rng.normal(0.0004, 0.006, (6289, 12)), label="dsr")
        row = family_deflated_sharpe(family, member="member_000")
        # Feeding the annualized Sharpe with a daily session count saturates the
        # statistic at exactly 1.0; the conversion is what prevents that.
        illustrative = float(row["Lower-bound deflated probability (illustrative)"])
        self.assertLess(illustrative, 1.0)
        self.assertEqual(int(row["Declared trials"]), 12)
        reported = float(row["Per-period Sharpe"])
        column = family.paths["member_000"]
        self.assertAlmostEqual(
            reported, float(column.mean() / column.std(ddof=0)), places=12
        )
        # The annualized Sharpe is fifteen-odd times larger; reporting it here
        # would be the exact input error that saturates the statistic.
        self.assertLess(abs(reported), 1.0)

    def test_the_reported_threshold_is_the_one_the_statistic_deflates_by(self) -> None:
        # The reason the illustrative number does not discriminate inside a
        # family is the threshold, not floating point: recomputing the deflated
        # probability at a Sharpe exactly equal to the reported threshold must
        # land on one half, which is only true if the reported threshold is the
        # same Gumbel expected maximum inference.deflated_sharpe_ratio uses.
        rng = np.random.default_rng(502)
        family = matrix_family(rng.normal(0.0004, 0.006, (4000, 17)), label="thresh")
        row = family_deflated_sharpe(family, member="benchmark")
        threshold = float(row["Deflation threshold (per period)"])
        self.assertGreater(threshold, 0.0)
        at_threshold = inference.deflated_sharpe_ratio(
            threshold,
            trial_count=int(row["Declared trials"]),
            trial_sharpe_variance=float(row["Trial Sharpe variance (per period)"]),
            sessions=int(row["Observations"]),
            skewness=0.0,
            excess_kurtosis=2.0,
        )
        self.assertAlmostEqual(float(at_threshold.value), 0.5, places=12)
        self.assertAlmostEqual(
            float(row["Deflation threshold (annualized)"]),
            threshold * math.sqrt(252),
            places=12,
        )
        # The disclosure that matters: the threshold is far below every member,
        # so the statistic is not discriminating within the family.
        annualized = float(row["Per-period Sharpe"]) * math.sqrt(252)
        self.assertLess(float(row["Deflation threshold (annualized)"]), abs(annualized))

    def test_incomplete_family_refuses(self) -> None:
        rng = np.random.default_rng(501)
        family = matrix_family(
            rng.normal(0.0004, 0.006, (2000, 6)),
            label="dsr-incomplete",
            declared_extra=("absent",),
        )
        row = family_deflated_sharpe(family)
        self.assertEqual(row["Status"], NOT_ESTIMABLE)
        self.assertIsNone(row["Value"])


class TestPathStatistics(unittest.TestCase):
    def test_cagr_is_annualized_on_elapsed_calendar_time(self) -> None:
        # The session grid is every weekday, so it carries about 261 sessions
        # per calendar year rather than 252.  Annualizing by
        # observations/annualization inflates the elapsed period by 3.5% and
        # understates CAGR by roughly the same proportion against
        # strategy.performance_metrics, with nothing in the artifact to say so.
        rate = 0.0004
        periods = 2610
        index = pd.bdate_range("1990-01-01", periods=periods)
        returns = pd.Series(rate, index=index)
        statistics = validation._path_statistics(
            returns, annualization=252, hac_lags=21
        )
        years = (index[-1] - index[0]).days / 365.2425
        expected = (1.0 + rate) ** (periods / years) - 1.0
        self.assertAlmostEqual(statistics["cagr"], expected, places=12)
        session_convention = (1.0 + rate) ** 252.0 - 1.0
        self.assertGreater(abs(statistics["cagr"] - session_convention), 1e-3)

    def test_the_period_start_anchor_matches_performance_metrics(self) -> None:
        # performance_metrics measures elapsed time from the session BEFORE the
        # first in-window return.  Passing that session reproduces it exactly,
        # which is what lets a validation CAGR be compared with the canonical
        # bundle instead of merely resembling it.
        frames = synthetic_frames(markets=2, sessions=900)
        config = synthetic_config()
        result = run_backtest(config, **frames)
        start, end = "1991-01-01", frames["prices"].index[-1].date().isoformat()
        metrics = performance_metrics(result, start, end, 252)
        window = result.daily.loc[start:end, "net_return"].astype(float)
        location = result.daily.index.get_indexer([window.index[0]])[0]
        anchor = result.daily.index[location - 1]
        statistics = validation._path_statistics(
            window, annualization=252, hac_lags=21, period_start=anchor
        )
        self.assertAlmostEqual(statistics["cagr"], float(metrics["CAGR"]), places=12)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_reproduces_every_p_value(self) -> None:
        rng = np.random.default_rng(600)
        family = matrix_family(rng.normal(0.0, 0.005, (1200, 8)), label="repeat")
        kwargs = {"block_lengths": (21, 63), "samples": 499}
        first = family_wise_report(family, **kwargs)
        second = family_wise_report(family, **kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_reality_check_matches_whether_run_alone_or_jointly(self) -> None:
        rng = np.random.default_rng(601)
        family = matrix_family(rng.normal(0.0, 0.005, (1200, 8)), label="shared")
        kwargs = {
            "statistics": (MEAN_STATISTIC,),
            "block_lengths": (63,),
            "samples": 499,
        }
        alone = reality_check(family, **kwargs)
        joint = family_wise_report(family, **kwargs)
        joint_rc = joint.loc[joint["Procedure"] == REALITY_CHECK].reset_index(drop=True)
        pd.testing.assert_series_equal(
            alone["p value"], joint_rc["p value"], check_names=False
        )


class TestWalkForwardStateContinuity(unittest.TestCase):
    """The tests that catch a restarted fold NAV.

    A per-fold engine call with fresh initial capital re-warms the portfolio
    risk scalar to 1.0 at every boundary, which exceeds the strategy's measured
    average scalar, so every fold would start over-levered relative to the true
    path.  The identity below fails immediately for any such implementation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.frames = synthetic_frames()
        cls.config = synthetic_config()
        cls.baseline = run_backtest(cls.config, **cls.frames)
        cls.end = cls.frames["prices"].index[-1].date().isoformat()

    def walk_forward_config(self, **overrides: Any) -> WalkForwardConfig:
        base = {
            "boundaries": ("1994-01-01", "1995-01-01", "1996-01-01", "1997-01-01"),
            "anchor": "1990-01-01",
            "end": self.end,
        }
        base.update(overrides)
        return WalkForwardConfig(**base)

    def test_constant_parameters_reproduce_the_single_shot_backtest(self) -> None:
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
        )
        self.assertEqual(report.mode, WALK_FORWARD_STABILITY_MODE)
        np.testing.assert_allclose(
            report.daily["net_return"].to_numpy(dtype=float),
            self.baseline.daily["net_return"].to_numpy(dtype=float),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            report.daily["nav"].to_numpy(dtype=float),
            self.baseline.daily["nav"].to_numpy(dtype=float),
            atol=1e-9,
        )

    def test_a_pinned_selector_reproduces_that_candidate_exactly(self) -> None:
        # Exercises the full selection path with a non-default specification,
        # so the identity covers the splice itself rather than only the
        # degenerate single-candidate case.
        candidates = (
            WalkForwardCandidate(
                "fast", "126-session trend lookback", (("trend_lookback", 126),)
            ),
            WalkForwardCandidate("slow", "252-session trend lookback"),
        )
        fast = run_backtest(candidates[0].apply(self.config), **self.frames)
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
            candidates=candidates,
            select_fn=lambda training: ("fast", float("nan")),
        )
        self.assertEqual(report.mode, WALK_FORWARD_SELECTION_MODE)
        np.testing.assert_allclose(
            report.daily["net_return"].to_numpy(dtype=float),
            fast.daily["net_return"].to_numpy(dtype=float),
            atol=1e-12,
        )

    def test_a_fold_inherits_the_book_the_previous_fold_left(self) -> None:
        # The discriminating test.  With the anchor specification held before
        # the first boundary and a different specification selected from that
        # boundary on, the spliced intents match the selected candidate exactly
        # while the realised path does not - because the book crossing the
        # boundary is the one the previous regime left.  An implementation that
        # restarted the fold's NAV would instead reproduce the candidate's own
        # standalone path here, which is precisely the flattering error.
        candidates = (
            WalkForwardCandidate("slow", "252-session trend lookback"),
            WalkForwardCandidate(
                "fast", "126-session trend lookback", (("trend_lookback", 126),)
            ),
        )
        fast = run_backtest(candidates[1].apply(self.config), **self.frames)
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
            candidates=candidates,
            select_fn=lambda training: ("fast", float("nan")),
        )
        boundary = pd.Timestamp("1994-01-01")
        intents = report.decision_intents
        after_decisions = intents.index >= boundary
        pd.testing.assert_frame_equal(
            intents.loc[after_decisions],
            fast.target_positions.loc[after_decisions],
        )
        after = report.daily.index >= boundary
        stitched = report.daily.loc[after, "net_return"].to_numpy(dtype=float)
        standalone = fast.daily.loc[after, "net_return"].to_numpy(dtype=float)
        self.assertGreater(float(np.abs(stitched - standalone).max()), 1e-6)

    def test_spliced_intents_equal_the_selected_candidate_row_by_row(self) -> None:
        candidates = (
            WalkForwardCandidate("slow", "252-session trend lookback"),
            WalkForwardCandidate(
                "fast", "126-session trend lookback", (("trend_lookback", 126),)
            ),
        )
        order = iter(["fast", "slow", "fast", "slow"])
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
            candidates=candidates,
            select_fn=lambda training: (next(order), 0.0),
        )
        runs = {
            candidate.name: run_backtest(candidate.apply(self.config), **self.frames)
            for candidate in candidates
        }
        intents = report.decision_intents
        for _, fold in report.folds.iterrows():
            start = pd.Timestamp(fold["segment_start"])
            stop = pd.Timestamp(fold["segment_end"])
            window = (intents.index >= start) & (intents.index <= stop)
            if not window.any():
                continue
            expected = runs[fold["selected_variant"]].target_positions.loc[window]
            with self.subTest(fold=int(fold["fold"])):
                pd.testing.assert_frame_equal(intents.loc[window], expected)

    def test_the_folds_partition_the_stitched_path_exactly(self) -> None:
        # Every session between the first boundary and the end belongs to one
        # fold and no session belongs to two.  A gap would silently drop a
        # segment from the diagnostic; an overlap would double-count one.
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
        )
        covered = 0
        edges: list[pd.Timestamp] = []
        for _, fold in report.folds.iterrows():
            covered += int(fold["fold_sessions"])
            edges.append(pd.Timestamp(fold["segment_start"]))
        self.assertEqual(covered, len(report.stitched_path))
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(
            report.stitched_path.index[0],
            report.daily.loc[report.daily.index >= edges[0]].index[0],
        )

    def test_walk_forward_efficiency_compares_one_span_with_itself(self) -> None:
        # With no selection the stitched path IS the frozen specification's
        # path, so the efficiency must be exactly 1.0.  Dividing by the anchor
        # window's Sharpe instead compares 1995-2014 against 1990-2014 and
        # books the training years as inefficiency in a procedure that performs
        # no selection at all.
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
        )
        summary = report.summary
        self.assertEqual(summary["mode"], WALK_FORWARD_STABILITY_MODE)
        self.assertEqual(float(summary["walk_forward_efficiency"]), 1.0)
        self.assertEqual(
            float(summary["anchor_specification_stitched_span_sharpe"]),
            float(summary["stitched_sharpe"]),
        )
        # The full-window figure is still published, under a name that says so,
        # and on this fixture it is a different number.
        self.assertNotAlmostEqual(
            float(summary["anchor_specification_full_window_sharpe"]),
            float(summary["stitched_sharpe"]),
            places=6,
        )

    def test_a_fold_zero_switch_away_from_the_warm_up_is_counted(self) -> None:
        # Every decision before the first boundary comes from the first
        # declared candidate, so a fold-zero selection of anything else is a
        # real discontinuity in the spliced intent frame.  Reporting it as no
        # switch understates the count the intent-state caveat leans on.
        candidates = (
            WalkForwardCandidate("slow", "252-session trend lookback"),
            WalkForwardCandidate(
                "fast", "126-session trend lookback", (("trend_lookback", 126),)
            ),
        )
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
            candidates=candidates,
            select_fn=lambda training: ("fast", 0.0),
        )
        self.assertEqual(report.summary["warm_up_variant"], "slow")
        self.assertTrue(bool(report.folds.loc[0, "selection_switched"]))
        self.assertEqual(int(report.summary["selection_switches"]), 1)
        # Selecting the warm-up regime itself is not a switch.
        held = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
            candidates=candidates,
            select_fn=lambda training: ("slow", 0.0),
        )
        self.assertEqual(int(held.summary["selection_switches"]), 0)

    def test_switching_regimes_is_charged_as_turnover(self) -> None:
        candidates = (
            WalkForwardCandidate("slow", "252-session trend lookback"),
            WalkForwardCandidate(
                "fast", "126-session trend lookback", (("trend_lookback", 126),)
            ),
        )
        order = iter(["slow", "fast", "slow", "fast"])
        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=self.walk_forward_config(),
            candidates=candidates,
            select_fn=lambda training: (next(order), 0.0),
        )
        self.assertEqual(int(report.summary["selection_switches"]), 3)
        self.assertGreater(
            float(report.summary["stitched_annual_rebalance_turnover_contracts"]),
            0.0,
        )


class TestWalkForwardCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frames = synthetic_frames()
        cls.config = synthetic_config()
        cls.end = cls.frames["prices"].index[-1].date().isoformat()

    def test_the_selector_never_sees_a_date_at_or_after_its_boundary(self) -> None:
        boundaries = ("1994-01-01", "1995-01-01", "1996-01-01", "1997-01-01")
        seen: list[pd.Timestamp] = []

        def recording(training: pd.DataFrame) -> tuple[str, float]:
            seen.append(training.index.max())
            return "slow", 0.0

        anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=WalkForwardConfig(
                boundaries=boundaries, anchor="1990-01-01", end=self.end
            ),
            candidates=(
                WalkForwardCandidate("slow", "252-session trend lookback"),
                WalkForwardCandidate(
                    "fast", "126-session trend lookback", (("trend_lookback", 126),)
                ),
            ),
            select_fn=recording,
        )
        self.assertEqual(len(seen), len(boundaries))
        for observed, boundary in zip(seen, boundaries):
            with self.subTest(boundary=boundary):
                self.assertLess(observed, pd.Timestamp(boundary))

    def test_an_engineered_future_cannot_change_an_early_selection(self) -> None:
        # Two candidates share one decision frame and differ only in the return
        # path the selector scores.  "early" earns before the pivot and loses
        # after it; "late" does the reverse.  Over the whole history "late" is
        # the better path, so a selector reading any future data would choose it
        # at the first boundary.  Reading only the training span, it cannot.
        baseline = run_backtest(self.config, **self.frames)
        index = baseline.daily.index
        pivot = pd.Timestamp("1993-06-01")
        before = (index < pivot).astype(float)
        rng = np.random.default_rng(20_260_806)
        noise = rng.normal(0.0, 0.004, len(index))
        edge = np.where(before > 0, 0.001, -0.001)
        early = pd.Series(edge + noise, index=index)
        late = pd.Series(-edge + noise, index=index)

        def runner(candidate_config: Any) -> StubResult:
            path = early if candidate_config.trend_lookback == 252 else late
            daily = baseline.daily.copy()
            daily["net_return"] = path
            return StubResult(daily=daily, target_positions=baseline.target_positions)

        candidates = (
            WalkForwardCandidate("early", "engineered pre-pivot edge"),
            WalkForwardCandidate(
                "late", "engineered post-pivot edge", (("trend_lookback", 126),)
            ),
        )
        walk_forward = WalkForwardConfig(
            boundaries=("1992-01-01", "1993-01-01", "1998-01-01"),
            anchor="1990-01-01",
            end=self.end,
        )
        full_history_choice = late.loc[index >= pd.Timestamp("1990-01-01")].mean()
        self.assertGreater(full_history_choice, early.mean())

        report = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=walk_forward,
            candidates=candidates,
            run_fn=runner,
        )
        selections = list(report.folds["selected_variant"])
        self.assertEqual(selections, ["early", "early", "late"])

    def test_truncating_the_future_leaves_earlier_folds_unchanged(self) -> None:
        # The mechanical form of the causality contract: a fold statistic that
        # moved when later data were removed would prove a forward read.
        boundaries = ("1994-01-01", "1995-01-01", "1996-01-01", "1997-01-01")
        full = anchored_walk_forward(
            self.config,
            frames=self.frames,
            walk_forward=WalkForwardConfig(
                boundaries=boundaries, anchor="1990-01-01", end=self.end
            ),
        )
        cut = pd.Timestamp("1997-01-01")
        truncated_frames = {
            name: frame.loc[frame.index < cut] if isinstance(frame.index, pd.DatetimeIndex)
            else frame
            for name, frame in self.frames.items()
        }
        truncated = anchored_walk_forward(
            self.config,
            frames=truncated_frames,
            walk_forward=WalkForwardConfig(
                boundaries=boundaries[:3],
                anchor="1990-01-01",
                end="1996-12-31",
            ),
        )
        columns = ["fold_cagr", "fold_sharpe", "fold_max_drawdown", "fold_hit_rate"]
        np.testing.assert_allclose(
            full.folds.loc[:1, columns].to_numpy(dtype=float),
            truncated.folds.loc[:1, columns].to_numpy(dtype=float),
            atol=1e-12,
        )


class TestWalkForwardValidation(unittest.TestCase):
    def test_configuration_rejects_impossible_schedules(self) -> None:
        cases = {
            "no boundary": {"boundaries": ()},
            "unsorted": {"boundaries": ("1996-01-01", "1995-01-01")},
            "duplicated": {"boundaries": ("1995-01-01", "1995-01-01")},
            "before the anchor": {"boundaries": ("1989-01-01",)},
            "after the end": {"boundaries": ("2020-01-01",)},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                WalkForwardConfig(anchor="1990-01-01", end="2014-12-31", **kwargs)

    def test_partial_frames_are_refused(self) -> None:
        frames = synthetic_frames(markets=2, sessions=400)
        frames.pop("delivery_months")
        with self.assertRaises(ValueError) as raised:
            anchored_walk_forward(
                synthetic_config(),
                frames=frames,
                walk_forward=WalkForwardConfig(
                    boundaries=("1991-01-01",), anchor="1990-01-01", end="1991-06-28"
                ),
            )
        self.assertIn("delivery_months", str(raised.exception))

    def test_execution_ledger_overrides_are_refused_not_silently_dropped(self) -> None:
        # The splice carries the monthly intent frame and simulates ONE ledger,
        # built from the base configuration.  A candidate declaring different
        # costs, capacity, leverage or timing is scored under its own
        # specification and would then be replayed under the base one, with
        # nothing in the fold table recording the substitution.  Measured on
        # this fixture before the guard existed: a candidate with
        # commission_per_contract = 50.0, pinned as the selection, replayed at
        # the base annual cost drag of 0.00327 instead of its own 0.03330.
        frames = synthetic_frames(markets=2, sessions=900)
        schedule = WalkForwardConfig(
            boundaries=("1992-01-01", "1993-01-01"),
            anchor="1990-01-01",
            end="1993-06-01",
        )
        refused = {
            "half_spread_ticks": 5.0,
            "slippage_ticks": 4.0,
            "commission_per_contract": 50.0,
            "exchange_and_regulatory_fees_per_contract": 25.0,
            "impact_bps_at_full_participation": 100.0,
            "max_gross_notional_multiple": 1.5,
            "max_static_margin_fraction": 0.10,
            "max_rebalance_participation": 0.01,
            "cap_fills_at_realized_capacity": False,
            "execution_delay_sessions": 1,
            "execution_timing": "next_open",
            "integer_contracts": False,
            "charge_roll_costs": False,
            "initial_capital": 5_000_000.0,
            "launch_date": "1991-01-01",
            "volume_gate_window": 30,
        }
        for key, value in refused.items():
            with self.subTest(field=key):
                with self.assertRaises(ValueError) as raised:
                    anchored_walk_forward(
                        synthetic_config(),
                        frames=frames,
                        walk_forward=schedule,
                        candidates=(
                            WalkForwardCandidate("base", "base configuration"),
                            WalkForwardCandidate("over", f"{key}", ((key, value),)),
                        ),
                    )
                self.assertIn(key, str(raised.exception))
        # Intent-level dimensions are still selectable.
        anchored_walk_forward(
            synthetic_config(),
            frames=frames,
            walk_forward=schedule,
            candidates=(
                WalkForwardCandidate("base", "base configuration"),
                WalkForwardCandidate(
                    "fast", "126-session lookback", (("trend_lookback", 126),)
                ),
            ),
            select_fn=lambda training: ("fast", 0.0),
        )

    def test_the_spliceable_allow_list_excludes_every_ledger_field(self) -> None:
        # Structural rather than enumerated: the allow-list is checked against
        # the engine source, so a field that starts being read by the execution
        # ledger cannot stay spliceable by oversight.
        import inspect
        import re

        from delta1_strategy.research import strategy as strategy_module

        sources = "".join(
            inspect.getsource(function)
            for function in (
                strategy_module._simulate_execution,
                strategy_module._limit_target_contracts,
                validation._execution_inputs,
            )
        )
        ledger_fields = set(re.findall(r"config\.(\w+)", sources))
        self.assertIn("half_spread_ticks", ledger_fields)
        self.assertIn("max_gross_notional_multiple", ledger_fields)
        overlap = ledger_fields & validation.SPLICEABLE_CONFIG_FIELDS
        self.assertEqual(overlap, set())
        declared = {field.name for field in dataclasses.fields(StrategyConfig)}
        self.assertEqual(validation.SPLICEABLE_CONFIG_FIELDS - declared, set())

    def test_an_anchor_without_enough_history_is_refused(self) -> None:
        frames = synthetic_frames(markets=2, sessions=800)
        with self.assertRaises(ValueError) as raised:
            anchored_walk_forward(
                synthetic_config(),
                frames=frames,
                walk_forward=WalkForwardConfig(
                    boundaries=("1990-06-01",),
                    anchor="1990-01-01",
                    end="1992-12-31",
                ),
            )
        self.assertIn("warm", str(raised.exception))


class TestReportConventions(unittest.TestCase):
    def test_no_emitted_column_can_read_as_a_promotion_decision(self) -> None:
        rng = np.random.default_rng(700)
        family = matrix_family(rng.normal(0.0, 0.005, (6289, 12)), label="labels")
        frames = [
            family_wise_report(family, block_lengths=(21,), samples=99),
            cscv_pbo(family, submatrices=16).to_frame(),
            cscv_pbo(family, submatrices=16).splits,
            pd.DataFrame(validation.WALK_FORWARD_COLUMNS).T,
            family_deflated_sharpe(family).to_frame().T,
        ]
        for position, frame in enumerate(frames):
            for column in frame.columns:
                lowered = str(column).lower()
                for banned in BANNED_SUBSTRINGS:
                    with self.subTest(frame=position, column=column, banned=banned):
                        self.assertNotIn(banned, lowered)

    def test_walk_forward_columns_carry_no_banned_substring(self) -> None:
        for column in validation.WALK_FORWARD_COLUMNS:
            for banned in BANNED_SUBSTRINGS:
                with self.subTest(column=column, banned=banned):
                    self.assertNotIn(banned, column.lower())

    def test_no_report_ranks_or_recommends(self) -> None:
        rng = np.random.default_rng(701)
        family = matrix_family(rng.normal(0.0, 0.005, (1000, 6)), label="ranking")
        report = family_wise_report(family, block_lengths=(21,), samples=99)
        for column in report.columns:
            lowered = column.lower()
            for banned in ("rank", "winner", "recommend", "score"):
                with self.subTest(column=column, banned=banned):
                    self.assertNotIn(banned, lowered)


if __name__ == "__main__":
    unittest.main()
