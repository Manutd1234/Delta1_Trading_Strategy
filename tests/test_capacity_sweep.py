"""Regression tests for the standalone capacity-sweep runner."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_capacity_sweep import (
    CAPITAL_MULTIPLES,
    FROZEN_INITIAL_CAPITAL,
    PUBLISHED_WINDOW,
    SHARPE_COLUMN,
    SHARPE_EROSION_THRESHOLDS,
    erosion_crossing_capital,
)

ARTIFACT = Path("outputs/validation/validation_capacity.csv")
METRICS = Path("outputs/strategy_metrics.csv")
DATA_DIR = Path("Round1AllData/Quant Researcher/Delta1")


class TestErosionCrossingConvention(unittest.TestCase):
    def test_bracketed_threshold_interpolates_in_log_capital(self) -> None:
        capital, method = erosion_crossing_capital((1e6, 1e7), (0.0, 0.2), 0.10)
        self.assertEqual(method, "interpolated_within_grid")
        self.assertAlmostEqual(capital, 10.0**6.5, places=4)

    def test_threshold_beyond_grid_is_labelled_extrapolation(self) -> None:
        capital, method = erosion_crossing_capital((1e6, 1e7), (0.0, 0.2), 0.40)
        self.assertEqual(method, "extrapolated_beyond_final_grid_level")
        self.assertAlmostEqual(math.log10(capital), 8.0, places=10)

    def test_flat_final_slope_refuses_to_extrapolate(self) -> None:
        capital, method = erosion_crossing_capital((1e6, 1e7), (0.05, 0.05), 0.40)
        self.assertTrue(math.isnan(capital))
        self.assertEqual(method, "not_estimable_final_slope_nonpositive")

    def test_a_single_completed_level_cannot_locate_a_crossing(self) -> None:
        capital, method = erosion_crossing_capital((1e6,), (0.0,), 0.10)
        self.assertTrue(math.isnan(capital))
        self.assertEqual(method, "not_estimable_insufficient_completed_levels")


@unittest.skipUnless(
    ARTIFACT.exists() and METRICS.exists() and DATA_DIR.exists(),
    "Capacity artifact, published metrics, or supplied DELTA1 data directory "
    "is not available",
)
class TestCapacityArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = pd.read_csv(ARTIFACT, float_precision="round_trip")
        cls.levels = cls.table.loc[
            cls.table["row_type"] == "capacity_level"
        ].reset_index(drop=True)
        cls.metrics = pd.read_csv(METRICS, float_precision="round_trip")

    def test_grid_equals_declared_manifest(self) -> None:
        self.assertEqual(
            list(self.levels["capital_multiple"]),
            [float(multiple) for multiple in CAPITAL_MULTIPLES],
        )
        self.assertEqual(
            list(self.levels["initial_capital_usd"]),
            [multiple * FROZEN_INITIAL_CAPITAL for multiple in CAPITAL_MULTIPLES],
        )

    def test_1x_row_reproduces_the_published_full_history_sharpe(self) -> None:
        published = float(
            self.metrics.loc[self.metrics["Window"] == PUBLISHED_WINDOW].iloc[0][
                SHARPE_COLUMN
            ]
        )
        row = self.levels.iloc[0]
        self.assertEqual(float(row["capital_multiple"]), 1.0)
        self.assertEqual(row["run_outcome"], "completed")
        self.assertAlmostEqual(float(row["sharpe"]), published, places=12)

    def test_cost_drag_is_non_decreasing_in_capital(self) -> None:
        completed = self.levels.loc[self.levels["run_outcome"] == "completed"]
        drags = completed["annual_cost_drag"].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(drags).all())
        self.assertTrue((np.diff(drags) >= -1e-15).all())

    def test_erosion_threshold_rows_state_their_convention(self) -> None:
        thresholds = self.table.loc[
            self.table["row_type"] == "sharpe_erosion_threshold"
        ]
        self.assertEqual(
            list(thresholds["erosion_threshold"]), list(SHARPE_EROSION_THRESHOLDS)
        )
        self.assertTrue(thresholds["estimation_method"].notna().all())
        self.assertTrue(thresholds["estimation_convention"].str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main()
