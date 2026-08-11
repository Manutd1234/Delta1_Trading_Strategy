"""The cost-breakeven artifact must match its manifest and the canonical bundle.

``scripts/run_cost_breakeven.py`` prices the one frozen strategy under a
declared ladder of execution-cost multiples.  Three things make that artifact
trustworthy, and each is pinned here: its 1.0x row is the published
full-history row of ``outputs/strategy_metrics.csv``, its grid is exactly the
manifest the script declared before reading any file, and its Sharpe falls
monotonically as costs rise -- the shape that makes a linear breakeven
estimate meaningful at all.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "Round1AllData/Quant Researcher/Delta1"))
ARTIFACT = ROOT_DIR / "outputs" / "validation" / "validation_cost_breakeven.csv"


def _load_script():
    """Import the sweep script by path, without running its main()."""
    path = ROOT_DIR / "scripts" / "run_cost_breakeven.py"
    spec = importlib.util.spec_from_file_location("run_cost_breakeven", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    ARTIFACT.exists() and DATA_DIR.exists(),
    "Cost-breakeven artifact or supplied DELTA1 data directory is not available",
)
class TestCostBreakevenArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_script()
        cls.frame = pd.read_csv(ARTIFACT)
        cls.grid = cls.frame.loc[cls.frame["row_type"].eq("grid")]

    def test_unit_multiplier_row_is_the_published_full_history_row(self) -> None:
        """The 1.0x run and strategy_metrics.csv must agree to 12 places."""
        row = self.grid.loc[self.grid["cost_multiplier"].eq(1.0)]
        self.assertEqual(len(row), 1)
        published = pd.read_csv(ROOT_DIR / "outputs" / "strategy_metrics.csv")
        full = published.loc[
            published["Window"].eq("1990-2014 full post-launch history")
        ]
        self.assertEqual(len(full), 1)
        for field, column in (
            ("net_sharpe", "Naive daily Sharpe (sqrt252, rf=0)"),
            ("net_cagr", "CAGR"),
            ("max_drawdown", "Max drawdown"),
            ("annual_cost_drag", "Annual cost drag"),
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    float(row.iloc[0][field]), float(full.iloc[0][column]), places=12
                )

    def test_sharpe_is_monotone_in_the_cost_multiplier(self) -> None:
        ordered = self.grid.sort_values("cost_multiplier")["net_sharpe"]
        self.assertTrue(
            ordered.is_monotonic_decreasing,
            f"net Sharpe is not monotone in the multiplier: {ordered.tolist()}",
        )

    def test_artifact_grid_equals_the_declared_manifest(self) -> None:
        self.assertEqual(
            tuple(self.grid["cost_multiplier"]), self.script.MULTIPLIER_GRID
        )

    def test_breakeven_rows_cover_both_declared_conventions(self) -> None:
        breakeven = self.frame.loc[self.frame["row_type"].eq("breakeven")]
        self.assertEqual(
            sorted(breakeven["crossing_metric"]), ["net_cagr", "net_sharpe"]
        )
        for _, row in breakeven.iterrows():
            with self.subTest(metric=row["crossing_metric"]):
                self.assertIn("estimate, not a measurement", row["convention"])
                self.assertEqual(
                    row["within_declared_grid"],
                    "EXTRAPOLATION" not in row["convention"],
                )


if __name__ == "__main__":
    unittest.main()
