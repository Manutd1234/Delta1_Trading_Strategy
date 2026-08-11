"""The jackknife artifact must agree with the frozen baseline and its manifest.

The artifact is descriptive: rows in declared manifest order, no verdicts.
These tests pin the structural facts --- the baseline row is the published
full-history result, the row order is the pre-registered order, and every
``markets_remaining`` count follows from the reference universe --- and leave
the jackknife figures themselves unasserted, because they are properties of
one realized path.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_universe_jackknife import (
    BASELINE_LABEL,
    EXCLUSION_MANIFEST,
    PUBLISHED_BASELINE_SHARPE,
    REPORT_COLUMNS,
    VALIDATION_STATUS,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "Round1AllData/Quant Researcher/Delta1"))
ARTIFACT = ROOT_DIR / "outputs" / "validation" / "validation_universe_jackknife.csv"
PUBLISHED_METRICS = ROOT_DIR / "outputs" / "strategy_metrics.csv"
FULL_HISTORY_WINDOW = "1990-2014 full post-launch history"


def _load_reference():
    """Import the reference file by path, the way a reader would run it."""
    path = ROOT_DIR / "reference" / "delta1_reference.py"
    spec = importlib.util.spec_from_file_location("delta1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    ARTIFACT.is_file() and PUBLISHED_METRICS.is_file() and DATA_DIR.exists(),
    "Jackknife artifact, published metrics, or supplied DELTA1 data are not available",
)
class TestUniverseJackknifeArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.read_csv(ARTIFACT)
        metrics = pd.read_csv(PUBLISHED_METRICS)
        cls.published = metrics.loc[metrics["Window"] == FULL_HISTORY_WINDOW].iloc[0]

    def test_the_baseline_row_is_the_published_full_history_result(self) -> None:
        row = self.frame.iloc[0]
        self.assertEqual(row["excluded_class"], BASELINE_LABEL)
        self.assertAlmostEqual(
            float(row["sharpe"]),
            float(self.published["Naive daily Sharpe (sqrt252, rf=0)"]),
            places=12,
        )
        self.assertAlmostEqual(float(row["cagr"]), float(self.published["CAGR"]), places=12)
        self.assertAlmostEqual(
            float(row["annualized_volatility"]),
            float(self.published["Annualized volatility"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(row["max_drawdown"]), float(self.published["Max drawdown"]), places=12
        )
        self.assertAlmostEqual(
            float(self.published["Naive daily Sharpe (sqrt252, rf=0)"]),
            PUBLISHED_BASELINE_SHARPE,
            places=12,
        )
        self.assertEqual(float(row["delta_sharpe_vs_baseline"]), 0.0)
        self.assertEqual(float(row["delta_cagr_vs_baseline"]), 0.0)

    def test_rows_follow_the_declared_manifest_order(self) -> None:
        self.assertEqual(list(self.frame.columns), REPORT_COLUMNS)
        self.assertEqual(
            list(self.frame["excluded_class"]),
            [BASELINE_LABEL] + [name for name, _ in EXCLUSION_MANIFEST],
        )
        self.assertTrue((self.frame["validation_status"] == VALIDATION_STATUS).all())

    def test_markets_remaining_follow_the_reference_universe_counts(self) -> None:
        reference = _load_reference()
        self.assertEqual(
            [name for name, _ in EXCLUSION_MANIFEST], list(reference.UNIVERSE)
        )
        total = sum(len(members) for members in reference.UNIVERSE.values())
        self.assertEqual(int(self.frame.iloc[0]["markets_remaining"]), total)
        for position, (name, members) in enumerate(EXCLUSION_MANIFEST, start=1):
            self.assertEqual(members, tuple(reference.UNIVERSE[name]))
            self.assertEqual(
                int(self.frame.iloc[position]["markets_remaining"]),
                total - len(members),
            )


if __name__ == "__main__":
    unittest.main()
