"""Pin the crisis-window artifact to its declared manifest and to the ledger."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_crisis_windows import (
    ANNUALIZATION,
    CRISIS_WINDOWS,
    VALIDATION_STATUS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "outputs" / "strategy_daily.csv"
ARTIFACT = REPO_ROOT / "outputs" / "validation" / "validation_crisis_windows.csv"


@unittest.skipUnless(
    LEDGER.exists() and ARTIFACT.exists(),
    "Frozen ledger or crisis-window artifact is not available",
)
class TestCrisisWindowArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = pd.read_csv(ARTIFACT)

    def test_rows_equal_the_declared_manifest(self) -> None:
        self.assertEqual(len(self.artifact), len(CRISIS_WINDOWS))
        for row, (label, start, end, anchor) in zip(
            self.artifact.itertuples(index=False), CRISIS_WINDOWS
        ):
            self.assertEqual(row.label, label)
            self.assertEqual(row.start, start)
            self.assertEqual(row.end, end)
            self.assertEqual(row.event_anchor, anchor)
            self.assertEqual(row.validation_status, VALIDATION_STATUS)

    def test_every_window_lies_inside_the_retrospective(self) -> None:
        for _, start, end, _ in CRISIS_WINDOWS:
            self.assertLessEqual(pd.Timestamp("1990-01-01"), pd.Timestamp(start))
            self.assertLess(pd.Timestamp(start), pd.Timestamp(end))
            self.assertLessEqual(pd.Timestamp(end), pd.Timestamp("2014-12-31"))

    def test_lehman_row_recomputes_from_the_ledger(self) -> None:
        daily = pd.read_csv(LEDGER, index_col="date", parse_dates=True)
        label, start, end, _ = next(
            entry for entry in CRISIS_WINDOWS if entry[0] == "Lehman"
        )
        returns = daily.loc[start:end, "net_return"]
        equity = (1.0 + returns).cumprod()
        running_peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
        row = self.artifact.loc[self.artifact["label"] == label].iloc[0]
        self.assertEqual(int(row["sessions"]), len(returns))
        self.assertAlmostEqual(
            float(row["cumulative_net_return"]), float(equity.iloc[-1] - 1.0), places=12
        )
        self.assertAlmostEqual(
            float(row["max_drawdown"]),
            float((equity / running_peak - 1.0).min()),
            places=12,
        )
        self.assertAlmostEqual(
            float(row["annualized_volatility"]),
            float(returns.std() * math.sqrt(ANNUALIZATION)),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
