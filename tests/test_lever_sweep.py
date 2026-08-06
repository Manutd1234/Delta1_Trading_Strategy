"""Regression tests for the standalone lever-sweep runner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_lever_sweep import bootstrap_breach_probability


class TestLeverSweepBootstrap(unittest.TestCase):
    def test_seeded_result_is_reproducible_and_records_its_design(self) -> None:
        returns = pd.Series(
            [0.004, -0.003, 0.002, -0.006, 0.005, 0.001, -0.002] * 20
        )
        first = bootstrap_breach_probability(
            returns,
            horizon=100,
            block_length=21,
            samples=100,
            seed=17,
        )
        second = bootstrap_breach_probability(
            returns,
            horizon=100,
            block_length=21,
            samples=100,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_samples"], 100)
        self.assertEqual(first["bootstrap_seed"], 17)
        self.assertEqual(first["bootstrap_expected_block_sessions"], 21)
        self.assertEqual(first["bootstrap_horizon_sessions"], 100)
        self.assertTrue(first["common_random_numbers"])
        self.assertEqual(len(str(first["bootstrap_index_sha256"])), 64)

    def test_equal_length_paths_use_the_same_index_matrix(self) -> None:
        first = pd.Series(np.linspace(-0.01, 0.01, 140))
        second = pd.Series(np.linspace(-0.02, 0.02, 140))
        first_report = bootstrap_breach_probability(
            first, horizon=100, block_length=21, samples=100, seed=17
        )
        second_report = bootstrap_breach_probability(
            second, horizon=100, block_length=21, samples=100, seed=17
        )

        self.assertEqual(
            first_report["bootstrap_index_sha256"],
            second_report["bootstrap_index_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
