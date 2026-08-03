from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from delta1_strategy.research.trade_sequence import (
    TRADE_SEQUENCE_COLUMNS,
    TradeSequenceMonteCarloConfig,
    _path_statistics,
    trade_sequence_monte_carlo_summary,
)


def synthetic_result(include_future: bool = False) -> SimpleNamespace:
    episodes = pd.DataFrame(
        {
            "Symbol": ["A", "B", "C", "D", "E", "F"],
            "Entry date": pd.to_datetime(
                [
                    "2000-01-01",
                    "2000-01-15",
                    "2000-02-01",
                    "2000-02-01",
                    "2000-03-01",
                    "2000-04-01",
                ]
            ),
            "Exit date": pd.to_datetime(
                [
                    "2000-02-01",
                    "2000-02-01",
                    "2000-03-01",
                    "2000-04-01",
                    "2000-05-01",
                    None,
                ]
            ),
            "Status": ["Closed", "Closed", "Closed", "Closed", "Closed", "Open"],
            "Net contribution": [0.04, -0.02, 0.01, -0.03, 0.06, 0.50],
        }
    )
    if include_future:
        episodes.loc[len(episodes)] = [
            "G",
            pd.Timestamp("2002-01-01"),
            pd.Timestamp("2002-02-01"),
            "Closed",
            -0.90,
        ]
    return SimpleNamespace(trade_episodes=episodes)


class TestTradeSequenceMonteCarlo(unittest.TestCase):
    def test_is_deterministic_and_reports_explicit_limitations(self) -> None:
        config = TradeSequenceMonteCarloConfig(samples=250, seed=19)
        windows = {"history": ("2000-01-01", "2000-12-31")}
        first = trade_sequence_monte_carlo_summary(
            synthetic_result(), windows, config=config
        )
        second = trade_sequence_monte_carlo_summary(
            synthetic_result(), windows, config=config
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.columns.tolist(), TRADE_SEQUENCE_COLUMNS)
        self.assertEqual(set(first["Closed episodes"]), {5})
        self.assertEqual(set(first["Maximum episodes sharing an exit date"]), {2})
        self.assertEqual(set(first["Replacement"]), {False, True})
        self.assertFalse(bool(first["Selection adjusted"].any()))
        self.assertFalse(bool(first["Dependence preserved"].any()))
        joined_limitations = " ".join(first["Limitations"].unique()).lower()
        self.assertIn("overlap", joined_limitations)
        self.assertIn("selection-adjusted", joined_limitations)
        self.assertIn("iid", joined_limitations)

    def test_permutation_terminal_outcome_is_fixed_but_order_risk_varies(self) -> None:
        config = TradeSequenceMonteCarloConfig(
            samples=500,
            seed=4,
            methods=("permutation",),
            capital_floor_fraction=0.45,
            drawdown_thresholds=(0.56,),
        )
        result = SimpleNamespace(
            trade_episodes=pd.DataFrame(
                {
                    "entry_date": pd.to_datetime(["2000-01-01", "2000-01-02"]),
                    "exit_date": pd.to_datetime(["2000-02-01", "2000-02-02"]),
                    "status": ["closed", "closed"],
                    "net_return_contribution": [-0.60, 0.10],
                }
            )
        )
        report = trade_sequence_monte_carlo_summary(
            result, {"two trades": ("2000-01-01", "2000-12-31")}, config=config
        )
        terminal = report.loc[report["Metric"].eq("Terminal contribution return")]
        self.assertTrue(np.allclose(terminal["Value"], -0.50))

        ruin = report.loc[
            report["Metric"].eq(
                "Capital-floor breach probability (risk-of-ruin proxy)"
            )
        ].iloc[0]
        self.assertAlmostEqual(float(ruin["Value"]), 0.50, delta=0.07)
        self.assertLess(float(ruin["Confidence lower"]), float(ruin["Value"]))
        self.assertGreater(float(ruin["Confidence upper"]), float(ruin["Value"]))

        drawdown = report.loc[
            report["Metric"].eq(
                "Maximum-drawdown breach probability (56% threshold)"
            )
        ].iloc[0]
        self.assertAlmostEqual(float(drawdown["Value"]), 0.50, delta=0.07)

    def test_drawdown_includes_starting_capital(self) -> None:
        stats = _path_statistics(np.array([[-0.10, 0.05], [0.10, -0.20]]))
        self.assertAlmostEqual(float(stats["Maximum drawdown"][0]), -0.10)
        self.assertAlmostEqual(float(stats["Maximum drawdown"][1]), -0.20 / 1.10)
        np.testing.assert_array_equal(
            stats["Longest losing streak (episodes)"], np.array([1, 1])
        )

    def test_future_episode_does_not_change_a_fixed_window(self) -> None:
        config = TradeSequenceMonteCarloConfig(samples=100, seed=3)
        windows = {"fixed": ("2000-01-01", "2000-12-31")}
        base = trade_sequence_monte_carlo_summary(
            synthetic_result(), windows, config=config
        )
        future = trade_sequence_monte_carlo_summary(
            synthetic_result(include_future=True), windows, config=config
        )
        pd.testing.assert_frame_equal(base, future)

    def test_zero_observed_breaches_still_has_nonzero_confidence_bound(self) -> None:
        config = TradeSequenceMonteCarloConfig(
            samples=100,
            methods=("permutation",),
            capital_floor_fraction=0.01,
            drawdown_thresholds=(0.90,),
        )
        report = trade_sequence_monte_carlo_summary(
            synthetic_result(),
            {"history": ("2000-01-01", "2000-12-31")},
            config=config,
        )
        probability = report.loc[
            report["Metric"].eq(
                "Capital-floor breach probability (risk-of-ruin proxy)"
            )
        ].iloc[0]
        self.assertEqual(float(probability["Value"]), 0.0)
        self.assertGreater(float(probability["Confidence upper"]), 0.0)

    def test_empty_input_and_invalid_config(self) -> None:
        empty = trade_sequence_monte_carlo_summary(
            SimpleNamespace(trade_episodes=pd.DataFrame()),
            {"empty": ("2000", "2001")},
        )
        self.assertTrue(empty.empty)
        self.assertEqual(empty.columns.tolist(), TRADE_SEQUENCE_COLUMNS)
        invalid_arguments = [
            {"samples": 0},
            {"samples": 10.0},
            {"seed": 1.5},
            {"methods": ()},
            {"methods": ("unknown",)},
            {"methods": ("permutation", "permutation")},
            {"return_scale": 0.0},
            {"capital_floor_fraction": 1.0},
            {"drawdown_thresholds": (0.0,)},
            {"drawdown_thresholds": (0.10, 0.10)},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                TradeSequenceMonteCarloConfig(**arguments)

    def test_reversed_window_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "end precedes start"):
            trade_sequence_monte_carlo_summary(
                synthetic_result(), {"bad": ("2001-01-01", "2000-01-01")}
            )


if __name__ == "__main__":
    unittest.main()
