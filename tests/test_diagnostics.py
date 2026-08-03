from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from delta1_strategy.research.diagnostics import (
    DEFAULT_DAILY_BLOCK_LENGTHS,
    DEFAULT_DAILY_HORIZONS_SESSIONS,
    _path_drawdown_statistics,
    causal_regime_report,
    daily_stationary_bootstrap_drawdown_summary,
    monthly_stationary_bootstrap_summary,
    trade_metrics_report,
)


def synthetic_result(months: int = 144) -> SimpleNamespace:
    index = pd.date_range("2000-01-31", periods=months, freq="ME")
    phase = np.arange(months, dtype=float)
    returns = 0.008 + 0.025 * np.sin(phase / 3.0) + 0.005 * np.cos(phase / 11.0)
    daily = pd.DataFrame(
        {
            "net_return": returns,
            "cost": 0.0002 + 0.00005 * (1 + np.sin(phase / 5.0)),
            "gross_notional_multiple": 3.0 + 0.2 * np.sin(phase / 7.0),
            "static_margin_fraction": 0.25 + 0.02 * np.cos(phase / 9.0),
            "max_order_participation": 0.01 + 0.004 * (1 + np.cos(phase / 4.0)),
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "X": np.sin(phase / 8.0),
            "Y": np.cos(phase / 10.0),
        },
        index=index,
    )
    episodes = pd.DataFrame(
        {
            "symbol": ["X", "X", "Y", "Y"],
            "asset_class": ["Rates", "Rates", "FX", "FX"],
            "entry_date": pd.to_datetime(
                ["2001-01-31", "2002-01-31", "2003-01-31", "2010-01-31"]
            ),
            "exit_date": pd.to_datetime(
                ["2001-06-30", "2002-06-30", "2003-06-30", None]
            ),
            "status": ["closed", "closed", "closed", "open"],
            "net_pnl_usd": [2_000.0, -1_000.0, 1_000.0, 500.0],
            "net_return_contribution": [0.002, -0.001, 0.001, 0.0005],
            "holding_sessions": [100, 105, 98, 200],
        }
    )
    return SimpleNamespace(daily=daily, signals=signals, trade_episodes=episodes)


def synthetic_daily_result(sessions: int = 756) -> SimpleNamespace:
    index = pd.bdate_range("2000-01-03", periods=sessions)
    phase = np.arange(sessions, dtype=float)
    returns = 0.0003 + 0.004 * np.sin(phase / 17.0) + 0.002 * np.cos(phase / 41.0)
    return SimpleNamespace(
        daily=pd.DataFrame({"net_return": returns}, index=index)
    )


class TestMonthlyStationaryBootstrap(unittest.TestCase):
    def test_is_deterministic_and_constrains_horizons_to_source_history(self) -> None:
        result = synthetic_result(72)
        windows = {"history": ("2000-01-01", "2005-12-31")}
        first = monthly_stationary_bootstrap_summary(
            result,
            windows,
            samples=250,
            block_lengths=(6, 12),
            horizons_months=(12, 36, 60, 120),
            seed=17,
        )
        second = monthly_stationary_bootstrap_summary(
            result,
            windows,
            samples=250,
            block_lengths=(6, 12),
            horizons_months=(12, 36, 60, 120),
            seed=17,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["Horizon months"]), {12, 36, 60})
        self.assertFalse(bool(first["Selection adjusted"].any()))
        self.assertTrue(first["Value"].notna().all())
        self.assertEqual(
            set(first["Drawdown resolution"]),
            {
                "month-end only; informational; may understate intramonth "
                "drawdown"
            },
        )
        self.assertIn(
            "Probability maximum drawdown exceeds 15%",
            set(first["Metric"]),
        )
        self.assertIn(
            "Probability capital falls below 50% of initial (diagnostic)",
            set(first["Metric"]),
        )


class TestDailyStationaryBootstrapDrawdown(unittest.TestCase):
    def test_predeclared_grid_is_one_three_six_months_and_ten_twenty_five_years(
        self,
    ) -> None:
        self.assertEqual(DEFAULT_DAILY_BLOCK_LENGTHS, (21, 63, 126))
        self.assertEqual(DEFAULT_DAILY_HORIZONS_SESSIONS, (2_520, 6_300))

    def test_is_deterministic_and_has_daily_resolution_schema(self) -> None:
        result = synthetic_daily_result(756)
        windows = {"history": ("2000-01-01", None)}
        arguments = {
            "samples": 80,
            "block_lengths_sessions": (21, 63),
            "horizons_sessions": (252, 504),
            "seed": 29,
        }
        first = daily_stationary_bootstrap_drawdown_summary(
            result,
            windows,
            **arguments,
        )
        second = daily_stationary_bootstrap_drawdown_summary(
            result,
            windows,
            **arguments,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["Method"]), {"stationary daily bootstrap"})
        self.assertEqual(
            set(first["Drawdown resolution"]),
            {"daily close; includes initial capital"},
        )
        self.assertEqual(set(first["Expected block sessions"]), {21, 63})
        self.assertEqual(set(first["Horizon sessions"]), {252, 504})
        self.assertEqual(
            set(first["Metric"]),
            {
                "Maximum drawdown",
                "Probability capital falls below 50% of initial (diagnostic)",
                "Probability maximum drawdown exceeds 15%",
                "Probability maximum drawdown exceeds 20%",
                "Probability maximum drawdown exceeds 30%",
                "Probability maximum drawdown exceeds 40%",
            },
        )
        self.assertNotIn("CAGR", set(first["Metric"]))
        self.assertNotIn("Monthly Sharpe", set(first["Metric"]))
        self.assertFalse(bool(first["Selection adjusted"].any()))

    def test_drawdown_includes_initial_capital(self) -> None:
        statistics = _path_drawdown_statistics(np.array([[-0.10]]))
        self.assertAlmostEqual(float(statistics["Maximum drawdown"][0]), -0.10)
        self.assertAlmostEqual(
            float(statistics["Minimum capital fraction"][0]),
            0.90,
        )

    def test_daily_path_catches_loss_recovered_before_month_end(self) -> None:
        daily_path = np.array([[-0.20, 0.25]])
        monthly_path = np.prod(1.0 + daily_path, axis=1, keepdims=True) - 1.0
        daily = _path_drawdown_statistics(daily_path)["Maximum drawdown"][0]
        month_end = _path_drawdown_statistics(monthly_path)["Maximum drawdown"][0]
        self.assertAlmostEqual(float(monthly_path[0, 0]), 0.0)
        self.assertAlmostEqual(float(month_end), 0.0)
        self.assertAlmostEqual(float(daily), -0.20)


class TestCausalRegimes(unittest.TestCase):
    def test_labels_and_report_are_truncation_invariant(self) -> None:
        full = synthetic_result(144)
        cutoff = pd.Timestamp("2008-12-31")
        truncated = SimpleNamespace(
            daily=full.daily.loc[:cutoff].copy(),
            signals=full.signals.loc[:cutoff].copy(),
            trade_episodes=full.trade_episodes.copy(),
        )
        full_report = causal_regime_report(
            full,
            start="2000-01-01",
            end="2008-12-31",
            minimum_history_months=36,
        )
        truncated_report = causal_regime_report(
            truncated,
            start="2000-01-01",
            end="2008-12-31",
            minimum_history_months=36,
        )
        pd.testing.assert_frame_equal(full_report, truncated_report)
        self.assertGreater(len(full_report), 0)
        forbidden = {"CAGR", "Max drawdown"}
        self.assertTrue(forbidden.isdisjoint(full_report.columns))


class TestTradeMetricsAndSchemas(unittest.TestCase):
    def test_reports_overall_asset_class_and_symbol_rows(self) -> None:
        result = synthetic_result()
        report = trade_metrics_report(
            result,
            {"history": ("2000-01-01", "2011-12-31")},
        )
        self.assertEqual(set(report["Scope"]), {"Overall", "Asset class", "Symbol"})
        overall = report.loc[report["Scope"].eq("Overall")].iloc[0]
        self.assertEqual(int(overall["Closed episodes"]), 3)
        self.assertEqual(int(overall["Open or censored episodes"]), 1)
        self.assertAlmostEqual(float(overall["Profit factor contribution"]), 3.0)
        self.assertAlmostEqual(float(overall["Win rate"]), 2 / 3)
        self.assertAlmostEqual(float(overall["Expectancy bps"]), 20 / 3)

    def test_accepts_canonical_title_case_episode_columns(self) -> None:
        result = synthetic_result()
        result.trade_episodes = result.trade_episodes.rename(
            columns={
                "symbol": "Symbol",
                "asset_class": "Asset class",
                "entry_date": "Entry date",
                "exit_date": "Exit date",
                "status": "Status",
                "net_pnl_usd": "Net P&L USD",
                "net_return_contribution": "Net contribution",
                "holding_sessions": "Holding sessions",
            }
        )
        report = trade_metrics_report(
            result,
            {"history": ("2000-01-01", "2011-12-31")},
        )
        overall = report.loc[report["Scope"].eq("Overall")].iloc[0]
        self.assertEqual(int(overall["Closed episodes"]), 3)
        self.assertAlmostEqual(float(overall["Profit factor contribution"]), 3.0)
        self.assertAlmostEqual(float(overall["Expectancy bps"]), 20 / 3)

    def test_profit_factor_uses_each_numeraire_sign(self) -> None:
        episodes = pd.DataFrame(
            {
                "symbol": ["ES", "ES"],
                "asset_class": ["Equity", "Equity"],
                "entry_date": pd.to_datetime(["2001-01-01", "2001-02-01"]),
                "exit_date": pd.to_datetime(["2001-01-10", "2001-02-10"]),
                "status": ["Closed", "Closed"],
                "net_pnl_usd": [10.0, -5.0],
                "net_return_contribution": [-0.02, 0.01],
                "holding_sessions": [7, 7],
            }
        )
        result = SimpleNamespace(trade_episodes=episodes)
        report = trade_metrics_report(
            result,
            {"history": ("2000-01-01", "2002-12-31")},
        )
        overall = report.loc[report["Scope"].eq("Overall")].iloc[0]
        self.assertAlmostEqual(float(overall["Profit factor contribution"]), 0.5)
        self.assertAlmostEqual(float(overall["Profit factor USD"]), 2.0)
        self.assertAlmostEqual(float(overall["Win rate"]), 0.5)

    def test_diagnostics_never_emit_target_or_verdict_columns(self) -> None:
        result = synthetic_result(72)
        windows = {"history": ("2000-01-01", "2005-12-31")}
        frames = [
            monthly_stationary_bootstrap_summary(
                result,
                windows,
                samples=50,
                block_lengths=(6,),
                horizons_months=(12,),
            ),
            daily_stationary_bootstrap_drawdown_summary(
                synthetic_daily_result(300),
                {"history": ("2000-01-01", None)},
                samples=25,
                block_lengths_sessions=(21,),
                horizons_sessions=(252,),
            ),
            causal_regime_report(
                result,
                start="2000-01-01",
                end="2005-12-31",
                minimum_history_months=36,
            ),
            trade_metrics_report(result, windows),
        ]
        forbidden = ("target", "pass", "fail", "verdict")
        for frame in frames:
            joined = " ".join(map(str, frame.columns)).lower()
            self.assertFalse(any(word in joined for word in forbidden))

    def test_empty_trade_episodes_return_zero_count_overall_row(self) -> None:
        result = SimpleNamespace(trade_episodes=pd.DataFrame())
        report = trade_metrics_report(result, {"empty": ("2000", "2001")})
        self.assertEqual(len(report), 1)
        self.assertEqual(int(report.iloc[0]["Closed episodes"]), 0)


if __name__ == "__main__":
    unittest.main()
