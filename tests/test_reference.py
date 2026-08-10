"""The short file and the long package must agree, or the short file is a lie.

`reference/delta1_reference.py` is the readable answer to the case: one file,
no package imports, the whole strategy visible top to bottom.  It is only
worth reading if it is the *same* strategy the hardened engine in
`src/delta1_strategy/` runs, so this module pins that equality on the supplied
history and on generated data.

The comparison is exact, not approximate.  Both implementations perform the
same arithmetic in the same order, so any float tolerance here would be
hiding a specification difference rather than absorbing numerical noise.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research.strategy import (
    StrategyConfig,
    load_prices,
    run_backtest,
    strategy_symbols,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "Round1AllData/Quant Researcher/Delta1"))
# CI generates its panel outside the tree, so the location is overridable.
SYNTHETIC_DIR = Path(
    os.environ.get("DELTA1_SYNTHETIC_DIR", ROOT_DIR / "examples" / "data" / "synthetic")
)


def _load_reference():
    """Import the reference file by path, the way a reader would run it."""
    path = ROOT_DIR / "reference" / "delta1_reference.py"
    spec = importlib.util.spec_from_file_location("delta1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load_reference()

# Columns the reference publishes that the canonical ledger also carries.
SHARED_COLUMNS = (
    "nav",
    "prior_nav_usd",
    "gross_pnl_usd",
    "transaction_cost_usd",
    "gross_return",
    "cost",
    "net_return",
    "gross_notional_multiple",
    "static_margin_fraction",
    "total_contract_turnover",
    "active_markets",
)

# Every metric the case notebook displays, mapped to its canonical name.  The
# ledger being identical does not make the *metrics* identical -- the two
# implementations could compute a published figure by different formulas, which
# is exactly how the cost-drag row once drifted.  Anything the notebook shows a
# reader is pinned here.
SHARED_METRICS = (
    ("CAGR", "CAGR"),
    ("Annualized volatility", "Annualized volatility"),
    ("Sharpe (rf=0)", "Naive daily Sharpe (sqrt252, rf=0)"),
    ("Sortino (rf=0)", "Sortino (rf=0)"),
    ("Max drawdown", "Max drawdown"),
    ("Calmar", "Calmar"),
    ("Positive months", "Positive months"),
    ("Annual cost drag", "Annual cost drag"),
    ("Average gross notional multiple", "Average gross notional multiple"),
    ("Average markets held", "Average markets held"),
)


class TestReferenceSpecification(unittest.TestCase):
    """Checks that need no market data."""

    def test_universe_matches_the_package(self) -> None:
        self.assertEqual(reference.SYMBOLS, strategy_symbols())
        self.assertNotIn("YXT", reference.SYMBOLS)
        self.assertNotIn("YYT", reference.SYMBOLS)

    def test_parameters_match_the_frozen_config(self) -> None:
        config = StrategyConfig(DATA_DIR)
        pairs = {
            "trend_lookback": config.trend_lookback,
            "basis_roll_window": config.basis_roll_window,
            "basis_lookback": config.basis_lookback,
            "signal_norm_window": config.signal_normalization_window,
            "signal_cap": config.signal_cap,
            "basis_weight": config.basis_weight,
            "fast_vol_span": config.fast_vol_span,
            "slow_vol_span": config.slow_vol_span,
            "shock_start": config.shock_start,
            "shock_full": config.shock_full,
            "shock_floor": config.shock_floor,
            "vol_span": config.vol_span,
            "risk_managed_window": config.risk_managed_window,
            "risk_managed_cap": config.risk_managed_cap,
            "target_vol": config.target_vol,
            "vol_decay": config.vol_decay,
            "vol_min_periods": config.vol_estimator_min_periods,
            "portfolio_vol_window": config.portfolio_vol_window,
            "min_risk_scalar": config.min_risk_scalar,
            "max_risk_scalar": config.max_risk_scalar,
            "max_gross_notional_multiple": config.max_gross_notional_multiple,
            "no_trade_buffer": config.no_trade_buffer,
            "volume_gate_window": config.volume_gate_window,
            "min_median_contracts": config.min_median_contracts,
            "max_participation": config.max_rebalance_participation,
            "max_roll_backlog_sessions": config.max_roll_backlog_sessions,
            "half_spread_ticks": config.half_spread_ticks,
            "slippage_ticks": config.slippage_ticks,
            "commission": config.commission_per_contract,
            "fees": config.exchange_and_regulatory_fees_per_contract,
            "impact_bps_at_full_participation": config.impact_bps_at_full_participation,
            "initial_capital": config.initial_capital,
            "launch": config.launch_date,
            "price_ffill_limit": config.price_ffill_limit,
            "annualization": config.annualization,
        }
        for key, expected in pairs.items():
            with self.subTest(parameter=key):
                self.assertEqual(reference.P[key], expected)

    def test_no_trade_buffer_holds_small_adjustments(self) -> None:
        desired = pd.DataFrame(
            {"A": [10.0, 11.0, 20.0]},
            index=pd.date_range("2020-01-31", periods=3, freq="ME"),
        )
        executed = reference.apply_no_trade_buffer(desired, 0.25)
        # 10 -> 11 is a 9% adjustment and is suppressed; 10 -> 20 is not.
        self.assertEqual(list(executed["A"]), [10.0, 10.0, 20.0])

    def test_shock_multiplier_only_ever_cuts(self) -> None:
        rng = np.random.default_rng(0)
        index = pd.bdate_range("2015-01-01", periods=600)
        prices = pd.DataFrame(
            {"A": 100 + np.cumsum(rng.normal(0, 1, len(index)))}, index=index
        )
        multiplier = reference.shock_multiplier(prices).dropna()
        self.assertTrue((multiplier <= 1.0 + 1e-12).all().all())
        self.assertTrue((multiplier >= reference.P["shock_floor"] - 1e-12).all().all())


def _assert_ledgers_identical(case: unittest.TestCase, ledger: pd.DataFrame, canonical: pd.DataFrame) -> None:
    case.assertTrue(ledger.index.equals(canonical.index))
    for column in SHARED_COLUMNS:
        with case.subTest(column=column):
            np.testing.assert_array_equal(
                ledger[column].to_numpy(dtype=float),
                canonical[column].to_numpy(dtype=float),
                err_msg=f"reference and package disagree on {column}",
            )


@unittest.skipUnless(SYNTHETIC_DIR.exists(), "Generated data directory is not available")
class TestReferenceMatchesPackageOnGeneratedData(unittest.TestCase):
    """Equality without the licensed history, so CI can run it."""

    def test_daily_ledger_is_bit_identical(self) -> None:
        canonical = run_backtest(StrategyConfig(SYNTHETIC_DIR)).daily
        ledger = reference.run(SYNTHETIC_DIR)
        _assert_ledgers_identical(self, ledger, canonical)


@unittest.skipUnless(DATA_DIR.exists(), "Supplied DELTA1 data directory is not available")
class TestReferenceMatchesPackageOnSuppliedData(unittest.TestCase):
    """The claim that matters: same numbers on the twenty-five-year history."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = run_backtest(StrategyConfig(DATA_DIR)).daily
        cls.ledger = reference.run(DATA_DIR)

    def test_daily_ledger_is_bit_identical(self) -> None:
        _assert_ledgers_identical(self, self.ledger, self.canonical)

    def test_headline_metrics_match_the_published_bundle(self) -> None:
        """Every metric the case notebook prints must equal the canonical bundle.

        An identical ledger does not imply identical *metrics*: the two
        implementations could compute a published figure by different formulas,
        which is exactly how the cost-drag row once drifted 0.83% against 1.04%.
        Each published window is checked separately, because a formula that
        weights time differently can agree on one span and disagree on another.
        """
        published = pd.read_csv(ROOT_DIR / "outputs" / "strategy_metrics.csv")
        windows = {
            "1990-2004 development history": ("1990-01-01", "2004-12-31"),
            "2005-2014 reused later diagnostic": ("2005-01-01", "2014-12-31"),
            "1990-2014 full post-launch history": ("1990-01-01", "2014-12-31"),
        }
        for window, (start, end) in windows.items():
            row = published.loc[published["Window"].eq(window)]
            self.assertEqual(len(row), 1, f"missing published window: {window}")
            measured = reference.metrics(self.ledger, start, end)
            for reference_key, published_key in SHARED_METRICS:
                with self.subTest(window=window, metric=published_key):
                    self.assertAlmostEqual(
                        float(measured[reference_key]),
                        float(row.iloc[0][published_key]),
                        places=12,
                    )

    def test_forecast_is_bounded_and_causal(self) -> None:
        data = reference.load_market_data(DATA_DIR)
        forecast = reference.build_forecast(data)
        self.assertLessEqual(float(forecast.abs().max().max()), 1.0)

        # Truncation invariance: recomputing on a prefix must not change any
        # forecast inside that prefix.  A forward-looking term would break this.
        cutoff = data["prices"].index[-500]
        truncated = {
            key: value.loc[:cutoff] if isinstance(value, pd.DataFrame) else value
            for key, value in data.items()
        }
        prefix = reference.build_forecast(truncated)
        overlap = prefix.index
        pd.testing.assert_frame_equal(prefix, forecast.loc[overlap], check_freq=False)


@unittest.skipUnless(DATA_DIR.exists(), "Supplied DELTA1 data directory is not available")
class TestReferenceReadsTheSameInputs(unittest.TestCase):
    def test_price_panel_matches_the_package_loader(self) -> None:
        expected = load_prices(DATA_DIR, strategy_symbols(), 10)
        actual = reference.load_market_data(DATA_DIR)["prices"]
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)


if __name__ == "__main__":
    unittest.main()
