"""Tests for the funded-account reporting view."""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research.collateral import (
    EXCESS_BASIS_LABEL,
    FUNDED_BASIS_LABEL,
    CollateralConfig,
    collateral_reconciliation_report,
    funded_ledger,
    funded_performance_report,
    funded_regime_report,
    load_financing_rate,
    replace_spread,
)

DATA_DIR = Path("Round1AllData/Quant Researcher/Delta1")


def synthetic_daily(returns: list[float], start: str = "2000-01-03") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(returns))
    return pd.DataFrame({"net_return": returns}, index=index)


def synthetic_rate(index: pd.DatetimeIndex, rate: float) -> pd.DataFrame:
    return pd.DataFrame({"implied_annual_rate": [rate] * len(index)}, index=index)


class TestCollateralConfig(unittest.TestCase):
    def test_rejects_implausible_conventions(self) -> None:
        for override in (
            {"daycount_basis": 252},
            {"cash_rate_spread_bps": -1.0},
            {"max_stale_rate_sessions": 0},
            {"annualization": 0},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    dataclasses.replace(CollateralConfig(), **override)


class TestFundedLedger(unittest.TestCase):
    def test_accrual_uses_calendar_days_not_sessions(self) -> None:
        # A Friday-to-Monday step must earn three days of interest, not one.
        daily = synthetic_daily([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rate = synthetic_rate(daily.index, 0.0365)
        funded = funded_ledger(daily, rate, period_start=daily.index[0])
        weekend = funded.loc[funded["accrual_days"] == 3.0]
        self.assertFalse(weekend.empty)
        self.assertTrue((funded["accrual_days"] >= 0).all())
        self.assertAlmostEqual(
            float(funded["accrual_days"].sum()),
            float((daily.index[-1] - daily.index[0]).days),
        )

    def test_rate_is_lagged_so_no_session_earns_its_own_print(self) -> None:
        daily = synthetic_daily([0.0] * 5)
        rate = synthetic_rate(daily.index, 0.05)
        funded = funded_ledger(daily, rate, period_start=daily.index[0])
        observation = pd.to_datetime(funded["rate_observation_date"]).dropna()
        self.assertTrue((observation.to_numpy() < observation.index.to_numpy()).all())
        self.assertEqual(float(funded["collateral_return"].iloc[0]), 0.0)

    def test_excess_returns_are_reproduced_unaltered(self) -> None:
        values = [0.01, -0.02, 0.003, 0.0, 0.004]
        daily = synthetic_daily(values)
        funded = funded_ledger(
            daily, synthetic_rate(daily.index, 0.04), period_start=daily.index[0]
        )
        np.testing.assert_allclose(
            funded["excess_net_return"].to_numpy(), np.array(values)
        )

    def test_refuses_to_invent_a_rate_beyond_the_staleness_limit(self) -> None:
        daily = synthetic_daily([0.0] * 12)
        sparse = synthetic_rate(daily.index[:1], 0.04)
        with self.assertRaises(ValueError):
            funded_ledger(
                daily,
                sparse,
                dataclasses.replace(CollateralConfig(), max_stale_rate_sessions=2),
                period_start=daily.index[0],
            )

    def test_wider_broker_spread_never_improves_the_result(self) -> None:
        daily = synthetic_daily([0.001] * 20)
        rate = synthetic_rate(daily.index, 0.05)
        base = funded_ledger(daily, rate, period_start=daily.index[0])
        stressed = funded_ledger(
            daily,
            rate,
            replace_spread(CollateralConfig(), 100.0),
            period_start=daily.index[0],
        )
        self.assertLessEqual(
            float(stressed["funded_nav"].iloc[-1]),
            float(base["funded_nav"].iloc[-1]),
        )


class TestFundedSharpeIsNotInflated(unittest.TestCase):
    """The definitional trap this module exists to prevent.

    Adding a financing rate to the numerator while still treating the hurdle
    as zero raises the reported ratio without changing anything real.  On the
    supplied history that mistake yields roughly 2.0, which is precisely the
    figure named as an aspiration, so it must be structurally impossible.
    """

    def test_funded_sharpe_equals_excess_sharpe(self) -> None:
        rng = np.random.default_rng(11)
        daily = synthetic_daily(list(rng.normal(0.0004, 0.005, 500)))
        rate = synthetic_rate(daily.index, 0.05)
        funded = funded_ledger(daily, rate, period_start=daily.index[0])
        report = funded_performance_report(funded).set_index("Basis")
        self.assertAlmostEqual(
            float(report.loc[FUNDED_BASIS_LABEL, "Sharpe excess of financing"]),
            float(report.loc[EXCESS_BASIS_LABEL, "Sharpe excess of financing"]),
            places=9,
        )

    def test_financing_raises_cagr_but_not_risk_adjusted_return(self) -> None:
        rng = np.random.default_rng(12)
        daily = synthetic_daily(list(rng.normal(0.0004, 0.005, 500)))
        funded = funded_ledger(
            daily, synthetic_rate(daily.index, 0.05), period_start=daily.index[0]
        )
        report = funded_performance_report(funded).set_index("Basis")
        self.assertGreater(
            float(report.loc[FUNDED_BASIS_LABEL, "CAGR"]),
            float(report.loc[EXCESS_BASIS_LABEL, "CAGR"]),
        )

    def test_no_report_column_advertises_a_zero_rate_sharpe(self) -> None:
        daily = synthetic_daily([0.001] * 30)
        funded = funded_ledger(
            daily, synthetic_rate(daily.index, 0.05), period_start=daily.index[0]
        )
        columns = funded_performance_report(funded).columns
        for column in columns:
            if "sharpe" in column.lower():
                self.assertIn("excess of financing", column.lower())


class TestReconciliation(unittest.TestCase):
    def test_all_checks_pass_on_a_consistent_ledger(self) -> None:
        rng = np.random.default_rng(13)
        daily = synthetic_daily(list(rng.normal(0.0003, 0.004, 400)))
        funded = funded_ledger(
            daily, synthetic_rate(daily.index, 0.04), period_start=daily.index[0]
        )
        report = collateral_reconciliation_report(funded, daily)
        self.assertTrue((report["status"] == "PASS").all(), report.to_string())

    def test_a_tampered_funded_return_is_detected(self) -> None:
        daily = synthetic_daily([0.001] * 40)
        funded = funded_ledger(
            daily, synthetic_rate(daily.index, 0.04), period_start=daily.index[0]
        )
        funded.loc[funded.index[5], "funded_net_return"] += 0.01
        report = collateral_reconciliation_report(funded, daily).set_index("check")
        self.assertEqual(report.loc["funded_return_identity", "status"], "BLOCKED")


class TestSuppliedFinancingSeries(unittest.TestCase):
    """Guards on the actual vendor file, skipped when data is absent."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (DATA_DIR / "Futures Data" / "&ZQ.csv").is_file():
            raise unittest.SkipTest("supplied financing series is unavailable")

    def test_unadjusted_series_spans_a_plausible_rate_range(self) -> None:
        rate = load_financing_rate(DATA_DIR)
        implied = rate["implied_annual_rate"]
        self.assertGreaterEqual(float(implied.min()), 0.0)
        self.assertLess(float(implied.max()), 0.20)
        self.assertLess(float(implied.min()), 0.03)

    def test_back_adjusted_series_is_rejected(self) -> None:
        # Back-adjustment preserves returns but destroys the price level the
        # rate transform depends on, so the companion file must not load.
        with self.assertRaises(ValueError):
            load_financing_rate(
                DATA_DIR,
                dataclasses.replace(CollateralConfig(), rate_file="&ZQ_CCB.csv"),
            )

    def test_regime_report_partitions_the_history(self) -> None:
        daily = synthetic_daily([0.0] * 300, start="2000-01-03")
        rate = load_financing_rate(DATA_DIR)
        funded = funded_ledger(daily, rate, period_start=daily.index[0])
        regimes = funded_regime_report(funded)
        self.assertFalse(regimes.empty)
        self.assertAlmostEqual(float(regimes["Share of sessions"].sum()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
