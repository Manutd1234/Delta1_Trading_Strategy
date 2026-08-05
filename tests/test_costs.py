from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from delta1_strategy.execution.costs import (
    CostCalibrationError,
    CostCalibrationPolicy,
    calibrate_execution_costs,
    cost_calibration_readiness_report,
    execution_cost_observations,
    raw_fill_sha256,
)


AS_OF = pd.Timestamp("2026-08-03 16:00", tz="UTC")


def fill_records(count: int = 40, **overrides: object) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number in range(count):
        side = "BUY" if number % 2 == 0 else "SELL"
        mid = 100.0
        fill_price = 100.30 if side == "BUY" else 99.70
        rows.append(
            {
                "fill_id": f"fill-{number}",
                "order_id": f"order-{number}",
                "contract_id": "ESU6" if number % 2 == 0 else "ESZ6",
                "root_symbol": "ES",
                "arrival_timestamp": (
                    AS_OF
                    - pd.Timedelta(days=number % 20)
                    - pd.Timedelta(milliseconds=500 + number % 5 * 100)
                ),
                "timestamp": AS_OF - pd.Timedelta(days=number % 20),
                "side": side,
                "quantity": 2,
                "order_type": "MARKETABLE_LIMIT" if number % 2 == 0 else "VWAP",
                "interval_volume": (2000, 400, 134)[number % 3],
                "volatility_regime": "LOW" if number % 2 == 0 else "HIGH",
                "arrival_bid": mid - 0.25,
                "arrival_ask": mid + 0.25,
                "fill_price": fill_price,
                "point_value_usd": 50.0,
                "commissions_and_fees_usd": 8.0,
                "is_roll": number < 10,
            }
        )
    frame = pd.DataFrame(rows)
    for column, value in overrides.items():
        frame[column] = pd.Series([value] * len(frame), index=frame.index)
    return frame


class CostObservationTests(unittest.TestCase):
    def test_spread_slippage_fee_decomposition_and_sell_sign(self) -> None:
        observations = execution_cost_observations(fill_records(2))
        # Half-spread is 0.25 points * $50 * 2 contracts.
        np.testing.assert_allclose(observations["quoted_half_spread_usd"], 25.0)
        # Both a buy above and sell below the midpoint have positive shortfall.
        np.testing.assert_allclose(observations["implementation_shortfall_usd"], 30.0)
        np.testing.assert_allclose(observations["residual_slippage_usd"], 5.0)
        np.testing.assert_allclose(observations["total_execution_cost_usd"], 38.0)

    def test_price_improvement_is_preserved_and_negative_prices_work(self) -> None:
        frame = fill_records(
            1,
            arrival_bid=-10.25,
            arrival_ask=-10.00,
            fill_price=-10.30,
            side="BUY",
        )
        observations = execution_cost_observations(frame)
        self.assertLess(observations.iloc[0]["implementation_shortfall_usd"], 0)
        self.assertTrue(np.isfinite(observations.iloc[0]["total_cost_bps"]))

    def test_invalid_fill_data_fails_closed(self) -> None:
        cases = (
            fill_records(2).drop(columns="arrival_bid"),
            fill_records(2).rename(columns={"point_value_usd": "point_value"}),
            fill_records(2, timestamp=pd.Timestamp("2026-08-03 16:00")),
            fill_records(2, arrival_bid=101, arrival_ask=100),
            fill_records(2, quantity=1.5),
            fill_records(2, is_roll=1),
            fill_records(2, arrival_timestamp=AS_OF + pd.Timedelta(seconds=1)),
            fill_records(2, interval_volume=1),
            fill_records(2, order_type="MARKET"),
            fill_records(2, volatility_regime=" "),
        )
        for frame in cases:
            with (
                self.subTest(columns=frame.columns.tolist()),
                self.assertRaises(CostCalibrationError),
            ):
                execution_cost_observations(frame)

    def test_order_identifiers_have_invariant_execution_identity(self) -> None:
        for column, replacement in (
            ("root_symbol", "NQ"),
            ("contract_id", "ESH7"),
            ("side", "SELL"),
            ("order_type", "VWAP"),
        ):
            frame = fill_records(2)
            frame.loc[:, "order_id"] = "shared-order"
            frame.loc[:, "side"] = "BUY"
            frame.loc[:, "order_type"] = "MARKETABLE_LIMIT"
            frame.loc[:, "contract_id"] = "ESU6"
            frame.loc[:, "volatility_regime"] = "LOW"
            frame.loc[:, "is_roll"] = True
            frame.loc[1, column] = replacement
            with self.subTest(column=column), self.assertRaisesRegex(
                CostCalibrationError, "order_id must map to invariant"
            ):
                execution_cost_observations(frame)


class CalibrationTests(unittest.TestCase):
    def test_calibration_passes_predeclared_sample_policy(self) -> None:
        fills = fill_records()
        calibration = calibrate_execution_costs(fills)
        self.assertEqual(calibration.iloc[0]["fill_count"], 40)
        self.assertEqual(calibration.iloc[0]["buy_fill_count"], 20)
        self.assertEqual(calibration.iloc[0]["sell_fill_count"], 20)
        self.assertEqual(calibration.iloc[0]["roll_fill_count"], 10)
        self.assertEqual(calibration.iloc[0]["order_count"], 40)
        self.assertEqual(calibration.iloc[0]["session_count"], 20)
        self.assertEqual(calibration.iloc[0]["delivery_cycle_count"], 2)
        self.assertEqual(calibration.iloc[0]["raw_fill_sha256"], raw_fill_sha256(fills))
        self.assertEqual(calibration.iloc[0]["order_type_count"], 2)
        self.assertEqual(calibration.iloc[0]["volatility_regime_count"], 2)
        self.assertEqual(calibration.iloc[0]["participation_bucket_count"], 3)
        self.assertGreaterEqual(
            calibration.iloc[0]["estimated_impact_bps_at_full_participation"], 0
        )
        report = cost_calibration_readiness_report(
            calibration,
            required_roots=["ES"],
            as_of=AS_OF,
            raw_fills=fills,
        )
        self.assertEqual(report.iloc[0]["status"], "PASS")

    def test_missing_stale_thin_one_sided_or_unrepresented_root_blocks(self) -> None:
        policy = CostCalibrationPolicy()
        stale_fills = fill_records(
            40,
            arrival_timestamp=AS_OF - pd.Timedelta(days=91, seconds=1),
            timestamp=AS_OF - pd.Timedelta(days=91),
        )
        cases = (
            (fill_records(10), "insufficient total orders"),
            (stale_fills, "stale"),
            (fill_records(40, side="BUY"), "sell orders"),
            (fill_records(40, is_roll=False), "roll orders"),
            (fill_records(40, order_type="MARKETABLE_LIMIT"), "order types"),
            (fill_records(40, volatility_regime="LOW"), "volatility regimes"),
            (fill_records(40, interval_volume=400), "participation buckets"),
        )
        for fills, reason in cases:
            with self.subTest(reason=reason):
                calibration = calibrate_execution_costs(fills)
                report = cost_calibration_readiness_report(
                    calibration,
                    required_roots=["ES"],
                    as_of=AS_OF,
                    policy=policy,
                    raw_fills=fills,
                )
                self.assertEqual(report.iloc[0]["status"], "BLOCKED")
                self.assertIn(reason, report.iloc[0]["reason"])

        fills = fill_records()
        missing = cost_calibration_readiness_report(
            calibrate_execution_costs(fills),
            required_roots=["NQ"],
            as_of=AS_OF,
            raw_fills=fills,
        )
        self.assertEqual(missing.iloc[0]["status"], "BLOCKED")

    def test_fill_fragmentation_cannot_satisfy_unique_order_gate(self) -> None:
        originals = fill_records(20, quantity=1, commissions_and_fees_usd=4.0)
        fragments = originals.copy()
        fragments["fill_id"] = fragments["fill_id"] + "-partial-2"
        fragments["timestamp"] = fragments["timestamp"] + pd.Timedelta(milliseconds=100)
        fills = pd.concat([originals, fragments], ignore_index=True)
        calibration = calibrate_execution_costs(fills)
        self.assertEqual(calibration.iloc[0]["fill_count"], 40)
        self.assertEqual(calibration.iloc[0]["order_count"], 20)
        report = cost_calibration_readiness_report(
            calibration,
            required_roots=["ES"],
            as_of=AS_OF,
            raw_fills=fills,
        )
        self.assertEqual(report.iloc[0]["status"], "BLOCKED")
        self.assertIn("insufficient total orders", report.iloc[0]["reason"])

    def test_distinct_sessions_recent_coverage_and_delivery_cycles_are_required(self) -> None:
        same_session = fill_records(
            timestamp=AS_OF,
            arrival_timestamp=AS_OF - pd.Timedelta(seconds=1),
        )
        one_cycle = fill_records(contract_id="ESU6")
        old_but_not_stale = fill_records()
        for number in old_but_not_stale.index:
            timestamp = AS_OF - pd.Timedelta(days=40 + number % 20)
            old_but_not_stale.loc[number, "timestamp"] = timestamp
            old_but_not_stale.loc[number, "arrival_timestamp"] = (
                timestamp - pd.Timedelta(seconds=1)
            )
        cases = (
            (same_session, "insufficient sessions"),
            (one_cycle, "insufficient delivery cycles"),
            (old_but_not_stale, "insufficient recent orders"),
            (old_but_not_stale, "insufficient recent sessions"),
        )
        for fills, reason in cases:
            with self.subTest(reason=reason):
                report = cost_calibration_readiness_report(
                    calibrate_execution_costs(fills),
                    required_roots=["ES"],
                    as_of=AS_OF,
                    raw_fills=fills,
                )
                self.assertEqual(report.iloc[0]["status"], "BLOCKED")
                self.assertIn(reason, report.iloc[0]["reason"])

    def test_raw_fill_digest_is_deterministic_and_readiness_recomputes_aggregate(self) -> None:
        fills = fill_records()
        calibration = calibrate_execution_costs(fills)
        shuffled = fills.sample(frac=1.0, random_state=17).reset_index(drop=True)
        self.assertEqual(raw_fill_sha256(fills), raw_fill_sha256(shuffled))

        no_raw = cost_calibration_readiness_report(
            calibration,
            required_roots=["ES"],
            as_of=AS_OF,
        )
        self.assertEqual(no_raw.iloc[0]["status"], "BLOCKED")
        self.assertIn("raw fills are missing", no_raw.iloc[0]["reason"])

        changed = fills.copy()
        changed.loc[0, "fill_price"] += 0.01
        wrong_raw = cost_calibration_readiness_report(
            calibration,
            required_roots=["ES"],
            as_of=AS_OF,
            raw_fills=changed,
        )
        self.assertEqual(wrong_raw.iloc[0]["status"], "BLOCKED")
        self.assertIn("raw-fill SHA256", wrong_raw.iloc[0]["reason"])

        forged = calibration.copy()
        forged.loc[0, "p75_total_cost_per_contract_usd"] += 1.0
        forged_report = cost_calibration_readiness_report(
            forged,
            required_roots=["ES"],
            as_of=AS_OF,
            raw_fills=fills,
        )
        self.assertEqual(forged_report.iloc[0]["status"], "BLOCKED")
        self.assertIn("aggregate does not match", forged_report.iloc[0]["reason"])

    def test_invalid_policy_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            CostCalibrationPolicy(minimum_fills_per_root=0)
        with self.assertRaises(ValueError):
            CostCalibrationPolicy(maximum_age_days=np.inf)
        with self.assertRaises(ValueError):
            CostCalibrationPolicy(maximum_observed_participation=0)
        with self.assertRaises(ValueError):
            CostCalibrationPolicy(minimum_sessions_per_root=0)


if __name__ == "__main__":
    unittest.main()
