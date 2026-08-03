from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from delta1_strategy.marketdata.contracts import (
    ContractDataError,
    SerialDataLimits,
    build_roll_plans,
    expand_roll_legs,
    serial_snapshot_validation_report,
    validated_serial_snapshot,
)


AS_OF = pd.Timestamp("2026-08-03 16:00:00", tz="UTC")


def serial_snapshot(**overrides: object) -> pd.DataFrame:
    rows = [
        {
            "timestamp": AS_OF - pd.Timedelta(seconds=30),
            "session_date": "2026-08-03",
            "root_symbol": "ES",
            "contract_id": "ESU6",
            "exchange": "CME",
            "currency": "USD",
            "delivery_type": "CASH_SETTLED",
            "expiry": "2026-09-18",
            "first_notice_date": None,
            "last_trade_date": "2026-09-18",
            "broker_liquidation_cutoff": pd.Timestamp(
                "2026-09-17 20:00:00", tz="UTC"
            ),
            "bid": 6300.00,
            "ask": 6300.25,
            "settlement": 6300.00,
            "volume": 10_000,
            "open_interest": 500_000,
            "point_value": 50.0,
            "notional_per_contract_usd": 315_000.0,
            "tick_size": 0.25,
            "margin_per_contract": 15_000.0,
            "margin_per_contract_usd": 15_000.0,
            "commission_per_contract": 2.50,
            "exchange_fee_per_contract": 1.50,
            "spec_effective_from": pd.Timestamp("2026-01-01", tz="UTC"),
            "spec_effective_to": pd.Timestamp("2027-01-01", tz="UTC"),
        },
        {
            "timestamp": AS_OF - pd.Timedelta(seconds=30),
            "session_date": "2026-08-03",
            "root_symbol": "ES",
            "contract_id": "ESZ6",
            "exchange": "CME",
            "currency": "USD",
            "delivery_type": "CASH_SETTLED",
            "expiry": "2026-12-18",
            "first_notice_date": None,
            "last_trade_date": "2026-12-18",
            "broker_liquidation_cutoff": pd.Timestamp(
                "2026-12-17 20:00:00", tz="UTC"
            ),
            "bid": 6320.00,
            "ask": 6320.25,
            "settlement": 6320.00,
            "volume": 1_000,
            "open_interest": 25_000,
            "point_value": 50.0,
            "notional_per_contract_usd": 316_000.0,
            "tick_size": 0.25,
            "margin_per_contract": 15_000.0,
            "margin_per_contract_usd": 15_000.0,
            "commission_per_contract": 2.50,
            "exchange_fee_per_contract": 1.50,
            "spec_effective_from": pd.Timestamp("2026-01-01", tz="UTC"),
            "spec_effective_to": pd.Timestamp("2027-01-01", tz="UTC"),
        },
    ]
    frame = pd.DataFrame(rows)
    for column, value in overrides.items():
        frame[column] = pd.Series([value] * len(frame), index=frame.index)
    return frame


class SerialSnapshotTests(unittest.TestCase):
    def test_valid_snapshot_passes_and_negative_prices_are_allowed(self) -> None:
        frame = serial_snapshot(bid=-10.25, ask=-10.00, settlement=-10.10)
        report = serial_snapshot_validation_report(frame, as_of=AS_OF)
        self.assertTrue(report["status"].eq("PASS").all())
        normalised = validated_serial_snapshot(frame, as_of=AS_OF)
        self.assertEqual(normalised["contract_id"].tolist(), ["ESU6", "ESZ6"])

    def test_naive_stale_crossed_and_present_day_specs_block(self) -> None:
        cases = (
            (serial_snapshot(timestamp=pd.Timestamp("2026-08-03 15:59:30")), "serial_timestamp_timezone"),
            (serial_snapshot(timestamp=AS_OF - pd.Timedelta(seconds=301)), "serial_data_freshness"),
            (serial_snapshot(bid=101.0, ask=100.0), "serial_market_values"),
            (
                serial_snapshot(
                    spec_effective_from=pd.Timestamp("2026-08-04", tz="UTC")
                ),
                "serial_effective_dated_specs",
            ),
        )
        for frame, gate in cases:
            with self.subTest(gate=gate):
                report = serial_snapshot_validation_report(frame, as_of=AS_OF)
                blocked = set(report.loc[report["status"].eq("BLOCKED"), "gate"])
                self.assertIn(gate, blocked)
                with self.assertRaises(ContractDataError):
                    validated_serial_snapshot(frame, as_of=AS_OF)

    def test_continuous_or_duplicate_identifier_blocks(self) -> None:
        frame = serial_snapshot()
        frame.loc[0, "contract_id"] = "ES"
        frame.loc[1, "contract_id"] = "ES"
        report = serial_snapshot_validation_report(frame, as_of=AS_OF)
        row = report.set_index("gate").loc["serial_contract_identifiers"]
        self.assertEqual(row["status"], "BLOCKED")

    def test_blank_venue_currency_and_zero_margin_block(self) -> None:
        cases = (
            (serial_snapshot(exchange=" "), "serial_venue_currency"),
            (serial_snapshot(currency=""), "serial_venue_currency"),
            (serial_snapshot(margin_per_contract=0), "serial_market_values"),
            (serial_snapshot(notional_per_contract_usd=0), "serial_market_values"),
            (serial_snapshot(margin_per_contract_usd=0), "serial_market_values"),
        )
        for frame, gate in cases:
            with self.subTest(gate=gate):
                report = serial_snapshot_validation_report(frame, as_of=AS_OF)
                self.assertEqual(report.set_index("gate").loc[gate, "status"], "BLOCKED")
                with self.assertRaises(ContractDataError):
                    validated_serial_snapshot(frame, as_of=AS_OF)

    def test_delivery_type_notice_and_broker_cutoff_fail_closed(self) -> None:
        cases = (
            serial_snapshot(delivery_type="UNKNOWN"),
            serial_snapshot(delivery_type="PHYSICAL", first_notice_date=None),
            serial_snapshot(
                delivery_type="PHYSICAL",
                first_notice_date="2026-09-17",
                broker_liquidation_cutoff=pd.Timestamp(
                    "2026-09-17 12:00:00", tz="UTC"
                ),
            ),
            serial_snapshot(
                broker_liquidation_cutoff=pd.Timestamp("2026-09-17 12:00:00")
            ),
        )
        for frame in cases:
            with self.subTest(delivery_type=frame.iloc[0]["delivery_type"]):
                report = serial_snapshot_validation_report(frame, as_of=AS_OF)
                self.assertEqual(
                    report.set_index("gate").loc[
                        "serial_delivery_controls", "status"
                    ],
                    "BLOCKED",
                )
                with self.assertRaises(ContractDataError):
                    validated_serial_snapshot(frame, as_of=AS_OF)

    def test_invalid_limits_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            SerialDataLimits(max_order_participation=0)
        with self.assertRaises(ValueError):
            SerialDataLimits(max_data_age_seconds=np.inf)
        with self.assertRaises(ValueError):
            SerialDataLimits(roll_calendar_days_before_deadline=0)


class RollPlanTests(unittest.TestCase):
    def snapshot_near_roll(self) -> pd.DataFrame:
        frame = serial_snapshot()
        expiring = frame["contract_id"].eq("ESU6")
        frame.loc[expiring, "last_trade_date"] = "2026-08-09"
        frame.loc[expiring, "expiry"] = "2026-08-09"
        frame.loc[expiring, "broker_liquidation_cutoff"] = pd.Timestamp(
            "2026-08-08 20:00:00", tz="UTC"
        )
        return frame

    def test_both_legs_are_capped_by_thinner_contract(self) -> None:
        plans = build_roll_plans(
            {"ESU6": 50},
            self.snapshot_near_roll(),
            as_of=AS_OF,
            limits=SerialDataLimits(max_order_participation=0.02),
        )
        row = plans.iloc[0]
        self.assertTrue(row["approved"])
        self.assertEqual(row["status"], "APPROVED_PARTIAL")
        self.assertEqual(row["capacity_contracts"], 20)
        self.assertEqual(row["roll_contracts"], 20)
        self.assertAlmostEqual(row["from_participation"], 0.002)
        self.assertAlmostEqual(row["to_participation"], 0.02)
        self.assertEqual(
            row["destination_liquidation_deadline"],
            pd.Timestamp("2026-12-17 20:00:00", tz="UTC"),
        )

        legs = expand_roll_legs(plans)
        self.assertEqual(legs["leg"].tolist(), ["CLOSE_EXPIRING", "OPEN_DEFERRED"])
        self.assertEqual(legs["side"].tolist(), ["SELL", "BUY"])
        self.assertTrue((legs["projected_participation"] <= 0.02).all())

    def test_short_roll_reverses_leg_sides(self) -> None:
        plans = build_roll_plans(
            {"ESU6": -5}, self.snapshot_near_roll(), as_of=AS_OF
        )
        legs = expand_roll_legs(plans)
        self.assertEqual(legs["side"].tolist(), ["BUY", "SELL"])

    def test_not_due_zero_volume_missing_contract_and_deadline_fail_closed(self) -> None:
        not_due = build_roll_plans({"ESU6": 1}, serial_snapshot(), as_of=AS_OF)
        self.assertEqual(not_due.iloc[0]["status"], "NOT_DUE")
        self.assertFalse(not_due.iloc[0]["approved"])

        thin = self.snapshot_near_roll()
        thin.loc[thin["contract_id"].eq("ESZ6"), "volume"] = 0
        blocked = build_roll_plans({"ESU6": 1}, thin, as_of=AS_OF)
        self.assertEqual(blocked.iloc[0]["status"], "BLOCKED")

        missing = build_roll_plans({"UNKNOWN": 1}, serial_snapshot(), as_of=AS_OF)
        self.assertEqual(missing.iloc[0]["status"], "BLOCKED")

        deadline_snapshot = self.snapshot_near_roll()
        deadline_time = pd.Timestamp("2026-08-09 16:00", tz="UTC")
        deadline_snapshot.loc[:, "timestamp"] = deadline_time - pd.Timedelta(seconds=5)
        deadline_snapshot.loc[:, "session_date"] = "2026-08-09"
        late = build_roll_plans({"ESU6": 1}, deadline_snapshot, as_of=deadline_time)
        self.assertEqual(late.iloc[0]["status"], "BLOCKED")
        self.assertIn("deadline", late.iloc[0]["reason"])

        unsafe_destination = self.snapshot_near_roll()
        unsafe_destination.loc[
            unsafe_destination["contract_id"].eq("ESZ6"),
            "broker_liquidation_cutoff",
        ] = pd.Timestamp("2026-08-09 20:00:00", tz="UTC")
        blocked_destination = build_roll_plans(
            {"ESU6": 1}, unsafe_destination, as_of=AS_OF
        )
        self.assertEqual(blocked_destination.iloc[0]["status"], "BLOCKED")
        self.assertIn("delivery-safe", blocked_destination.iloc[0]["reason"])

    def test_destination_retains_the_configured_delivery_buffer(self) -> None:
        snapshot = serial_snapshot()
        expiring = snapshot["contract_id"].eq("ESU6")
        deferred = snapshot["contract_id"].eq("ESZ6")
        snapshot.loc[expiring, ["last_trade_date", "expiry"]] = "2026-08-04"
        snapshot.loc[expiring, "broker_liquidation_cutoff"] = pd.Timestamp(
            "2026-08-04 20:00:00", tz="UTC"
        )
        # This deadline is later than the expiring contract's, but only six
        # calendar days after the decision versus the required seven.
        snapshot.loc[deferred, "broker_liquidation_cutoff"] = pd.Timestamp(
            "2026-08-09 20:00:00", tz="UTC"
        )
        plan = build_roll_plans({"ESU6": 1}, snapshot, as_of=AS_OF).iloc[0]
        self.assertFalse(plan["approved"])
        self.assertEqual(plan["status"], "BLOCKED")
        self.assertIn("delivery-safe", plan["reason"])

    def test_destination_must_match_source_venue_and_currency(self) -> None:
        destination = serial_snapshot()["contract_id"].eq("ESZ6")
        for column, value in (("exchange", "ICE"), ("currency", "EUR")):
            with self.subTest(column=column):
                snapshot = self.snapshot_near_roll()
                snapshot.loc[destination, column] = value
                plan = build_roll_plans({"ESU6": 1}, snapshot, as_of=AS_OF).iloc[0]
                self.assertFalse(plan["approved"])
                self.assertEqual(plan["status"], "BLOCKED")
                self.assertIn("delivery-safe", plan["reason"])

    def test_positions_must_be_integer(self) -> None:
        with self.assertRaises(ValueError):
            build_roll_plans({"ESU6": 1.5}, self.snapshot_near_roll(), as_of=AS_OF)


if __name__ == "__main__":
    unittest.main()
