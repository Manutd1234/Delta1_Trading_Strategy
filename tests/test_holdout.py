"""Tests for the one-shot out-of-sample custody ledger."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research.holdout import (
    SUBSET_LIMITATIONS,
    HoldoutConsumption,
    HoldoutDataset,
    HoldoutLedger,
    HoldoutLedgerError,
    HoldoutRegistration,
    evaluate_acceptance,
    verify_continuity,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
STAMP = "2026-01-02T00:00:00Z"
LATER = "2026-01-03T00:00:00Z"


def dataset(dataset_id: str = "d1") -> HoldoutDataset:
    return HoldoutDataset(
        dataset_id=dataset_id,
        description="sealed continuation series",
        file_sha256={"&ZB_CCB.csv": DIGEST_A},
        window_start="2015-01-01",
        window_end="2016-12-30",
        roots=("ZB",),
    )


def registration(dataset_id: str = "d1") -> HoldoutRegistration:
    return HoldoutRegistration(
        dataset=dataset(dataset_id),
        model_fingerprint_sha256=DIGEST_B,
        config_fingerprint_sha256=DIGEST_C,
        protocol_sha256=DIGEST_A,
        acceptance_criteria={"naive_sharpe": 0.0},
        limitations=SUBSET_LIMITATIONS,
        custodian="local repository",
    )


def consumption(record: HoldoutRegistration) -> HoldoutConsumption:
    return HoldoutConsumption(
        dataset_id=record.dataset.dataset_id,
        registration_sha256=record.registration_sha256,
        result_artifact_sha256=DIGEST_B,
        observed={"naive_sharpe": 0.55},
        completed_at=STAMP,
    )


class TestRecords(unittest.TestCase):
    def test_criteria_are_mandatory(self) -> None:
        # A look with no threshold written down beforehand is not an
        # evaluation; it is a peek.
        with self.assertRaises(ValueError):
            HoldoutRegistration(
                dataset=dataset(),
                model_fingerprint_sha256=DIGEST_B,
                config_fingerprint_sha256=DIGEST_C,
                protocol_sha256=DIGEST_A,
                acceptance_criteria={},
                limitations=SUBSET_LIMITATIONS,
                custodian="local",
            )

    def test_limitations_are_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutRegistration(
                dataset=dataset(),
                model_fingerprint_sha256=DIGEST_B,
                config_fingerprint_sha256=DIGEST_C,
                protocol_sha256=DIGEST_A,
                acceptance_criteria={"naive_sharpe": 0.0},
                limitations=(),
                custodian="local",
            )

    def test_window_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutDataset(
                dataset_id="d",
                description="x",
                file_sha256={"a.csv": DIGEST_A},
                window_start="2016-12-30",
                window_end="2015-01-01",
                roots=("ZB",),
            )


class TestLedger(unittest.TestCase):
    def test_a_second_look_is_refused(self) -> None:
        """The scarce resource is the look, not the bytes."""

        record = registration()
        ledger = HoldoutLedger().register(record, occurred_at=STAMP)
        ledger = ledger.consume(consumption(record), occurred_at=STAMP)
        with self.assertRaisesRegex(HoldoutLedgerError, "already been consumed"):
            ledger.consume(consumption(record), occurred_at=LATER)

    def test_consumption_before_registration_is_refused(self) -> None:
        record = registration()
        with self.assertRaisesRegex(HoldoutLedgerError, "before it is registered"):
            HoldoutLedger().consume(consumption(record), occurred_at=STAMP)

    def test_consumption_must_reconcile_with_what_was_sealed(self) -> None:
        record = registration()
        ledger = HoldoutLedger().register(record, occurred_at=STAMP)
        mismatched = HoldoutConsumption(
            dataset_id="d1",
            registration_sha256=DIGEST_C,
            result_artifact_sha256=DIGEST_B,
            observed={"naive_sharpe": 0.55},
            completed_at=STAMP,
        )
        with self.assertRaisesRegex(HoldoutLedgerError, "does not reconcile"):
            ledger.consume(mismatched, occurred_at=STAMP)

    def test_duplicate_registration_is_refused(self) -> None:
        ledger = HoldoutLedger().register(registration(), occurred_at=STAMP)
        with self.assertRaisesRegex(HoldoutLedgerError, "already registered"):
            ledger.register(registration(), occurred_at=LATER)

    def test_chain_detects_a_tampered_event(self) -> None:
        record = registration()
        ledger = HoldoutLedger().register(record, occurred_at=STAMP)
        ledger = ledger.consume(consumption(record), occurred_at=STAMP)
        broken = ledger.to_jsonl().replace('"naive_sharpe":0.55', '"naive_sharpe":9.99')
        with self.assertRaises(HoldoutLedgerError):
            HoldoutLedger.from_jsonl(broken)

    def test_round_trip_preserves_the_chain(self) -> None:
        record = registration()
        ledger = HoldoutLedger().register(record, occurred_at=STAMP)
        ledger = ledger.consume(consumption(record), occurred_at=STAMP)
        restored = HoldoutLedger.from_jsonl(ledger.to_jsonl())
        self.assertEqual(restored.head_sha256, ledger.head_sha256)
        self.assertTrue(restored.is_extension_of(ledger))

    def test_saving_cannot_rewrite_history(self) -> None:
        record = registration()
        ledger = HoldoutLedger().register(record, occurred_at=STAMP)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            ledger.save(path)
            unrelated = HoldoutLedger().register(registration("d2"), occurred_at=STAMP)
            with self.assertRaisesRegex(HoldoutLedgerError, "append-only"):
                unrelated.save(path)
            # Appending to the same chain is allowed.
            ledger.consume(consumption(record), occurred_at=STAMP).save(path)
            self.assertEqual(len(HoldoutLedger.load(path).events), 2)


class TestContinuity(unittest.TestCase):
    def test_matching_overlap_passes(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=10)
        canonical = pd.DataFrame({"ZB": np.arange(10.0)}, index=index)
        extension = pd.DataFrame(
            {"ZB": np.arange(5.0, 15.0)}, index=pd.bdate_range("2020-01-08", periods=10)
        )
        report = verify_continuity(canonical, extension)
        self.assertTrue((report["status"] == "PASS").all())

    def test_a_re_anchored_extension_is_blocked(self) -> None:
        """An offset means the vendor restated history; appending would corrupt it."""

        index = pd.bdate_range("2020-01-01", periods=10)
        canonical = pd.DataFrame({"ZB": np.arange(10.0)}, index=index)
        extension = pd.DataFrame(
            {"ZB": np.arange(5.0, 15.0) + 0.5},
            index=pd.bdate_range("2020-01-08", periods=10),
        )
        report = verify_continuity(canonical, extension)
        self.assertTrue((report["status"] == "BLOCKED").all())


class TestAcceptance(unittest.TestCase):
    def test_thresholds_are_lower_bounds(self) -> None:
        report = evaluate_acceptance(
            {"naive_sharpe": 0.55, "max_drawdown": -0.08},
            {"naive_sharpe": 0.0, "max_drawdown": -0.15},
        ).set_index("criterion")
        self.assertTrue(bool(report.loc["naive_sharpe", "met"]))
        self.assertTrue(bool(report.loc["max_drawdown", "met"]))

    def test_a_missing_metric_is_not_quietly_met(self) -> None:
        report = evaluate_acceptance({}, {"naive_sharpe": 0.0}).set_index("criterion")
        self.assertFalse(bool(report.loc["naive_sharpe", "met"]))


if __name__ == "__main__":
    unittest.main()
