from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from delta1_strategy.controls.evidence import (
    EvidenceRegistryError,
    load_evidence_registry,
    verified_evidence_flags,
    verify_evidence_registry,
)


MODEL = "1" * 64
CONFIG = "2" * 64
SOURCE = "3" * 64
SUBJECT = "4" * 64
AS_OF = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


class EvidenceFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "approval.txt"
        self.artifact.write_text("independently reviewed\n", encoding="utf-8")
        self.artifact_hash = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.manifest = self.root / "evidence.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def record(self, gate: str = "serial_contracts", **overrides: object) -> dict:
        values: dict[str, object] = {
            "gate": gate,
            "artifact_path": self.artifact.name,
            "artifact_sha256": self.artifact_hash,
            "issued_at": "2026-08-01T00:00:00Z",
            "expires_at": "2027-08-01T00:00:00+00:00",
            "model_fingerprint_sha256": MODEL,
            "config_fingerprint_sha256": CONFIG,
            "source_fingerprint_sha256": SOURCE,
            "author": "Strategy Author",
            "reviewer": "Independent Reviewer",
            "approved": True,
            "revoked": False,
        }
        values.update(overrides)
        return values

    def write_manifest(self, records: list[dict], **overrides: object) -> Path:
        document: dict[str, object] = {
            "schema_version": 1,
            "evidence": records,
        }
        document.update(overrides)
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        return self.manifest

    def verify(
        self,
        records: list[dict],
        gates: tuple[str, ...] = ("serial_contracts",),
        **overrides: object,
    ) -> pd.DataFrame:
        path = self.write_manifest(records)
        arguments: dict[str, object] = {
            "required_gates": gates,
            "expected_model_fingerprint": MODEL,
            "expected_config_fingerprint": CONFIG,
            "expected_source_fingerprint": SOURCE,
            "as_of": AS_OF,
        }
        arguments.update(overrides)
        return verify_evidence_registry(path, **arguments)


class TestEvidenceRegistry(EvidenceFixture):
    def test_valid_relative_artifact_and_independent_approval_pass(self) -> None:
        registry = load_evidence_registry(
            self.write_manifest([self.record()])
        )
        self.assertEqual(registry.schema_version, 1)
        self.assertEqual(len(registry.records), 1)
        self.assertEqual(len(registry.manifest_sha256), 64)

        report = verify_evidence_registry(
            registry,
            required_gates=("serial_contracts",),
            expected_model_fingerprint=MODEL,
            expected_config_fingerprint=CONFIG,
            expected_source_fingerprint=SOURCE,
            as_of="2026-08-03T12:00:00Z",
        )
        self.assertEqual(
            report.columns.tolist(),
            ["gate", "category", "critical", "status", "observed", "limit", "reason"],
        )
        self.assertEqual(report.iloc[0]["status"], "PASS")
        self.assertEqual(report.iloc[0]["observed"], self.artifact_hash)
        self.assertEqual(report.iloc[0]["reason"], "gate passed")
        self.assertEqual(
            verified_evidence_flags(report, ("serial_contracts",)),
            {"serial_contracts": True},
        )

    def test_loaded_registry_is_bound_to_unchanged_manifest_bytes(self) -> None:
        registry = load_evidence_registry(self.write_manifest([self.record()]))
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "evidence": []}),
            encoding="utf-8",
        )
        report = verify_evidence_registry(
            registry,
            required_gates=("serial_contracts",),
            expected_model_fingerprint=MODEL,
            expected_config_fingerprint=CONFIG,
            expected_source_fingerprint=SOURCE,
            as_of=AS_OF,
        )
        self.assertEqual(report.iloc[0]["status"], "BLOCKED")
        self.assertIn("manifest SHA-256 does not match", report.iloc[0]["reason"])

    def test_tampered_missing_and_unreadable_artifacts_block(self) -> None:
        path = self.write_manifest([self.record()])
        self.artifact.write_text("tampered\n", encoding="utf-8")
        report = verify_evidence_registry(
            path,
            required_gates=("serial_contracts",),
            expected_model_fingerprint=MODEL,
            expected_config_fingerprint=CONFIG,
            expected_source_fingerprint=SOURCE,
            as_of=AS_OF,
        )
        self.assertEqual(report.iloc[0]["status"], "BLOCKED")
        self.assertIn("SHA-256 does not match", report.iloc[0]["reason"])

        missing = self.verify(
            [self.record(artifact_path="not-there.txt")]
        )
        self.assertEqual(missing.iloc[0]["status"], "BLOCKED")
        self.assertIn("does not exist", missing.iloc[0]["reason"])

        directory = self.root / "not-a-file"
        directory.mkdir()
        not_file = self.verify(
            [self.record(artifact_path=directory.name)]
        )
        self.assertIn("not a regular file", not_file.iloc[0]["reason"])

    def test_timestamp_policy_requires_utc_current_ordered_evidence(self) -> None:
        cases = (
            (
                {"issued_at": "2026-08-01T00:00:00"},
                "explicit UTC offset",
            ),
            (
                {"issued_at": "2026-08-01T01:00:00+01:00"},
                "expressed in UTC",
            ),
            (
                {"issued_at": "2026-08-04T00:00:00Z"},
                "not yet been issued",
            ),
            (
                {"expires_at": "2026-08-03T12:00:00Z"},
                "has expired",
            ),
            (
                {
                    "issued_at": "2027-01-01T00:00:00Z",
                    "expires_at": "2026-01-01T00:00:00Z",
                },
                "later than issued_at",
            ),
        )
        for values, message in cases:
            with self.subTest(values=values):
                report = self.verify([self.record(**values)])
                self.assertEqual(report.iloc[0]["status"], "BLOCKED")
                self.assertIn(message, report.iloc[0]["reason"])

    def test_all_fingerprints_must_be_valid_and_match_frozen_run(self) -> None:
        fields = (
            "model_fingerprint_sha256",
            "config_fingerprint_sha256",
            "source_fingerprint_sha256",
        )
        for field in fields:
            with self.subTest(field=field):
                report = self.verify([self.record(**{field: "f" * 64})])
                self.assertEqual(report.iloc[0]["status"], "BLOCKED")
                self.assertIn("does not match the frozen run", report.iloc[0]["reason"])

        invalid_context = self.verify(
            [self.record()], expected_model_fingerprint="not-a-hash"
        )
        self.assertTrue(invalid_context["status"].eq("BLOCKED").all())
        self.assertIn("64-character", invalid_context.iloc[0]["reason"])

    def test_scoped_evidence_must_match_the_deployment_subject(self) -> None:
        matching = self.verify(
            [self.record(subject_sha256=SUBJECT)],
            expected_subject_sha256_by_gate={"serial_contracts": SUBJECT},
        )
        self.assertEqual(matching.iloc[0]["status"], "PASS")

        for supplied, message in (
            (None, "must be a 64-character"),
            ("5" * 64, "does not match deployment"),
        ):
            with self.subTest(supplied=supplied):
                record = self.record()
                if supplied is not None:
                    record["subject_sha256"] = supplied
                report = self.verify(
                    [record],
                    expected_subject_sha256_by_gate={"serial_contracts": SUBJECT},
                )
                self.assertEqual(report.iloc[0]["status"], "BLOCKED")
                self.assertIn(message, report.iloc[0]["reason"])

    def test_reviewer_must_be_independent_and_status_explicit(self) -> None:
        cases = (
            ({"reviewer": " strategy author "}, "independent"),
            ({"approved": False}, "not explicitly approved"),
            ({"approved": 1}, "not explicitly approved"),
            ({"revoked": True}, "is revoked"),
            ({"revoked": 0}, "revoked must be false"),
        )
        for values, message in cases:
            with self.subTest(values=values):
                report = self.verify([self.record(**values)])
                self.assertEqual(report.iloc[0]["status"], "BLOCKED")
                self.assertIn(message, report.iloc[0]["reason"])

    def test_missing_or_duplicate_gate_blocks_without_affecting_row_order(self) -> None:
        report = self.verify(
            [self.record("serial_contracts"), self.record("serial_contracts")],
            gates=("serial_contracts", "paper_trading"),
        )
        self.assertEqual(report["gate"].tolist(), ["serial_contracts", "paper_trading"])
        self.assertTrue(report["status"].eq("BLOCKED").all())
        self.assertIn("multiple", report.iloc[0]["reason"])
        self.assertIn("missing", report.iloc[1]["reason"])
        self.assertEqual(
            verified_evidence_flags(
                report, ("serial_contracts", "paper_trading")
            ),
            {"serial_contracts": False, "paper_trading": False},
        )

    def test_malformed_registry_and_as_of_fail_closed(self) -> None:
        self.manifest.write_text("not json", encoding="utf-8")
        with self.assertRaises(EvidenceRegistryError):
            load_evidence_registry(self.manifest)

        report = verify_evidence_registry(
            self.manifest,
            required_gates=("serial_contracts", "paper_trading"),
            expected_model_fingerprint=MODEL,
            expected_config_fingerprint=CONFIG,
            expected_source_fingerprint=SOURCE,
            as_of=AS_OF,
        )
        self.assertEqual(len(report), 2)
        self.assertTrue(report["status"].eq("BLOCKED").all())
        self.assertTrue(report["reason"].str.contains("registry is invalid").all())

        bad_as_of = self.verify([self.record()], as_of="2026-08-03T12:00:00")
        self.assertEqual(bad_as_of.iloc[0]["status"], "BLOCKED")
        self.assertIn("explicit UTC offset", bad_as_of.iloc[0]["reason"])

    def test_missing_required_record_field_invalidates_entire_registry(self) -> None:
        record = self.record()
        del record["artifact_sha256"]
        path = self.write_manifest([record])
        report = verify_evidence_registry(
            path,
            required_gates=("serial_contracts", "paper_trading"),
            expected_model_fingerprint=MODEL,
            expected_config_fingerprint=CONFIG,
            expected_source_fingerprint=SOURCE,
            as_of=AS_OF,
        )
        self.assertTrue(report["status"].eq("BLOCKED").all())
        self.assertTrue(report["reason"].str.contains("artifact_sha256").all())

    def test_flags_fail_closed_for_malformed_or_duplicate_reports(self) -> None:
        self.assertEqual(
            verified_evidence_flags(None, ("serial_contracts",)),
            {"serial_contracts": False},
        )
        malformed = pd.DataFrame({"gate": ["serial_contracts"]})
        self.assertEqual(
            verified_evidence_flags(malformed, ("serial_contracts",)),
            {"serial_contracts": False},
        )
        duplicate = pd.DataFrame(
            {
                "gate": ["serial_contracts", "serial_contracts"],
                "critical": [True, True],
                "status": ["PASS", "PASS"],
            }
        )
        self.assertEqual(
            verified_evidence_flags(duplicate, ("serial_contracts",)),
            {"serial_contracts": False},
        )


if __name__ == "__main__":
    unittest.main()
