from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from delta1_strategy.controls.production import (
    ArtifactBackedReadinessCheck,
    EXPECTED_READINESS_GATES,
    EXTERNAL_EVIDENCE_LABELS,
    ProductionLimits,
    build_order_intents,
    daily_frame_fingerprint,
    evaluate_runtime_health,
    overall_readiness_status,
    production_readiness_report,
)


CERTIFIED_BROKER_IDENTITY_SHA256 = "4" * 64
COMPLIANCE_POLICY_SHA256 = "5" * 64


class TestProductionLimitsAndEvidence(unittest.TestCase):
    def test_defaults_are_conservative(self) -> None:
        limits = ProductionLimits()
        self.assertEqual(limits.max_order_participation, 0.02)
        self.assertEqual(limits.max_gross_notional_multiple, 6.0)
        self.assertEqual(limits.max_margin_fraction, 0.35)
        self.assertEqual(limits.max_drawdown_fraction, 0.15)
        self.assertEqual(limits.max_data_age_seconds, 300.0)
        self.assertEqual(limits.max_reconciliation_age_seconds, 30.0)
        self.assertEqual(limits.max_runtime_health_age_seconds, 30.0)
        self.assertGreaterEqual(len(EXTERNAL_EVIDENCE_LABELS), 17)

    def test_invalid_limits_fail_fast(self) -> None:
        invalid = (
            {"max_order_participation": 0},
            {"max_order_participation": 1.01},
            {"max_gross_notional_multiple": np.inf},
            {"max_margin_fraction": -0.1},
            {"max_drawdown_fraction": np.nan},
            {"max_data_age_seconds": 0},
            {"max_reconciliation_age_seconds": 0},
            {"max_runtime_health_age_seconds": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ProductionLimits(**values)


class TestOrderIntents(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamp = pd.Timestamp("2025-01-02 16:00", tz="UTC")

    def build(self, **overrides: object) -> pd.DataFrame:
        inputs: dict[str, object] = {
            "current_contracts": {"ES": 0},
            "target_contracts": {"ES": 10},
            "notional_per_contract_usd": {"ES": 1_000.0},
            "margin_per_contract": {"ES": 500.0},
            "adv": {"ES": 100.0},
            "nav": 100_000.0,
            "decision_timestamp": self.timestamp,
            "limits": ProductionLimits(),
        }
        inputs.update(overrides)
        return build_order_intents(**inputs)

    def test_orders_are_deterministic_sorted_and_participation_capped(self) -> None:
        first = self.build(
            current_contracts={"NQ": 0, "ES": 0},
            target_contracts={"NQ": -1, "ES": 10},
            notional_per_contract_usd={"NQ": 1_000, "ES": 1_000},
            margin_per_contract={"NQ": 400, "ES": 500},
            adv={"NQ": 100, "ES": 100},
        )
        second = self.build(
            current_contracts={"ES": 0, "NQ": 0},
            target_contracts={"ES": 10, "NQ": -1},
            notional_per_contract_usd={"ES": 1_000, "NQ": 1_000},
            margin_per_contract={"ES": 500, "NQ": 400},
            adv={"ES": 100, "NQ": 100},
        )
        self.assertEqual(first["symbol"].tolist(), ["ES", "NQ"])
        self.assertEqual(first["order_id"].tolist(), second["order_id"].tolist())
        es = first.set_index("symbol").loc["ES"]
        self.assertEqual(es["requested_quantity"], 10)
        self.assertEqual(es["capped_quantity"], 2)
        self.assertEqual(es["quantity"], 2)
        self.assertEqual(es["side"], "BUY")
        self.assertAlmostEqual(es["projected_participation"], 0.02)
        self.assertEqual(es["projected_notional_usd"], 2_000)
        self.assertEqual(es["projected_margin_usd"], 1_000)
        self.assertTrue(es["approved"])
        self.assertEqual(es["status"], "APPROVED_PARTIAL")

    def test_portfolio_breach_blocks_risk_increase(self) -> None:
        row = self.build(
            current_contracts={"ES": 5},
            target_contracts={"ES": 6},
            notional_per_contract_usd={"ES": 200_000},
            adv={"ES": 1_000},
        ).iloc[0]
        self.assertFalse(row["approved"])
        self.assertEqual(row["status"], "BLOCKED")
        self.assertIn("gross notional", row["reason"])

    def test_reduce_only_order_is_allowed_during_a_limit_breach(self) -> None:
        row = self.build(
            current_contracts={"ES": 10},
            target_contracts={"ES": 0},
            notional_per_contract_usd={"ES": 100_000},
            margin_per_contract={"ES": 10_000},
            adv={"ES": 100},
        ).iloc[0]
        self.assertTrue(row["approved"])
        self.assertEqual(row["side"], "SELL")
        self.assertEqual(row["status"], "APPROVED_REDUCE_ONLY")
        self.assertEqual(row["style"], "REDUCE_ONLY")

    def test_missing_or_invalid_inputs_never_approve_an_order(self) -> None:
        cases = (
            {"notional_per_contract_usd": {}},
            {"adv": {"ES": 0}},
            {"nav": np.nan},
            {"decision_timestamp": None},
            {"decision_timestamp": pd.Timestamp("2025-01-02 16:00")},
            {"target_contracts": {"ES": 1.5}},
        )
        for values in cases:
            with self.subTest(values=values):
                row = self.build(**values).iloc[0]
                self.assertFalse(row["approved"])
                self.assertEqual(row["status"], "BLOCKED")

    def test_no_change_is_not_an_approved_order(self) -> None:
        row = self.build(current_contracts={"ES": 3}, target_contracts={"ES": 3}).iloc[0]
        self.assertEqual(row["side"], "HOLD")
        self.assertFalse(row["approved"])
        self.assertEqual(row["status"], "NO_ACTION")

    def test_exact_usd_notional_is_used_without_price_inference(self) -> None:
        row = self.build(
            target_contracts={"CL": 1},
            current_contracts={"CL": 0},
            notional_per_contract_usd={"CL": 10_000},
            margin_per_contract={"CL": 2_000},
            adv={"CL": 1_000},
        ).iloc[0]
        self.assertTrue(row["approved"])
        self.assertEqual(row["projected_notional_usd"], 10_000)


class TestRuntimeHealth(unittest.TestCase):
    POSITION_DIGEST = "a" * 64

    @classmethod
    def valid_inputs(cls, now: pd.Timestamp) -> dict[str, object]:
        return {
            "data_timestamp": now - pd.Timedelta(seconds=30),
            "reconciled_at": now - pd.Timedelta(seconds=20),
            "monitoring_evaluated_at": now - pd.Timedelta(seconds=10),
            "kill_switch_checked_at": now - pd.Timedelta(seconds=5),
            "position_snapshot_sha256": cls.POSITION_DIGEST,
            "current_time": now,
            "nav": 100,
            "peak_nav": 105,
            "gross_notional": 500,
            "margin_requirement": 20,
            "broker_connected": True,
            "broker_reconciled": True,
            "monitoring_healthy": True,
            "kill_switch_ready": True,
        }

    def test_complete_fresh_snapshot_passes(self) -> None:
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        health = evaluate_runtime_health(**self.valid_inputs(now))
        self.assertTrue(health.healthy)
        self.assertEqual(health.status, "PASS")
        self.assertEqual(health.reasons, ())
        self.assertEqual(health.data_age_seconds, 30)
        self.assertEqual(health.evaluated_at, now)
        self.assertEqual(health.data_timestamp, now - pd.Timedelta(seconds=30))
        self.assertEqual(health.reconciled_at, now - pd.Timedelta(seconds=20))
        self.assertEqual(
            health.monitoring_evaluated_at,
            now - pd.Timedelta(seconds=10),
        )
        self.assertEqual(
            health.kill_switch_checked_at,
            now - pd.Timedelta(seconds=5),
        )
        self.assertEqual(health.position_snapshot_sha256, self.POSITION_DIGEST)
        self.assertEqual(health.reconciliation_age_seconds, 20)
        self.assertEqual(health.monitoring_age_seconds, 10)
        self.assertEqual(health.kill_switch_age_seconds, 5)

    def test_defaults_stale_future_and_breaches_fail_closed(self) -> None:
        self.assertFalse(evaluate_runtime_health().healthy)
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        common = self.valid_inputs(now)
        common.pop("data_timestamp")
        common["peak_nav"] = 100
        common["gross_notional"] = 100
        common["margin_requirement"] = 10
        for timestamp in (now - pd.Timedelta(seconds=301), now + pd.Timedelta(seconds=1)):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(evaluate_runtime_health(data_timestamp=timestamp, **common).healthy)
        breached_inputs = self.valid_inputs(now)
        breached_inputs.update(
            data_timestamp=now - pd.Timedelta(seconds=21),
            nav=100,
            peak_nav=125,
            gross_notional=601,
            margin_requirement=36,
        )
        breached = evaluate_runtime_health(**breached_inputs)
        self.assertFalse(breached.healthy)
        self.assertGreaterEqual(len(breached.reasons), 3)

    def test_timezone_naive_health_timestamps_fail_closed(self) -> None:
        naive = pd.Timestamp("2025-01-02 16:00")
        health = evaluate_runtime_health(
            data_timestamp=naive,
            reconciled_at=naive,
            monitoring_evaluated_at=naive,
            kill_switch_checked_at=naive,
            position_snapshot_sha256=self.POSITION_DIGEST,
            current_time=naive,
            nav=100,
            peak_nav=100,
            gross_notional=100,
            margin_requirement=10,
            broker_connected=True,
            broker_reconciled=True,
            monitoring_healthy=True,
            kill_switch_ready=True,
        )
        self.assertFalse(health.healthy)
        self.assertIsNone(health.evaluated_at)
        self.assertIsNone(health.data_timestamp)
        self.assertIsNone(health.reconciled_at)
        self.assertIsNone(health.monitoring_evaluated_at)
        self.assertIsNone(health.kill_switch_checked_at)

    def test_missing_evidence_timestamp_or_position_digest_fails_closed(self) -> None:
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        for field in (
            "reconciled_at",
            "monitoring_evaluated_at",
            "kill_switch_checked_at",
            "position_snapshot_sha256",
        ):
            inputs = self.valid_inputs(now)
            inputs[field] = None
            with self.subTest(field=field):
                self.assertFalse(evaluate_runtime_health(**inputs).healthy)

        for digest in ("A" * 64, "a" * 63, "g" * 64, 123):
            inputs = self.valid_inputs(now)
            inputs["position_snapshot_sha256"] = digest
            with self.subTest(digest=digest):
                health = evaluate_runtime_health(**inputs)
                self.assertFalse(health.healthy)
                self.assertIn("position snapshot digest", " ".join(health.reasons))

    def test_evidence_timestamps_must_be_ordered(self) -> None:
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        cases = (
            (
                "reconciled_at",
                now - pd.Timedelta(seconds=31),
                "reconciliation timestamp predates market data",
            ),
            (
                "monitoring_evaluated_at",
                now - pd.Timedelta(seconds=21),
                "monitoring timestamp predates reconciliation",
            ),
            (
                "kill_switch_checked_at",
                now - pd.Timedelta(seconds=11),
                "kill-switch timestamp predates monitoring evaluation",
            ),
        )
        for field, value, reason in cases:
            inputs = self.valid_inputs(now)
            inputs[field] = value
            with self.subTest(field=field):
                health = evaluate_runtime_health(**inputs)
                self.assertFalse(health.healthy)
                self.assertIn(reason, health.reasons)

    def test_reconciliation_monitoring_and_kill_switch_checks_must_be_fresh(
        self,
    ) -> None:
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        stale_reconciliation = self.valid_inputs(now)
        stale_reconciliation.update(
            data_timestamp=now - pd.Timedelta(seconds=40),
            reconciled_at=now - pd.Timedelta(seconds=31),
        )
        health = evaluate_runtime_health(**stale_reconciliation)
        self.assertIn("broker reconciliation is stale", health.reasons)

        stale_runtime = self.valid_inputs(now)
        stale_runtime.update(
            data_timestamp=now - pd.Timedelta(seconds=50),
            reconciled_at=now - pd.Timedelta(seconds=40),
            monitoring_evaluated_at=now - pd.Timedelta(seconds=35),
            kill_switch_checked_at=now - pd.Timedelta(seconds=31),
            limits=ProductionLimits(max_reconciliation_age_seconds=60),
        )
        health = evaluate_runtime_health(**stale_runtime)
        self.assertIn("monitoring evaluation is stale", health.reasons)
        self.assertIn("kill-switch check is stale", health.reasons)


def safe_result() -> SimpleNamespace:
    daily = pd.DataFrame(
        {
            "nav": [100.0, 102.0, 99.0, 103.0],
            "equity": [1.0, 1.02, 0.99, 1.03],
            "net_return": [0.0, 0.02, -0.0294118, 0.040404],
            "cost": [0.0, 0.001, 0.001, 0.001],
            "gross_notional_multiple": [0.0, 2.0, 3.0, 1.0],
            "static_margin_fraction": [0.0, 0.10, 0.20, 0.05],
            "max_order_participation": [0.0, 0.01, 0.02, 0.01],
            "max_rebalance_participation": [0.0, 0.01, 0.02, 0.01],
            "max_roll_participation_proxy": [0.0, 0.01, 0.02, 0.01],
            "pending_markets": [0, 0, 1, 0],
        },
        index=pd.date_range("2025-01-01", periods=4),
    )
    return SimpleNamespace(daily=daily)


def artifact_backed_check(
    root: Path,
    result: SimpleNamespace,
) -> tuple[ArtifactBackedReadinessCheck, dict[str, Path]]:
    """Create a minimal, internally consistent deployment bundle for tests."""

    output_dir = root / "outputs"
    output_dir.mkdir()
    implementation = root / "model.py"
    implementation.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    implementation_hashes = {
        implementation.name: hashlib.sha256(implementation.read_bytes()).hexdigest()
    }
    model_fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}:{digest}"
            for name, digest in sorted(implementation_hashes.items())
        ).encode("utf-8")
    ).hexdigest()

    config_text = json.dumps(
        {"engine": "test", "target_vol": 0.07},
        indent=2,
        sort_keys=True,
    )
    config_path = output_dir / "strategy_config.json"
    config_path.write_text(config_text + "\n", encoding="utf-8")
    config_fingerprint = hashlib.sha256(config_text.encode("utf-8")).hexdigest()

    source_row_hash = "a" * 64
    source_manifest = output_dir / "source_manifest.csv"
    source_manifest.write_text(
        "relative_path,bytes,sha256\n"
        f"raw.csv,1,{source_row_hash}\n",
        encoding="utf-8",
    )
    source_fingerprint = hashlib.sha256(
        f"raw.csv:{source_row_hash}".encode("utf-8")
    ).hexdigest()

    daily_path = output_dir / "strategy_daily.csv"
    result.daily.to_csv(daily_path, index_label="date")
    output_paths = (config_path, source_manifest, daily_path)
    output_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_paths
    }
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "implementation_fingerprint_sha256": model_fingerprint,
                "config_sha256": config_fingerprint,
                "source_fingerprint_sha256": source_fingerprint,
                "daily_fingerprint_sha256": daily_frame_fingerprint(result.daily),
                "implementation_files": implementation_hashes,
                "output_files": output_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = root / "approval.txt"
    artifact.write_text("approved evidence\n", encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    records = [
        {
            "gate": gate,
            "artifact_path": (
                str(manifest.relative_to(root))
                if gate == "frozen_model_and_change_control"
                else artifact.name
            ),
            "artifact_sha256": (
                manifest_hash
                if gate == "frozen_model_and_change_control"
                else artifact_hash
            ),
            "issued_at": "2026-08-01T00:00:00Z",
            "expires_at": "2027-08-01T00:00:00Z",
            "model_fingerprint_sha256": model_fingerprint,
            "config_fingerprint_sha256": config_fingerprint,
            "source_fingerprint_sha256": source_fingerprint,
            "author": "Control Owner",
            "reviewer": "Independent Reviewer",
            "approved": True,
            "revoked": False,
            **(
                {"subject_sha256": CERTIFIED_BROKER_IDENTITY_SHA256}
                if gate == "certified_broker_adapter"
                else {"subject_sha256": COMPLIANCE_POLICY_SHA256}
                if gate == "compliance_approval"
                else {}
            ),
        }
        for gate in EXTERNAL_EVIDENCE_LABELS
    ]
    registry = root / "evidence.json"
    registry.write_text(
        json.dumps({"schema_version": 1, "evidence": records}),
        encoding="utf-8",
    )
    check = ArtifactBackedReadinessCheck(
        result=result,
        evidence_registry=registry,
        model_fingerprint_sha256=model_fingerprint,
        config_fingerprint_sha256=config_fingerprint,
        source_fingerprint_sha256=source_fingerprint,
        certified_broker_identity_sha256=CERTIFIED_BROKER_IDENTITY_SHA256,
        compliance_policy_sha256=COMPLIANCE_POLICY_SHA256,
        run_manifest_path=manifest,
        deployment_root=root,
    )
    return check, {
        "manifest": manifest,
        "implementation": implementation,
        "daily": daily_path,
    }


class TestArtifactBackedReadiness(unittest.TestCase):
    AS_OF = "2026-08-03T00:00:00Z"

    def test_daily_fingerprint_is_deterministic_and_content_bound(self) -> None:
        first = safe_result().daily
        second = first.copy()
        self.assertEqual(
            daily_frame_fingerprint(first),
            daily_frame_fingerprint(second),
        )
        second.iloc[0, second.columns.get_loc("net_return")] = 0.001
        self.assertNotEqual(
            daily_frame_fingerprint(first),
            daily_frame_fingerprint(second),
        )

    def test_valid_bundle_and_in_memory_daily_ledger_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = safe_result()
            check, _ = artifact_backed_check(Path(directory), result)
            report = check(as_of=self.AS_OF)
        self.assertEqual(overall_readiness_status(report), "READY")

    def test_missing_bundle_fails_closed(self) -> None:
        result = safe_result()
        check = ArtifactBackedReadinessCheck(
            result=result,
            evidence_registry="missing-evidence.json",
            model_fingerprint_sha256="1" * 64,
            config_fingerprint_sha256="2" * 64,
            source_fingerprint_sha256="3" * 64,
            certified_broker_identity_sha256=CERTIFIED_BROKER_IDENTITY_SHA256,
            compliance_policy_sha256=COMPLIANCE_POLICY_SHA256,
        )
        report = check(as_of=self.AS_OF)
        self.assertEqual(overall_readiness_status(report), "BLOCKED")
        self.assertTrue(
            report.loc[report["category"].eq("external evidence"), "reason"]
            .str.contains("deployment bundle verification failed")
            .all()
        )

    def test_changed_file_or_daily_ledger_fails_closed(self) -> None:
        for target in ("implementation", "daily_result"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                result = safe_result()
                check, paths = artifact_backed_check(Path(directory), result)
                if target == "implementation":
                    paths["implementation"].write_text(
                        "MODEL_VERSION = 2\n",
                        encoding="utf-8",
                    )
                else:
                    result.daily.iloc[
                        0,
                        result.daily.columns.get_loc("net_return"),
                    ] = 0.001
                report = check(as_of=self.AS_OF)
                self.assertEqual(overall_readiness_status(report), "BLOCKED")
                reasons = report.loc[
                    report["category"].eq("external evidence"),
                    "reason",
                ]
                self.assertTrue(
                    reasons.str.contains("deployment bundle verification failed").all()
                )

    def test_missing_daily_fingerprint_or_malformed_manifest_fails_closed(self) -> None:
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                result = safe_result()
                check, paths = artifact_backed_check(Path(directory), result)
                if malformed:
                    paths["manifest"].write_text("not json", encoding="utf-8")
                else:
                    document = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                    del document["daily_fingerprint_sha256"]
                    paths["manifest"].write_text(
                        json.dumps(document),
                        encoding="utf-8",
                    )
                report = check(as_of=self.AS_OF)
                self.assertEqual(overall_readiness_status(report), "BLOCKED")

    def test_rewritten_daily_file_and_self_declared_hashes_still_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = safe_result()
            check, paths = artifact_backed_check(Path(directory), result)
            result.daily.iloc[
                0,
                result.daily.columns.get_loc("net_return"),
            ] = 0.001
            result.daily.to_csv(paths["daily"], index_label="date")
            document = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            document["daily_fingerprint_sha256"] = daily_frame_fingerprint(
                result.daily
            )
            document["output_files"]["strategy_daily.csv"] = hashlib.sha256(
                paths["daily"].read_bytes()
            ).hexdigest()
            paths["manifest"].write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = check(as_of=self.AS_OF)
        self.assertEqual(overall_readiness_status(report), "BLOCKED")
        reasons = report.loc[
            report["category"].eq("external evidence"),
            "reason",
        ]
        self.assertTrue(reasons.str.contains("exact run manifest").all())


class TestProductionReadiness(unittest.TestCase):
    def test_all_critical_gates_must_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "approval.txt"
            artifact.write_text("approved evidence\n", encoding="utf-8")
            artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            model, config, source = "1" * 64, "2" * 64, "3" * 64
            records = [
                {
                    "gate": gate,
                    "artifact_path": artifact.name,
                    "artifact_sha256": artifact_hash,
                    "issued_at": "2026-08-01T00:00:00Z",
                    "expires_at": "2027-08-01T00:00:00Z",
                    "model_fingerprint_sha256": model,
                    "config_fingerprint_sha256": config,
                    "source_fingerprint_sha256": source,
                    "author": "Control Owner",
                    "reviewer": "Independent Reviewer",
                    "approved": True,
                    "revoked": False,
                    **(
                        {"subject_sha256": CERTIFIED_BROKER_IDENTITY_SHA256}
                        if gate == "certified_broker_adapter"
                        else {"subject_sha256": COMPLIANCE_POLICY_SHA256}
                        if gate == "compliance_approval"
                        else {}
                    ),
                }
                for gate in EXTERNAL_EVIDENCE_LABELS
            ]
            registry = root / "evidence.json"
            registry.write_text(
                json.dumps({"schema_version": 1, "evidence": records}),
                encoding="utf-8",
            )
            report = production_readiness_report(
                safe_result(),
                evidence_registry=registry,
                model_fingerprint_sha256=model,
                config_fingerprint_sha256=config,
                source_fingerprint_sha256=source,
                certified_broker_identity_sha256=CERTIFIED_BROKER_IDENTITY_SHA256,
                compliance_policy_sha256=COMPLIANCE_POLICY_SHA256,
                evidence_as_of="2026-08-03T00:00:00Z",
            )
        critical = report.loc[report["critical"].eq(True)]
        self.assertTrue(critical["status"].eq("PASS").all())
        self.assertEqual(overall_readiness_status(report), "READY")

    def test_external_evidence_defaults_to_blocked(self) -> None:
        report = production_readiness_report(safe_result())
        external = report.loc[report["category"].eq("external evidence")]
        self.assertEqual(len(external), len(EXTERNAL_EVIDENCE_LABELS))
        self.assertTrue(external["status"].eq("BLOCKED").all())
        self.assertEqual(overall_readiness_status(report), "BLOCKED")

    def test_missing_columns_nonfinite_values_and_limit_breaches_block(self) -> None:
        missing = production_readiness_report(SimpleNamespace(daily=pd.DataFrame({"nav": [100]})))
        self.assertEqual(overall_readiness_status(missing), "BLOCKED")

        result = safe_result()
        result.daily.loc[result.daily.index[-1], "gross_notional_multiple"] = 6.01
        result.daily.loc[result.daily.index[-1], "max_rebalance_participation"] = 0.021
        report = production_readiness_report(result)
        blocked = set(report.loc[report["status"].eq("BLOCKED"), "gate"])
        self.assertIn("historical_gross_notional", blocked)
        self.assertIn("historical_rebalance_participation", blocked)

        margin = report.set_index("gate").loc["historical_static_margin_proxy"]
        self.assertFalse(bool(margin["critical"]))
        self.assertEqual(margin["status"], "INFORMATIONAL")

    def test_negative_risk_metrics_and_initial_equity_loss_block(self) -> None:
        result = safe_result()
        result.daily.loc[:, "gross_notional_multiple"] = -0.1
        result.daily.loc[:, "equity"] = [0.80, 0.82, 0.83, 0.84]
        report = production_readiness_report(result)
        blocked = set(report.loc[report["status"].eq("BLOCKED"), "gate"])
        self.assertIn("historical_gross_notional", blocked)
        self.assertIn("historical_drawdown", blocked)

        drawdown = report.set_index("gate").loc["historical_drawdown"]
        self.assertAlmostEqual(drawdown["observed"], 0.20)

    def test_drawdown_requires_canonical_equity_and_never_falls_back_to_nav(
        self,
    ) -> None:
        result = safe_result()
        result.daily = result.daily.drop(columns="equity")
        report = production_readiness_report(result).set_index("gate")
        self.assertEqual(report.loc["required_daily_columns", "status"], "BLOCKED")
        self.assertIn("equity", report.loc["required_daily_columns", "observed"])
        self.assertEqual(report.loc["historical_drawdown", "status"], "BLOCKED")
        self.assertTrue(pd.isna(report.loc["historical_drawdown", "observed"]))

    def test_overall_status_rejects_malformed_or_vacuous_reports(self) -> None:
        self.assertEqual(overall_readiness_status(None), "BLOCKED")
        self.assertEqual(overall_readiness_status(pd.DataFrame()), "BLOCKED")
        self.assertEqual(
            overall_readiness_status(pd.DataFrame({"critical": [False], "status": ["PASS"]})),
            "BLOCKED",
        )

        valid = production_readiness_report(safe_result())
        valid.loc[valid["category"].eq("external evidence"), "status"] = "PASS"
        self.assertEqual(set(valid["gate"]), EXPECTED_READINESS_GATES)
        self.assertEqual(overall_readiness_status(valid), "READY")

        malformed = {
            "one fabricated pass": pd.DataFrame(
                {"gate": ["backtest_daily_available"], "critical": [True], "status": ["PASS"]}
            ),
            "missing gate": valid.iloc[:-1].copy(),
            "duplicate gate": pd.concat([valid, valid.iloc[[0]]], ignore_index=True),
            "extra critical gate": pd.concat(
                [
                    valid,
                    pd.DataFrame(
                        {"gate": ["unexpected"], "critical": [True], "status": ["PASS"]}
                    ),
                ],
                ignore_index=True,
            ),
            "non boolean critical": valid.assign(critical="True"),
        }
        for label, report in malformed.items():
            with self.subTest(label=label):
                self.assertEqual(overall_readiness_status(report), "BLOCKED")


if __name__ == "__main__":
    unittest.main()
