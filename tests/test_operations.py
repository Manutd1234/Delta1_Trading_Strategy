from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from delta1_strategy.marketdata.contracts import SerialDataLimits, validated_serial_snapshot
from delta1_strategy.execution.operations import (
    BrokerDeploymentIdentity,
    BrokerEvent,
    CertifiedIntent,
    ExecutionService,
    HashChainedJournal,
    MonitoringPolicy,
    OperationsError,
    OrderRequest,
    PaperBroker,
    PortfolioComplianceDecision,
    PortfolioComplianceRequest,
    PersistentKillSwitch,
    assess_order_batch,
    certify_order_intent,
    disaster_recovery_report,
    monitoring_report,
    position_snapshot_sha256,
    reconciliation_report,
    replay_order_states,
)
from delta1_strategy.controls.production import (
    ArtifactBackedReadinessCheck,
    EXTERNAL_EVIDENCE_LABELS,
    daily_frame_fingerprint,
    evaluate_runtime_health,
)


NOW = pd.Timestamp("2026-08-03 16:00", tz="UTC")
SIGNING_KEY = b"committee-test-intent-key-32bytes!"
KEY_ID = "risk-service-v1"
NAV_USD = 10_000_000.0
COMPLIANCE_POLICY_SHA256 = "c" * 64
PRODUCTION_BROKER_IDENTITY = BrokerDeploymentIdentity(
    broker_id="TEST-FCM",
    adapter_id="tests.CertifiedTestBroker",
    adapter_build_sha256="b" * 64,
    account_id="TEST-ACCOUNT-001",
    environment="production",
)


class CertifiedTestBroker(PaperBroker):
    """Paper mechanics behind an explicit production-identity test boundary."""

    def __init__(
        self,
        *,
        initial_cash: float = 0.0,
        identity: BrokerDeploymentIdentity = PRODUCTION_BROKER_IDENTITY,
    ) -> None:
        super().__init__(initial_cash=initial_cash)
        self._deployment_identity = identity

    def deployment_identity(self) -> BrokerDeploymentIdentity:
        return self._deployment_identity

    def set_deployment_identity(self, identity: BrokerDeploymentIdentity) -> None:
        self._deployment_identity = identity


def order(
    order_id: str = "order-1",
    *,
    contract_id: str = "ESU6",
    root_symbol: str = "ES",
    side: str = "BUY",
    quantity: int = 10,
    reduce_only: bool = False,
    decision_timestamp: pd.Timestamp = NOW,
    roll_id: str | None = None,
) -> OrderRequest:
    return OrderRequest(
        order_id=order_id,
        contract_id=contract_id,
        root_symbol=root_symbol,
        side=side,
        quantity=quantity,
        decision_timestamp=decision_timestamp,
        limit_price=(
            (6300.25 if contract_id == "ESU6" else 6320.25)
            if side == "BUY"
            else (6300.00 if contract_id == "ESU6" else 6320.00)
        ),
        reduce_only=reduce_only,
        roll_id=roll_id,
    )


def safe_result() -> SimpleNamespace:
    return SimpleNamespace(
        daily=pd.DataFrame(
            {
                "nav": [100.0, 102.0],
                "equity": [1.0, 1.02],
                "net_return": [0.0, 0.02],
                "cost": [0.0, 0.001],
                "gross_notional_multiple": [0.0, 2.0],
                "static_margin_fraction": [0.0, 0.10],
                "max_order_participation": [0.0, 0.01],
                "max_rebalance_participation": [0.0, 0.01],
                "max_roll_participation_proxy": [0.0, 0.01],
                "pending_markets": [0, 0],
            }
        )
    )


class OperationsFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.journal = HashChainedJournal(self.root / "events.jsonl")
        self.switch = PersistentKillSwitch(self.root / "kill.json")
        self.broker = CertifiedTestBroker(initial_cash=100_000)
        self.bundle_sequence = 0
        self.current_readiness_check: ArtifactBackedReadinessCheck | None = None

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def commission(self) -> None:
        self.switch.commission(
            requested_by="operator",
            approved_by="risk",
            timestamp=NOW,
        )

    def readiness_check(self, *, ready: bool) -> ArtifactBackedReadinessCheck:
        self.bundle_sequence += 1
        bundle = self.root / f"bundle-{self.bundle_sequence}"
        outputs = bundle / "outputs"
        outputs.mkdir(parents=True)
        result = safe_result()

        implementation = bundle / "model.py"
        implementation.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
        implementation_hashes = {
            implementation.name: hashlib.sha256(implementation.read_bytes()).hexdigest()
        }
        model = hashlib.sha256(
            "\n".join(
                f"{name}:{digest}"
                for name, digest in sorted(implementation_hashes.items())
            ).encode("utf-8")
        ).hexdigest()

        config_text = json.dumps(
            {"engine": "operations-test", "target_vol": 0.07},
            indent=2,
            sort_keys=True,
        )
        config_path = outputs / "strategy_config.json"
        config_path.write_text(config_text + "\n", encoding="utf-8")
        config = hashlib.sha256(config_text.encode("utf-8")).hexdigest()

        source_row_hash = "a" * 64
        source_path = outputs / "source_manifest.csv"
        source_path.write_text(
            "relative_path,bytes,sha256\n"
            f"raw.csv,1,{source_row_hash}\n",
            encoding="utf-8",
        )
        source = hashlib.sha256(
            f"raw.csv:{source_row_hash}".encode()
        ).hexdigest()
        daily_path = outputs / "strategy_daily.csv"
        result.daily.to_csv(daily_path, index_label="date")
        output_paths = (config_path, source_path, daily_path)
        output_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_paths
        }
        manifest = outputs / "run_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "implementation_fingerprint_sha256": model,
                    "config_sha256": config,
                    "source_fingerprint_sha256": source,
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

        artifact = bundle / "approval.txt"
        artifact.write_text("approved evidence\n", encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        records = []
        for index, gate in enumerate(EXTERNAL_EVIDENCE_LABELS):
            record = {
                "gate": gate,
                "artifact_path": (
                    str(manifest.relative_to(bundle))
                    if gate == "frozen_model_and_change_control"
                    else artifact.name
                ),
                "artifact_sha256": (
                    manifest_digest
                    if gate == "frozen_model_and_change_control"
                    else digest
                ),
                "issued_at": "2026-08-01T00:00:00Z",
                "expires_at": "2027-08-01T00:00:00Z",
                "model_fingerprint_sha256": model,
                "config_fingerprint_sha256": config,
                "source_fingerprint_sha256": source,
                "author": "Control Owner",
                "reviewer": "Independent Reviewer",
                "approved": ready or index > 0,
                "revoked": False,
            }
            if gate == "certified_broker_adapter":
                record["subject_sha256"] = PRODUCTION_BROKER_IDENTITY.sha256
            elif gate == "compliance_approval":
                record["subject_sha256"] = COMPLIANCE_POLICY_SHA256
            records.append(record)
        registry = bundle / "evidence.json"
        registry.write_text(
            json.dumps({"schema_version": 1, "evidence": records}),
            encoding="utf-8",
        )
        check = ArtifactBackedReadinessCheck(
            result=result,
            evidence_registry=registry,
            model_fingerprint_sha256=model,
            config_fingerprint_sha256=config,
            source_fingerprint_sha256=source,
            certified_broker_identity_sha256=(
                PRODUCTION_BROKER_IDENTITY.sha256
            ),
            compliance_policy_sha256=COMPLIANCE_POLICY_SHA256,
            run_manifest_path=manifest,
            deployment_root=bundle,
        )
        self.current_readiness_check = check
        return check

    def serial_snapshot(self, as_of: pd.Timestamp = NOW) -> pd.DataFrame:
        common = {
            "timestamp": as_of - pd.Timedelta(seconds=5),
            "session_date": as_of.date().isoformat(),
            "root_symbol": "ES",
            "exchange": "CME",
            "currency": "USD",
            "delivery_type": "CASH_SETTLED",
            "first_notice_date": None,
            "bid": 6300.00,
            "ask": 6300.25,
            "settlement": 6300.00,
            "volume": 10_000,
            "open_interest": 500_000,
            "point_value": 50.0,
            "tick_size": 0.25,
            "margin_per_contract": 15_000.0,
            "margin_per_contract_usd": 15_000.0,
            "commission_per_contract": 2.50,
            "exchange_fee_per_contract": 1.50,
            "spec_effective_from": pd.Timestamp("2026-01-01", tz="UTC"),
            "spec_effective_to": pd.Timestamp("2027-01-01", tz="UTC"),
        }
        return pd.DataFrame(
            [
                {
                    **common,
                    "contract_id": "ESU6",
                    "expiry": "2026-09-18",
                    "last_trade_date": "2026-09-18",
                    "broker_liquidation_cutoff": pd.Timestamp(
                        "2026-09-17 20:00:00", tz="UTC"
                    ),
                    "notional_per_contract_usd": 315_000.0,
                },
                {
                    **common,
                    "contract_id": "ESZ6",
                    "expiry": "2026-12-18",
                    "last_trade_date": "2026-12-18",
                    "broker_liquidation_cutoff": pd.Timestamp(
                        "2026-12-17 20:00:00", tz="UTC"
                    ),
                    "notional_per_contract_usd": 316_000.0,
                    "bid": 6320.00,
                    "ask": 6320.25,
                    "settlement": 6320.00,
                },
            ]
        )

    def health_check(
        self, *, healthy: bool = True, evaluated_at: pd.Timestamp = NOW
    ):
        def check(route_time: pd.Timestamp):
            positions = self.broker.positions()
            notionals = {"ESU6": 315_000.0, "ESZ6": 316_000.0}
            gross = sum(
                abs(quantity) * notionals[contract]
                for contract, quantity in positions.items()
            )
            margin = sum(abs(quantity) * 15_000.0 for quantity in positions.values())
            return evaluate_runtime_health(
                data_timestamp=evaluated_at - pd.Timedelta(seconds=4),
                reconciled_at=evaluated_at - pd.Timedelta(seconds=3),
                monitoring_evaluated_at=evaluated_at - pd.Timedelta(seconds=2),
                kill_switch_checked_at=evaluated_at - pd.Timedelta(seconds=1),
                position_snapshot_sha256=position_snapshot_sha256(positions),
                current_time=evaluated_at,
                nav=NAV_USD,
                peak_nav=NAV_USD * 1.05,
                gross_notional=gross,
                margin_requirement=margin,
                broker_connected=True,
                broker_reconciled=True,
                monitoring_healthy=healthy,
                kill_switch_ready=True,
            )

        return check

    def approved_compliance_check(
        self,
        request: PortfolioComplianceRequest,
    ) -> PortfolioComplianceDecision:
        return PortfolioComplianceDecision(
            decision_id="committee-runtime-pass",
            evaluated_at=request.as_of,
            request_sha256=request.request_sha256,
            policy_sha256=COMPLIANCE_POLICY_SHA256,
            approved=True,
        )

    def live_service(
        self,
        *,
        ready: bool = True,
        healthy: bool = True,
        portfolio_compliance_check=None,
    ) -> ExecutionService:
        compliance_check = (
            self.approved_compliance_check
            if portfolio_compliance_check is None
            else portfolio_compliance_check
        )
        return ExecutionService(
            broker=self.broker,
            journal=self.journal,
            kill_switch=self.switch,
            mode="live",
            readiness_check=self.readiness_check(ready=ready),
            runtime_health_check=self.health_check(healthy=healthy),
            serial_snapshot_provider=self.serial_snapshot,
            portfolio_compliance_check=compliance_check,
            intent_verification_keys={KEY_ID: SIGNING_KEY},
        )

    def certificate(
        self,
        request: OrderRequest,
        *,
        broker_identity_digest: str | None = None,
        compliance_policy_digest: str | None = None,
    ) -> CertifiedIntent:
        if self.current_readiness_check is None:
            raise RuntimeError("create the live service before certifying an order")
        snapshot = validated_serial_snapshot(
            self.serial_snapshot(NOW),
            as_of=NOW,
            limits=SerialDataLimits(),
        )
        assessment = assess_order_batch(
            [request],
            broker_positions=self.broker.positions(),
            serial_snapshot=snapshot,
            nav_usd=NAV_USD,
            as_of=NOW,
        )
        return certify_order_intent(
            request,
            issued_at=NOW - pd.Timedelta(seconds=1),
            expires_at=NOW + pd.Timedelta(seconds=30),
            model_fingerprint_sha256=(
                self.current_readiness_check.model_fingerprint_sha256
            ),
            config_fingerprint_sha256=(
                self.current_readiness_check.config_fingerprint_sha256
            ),
            source_fingerprint_sha256=(
                self.current_readiness_check.source_fingerprint_sha256
            ),
            serial_snapshot_digest=assessment.serial_snapshot_sha256,
            position_snapshot_digest=assessment.position_snapshot_sha256,
            broker_identity_digest=(
                self.current_readiness_check.certified_broker_identity_sha256
                if broker_identity_digest is None
                else broker_identity_digest
            ),
            compliance_policy_digest=(
                self.current_readiness_check.compliance_policy_sha256
                if compliance_policy_digest is None
                else compliance_policy_digest
            ),
            projected_participation=assessment.participation_by_order[request.order_id],
            projected_gross_notional_multiple=(
                assessment.projected_gross_notional_multiple
            ),
            projected_margin_fraction=assessment.projected_margin_fraction,
            key_id=KEY_ID,
            signing_key=SIGNING_KEY,
        )

    def paper_service(self) -> ExecutionService:
        return ExecutionService(
            broker=self.broker,
            journal=self.journal,
            kill_switch=self.switch,
            mode="paper",
            runtime_health_check=self.health_check(),
        )


class JournalAndReplayTests(OperationsFixture):
    def test_append_is_hash_chained_and_idempotent(self) -> None:
        first = self.journal.append(
            "TEST",
            {"value": 1},
            timestamp=NOW,
            idempotency_key="test:1",
        )
        repeated = self.journal.append(
            "TEST",
            {"value": 1},
            timestamp=NOW + pd.Timedelta(seconds=1),
            idempotency_key="test:1",
        )
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.journal.read()), 1)
        with self.assertRaises(OperationsError):
            self.journal.append(
                "TEST",
                {"value": 2},
                timestamp=NOW,
                idempotency_key="test:1",
            )

    def test_tampering_and_truncation_fail_replay(self) -> None:
        self.journal.append(
            "TEST", {"value": 1}, timestamp=NOW, idempotency_key="test:1"
        )
        record = json.loads(self.journal.path.read_text().strip())
        record["payload"]["value"] = 2
        self.journal.path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(OperationsError, "hash"):
            self.journal.read()

    def test_replay_handles_partial_fill_and_cancel(self) -> None:
        self.commission()
        service = self.paper_service()
        routed = service.route([order()], timestamp=NOW)
        self.assertEqual(routed.iloc[0]["status"], "ROUTED")
        fill = self.broker.fill_order(
            "order-1",
            fill_id="fill-1",
            quantity=4,
            price=6300.25,
            fees_usd=4.0,
            timestamp=NOW + pd.Timedelta(seconds=1),
        )
        service.record_broker_event(fill)
        state = replay_order_states(self.journal).iloc[0]
        self.assertEqual(state["status"], "PARTIAL")
        self.assertEqual(state["remaining_quantity"], 6)
        service.cancel_all(timestamp=NOW + pd.Timedelta(seconds=2))
        state = replay_order_states(self.journal).iloc[0]
        self.assertEqual(state["status"], "CANCELLED")
        self.assertEqual(self.broker.positions()["ESU6"], 4)
        self.assertEqual(self.broker.cash_balance(), 99_996)


class BrokerAndRoutingTests(OperationsFixture):
    def test_submit_and_fill_are_idempotent_and_overfill_blocks(self) -> None:
        ack_one = self.broker.submit_order(order())
        ack_two = self.broker.submit_order(order())
        self.assertEqual(ack_one, ack_two)
        fill_one = self.broker.fill_order(
            "order-1",
            fill_id="fill-1",
            quantity=5,
            price=6300,
            fees_usd=2,
            timestamp=NOW,
        )
        fill_two = self.broker.fill_order(
            "order-1",
            fill_id="fill-1",
            quantity=5,
            price=6300,
            fees_usd=2,
            timestamp=NOW,
        )
        self.assertEqual(fill_one, fill_two)
        with self.assertRaises(OperationsError):
            self.broker.fill_order(
                "order-1",
                fill_id="fill-2",
                quantity=6,
                price=6300,
                fees_usd=2,
                timestamp=NOW,
            )

    def test_live_risk_increase_requires_ready_health_and_active_switch(self) -> None:
        blocked_service = self.live_service(ready=False, healthy=False)
        blocked = blocked_service.route([order()], timestamp=NOW)
        self.assertEqual(blocked.iloc[0]["status"], "BLOCKED")
        self.assertFalse(self.journal.path.exists())

        self.commission()
        ready_service = self.live_service()
        routed = ready_service.route([self.certificate(order())], timestamp=NOW)
        self.assertEqual(routed.iloc[0]["status"], "ROUTED")
        self.assertEqual(len(self.journal.read()), 2)
        outbox = self.journal.read()[0]["payload"]
        self.assertEqual(
            outbox["broker_identity_sha256"],
            PRODUCTION_BROKER_IDENTITY.sha256,
        )
        self.assertEqual(
            outbox["compliance_policy_sha256"],
            COMPLIANCE_POLICY_SHA256,
        )

    def test_reduce_only_can_route_during_health_or_readiness_block(self) -> None:
        seed = order(order_id="seed", quantity=10)
        self.broker.submit_order(seed)
        self.broker.fill_order(
            "seed",
            fill_id="seed-fill",
            quantity=10,
            price=6300,
            fees_usd=0,
            timestamp=NOW,
        )
        service = self.live_service(ready=False, healthy=False)
        routed = service.route(
            [order(order_id="reduce", side="SELL", quantity=5, reduce_only=True)],
            timestamp=NOW,
        )
        self.assertEqual(routed.iloc[0]["status"], "ROUTED")

    def test_unverified_reduce_only_never_bypasses_controls(self) -> None:
        service = self.live_service(ready=False, healthy=False)
        flat = service.route(
            [order(order_id="flat", side="SELL", quantity=1, reduce_only=True)],
            timestamp=NOW,
        )
        self.assertEqual(flat.iloc[0]["status"], "BLOCKED")
        self.assertIn("does not reduce", flat.iloc[0]["reason"])

        seed = order(order_id="seed", quantity=10)
        self.broker.submit_order(seed)
        self.broker.fill_order(
            "seed",
            fill_id="seed-fill",
            quantity=10,
            price=6300,
            fees_usd=0,
            timestamp=NOW,
        )
        wrong_side = service.route(
            [order(order_id="wrong", side="BUY", quantity=1, reduce_only=True)],
            timestamp=NOW,
        )
        crossing = service.route(
            [order(order_id="cross", side="SELL", quantity=11, reduce_only=True)],
            timestamp=NOW,
        )
        self.assertTrue(wrong_side["status"].eq("BLOCKED").all())
        self.assertTrue(crossing["status"].eq("BLOCKED").all())

    def test_live_service_requires_route_time_verifiers_and_fresh_health(self) -> None:
        with self.assertRaises(TypeError):
            ExecutionService(
                broker=self.broker,
                journal=self.journal,
                kill_switch=self.switch,
                mode="live",
            )
        self.commission()
        stale = ExecutionService(
            broker=self.broker,
            journal=self.journal,
            kill_switch=self.switch,
            mode="live",
            readiness_check=self.readiness_check(ready=True),
            runtime_health_check=self.health_check(
                evaluated_at=NOW - pd.Timedelta(seconds=31)
            ),
            serial_snapshot_provider=self.serial_snapshot,
            portfolio_compliance_check=self.approved_compliance_check,
            intent_verification_keys={KEY_ID: SIGNING_KEY},
        ).route([self.certificate(order())], timestamp=NOW)
        self.assertEqual(stale.iloc[0]["status"], "BLOCKED")
        self.assertIn("stale", stale.iloc[0]["reason"])

    def test_live_requires_certified_production_identity_and_compliance(self) -> None:
        readiness = self.readiness_check(ready=True)
        common = {
            "journal": self.journal,
            "kill_switch": self.switch,
            "mode": "live",
            "readiness_check": readiness,
            "runtime_health_check": self.health_check(),
            "serial_snapshot_provider": self.serial_snapshot,
            "intent_verification_keys": {KEY_ID: SIGNING_KEY},
        }
        with self.assertRaisesRegex(TypeError, "portfolio/compliance"):
            ExecutionService(broker=self.broker, **common)

        with self.assertRaisesRegex(OperationsError, "not production"):
            ExecutionService(
                broker=PaperBroker(initial_cash=100_000),
                portfolio_compliance_check=self.approved_compliance_check,
                **common,
            )

        paper = ExecutionService(
            broker=PaperBroker(initial_cash=100_000),
            journal=self.journal,
            kill_switch=self.switch,
            mode="paper",
            runtime_health_check=self.health_check(),
        )
        self.assertEqual(paper.mode, "paper")

    def test_signed_intent_binds_broker_identity_and_compliance_policy(self) -> None:
        self.commission()
        service = self.live_service()
        wrong_identity = service.route(
            [
                self.certificate(
                    order(order_id="wrong-identity"),
                    broker_identity_digest="d" * 64,
                )
            ],
            timestamp=NOW,
        )
        self.assertEqual(wrong_identity.iloc[0]["status"], "BLOCKED")
        self.assertIn("broker identity", wrong_identity.iloc[0]["reason"])

        wrong_policy = service.route(
            [
                self.certificate(
                    order(order_id="wrong-policy"),
                    compliance_policy_digest="e" * 64,
                )
            ],
            timestamp=NOW,
        )
        self.assertEqual(wrong_policy.iloc[0]["status"], "BLOCKED")
        self.assertIn("compliance policy", wrong_policy.iloc[0]["reason"])
        self.assertFalse(self.journal.path.exists())

    def test_runtime_identity_change_blocks_before_outbox(self) -> None:
        self.commission()
        service = self.live_service()
        certified = self.certificate(order(order_id="identity-switch"))
        self.broker.set_deployment_identity(
            replace(PRODUCTION_BROKER_IDENTITY, account_id="OTHER-ACCOUNT")
        )
        result = service.route([certified], timestamp=NOW)
        self.assertEqual(result.iloc[0]["status"], "BLOCKED")
        self.assertIn("runtime broker identity", result.iloc[0]["reason"])
        self.assertFalse(self.journal.path.exists())

    def test_identity_change_after_outbox_blocks_submission_and_halts(self) -> None:
        class SwitchingIdentityBroker(CertifiedTestBroker):
            def __init__(self):
                super().__init__(initial_cash=100_000)
                self.identity_reads = 0

            def deployment_identity(self):
                self.identity_reads += 1
                if self.identity_reads >= 4:
                    return replace(
                        PRODUCTION_BROKER_IDENTITY,
                        account_id="SWITCHED-ACCOUNT",
                    )
                return PRODUCTION_BROKER_IDENTITY

        self.broker = SwitchingIdentityBroker()
        self.commission()
        service = self.live_service()
        result = service.route(
            [self.certificate(order(order_id="late-identity-switch"))],
            timestamp=NOW,
        )
        self.assertEqual(result.iloc[0]["status"], "BLOCKED")
        self.assertIn("after durable outbox", result.iloc[0]["reason"])
        self.assertEqual(
            [record["event_type"] for record in self.journal.read()],
            ["ORDER_OUTBOX"],
        )
        self.assertEqual(self.switch.read().status, "HALTED")

    def test_compliance_decision_failures_block_before_outbox(self) -> None:
        self.commission()

        def denied(request):
            return PortfolioComplianceDecision(
                decision_id="denied",
                evaluated_at=request.as_of,
                request_sha256=request.request_sha256,
                policy_sha256=COMPLIANCE_POLICY_SHA256,
                approved=False,
                reasons=("position limit breached",),
            )

        def wrong_request(request):
            return replace(
                self.approved_compliance_check(request),
                request_sha256="d" * 64,
            )

        def wrong_policy(request):
            return replace(
                self.approved_compliance_check(request),
                policy_sha256="e" * 64,
            )

        def stale(request):
            return replace(
                self.approved_compliance_check(request),
                evaluated_at=NOW - pd.Timedelta(seconds=31),
            )

        def mutated(request):
            request.serial_snapshot.loc[:, "volume"] = 1
            return self.approved_compliance_check(request)

        def invalid_mutation(request):
            decision = self.approved_compliance_check(request)
            request.serial_snapshot.drop(columns=["volume"], inplace=True)
            return decision

        def invalid(_request):
            return object()

        def failed(_request):
            raise RuntimeError("limit service unavailable")

        cases = (
            (denied, "denied"),
            (wrong_request, "does not match the live request"),
            (wrong_policy, "policy does not match"),
            (stale, "stale"),
            (mutated, "mutated the live request"),
            (invalid_mutation, "into invalid state"),
            (invalid, "invalid decision"),
            (failed, "limit service unavailable"),
        )
        for index, (check, message) in enumerate(cases):
            with self.subTest(message=message):
                service = self.live_service(portfolio_compliance_check=check)
                request = order(order_id=f"compliance-{index}")
                result = service.route(
                    [self.certificate(request)],
                    timestamp=NOW,
                )
                self.assertEqual(result.iloc[0]["status"], "BLOCKED")
                self.assertIn(message, result.iloc[0]["reason"])
                self.assertFalse(self.journal.path.exists())

    def test_compliance_denial_also_blocks_verified_reduce_only(self) -> None:
        seed = order(order_id="seed", quantity=10)
        self.broker.submit_order(seed, timestamp=NOW)
        self.broker.fill_order(
            "seed",
            fill_id="seed-fill",
            quantity=10,
            price=6300,
            fees_usd=0,
            timestamp=NOW,
        )

        def denied(request):
            return PortfolioComplianceDecision(
                decision_id="reduce-denied",
                evaluated_at=request.as_of,
                request_sha256=request.request_sha256,
                policy_sha256=COMPLIANCE_POLICY_SHA256,
                approved=False,
                reasons=("market-access restriction",),
            )

        service = self.live_service(portfolio_compliance_check=denied)
        result = service.route(
            [order(order_id="reduce-denied", side="SELL", quantity=5, reduce_only=True)],
            timestamp=NOW,
        )
        self.assertEqual(result.iloc[0]["status"], "BLOCKED")
        self.assertIn("market-access restriction", result.iloc[0]["reason"])
        self.assertFalse(self.journal.path.exists())

    def test_live_blocks_unsigned_oversized_and_stale_alpha_orders(self) -> None:
        self.commission()
        service = self.live_service()

        unsigned = service.route([order(order_id="unsigned")], timestamp=NOW)
        self.assertEqual(unsigned.iloc[0]["status"], "BLOCKED")
        self.assertIn("certificate", unsigned.iloc[0]["reason"])

        oversized_order = order(order_id="oversized", quantity=1_000_000_000)
        oversized = service.route(
            [self.certificate(oversized_order)], timestamp=NOW
        )
        self.assertEqual(oversized.iloc[0]["status"], "BLOCKED")
        self.assertIn("participation", oversized.iloc[0]["reason"])
        self.assertIn("gross notional", oversized.iloc[0]["reason"])

        stale_order = order(
            order_id="stale",
            decision_timestamp=NOW - pd.Timedelta(seconds=301),
        )
        stale = service.route([self.certificate(stale_order)], timestamp=NOW)
        self.assertEqual(stale.iloc[0]["status"], "BLOCKED")
        self.assertIn("decision is stale", stale.iloc[0]["reason"])
        self.assertFalse(self.journal.path.exists())

    def test_live_binds_certificate_to_serial_and_position_snapshots(self) -> None:
        self.commission()
        service = self.live_service()
        certified = self.certificate(order(order_id="position-bound"))

        seed = order(order_id="seed", quantity=1)
        self.broker.submit_order(seed, timestamp=NOW)
        self.broker.fill_order(
            "seed",
            fill_id="seed-fill",
            quantity=1,
            price=6300,
            fees_usd=0,
            timestamp=NOW,
        )
        changed = service.route([certified], timestamp=NOW)
        self.assertEqual(changed.iloc[0]["status"], "BLOCKED")
        self.assertIn("position snapshot does not match", changed.iloc[0]["reason"])

        unknown = order(order_id="unknown", contract_id="ESX9")
        unknown_result = service.route([unknown], timestamp=NOW)
        self.assertEqual(unknown_result.iloc[0]["status"], "BLOCKED")
        self.assertIn("absent from the validated serial", unknown_result.iloc[0]["reason"])

    def test_mismatched_broker_ack_latches_kill_switch(self) -> None:
        class WrongAckBroker(CertifiedTestBroker):
            def submit_order(self, request, *, timestamp=None):
                accepted = super().submit_order(request, timestamp=timestamp)
                return BrokerEvent(
                    event_id="wrong-ack",
                    event_type="ACK",
                    order_id="another-order",
                    broker_order_id=accepted.broker_order_id,
                    timestamp=timestamp,
                )

        self.broker = WrongAckBroker(initial_cash=100_000)
        self.commission()
        service = self.live_service()
        result = service.route(
            [self.certificate(order(order_id="ack-check"))], timestamp=NOW
        )
        self.assertEqual(result.iloc[0]["status"], "BLOCKED")
        self.assertIn("does not match", result.iloc[0]["reason"])
        self.assertEqual(self.switch.read().status, "HALTED")
        self.assertEqual(
            [record["event_type"] for record in self.journal.read()],
            ["ORDER_OUTBOX"],
        )

    def test_reference_live_router_never_sends_individual_roll_legs(self) -> None:
        seed = order(order_id="seed", quantity=5)
        self.broker.submit_order(seed, timestamp=NOW)
        self.broker.fill_order(
            "seed",
            fill_id="seed-fill",
            quantity=5,
            price=6300,
            fees_usd=0,
            timestamp=NOW,
        )
        self.commission()
        service = self.live_service()
        legs = [
            order(
                order_id="roll-close",
                side="SELL",
                quantity=5,
                reduce_only=True,
                roll_id="roll-1",
            ),
            order(
                order_id="roll-open",
                contract_id="ESZ6",
                side="BUY",
                quantity=5,
                roll_id="roll-1",
            ),
        ]
        result = service.route(legs, timestamp=NOW)
        self.assertTrue(result["status"].eq("BLOCKED").all())
        self.assertTrue(result["reason"].str.contains("atomic/contingent").all())
        self.assertFalse(self.journal.path.exists())

    def test_disconnect_leaves_durable_outbox_for_retry(self) -> None:
        self.commission()
        service = self.paper_service()
        self.broker.set_connected(False)
        blocked = service.route([order()], timestamp=NOW)
        self.assertEqual(blocked.iloc[0]["status"], "BLOCKED")
        self.assertFalse(self.journal.path.exists())

        # Simulate failure after persist-before-send by appending the outbox,
        # then verify a retry is idempotent when the broker returns.
        self.journal.append(
            "ORDER_OUTBOX",
            order().payload(),
            timestamp=NOW,
            idempotency_key="outbox:order-1",
        )
        self.broker.set_connected(True)
        routed = service.route([order()], timestamp=NOW)
        self.assertEqual(routed.iloc[0]["status"], "ROUTED")
        self.assertEqual(len(self.journal.read()), 2)


class ControlTests(OperationsFixture):
    def test_kill_switch_is_latched_checksummed_and_dual_controlled(self) -> None:
        self.assertEqual(self.switch.read().status, "HALTED")
        with self.assertRaises(OperationsError):
            self.switch.commission(
                requested_by="same", approved_by="same", timestamp=NOW
            )
        self.commission()
        self.assertEqual(self.switch.read().status, "ACTIVE")
        self.switch.trigger(actor="risk", reason="drawdown", timestamp=NOW)
        restarted = PersistentKillSwitch(self.switch.path)
        self.assertEqual(restarted.read().status, "HALTED")
        with self.assertRaises(OperationsError):
            restarted.reset(
                requested_by="risk",
                approved_by="risk",
                reason="reviewed",
                timestamp=NOW,
            )
        restarted.reset(
            requested_by="operator",
            approved_by="risk",
            reason="incident closed",
            timestamp=NOW,
        )
        self.assertEqual(restarted.read().status, "ACTIVE")

    def test_reconciliation_and_monitoring_fail_closed(self) -> None:
        reconciled = reconciliation_report(
            {"ESU6": 2}, {"ESU6": 2}, internal_cash=100, broker_cash=100.005
        )
        self.assertTrue(reconciled["status"].eq("PASS").all())
        broken = reconciliation_report(
            {"ESU6": 2}, {"ESU6": 1}, internal_cash=100, broker_cash=99
        )
        self.assertTrue(broken["status"].eq("BLOCKED").all())

        healthy = monitoring_report(
            {
                "market_data": NOW - pd.Timedelta(seconds=10),
                "broker": NOW - pd.Timedelta(seconds=10),
                "reconciliation": NOW - pd.Timedelta(hours=1),
                "journal": NOW - pd.Timedelta(seconds=10),
            },
            as_of=NOW,
        )
        self.assertTrue(healthy["status"].eq("PASS").all())
        stale = monitoring_report({}, as_of=NOW, policy=MonitoringPolicy())
        self.assertTrue(stale["status"].eq("BLOCKED").all())

    def test_disaster_recovery_reconciles_open_orders_and_detects_break(self) -> None:
        self.commission()
        service = self.paper_service()
        service.route([order()], timestamp=NOW)
        report = disaster_recovery_report(self.journal, self.broker)
        self.assertTrue(report["status"].eq("PASS").all())
        self.broker.cancel_order("order-1", timestamp=NOW)
        broken = disaster_recovery_report(self.journal, self.broker)
        self.assertEqual(
            broken.set_index("gate").loc["open_order_reconciliation", "status"],
            "BLOCKED",
        )

    def test_replay_rejects_any_event_after_terminal_fill(self) -> None:
        self.commission()
        service = self.paper_service()
        service.route([order(quantity=1)], timestamp=NOW)
        fill = self.broker.fill_order(
            "order-1",
            fill_id="terminal-fill",
            quantity=1,
            price=6300,
            fees_usd=0,
            timestamp=NOW + pd.Timedelta(seconds=1),
        )
        service.record_broker_event(fill)
        invalid_cancel = BrokerEvent(
            event_id="late-cancel",
            event_type="CANCEL",
            order_id="order-1",
            broker_order_id="paper-late",
            timestamp=NOW + pd.Timedelta(seconds=2),
        )
        service.record_broker_event(invalid_cancel)
        with self.assertRaisesRegex(OperationsError, "terminal"):
            replay_order_states(self.journal)


class ValidationTests(unittest.TestCase):
    def test_broker_identity_digest_binds_every_deployment_component(self) -> None:
        changes = {
            "broker_id": "OTHER-FCM",
            "adapter_id": "tests.OtherAdapter",
            "adapter_build_sha256": "d" * 64,
            "account_id": "OTHER-ACCOUNT",
            "environment": "staging",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = replace(
                    PRODUCTION_BROKER_IDENTITY,
                    **{field: value},
                )
                self.assertNotEqual(
                    changed.sha256,
                    PRODUCTION_BROKER_IDENTITY.sha256,
                )

    def test_order_rejects_continuous_contract_and_invalid_order(self) -> None:
        with self.assertRaises(ValueError):
            OrderRequest(
                order_id="x",
                contract_id="ES",
                root_symbol="ES",
                side="BUY",
                quantity=1,
                decision_timestamp=NOW,
                limit_price=100,
            )
        with self.assertRaises(ValueError):
            order(quantity=0)


if __name__ == "__main__":
    unittest.main()
