from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone, UTC

from delta1_strategy.research.registry import (
    FrozenPayload,
    RegistryEvent,
    ResearchCandidate,
    ResearchRegistryError,
    ResearchResultLink,
    ResearchTrialRegistry,
    ResearchWindow,
    ZERO_SHA256,
    content_sha256,
)


NOW = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)
FORMULA = "1" * 64
CONFIG = "2" * 64
SOURCE = "3" * 64
RESULT = "4" * 64
METRICS = "5" * 64
MANIFEST = "6" * 64


def candidate(candidate_id: str = "candidate-1", *, batch_id: str = "batch-1") -> ResearchCandidate:
    return ResearchCandidate(
        batch_id=batch_id,
        candidate_id=candidate_id,
        hypothesis="Slower multi-horizon trend may reduce turnover without using validation data.",
        formula_fingerprint_sha256=FORMULA,
        config_fingerprint_sha256=CONFIG,
        source_fingerprint_sha256=SOURCE,
        training_window=ResearchWindow(date(1990, 1, 1), date(1999, 12, 31)),
        validation_window=ResearchWindow(date(2000, 1, 1), date(2004, 12, 31)),
        costs=FrozenPayload.freeze(
            {
                "commission_model": "effective-dated-external",
                "spread_model": "quoted-half-spread",
                "slippage_model": "participation-conditioned",
            }
        ),
        risk_budget=FrozenPayload.freeze(
            {
                "portfolio_volatility_limit": 0.10,
                "gross_notional_limit": 4.0,
                "drawdown_shutdown": 0.15,
            }
        ),
        planned_metrics=(
            "net_sharpe",
            "sortino",
            "maximum_drawdown",
            "profit_factor",
            "expectancy_per_trade",
        ),
    )


def linked(registry: ResearchTrialRegistry, candidate_id: str = "candidate-1") -> ResearchTrialRegistry:
    return registry.link_result(
        candidate_id=candidate_id,
        result_id=f"result-{candidate_id}",
        result_artifact_sha256=RESULT,
        metrics_artifact_sha256=METRICS,
        run_manifest_sha256=MANIFEST,
        completed_at=NOW + timedelta(hours=1),
        occurred_at=NOW + timedelta(hours=2),
    )


class ResearchRecordTests(unittest.TestCase):
    def test_frozen_payload_detaches_mutable_input_and_verifies_hash(self) -> None:
        source = {"commission": 2.5, "nested": {"slippage": 1.0}}
        frozen = FrozenPayload.freeze(source)
        source["commission"] = 99.0
        detached = frozen.values()
        detached["commission"] = 10.0
        self.assertEqual(frozen.values()["commission"], 2.5)
        with self.assertRaisesRegex(ValueError, "does not match"):
            replace(frozen, sha256="f" * 64)

    def test_candidate_is_frozen_and_windows_cannot_overlap(self) -> None:
        item = candidate()
        with self.assertRaises(FrozenInstanceError):
            item.hypothesis = "changed after observation"  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "ordered and disjoint"):
            replace(
                item,
                validation_window=ResearchWindow(date(1999, 1, 1), date(2004, 1, 1)),
            )

    def test_planned_metrics_are_required_and_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            replace(candidate(), planned_metrics=())
        with self.assertRaisesRegex(ValueError, "duplicates"):
            replace(candidate(), planned_metrics=("Sharpe", "sharpe"))


class ResearchTrialRegistryTests(unittest.TestCase):
    def test_append_returns_new_content_addressed_registry(self) -> None:
        empty = ResearchTrialRegistry()
        registered = empty.register_candidate(candidate(), occurred_at=NOW)
        completed = linked(registered)

        self.assertEqual(empty.events, ())
        self.assertEqual(empty.max_candidates_per_batch, 3)
        self.assertNotEqual(registered.head_sha256, ZERO_SHA256)
        self.assertNotEqual(completed.head_sha256, registered.head_sha256)
        self.assertEqual(completed.events[-1].previous_event_sha256, registered.head_sha256)
        self.assertTrue(completed.is_extension_of(registered))
        self.assertFalse(registered.is_extension_of(completed))
        self.assertEqual(completed.results[0].candidate_sha256, candidate().candidate_sha256)

    def test_duplicate_and_mutated_candidate_ids_are_rejected(self) -> None:
        first = ResearchTrialRegistry().register_candidate(candidate(), occurred_at=NOW)
        with self.assertRaisesRegex(ResearchRegistryError, "already registered"):
            first.register_candidate(candidate(), occurred_at=NOW + timedelta(seconds=1))

        mutated = replace(candidate(), hypothesis="A post-registration mutation.")
        with self.assertRaisesRegex(ResearchRegistryError, "was mutated"):
            first.register_candidate(mutated, occurred_at=NOW + timedelta(seconds=1))

    def test_candidate_budget_is_enforced_per_batch(self) -> None:
        registry = ResearchTrialRegistry()
        for index in range(3):
            registry = registry.register_candidate(
                candidate(f"candidate-{index + 1}"),
                occurred_at=NOW + timedelta(seconds=index),
            )
        with self.assertRaisesRegex(ResearchRegistryError, "exceeds its candidate budget"):
            registry.register_candidate(
                candidate("candidate-4"),
                occurred_at=NOW + timedelta(seconds=3),
            )

        other_batch = registry.register_candidate(
            candidate("candidate-4", batch_id="batch-2"),
            occurred_at=NOW + timedelta(seconds=3),
        )
        self.assertEqual(len(other_batch.candidates), 4)

    def test_non_utc_and_backdated_events_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expressed in UTC"):
            ResearchTrialRegistry().register_candidate(
                candidate(),
                occurred_at=datetime(
                    2026,
                    8,
                    4,
                    16,
                    0,
                    tzinfo=timezone(timedelta(hours=8)),
                ),
            )
        registry = ResearchTrialRegistry().register_candidate(candidate(), occurred_at=NOW)
        with self.assertRaisesRegex(ResearchRegistryError, "backdated"):
            registry.register_candidate(
                candidate("candidate-2"),
                occurred_at=NOW - timedelta(seconds=1),
            )

    def test_result_before_registration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ResearchRegistryError, "precede candidate registration"):
            linked(ResearchTrialRegistry())

    def test_result_link_is_single_and_immutable(self) -> None:
        registry = ResearchTrialRegistry().register_candidate(candidate(), occurred_at=NOW)
        completed = linked(registry)
        with self.assertRaisesRegex(ResearchRegistryError, "linkage is immutable"):
            completed.link_result(
                candidate_id="candidate-1",
                result_id="replacement-result",
                result_artifact_sha256="f" * 64,
                metrics_artifact_sha256=METRICS,
                run_manifest_sha256=MANIFEST,
                completed_at=NOW + timedelta(hours=2),
                occurred_at=NOW + timedelta(hours=3),
            )

    def test_result_completion_cannot_predate_registration(self) -> None:
        registry = ResearchTrialRegistry().register_candidate(candidate(), occurred_at=NOW)
        with self.assertRaisesRegex(ResearchRegistryError, "completed before"):
            registry.link_result(
                candidate_id="candidate-1",
                result_id="result-early",
                result_artifact_sha256=RESULT,
                metrics_artifact_sha256=METRICS,
                run_manifest_sha256=MANIFEST,
                completed_at=NOW - timedelta(seconds=1),
                occurred_at=NOW + timedelta(seconds=1),
            )

    def test_tampered_event_chain_and_result_candidate_hash_are_rejected(self) -> None:
        registered = ResearchTrialRegistry().register_candidate(candidate(), occurred_at=NOW)
        event = registered.events[0]
        with self.assertRaisesRegex(ResearchRegistryError, "hash chain"):
            ResearchTrialRegistry(events=(replace(event, previous_event_sha256="f" * 64),))

        bad_result = ResearchResultLink(
            result_id="bad-result",
            candidate_id="candidate-1",
            candidate_sha256="f" * 64,
            result_artifact_sha256=RESULT,
            metrics_artifact_sha256=METRICS,
            run_manifest_sha256=MANIFEST,
            completed_at=NOW + timedelta(seconds=1),
        )
        bad_event = RegistryEvent(
            sequence=2,
            event_type="LINK_RESULT",
            occurred_at=NOW + timedelta(seconds=2),
            previous_event_sha256=event.event_sha256,
            registry_policy_sha256=registered.policy_sha256,
            result=bad_result,
        )
        with self.assertRaisesRegex(ResearchRegistryError, "fingerprint does not reconcile"):
            ResearchTrialRegistry(events=(event, bad_event))

    def test_content_hash_changes_when_frozen_candidate_content_changes(self) -> None:
        original = candidate()
        changed = replace(original, hypothesis="A separately declared hypothesis.")
        self.assertNotEqual(content_sha256(original), content_sha256(changed))


if __name__ == "__main__":
    unittest.main()
