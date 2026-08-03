"""Append-only governance for pre-registered research trials.

The registry records hypotheses and immutable evidence links.  It deliberately
contains no parameter search, optimizer, target-return logic, or strategy
selection rule.  Appends return a new frozen registry whose content-addressed
event chain retains the complete prior registry state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any


SCHEMA_VERSION = 1
ZERO_SHA256 = "0" * 64
REGISTER_CANDIDATE = "REGISTER_CANDIDATE"
LINK_RESULT = "LINK_RESULT"


class ResearchRegistryError(ValueError):
    """Raised when a research event would violate registry governance."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _utc(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            timestamp = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{label} is not a valid ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{label} must be a UTC timestamp")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be expressed in UTC")
    return timestamp.astimezone(timezone.utc)


def _date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO calendar date") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc(value, "timestamp").isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _json_value(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("payload must contain finite JSON-compatible values") from exc


def content_sha256(record: Any) -> str:
    """Hash a record with domain separation by its concrete record type."""

    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("record must be a dataclass instance")
    payload = {
        "record_type": record.__class__.__qualname__,
        "schema_version": SCHEMA_VERSION,
        "payload": asdict(record),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenPayload:
    """Canonical, deeply immutable JSON used for costs and risk budgets."""

    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        text = _text(self.canonical_json, "canonical_json")
        try:
            parsed = json.loads(
                text,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid numeric constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("canonical_json must be valid finite JSON") from exc
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("frozen payload must be a non-empty JSON object")
        canonical = _canonical_json(parsed)
        if canonical != text:
            raise ValueError("canonical_json is not in canonical form")
        observed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if _sha256(self.sha256, "sha256") != observed:
            raise ValueError("frozen payload SHA-256 does not match its content")

    @classmethod
    def freeze(cls, values: Mapping[str, Any]) -> FrozenPayload:
        if not isinstance(values, Mapping) or not values:
            raise ValueError("values must be a non-empty mapping")
        canonical = _canonical_json(dict(values))
        return cls(
            canonical_json=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def values(self) -> dict[str, Any]:
        """Return a detached copy; mutating it cannot alter the frozen record."""

        return json.loads(self.canonical_json)


@dataclass(frozen=True)
class ResearchWindow:
    """Inclusive calendar-date window fixed before a candidate is evaluated."""

    start: date
    end: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _date(self.start, "window start"))
        object.__setattr__(self, "end", _date(self.end, "window end"))
        if self.start > self.end:
            raise ValueError("research window start must not follow its end")


@dataclass(frozen=True)
class ResearchCandidate:
    """One fully specified hypothesis registered before result observation."""

    batch_id: str
    candidate_id: str
    hypothesis: str
    formula_fingerprint_sha256: str
    config_fingerprint_sha256: str
    source_fingerprint_sha256: str
    training_window: ResearchWindow
    validation_window: ResearchWindow
    costs: FrozenPayload
    risk_budget: FrozenPayload
    planned_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("batch_id", "candidate_id", "hypothesis"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "formula_fingerprint_sha256",
            "config_fingerprint_sha256",
            "source_fingerprint_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.training_window, ResearchWindow):
            raise TypeError("training_window must be a ResearchWindow")
        if not isinstance(self.validation_window, ResearchWindow):
            raise TypeError("validation_window must be a ResearchWindow")
        if self.training_window.end >= self.validation_window.start:
            raise ValueError("training and validation windows must be ordered and disjoint")
        if not isinstance(self.costs, FrozenPayload):
            raise TypeError("costs must be a FrozenPayload")
        if not isinstance(self.risk_budget, FrozenPayload):
            raise TypeError("risk_budget must be a FrozenPayload")
        if isinstance(self.planned_metrics, (str, bytes)):
            raise ValueError("planned_metrics must be a non-empty sequence")
        try:
            metrics = tuple(_text(value, "planned metric") for value in self.planned_metrics)
        except TypeError as exc:
            raise ValueError("planned_metrics must be a non-empty sequence") from exc
        if not metrics:
            raise ValueError("planned_metrics must not be empty")
        if len({metric.casefold() for metric in metrics}) != len(metrics):
            raise ValueError("planned_metrics contains duplicates")
        object.__setattr__(self, "planned_metrics", metrics)

    @property
    def candidate_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ResearchResultLink:
    """Immutable link from a registered candidate to result artifacts."""

    result_id: str
    candidate_id: str
    candidate_sha256: str
    result_artifact_sha256: str
    metrics_artifact_sha256: str
    run_manifest_sha256: str
    completed_at: datetime

    def __post_init__(self) -> None:
        for name in ("result_id", "candidate_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "candidate_sha256",
            "result_artifact_sha256",
            "metrics_artifact_sha256",
            "run_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "completed_at", _utc(self.completed_at, "completed_at"))


def _policy_sha256(max_candidates_per_batch: int) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "max_candidates_per_batch": max_candidates_per_batch,
            }
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RegistryEvent:
    """One hash-linked candidate-registration or result-link event."""

    sequence: int
    event_type: str
    occurred_at: datetime
    previous_event_sha256: str
    registry_policy_sha256: str
    candidate: ResearchCandidate | None = None
    result: ResearchResultLink | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("event sequence must be a positive integer")
        event_type = _text(self.event_type, "event_type").upper()
        if event_type not in {REGISTER_CANDIDATE, LINK_RESULT}:
            raise ValueError("unsupported registry event type")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at, "occurred_at"))
        for name in ("previous_event_sha256", "registry_policy_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if event_type == REGISTER_CANDIDATE:
            if not isinstance(self.candidate, ResearchCandidate) or self.result is not None:
                raise ValueError("candidate registration requires only a ResearchCandidate")
        elif not isinstance(self.result, ResearchResultLink) or self.candidate is not None:
            raise ValueError("result linkage requires only a ResearchResultLink")

    @property
    def event_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class ResearchTrialRegistry:
    """Immutable append-only event chain enforcing a bounded research batch."""

    max_candidates_per_batch: int = 3
    events: tuple[RegistryEvent, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_candidates_per_batch, bool)
            or not isinstance(self.max_candidates_per_batch, int)
            or self.max_candidates_per_batch < 1
        ):
            raise ValueError("max_candidates_per_batch must be a positive integer")
        try:
            events = tuple(self.events)
        except TypeError as exc:
            raise ValueError("events must be an iterable of RegistryEvent values") from exc
        if any(not isinstance(event, RegistryEvent) for event in events):
            raise ValueError("events must contain only RegistryEvent values")
        object.__setattr__(self, "events", events)
        self._validate_chain()

    @property
    def policy_sha256(self) -> str:
        return _policy_sha256(self.max_candidates_per_batch)

    @property
    def head_sha256(self) -> str:
        return self.events[-1].event_sha256 if self.events else ZERO_SHA256

    @property
    def candidates(self) -> tuple[ResearchCandidate, ...]:
        return tuple(
            event.candidate
            for event in self.events
            if event.event_type == REGISTER_CANDIDATE and event.candidate is not None
        )

    @property
    def results(self) -> tuple[ResearchResultLink, ...]:
        return tuple(
            event.result
            for event in self.events
            if event.event_type == LINK_RESULT and event.result is not None
        )

    def _validate_chain(self) -> None:
        previous_hash = ZERO_SHA256
        previous_time: datetime | None = None
        candidates: dict[str, tuple[ResearchCandidate, datetime]] = {}
        batch_counts: dict[str, int] = {}
        result_candidate_ids: set[str] = set()
        result_ids: set[str] = set()

        for expected_sequence, event in enumerate(self.events, start=1):
            if event.sequence != expected_sequence:
                raise ResearchRegistryError("registry event sequence is not contiguous")
            if event.previous_event_sha256 != previous_hash:
                raise ResearchRegistryError("registry event hash chain does not reconcile")
            if event.registry_policy_sha256 != self.policy_sha256:
                raise ResearchRegistryError("registry event policy fingerprint does not match")
            if previous_time is not None and event.occurred_at < previous_time:
                raise ResearchRegistryError("registry contains a backdated event")

            if event.event_type == REGISTER_CANDIDATE:
                candidate = event.candidate
                assert candidate is not None
                prior = candidates.get(candidate.candidate_id)
                if prior is not None:
                    if prior[0].candidate_sha256 == candidate.candidate_sha256:
                        raise ResearchRegistryError("candidate is already registered")
                    raise ResearchRegistryError("registered candidate ID was mutated")
                batch_counts[candidate.batch_id] = batch_counts.get(candidate.batch_id, 0) + 1
                if batch_counts[candidate.batch_id] > self.max_candidates_per_batch:
                    raise ResearchRegistryError("research batch exceeds its candidate budget")
                candidates[candidate.candidate_id] = (candidate, event.occurred_at)
            else:
                result = event.result
                assert result is not None
                registered = candidates.get(result.candidate_id)
                if registered is None:
                    raise ResearchRegistryError("result cannot precede candidate registration")
                candidate, registered_at = registered
                if result.candidate_sha256 != candidate.candidate_sha256:
                    raise ResearchRegistryError("result candidate fingerprint does not reconcile")
                if result.candidate_id in result_candidate_ids:
                    raise ResearchRegistryError("candidate result linkage is immutable")
                if result.result_id in result_ids:
                    raise ResearchRegistryError("result_id is already linked")
                if result.completed_at < registered_at:
                    raise ResearchRegistryError("result completed before candidate registration")
                if result.completed_at > event.occurred_at:
                    raise ResearchRegistryError("result completion postdates its linkage event")
                result_candidate_ids.add(result.candidate_id)
                result_ids.add(result.result_id)

            previous_hash = event.event_sha256
            previous_time = event.occurred_at

    def register_candidate(
        self, candidate: ResearchCandidate, *, occurred_at: Any
    ) -> ResearchTrialRegistry:
        """Append one pre-specified candidate and return the new registry head."""

        if not isinstance(candidate, ResearchCandidate):
            raise TypeError("candidate must be a ResearchCandidate")
        event = RegistryEvent(
            sequence=len(self.events) + 1,
            event_type=REGISTER_CANDIDATE,
            occurred_at=occurred_at,
            previous_event_sha256=self.head_sha256,
            registry_policy_sha256=self.policy_sha256,
            candidate=candidate,
        )
        return ResearchTrialRegistry(
            max_candidates_per_batch=self.max_candidates_per_batch,
            events=self.events + (event,),
        )

    def link_result(
        self,
        *,
        candidate_id: str,
        result_id: str,
        result_artifact_sha256: str,
        metrics_artifact_sha256: str,
        run_manifest_sha256: str,
        completed_at: Any,
        occurred_at: Any,
    ) -> ResearchTrialRegistry:
        """Append a single immutable result link for an existing candidate."""

        normalized_id = _text(candidate_id, "candidate_id")
        matches = [candidate for candidate in self.candidates if candidate.candidate_id == normalized_id]
        if not matches:
            raise ResearchRegistryError("result cannot precede candidate registration")
        candidate = matches[0]
        result = ResearchResultLink(
            result_id=result_id,
            candidate_id=normalized_id,
            candidate_sha256=candidate.candidate_sha256,
            result_artifact_sha256=result_artifact_sha256,
            metrics_artifact_sha256=metrics_artifact_sha256,
            run_manifest_sha256=run_manifest_sha256,
            completed_at=completed_at,
        )
        event = RegistryEvent(
            sequence=len(self.events) + 1,
            event_type=LINK_RESULT,
            occurred_at=occurred_at,
            previous_event_sha256=self.head_sha256,
            registry_policy_sha256=self.policy_sha256,
            result=result,
        )
        return ResearchTrialRegistry(
            max_candidates_per_batch=self.max_candidates_per_batch,
            events=self.events + (event,),
        )

    def is_extension_of(self, earlier: ResearchTrialRegistry) -> bool:
        """Return whether this registry preserves every event in ``earlier``."""

        if not isinstance(earlier, ResearchTrialRegistry):
            raise TypeError("earlier must be a ResearchTrialRegistry")
        return (
            self.max_candidates_per_batch == earlier.max_candidates_per_batch
            and len(self.events) >= len(earlier.events)
            and self.events[: len(earlier.events)] == earlier.events
        )


__all__ = [
    "FrozenPayload",
    "LINK_RESULT",
    "REGISTER_CANDIDATE",
    "RegistryEvent",
    "ResearchCandidate",
    "ResearchRegistryError",
    "ResearchResultLink",
    "ResearchTrialRegistry",
    "ResearchWindow",
    "SCHEMA_VERSION",
    "ZERO_SHA256",
    "content_sha256",
]
