"""Artifact-backed production-readiness evidence verification.

The research backtest cannot prove operational readiness.  This module binds
each external-evidence gate to a file, immutable run fingerprints, a validity
window, and an independent approval.  Missing, malformed, stale, tampered, or
revoked evidence always produces a ``BLOCKED`` result.

The verification report deliberately uses the same seven columns as
``production.production_readiness_report`` so callers can concatenate the two
reports before calculating an overall readiness status.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import pandas as pd


PASS = "PASS"
BLOCKED = "BLOCKED"
SCHEMA_VERSION = 1
REPORT_COLUMNS = [
    "gate",
    "category",
    "critical",
    "status",
    "observed",
    "limit",
    "reason",
]
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class EvidenceRegistryError(ValueError):
    """Raised when an evidence manifest cannot be loaded structurally."""


@dataclass(frozen=True)
class EvidenceRecord:
    """One external-evidence assertion loaded from the registry manifest.

    Values remain in their JSON representation until verification.  Keeping
    parsing and policy checks in :func:`verify_evidence_registry` lets that
    function return a fail-closed report instead of leaking validation errors
    to a live caller.
    """

    gate: Any
    artifact_path: Any
    artifact_sha256: Any
    issued_at: Any
    expires_at: Any
    model_fingerprint_sha256: Any
    config_fingerprint_sha256: Any
    source_fingerprint_sha256: Any
    author: Any
    reviewer: Any
    approved: Any
    revoked: Any
    subject_sha256: Any = None


@dataclass(frozen=True)
class EvidenceRegistry:
    """A loaded manifest and the digest of the exact bytes that were parsed."""

    manifest_path: Path
    manifest_sha256: str
    schema_version: int
    records: tuple[EvidenceRecord, ...]


_RECORD_FIELDS = tuple(
    item.name for item in fields(EvidenceRecord) if item.name != "subject_sha256"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence_registry(path: str | os.PathLike[str]) -> EvidenceRegistry:
    """Load an evidence registry without treating it as trusted.

    Structural errors raise :class:`EvidenceRegistryError`.  Production code
    should normally call :func:`verify_evidence_registry`, which catches these
    errors and converts them into ``BLOCKED`` rows.
    """

    try:
        manifest_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvidenceRegistryError(f"evidence manifest is unavailable: {exc}") from exc
    if not manifest_path.is_file():
        raise EvidenceRegistryError("evidence manifest is not a regular file")

    try:
        manifest_bytes = manifest_path.read_bytes()
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceRegistryError(f"evidence manifest cannot be parsed: {exc}") from exc

    if not isinstance(document, dict):
        raise EvidenceRegistryError("evidence manifest must be a JSON object")
    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise EvidenceRegistryError(
            f"schema_version must be the integer {SCHEMA_VERSION}"
        )
    raw_records = document.get("evidence")
    if not isinstance(raw_records, list):
        raise EvidenceRegistryError("evidence must be a JSON array")

    records: list[EvidenceRecord] = []
    for position, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise EvidenceRegistryError(
                f"evidence record {position} must be a JSON object"
            )
        missing = [name for name in _RECORD_FIELDS if name not in raw_record]
        if missing:
            raise EvidenceRegistryError(
                f"evidence record {position} is missing: {', '.join(missing)}"
            )
        gate = raw_record["gate"]
        if not isinstance(gate, str) or not gate.strip():
            raise EvidenceRegistryError(
                f"evidence record {position} has an invalid gate"
            )
        records.append(
            EvidenceRecord(
                **{name: raw_record[name] for name in _RECORD_FIELDS},
                subject_sha256=raw_record.get("subject_sha256"),
            )
        )

    return EvidenceRegistry(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        schema_version=schema_version,
        records=tuple(records),
    )


def _parse_utc_timestamp(value: Any, label: str) -> tuple[datetime | None, str | None]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None, f"{label} is not a valid ISO-8601 timestamp"
    else:
        return None, f"{label} is missing or invalid"

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, f"{label} must include an explicit UTC offset"
    if parsed.utcoffset() != timedelta(0):
        return None, f"{label} must be expressed in UTC"
    return parsed.astimezone(UTC), None


def _normalise_sha256(value: Any, label: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        return None, f"{label} must be a 64-character SHA-256 digest"
    return value.lower(), None


def _normalise_identity(value: Any, label: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{label} is missing or invalid"
    return value.strip(), None


def _blocked_rows(gates: Sequence[str], reason: str) -> pd.DataFrame:
    targets = list(gates) or ["evidence_registry"]
    return pd.DataFrame(
        [
            {
                "gate": gate,
                "category": "external evidence",
                "critical": True,
                "status": BLOCKED,
                "observed": None,
                "limit": "valid artifact-backed approval",
                "reason": reason,
            }
            for gate in targets
        ],
        columns=REPORT_COLUMNS,
    )


def _required_gate_list(values: Any) -> tuple[list[str], str | None]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return [], "required_gates must be a non-empty sequence of gate names"
    gates: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            return [], "required_gates contains an invalid gate name"
        gates.append(value.strip())
    if not gates:
        return [], "required_gates must not be empty"
    if len(gates) != len(set(gates)):
        return [], "required_gates contains duplicate gate names"
    return gates, None


def _registry_integrity_error(registry: EvidenceRegistry) -> str | None:
    if (
        type(registry.schema_version) is not int
        or registry.schema_version != SCHEMA_VERSION
    ):
        return f"loaded registry schema_version must be {SCHEMA_VERSION}"
    declared_hash, error = _normalise_sha256(
        registry.manifest_sha256, "manifest_sha256"
    )
    if error:
        return error
    if not isinstance(registry.manifest_path, Path):
        return "loaded registry manifest_path is invalid"
    try:
        if not registry.manifest_path.is_file():
            return "loaded registry manifest is unavailable"
        observed_hash = _sha256_file(registry.manifest_path)
    except OSError:
        return "loaded registry manifest cannot be read"
    if observed_hash != declared_hash:
        return "loaded registry manifest SHA-256 does not match"
    if not isinstance(registry.records, tuple) or not all(
        isinstance(record, EvidenceRecord) for record in registry.records
    ):
        return "loaded registry records are malformed"
    return None


def _verify_record(
    record: EvidenceRecord,
    registry: EvidenceRegistry,
    *,
    expected_model: str,
    expected_config: str,
    expected_source: str,
    as_of: datetime,
) -> tuple[list[str], str | None]:
    reasons: list[str] = []

    declared_artifact_hash, error = _normalise_sha256(
        record.artifact_sha256, "artifact_sha256"
    )
    if error:
        reasons.append(error)

    artifact_path: Path | None = None
    if not isinstance(record.artifact_path, str) or not record.artifact_path.strip():
        reasons.append("artifact_path is missing or invalid")
    else:
        candidate = Path(record.artifact_path.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = registry.manifest_path.parent / candidate
        try:
            artifact_path = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            reasons.append("evidence artifact does not exist")
        if artifact_path is not None:
            if not artifact_path.is_file():
                reasons.append("evidence artifact is not a regular file")
            elif declared_artifact_hash is not None:
                try:
                    observed_hash = _sha256_file(artifact_path)
                except OSError:
                    reasons.append("evidence artifact cannot be read")
                else:
                    if observed_hash != declared_artifact_hash:
                        reasons.append("evidence artifact SHA-256 does not match")

    issued_at, error = _parse_utc_timestamp(record.issued_at, "issued_at")
    if error:
        reasons.append(error)
    expires_at, error = _parse_utc_timestamp(record.expires_at, "expires_at")
    if error:
        reasons.append(error)
    if issued_at is not None and expires_at is not None:
        if expires_at <= issued_at:
            reasons.append("expires_at must be later than issued_at")
        if issued_at > as_of:
            reasons.append("evidence has not yet been issued")
        if expires_at <= as_of:
            reasons.append("evidence has expired")

    fingerprints = (
        (
            record.model_fingerprint_sha256,
            expected_model,
            "model_fingerprint_sha256",
        ),
        (
            record.config_fingerprint_sha256,
            expected_config,
            "config_fingerprint_sha256",
        ),
        (
            record.source_fingerprint_sha256,
            expected_source,
            "source_fingerprint_sha256",
        ),
    )
    for supplied, expected, label in fingerprints:
        digest, error = _normalise_sha256(supplied, label)
        if error:
            reasons.append(error)
        elif digest != expected:
            reasons.append(f"{label} does not match the frozen run")

    author, error = _normalise_identity(record.author, "author")
    if error:
        reasons.append(error)
    reviewer, error = _normalise_identity(record.reviewer, "reviewer")
    if error:
        reasons.append(error)
    if (
        author is not None
        and reviewer is not None
        and author.casefold() == reviewer.casefold()
    ):
        reasons.append("reviewer must be independent from the author")

    if record.approved is not True:
        reasons.append("evidence is not explicitly approved")
    if record.revoked is not False:
        reasons.append(
            "evidence is revoked" if record.revoked is True else "revoked must be false"
        )

    return list(dict.fromkeys(reasons)), declared_artifact_hash


def verify_evidence_registry(
    registry_or_path: EvidenceRegistry | str | os.PathLike[str],
    *,
    required_gates: Sequence[str],
    expected_model_fingerprint: Any,
    expected_config_fingerprint: Any,
    expected_source_fingerprint: Any,
    expected_subject_sha256_by_gate: Mapping[str, Any] | None = None,
    as_of: Any = None,
) -> pd.DataFrame:
    """Return one fail-closed readiness row for every required evidence gate.

    ``as_of`` defaults to the current UTC time.  It may be a timezone-aware
    UTC :class:`datetime.datetime` or an ISO-8601 UTC string.  All three
    expected fingerprints must be SHA-256 digests for the frozen run.
    """

    gates, error = _required_gate_list(required_gates)
    if error:
        return _blocked_rows([], error)

    expected: list[str] = []
    for value, label in (
        (expected_model_fingerprint, "expected_model_fingerprint"),
        (expected_config_fingerprint, "expected_config_fingerprint"),
        (expected_source_fingerprint, "expected_source_fingerprint"),
    ):
        digest, digest_error = _normalise_sha256(value, label)
        if digest_error or digest is None:
            return _blocked_rows(
                gates,
                digest_error or f"{label} is invalid",
            )
        expected.append(digest)

    if as_of is None:
        as_of_time = datetime.now(UTC)
    else:
        as_of_time, error = _parse_utc_timestamp(as_of, "as_of")
        if error or as_of_time is None:
            return _blocked_rows(gates, error or "as_of is invalid")

    if isinstance(registry_or_path, EvidenceRegistry):
        registry = registry_or_path
    else:
        try:
            registry = load_evidence_registry(registry_or_path)
        except (EvidenceRegistryError, OSError, TypeError, ValueError) as exc:
            return _blocked_rows(gates, f"evidence registry is invalid: {exc}")

    integrity_error = _registry_integrity_error(registry)
    if integrity_error:
        return _blocked_rows(gates, f"evidence registry is invalid: {integrity_error}")

    subject_expectations = dict(expected_subject_sha256_by_gate or {})
    rows: list[dict[str, Any]] = []
    for gate in gates:
        matches = [record for record in registry.records if record.gate == gate]
        if len(matches) != 1:
            reason = (
                "required evidence record is missing"
                if not matches
                else "multiple evidence records exist for this gate"
            )
            rows.append(
                {
                    "gate": gate,
                    "category": "external evidence",
                    "critical": True,
                    "status": BLOCKED,
                    "observed": None,
                    "limit": "valid artifact-backed approval",
                    "reason": reason,
                }
            )
            continue

        record = matches[0]
        reasons, artifact_hash = _verify_record(
            record,
            registry,
            expected_model=expected[0],
            expected_config=expected[1],
            expected_source=expected[2],
            as_of=as_of_time,
        )
        if gate in subject_expectations:
            expected_subject, expected_error = _normalise_sha256(
                subject_expectations[gate], f"expected {gate} subject_sha256"
            )
            observed_subject, observed_error = _normalise_sha256(
                record.subject_sha256, f"{gate} subject_sha256"
            )
            if expected_error:
                reasons.append(expected_error)
            if observed_error:
                reasons.append(observed_error)
            if (
                expected_subject is not None
                and observed_subject is not None
                and observed_subject != expected_subject
            ):
                reasons.append(f"{gate} subject_sha256 does not match deployment")
        rows.append(
            {
                "gate": gate,
                "category": "external evidence",
                "critical": True,
                "status": BLOCKED if reasons else PASS,
                "observed": artifact_hash,
                "limit": "valid artifact-backed approval",
                "reason": "; ".join(reasons) if reasons else "gate passed",
            }
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def verified_evidence_flags(
    report: Any, required_gates: Sequence[str]
) -> dict[str, bool]:
    """Convert a verification report to strict booleans for legacy adapters.

    A gate is true only when exactly one critical row explicitly says ``PASS``.
    A malformed report, duplicate row, or missing gate returns ``False``.
    """

    gates, error = _required_gate_list(required_gates)
    if error:
        return {}
    flags = {gate: False for gate in gates}
    if not isinstance(report, pd.DataFrame):
        return flags
    required_columns = {"gate", "critical", "status"}
    if not required_columns.issubset(report.columns):
        return flags
    for gate in gates:
        rows = report.loc[report["gate"].eq(gate)]
        critical = rows.iloc[0]["critical"] if len(rows) == 1 else None
        flags[gate] = bool(
            len(rows) == 1
            and pd.api.types.is_bool(critical)
            and bool(critical)
            and rows.iloc[0]["status"] == PASS
        )
    return flags


__all__ = [
    "BLOCKED",
    "PASS",
    "EvidenceRecord",
    "EvidenceRegistry",
    "EvidenceRegistryError",
    "load_evidence_registry",
    "verified_evidence_flags",
    "verify_evidence_registry",
]
