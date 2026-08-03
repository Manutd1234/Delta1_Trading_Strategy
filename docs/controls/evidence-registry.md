# Deployment evidence registry

`examples/evidence/production_evidence.example.json` is intentionally empty and therefore
blocks deployment. Copy it outside the repository's generated-output area and
add exactly one independently approved record for every gate listed in
`delta1_strategy.controls.production.EXTERNAL_EVIDENCE_LABELS`.

Each record has this schema:

```json
{
  "gate": "serial_contract_data",
  "artifact_path": "artifacts/serial-data-acceptance.pdf",
  "artifact_sha256": "64 lowercase hexadecimal characters",
  "issued_at": "2026-08-03T00:00:00Z",
  "expires_at": "2027-08-03T00:00:00Z",
  "model_fingerprint_sha256": "frozen implementation fingerprint",
  "config_fingerprint_sha256": "frozen configuration fingerprint",
  "source_fingerprint_sha256": "frozen source-data fingerprint",
  "subject_sha256": "gate-specific deployment subject, where required",
  "author": "Control Owner",
  "reviewer": "Independent Reviewer",
  "approved": true,
  "revoked": false
}
```

The verifier reads the artifact, checks its SHA-256, requires UTC validity
dates, binds the approval to the exact frozen run, and rejects self-review,
duplicates, missing records, expired records, revocation, or any changed
manifest bytes. The `certified_broker_adapter` record must bind
`subject_sha256` to the selected production broker/adapter/account/environment
identity; the `compliance_approval` record must bind it to the approved policy
digest. A repository author must not create approval artifacts on behalf of
independent risk, compliance, operations, or broker reviewers.

The empty example is not a checklist that can be edited to `true`. Evidence is
accepted only through `delta1_strategy.controls.evidence.verify_evidence_registry`
and the verified registry inputs of
`delta1_strategy.controls.production.production_readiness_report`.

This launch-evidence registry is separate from
`delta1_strategy.research.registry.ResearchTrialRegistry`, which governs
prospective candidate registration and result links. A valid research trial
chain cannot substitute for any independent production-evidence record.
