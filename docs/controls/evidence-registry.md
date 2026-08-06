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

Research artifacts are not evidence records either, however carefully they are
constructed. `outputs/holdout/`, `outputs/validation/` and `outputs/etf/`
contain out-of-sample material — a 2015-2016 twelve-root subset ledger, a
stitched 1995-2014 walk-forward, and a ten-year ETF record with a contiguous
five-year sealed block — and none of them may be registered against
`independent_holdout` or `forward_paper_trading`. Each is a replay on a vendor
panel already in the researcher's hands, produced by the same party that wrote
the specification, with no independent custodian, no post-freeze exposure and no
elapsed forward time. A hash-linked custody replay proves that a file was not
read; it does not prove that a reviewer was independent. Registering one would
be self-approval with extra steps, which is the failure mode
`verify_evidence_registry` exists to reject.

The same applies to the fingerprints these studies move. Adding the study
modules changed the run manifest's `implementation_fingerprint_sha256` — the
hashed file count moved from 22 to 29 — while `config_sha256` and the engine's
daily ledger fingerprint are byte-identical to the run before the work. Every
record above binds `model_fingerprint_sha256` to that implementation
fingerprint, so records issued against the earlier value no longer bind and must
be reissued. A changed fingerprint invalidates the approval regardless of
whether the numbers it approved moved; that is the intended behaviour, not an
inconvenience to be waived.
