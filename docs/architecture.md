# Repository architecture

```text
src/delta1_strategy/
  cli.py
  research/      causal strategy, diagnostics, Monte Carlo and trial registry
  marketdata/    serial-contract validation and roll planning
  controls/      readiness, runtime risk, evidence and treasury validation
  execution/     cost calibration, order routing and operational controls
docs/            model, methodology, architecture and runbooks
notebooks/       executed committee review
scripts/         reproducible notebook builder
examples/        fail-closed evidence examples
outputs/         canonical generated research bundle
tests/           unit, integration and artifact tests
```

Research never routes orders. Execution depends on validated market data and
controls, while the command-line layer orchestrates the canonical research
run. The run manifest hashes every Python file under `src/delta1_strategy` and
every canonical output. Moving or editing implementation files invalidates the
fingerprint and requires a regenerated bundle and new approvals.

The live execution boundary requires a production broker identity whose digest
matches independently certified evidence, a signed intent bound to that
identity and an approved policy digest, and a fresh external compliance
decision bound to the exact batch. Identity is rechecked before durable outbox
and broker submission. These interfaces are implemented; the external signer,
policy artifact/provider, certified adapter and broker evidence are not.

`delta1_strategy.research.registry` provides an immutable hash-linked trial
registry with a default maximum of three candidates per batch and one result
link per registered candidate. It makes the prospective governance rule
testable, but a local registry is not proof of independent preregistration,
custody, completeness of old trials, or out-of-sample performance.

`delta1_strategy.controls.treasury` defines content-addressed settlement,
cash, collateral, margin, funding and balanced-journal evidence records plus a
fail-closed validation boundary. It is an integration scaffold, not a broker
adapter, funded-account simulation, live treasury ledger, or external launch
record. The required operating evidence remains described in
[`controls/funding-and-margin.md`](controls/funding-and-margin.md).

`scripts/build_committee_notebook.py` writes
`notebooks/global_futures_trend_basis_committee_review.ipynb`. The notebook is
a read-only presentation layer: it imports the installed package version,
requires that version to match both generated manifests, verifies source and
output hashes, and only then reads the canonical CSV/JSON artifacts. Build it
after running `delta1-strategy`; execute it only from a clean kernel after the
integrity check passes. A manifest that names removed root-level modules or an
earlier engine version is stale and must be regenerated, not waived.

The ignored `Round1AllData/` directory remains an external local data mount.
It is not packaged or committed. Only files listed and hashed in
`outputs/source_manifest.csv` are consumed by the canonical run.
