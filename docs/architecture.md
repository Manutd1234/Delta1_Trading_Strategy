# Repository architecture

```text
src/delta1_strategy/
  cli.py
  research/      causal strategy, diagnostics, Monte Carlo, trial registry,
                 published-rule benchmarks, validation estimators, bound
                 sensitivity and the ETF regime/allocation sleeve
  marketdata/    serial-contract validation, roll planning and the ETF panel
  controls/      readiness, runtime risk, evidence and treasury validation
  execution/     cost calibration, order routing and operational controls
docs/            model, methodology, architecture, findings and runbooks
notebooks/       executed committee review
scripts/         reproducible notebook builder and research runners
examples/        fail-closed evidence examples
outputs/         canonical generated research bundle and per-study subdirectories
tests/           unit, integration and artifact tests
```

The `research` package separates the frozen strategy from the studies that
measure it. `strategy` is the engine; `levers`, `friction`, `diagnostics`,
`trade_sequence`, `drawdown`, `collateral`, `holdout`, `attribution`, `bounds`,
`benchmarks`, `validation`, `regimes` and `allocation` measure it under varied
assumptions; `inference` supplies the statistics they all share. A study may
read the engine; it may not change it, and it may not write into the canonical
bundle.

Seven modules were added after the correctness audit — four research studies of
the incumbent, plus the three that carry the ETF sleeve. Each answers a question
the package could previously only assert an answer to:

- `research.attribution` asks whether a proposed risk control is an active
  constraint at all and whether the loss it targets is the kind of loss the
  strategy actually suffers. It exists so that a bound sweep is interpretable
  rather than merely numeric. See
  [`drawdown-attribution-findings.md`](drawdown-attribution-findings.md).
- `research.bounds` sweeps the position-magnitude family at *matched* realized
  volatility, so a variant's effect on the shape of the loss distribution is
  separated from the de-levering any uniform rescale would also buy. Its most
  useful output is a refusal: re-running the unchanged incumbent at five
  volatility budgets the match tolerance accepts moves
  drawdown-per-unit-volatility across a band of 0.0588, wider than any matched
  delta the sweep measures, so 49 of 50 shape rows publish `not_estimable` with
  that floor printed beside them.
- `research.validation` implements the three estimators
  [`research-methodology.md`](research-methodology.md) names as required and
  which previously existed only in prose: anchored expanding walk-forward,
  White's Reality Check with Hansen's SPA, and CSCV/PBO with a mandatory
  `NOT_ESTIMABLE` refusal path. Its family assembler is what makes the
  family-wise promotion gate executable, and it validates that every member's
  path is synchronized on one common index before any statistic is computed.
  `family_deflated_sharpe` returns `NOT_ESTIMABLE` by design: an earlier
  revision returned a number that cleared the 95% gate for every configuration
  including the worst, and a statistic that passes everything is not a gate.
- `research.benchmarks` replicates published trend-following rules on the
  identical panel and pushes them through `strategy._simulate_execution` with
  the same engine and all common execution and cost assumptions, so costs,
  integer contracts, participation mechanics, roll turnover and FX are
  identical by construction. Rule-specific gross-notional-cap releases are
  declared in the artifact, and capped MOP is reported separately. A seam check asserts
  that the incumbent's own decision frame replayed through that path reproduces
  the canonical ledger exactly — maximum absolute daily net-return deviation
  0.0. `strategy.py` is unmodified.

`marketdata.etfs` and `research.regimes`/`research.allocation` carry the ETF
dynamic regime-allocation sleeve. It exists because the futures panel ends
2014-12-31 while the ETF panel ends 2018-12-31, so it is the only route in this
repository to a contiguous multi-year forward block. Its universe is
pre-declared by asset class, inception and liquidity, never by return, and every
artifact it emits carries the panel's survivorship disclosure: all 745 supplied
funds are alive on the final session, so no delisted fund is present. The sleeve
loses to a passive 60/40 on that block and the finding is published as such; see
[`etf-regime-allocation-findings.md`](etf-regime-allocation-findings.md).

Research never routes orders. Execution depends on validated market data and
controls, while the command-line layer orchestrates the canonical research
run. The run manifest hashes every Python file under `src/delta1_strategy` and
every canonical output. Moving or editing implementation files invalidates the
fingerprint and requires a regenerated bundle and new approvals.

`implementation_fingerprint_sha256` and `daily_fingerprint_sha256` are separate
on purpose, and the seven added modules are the case that shows why. Adding them
moved the hashed file count from 22 to 29 and therefore changed the
implementation fingerprint, while `config_sha256` and the daily fingerprint are
byte-identical to the run before the work
(`daily_fingerprint_sha256 = 8a4f63aaacfe32671c3aa8f120198fd9e256bb839d8ca2ff71ddec42466a4ac8`).
A changed implementation fingerprint means the deployment bundle is stale and
must be regenerated and re-approved; an unchanged daily fingerprint is the
evidence that no study touched the engine's ledger. Neither substitutes for the
other, and neither may be waived on the strength of the other.

## Where each study writes

Studies never write into the canonical bundle: each runner takes its own
`--output-dir` and never calls `save_outputs`. Every runner that measures the
incumbent on the futures panel first reconciles its own baseline replay against
the frozen manifest's `daily_fingerprint_sha256` through
`levers.assert_baseline_fingerprint`, so a wiring mistake fails before a number
is published. The ETF runner is the exception and cannot do this: it reads a
different panel and has no incumbent baseline to reconcile, which is one more
reason its output is a separate lineage rather than a continuation of this one.

| Runner | Output directory | Findings document |
|---|---|---|
| `scripts/run_lever_sweep.py` | `outputs/levers/` | [`lever-program-findings.md`](lever-program-findings.md) |
| `scripts/run_drawdown_attribution.py` | `outputs/attribution/` | [`drawdown-attribution-findings.md`](drawdown-attribution-findings.md) |
| `scripts/run_bounds_sweep.py` | `outputs/bounds/` | [`drawdown-attribution-findings.md`](drawdown-attribution-findings.md) |
| `scripts/run_benchmark_comparison.py` | `outputs/benchmarks/` | [`benchmark-and-validation-findings.md`](benchmark-and-validation-findings.md) |
| `scripts/run_validation_suite.py` | `outputs/validation/` | [`benchmark-and-validation-findings.md`](benchmark-and-validation-findings.md) |
| `scripts/run_etf_regime_allocation.py` | `outputs/etf/` | [`etf-regime-allocation-findings.md`](etf-regime-allocation-findings.md) |
| `scripts/run_holdout_evaluation.py` | `outputs/holdout/` | [`lever-program-findings.md`](lever-program-findings.md) |

`run_validation_suite.py` additionally reads `outputs/levers`, which is the one
ordering constraint between runners. `run_holdout_evaluation.py` appends to a
hash-linked ledger that refuses a second scoring of the same dataset, so it is
not idempotent by design.

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
