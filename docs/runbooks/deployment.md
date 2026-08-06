# Best Available Hardened Strategy — deployment runbook v3.2.1

**Technical strategy:** Diversified Global-Futures Time-Series Momentum Plus
Basis-Momentum Portfolio.

The best-available designation identifies the selected repository version; it
does not authorize paper or live deployment.

## Current decision

**NO-GO for live capital.** The only authorized next environment is a
prospective paper/shadow deployment after current serial data and
effective-dated specifications pass validation. The supplied continuous data
end in 2014 and cannot satisfy the external gates.

The research out-of-sample records added since the correctness audit do not
change this decision and must not be presented at a launch review as if they
did. The 2015-2016 twelve-root subset, the stitched 1995-2014 walk-forward and
the ETF sealed block are replays on vendor panels already in hand. None is
post-freeze, independently custodied or prospective, so none satisfies
`independent_holdout` or `forward_paper_trading`.

## System boundary

1. `delta1_strategy.research.strategy` is a deterministic research ledger; it never routes orders.
2. `delta1_strategy.marketdata.contracts` rejects continuous IDs, stale/naive timestamps, invalid USD
   risk values, crossed quotes, ineffective specifications and unsafe rolls.
3. `delta1_strategy.execution.costs` accepts raw timestamped quote/fill evidence and fails closed on
   unrepresentative order/session/cycle coverage.
4. `delta1_strategy.controls.production` owns pre-trade limits, runtime health and artifact-bound
   launch readiness.
5. `delta1_strategy.controls.treasury` validates externally supplied settlement,
   cash, collateral, margin, funding and journal evidence; it does not supply
   those records or operate a live account.
6. `delta1_strategy.execution.operations` owns authenticated intents, route-time recomputation,
   broker submission, journal/OMS, reconciliation, monitoring, DR and kill.
7. `delta1_strategy.research.drawdown` is diagnostic only and cannot clear a launch gate.
8. `delta1_strategy.research.attribution`, `.bounds`, `.benchmarks`,
   `.validation`, `.regimes`, `.allocation` and `delta1_strategy.marketdata.etfs`
   are study modules. They never route orders, never write into the canonical
   bundle, and cannot clear a launch gate. Their addition changed
   `implementation_fingerprint_sha256` — the hashed file count moved from 22 to
   29 — while `config_sha256` and `daily_fingerprint_sha256` are unchanged. A
   deployment bundle frozen before that change is stale and must be regenerated
   and re-approved, even though the engine's ledger is byte-identical.

## Before paper/shadow trading

- Freeze Git commit, v3.2.1 configuration, environment, source/output manifest,
  implementation fingerprint and acceptance criteria.
- Ingest licensed serial contracts with UTC timestamps, venue sessions,
  bid/ask, volume, open interest, expiry, FND/LTD, explicit USD notional/margin,
  effective-dated valuation, ticks, native margin and fees.
- Run `serial_snapshot_validation_report` every cycle and
  `build_roll_plans` daily. Carry residuals; never hold through the liquidation
  deadline.
- Deploy the intent signer separately from the router. Restrict and rotate its
  keys; the router accepts only exact unexpired certificates bound to the
  certified broker identity and approved compliance-policy digest.
- Deploy an independent portfolio/compliance policy service. It must return a
  fresh decision bound to the exact order batch, positions, serial snapshot,
  NAV and production-broker identity. The repository supplies the request and
  decision contracts, not the signer, policy artifact, approval or service.
- Certify the selected adapter build, broker/account/environment identity and
  bind that identity digest to the external evidence registry.
- Commission reconciliation, monitoring, durable journal storage/backups and
  the kill switch with separate operator/risk identities.
- Predeclare paper, drawdown, cost, roll and recovery acceptance criteria. The
  drawdown criteria must not be written as if the position-magnitude bounds
  enforce them. `max_risk_scalar` and `min_risk_scalar` bind on 0 of 300
  historical monthly decisions and `max_gross_notional_multiple` on 24 of 6,523
  sessions; measured drawdowns are broad accuracy failures at slightly *lower*
  volatility and a *smaller* book, not size failures. Tightening the scalar
  ceiling to 1.00 at matched risk raises simulated P(drawdown > 15%) from 4.90%
  to 5.95%. Treat those bounds as compliance ceilings and the latched halt plus
  the allocation decision as the drawdown response.
- Do not count the bundled `PaperBroker` as a qualifying forward record; it
  lacks price marks, margin, collateral, funding and liquidation.

## Route-time sequence

1. Verify the read-only deployment bundle and evidence registry against frozen
   implementation/config/source/output fingerprints.
2. Read the broker deployment identity and require the production
   broker/adapter-build/account/environment digest to match certified evidence.
3. Load a fresh validated serial snapshot and exact broker positions/open
   orders.
4. Verify ordered timestamps: market data ≤ reconciliation ≤ monitoring ≤ kill
   check ≤ health evaluation ≤ route time.
5. Match the broker-position SHA-256 and recompute current USD gross/margin.
6. Obtain a fresh exact-request portfolio/compliance decision and verify its
   approved policy digest.
7. Verify the exact intent signature, validity interval, deployment
   fingerprints, serial/position digests, broker-identity digest and
   compliance-policy digest.
8. Recompute order participation, delivery safety, and whole-batch projected
   gross/margin. Any missing input blocks risk.
9. Re-read the broker identity, then persist the exact order and broker/policy
   digests to the locked hash-chained outbox.
10. Re-read the broker identity immediately before submission; any change
    blocks the order and latches the kill switch.
11. Require an ACK for the same order ID at a valid timestamp and a uniquely
   bound broker order ID. Any mismatch latches the kill switch.
12. Journal fills/rejects/cancels, update positions from fills only, reconcile,
   back up and cold-replay.

Live rolls must be submitted through a selected-broker workflow that keeps the
close/open legs controlled as one group. A standalone opening leg is not an
acceptable roll implementation.

## Kill and recovery

Trigger the kill switch for stale data, failed signatures, broker disconnect,
position/cash/open-order break, invalid serial data, delivery risk, cost or
exposure breach, drawdown breach, ACK mismatch, journal/replay failure,
monitoring failure, or operator uncertainty.

The switch is latched across restart. While halted, new risk is blocked.
Emergency reduction requires fresh reconciled broker positions and serial
data, no open-order conflict, no zero crossing and valid broker ACK handling.
Cancel orders, preserve logs, reconcile, open an incident and determine whether
the broker accepted any unresolved outbox record.

Reset requires a named requester and different approver, documented root
cause, clean reconciliation, successful cold replay, healthy monitoring,
tested broker connectivity, and any required risk/compliance approval.

Before launch and at least quarterly, drill timeouts before/after acceptance,
wrong/duplicate/out-of-order ACKs, partial fill then restart, stale data,
position/cash breaks, one-leg roll failure, margin jump, journal corruption,
kill with open orders, key rotation and cold recovery.

## Launch evidence

All gates in
`delta1_strategy.controls.production.EXTERNAL_EVIDENCE_LABELS` are
conjunctive. Historical Sharpe, CAGR, profit factor, Monte Carlo, or local tests
cannot authorize capital. Launch requires current serial data, calibrated
execution, cash/margin controls, frozen-model/holdout/paper evidence, certified
broker and operations, a broker-certification subject matching the selected
deployment identity, a compliance-approval subject matching the signed policy
digest, successful DR/kill drills, and independent risk/compliance/model
approval.

Walk-forward, benchmark, family-wise, CSCV and ETF artifacts cannot authorize
capital either, and they are not evidence records. `outputs/holdout/`,
`outputs/validation/` and `outputs/etf/` are research output, not registered
evidence, and cannot be filed against `independent_holdout` or
`forward_paper_trading` — see
[`../controls/evidence-registry.md`](../controls/evidence-registry.md). If a
review presents them, the correct reading is the one their own artifacts carry:
the futures walk-forward is out of sample with respect to a selector only, and
the ETF sleeve loses to a passive 60/40 on the one contiguous forward block that
exists anywhere in this repository.
