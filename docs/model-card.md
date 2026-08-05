# Best Available Hardened Strategy — model card v3.2.1

**Technical strategy:** Diversified Global-Futures Time-Series Momentum Plus
Basis-Momentum Portfolio.

“Best Available Hardened Strategy” designates the latest selected repository
specification. It does not override the evidence, validation, or deployment
statuses below.

## Decision and intended use

**Deployment status: BLOCKED.** The repository is approved for reproducible
research review, serial-data integration, and a future prospective paper/shadow
program. It is not approved for live capital.

Intended users are quantitative research, independent model validation,
execution, risk, operations, compliance, market data, and the investment
committee. Prohibited uses include routing continuous symbols, using the
bundled paper broker with capital, treating reused history as a holdout, or
self-approving missing evidence.

## Frozen research specification

- Universe: 59 supported futures in six asset classes. YXT/YYT are excluded
  until rate-dependent, effective-dated valuation functions are available.
- Forecast: equal blend of 252-session sign trend and causal basis momentum,
  with trend-only fallback before basis history is available.
- Risk: causal market volatility, volatility-shock taper, flat market risk
  budgets, causal portfolio scalar, integer contracts, 7% annual risk target,
  25% no-trade region, and 5x gross-intent buffer.
- Schedule: completed month-end intent, later positive-volume-session close
  research fill, no P&L before fill.
- Launch: USD 1,000,000 and zero positions on 1990-01-01; earlier data warm
  causal estimates only.
- Independent live ceilings: 2% order participation, 6x gross notional, 35%
  effective-dated margin, and 15% drawdown.
- No optimized stop-loss or take-profit. The latched runtime halt cannot
  guarantee that a market gap will stop at 15%.

The current catalogue margin snapshot does not resize historical positions.
Its historical series is an informational proxy only.

## Historical evidence

All reported windows are `retrospective_reused_history`. Authoritative point
estimates live in `outputs/strategy_metrics.csv` and
`outputs/strategy_trade_metrics.csv`; the committee notebook reads them only
after verifying the v3.2.1 run manifest. They are not duplicated here because a
code, data, cost, or execution change requires regenerated artifacts. Episodes
overlap and share portfolio NAV, so they are not independent trials.

CAGR is futures excess-return CAGR under zero cash yield. Collateral income,
variation-margin funding, forced liquidation and broker financing are absent.
Against the user-supplied descriptive bands, historical Sharpe is
institutional while CAGR is conservative. The model is not labeled high alpha.

Twenty percent CAGR and approximately 2.0 Sharpe are aspirations, not an
optimization, promotion, or launch objective. The committee notebook derives
the arithmetic relationship from the generated volatility rather than using a
hard-coded claim. Bounded signal and portfolio challengers did not establish a
robust joint improvement. A 9.3% volatility-target sensitivity was rejected
because it bought exposure rather than alpha, failed to establish the joint
aspiration on full and later reused history, and left inadequate drawdown
headroom. Further 1990–2014 tuning is prohibited.

## Monte Carlo scope

The primary path diagnostic is a stationary bootstrap of daily net returns
using fixed 21/63/126-session blocks and 10/25-year horizons. Drawdown includes
initial capital and is measured at daily close. The monthly bootstrap is the
CAGR/Sharpe view; its month-end drawdown output is informational because it can
hide intramonth losses.

The episode permutation/IID bootstrap is secondary. Neither method recreates
signals, serial rolls, margin calls, funding, rejected orders, market impact,
broker failure, or unseen regimes. Capital-floor statistics are sensitivity
proxies, not calibrated live risk of ruin. Monte Carlo cannot create alpha.

## Principal limitations

1. The supplied data are vendor continuous/back-adjusted series, not
   exchange-listed old/new serial contracts with executable timestamps.
2. Back-adjusted roll-day P&L cannot be reconciled to the old contract held
   until the assumed close. The generated continuous-roll proxy is a warning,
   not a serial-contract capacity result.
3. Catalogue point values, ticks and margin are undated snapshots. Effective-
   dated valuation, venue calendars, FX, collateral, portfolio margin and
   position-limit history are required.
4. Research order *size* uses only lagged volume. The *fill* is additionally
   bounded by the executing session's realized depth, with the residual
   deferred, so realized participation is bounded by construction. That is a
   deliberate use of same-session information: it conditions the market's
   response, never the decision, and can only truncate a fill, never enlarge
   one. The next-close fill remains an approximation: an intraday POV algorithm
   would receive VWAP-like fills, while an auction order cannot claim
   deterministic close execution. Roll turnover is spread across sessions at
   the same limit and priced against the parent roll's participation, so
   spreading preserves the charge rather than discounting it.
5. Costs are transparent assumptions and stresses, not representative live
   calibration.
6. Source hashes prove byte identity, not point-in-time membership,
   survivorship freedom, vendor correctness, or live tradability.
7. Every return period has been inspected/reused. No untouched holdout or
   forward track record exists.

## Production controls

- `delta1_strategy.marketdata.contracts`: fresh UTC serial schema, explicit USD notional/margin,
  effective intervals, compatible later roll destination, FND/LTD and two-leg
  participation controls.
- `delta1_strategy.execution.costs`: order-weighted raw-fill-hash-bound calibration with unique order,
  session, recency, side, style, regime and delivery-cycle gates.
- `delta1_strategy.controls.production`: explicit notional pre-trade limits, ordered timestamped
  runtime health, full readiness schema, deployment-bundle fingerprints, and
  broker-identity/compliance-policy evidence-subject binding.
- `delta1_strategy.controls.evidence`: expiring artifact hashes, frozen fingerprints, independent
  review and revocation checks.
- `delta1_strategy.controls.treasury`: immutable settlement, cash, collateral,
  margin, funding and balanced-journal evidence validation. This is a
  fail-closed integration scaffold, not broker-supplied evidence or a funded
  live ledger.
- `delta1_strategy.execution.operations`: authenticated exact intents bound to
  certified broker-identity and compliance-policy digests; fresh exact-batch
  portfolio/compliance decisions; serial/broker-position binding; batch
  exposure controls; repeated broker-identity checks; durable locked outbox;
  ACK correlation, terminal-state replay, reconciliation, monitoring, DR and
  latched kill.

Software tests establish local behavior only. They do not establish external
data quality, network/broker behavior, elapsed paper evidence, or independent
approval. In particular, the repository does not supply the independent intent
signer, approved compliance-policy artifact/provider, certified production
adapter or broker-identity evidence required by those interfaces.

## Minimum path to another launch review

1. Freeze v3.2.1, dependencies, data and acceptance criteria.
2. Integrate licensed current serial data, venue calendars and effective-dated
   valuation/specification/margin/fee schedules.
3. Calibrate execution with representative quotes and fills.
4. Add collateral, variation margin, funding and forced-liquidation controls.
5. Evaluate genuinely post-freeze data without model changes.
6. Complete predeclared paper/shadow acceptance across rebalance and roll
   cycles.
7. Deploy and certify the selected broker adapter and identity, separately
   controlled signer, approved compliance-policy provider, atomic roll
   workflow, reconciliation, monitoring, incident response, kill and recovery
   drills.
8. Obtain independent model-risk, risk, operations, compliance, market-access,
   delivery and position-limit approvals.

Until every critical record is verified, `overall_readiness_status` remains
`BLOCKED`. See the [deployment runbook](runbooks/deployment.md) and
[evidence-registry guide](controls/evidence-registry.md).
