# Best Available Hardened Strategy

**Technical strategy:** Diversified Global-Futures Time-Series Momentum Plus
Basis-Momentum Portfolio.

One selected research specification, one reconciled research ledger, and a
separate fail-closed paper/live execution boundary.

“Best Available Hardened Strategy” is the repository designation for the
latest adopted specification after the correctness and controls audit. It is
not a claim of universal optimality, independent validation, or live approval.

> **Launch decision: BLOCKED — no live-capital authorization.**
>
> The repository now fails closed on missing or stale operational evidence,
> but the supplied history is continuous futures ending on 2014-12-31. Current
> serial contracts, representative quotes/fills, a prospective paper record, a
> certified broker deployment, and independent approvals are not present.

See the [model card](docs/model-card.md) for intended use and limitations and
the [deployment runbook](docs/runbooks/deployment.md) for the operating sequence.
The [research methodology](docs/research-methodology.md) defines future alpha
promotion and multiple-testing rules; [architecture](docs/architecture.md)
documents the consolidated package layout.

## Canonical result after the correctness audit

The v3.2.1 strategy combines a 252-session time-series trend forecast with
basis momentum across 59 supported futures. It targets 7% annualized portfolio
volatility, trades integer contracts monthly, and deducts modeled spread,
slippage, fees, impact, and continuous-roll costs.

Point estimates are deliberately not copied into this README because they
become stale whenever the engine, data, costs, or execution controls change.
The authoritative values are generated in
[`outputs/strategy_metrics.csv`](outputs/strategy_metrics.csv), with the exact
configuration and package version in
[`outputs/strategy_config.json`](outputs/strategy_config.json). The committee
notebook verifies both files against
[`outputs/run_manifest.json`](outputs/run_manifest.json) before displaying
CAGR, daily/monthly/HAC Sharpe, Sortino, drawdown, profit factors, expectancy,
cost drag, gross exposure, and participation.

Research rebalance size is capped from lagged median volume so it does not use
the completed execution session to decide order size. Realized participation
is then reported against that session's actual volume and can breach the
independent live gate when volume falls. Continuous-roll participation and the
current-catalogue static-margin series remain informational proxies: they are
not expiry-specific capacity or point-in-time historical margin evidence. Live
orders require current serial liquidity, a route-time participation check and
effective-dated USD margin.

Reported CAGR is a futures **excess-return CAGR**: cash collateral earns zero
and the research ledger omits collateral yield, variation-margin funding, and
forced liquidation. It is not a deployable funded-account CAGR.

## Return aspiration—not an optimization constraint

Twenty percent CAGR and approximately 2.0 Sharpe are aspirations, not search,
promotion, or launch constraints. Every supplied return period has already
been inspected, so tuning parameters against either number would create a
target-fitted result rather than independent evidence.

A bounded reused-history audit tested simple trend ensembles and portfolio
risk allocations at the same risk budget; none improved both durable return
and Sharpe. A separate 9.3% volatility-target sensitivity was also rejected.
It increased exposure rather than alpha, did not establish the joint
aspiration across the full and later reused windows, left insufficient
headroom beneath the drawdown policy, and produced materially higher simulated
15% drawdown-breach frequencies. It is not the adopted configuration and is
not promoted as a canonical result.

The committee notebook calculates, from the generated full-history volatility,
the Sharpe required for a 20% geometric return and the CAGR implied by a 2.0
Sharpe under a clearly labeled lognormal approximation. Leverage cannot create
the missing risk-adjusted edge.

The defensible conclusion is **institutional historical risk-adjusted
characteristics with conservative CAGR**, not “high alpha.” New alpha must be
frozen before evaluation on new serial-contract data and prospective paper
trading. Useful methodological references include
[Basis-Momentum](https://doi.org/10.1111/jofi.12738),
[White’s Reality Check](https://doi.org/10.1111/1468-0262.00152), and
[the Stationary Bootstrap](https://doi.org/10.1080/01621459.1994.10476870).

## What the hardened v3.2.1 strategy corrected

- Removed the current static-margin snapshot from 1990–2014 position sizing.
- Excluded YXT and YYT because a constant point value and `price × multiplier`
  cannot exactly value their yield-quoted P&L and notional. This is a
  correctness exclusion, not return-ranked universe selection.
- Kept the 5x research gross-intent buffer and independent 6x live ceiling.
- Corrected contribution profit factor/payoff classification when USD P&L and
  NAV-normalized contribution have different signs.
- Rejected infinite prices/volumes, negative volume, and premature terminal
  mid-month decisions.
- Added daily-close stationary-bootstrap drawdown paths that include initial
  capital. Month-end bootstrap drawdown is now explicitly informational.
- Required explicit USD notional and margin per serial contract; a roll
  destination must be compatible, later and outside the delivery buffer.
- Made production readiness require the complete unique gate schema; a
  fabricated subset cannot return `READY`.
- Bound runtime health to ordered fresh data/reconciliation/monitoring/kill
  timestamps and an exact broker-position SHA-256.
- Bound live alpha orders to authenticated exact intents and rechecked fresh
  serial data, participation, aggregate gross, margin, delivery safety and
  broker positions immediately before routing.
- Verified broker ACK type, order ID, timestamp and broker-order-ID binding;
  mismatches latch the kill switch.
- Added an inter-process journal lock and rejected events after terminal OMS
  states.
- Made execution-cost calibration order-weighted and raw-fill-hash-bound, with
  unique-order, session, recency, side, style, regime and delivery-cycle gates.
- Removed execution-session completed-volume look-ahead from close-order
  sizing; order capacity now uses lagged volume only and realized participation
  remains visible against actual volume.
- Corrected the optional equal-asset-class allocator so excluded or subset
  contracts cannot be reintroduced through catalogue indexing.
- Required fill-time USD point values in cost calibration and aligned the
  drawdown-overlay CAGR/MDD convention with the canonical ledger.
- Consolidated the implementation into the installable `delta1_strategy`
  package, with separate research, market-data, controls and execution layers.

These changes make the code safer and the research more correct. They do not
create the external facts required for deployment.

## Strategy specification

- 59 supported futures across equity indices, government bonds, FX, energy,
  metals, and agriculture/livestock.
- Equal blend of a 252-session sign-trend forecast and year-on-year change in
  trailing realized roll yield.
- Causal per-market volatility sizing, volatility-shock taper, flat
  pre-forecast market risk budgets, and a causal portfolio risk scalar.
- 7% portfolio risk budget, 25% no-trade region, monthly decisions, 5x research
  gross-intent buffer, and a 2% lagged-volume sizing fraction. The independent
  2% live gate is applied to current serial-contract liquidity at route time.
- Month-end signals queue for a later positive-volume session. The daily
  research approximation fills at that session’s close; completed same-session
  volume and exact close execution are not a live execution proof.
- Integer contracts and a self-financing USD NAV ledger. Back-adjusted series
  drive research P&L; unadjusted active prices approximate economic exposure.
- No optimized stop-loss/take-profit. Diversification, volatility sizing,
  exposure limits, and the separate latched runtime halt manage risk.

The supplied continuous data cannot reconcile old/new serial roll prices,
volumes, FND/LTD, contract vintages, or auction/VWAP fills. The generated roll
proxy keeps that limitation visible rather than declaring the roll executable.

## Monte Carlo scope

`delta1_strategy.research.diagnostics` generates the primary stationary
bootstrap from daily net returns using fixed 21/63/126-session blocks and
fixed 10/25-year horizons.
Drawdown is measured at daily close and includes the initial capital high-water
mark. The monthly bootstrap remains the CAGR/Sharpe uncertainty view.

`delta1_strategy.research.trade_sequence` also provides permutation and
with-replacement episode-order diagnostics. Episodes overlap and share
portfolio NAV, so their capital-floor statistic is a sensitivity proxy—not a
live probability of ruin. None of the simulations recreates serial contracts,
margin calls, rejected orders, gaps, funding, market impact, or unseen regimes.
Monte Carlo cannot manufacture a higher expected return or Sharpe.

## Production boundary

| Module | Responsibility | Remaining external dependency |
|---|---|---|
| `delta1_strategy.research` | Causal signals, integer research ledger, metrics and robustness diagnostics | Serial execution and post-2014 data |
| `delta1_strategy.marketdata` | Fresh serial schema, USD risk values, expiry and roll controls | Licensed point-in-time feed/specifications |
| `delta1_strategy.execution` | Fill calibration, authenticated intents, route-time risk and broker operations | Representative fills and certified broker services |
| `delta1_strategy.controls` | Runtime health, evidence, treasury validation and conjunctive readiness | Broker/clearing records, authenticated approvals and deployed control owners |

Live risk-increasing orders require all of the following at route time:

1. the complete artifact-backed readiness report is `READY`;
2. the runtime production broker identity matches the certified adapter,
   account and environment named by the evidence record;
3. a fresh external portfolio/compliance decision approves the exact order
   batch, positions, serial snapshot, NAV, broker identity and signed policy;
4. the exact intent certificate matches the frozen model/config/source,
   current serial/position snapshots, broker identity and compliance-policy
   digests;
5. data, reconciliation, monitoring and kill checks are ordered and fresh;
6. current and projected positions reconcile under exact USD notional/margin;
7. participation, gross, margin and delivery controls pass;
8. the kill switch is active and the broker is connected; and
9. the returned broker acknowledgement matches the persisted order.

Broker identity is checked at service initialization, again before the durable
outbox write, and again immediately before submission. The repository defines
these fail-closed interfaces, but it does not supply the independent intent
signer, approved compliance-policy artifact/provider, certified production
adapter or broker evidence.

Verified emergency reductions may bypass alpha-readiness and normal health
gates, but they still require fresh reconciled positions, valid serial data,
no crossing through zero, no conflicting open order, and broker ACK integrity.

The bundled `PaperBroker` is deterministic test plumbing. It has no price
marks, variation margin, collateral/funding ledger, liquidation engine, or
certified connection, so its activity is not a qualifying forward record.

## What still blocks deployment

Every critical gate is conjunctive. Required external evidence includes:

1. tradeable serial-contract data and roll liquidity;
2. timestamped current market data and venue calendars;
3. effective-dated contract valuation, specifications, margin and fees;
4. representative quote/fill calibration;
5. a point-in-time universe/security master;
6. cash, variation margin, collateral, funding and forced-liquidation controls;
7. a frozen model and authenticated change control;
8. a genuinely post-freeze independent holdout;
9. prospective paper/shadow acceptance;
10. a certified selected-broker adapter, deployment-identity evidence,
    separately controlled intent signer, approved compliance-policy provider
    and atomic roll workflow;
11. broker/clearing reconciliation, monitoring and incident response;
12. disaster-recovery and kill-switch drill evidence; and
13. compliance, market access, delivery, position-limit and independent model
    approvals.

No backtest or local unit test can manufacture these records. Until they are
provided and verified, the output remains `BLOCKED`.

## Reproduce

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"

delta1-strategy \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs

python -m unittest discover -s tests -v

python scripts/build_committee_notebook.py

jupyter nbconvert --to notebook --execute --inplace \
  notebooks/global_futures_trend_basis_committee_review.ipynb \
  --ExecutePreprocessor.timeout=900
```

The generated committee notebook refuses a stale bundle whose manifest engine
version or implementation hashes do not match the installed v3.2.1 package. The
[committee notebook](notebooks/global_futures_trend_basis_committee_review.ipynb)
reads only hashed canonical outputs; its executed review copy must contain
embedded committee charts. Important artifacts include
`outputs/strategy_metrics.csv`, `outputs/strategy_daily.csv`,
`outputs/strategy_market_daily.csv.gz`, `outputs/strategy_trade_metrics.csv`,
`outputs/strategy_friction_stress.csv`,
`outputs/strategy_daily_drawdown_monte_carlo.csv`,
`outputs/strategy_monte_carlo_summary.csv`,
`outputs/strategy_trade_sequence_monte_carlo.csv`,
`outputs/production_readiness.csv`, `outputs/strategy_config.json`, and
`outputs/run_manifest.json`.

None of these artifacts alone authorizes trading. See [LICENSE](LICENSE).
