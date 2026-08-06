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
>
> Out-of-sample records now exist inside the repository — a 2015–2016 futures
> subset, a 20.7-year stitched futures walk-forward, and a ten-year ETF record
> containing a contiguous five-year sealed block. Each is a replay on a vendor
> panel already in hand, not a prospective track record. None satisfies
> `independent_holdout` or `forward_paper_trading`, and none changes the status
> above.

See the [model card](docs/model-card.md) for intended use and limitations and
the [deployment runbook](docs/runbooks/deployment.md) for the operating sequence.
The [research methodology](docs/research-methodology.md) defines future alpha
promotion and multiple-testing rules; [architecture](docs/architecture.md)
documents the consolidated package layout. Three findings documents carry the
measurements this README summarizes:
[drawdown attribution](docs/drawdown-attribution-findings.md),
[benchmarks and validation](docs/benchmark-and-validation-findings.md), and
[the ETF regime-allocation sleeve](docs/etf-regime-allocation-findings.md).

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

Order **size** is capped from lagged median volume, so the completed execution
session never decides how much to trade. The **fill** is separately capped at
the participation limit of the volume that session actually traded, and any
residual carries to the next session — the same partial-fill-and-defer rule the
live roll control applies. The two bounds use different information sets on
purpose: sizing from realized volume would be an optimistic look-ahead, while
capping a fill can only ever truncate it. Realized participation is therefore
bounded by construction rather than merely reported, and
`capacity_deferred_contracts` records what depth refused.

Roll turnover is bounded the same way. A delivery change obliges the book to
transfer every contract held through it, two contracts of turnover each; that
obligation is tracked as `roll_backlog_contracts` and worked off in
capacity-sized slices. Slices are priced against the parent roll's
participation, so spreading preserves the total charge instead of earning a
square-root discount, and a backlog that cannot clear within one roll cycle
fails closed rather than silently transferring for free.

The current-catalogue static-margin series remains an informational proxy: it
is not point-in-time historical margin evidence. Live orders still require
current serial liquidity, a route-time participation check and effective-dated
USD margin.

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
characteristics with conservative CAGR**, not “high alpha” — and the two
sections below narrow even that. Against published rules replicated on the
identical panel the incumbent's advantage is risk control rather than signal;
the 4.64% spanning alpha is an optimistically biased reused-history estimate,
not an unbiased estimate of edge; no
family-wise procedure rejects on Sharpe across seventeen declared
configurations; and CSCV on the monthly paths the methodology permits returns a
probability of backtest overfitting of **0.42**, near the 0.5 signature of pure
overfitting. New alpha must be frozen before evaluation on data the pipeline
has never read. Useful methodological references include
[Basis-Momentum](https://doi.org/10.1111/jofi.12738),
[White’s Reality Check](https://doi.org/10.1111/1468-0262.00152), and
[the Stationary Bootstrap](https://doi.org/10.1080/01621459.1994.10476870).

## Drawdown levers are checked for activity before they are swept

`delta1_strategy.research.attribution` asks whether a proposed risk control is
an active constraint at all, and whether the loss it targets is the kind of loss
this strategy suffers, before a sweep spends compute measuring its effect.

For the position-magnitude family the answer is that it is not.
`max_risk_scalar` (2.00) and `min_risk_scalar` (0.25) never bind: over
1990–2014 the portfolio volatility-target multiplier stays inside
`[0.294, 1.846]`, so lowering either changes nothing. Realized volatility inside
drawdowns is 96.9% of the unconditional level while the book is 21.5% *smaller*
and the daily hit rate falls 8.9 percentage points — an accuracy failure, not a
size failure — and the losses are broad, with 59%–86% of the traded book losing
in every episode deeper than 5%. `signal_cap` is not a ceiling at all;
`basis_momentum` divides by it, so lowering it raises exposure.

[`docs/drawdown-attribution-findings.md`](docs/drawdown-attribution-findings.md)
records the measurements, the uniform-de-lever frontier that any magnitude lever
has to beat, and the two configurations that change the shape of the loss
distribution rather than its scale — trend-speed diversification, which pays for
the shape in return, and conditioning exposure on cross-market agreement, which
is the only one measured that improves every axis at once. Both were found by
looking at this history, so both are pre-registration candidates rather than
results. The conditioner's +0.054 Sharpe is not selection adjusted, and paired
resampling of that exact risk-matched path leaves its lower confidence bound
below zero.

`delta1_strategy.research.bounds` then sweeps the whole magnitude family at
*matched* realized volatility, so a variant's effect on the shape of the loss
distribution is separated from the de-levering any uniform rescale also buys.
Its most useful output is a refusal. Re-running the **unchanged** incumbent at
five different volatility budgets that all satisfy the match tolerance moves
drawdown-per-unit-volatility across a band of 0.0588 — wider than any delta the
sweep measures. Forty-nine of fifty shape rows therefore publish
`not_estimable` with that floor printed beside them: **no bound setting on any
swept axis changes the shape of the loss distribution by more than the
measurement can resolve.**

What does resolve is Sharpe, and it is not a drawdown result: `shock_floor`
0.75 → 0.25 gives +0.0225 with a 90% interval of [0.0034, 0.0416], an order of
magnitude below the +0.10 promotion gate. Forward breach risk moves the wrong
way — tightening `max_risk_scalar` to 1.00 at matched risk *raises*
P(drawdown > 15%) from 4.90% to 5.95%.

## The strategy is now measured against published rules and a declared family

Two things the repository previously asserted rather than demonstrated are now
executable, and both produce uncomfortable numbers.

`delta1_strategy.research.benchmarks` replicates Moskowitz-Ooi-Pedersen (2012),
Hurst-Ooi-Pedersen (2017), Baltas-Kosowski, a MACD/EWMAC crossover,
Barroso-Santa-Clara and Moreira-Muir volatility scaling, and long-only
references, on the identical panel through the same engine and all common
execution and cost assumptions. Rule-specific gross-cap departures are
declared, and capped MOP is reported separately.
The incumbent beats every one of them on Sharpe — but MOP TSMOM earns *more*
CAGR at 1.55x the volatility, so the incumbent's advantage is risk control
rather than signal. A joint spanning regression leaves **4.64% annualized alpha
at HAC t = 6.04**, with the only significant loading on MOP TSMOM at 0.509. That
figure is optimistically biased in the incumbent's favour: the benchmark rules
ran cold and unrefitted while the incumbent's parameters were chosen with this
panel visible.

`delta1_strategy.research.validation` implements the three estimators
[`research-methodology.md`](docs/research-methodology.md) names as required and
which previously existed only in prose — anchored expanding walk-forward,
White's Reality Check with Hansen's SPA, and CSCV/PBO with a mandatory
`NOT_ESTIMABLE` refusal path. Across seventeen declared configurations **no
procedure rejects on Sharpe at any block length** (White RC 0.642–0.660,
Hansen SPA consistent 0.277–0.321): the best in-sample member sits +0.088 above
the incumbent and is indistinguishable from it once the search is priced. On the
monthly paths the methodology permits, **PBO is 0.42**. The daily matrix gives
0.073 and is emitted only as a labelled secondary estimate; 0.42 is the number
to quote. `family_deflated_sharpe` returns `NOT_ESTIMABLE` for every member,
because a family declared today is a lower bound on this lineage's search rather
than its trial count.

[`docs/benchmark-and-validation-findings.md`](docs/benchmark-and-validation-findings.md)
carries both, including the selector path's 0.165 Sharpe shortfall. Only the
1995 fold selected a non-baseline variant, but continuous state carry prevents
assigning the full gap to that fold alone.

## What out-of-sample evidence exists, and what each piece is worth

The repository previously had one narrow forward record. It now has three, and
listing them together is the only way to keep any of them from being read as
more than it is. Each carries its limitation inline because the limitation is
the point.

| Record | Span | What it is out of sample with respect to | What it is not |
|---|---|---|---|
| 2015–2016 futures subset (`outputs/holdout/`) | 522 sessions, 12 of 59 roots | data the canonical source manifest proves the pipeline never read | trend-sleeve only, no basis sleeve, no equity/FX/energy/ags exposure; a subset consistency diagnostic, and the ledger refuses a second look |
| Stitched futures walk-forward (`outputs/validation/`) | 1995-01-02 → 2014-12-31, 5,218 sessions (20.7 252-session-equivalent years; 20.0 elapsed calendar years) | the **selector only** | the replayed specification was written with this window already read, so the segments are not out of sample with respect to specification choices |
| ETF regime-allocation sleeve (`outputs/etf/`) | 2009-01-02 → 2018-12-31, 2,516 sessions, of which 2014–2018 is a contiguous sealed block | the selector throughout, and the development replay's custody over the sealed block; later sealed folds fit on earlier sealed sessions | a survivors-only panel; the sealed block holds no full equity bear market, so full-bear conditional performance is not estimable, although the candidates' own defensive gates did operate; the universe rule and cost model were written by someone who had read the whole panel |

Two results have to be read next to each other. Letting the trend lookback be
chosen out of sample cost 0.165 Sharpe and deepened maximum drawdown from
−11.85% to −18.19% — **through the 15% drawdown policy**. The only
selected-variant difference is the 1995 fold; the splice-once replay carries
that fold's book and NAV state forward, so the full metric gap is not a
fold-local attribution. And the
ETF sleeve, the only contiguous five-year forward block available anywhere in
this repository, **loses**: −5.93% annualized against a monthly-rebalanced
60/40 at *t* = −3.48, and it also loses to a declared candidate without the
annual walk-forward selector. It is
reported because it is the answer, not despite it.

The honest reading is that the validation machinery works and returns
unflattering answers. It is not that a forward record now exists. See
[`docs/etf-regime-allocation-findings.md`](docs/etf-regime-allocation-findings.md)
for the audited session accounting, the survivorship disclosure and the four
vendor defects the panel carries.

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
  sizing; order size uses lagged volume only. The fill is separately bounded by
  the executing session's realized depth with the residual deferred, so
  realized participation is now bounded rather than merely observed.
- Bounded roll turnover, which previously bypassed every capacity cap and could
  demand one and a half times a thin session's entire volume. The transfer
  obligation is now tracked as a quantity and worked off in slices priced
  against the parent roll, so no uncosted transfer and no split discount.
- Corrected the optional equal-asset-class allocator so excluded or subset
  contracts cannot be reintroduced through catalogue indexing.
- Corrected the higher-moment term in `research.inference.deflated_sharpe_ratio`.
  It divided excess kurtosis by four; Bailey and Lopez de Prado specify
  `(gamma_4 - 1) / 4` on non-excess kurtosis, so the constant was dropped and
  with it the whole `+0.5 * SR^2` term a Gaussian series carries. The statistic
  was understating its own variance adjustment. Regression tests now pin the
  Gaussian identity `sqrt(1 + SR^2 / 2)`.
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
- No optimized stop-loss/take-profit. Risk is managed by diversification,
  causal volatility sizing and the separate latched runtime halt. The named
  position-magnitude bounds are not part of that list on the evidence:
  `max_risk_scalar` and `min_risk_scalar` bind on 0 of 300 monthly decisions,
  `max_gross_notional_multiple` on 24 of 6,523 sessions, and the measured
  drawdown failure mode is forecast accuracy rather than position size. A
  magnitude limit is a compliance ceiling here, not a drawdown control.

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
| `delta1_strategy.research` | Causal signals, integer research ledger, metrics, robustness diagnostics, published-rule replication and the family-wise/walk-forward/CSCV estimators | Serial execution and post-2014 futures data |
| `delta1_strategy.marketdata` | Fresh serial schema, USD risk values, expiry and roll controls, and the survivors-only ETF panel loader | Licensed point-in-time feed/specifications; a delisting-complete fund universe |
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
8. a genuinely post-freeze independent holdout — the 2015–2016 subset, the
   stitched walk-forward and the ETF sealed block are replays on panels already
   in hand and satisfy none of it;
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
pip install -e ".[notebook,dev]"
```

The licensed history cannot be redistributed, so start with generated data in
the same vendor layout. This runs the whole pipeline and, more importantly,
lets the causality tests execute:

```bash
python scripts/make_synthetic_data.py --output-dir examples/data/synthetic
delta1-strategy --data-dir examples/data/synthetic --output-dir /tmp/synthetic-outputs
DELTA1_DATA_DIR=examples/data/synthetic python -m unittest \
  tests.test_strategy.TestSuppliedDataIntegration -v
```

Those prices are driftless and independent by construction. Any performance
measured on them is an arithmetic check, never evidence — what they establish
is that the ledger reconciles and the engine has no look-ahead. See
[docs/causality.md](docs/causality.md) for what is being proved and why the
truncation-invariance test is the one that matters.

With the licensed data in place:

```bash
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

### The studies

Each study is a separate runner writing to its own subdirectory, and none of
them modifies the canonical bundle. Every runner that measures the incumbent on
the futures panel first reconciles its baseline replay against the frozen run
manifest's daily fingerprint, so a wiring mistake fails before a number is
published. The ETF runner reads a different panel and has no incumbent baseline
to reconcile.

```bash
DATA="Round1AllData/Quant Researcher/Delta1"

python scripts/run_lever_sweep.py            --data-dir "$DATA" --output-dir outputs/levers
python scripts/run_drawdown_attribution.py   --data-dir "$DATA" --output-dir outputs/attribution
python scripts/run_bounds_sweep.py           --data-dir "$DATA" --output-dir outputs/bounds
python scripts/run_benchmark_comparison.py   --data-dir "$DATA" --output-dir outputs/benchmarks
python scripts/run_validation_suite.py       --data-dir "$DATA" --output-dir outputs/validation
python scripts/run_etf_regime_allocation.py  --data-dir "$DATA" --output-dir outputs/etf
```

`run_validation_suite.py` also reads `outputs/levers`, so run the lever sweep
first if the CSCV refusal on that four-variant family is wanted. The 2015–2016
subset holdout is deliberately harder to run twice: it needs the continuation
and FX extracts and an explicit `--as-of` stamp, and its append-only ledger
refuses a second scoring of the same dataset.

```bash
python scripts/run_holdout_evaluation.py \
  --data-dir "$DATA" \
  --extension-dir "Round1AllData/Quant Researcher/FXFI/Futures Data" \
  --forex-dir "Round1AllData/Quant Researcher/FXFI/Forex Data" \
  --output-dir outputs/holdout \
  --as-of 2026-08-06T00:00:00Z
```

| Directory | Study | Findings document |
|---|---|---|
| `outputs/` | canonical v3.2.1 research bundle and manifests | this README |
| `outputs/levers/` | CAGR/Sharpe levers under a drawdown-risk constraint | [lever program](docs/lever-program-findings.md) |
| `outputs/attribution/` | bound activity, drawdown anatomy, de-lever frontier | [drawdown attribution](docs/drawdown-attribution-findings.md) |
| `outputs/bounds/` | position-magnitude sweep at matched realized volatility | [drawdown attribution](docs/drawdown-attribution-findings.md) |
| `outputs/benchmarks/` | published-rule replication, spanning regression, sign-flip null | [benchmarks and validation](docs/benchmark-and-validation-findings.md) |
| `outputs/validation/` | walk-forward, family-wise inference, CSCV/PBO | [benchmarks and validation](docs/benchmark-and-validation-findings.md) |
| `outputs/etf/` | ETF regime-allocation sleeve and its out-of-sample accounting | [ETF regime allocation](docs/etf-regime-allocation-findings.md) |
| `outputs/holdout/` | 2015–2016 twelve-root subset ledger | [lever program](docs/lever-program-findings.md) |

None of these artifacts alone authorizes trading. See [LICENSE](LICENSE).
