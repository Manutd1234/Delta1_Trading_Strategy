# Best Available Hardened Strategy — model card v3.2.1

**Technical strategy:** Diversified Global-Futures Time-Series Momentum Plus
Basis-Momentum Portfolio.

“Best Available Hardened Strategy” designates the latest selected repository
specification. It does not override the evidence, validation, or deployment
statuses below.

## Decision and intended use

**Deployment status: BLOCKED.** Nothing added since the correctness audit
changes that. The repository is approved for reproducible research review,
serial-data integration, and a future prospective paper/shadow program. It is
not approved for live capital.

Intended users are quantitative research, independent model validation,
execution, risk, operations, compliance, market data, and the investment
committee. The benchmark replication, family-wise inference, walk-forward,
bound-sensitivity and ETF studies exist to be read as constraints on what may be
claimed, not as support for a claim.

Prohibited uses include routing continuous symbols, using the bundled paper
broker with capital, treating reused history as a holdout, or self-approving
missing evidence. Also prohibited, and specific to the evidence added since the
audit:

- presenting the stitched 1995–2014 walk-forward, the 2015–2016 twelve-root
  subset, or the ETF sealed block as a prospective track record, or as
  satisfying `independent_holdout` or `forward_paper_trading`;
- quoting the daily-matrix PBO of 0.073 as the headline figure — the monthly
  paths the methodology permits give **0.42**, and that is the number;
- quoting the spanning alpha of 4.64% as an unbiased estimate of edge rather
  than an optimistically biased reused-history estimate;
- quoting the sign-permutation *p*-value as smaller than 0.000999, which is the
  resolution floor at 1,000 permutations; and
- citing the position-magnitude bounds as drawdown controls.

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

The position-magnitude bounds listed above are not risk controls on the
evidence. `max_risk_scalar` (2.00) and `min_risk_scalar` (0.25) bind on 0 of
6,523 sessions over 1990–2014; the realized multiplier stays inside
`[0.294, 1.846]`. `max_gross_notional_multiple` binds on 24 of 6,523 sessions.
They are retained as compliance ceilings, and lowering them is an allocation
decision rather than a drawdown decision. See
[`drawdown-attribution-findings.md`](drawdown-attribution-findings.md).

## Historical evidence

All reported windows are `retrospective_reused_history`. Authoritative point
estimates live in `outputs/strategy_metrics.csv` and
`outputs/strategy_trade_metrics.csv`, and are read only after the v3.2.1 run
manifest verifies. They are not duplicated here because a
code, data, cost, or execution change requires regenerated artifacts. Episodes
overlap and share portfolio NAV, so they are not independent trials.

CAGR is futures excess-return CAGR under zero cash yield. Collateral income,
variation-margin funding, forced liquidation and broker financing are absent.
Against the user-supplied descriptive bands, historical Sharpe is
institutional while CAGR is conservative. The model is not labeled high alpha.

Twenty percent CAGR and approximately 2.0 Sharpe are aspirations, not an
optimization, promotion, or launch objective. The arithmetic relationship is
derived from the generated volatility rather than from a hard-coded claim. Bounded signal and portfolio challengers did not establish a
robust joint improvement. A 9.3% volatility-target sensitivity was rejected
because it bought exposure rather than alpha, failed to establish the joint
aspiration on full and later reused history, and left inadequate drawdown
headroom. Further 1990–2014 tuning is prohibited.

### External reference points and multiple-testing results

The descriptive bands are a classification of a number, not a comparison of a
strategy. Published rules replicated on the identical panel through the same
engine and all common execution and cost assumptions supply the comparison;
rule-specific gross-cap departures are declared and capped MOP is reported
separately. Family-wise inference supplies the multiple-testing correction. Both are computed by
`delta1_strategy.research.benchmarks` and
`delta1_strategy.research.validation`; artifacts are in `outputs/benchmarks/`
and `outputs/validation/`, and
[`benchmark-and-validation-findings.md`](benchmark-and-validation-findings.md)
carries the detail.

| Result | Value | What it does not establish |
|---|---|---|
| Sharpe against published trend rules | incumbent highest of the replicated set | Moskowitz-Ooi-Pedersen TSMOM earns *more* CAGR at 1.55x the volatility, so the advantage is risk control, not signal |
| Spanning alpha, Newey-West 21 lags | 4.64%/yr, HAC *t* = 6.04, R² 0.735 | an optimistically biased reused-history estimate: the benchmark rules ran cold and unrefitted while the incumbent's parameters were chosen with this panel visible, and the size of that selection bias is unmeasurable here |
| White Reality Check / Hansen SPA on Sharpe, 17 declared configurations | no rejection at any block length (RC 0.642–0.660, SPA consistent 0.277–0.321) | Hansen SPA rejects on annualized mean at the resolution floor because a member raises the volatility target 7%→8%, lifting the annualized arithmetic mean 1.89 percentage points on a path correlated 0.994; White RC does not (0.087–0.106) — leverage, not skill |
| CSCV/PBO, monthly paths | **0.42** | near the 0.5 signature of pure overfitting; the daily-matrix 0.073 is a labelled secondary estimate and is not the headline |
| `family_deflated_sharpe` | `NOT_ESTIMABLE` for every member | a family declared today is a lower bound on this lineage's search, not its trial count |
| Sign-permutation null, 1,000 block flips | incumbent at the 100th percentile, *p* = 0.000999 | resolution floor at B = 1,000; a rejection of pure sign noise, not evidence of a persistent edge |

## Evaluation data and out-of-sample record

Three records exist and none of them is a forward track record. They are listed
together because reading any one alone overstates it.

| Record | Artifacts | Span | Result | Limitation carried inline |
|---|---|---|---|---|
| 2015–2016 futures subset | `outputs/holdout/` | 522 sessions, 12 of 59 roots | annualized return +3.71%, Sharpe +0.56, max DD −8.02% | trend sleeve only; the vendor supplies no unadjusted post-2014 series so basis momentum cannot be reconstructed; no equity, FX, energy or agricultural exposure; ~500 sessions cannot resolve a Sharpe difference of the size the gates require; the append-only ledger refuses a second look |
| Stitched futures walk-forward | `outputs/validation/` | 1995-01-02 → 2014-12-31, 5,218 sessions (20.7 252-session-equivalent years; 20.0 elapsed calendar years), 20 folds, pairwise disjoint | selector active: CAGR 11.59%, Sharpe 1.418, HAC 1.321, max DD −18.19%, efficiency 0.896. Frozen specification: CAGR 12.93%, Sharpe 1.583, HAC 1.476, max DD −11.85% | out of sample with respect to the **selector only**; the replayed specification was written with this window already read. Selection cost 0.165 Sharpe and pushed maximum drawdown through the 15% policy. Only 1995 selects a non-baseline variant; continuous state carry means the full gap is not a fold-local attribution |
| ETF regime-allocation sleeve | `outputs/etf/` | rolling 2009-01-02 → 2018-12-31, 2,516 sessions, 10 complete calendar years, of which 2014-01-02 → 2018-12-31 is a contiguous sealed block of 1,258 sessions; zero sessions double counted | rolling: CAGR 3.32%, Sharpe 0.661, HAC 0.744, max DD −7.47%. Sealed block: CAGR 1.59%, Sharpe 0.389, HAC 0.424, max DD −7.47% | **it loses.** −5.93%/yr against a monthly-rebalanced 60/40 (*t* = −3.48, unadjusted 95% one-sided upper bound −3.13%) and −5.17%/yr against buy-and-hold (*t* = −3.45); it also loses to a declared candidate without the annual walk-forward selector. See the survivorship and state-coverage disclosures below |

Two disclosures apply to the ETF record specifically and must travel with any
citation of it.

**Survivorship.** All 745 supplied ETF files end 2018-12-31 with positive volume
on that exact session. The distribution of "last session with positive volume"
is 745 on that date and zero on every other. No closed or delisted fund is
present in the extract. It is a survivors-only panel. The sleeve therefore performs no cross-sectional
selection: the universe is eleven large broad index trackers (SPY, IWM, EFA,
EEM, IYR, SHY, IEF, TLT, LQD, GLD, DBC) chosen on asset-class coverage,
inception date and liquidity only, never on return. That reduces exposure to the
defect without removing it, and choosing those eleven remains a judgement made
by someone who had seen the panel.

**Full-bear conditional performance is not estimable.** The sealed block
contains no full equity bear market. The Daniel-Moskowitz bear state, used as an
external coverage diagnostic rather than an allocator input, is unoccupied
across all 1,258 sessions and `sealed_block_state_coverage` returns
`NOT_ESTIMABLE`. The candidates' own lagged Faber and time-series-momentum gates
did operate and the sealed path held 33.39% mean cash. The sleeve is not shown
to be worthless; it is also not rescued from its observed underperformance by
the absence of a full-bear test.

The sealed block is **not** sealed before every fitting decision. What is proven
is narrower and is proven by execution: the development replay read no sealed
row, verified by a byte-identical custody replay over 1,990 sessions with
maximum absolute difference exactly 0.0. The universe rule, candidate set, cost
model, risk budget, boundary schedule and purge/embargo lengths were all written
by someone who had read the whole panel. The annual selector also refits inside
the sealed block, so later sealed folds train on earlier sealed sessions.

The vendor panel also carries four defects the loader corrects rather than
inherits: the `Dividend` column is identically 0.0 in all 745 files, the three
`Constituent_` columns are identically 0, `Volume` is back-adjusted so
`Volume × Unadjusted Close` is wrong by the adjustment factor, and
`first_quoted_date` is D/M/YY and silently mis-parses as M/D/YY on 310 of 745
rows.

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
   survivorship freedom, vendor correctness, or live tradability. The ETF panel
   is demonstrably survivors-only, and a hash cannot detect that.
7. Every 1990–2014 futures return period has been inspected and reused. **No
   prospective forward track record exists**, and no record here is post-freeze
   or independently custodied. The three out-of-sample records tabulated above
   are replays on vendor panels already in hand: two years on a twelve-root
   subset, twenty years out of sample with respect to a selector only, and ten
   years on a separate survivors-only ETF panel whose sealed five-year block
   contains no full equity bear, so bear-conditional performance is not
   estimable. None of them is the
   `independent_holdout` gate and none is the `forward_paper_trading` gate.
8. The strategy's edge is not established. No family-wise procedure rejects on
   Sharpe over seventeen declared configurations, CSCV/PBO on the permitted
   monthly paths is 0.42, and the 4.64% spanning alpha is an optimistically
   biased reused-history estimate with unmeasurable selection bias. What is defensible is that
   the incumbent achieves comparable return to published trend rules at
   materially lower risk on this panel.
9. The position-magnitude bounds do not manage drawdown. Neither named scalar
   bound binds; realized volatility inside drawdowns deeper than 5% is 96.9% of
   the unconditional level while the book is 21.5% smaller; the daily hit rate
   falls from 55.48% to 46.56%; and 59%–86% of the traded book loses in every
   episode. The failure mode is forecast accuracy across many correlated
   markets, which no size ceiling reaches. Uniform de-levering leaves Sharpe
   invariant at 1.5895 and makes Calmar slightly worse.
10. Capacity on this cost model is small and is bounded by roll completion,
    not price impact. Replaying the frozen configuration at larger initial
    capital (`outputs/validation/validation_capacity.csv`), every level from
    $5M upward aborts on the 21-session roll-completion guard in thin markets
    (SJB first, then RS and GF); the working capacity sits between $2M and
    $5M. The headline Sharpe describes a book that cannot be scaled past that
    bound under these participation limits, and no impact-cost erosion figure
    is estimable because the guard binds first.

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

## Research estimators the methodology names, now in code

Gates that existed only as prose are executable, which means a future review can
run them rather than assert them. Existence is not a result: three of the four
returned refusals or unfavourable answers on this data.

- `delta1_strategy.research.validation`: `anchored_walk_forward`,
  `reality_check`/`hansen_spa`/`family_wise_report`, `cscv_pbo` with a mandatory
  `NOT_ESTIMABLE` refusal path, and `family_deflated_sharpe`. `assemble_family`
  validates that every member's path is synchronized on one common index before
  any statistic is computed.
- `delta1_strategy.research.benchmarks`: published-rule replication driven
  through `strategy._simulate_execution` with the incumbent's own
  `StrategyConfig`, so costs, integer contracts, participation caps, roll
  turnover and FX are identical by construction. A seam check reproduces the
  canonical ledger from the incumbent's own decision frame with maximum absolute
  daily deviation exactly 0.0. `strategy.py` is unmodified.
- `delta1_strategy.research.attribution` and `delta1_strategy.research.bounds`:
  bound-activity measurement before a sweep is run, then a magnitude sweep at
  matched realized volatility with a published resolution floor.
- `delta1_strategy.marketdata.etfs`, `delta1_strategy.research.regimes` and
  `delta1_strategy.research.allocation`: the ETF sleeve, its pre-declared
  universe and its custody replay.

## Minimum path to another launch review

1. Freeze v3.2.1, dependencies, data and acceptance criteria.
2. Integrate licensed current serial data, venue calendars and effective-dated
   valuation/specification/margin/fee schedules.
3. Calibrate execution with representative quotes and fills.
4. Add collateral, variation margin, funding and forced-liquidation controls.
5. Evaluate genuinely post-freeze data without model changes. The estimators
   for that evaluation exist — `research.validation.anchored_walk_forward` for
   the replay and `family_wise_report` for the multiplicity correction — so what
   is missing is the data and the custody, not the method.
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
[evidence-registry guide](controls/evidence-registry.md). The measurements
summarized above are recorded in
[drawdown attribution](drawdown-attribution-findings.md),
[benchmarks and validation](benchmark-and-validation-findings.md),
[the ETF regime-allocation sleeve](etf-regime-allocation-findings.md) and
[the lever program](lever-program-findings.md).
