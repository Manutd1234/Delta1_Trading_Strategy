# ETF Dynamic Regime Allocation — Findings

Two requirements motivated this sleeve:

- **explore momentum and trend-following CTA strategies, dynamic regime
  allocation for ETFs, or volatility strategies**; and
- **at least five years of out-of-sample forecast, with rolling walk-forward
  analysis recommended to conserve data.**

The incumbent futures strategy already satisfies the first clause of the first
requirement. It cannot satisfy the second on its own data: the Delta1 futures
panel ends 2014-12-31, and the only continuation available (FXFI, 27 roots) ends
2016-12-30 and covers twelve tradeable roots — two years, not five, and already
consumed by a one-shot holdout ledger that refuses a second look.

The ETF panel ends **2018-12-31**. It is the only dataset here that can carry a
contiguous multi-year forward block, and "dynamic regime allocation for ETFs" is
named in the first requirement. One sleeve therefore serves both.

**The headline result is negative and is reported as such.** The allocator
loses to a passive 60/40 on the realized out-of-sample path, and the unadjusted
paired sampling interval remains below zero. The artifacts do not identify a
unique structural cause or supply a selection-adjusted promotion decision.

## The out-of-sample record is real and it is auditable

| Claim | Window | Sessions | Calendar years | Double counted |
|---|---|---|---|---|
| In-sample training span | 2006-02-06 → 2008-12-31 | 732 | — | — |
| Rolling walk-forward | 2009-01-02 → 2013-12-31 | 1,258 | 5 | 0 |
| Sealed contiguous block | 2014-01-02 → 2018-12-31 | 1,258 | 5 | 0 |
| **Combined out of sample** | 2009-01-02 → 2018-12-31 | **2,516** | **10** | **0** |

Ten complete calendar years, of which five are a contiguous block. The fold
windows are pairwise disjoint and their union equals the stitched index exactly;
that is asserted in code rather than described. An independent recount from the
stitched daily file returns 2,516 and 1,258 rows respectively, with a unique
monotonic index.

Ten annual anchored-expanding folds, training growing 732 → 984 → 1,236 → …,
training strictly preceding test in every fold.

The annual selector continues refitting inside the 2014–2018 block: later
sealed folds use earlier sealed sessions as training data. “Sealed” proves
development-input custody, not a frozen five-year policy.

## What it earned

These are fully funded total-return paths whose residual “cash” is SHY and
therefore carries duration. The panel has no risk-free or financing series, so
every Sharpe and HAC Sharpe uses `rf=0`; it is not an excess-return statistic.

Rolling out-of-sample, 2009–2018:

| Path | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| **ETF regime allocation** | 3.32% | 5.14% | **0.661** | −7.47% | 0.444 |
| Faber trend, inverse volatility | 3.55% | 5.05% | 0.716 | −7.09% | 0.501 |
| Momentum, inverse volatility | 3.79% | 5.18% | 0.746 | −7.42% | 0.511 |
| Equal-weight panel | 6.35% | 9.71% | 0.684 | −17.18% | 0.370 |
| 60/40 buy and hold | 8.56% | 8.31% | **1.031** | −13.73% | 0.623 |
| 60/40 rebalanced monthly | 9.31% | 9.00% | **1.036** | −17.49% | 0.533 |

Sealed contiguous block, 2014–2018:

| Path | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| **ETF regime allocation** | 1.59% | 4.31% | **0.389** | −7.47% | 0.214 |
| Faber trend, inverse volatility | 2.33% | 4.06% | 0.588 | −7.09% | 0.328 |
| 60/40 rebalanced monthly | 6.45% | 7.31% | **0.892** | −10.56% | 0.611 |
| 60/40 buy and hold | 6.46% | 8.11% | 0.814 | −13.01% | 0.497 |

Paired against the references over the out-of-sample record, the unadjusted
sampling intervals remain below zero. They describe sampling uncertainty on
this realized panel; they are not selection adjusted and do not license
promotion.

| Comparison | Differential | *t* | 95% one-sided upper bound |
|---|---|---|---|
| minus 60/40 rebalanced monthly | **−5.93%** | −3.48 | −3.13% |
| minus 60/40 buy and hold | −5.17% | −3.45 | −2.71% |
| minus equal-weight panel | −3.24% | −1.92 | −0.46% |

The allocator also loses to a simpler declared candidate.
`faber_trend_inverse_volatility` retains a lagged Faber trend gate but omits the
annual walk-forward selector, and it beats the allocator on both windows.
Whatever annual candidate switching and selection add here, it is not
performance on these realized paths.

In-sample the sleeve earned Sharpe 0.657. The pre-sealed 2009–2013 rolling slice
earned 0.874, the combined 2009–2018 path earned 0.661, and the sealed block
earned 0.389. The pre-sealed window flattered it; the sealed block did not.

## What the sealed block leaves unresolved

**The sealed block contains no full equity bear market.** The Daniel-Moskowitz
bear state is unoccupied across all 1,258 sessions, and
`sealed_block_state_coverage` returns `NOT_ESTIMABLE` rather than a number. That
state is an external coverage diagnostic, not an allocator input. The declared
candidates' lagged Faber and time-series-momentum gates did operate materially:
the sealed stitched path held 33.39% mean cash and its risk weight ranged from
0.186 to 0.953. What was not observed is performance through a full equity bear,
while the realized path underperformed; the artifacts do not isolate a
counterfactual benefit or cost for each individual de-risking decision.

That makes the result narrower than it looks in both directions. The sleeve is
**not shown to be worthless** — its full-bear conditional performance is not
estimable. It is also **not rescued** by that observation: the candidate gates
were active and the observed underperformance remains the measured result.

The one full equity bear market in the panel, 2007–2009, straddles the first
fold boundary: 380 of its 504 sessions are training span and 124 fall inside
fold 0. It is published as three windows per path, with the whole episode
labelled as straddling rather than claimed wholly in-sample.

## What the sealed block does and does not prove

An earlier draft of this work claimed the block was "sealed before any fitting
decision." **That claim was wrong and has been removed from the artifacts.**

What is proven is narrower and is proven by execution: the development replay
read no sealed row, verified by a byte-identical custody replay over 1,990
sessions with maximum absolute difference exactly 0.0.

What is *not* true is that no fitting decision saw the block. The universe rule,
the candidate set, the cost model, the risk budget, the boundary schedule and
the purge/embargo lengths were all written by someone who had read the whole
panel. The declared liquidity screen is measured over 2006-02-03 → 2018-12-31,
a window that includes the sealed block. In addition, the annual selector
refits inside the block, so later sealed folds train on earlier sealed sessions.
A sealed block inside a panel the
researcher has already seen is weaker evidence than a genuine forward record,
and the distance between the two is not measurable from inside this repository.

## The panel's defect, and the design that accommodates it

**All 745 supplied ETF files end 2018-12-31 with positive volume on that exact
session.** The distribution of "last session with positive volume" is 745 on
that date and zero on every other. No closed or delisted fund is present in the
extract. It is a survivors-only panel.

Cross-sectional selection across such a panel is close to meaningless, so this
sleeve does not do it. The universe is eleven large, broad index-trackers —
SPY, IWM, EFA, EEM, IYR, SHY, IEF, TLT, LQD, GLD, DBC — chosen on asset-class
coverage, inception date and liquidity only, never on return, with the reason
recorded per fund in code. Using broad, liquid funds reduces exposure to the
defect without removing it. Choosing
those eleven remains a judgement made by someone who had seen the panel.

Three vendor facts the scouting corrected, each of which would have silently
corrupted results:

- The `Dividend` column is **identically 0.0** in all 745 files on every
  session. Total return comes from the adjusted `Close` alone.
- The three `Constituent_` columns are **identically 0**. ETFs are not S&P index
  constituents; there is no point-in-time membership here.
- `Volume` is **back-adjusted**, not a share count, so `Volume × Unadjusted
  Close` is wrong by the adjustment factor (median ratio 0.71 on SPY).
  `Turnover` is the correct liquidity screen.

Also: `first_quoted_date` is D/M/YY and *silently* mis-parses as M/D/YY on the
310 rows where day ≤ 12, while raising on the other 435. `Unadjusted Close`
retains splits — 223 funds, 395 events — and must never be differenced.

## What this establishes

The five-year out-of-sample requirement is **satisfiable on this data, and is
satisfied** — ten audited years by the rolling route, five contiguous by the
simplest route, with zero double counting.

**This particular sleeve is not validated by it.** It underperforms a passive
60/40 by roughly six percentage points a year with a *t* of −3.5, it
underperforms a declared candidate without the annual walk-forward selector,
and full-equity-bear conditional performance is not estimable in the test
window.

The honest use of this result is as a demonstration that the validation
machinery works and produces an unflattering answer when the strategy deserves
one — not as a candidate for capital.

## Reproduce

```bash
python scripts/run_etf_regime_allocation.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs/etf
```
