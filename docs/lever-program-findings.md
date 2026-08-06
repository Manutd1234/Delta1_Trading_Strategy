# Lever Program Findings

Objective: raise CAGR and Sharpe without sacrificing drawdown.

Except for the explicitly labelled continuation and ETF diagnostics near the
end, results below are the reused 1990-2014 history. Nothing here promotes a
configuration. `docs/research-methodology.md` makes the Deflated Sharpe Ratio a
conjunctive promotion gate and the incumbent's trial count is unrecoverable, so
`research.inference.promotion_support` returns `NOT_ESTIMABLE` and refuses
promotion for every configuration measured on this data, whatever its point
estimates. These numbers price decisions; they do not make them.

Since this was written, the estimators `docs/research-methodology.md` names have
been implemented in `research.validation` and pointed back at this sweep. Both
answers belong here, at the top, because they qualify everything below:

- **Family-wise, on the four variants of this sweep.** White's Reality Check and
  Hansen's SPA do not reject on Sharpe at any block length — *p* between 0.54
  and 0.67. They reject on the annualized mean at *p* ≈ 1e-4, which is the
  risk-budget rescale showing up exactly where "CAGR is purchasable, Sharpe is
  not" predicts it will.
- **CSCV refuses.** Fed this same four-variant family, `cscv_pbo` returns
  `NOT_ESTIMABLE` for all four statistics: four configurations quantise the
  out-of-sample rank in steps of 1/5, which measures the grid rather than the
  overfitting, and the floor is ten. The sweep is too small to be its own
  overfitting diagnostic, and the code says so rather than printing a number.

Artifacts: `outputs/validation/validation_family_wise_lever_sweep.csv` and
`outputs/validation/validation_cscv_lever_sweep.csv`.

## The current sweep is a risk-budget frontier, and the result is a refusal

The executable sweep registers exactly four configurations: the frozen 7.0%
risk budget and otherwise unchanged 7.5%, 8.0% and 8.5% budgets. Earlier
cost-lever variants are not in the current registry or artifacts and therefore
cannot support a claim here.

"Without sacrificing drawdown" must mean forward path risk rather than the one
historical maximum. The 7.5% row demonstrates why: its realized maximum drawdown
is fractionally shallower than the baseline's, yet its ten-year bootstrap breach
probability is higher on both the excess and funded bases.

Each breach estimate below uses one common-random-number matrix of 2,000
stationary-bootstrap paths, a 63-session expected block and a 2,520-session
horizon. The artifact records those settings and the seed.

| Configuration | Excess CAGR | Funded CAGR | Sharpe | HAC | Hist. MDD | Excess P(DD>15%) | Funded P(DD>15%) | Peak order participation |
|---|---|---|---|---|---|---|---|---|
| Baseline, 7.0% budget | 13.19% | 17.01% | 1.590 | 1.489 | −11.85% | 4.05% | 0.90% | 2.00% |
| 7.5% risk budget | 14.32% | 18.17% | 1.601 | 1.492 | −11.84% | 6.70% | 1.90% | 2.00% |
| 8.0% risk budget | 15.31% | 19.20% | 1.597 | 1.482 | −13.35% | 10.80% | 4.40% | 2.00% |
| 8.5% risk budget | 16.28% | 20.20% | 1.602 | 1.493 | −13.09% | 13.90% | 6.75% | 2.00% |

The 8.5% budget buys the requested 20% funded CAGR, but it raises the
excess-basis 15% drawdown-breach probability from 4.05% to 13.90% and the
funded probability from 0.90% to 6.75%. Even the smallest increase fails the
no-sacrifice reading: the 7.5% row raises funded breach risk from 0.90% to 1.90% while its favorable
historical maximum drawdown differs from the baseline by less than one basis
point. No row satisfies the stated objective.

## Sharpe barely moves, and the movement is not certifiable

The paired block-63 comparisons use shared bootstrap index draws. Every lower
bound remains below zero, and every point estimate is below its own minimum
detectable effect.

| Variant | ΔSharpe | 95% lower bound | Detection floor | Correlation |
|---|---|---|---|---|
| 7.5% risk budget | +0.0114 | −0.0151 | 0.0264 | 0.9960 |
| 8.0% risk budget | +0.0073 | −0.0232 | 0.0304 | 0.9940 |
| 8.5% risk budget | +0.0128 | −0.0276 | 0.0401 | 0.9914 |

The family-wise procedures reach the same answer: none rejects on Sharpe at any
block length, while every annualized-mean row rejects because the experiment is
a volatility-budget rescale. The honest summary is **CAGR is purchasable, Sharpe
is not**. Financing and leverage change the return level; neither establishes a
new risk-adjusted edge.

Two later measurements reach the same conclusion from outside this sweep, and
neither was available when the paragraph above was written. Across seventeen
declared configurations of the incumbent, no family-wise procedure rejects on
Sharpe at any block length. Hansen SPA rejects the annualized mean at the
*p* ≈ 1e-4 resolution floor on the strength of one member raising the volatility
target from 7% to 8%, while White Reality Check does not. And
Barroso-Santa-Clara volatility scaling applied to the incumbent's own decision
frame delivers 19.27% CAGR at Sharpe 1.588 against 13.19% at 1.590 — a
time-varying volatility-management overlay that raises return and risk without
establishing a Sharpe improvement. Both are in
`docs/benchmark-and-validation-findings.md`.

## The funded basis, and the trap inside it

Recognizing collateral yield on the 1990-2014 Fed Funds path adds **3.82
percentage points of CAGR (13.19% → 17.01%) and improves both the historical
drawdown (−11.18% vs −11.85%) and the breach probability (0.90% vs 4.05%)**.
Carry adds drift without adding market exposure and added only about 0.005
percentage points of realized volatility here, so it improves the measured
drawdown distribution. That is the largest single step toward 20%, and it is
accounting rather than alpha.

Two disclosures are mandatory.

**The Sharpe trap.** Adding the financing rate to the numerator while still
labelling the hurdle zero yields a Sharpe of **2.004** — the exact figure named
as an aspiration — on a strategy whose risk-adjusted return has not changed.
The correct excess-of-financing Sharpe is **1.590, unchanged by construction**.
`collateral_reconciliation_report` checks this identity rather than trusting it,
and `tests/test_collateral.py` asserts no published Sharpe column can be
computed any other way. This check caught the error in its own implementation
during development.

**Regime dependence.** The 3.27% mean spans very different worlds, and a single
blended uplift implies a forward expectation the data does not support:

| Rate regime | Share of sessions | Average rate | Annualized contribution |
|---|---|---|---|
| Above 4% | 45.7% | 5.57% | 5.66% |
| 1% to 4% | 28.5% | 2.39% | 2.42% |
| Below 1% | 25.8% | 0.18% | 0.18% |

In a zero-rate regime this lever is worth nothing. The series ends at 0.115% in
December 2014. Report the contribution per regime, never blended.

The view models the yield leg only. Variation-margin financing and forced
liquidation are absent, so it is systematically optimistic; the artifacts carry
`collateral_yield_leg_only; variation_margin_funding_and_liquidation_absent`.

## What was dropped, and why

- **Drawdown overlay integration** — an archived diagnostic was dropped before
  the current registry because it did not establish forward risk reduction. No
  dedicated current artifact persists it, so no quantitative effect is claimed.
- **Correlation-aware risk budget** — a constant exposure rescale is largely
  reversed by the downstream EWMA portfolio target. The old numerical
  diagnostic is not a current artifact and is not used as evidence here.
- **Breadth expansion** — archived screening found missing valuation inputs,
  duplicate exposures and thin residual candidates. Those counts are not
  persisted by the current sweep, so they are not repeated as current results.
- **Capacity as a return lever** — the entire execution dimension is worth about
  0.11 Sharpe end to end. In the current post-fix friction artifact, halving the
  participation limit moves Sharpe by +0.0096. It remains in scope as
  compliance work only; that reused-history sensitivity is not a promotable
  return result.

## Defects this work surfaced — since resolved

All three were fixed in the capacity work that followed; the diagnoses are kept
because they are the interesting part.

1. **Roll turnover bypassed the capacity clip.** `rebalance_capacity` clipped
   only `desired_change`; `roll_adjusted_turnover` never passed through it. That
   was the source of the 1.500 peak order participation reading — a roll into a
   thin holiday session on 1990-12-31, not a sizing failure. Roll obligations
   are now tracked as a quantity and worked off in capacity-sized slices.
2. **The participation gate degraded under every return-improving lever**, from
   6.82% toward 7.95% against a 2% limit it already failed. It is now bounded by
   construction, so it can no longer degrade: peak rebalance participation is
   0.0199 and peak order participation 0.0200.
3. **The breach was a collapsed denominator, not a large order.** Eight of 8,378
   fills exceeded 2%; maximum ex-ante participation was 1.97%, so the causal
   sizing rule never breached. The worst case was a 6-contract order into a
   session that traded 88 contracts against a 3,918 trailing median. The fix
   caps the fill and defers the residual rather than redefining the metric.

The correction cost essentially nothing: CAGR +0.46 bp, Sharpe −0.0005, and
that is noise from eight order-days in 9,683 sessions, not a lever.

## The 2015-2016 subset, and where it now sits

`scripts/run_holdout_evaluation.py` scores the frozen specification once on the
2015-2016 continuation series, which the canonical source manifest proves the
pipeline has never read. Twelve of fifty-nine roots survive the data
constraints — the eleven government bonds plus gold — and the basis sleeve
cannot be reconstructed without unadjusted prices, so this is a **subset
consistency diagnostic, not a holdout for the flagship**.

Over 522 sessions: annualized return **+3.71%**, Sharpe **+0.56**, volatility
6.72%, maximum drawdown −8.02%. All three pre-registered criteria were met.

The Sharpe is well below the 1.59 reported in sample. On a twelve-root
trend-only subset over two years that is neither surprising nor evidence of
decay — the standard error at this sample size is far too wide to distinguish
the two — but it is the number, and it is recorded where a second look at the
same data will be refused.

When this section was written it was the repository's only out-of-sample
record. It no longer is, and the two additions make it narrower rather than
stronger:

- A stitched anchored walk-forward covering 1995-01-02 to 2014-12-31 — 5,218
  sessions, 20.7 252-session-equivalent years, twenty pairwise-disjoint folds — now exists in
  `outputs/validation/`. It is out of sample with respect to the **selector
  only**; the specification it replays was written with that window already
  read. Its result is unfavourable to selection: letting the trend lookback be
  chosen out of sample cost 0.165 Sharpe and deepened maximum drawdown from
  −11.85% to −18.19%, through the 15% drawdown policy. Only the 1995 fold
  selects a non-baseline variant; the splice-once replay carries its state
  forward, so the full gap is not attributed to that fold alone.
- A ten-year ETF record with a contiguous five-year sealed block now exists in
  `outputs/etf/`. It is on a different, survivors-only panel, and the sleeve it
  scores **loses** to a passive 60/40 by 5.93 percentage points a year at
  *t* = −3.48.

None of the three is a post-freeze, independently custodied, prospective record.
Counting them together does not produce one.

## Reproduce

```bash
python scripts/run_lever_sweep.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs/levers
```

The harness verifies the baseline reproduces the frozen run manifest's daily
fingerprint before any variant runs. This is not ceremony: omitting the
`delivery_months` frame alone disables every roll cost and flatters the result
by **+0.92 percentage points of CAGR and +0.101 Sharpe** — close enough to the
promotion gate to manufacture a passing lever out of a wiring mistake.

The family-wise and CSCV results quoted at the top of this document come from
the validation suite, which reads `outputs/levers` and must therefore run after
the sweep:

```bash
python scripts/run_validation_suite.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs/validation \
  --levers-dir outputs/levers
```

The 2015-2016 scoring is deliberately not repeatable. It needs the continuation
and FX extracts and an explicit `--as-of` stamp, and it appends to a hash-linked
ledger that refuses a second scoring of the same dataset:

```bash
python scripts/run_holdout_evaluation.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --extension-dir "Round1AllData/Quant Researcher/FXFI/Futures Data" \
  --forex-dir "Round1AllData/Quant Researcher/FXFI/Forex Data" \
  --output-dir outputs/holdout \
  --as-of 2026-08-06T00:00:00Z
```
