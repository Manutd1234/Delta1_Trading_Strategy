# Pre-registered specification: Breadth TSMOM (v2)

This file freezes every specification choice for the v2 strategy before the
single official evaluation run. Anything not listed here is inherited
unchanged from the v1 pipeline (`delta1_cta.py`, `institutional_strategy.py`).
Deviations after the freeze must be appended to the log at the bottom.

## Honest provenance

The supplied data end on 2014-12-31 and the 2005-2014 window was already used
to report v1 results (Adaptive TSMOM, Sharpe 0.70), so **no untouched holdout
exists**. 2005-2014 is therefore labeled *evaluation window (second use)*
everywhere, never "out-of-sample". During v2 development, aggregate 2005-2014
statistics of early design variants were observed before this freeze; the
protections against self-deception are structural, not procedural: the primary
evidence window is 1990-2004, every design choice below is justified by
literature or pre-2005 evidence and recorded as such, all robustness exhibits
report both windows side by side, and the claim rule was fixed before the
official run. Post-2014 locked-holdout validation remains required before any
capital deployment.

## Universe (43 markets = 56 USD contracts − 13 named exclusions)

Rule: every USD-denominated `_CCB` contract in the supplied catalogue is a
member unless excluded for one of three reasons — duplicate exposure, broken
sizing mechanics, or a numeric liquidity rule (median daily dollar volume
below ≈ $100M over the sample). The exclusions: WBS, LSU, LRC, MWE, GD
(duplicates/baskets), DX (currency basket of members), VX (not delta-one),
ZQ (near-zero price volatility at the zero lower bound breaks 1/vol sizing),
LBS, ZR, DC, OJ, ZO (liquidity rule). Classes: 7 equity indices, 4 government
bonds, 8 FX, 6 energy, 5 metals, 13 agriculture & livestock.

Point-in-time entry gate: a market is tradeable on day *t* only if its
trailing 60-session median reported volume exceeds 1,000 contracts (reads only
past data; keeps 6N out until mid-2007 because its 2005-06 marks carry zero
reported volume). Exchange closures up to 10 business days are bridged by
carrying the last price (zero P&L), so recurring holiday closures (HTW at
Lunar New Year) hold positions instead of forcing round trips.

## Forecast

**Headline: the v1 forecast, unchanged** — `sign(P_t − P_{t−252})`. The single
intervention in v2 is breadth. Decision grounds: minimal-change principle;
fundamental law of active management (Moskowitz-Ooi-Pedersen 2012 trade 58
markets, Hurst-Ooi-Pedersen 2017 trade 67); pre-2005 evidence. The review
panel's a-priori preference for an equal-weight candidate ensemble was
considered and overridden on pre-2005 evidence (ensemble 1.62 vs 1.70 sign-12m,
1990-2004, pre-gate spec) and on the minimal-change principle; the ensemble is
reported prominently as the first alternative.

Candidate set for the training experiment (fixed): C1 sign 252; C2 sign
63/126/252; C3 strength 252; C4 strength 63/126/252; C5 strength
21/63/126/252. Strength signals use double normalization (Baz et al. 2015):
h-day move ÷ (EWM-60 daily vol × √h), then ÷ its own trailing 252-day standard
deviation, clipped at ±2 and rescaled to [−1, 1]. The primary blend excludes
the 21-day horizon (near information-free at a monthly rebalance, net of
costs); C5 retains it as the ablation. 12-month momentum uses the full window
(no skip-month: that convention is a cross-sectional equity artifact; MOP 2012
document short-horizon continuation in futures).

## Risk and execution (shared by every variant)

Flat equal risk across the 6 classes, equal risk within class, 1/dollar-vol
sizing (EWM span 60). Decision grounds for flat-6 over the hierarchical
4-super-class budget: pre-2005 evidence under the final spec (Sharpe 1.57 vs
1.52) and breadth proportionality (commodities are 24 of 43 markets); the
hierarchical budget is a stress-table row. Portfolio volatility target 10%,
63-day realized vol, leverage clipped [0.25, 2]. Vol-shock taper: fast 20 /
slow 120 EWM ratio, taper 1.35→2.00 to floor 0.75. No-trade buffer 25%.
Month-end decisions activate next business day. Costs: half-tick spread +
$2.50 per contract, one-way.

## Walk-forward training experiment

Annual model selection at each December year-end from 1990: choose the
candidate with the highest trailing 60-month net Sharpe (60m fixed as primary;
36m and 120m reported as sensitivity), apply it for the following calendar
year. Seeding: no exposure before the first selection takes effect; composite
metrics reported from 1992. The composite signal is re-run through the full
shared pipeline (sizing, leverage, buffer, costs) in one forward pass so
switch trades are charged.

## Evaluation and claim rule

Primary evidence window **1990-2004**; second-use window **2005-2014**;
benchmark: v1 Adaptive TSMOM (22 markets). Statistics: paired block bootstrap
of the monthly Sharpe difference (6-month blocks, 2,000 samples, seed 7); PSR
against the baseline's realized Sharpe; deflated Sharpe with trial count 26
(5 candidates + ensemble + walk-forward + 3 selection windows + 2 class
schemes + 3 buffer settings + v1's 12-cell robustness grid); CSCV PBO with 16
partitions on the candidate set.

**Claim rule (frozen):** the improvement is claimed only if (a) the 90%
paired-bootstrap interval of the Sharpe difference excludes zero in 1990-2004,
AND (b) the 2005-2014 point estimate is positive with the headline beating the
baseline in at least 6 of 10 calendar years. Otherwise the result is reported
as "not decisive" and the v1 strategy stands.

## Deviation log

- 2026-08-02, after the official run: fixed a unit bug in the deflated-Sharpe
  helper (the trial-Sharpe dispersion, declared in annualized units, was
  multiplied by √12 a second time, producing an absurd benchmark of ~1.7).
  The fix affects only this diagnostic; the strategy, the claim rule, and its
  inputs (paired bootstrap, win years) are untouched.
- 2026-08-02, after adversarial code review: (1) fixed a leverage boot-up
  defect — a structurally flat book (zero realized volatility) was mapped to
  the 2× leverage cap instead of neutral 1.0, and the trailing volatility
  window stayed diluted by flat days for one quarter after going live; the
  walk-forward composite therefore ran near 2× leverage in early 1991. The
  headline strategy is unaffected inside both evaluation windows (its history
  begins in 1979). (2) The walk-forward composite is now measured from 1992 in
  the comparison table, as this document always specified. (3) The DSR trial
  count was corrected from 26 to an honest 34, the CSCV-PBO median convention
  made conservative (ties count as overfit), and the PSR benchmark switched to
  the same monthly estimator as the PSR itself. Outputs were regenerated after
  these fixes; the claim-rule inputs (paired bootstrap, win years) and the
  headline results are unchanged.
