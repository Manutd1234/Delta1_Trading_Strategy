# Lever Program Findings

Objective: raise CAGR and Sharpe without sacrificing drawdown.

All results below are the reused 1990-2014 history. Nothing here promotes a
configuration. `docs/research-methodology.md` makes the Deflated Sharpe Ratio a
conjunctive promotion gate and the incumbent's trial count is unrecoverable, so
`research.inference.promotion_support` returns `NOT_ESTIMABLE` and refuses
promotion for every configuration measured on this data, whatever its point
estimates. These numbers price decisions; they do not make them.

## The constraint had to be defined before anything could be measured

"Without sacrificing drawdown" has two readings that give opposite answers.

| Configuration | CAGR | Sharpe | Historical MDD | P(MDD>15%) over 10y |
|---|---|---|---|---|
| Base (excess basis) | 13.19% | 1.590 | −11.83% | 3.85% |
| + drawdown overlay | 12.80% | 1.563 | −10.54% | 4.7% |
| + overlay, rescaled to match historical MDD | 14.50% | 1.563 | −11.83% | 9.6% |

The overlay lowers the historical maximum drawdown by 1.30 percentage points
and leaves forward breach probability unchanged. It clipped one 2008-09 path
rather than reducing risk, which its 0.985 average exposure multiplier already
implied. Rescaling on the strength of it doubles the breach probability while
the historical number stays put.

**Historical maximum drawdown is therefore not usable as the constraint.** Every
result below holds the bootstrap breach probability instead, measured on a
common index draw so configurations are compared on identical resamples.

The overlay finances a risk-neutral rescale of k = 0.981 — no headroom at all.
It is not part of any recommended configuration.

## What the levers deliver

Measured with paired common random numbers throughout (2,000 paths, block 63,
10-year horizon).

| Configuration | Excess CAGR | Funded CAGR | Sharpe | HAC | Hist. MDD | Excess P(DD>15%) | Funded P(DD>15%) | Peak participation |
|---|---|---|---|---|---|---|---|---|
| Baseline, 7.0% budget | 13.18% | 17.01% | 1.590 | 1.492 | −11.83% | 3.85% | 1.05% | 6.82% |
| Cost levers, 7.0% | 13.46% | 17.30% | 1.627 | 1.517 | −11.09% | 2.95% | 0.75% | 7.95% |
| Cost levers, 8.0% | 15.36% | 19.26% | 1.617 | 1.510 | −11.93% | 8.45% | 3.05% | 7.95% |
| Cost levers, 8.5% | 16.19% | 20.12% | 1.605 | 1.499 | −12.45% | 12.45% | 5.55% | 9.09% |
| Cost levers, 9.0% | 17.24% | 21.21% | 1.614 | 1.496 | −13.24% | 18.85% | 9.05% | 10.23% |

Two defensible answers follow, depending on which baseline risk level the
committee treats as the budget.

**Strict, funded against funded (0.75% vs 1.05%).** Keep the 7.0% risk budget
and take the cost levers only: **17.30% funded CAGR, Sharpe 1.627, historical
drawdown −11.09%, breach probability 0.75%.** Better than the incumbent on
every axis simultaneously. No risk budget is spent.

**Against the currently reported and accepted level (3.85%).** Raise the budget
to 8.0%: **19.26% funded CAGR at 3.05% funded breach probability**, still under
the risk the committee already accepts on today's reported basis.

**20% CAGR requires an 8.5% budget, which breaches both readings** — funded
breach probability 5.55% against a 1.05% strict and 3.85% accepted baseline.
The last 0.7 percentage points of CAGR cost roughly a doubling of drawdown risk.

## Sharpe barely moves, and the movement is not certifiable

This is the finding that most constrains what can be claimed.

| Variant | ΔSharpe | 95% lower bound | Detection floor | Correlation |
|---|---|---|---|---|
| Risk-scalar pass-through | +0.0221 | −0.0219 | 0.0431 | 0.9908 |
| Cost-aware buffer | −0.0031 | −0.0432 | 0.0401 | 0.9927 |
| Both | +0.0372 | −0.0121 | 0.0483 | 0.9885 |

Every lower bound sits below zero. The best point estimate (+0.037) is smaller
than its own minimum detectable effect (0.048), so it cannot be distinguished
from zero on 25 years of daily data even with paired resampling.

For scale, the unpaired bootstrap standard error of the headline Sharpe is
**0.201**, which makes the +0.10 promotion gate half of one standard deviation.
Pairing cuts the standard error to about 0.03 at these correlations — a
seven-fold gain, and still not enough. Comparing headline Sharpes without
pairing is not a weak test; it is not a test.

The honest summary: **CAGR is purchasable, Sharpe is not.** Rescaling is
Sharpe-neutral by construction, financing is Sharpe-neutral when measured
correctly, and the cost levers move it by less than the data can resolve.

## The funded basis, and the trap inside it

Recognizing collateral yield on the 1990-2014 Fed Funds path adds **3.82
percentage points of CAGR (13.19% → 17.01%) and improves both the historical
drawdown (−11.18% vs −11.83%) and the breach probability (1.05% vs 3.85%)**.
Carry adds drift without adding volatility, so it strictly improves the
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

- **Drawdown overlay integration** — no forward risk reduction; finances k = 0.981.
- **Correlation-aware risk budget** — mean pairwise P&L correlation is 0.0466
  (effective N ≈ 15.9), so the budget scales targets by about 0.59, and the EWMA
  portfolio scalar reverses roughly 99% of any constant rescale (median relative
  difference 0.7% at k = 0.6). The level effect is a no-op by construction.
- **Breadth expansion** — the HKD, KRW and SGD series required to value HSI,
  MHI, KOS and SSG are absent; the short-rate roots fail the same
  `price x multiplier` correctness test that excluded YXT and YYT; roughly 40%
  of the remainder duplicate held exposures. What survives is 8-12 thin
  softs and ags, which worsens the participation gate.
- **Capacity as a return lever** — the entire execution dimension is worth about
  0.11 Sharpe end to end. Halving the participation limit moves Sharpe by
  −0.0003. It remains in scope as compliance work only.

## Outstanding defects this work surfaced

1. **Roll turnover bypasses the capacity clip.** `rebalance_capacity` clips only
   `desired_change`; `roll_adjusted_turnover` is never passed through it. This
   is the actual source of the 1.500 peak order participation reading — a roll
   into a thin holiday session on 1990-12-31, not a sizing failure.
2. **The participation gate degrades under every return-improving lever**,
   from 6.82% to 7.95% and beyond, against a 2% limit it already fails. Any
   configuration recommended here inherits that breach.
3. **The participation breach is a collapsed denominator, not a large order.**
   Eight of 8,378 fills exceed 2%; maximum ex-ante participation is 1.97%, so
   the causal sizing rule never breaches. The worst case is a 6-contract order
   into a session that traded 88 contracts against a 3,918 trailing median.

## Reproduce

```bash
python scripts/run_lever_sweep.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs/levers
```

The harness verifies the baseline reproduces the frozen run manifest's daily
fingerprint before any variant runs. This is not ceremony: omitting the
`delivery_months` frame alone disables every roll cost and flatters the result
by **+0.83 percentage points of CAGR and +0.090 Sharpe** — close enough to the
promotion gate to manufacture a passing lever out of a wiring mistake.
