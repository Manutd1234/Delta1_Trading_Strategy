# Drawdown Attribution Findings

Question asked: can the drawdown be reduced by lowering the maximum and minimum
position bound?

Answer: **not by those two bounds, because neither is an active constraint.**
The reasoning below is worth more than the answer, because the same three
measurements decide whether *any* proposed risk control can work before a sweep
is run on it.

All figures are the reused 1990-2014 history, 6,523 in-window sessions. The core
attribution tables reproduce from `scripts/run_drawdown_attribution.py`; the
paired-resampling, quintile, split-sample and parameter-surface diagnostics in
section 5 were derived independently from the published daily and conditioner
artifacts and are not emitted as dedicated CSVs by that runner. The baseline
reconciles against the frozen run manifest's daily fingerprint before anything
is measured. Nothing here promotes a configuration;
`docs/research-methodology.md` prohibits selecting one on this data.

## 1. The two named bounds never bind

`min_risk_scalar` and `max_risk_scalar` clip the EWMA portfolio
volatility-target multiplier. Over twenty-five years that multiplier stays
inside `[0.294, 1.846]`.

| Bound | Side | Limit | Binding share | Nearest realized value | Headroom | Activity |
|---|---|---|---|---|---|---|
| `max_risk_scalar` | upper | 2.00 | 0.000% | 1.846 | +0.154 | **inert** |
| `min_risk_scalar` | lower | 0.25 | 0.000% | 0.294 | +0.044 | **inert** |
| `max_gross_notional_multiple` | upper | 5.00 | 0.368% | 5.112 | −0.112 | active |

Lowering `min_risk_scalar` is a no-op at any value at or below 0.294: the
strategy has never wanted to de-lever that far, so a deeper floor changes
nothing. Lowering `max_risk_scalar` is a no-op until it crosses 1.846, and even
then it touches almost nothing — the scalar is above 1.50 on 1.0% of sessions,
above 1.25 on 6.3%, and above 1.00 on 20.1%. To make the ceiling bite one has to
push it under 1.25, at which point it is no longer a tail control; it is a
smaller allocation.

A related trap, since it looks like a third member of the family:
**`signal_cap` is not a ceiling.** `basis_momentum` divides the clipped z-score
by `cap` before returning, so the sleeve is always in `[-1, 1]` and *lowering*
`signal_cap` raises average exposure by saturating sooner. It is the slope of
the sleeve map. The basis z-score already saturates on 31.8% of live
market-sessions, and the blended forecast reaches its ±1 limit on 21.8%.
`research.attribution.magnitude_bound_specs` deliberately excludes it from the
bound table with that reason attached.

## 2. The drawdowns are not volatility events, and the book was already smaller

| Sample | Sessions | Ann. return | Ann. volatility | Daily hit rate | Mean gross | Vol ratio | Exposure ratio |
|---|---|---|---|---|---|---|---|
| All sessions | 6,523 | +12.27% | 7.72% | 55.48% | 2.64× | 1.000 | 1.000 |
| In drawdown deeper than 5% | 958 | −13.31% | 7.48% | 46.56% | 2.07× | **0.969** | **0.785** |
| Outside | 5,565 | +16.67% | 7.73% | 57.02% | 2.74× | 1.001 | 1.037 |

Realized volatility inside drawdowns is 96.9% of the unconditional level, and
the book was 21.5% *smaller* while it was losing. What moved is the hit rate:
55.5% to 46.6%, a fall of 8.9 percentage points.

That is an accuracy failure, not a size failure. The volatility target had
already cut exposure — during the 2008-09 drawdown the risk scalar averaged
0.468 and reached 0.294 — and the strategy still lost, because the forecast was
wrong. A magnitude ceiling addresses a failure mode this strategy does not have.

## 3. The losses are broad, so no per-market ceiling reaches them

Seventeen episodes deeper than 5%. In every one, most of the traded book lost.

| Episode | Depth | Markets losing | Breadth | Share of loss from worst 3 | Asset classes losing |
|---|---|---|---|---|---|
| 2008-07 → 2009-06 | −11.85% | 35 / 59 | 59% | 17.9% | 5 / 6 |
| 1994-01 → 1994-04 | −9.90% | 23 / 34 | 68% | 26.6% | 5 / 6 |
| 2011-05 → 2012-11 | −9.60% | 37 / 59 | 63% | 25.1% | 5 / 6 |
| 2006-04 → 2006-12 | −8.12% | 37 / 49 | 76% | 20.3% | 5 / 6 |

Across all seventeen, breadth of loss runs 59%–86%. In the sixteen longer
episodes the three worst markets carry 15%–40% of the total loss; the
four-session August 1990 episode with a 27-market book is the outlier at 68.8%.
Four to six of six asset classes lose in every episode.

A per-market position ceiling can only help when a few markets produce the loss.
Here the loss is a correlation event: many markets reversing together. In every
episode except the four-session outlier, clipping the worst single market leaves
at least 84.5% of the loss untouched; even in that outlier it leaves 69.1%.

## 4. What pure de-levering already buys, and the invariant that exposes it

| Exposure multiplier | CAGR | Ann. vol | Max drawdown | Calmar | **Drawdown / volatility** | Sharpe |
|---|---|---|---|---|---|---|
| 1.0 | 13.19% | 7.72% | −11.85% | 1.113 | **1.535** | 1.5895 |
| 0.8 | 10.47% | 6.17% | −9.56% | 1.096 | **1.548** | 1.5895 |
| 0.6 | 7.80% | 4.63% | −7.23% | 1.079 | **1.561** | 1.5895 |
| 0.5 | 6.47% | 3.86% | −6.05% | 1.070 | **1.568** | 1.5895 |

Sharpe is invariant to the last decimal, drawdown per unit of volatility barely
moves, and Calmar gets slightly *worse* as exposure falls. This idealized
frontier scales the already-net return path, including its fixed-cost drag, so
it is conservative in the lever's favour; a real smaller book would not make
fixed costs shrink proportionally.

This is the line every magnitude lever has to beat. Any configuration that
reports a smaller drawdown while leaving drawdown-per-unit-of-volatility at 1.53
has not reduced risk. It has reduced the allocation, and the committee can do
that with one number and without a research project.

## 5. Two things that do change the shape — and what each costs

Both are measured at **matched volatility**, rescaled to the incumbent's 7.72%,
so the scale effect above is removed and only the shape difference remains.

| Configuration | Ann. vol | Sharpe | Max DD | DD / vol |
|---|---|---|---|---|---|
| Incumbent (252-session trend) | 7.72% | 1.590 | −11.85% | 1.535 |
| Equal blend of 63/126/252-session speeds | 7.72% | 1.394 | −10.59% | **1.371** |
| Exposure halved in the top correlation quintile | 7.72% | 1.644 | −10.56% | **1.369** |

**Trend-speed diversification** improves the shape — drawdown per unit of
volatility falls from 1.535 to 1.371 — and pays for it in return. The 63- and
126-session speeds earn Sharpe 1.04 and 1.05 standalone against 1.59 for the
252, and their pairwise return correlations with it are 0.55 and 0.69, so
blending dilutes the edge faster than it diversifies the risk. Sharpe falls.
This comparison is also biased toward the incumbent: its 252-session lookback
was selected on this same history and the two alternatives were not.

**Conditioning exposure on cross-market agreement** is the only lever measured
here that improves every displayed risk-adjusted and loss-shape axis
simultaneously. The conditioner is the mean
off-diagonal correlation of per-market P&L over a trailing 63-session window,
lagged so the value labelling session *t* uses data through *t−2*.
`tests/test_attribution.py` proves it truncation-invariant.

It is elevated where the failure mode is: 0.105 inside drawdowns deeper than 5%
against 0.064 unconditionally, a ratio of 1.64. Sorted into quintiles, the
forward 21-session return in the top quintile annualizes to +1.7% against
+13.6% to +16.9% in the other four, with a Spearman rank correlation of −0.155.

**It is not promotable, and the reasons are not formalities.** The threshold is
the in-sample 80th percentile of the same sample the result is measured on. The
Sharpe gain, +0.054, is not selection adjusted, and paired resampling of this
exact risk-matched path leaves its lower confidence bound below zero — the
measurement cannot separate it from zero. The split sample is directionally
consistent but far from stable: Spearman −0.076 over 1990-2002 against −0.211
over 2003-2014, with the top-quintile penalty roughly doubling in the second
half. And the conditioner is estimated from realized P&L, so it embeds the
strategy's own position sizes and is partly endogenous.

What counts in its favour is that the parameter surface is flat rather than a
single lucky cell. Across five estimation windows and nine threshold/depth
combinations — fourteen settings in all — every one improves drawdown per unit
of volatility (range 1.307 to 1.462 against the incumbent's 1.535) and every one
improves Sharpe (1.596 to 1.672 against 1.590).

| Window | 21 | 42 | 63 | 126 | 252 |
|---|---|---|---|---|---|
| DD / vol at q=0.80, ×0.5 | 1.360 | 1.372 | 1.369 | 1.369 | 1.376 |
| Sharpe | 1.664 | 1.657 | 1.644 | 1.672 | 1.596 |

Flatness rules out a knife-edge fit. It does not rule out the larger problem,
which is that the idea itself was found by looking at this data, and no amount
of internal robustness fixes that.

What it is: a hypothesis with an economic mechanism that matches the measured
failure mode, generated by looking at this data, and therefore requiring
pre-registration under `docs/research-methodology.md` and evaluation on history
that has not been seen. `research.attribution.conditional_exposure_diagnostic`
stamps that disclosure on every row it emits.

## Reproduce

```bash
python scripts/run_drawdown_attribution.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs/attribution
```
