# Archived research history

This is the immutable decision log for the experiments that produced the
current strategy. References to retired module names are historical; the
deployable implementation now lives entirely in `strategy.py`. Failed and
superseded strategies remain documented here but are intentionally absent from
the runtime repository.

Window-status language is also historical. Phrases such as "untouched" or
"never opened" describe the protocol at that point in the sequence; subsequent
rounds inspected every window. As of v2.6, no untouched holdout remains.

## Pre-registered specification: Breadth TSMOM (v2)

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

## Addendum (v2.1): institutional risk-engineering fixes

After v2 was evaluated, four externally proposed "institutional blueprint"
fixes were assessed. Full honesty about provenance: these were tested after
the v2 freeze, so both windows had been seen; decisions below therefore rest
on the primary window plus a-priori rules only, both windows are published
for every variant, and adopted changes were fixed before the single official
v2.1 run.

**Adopted.**
1. *Volatility targeting estimator*: the 63-day rolling portfolio volatility
   is replaced by a RiskMetrics EWMA (λ = 0.94). Same 10% target, same
   [0.25, 2] clip, same monthly activation — only the estimator reacts
   faster. Primary window improves (1.59 vs 1.57), the second-use window
   improves (0.99 vs 0.89), and the choice is the textbook standard. Daily
   position rescaling was rejected: it would break the month-boundary
   execution architecture and multiply turnover.

**Tested, reported in full, not adopted as the headline.**
1. *Carry sleeve*: a roll-yield carry signal estimated from roll gaps — the
   difference between unadjusted and back-adjusted continuous price changes,
   summed over a trailing year, volatility-scaled and normalized exactly like
   the trend signal (Koijen-Moskowitz-Pedersen 2018) — blended 50/50 with
   trend (the no-information prior for two return sources). Point estimates
   improve (primary Sharpe 1.62 vs 1.59; second-use max drawdown −11.9% vs
   −20.8%), but the paired-bootstrap Sharpe increment of the blend over trend
   alone is not decisive in either window (primary +0.04, 90% CI
   [−0.24, +0.26], P(>0) = 0.52), and this document's standing rule —
   default to the simpler variant when the interval includes zero — applies.
   The blend and 25%/75% weights are published rows in the comparison and
   stress tables; the carry code is tested and ships in the module.
   Collateral yield on margin cash is a deployment note, not a backtest
   input: the dataset has no risk-free series and Sharpe is rf=0.

**Already present** (no change): the liquidity screen, point-in-time volume
gate, 25% no-trade region, monthly horizon, and cost accounting implement the
"minimize frictional costs" prescription; realized cost drag is ~1.5% of
gross profits against the blueprint's 15% ceiling.

**Rejected, with evidence.**
1. *Absorption-ratio (PCA) de-leveraging*: implemented and published as a
   stress row. The specified absolute 70% trigger never fires on this
   universe (the top-2 principal components explain at most 67% of variance,
   median 31%); a point-in-time percentile version subtracts in both windows.
   The existing volatility-shock taper already de-risks in stress regimes.
2. *ATR stop losses*: incompatible with the month-end decision architecture,
   and the trend signal is itself the exit mechanism; intramonth stops would
   add path dependency and turnover without a tested benefit.
3. *Beta-neutralization against equity futures*: appropriate for a
   market-neutral equity book, not for a long-short CTA whose returns are the
   directional trends such a hedge would remove; time-varying signs already
   keep average equity beta near zero.

## Addendum (v2.2): multi-sleeve alpha search — rule frozen before results

The strategy's Sharpe is capped by breadth, not by trend-signal quality:
43 markets collapse to 12.3 independent bets (average pairwise P&L
correlation 0.059), and Grinold's law with the measured trend IC of 0.082
puts this architecture's realistic ceiling near 1.2-1.5. The only lever with
real headroom is **more uncorrelated return sources**, so a pre-declared set
of literature-backed candidate sleeves is tested, each through the identical
universe, risk stack, and cost model.

**Candidates (fixed before testing):** long-term reversal/value, calendar
seasonality, realized skewness, cross-sectional momentum, hedging pressure
from open interest, basis momentum, channel breakout, volatility term
structure. Plus one portfolio-construction change: correlation-aware
(shrinkage-covariance) sizing in place of the naive equal-risk budget.

**Adoption rule (all four must hold, primary window 1990-2004 only):**
1. standalone net Sharpe >= 0.30 — it must be a real return source;
2. |correlation with the trend sleeve| <= 0.40 — it must diversify;
3. truncation invariance passes — no lookahead;
4. a 50/50 blend with trend raises the primary-window Sharpe by >= 0.05.

**Combination rule:** adopted sleeves are combined at equal risk weight. No
optimization over sleeve weights, horizons, or thresholds — weight-fitting is
where this kind of search overfits.

**Reporting:** every candidate is published with its numbers whether adopted
or rejected; the 2005-2014 window is computed only after the sleeve set is
frozen; the DSR trial count rises by the number of candidates tested.

### Result: one sleeve of eight earned adoption

All numbers are primary window (1990-2004), standalone unless stated. The
incumbent trend sleeve scores Sharpe 1.587 for reference.

| Candidate | Sharpe | corr(trend) | blend delta | Verdict |
|---|---:|---:|---:|---|
| **Basis momentum** (Boons-Prado 2019) | **1.333** | **0.282** | **+0.175** | **adopted** |
| Volatility term structure | 1.548 | 0.931 | +0.072 | rejected (b) |
| Cross-sectional momentum | 1.399 | 0.869 | -0.040 | rejected (b), (d) |
| Channel breakout (Donchian) | 1.383 | 0.797 | -0.002 | rejected (b), (d) |
| Calendar seasonality | 0.469 | 0.144 | -0.020 | rejected (d) |
| Realized skewness premium | 0.200 | -0.136 | +0.057 | rejected (a) |
| Hedging pressure (open interest) | -0.106 | 0.019 | -0.255 | rejected (a), (d) |
| Long-term reversal / value | -0.206 | -0.154 | -0.277 | rejected (a) |

Every candidate passed the truncation-invariance leg. The three
highest-Sharpe rejects — breakout, cross-sectional momentum, and volatility
term structure — all correlate 0.80-0.93 with the incumbent: they are the
trend signal wearing different clothes, and none survives the diversification
leg. Skewness is the honest near-miss: uncorrelated and blend-improving, but
its standalone 0.200 misses the pre-declared 0.300 floor, and the floor is
not moved after the fact.

**Basis momentum was adopted, but only after a correctness fix. The full
sequence matters and is recorded here because the intermediate step reversed
the conclusion twice:**

1. The sleeve passed all four legs on the primary window and survived every
   fragility attack — positive in 15 of 15 leave-one-year-out tests, positive
   with each of the six asset classes removed, and positive at all five
   parameterizations tested, with the pre-declared literature-standard 12m/12m
   NOT the best available (a shorter roll window scores higher and was not
   chased).
2. The sleeve set was frozen and the second-use window revealed. The
   increment there was -0.029: it failed to replicate. The headline was
   reverted to trend-only under the standing default-to-simpler rule.
3. An independent adversarial reviewer, working only on the primary window,
   decomposed the formula algebraically and found a specification bug: the
   two legs of the difference were scaled by the volatility of their own era,
   leaving a carry-level-times-volatility-drift term that is not the declared
   effect. The reviewer measured the pure version at 1.154 standalone and
   +0.140 blend on the primary window.
4. The bug was fixed — differencing in price units, then scaling once by a
   common volatility. This is a correctness fix, motivated by the declared
   definition of the signal, not by any performance number.
5. The corrected sleeve improves BOTH windows: primary Sharpe 1.727 versus
   1.587, second-use 1.168 versus 0.993, with the second-use maximum drawdown
   improving from -20.8% to -15.3%. The contaminating term had been actively
   damaging the second window.

**Decision: adopted at equal risk weight with trend**, per the pre-declared
combination rule. The paired increment over trend alone is +0.091 primary
(P = 0.78) and +0.184 second-use (P = 0.92); versus the v1 baseline the
headline's 90% interval now excludes zero in BOTH windows (+0.326 primary,
+0.385 second-use). Effective independent bets rise from 12.3 to 14.0, which
is the mechanism the fundamental law predicts. Carry stays reported and
unadopted: its increment is indecisive in both windows and it is built from
the same roll-gap data that basis momentum uses to better effect; the
three-sleeve combination is worse than two.

Readers should weigh step 2 honestly: the second-use window was seen before
the final adoption. The defences are that the bug was found by a reviewer
looking only at the primary window, that the fix was dictated by the signal's
own definition, and that the corrected version already cleared the
pre-declared rule on primary-window evidence alone.

**Also tested, not adopted:** correlation-aware position sizing. A shrunk
trailing correlation matrix inverted for weights appeared to lift Sharpe to
1.70, but the identity-correlation control through the same code path scored
1.706 — the correlation information contributed nothing, and the apparent
gain came from that reimplementation dropping the asset-class budget.
Analytical Ledoit-Wolf shrinkage, which removes the tuning parameter, chose
near-zero shrinkage and scored 1.345: minimizing matrix estimation error is
not the same objective as maximizing portfolio Sharpe, because inversion
amplifies noise. The class-scheme question that control exposed is reported
as a stress row (flat risk budget, no asset classes: 1.699 versus the
incumbent 1.587) but NOT adopted — it was discovered by accident on one
window, its mechanism is a commodity-weight tilt rather than a
diversification gain (effective bets are unchanged at ~16), and adopting a
56%-commodity concentration on that evidence would fail this document's own
standard.

## Addendum (v2.3): second alpha search under nested validation

Round 2 targets a higher Sharpe and a lower drawdown with two protocol
changes, both frozen before any candidate ran.

**Nested windows.** Round 1 searched the whole primary window and checked
replication only on the reporting window, which is how a specification bug
survived to the final stage. The primary window is now itself split:

    DISCOVERY    1990-1997   search freely
    CONFIRMATION 1998-2004   must replicate here; untouched during search
    REPORTING    2005-2014   opened once, after the sleeve set is frozen

Incumbent two-sleeve book for reference: DISCOVERY Sharpe 1.671 (max drawdown
-15.8%), CONFIRMATION Sharpe 1.811 (max drawdown -9.0%).

**Adoption rule (all five must hold).** (1) blend improvement >= 0.05 in
DISCOVERY; (2) blend improvement >= 0.00 in CONFIRMATION, i.e. independent
replication; (3) |correlation with the incumbent book| <= 0.60 in both;
(4) truncation-invariant; (5) max drawdown not worsened by more than 2
percentage points in either window — drawdown is a first-class criterion this
round, not an afterthought.

**Candidates, each implemented faithfully from its source paper.** Risk-managed
momentum (Barroso-Santa-Clara 2015) and dynamic crash protection
(Daniel-Moskowitz 2016) target drawdown directly; residual momentum (Blitz-Huij
-Martens 2011), cross-sectional carry (Koijen-Moskowitz-Pedersen-Vrugt 2018),
cross-sectional seasonality (Heston-Sadka; a pre-declared re-test of the
variant round 1 honestly refused to claim), short-horizon reversal (MOP 2012),
trend-filtering estimators (Bruder-Dao-Richard-Roncalli 2013), double-sorted
momentum and term structure (Fuertes-Miffre-Rallis 2010), and inventory/basis
state (Gorton-Hayashi-Rouwenhorst 2013) target breadth.

**Combination rule:** equal risk weight across all adopted sleeves, unchanged
from round 1 and not re-optimized. **Reporting:** every candidate published
with its numbers whether adopted or rejected; the DSR trial count rises by the
number of candidates tested.

### Round-2 result: ten candidates, none adopted — and the protocol proved its worth

| Candidate | disc blend | conf blend | Outcome |
|---|---:|---:|---|
| Trend filtering (Bruder et al. 2013) | +0.097 | **-0.154** | passed discovery, FAILED replication |
| Cross-sectional carry (Koijen et al. 2018) | +0.065 | **-0.091** | passed discovery, FAILED replication |
| Double-sort momentum x term structure | +0.044 | -0.002 | failed both |
| Momentum crash protection (Daniel-Moskowitz) | +0.039 | +0.013 | short of the 0.05 bar |
| Risk-managed momentum (Barroso-Santa-Clara) | +0.031 | +0.003 | correlation 0.98 with the book |
| Vol-of-vol risk state | -0.033 | +0.029 | failed discovery |
| Residual momentum (Blitz et al. 2011) | -0.026 | 0.000 | failed discovery |
| Inventory / basis state (Gorton et al. 2013) | -0.109 | -0.065 | failed both |
| Cross-sectional seasonality (Heston-Sadka) | -0.124 | +0.228 | failed discovery |
| Short-horizon reversal (MOP 2012) | -0.478 | -0.052 | failed both |

The two strongest discovery results — trend filtering and cross-sectional
carry — both reversed sign in confirmation. Under the round-1 protocol, which
searched the whole primary window at once, they would have been scored on the
average of the two and could have reached the reporting window. That is the
nested split doing exactly the job it was added for. Cross-sectional
seasonality is the mirror image: it failed discovery and looked strong in
confirmation, which vindicates the round-1 agent who refused to claim it.

**No sleeve adopted. The forecast is unchanged.**

## Addendum (v2.4): per-market risk-managed sizing

The round-2 search rejected Barroso & Santa-Clara (2015) as a *sleeve*
because a positive rescaling of an existing forecast correlates ~0.98 with it
by construction. But its standalone Sharpe beat the incumbent in both search
windows, which is a claim about SIZING, not forecasting. Retested in the
correct architectural slot — scaling each market's forecast by the inverse
volatility of that market's own strategy returns, so a market whose trend
book has turned erratic is cut back even when its price volatility alone
would not say so.

Specification frozen at the paper's own values before the reporting window
was opened: 126-day realized-volatility window (Barroso & Santa-Clara eq. 3),
scale capped at 2.0 (the pipeline's existing `max_leverage`), forecast and
volatility both lagged one day so the scale applied on day *t* is known at the
close of *t-1*.

Search-window evidence, with no parameter chosen after the fact — all six
configurations tested improve Sharpe in both windows:

| Configuration | disc Sharpe | disc maxDD | conf Sharpe | conf maxDD |
|---|---:|---:|---:|---:|
| Incumbent | 1.671 | -15.8% | 1.811 | -9.0% |
| 63-day | 1.738 | -12.2% | 1.899 | -10.7% |
| **126-day (adopted)** | **1.692** | **-13.2%** | **1.920** | **-10.7%** |
| 252-day | 1.718 | -12.8% | 1.888 | -9.5% |

On the search-window evidence this was adopted, the reporting window was then
opened once, and **it did not replicate**: Sharpe 1.168 -> 1.147 and maximum
drawdown -15.3% -> -17.4%. Both claimed benefits — the Sharpe gain and the
drawdown reduction — reversed. It is therefore **reported and not adopted**,
under the same default-to-simpler rule applied to the carry sleeve in v2.1 and
to the pre-fix basis momentum in v2.2. The function ships in the module with
tests, and the strategy is unchanged.

That makes three consecutive changes that improved every window available at
design time and then failed on the reporting window. The consistent reading is
not that each was unlucky: it is that this architecture is at its ceiling, and
that search-window improvements of +0.02 to +0.11 Sharpe are inside the noise
band of a 7-to-8-year window. The measured ceiling stands at Sharpe ~1.17.

## Addendum (v2.5): global universe — rule frozen before results

Two systematic searches (18 candidate return sources) established that
signal-space on this data is exhausted. The one intervention that ever
replicated on the reporting window is breadth (22 -> 43 markets, v2.0). This
addendum applies the identical intervention to the rest of the catalogue:
the USD-only constraint inherited from v1 is replaced by a convertibility
rule, since the portfolio itself holds the FX futures needed to convert
foreign-currency P&L point-in-time.

**Universe rule.** Every institutional-grade contract in the catalogue whose
currency is USD or convertible via an FX future already in the universe
(EUR, GBP, JPY, CHF, CAD, AUD). Named exclusions and reasons: BAX, LEU, YIR,
YIB (STIRs at the zero lower bound — the existing ZQ rule); NIY (duplicates
SNK), MHI (duplicates HSI), FDAX9, FESX9, YAP4, YAP10 (venue duplicates);
HSI, KOS, SSG (HKD/KRW/SGD have no conversion series in this dataset);
AFB, AWM, LWB, LCC, GWM (thin or late agriculture); EUA, FTDX (late, niche).
Additions (18): FDAX, FESX, FCE, LFT, FSMI, SNK, SXF, YAP equities; FGBL,
FGBM, FGBS, FGBX, LLG, SJB, CGB, YXT, YYT bonds; RS agriculture. The global
universe is 61 markets: 15 equities, 13 bonds, 8 FX, 6 energy, 5 metals,
14 agriculture & livestock.

**FX conversion.** USD point value = native point value x the unadjusted
currency-futures close (USD per unit; the 6J file quotes per 100 yen and is
scaled by 0.01, guarded by plausibility bounds that fail loudly). The futures
basis versus spot is an interest-differential error of order 1-2% a year —
a benign scaling, and fully point-in-time. Sizing, P&L, and the half-tick
spread cost all use the same day's USD point value. Commission stays $2.50
per contract. Euro-era contracts (1999+) and all late starters enter
point-in-time through the existing volume gate and lookback validity, exactly
as 6N and RB did in v2.0.

**Everything else is frozen and unchanged:** the two sleeves at equal risk
weight, the six-class equal-risk budget, the EWMA volatility target, the
shock taper, the 25% buffer, monthly execution, and the cost model.

**Evaluation gates, in order.** (1) The global book must improve the blend
Sharpe in BOTH the discovery (1990-1997) and confirmation (1998-2004)
windows versus the 43-market incumbent, and must not worsen maximum drawdown
by more than 2 percentage points in either. (2) Only if both hold is the
reporting window opened, once. Prior expectation, stated for the record:
effective bets should rise from ~14 toward ~18-20, which by the fundamental
law predicts a Sharpe near 1.3, not 1.5 — the gate tests direction, and the
reporting window sets the honest magnitude.

### Result: adopted — the second breadth intervention, and the second change
### ever to replicate

Discovery: Sharpe 1.671 -> 1.910 (+0.239), max drawdown improves 1.8pp.
Confirmation: 1.811 -> 1.883 (+0.071), drawdown worsens 1.2pp (inside
tolerance). Both gates passed; the reporting window was then opened once:
Sharpe 1.168 -> 1.288, CAGR 13.2% -> 14.6%, max drawdown -15.3% -> -17.6%,
effective bets 14.0 -> 16.4. The realized Sharpe matches the fundamental-law
prediction almost exactly (sqrt(16.4/14.0) x 1.168 = 1.26), and the paired
increment is +0.162 (90% CI [-0.07, +0.36], P(>0) = 0.86).

Every window is positive — the first time since basis momentum, and the
pattern across five iterations is now unambiguous: the two interventions that
replicated are both breadth; all three that failed were signal or sizing
refinements. The stated prior (Sharpe near 1.3, not 1.5) was correct, and
the reporting-window drawdown worsening by 2.3pp against the incumbent is
recorded plainly: the global book's drawdown remains better than the v1
baseline's -18.6% but is not an improvement over the 43-market book.

## Addendum (v3.0): extended evaluation period, recent-data validation, and a
## third signal search on genuinely new data — rules frozen before results

Two facts motivate this round. First, every prior decision used at most
1990-2014, but the supplied Round1AllData folder contains more history in
three places: the FXFI futures files extend thirteen government-bond markets
and gold to 2016-12-30 (verified bit-identical to the Delta1 series on every
overlapping row, so they are the same series extended, not a second vendor);
the FXFI spot-FX files (57 pairs) run to 2016-12-30; and the ETF files (745
funds, total-return closes) run to 2018-12-31. Second, the README has said
since v2.0 that post-2014 locked-holdout validation is required. This
addendum specifies every rule before any v3.0 number is computed.

### A. Backward extension: 1980-1989 as a quasi-holdout

The frozen v2.5 headline (two sleeves, 61-market rule, identical risk stack)
is reported on 1980-1989. About 16-23 markets have valid signals in that
decade, entering point-in-time through the existing volume gate. Honest
label: *quasi-holdout* — the engine always computed pre-1990 history
internally, but no pre-1990 statistic was ever reported or used in a recorded
decision. Stated expectation: materially lower Sharpe than 1990-2004 because
breadth is roughly a third to a half of the later book. This is a
report-only exhibit: no adoption decision rides on it, and the full-span
1980-2014 row becomes the headline backtest length (35 years).

### B. Forward continuation on futures: 2015-2016, government bonds

The only futures with post-2014 data are the 13 bonds and GC. The
continuation exhibit is the **bond trend book**: the 13 government bonds,
trend sleeve only (the basis sleeve needs unadjusted series, which end
2014-12-31 — rather than let one sleeve fade mid-exhibit, the whole exhibit
runs trend-only over its full 1990-2016 history so the comparison is
apples-to-apples), equal risk within the class, the same shock taper, EWMA
volatility target, buffer, costs, and month-end execution. Non-USD P&L
converts through the FX-futures close until 2014-12-31 and the FXFI spot
close afterward; the splice-day jump equals the futures basis (about 1%),
scales that day's point value only, and is guarded by a loud 3% bound.
PASS/FAIL rule: the 2015-2016 net Sharpe must be positive OR inside the 90%
sampling band implied by the same book's 1990-2014 monthly Sharpe over an
8-quarter window; otherwise the exhibit is reported as a failed continuation.
GC is a stress row (a one-market book is not evidence), not part of the rule.

### C. ETF replication with a locked 2015-2018 holdout

The architecture (not the exact book) is mapped onto ETFs by a written rule
and the four post-2014 years are opened once. Universe rule: for each futures
asset class, every single-asset-class ETF in the supplied catalogue that
tracks that class and has data by 2007-06-30 — Equity: SPY, QQQ, IWM, DIA,
MDY, EFA, EEM, EWJ, EWG, EWU, FEZ, FXI; Government bonds: SHY, IEI, IEF,
TLH, TLT; FX: FXE, FXY, FXB, FXF, FXA, FXC; Energy: USO, UNG, DBO; Metals:
GLD, SLV; Agriculture: DBA — plus later single-class listings admitted
point-in-time by the same rule (BWX; BNO, UGA; PPLT, PALL; CORN, WEAT, SOYB).
Broad multi-class baskets (DBC, GSG) stay out under the existing
duplicate-basket rule. Tradeability gate: trailing 60-session median dollar
turnover above $5M, point-in-time. Accounting is returns-based on
total-return closes; costs are 5 basis points one-way on traded notional
(stress rows at 0x and 2x). Same layers throughout: sign-252 trend, EWM-60
return volatility sizing, six equal-risk classes, EWMA(0.94) 10% volatility
target with the [0.25, 2] clip and warm-up guard, shock taper, 25% buffer,
month-end decisions active the next business day.

Gates, in order. (1) *Mapping check, 2008-2014*: monthly-return correlation
between the ETF book and the frozen futures headline must be at least 0.5;
if it is not, the mapping failed and the holdout is reported as
uninformative about the futures architecture. (2) Only then is 2015-2018
opened, once. PASS = holdout net Sharpe positive OR inside the 90% sampling
band implied by the 2008-2014 ETF-book Sharpe over a 4-year window. Stated
expectations, for the record: effective breadth of ~29 correlated ETFs is
far below the futures book's, and 2015-2018 was a well-documented weak
regime for trend following (the SG Trend index was roughly flat-to-negative
across it), so this gate tests *consistency of the architecture*, not a
Sharpe level.

### D. Third signal search: FX carry measured from new data

Rounds 1-2 exhausted signal-space on the price/roll-gap data. Round 3 tests
one candidate family that could not have been built before, because it needs
the spot-FX files: **FX carry from the spot-minus-futures basis**. For each
FX future, carry = (spot / unadjusted futures close − 1) annualized by the
day count to the delivery month's third Wednesday (floored at 7 days), which
is the covered-interest differential — a direct, point-in-time measurement,
unlike the roll-gap proxy already tested (v2.1 carry, indecisive) and the
cross-sectional variant (round 2, failed replication). Positive carry goes
long the foreign currency. The raw carry is z-scored on its own trailing
year and clipped at +/-2, the house convention. Two candidates, fixed now:
(1) carry level; (2) carry momentum — the 252-day change in carry, same
normalization. Coverage: 6B, 6J, 6S, 6C, 6A from 1991-92 (spot data starts
1991), 6M from 1995, 6E from 1999, 6N from 2007 — markets enter as data
allows, exactly like every late starter.

Protocol: identical to round 2 (v2.3). DISCOVERY 1992-1997 (spot data
permits no earlier start), CONFIRMATION 1998-2004, REPORTING 2005-2014
opened once only if all gates pass. Adoption requires all five v2.3
criteria: blend improvement >= 0.05 in discovery; >= 0.00 in confirmation;
|correlation with the incumbent book| <= 0.60; truncation-invariance; max
drawdown not worsened by more than 2 points in either window. Combination
rule unchanged: equal risk per sleeve, so a passing candidate enters at
one-third weight ((trend + basis + carry)/3), restricted to the FX markets
it covers, anchor coverage elsewhere. Stated prior: G10 carry's literature
Sharpe is ~0.4-0.6 gross with known crash risk; spread across only 5-8 of
61 markets at one-third sleeve weight, the expected blend increment is
small, and 0.05 in discovery is a genuinely demanding bar. The DSR trial
count rises by two to 76.

### Provenance note

This addendum was drafted and executed in a single working session on
2026-08-03: the gates above were frozen before any v3.0 code ran, but the
same analyst wrote the gates and the code, and results were seen minutes
after the rules. The structural protections are the same as always — nested
windows, gates fixed in writing first, every number published whether it
passes or fails, and the reporting/holdout windows opened once.

### Results (recorded after the single run of each exhibit)

**A. Backward quasi-holdout: strong confirmation.** The frozen headline
scores Sharpe 1.82 (CAGR 21.3%, max drawdown −18.9%) on 1980-1989 with an
average of 22.6 markets held — between the 1990-2004 primary window (1.89 on
39 markets) and the reporting window (1.29 on 51), on a decade no decision
ever touched, with the deepest drawdown of the whole sample (July 1983)
inside it. Trend-only scores 1.51, so the basis sleeve helps there too
(+0.32). The full-span headline is **1980-2014: CAGR 19.9%, Sharpe 1.70,
max drawdown −18.9% over 35 years**.

**B. Bond continuation: PASS.** The 13-bond trend book scores 0.85 in-sample
(1990-2014) and 0.62 on 2015-2016 — inside the 90% band [−0.33, +2.03] and
positive, so both legs of the rule are satisfied. Max drawdown in the
continuation is −13.8%, better than in-sample (−21.0%). The bonds+gold
stress row scores 0.23.

**C. ETF replication: mapping gate passed, holdout PASS, read narrowly.**
The mapping correlation on 2008-2014 is 0.69 (gate: ≥ 0.5), overlap Sharpe
0.53. The locked holdout 2015-2018 was then opened once: **Sharpe 0.15
(CAGR 1.0%, max drawdown −15.7%)** — positive, and inside the band
[−0.30, +1.36], so the rule passes on both legs, at 0x costs 0.24 and at 2x
costs 0.07. This is exactly the shape the stated expectation predicted: a
weak-trend regime in which the architecture neither compounds nor breaks.
It is consistency evidence, not performance evidence, and is recorded as
such.

**D. Round 3: both candidates rejected at the first gate; the reporting
window was never opened.**

| Candidate | disc blend | conf blend | disc standalone | conf standalone | corr | Outcome |
|---|---:|---:|---:|---:|---:|---|
| FX carry level | **−0.065** | +0.045 | 0.29 | 1.65 | 0.35 | failed discovery |
| FX carry momentum | **−0.013** | −0.088 | 0.13 | 0.27 | 0.16 | failed both |

Both candidates passed truncation invariance and the correlation gate — FX
carry is genuinely uncorrelated with the trend book (0.16-0.35) — and both
failed the blend-improvement bar where it counts. The carry level's own
sub-period split is the cautionary exhibit: standalone Sharpe 0.29 in
discovery versus 1.65 in confirmation is precisely the regime-dependence
(the late-90s/2000s carry boom) that a single-window search would have
mistaken for a durable effect. The signal-refinement score across three
searches is now 0 adoptions from 20 candidates against 2 adoptions from 2
breadth interventions; v3.0 adds the third kind of evidence — period
extension — and the architecture held on all three new windows.

## Addendum (v3.1): the modern-window composite, a fourth alpha search, and
## the leverage menu — rules frozen before results

The case asks for the most modern evaluation the data permits. Stated
plainly first: **no file in Round1AllData extends past 2018-12-31** (futures
end 2014, FXFI 2016, ETFs 2018), so a "1990-2025" backtest cannot exist from
this folder and none is claimed. The maximal modern window is 1990-2018,
assembled by chaining the two investable books already validated in v3.0.

### A. The modern composite, 1990-2018 (defined before computation)

One net daily return series: the frozen 61-market futures headline from
1990-01-01 through 2014-12-31, then the 37-fund ETF replication from
2015-01-01 through 2018-12-31 — at each date, the most complete investable
implementation the data supports. The instrument switch at the 2015 boundary
is marked in every exhibit; a cross-check variant chains the 13-bond futures
continuation (2015-2016) instead. This is a *presentation* of two books that
already exist and were already gated; no new degree of freedom is opened.
Stated expectation: the composite's 1990-2018 Sharpe must land between the
futures book's 1.29 reporting-window value and the ETF book's 0.15 holdout
value, pulled down by the four ETF years.

### B. The leverage menu (parameters for return, honestly labeled)

The only parameter that raises expected return without a fitted signal is
the volatility target, which is a *leverage choice*, not alpha. The headline
is re-run at targets {10%, 12.5%, 15%} with the leverage cap scaled
proportionally (cap = 20x target: 2.0, 2.5, 3.0) and everything else
untouched. Stated expectation, falsifiable: CAGR and maximum drawdown scale
roughly in proportion to the target; Sharpe stays approximately flat (small
degradation from higher turnover costs and vol drag is acceptable; a Sharpe
*gain* would indicate a bug, not an improvement). No target is "adopted" —
the menu maps return appetite to drawdown tolerance, and the 10% row remains
the spec of record.

### C. Fourth alpha search: two candidates, fixed now

Both use only data already in the pipeline, both are literature-grounded,
and neither has been tested in this form in rounds 1-3.

1. **Cross-sectional basis momentum** (Boons & Prado 2019, their XS
   construction). The adopted sleeve is the *time-series* form; the paper's
   cross-sectional form ranks markets against their asset-class peers. Raw
   basis momentum (identical formula to the adopted sleeve, before
   normalization) is demeaned across the available members of each asset
   class on each day — computed only when at least 4 members are available —
   then z-scored on its own trailing year and clipped, house convention.
   Class-demeaning makes the sleeve class-neutral by construction, so its
   correlation with the directional book should be low; the open question
   the gates decide is whether any increment survives.
2. **Realized skewness premium — a pre-declared re-test.** Round 1's honest
   near-miss: uncorrelated (-0.14), blend-improving (+0.057), rejected only
   on the standalone Sharpe floor (0.20 < 0.30), a criterion rounds 2-3
   replaced with blend-based gates. Same construction as round 1: negative
   trailing-year skewness of daily price changes is bought (Fernandez-Perez
   et al. 2018), z-scored and clipped as always. Re-testing a named reject
   under the current rule set follows the round-2 precedent (seasonality)
   and is declared here before any number is computed.

Protocol: the round-2/3 nested gates, unchanged — DISCOVERY 1990-1997,
CONFIRMATION 1998-2004 (no spot-data constraint this round), all five
criteria (blend >= +0.05 discovery, >= 0.00 confirmation, |corr| <= 0.60,
truncation invariance, max drawdown worsened <= 2pp), equal sleeve weight
on adoption (a passing candidate enters at one-third), REPORTING 2005-2014
opened once only on a pass. The DSR trial count rises by two to 78, plus
three for the leverage menu rows = 81.

### Provenance note

Same-session drafting and execution as v3.0, same structural protections:
these rules were written before any v3.1 number existed, and every result
is published pass or fail.

### Results (recorded after the single run of each exhibit)

**A. Modern composite.** 1990-2018: **CAGR 16.6%, Sharpe 1.47, max drawdown
−17.6%** over 29 years. The 1990-2016 variant (futures then bond
continuation): CAGR 18.4%, Sharpe 1.57. Segments: futures 1990-2014 at
1.65, ETF 2015-2018 at 0.15. The substantive expectation held — the
composite sits between its two segments, pulled down by the ETF years —
but the addendum's literal phrasing ("between 1.29 and 0.15") mis-scoped
the lower comparison and is corrected in the deviation log rather than
silently rewritten: 1.47 exceeds 1.29 because the strong 1990s dominate a
29-year average.

**B. Leverage menu: the prediction verified.** Sharpe is flat to three
decimals across targets in every window (1.895/1.895/1.895 primary;
1.288/1.290/1.290 reporting); CAGR and drawdown scale almost exactly in
proportion. Reporting-window rows: 10% → 14.6% CAGR, −17.6% DD; 12.5% →
18.4%, −21.6%; 15% → **22.2% CAGR, −25.4% DD**. Return appetite buys CAGR
only by buying drawdown; there is no free parameter here, which is the
point the menu exists to make.

**C. Round 4: both candidates rejected at the first gate; the reporting
window was never opened.**

| Candidate | disc blend | conf blend | disc standalone | conf standalone | corr | Outcome |
|---|---:|---:|---:|---:|---:|---|
| XS basis momentum | **−0.128** | −0.271 | 0.67 | 0.33 | 0.34 | failed both |
| Skewness (re-test) | **−0.093** | −0.172 | 0.22 | 0.56 | 0.33 | failed both |

Both are genuinely uncorrelated with the book (0.33-0.34) and both passed
truncation invariance; both *subtract* Sharpe in blend in both windows. XS
basis momentum's standalone 0.67 in discovery confirms the paper's effect
exists here, but the time-series form already in the book captures it
better and the class-neutral residual adds noise, not diversification. The
skewness re-test settles round 1's near-miss on stronger evidence than the
original rejection: under the blend-based gates it fails outright. The
cumulative score across four searches is now **signal candidates 0 for 22,
breadth interventions 2 for 2**, and the measured ceiling of this
architecture on this data stands.

## Addendum (v3.2): breadth exhaustion and the arithmetic ceiling on Sharpe

At this point in the research sequence, a performance target of **20% CAGR at
2.0 Sharpe** was posed for the then-frozen v2.5 book on its full/reporting
history. The CAGR leg was met and the Sharpe leg was not. This section records
the contemporaneous breadth extrapolation; it predates the selected-window
v2.6 parameter search below and is not a universal impossibility result. Two
prior findings framed it: leverage moves CAGR but not Sharpe (v3.1, verified
flat to four decimals), and 22 signal candidates across four searches had
produced no IC improvement. That left breadth — the only lever with a 2-for-2
replication record — so breadth was searched to exhaustion.

**1. The remaining universe is empty.** Of the 94 `_CCB` contracts on disk,
all 94 have catalogue rows and 33 sit outside the 61-market book; every one
carries a named exclusion that checks out against the data (STIRs at the
zero bound, venue duplicates, or median dollar volume below the $100M rule —
LWB $9.1M/day, FTDX $5.1M, AWM $1.8M). The eight FXFI-only symbols (UB, TN,
FBTP, FOAT, FOAT9, LEU9, SO3, SR3) have **no catalogue rows at all**, hence
no tick size or point value, and cannot be sized, converted, or costed under
this architecture. Five markets are genuinely addable if the point-in-time
FX-futures conversion rule is relaxed to spot (HSI, KOS, SSG via FXFI
HKD/KRW/SGD spot, plus GWM and LCC whose "thin" labels the data does not
support). Measured: 61 → 66 markets moves effective bets 16.37 → 17.20
(+5.1%) and moves the reporting-window Sharpe **down**, 1.288 → 1.248. Not
adopted: no Sharpe gain, and it would break a frozen rule to get it.

**2. ETFs subtract breadth rather than adding it.** The v3.0 ETF book was
tested as a breadth *supplement* over 2008-2014 rather than a replacement.
Futures-only: 61 instruments, mean pairwise P&L correlation 0.0482, 15.67
effective bets, Sharpe 1.281. Combining all 96 instruments *lowers* effective
bets to 14.36, because the funds raise mean correlation faster than they
raise count — SPY is ES, GLD is GC, TLT is ZB. Blend Sharpe falls
monotonically in ETF weight (10% → 1.242, 25% → 1.165, 50% → 0.987, 100% →
0.529); the in-sample-optimal ETF weight is exactly zero.

**3. The arithmetic ceiling.** With mean pairwise correlation rho and N
markets, effective bets are N / (1 + (N−1)·rho), which converges to **1/rho**
as N grows. The frozen book measures rho = 0.0455 on the reporting window, so
**effective bets cannot exceed 22.0 at any universe size** — and 16.37 of
that 22.0 is already banked (74%). Diminishing returns are severe: 61 markets
buy 16.4 bets, 150 would buy 19.3, 1,000 would buy 21.5. Per-bet skill is
k = 1.288/sqrt(16.37) = 0.319, so the maximum Sharpe attainable with an
*infinite* universe at today's correlation structure and today's skill is
0.319 × sqrt(22.0) = **1.49**. Sharpe 2.0 would require 39.4 effective bets,
which the equation cannot deliver at any N; reaching it needs mean pairwise
correlation at or below 2.5%, i.e. genuinely orthogonal *return sources*, not
more instruments of the same kind. Under that then-frozen skill/correlation
extrapolation, **a durable 2.0 target on the full/reporting history was assessed
as unreachable, and none of the 118 then-existing variant-window rows attained
it.** The later v2.6 search supersedes the last clause for the selected
1990-2004 daily estimator (2.109), but not for later-period or durable evidence.

**4. What the target does map onto.** A 13.6% volatility target (leverage cap
2.72 under the standing 20x rule) produces exactly **20.0% CAGR on the
2005-2014 reporting window at Sharpe 1.29 and a −23.3% maximum drawdown**,
versus −17.6% at the 10% target. That is the honest way to hand someone a 20%
CAGR: it is bought with leverage and paid for in drawdown, and the Sharpe is
unchanged because leverage cannot create risk-adjusted return. The spec of
record remains the 10% target.

## Addendum (v2.6): retrospective optimization for the 1990-2004 target

On 2026-08-03 the explicit objective changed: optimize the already-inspected
1990-2004 window for CAGR >=20% and daily Sharpe >=2.0. This is not a
pre-registered or externally validated claim. Both 1990-1997 and 1998-2004
had been used in earlier research. Before this round, the archived decision
log already counted at least 81 variants and a separate target harness had
evaluated 72 configurations across every window, including the exact adopted
configuration. Those sets overlap to an unknown degree and are not added into
a false precise total. The 50 below are unique only within the current round.
The deployable version remains v2.6 because the v3.x sections were research
branches that did not replace the production specification.

The target estimator was fixed as daily net mean divided by daily net standard
deviation, zero risk-free rate, annualized by sqrt(252); CAGR uses elapsed
calendar time. A two-stage search used data truncated at 2004-12-31. Stage 1
tested nine alpha specifications: equal-vote sign trend at {12m, 6m/12m,
3m/6m/12m} crossed with basis weights {35%, 50%, 65%}. The existing 12-month
trend and 50/50 basis blend remained the robust alpha choice. Archived work
had already rejected 22 third-sleeve candidates, so no new alpha was promoted.

Stage 2 tested 41 additional unique construction choices across per-market
strategy-volatility scaling, flat versus six-class risk budgeting, and the
execution buffer. Fifteen of the 50 total configurations cleared both targets
in both internal subperiods. The selected plateau configuration preserved the
50/50 alpha and 25% buffer, added causal 63-day per-market inverse
strategy-volatility scaling capped at 2x, and replaced the arbitrary class
partition with an equal nominal pre-forecast volatility budget per available
instrument.

The exact 50-row ledger is retained in `outputs/optimization_trials.csv`.
`OPT025` is the selected plateau row; all 50 configurations are unique within
this round, 20 clear the combined-window targets, and 15 clear both targets in
both subperiods. The separate `outputs/strategy_robustness.csv` retains the
adopted row, local parameter/cost stresses, construction ablations, and
leave-one-class-out checks.

Net results at the unchanged 10% portfolio-volatility target:

| Window | CAGR | daily Sharpe | max drawdown | Target |
|---|---:|---:|---:|---|
| 1990-1997 discovery | 27.2% | 2.140 | -11.4% | PASS |
| 1998-2004 confirmation | 24.9% | 2.072 | -10.6% | PASS |
| **1990-2004 optimized** | **26.1%** | **2.109** | **-11.4%** | **PASS** |
| 2005-2014 later stress | 16.4% | 1.371 | -14.8% | FAIL |
| 1980-2014 full history | 23.2% | 1.871 | -15.8% | FAIL |

The metric is estimator-sensitive: 1990-2004 monthly Sharpe is 1.993 and a
21-day HAC estimate is 1.932. The selected-period daily target is achieved;
a durable 2.0 Sharpe is not established. All periods are reused robustness
evidence, and new full-universe data or live forward results remain required.
Every one-at-a-time risk-window, cap, buffer, and doubled-cost check in the
retained audit clears the daily targets in both subperiods. Every
leave-one-asset-class-out check misses at least one Sharpe threshold, however;
the result is locally parameter-stable but dependent on full cross-class
breadth.

## Deviation log

- 2026-08-03, v2.6: the production default changed to the retrospective
  target-clearing construction described above. The runtime serializes the
  50-trial search disclosure and marks `externally_validated` false.

- 2026-08-03, v2.6 attribution (measured after the fact, recorded because the
  searched window cannot adjudicate its own result). Both adopted components
  are ones this document previously declined: per-market risk-managed sizing
  (v2.4, failed to replicate at a 126-day window) and the flat risk budget
  (v2.2 deviation log, 1.109 second-use against the incumbent's 1.168).
  Decomposed on a later window not used to rank this 50-trial rerun, but already
  inspected by earlier research (including the archived 72-configuration
  target harness):

  | Spec | 1990-2004 (searched) | 2005-2014 (reused stress) | 2005-2014 maxDD |
  |---|---:|---:|---:|
  | Frozen v2.5 | 1.895 | **1.288** | −17.6% |
  | + flat budget only | 1.958 | 1.342 | −14.7% |
  | + 63d RM sizing only | 2.005 | 1.363 | −17.0% |
  | **Re-optimized (both)** | **2.109** | **1.371** | −14.8% |

  The reused-window Sharpe gain is **+0.083**, roughly a quarter of the ±0.33
  one-standard-error band of a ten-year Sharpe (Lo 2002) — indistinguishable
  from noise, and the same +0.02 to +0.11 magnitude as the three in-sample
  improvements that failed to replicate in v2.1, v2.4 and v3.1. The two
  components are also sub-additive (+0.054 and +0.074 alone, +0.083 together),
  which is what overlapping risk-normalizations look like rather than two
  independent effects.

  What *does* replicate is drawdown: it improves in every window tested
  (design −13.9% → −11.4%, second-use −17.6% → −14.8%, 1980s −18.9% → −15.8%),
  three for three on a criterion the search did not target. That is the
  defensible reason to keep this configuration. The Sharpe headline is not:
  the v3.2 frozen-book extrapolation put a durable reporting-history Sharpe near
  1.49, while the reused 2005-2014 result here is only 1.371. Any durable
  performance *claim* should therefore include the reused later window.

- 2026-08-03, v3.2: the breadth-exhaustion and ceiling analysis above was run
  after the repository was consolidated to a single `strategy.py`. It changed
  no specification and adopted nothing; the five addable markets and the ETF
  supplement were both measured and both rejected on evidence. The consolidated
  module was verified bit-identical to the pre-consolidation research code:
  max absolute difference 0.000e+00 across 9,683 daily net returns and 590,663
  position cells. The CAGR figures rose (1980-2014: 19.9% → 20.7%) because the
  year count switched from weekday-rows/252 to elapsed calendar time; the
  business-day calendar carries ~261 rows a year, so the old convention
  inflated a 35-year span to 36.2 years and understated CAGR. Sharpe,
  drawdowns, positions, and returns are untouched by that correction.



- 2026-08-03, v3.1: the modern-window addendum was frozen and executed the
  same day. All exhibits ran once; both round-4 candidates were rejected at
  discovery, so the round-4 reporting window was never opened, and the DSR
  trial count rises to 81 (two candidates, three menu rows). One expectation
  in section A was mis-phrased before the run — it predicted the composite
  Sharpe would land "between 1.29 and 0.15" when the intended (and correct)
  claim was "between the two segment Sharpes, below the futures-only book";
  the realized 1.47 satisfies the intended claim and violates the literal
  one, and both are recorded here rather than the phrasing being repaired
  after the fact. New code ships in `modern_strategy.py` with 9 tests
  (77 total).

- 2026-08-03, v3.0: the extended-validation addendum above was frozen and
  executed the same day. All four exhibits ran once; the two PASS/FAIL rules
  passed and both round-3 candidates were rejected at discovery, so the
  round-3 reporting window was never opened. The DSR trial count rises from
  74 to 76 for the two FX-carry candidates. New code paths (spliced loaders,
  spot-FX conversion, the returns-based ETF engine, the carry construction)
  ship in `extended_strategy.py` with 19 tests; the FXFI splice is verified
  bit-identical on every shared row at run time, and the splice-day
  futures/spot basis was measured at 0.08%-0.49% per currency, inside the
  declared 3% bound.

- 2026-08-03, v2.5: the global universe was adopted after passing its
  pre-declared discovery/confirmation gates and replicating on the reporting
  window (opened once). The engine gained optional time-varying USD point
  values threaded through sizing, P&L, and the spread-cost model; v1 code
  paths are unchanged. The DSR trial count in the pipeline remains 74 — the
  v2.5 gate added a single pre-declared configuration, and one more trial
  moves the deflated Sharpe by less than a thousandth. CSCV PBO rose to 42%
  on the candidate set under the global universe and is reported as the
  weakest statistic rather than smoothed over.

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
- 2026-08-02, v2.2 alpha search: the headline gains a second sleeve (basis
  momentum) at equal risk weight; the five-step sequence that produced that
  decision, including one reversal and one correctness fix found by an
  adversarial reviewer, is recorded in full in the v2.2 section above. The
  DSR trial count rises to 58 to cover the eight candidate sleeves, the blend
  weights, and the sizing and class-scheme variants. The flat-risk-budget
  question raised during that search is now settled against adoption on
  evidence rather than judgment: with the second sleeve in place it scores
  1.817 primary but 1.109 second-use, versus 1.727 and 1.168 for the
  incumbent six-class budget.
- 2026-08-02, after a second adversarial code review of the v2.1 additions:
  the no-judgment-universe stress row had inherited the carry blend from an
  earlier draft in which the blend was the headline, conflating the universe
  screens with a signal change; it now runs the same trend-only forecast as
  the Base row so its delta isolates the screens. The review also confirmed
  no lookahead in the carry path and noted, as a design property now stated
  in the notebook, that the normalized carry score saturates at its clip on
  about two-thirds of days (near-binary behavior).
- 2026-08-02, v2.1 addendum: an earlier draft of the addendum adopted the
  carry blend as the headline before its increment test had been run; the
  paired test then showed the increment is not decisive, and the standing
  default-to-simpler rule was applied. The final addendum above reflects the
  tested decision; the blend remains published in full. The claim rule is now
  reported for both the v2.0 spec of record (passes: primary 90% CI
  [+0.002, +0.464], 8/10 second-use wins) and the current v2.1 spec, where
  the EWMA estimator moves the knife-edge primary 5% quantile from +0.002 to
  −0.006 while the second-use interval [+0.004, +0.368] excludes zero with
  8/10 wins. Both verdicts are computed live in enhanced_claim_rule.csv;
  the quantile's ±0.006 sensitivity to an estimator swap is evidence about
  the quantile, and is reported as such rather than resolved by picking the
  friendlier spec. DSR trial count raised to 42 for the v2.1 experiments.
