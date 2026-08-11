# Benchmark and Validation Findings

Two questions this repository could not previously answer from inside itself:

1. **Is a 1.59 Sharpe good?** There was no strategy benchmark anywhere. The
   committee notebook carried descriptive bands — "Conservative",
   "Institutional", "High alpha" — pasted against the incumbent's own numbers,
   which is a classification of a number rather than a comparison of a strategy.
2. **Does the declared methodology exist in code?**
   `docs/research-methodology.md` names anchored expanding walk-forward as the
   primary diagnostic, a family-wise max-statistic procedure as a promotion gate,
   and CSCV/PBO as a secondary diagnostic. None of the three existed. The gates
   were prose.

Both are now answered by executable code. Everything below is reused 1990–2014
history unless stated otherwise, and none of it promotes a configuration.

## 1. Published trend following, replicated on the identical panel and ledger

Each rule builds a monthly decision frame and is handed to
`strategy._simulate_execution` with the same engine and all common execution
and cost assumptions. Costs, integer contracts, the 2% participation cap,
realized-capacity fill truncation, roll turnover and FX are therefore identical
*by construction*, not by inspection. Rule-specific releases of the gross
notional cap are declared in the table and artifact; capped MOP is reported
separately. `strategy.py` is unmodified.

The seam is proven rather than asserted: replaying the incumbent's own decision
frame through the benchmark execution path reproduces the canonical ledger with
**maximum absolute daily net-return deviation of exactly 0.0**
(`outputs/benchmarks/benchmark_seam_check.csv`).

| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar | MDD/vol | ρ incumbent |
|---|---|---|---|---|---|---|---|
| **Incumbent v3.2.1** | 13.19% | 7.72% | **1.590** | −11.85% | 1.113 | 1.53 | 1.000 |
| Moskowitz-Ooi-Pedersen (2012) TSMOM | 13.84% | 11.95% | 1.108 | −21.22% | 0.652 | 1.78 | 0.855 |
| MOP under the incumbent's leverage cap | 12.73% | 11.57% | 1.058 | −21.46% | 0.593 | 1.85 | 0.851 |
| Hurst-Ooi-Pedersen (2017) 1/3/12 blend | 11.47% | 9.62% | 1.138 | −15.50% | 0.740 | 1.61 | 0.646 |
| Baltas-Kosowski trend *t*-statistic | 10.00% | 11.63% | 0.850 | −26.04% | 0.384 | 2.24 | 0.763 |
| Baz MACD/EWMAC crossover | 8.21% | 7.93% | 1.002 | −13.66% | 0.601 | 1.72 | 0.638 |
| Long-only equal risk | 9.34% | 13.42% | 0.710 | −42.26% | 0.221 | 3.15 | 0.184 |
| Long-only equal risk, volatility matched | 6.50% | 7.72% | 0.826 | −26.52% | 0.245 | 3.43 | 0.201 |
| Long-only equal notional | 2.68% | 6.40% | 0.431 | −27.07% | 0.099 | 4.23 | 0.006 |
| Barroso-Santa-Clara scaling on MOP | 16.69% | 12.65% | 1.242 | −17.01% | 0.981 | 1.34 | 0.867 |
| Barroso-Santa-Clara scaling on incumbent | 19.27% | 11.11% | 1.588 | −16.64% | 1.158 | 1.50 | 0.984 |
| Moreira-Muir, expanding causal constant | 9.53% | 6.32% | 1.423 | **−8.65%** | 1.101 | **1.37** | 0.852 |
| Moreira-Muir, full-sample non-causal constant | 9.20% | 6.06% | 1.434 | **−8.27%** | 1.113 | **1.36** | 0.832 |

**The incumbent's edge is risk control, not signal.** MOP TSMOM earns *more*
CAGR — 13.84% against 13.19% — at 11.95% volatility against 7.72%. The
incumbent runs comparable return at 65% of the risk with slightly more than half
the drawdown. That is a defensible claim; "better signal" is not.

**Barroso-Santa-Clara on the incumbent is a volatility-management overlay, not
a new signal.** Its causal 126-session weights vary through time, but the result
is 19.27% CAGR at Sharpe 1.588 against the incumbent's 13.19% at 1.590: higher
return came with higher risk and essentially unchanged Sharpe, which is the
same conclusion `docs/lever-program-findings.md` reached from the other
direction.

**Moreira-Muir on the incumbent is the one benchmark relevant to drawdown.** The
implementable expanding-constant row cuts maximum drawdown to −8.65% and
drawdown-per-unit-volatility to 1.37 against 1.53, at a cost of 0.167 Sharpe. The
paper-definition full-sample constant is reported separately because it uses
future volatility and is therefore non-causal; its 1.434 Sharpe is 0.011 *above*
the causal row, not below it. It belongs in the same family as the
shape-improving levers in
[`drawdown-attribution-findings.md`](drawdown-attribution-findings.md).

### The spanning test

Regressing the incumbent's daily net returns on four published rules plus the
long-only equal-risk control jointly, Newey-West with 21 lags:

| | Coefficient | HAC *t* |
|---|---|---|
| **Intercept (alpha)** | **4.64% / yr** | **6.04** |
| MOP TSMOM | 0.509 | 15.16 |
| HOP trend blend | 0.036 | 1.46 |
| Baltas-Kosowski | 0.004 | 0.16 |
| Baz MACD/EWMAC | 0.031 | 1.08 |
| Long-only equal risk | 0.023 | 1.58 |

R² 0.735. The incumbent is approximately **half a unit of published time-series
momentum plus 4.6% a year that published trend following does not span**. Only
the MOP loading is significant; the other four coefficients are not individually
significant at conventional levels once it is present.

**4.64% is an optimistically biased reused-history estimate, not an unbiased
estimate of edge.** The benchmark rules were run cold and unrefitted on this
panel, while the incumbent's parameters were chosen with the panel visible.
That asymmetry biases the comparison in the incumbent's favour in expectation;
its size is unmeasurable here because the lineage's trial count is unrecoverable.

### The no-skill null

One thousand block sign-flips of the incumbent's own monthly decision frame use
the same sizing magnitudes and the same cost, capacity and execution model, with
each permuted path re-executed. Null Sharpe mean −0.199
(negative because random trading still pays costs), 95th percentile +0.242, 99th
+0.417. The incumbent's 1.590 sits at the **100th percentile**, empirical
one-sided p = 0.000999, which is the resolution floor at 1,000 permutations and
must not be quoted as smaller.

## 2. The methodology, now executable

### Anchored expanding walk-forward

Twenty annual boundaries, anchor 1990-01-01, selection over
`trend_lookback ∈ {126, 189, 252, 315}`, one chronological replay.

| | Selector active | Specification frozen |
|---|---|---|
| Stitched span | 1995-01-02 → 2014-12-31 (selector-only out of sample) | same dates; frozen-specification stability description |
| Sessions | **5,218 (20.7 252-session-equivalent years; 20.0 elapsed calendar years)** | 5,218 |
| CAGR | 11.59% | 12.93% |
| Sharpe | 1.418 | 1.583 |
| HAC Sharpe | 1.321 | 1.476 |
| Max drawdown | **−18.19%** | −11.85% |
| Walk-forward efficiency | 0.896 | 1.000 |
| Selection switches | 2 of 20 | 0 |

Letting the trend lookback be chosen out of sample costs 0.165 Sharpe and
deepens maximum drawdown by 6.34 percentage points — **through the 15% drawdown
policy**.

**The only selected-variant difference is fold 0.** Calendar 1995, selected on 1,305
training sessions of 1990–1994, picked `trend_315` and returned −11.61% CAGR at
fold Sharpe −1.58 and fold drawdown −17.38%. The frozen specification's same
fold returned +10.25% at Sharpe +1.32. The selector then chose the baseline for
folds 1–19. Because the book is simulated once and carries NAV, positions and
backlog across boundaries, the fold-0 state difference propagates into later
fold metrics even after the selected variant matches. One parameter choice made
on five years of history therefore shaped the full stitched result. That is a
stronger argument for freezing a specification before evaluation than any
amount of policy language.

The construction is splice-then-simulate-once: each candidate's decision frame
is built over the whole history, the selected regimes are spliced into one
frame, and `_simulate_execution` runs once, so NAV, the executed book, the roll
backlog and the cost ledger cross every boundary intact and switching turnover
is charged. It is pinned by an elementwise identity — with a constant parameter
the spliced replay reproduces the single-shot backtest to 1e-12 — and by a test
asserting a fold inherits the previous regime's book rather than reproducing the
incoming candidate's standalone path, which is precisely the flattering error a
per-fold NAV restart produces.

### Family-wise inference: no rejection anywhere on Sharpe

Seventeen declared configurations, benchmark the incumbent, 6,523 daily
sessions, B = 10,000.

| Procedure | block 21 | block 63 | block 126 |
|---|---|---|---|
| White Reality Check | 0.655 | 0.660 | 0.642 |
| Hansen SPA lower | 0.246 | 0.242 | 0.236 |
| Hansen SPA consistent | 0.321 | 0.310 | 0.277 |
| Hansen SPA upper | 0.395 | 0.375 | 0.357 |

The best in-sample member, `basis_060` at Sharpe 1.677 — **+0.088 over the
incumbent** — is not distinguishable from the incumbent once the sixteen-way
search is priced.

The Hansen SPA annualized-mean rows reject at the p ≈ 1e-4 resolution floor,
while White's Reality Check does not (p = 0.087–0.106). The SPA rejection is an
artefact: `risk_080` merely raises the volatility target from 7% to 8%, lifting
the annualized arithmetic mean by 1.89 percentage points on a path correlated
**0.994** with the
incumbent. Only the Sharpe rows bear on an improvement claim. This is exactly
why `docs/research-methodology.md` specifies the primary endpoint at the same
ex-ante risk.

### CSCV / PBO: the permitted monthly estimate and the daily sensitivity

On the synchronized monthly paths required by the methodology, PBO = **0.4193**
over 12,870 combinations of the seventeen-configuration family (S = 16, 18
monthly observations per submatrix, 12 months dropped from the oldest end to
make the history divide). Probability of loss is 0.000. The daily matrix produces
0.0733 (407 observations per submatrix, 11 sessions dropped), but it is emitted
only as an explicitly labelled secondary sensitivity; **0.42 is the PBO to
quote**.

The permitted PBO is near the 0.5 signature of pure overfitting, while SPA at
p = 0.32 says no selected configuration is distinguishable from the incumbent.
The statistics answer different questions: PBO describes selection instability
inside the declared family; SPA tests whether the best apparent improvement
survives the family-wise correction. Neither supplies a promotion decision on
this reused history.

Two honest qualifications. The monthly degradation slope of −0.969 is largely
mechanical: CSCV's halves are complements of one fixed history, which forces a
negative slope whenever one configuration wins most splits. And the much lower
daily estimate describes a narrow, highly correlated design space at a frequency
that the methodology does not permit for the headline PBO. A wider or more
adversarial grid could change either estimate.

### The refusals fire on real data

Fed the existing four-variant lever sweep in `outputs/levers`, CSCV returns
`NOT_ESTIMABLE` for all four statistics: *"4 configurations quantise the
out-of-sample rank in steps of 1/5, which measures the grid rather than the
overfitting; the floor is 10."* That is
`docs/research-methodology.md`'s "not estimable, never PASS" enforced by code
rather than by the reader.

The illustrative lower-bound deflated probabilities are numerically near
saturation for all seventeen members: they range from 0.9998407793 to
0.9999999999947, none is exactly 1.0, and fourteen round to 1.000000 at six
decimal places. The deflated-Sharpe result itself is `NOT_ESTIMABLE` for every
member because seventeen configurations *declared today* are not the lineage's
unrecoverable true trial count; `inference.promotion_support` therefore blocks
promotion.

## 3. Descriptive robustness of the frozen baseline

Four analyses added after the validation suite, all of them descriptions of the
one frozen configuration — no family, no selection, and nothing here altered a
parameter. Each artifact carries its own status label and the runner that wrote
it reconciles its baseline before emitting a number
(`outputs/validation/validation_*.csv`).

**Crisis windows** (`validation_crisis_windows.csv`). Eight windows declared
from public event dates — an invasion, a rate hike, a bankruptcy filing — not
from the equity curve. The strategy is net positive in six of eight: +14.2%
through the 1990 Gulf shock, +39.9% through the dot-com unwind, +3.4% through
Lehman's quarter. The two losses are LTCM/Russia (−2.9% over 65 sessions) and
the 2011 US downgrade (−3.7% over 45 sessions). Worst within-window drawdown
across all eight: −8.9% (1994 bond selloff). None of these windows was chosen
by looking at the result.

**Cost breakeven** (`validation_cost_breakeven.csv`). All five execution-cost
inputs scaled jointly from 0.5x to 16x, everything else frozen. Net Sharpe is
strictly decreasing in the multiplier; net CAGR crosses zero at an interpolated
**12.7x** the modeled costs and net Sharpe at **12.8x** — both crossings inside
the declared grid. The 1.0x row reproduces the published metrics row and the
2.0x row reproduces the friction stress's `double_all_execution_costs` to
twelve decimal places before anything is written.

**Capacity** (`validation_capacity.csv`). The frozen configuration replayed at
1x–100x the $1M initial capital. The binding constraint is not impact-cost
Sharpe erosion — at 2x the Sharpe is marginally *higher* (integer-contract
granularity improves target tracking faster than the extra ~4bp of cost drag
hurts). From **$5M upward every replay aborts on the 21-session roll-completion
guard** in thin markets (SJB first, then RS, GF), so on this cost model the
working capacity sits **between $2M and $5M** and is set by roll completion,
not by price impact. The Sharpe-erosion thresholds the sweep was designed to
interpolate are therefore not estimable, and the artifact records that refusal
rather than a number.

**Leave-one-sector-out** (`validation_universe_jackknife.csv`). The frozen
configuration replayed six times, each with one asset class removed. Full-book
Sharpe 1.59; the worst single exclusion (equity indices, 15 markets removed)
leaves 1.36, and no exclusion pushes maximum drawdown past −15.2%
(government bonds out). The prose claim that no single market or class carries
the result now has a table behind it.

## What none of this establishes

Every number above is reused 1990–2014 history. The walk-forward's segments are
out of sample with respect to the **selector only** — the specification being
replayed was written with this window already read. The seventeen configurations
are ones a researcher might plausibly have tried, not the ones that were tried,
so every family-wise p-value corrects for a family declared today and is a lower
bound on the correction actually required.

Genuine out-of-sample evidence needs data the pipeline has never read. The
2015–2016 futures continuation supplies two years on twelve roots
(`outputs/holdout/`). The contiguous five-year forward block lives in the ETF
sleeve, whose panel runs to 2018-12-31 — see
[`etf-regime-allocation-findings.md`](etf-regime-allocation-findings.md).

## Reproduce

```bash
python scripts/run_benchmark_comparison.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" --output-dir outputs/benchmarks

python scripts/run_validation_suite.py \
  --data-dir "Round1AllData/Quant Researcher/Delta1" --output-dir outputs/validation

# descriptive robustness of the frozen baseline
python scripts/run_crisis_windows.py
python scripts/run_cost_breakeven.py     --data-dir "Round1AllData/Quant Researcher/Delta1"
python scripts/run_capacity_sweep.py     --data-dir "Round1AllData/Quant Researcher/Delta1"
python scripts/run_universe_jackknife.py --data-dir "Round1AllData/Quant Researcher/Delta1"
```
