# Research methodology and alpha-promotion policy

## Current evidence status

Every supplied futures observation through 2014-12-31 is retrospective reused
history. The 1990–2004 and 2005–2014 windows are diagnostics, not independent
holdouts. Archived Git history records many earlier signal, universe, risk and
leverage trials, but complete synchronized return paths no longer exist for
every trial. The lineage's global trial count therefore cannot be reconstructed,
and neither can a Deflated Sharpe Ratio or a Probability of Backtest Overfitting
that corrects for it.

The requested 20% CAGR and approximately 2.0 Sharpe are aspirations, not
optimization objectives. Parameter search against those numbers is prohibited.
At the incumbent's observed volatility, that joint target requires a genuinely
new orthogonal return source rather than leverage or a favorable resample.

The bounded 9.3% volatility-target sensitivity is a rejection diagnostic, not
a candidate specification. It increased exposure and limit pressure without a
durable Sharpe improvement, did not establish the joint aspiration in full and
later reused history, and left too little room beneath the 15% drawdown policy.
Daily stationary-block paths showed materially greater drawdown-breach risk.
Because this sensitivity is not a canonical v3.2.1 output, its point estimates
must not be copied into committee exhibits or mistaken for a validated
frontier. Any future risk-target change is a capital-policy decision made only
after post-freeze evidence; it is never an alpha result.

Two things have changed since the paragraphs above were first written, and
neither softens them.

**The estimators now run.** `delta1_strategy.research.validation` computes CSCV/
PBO over a family *declared today*: 0.42 on the monthly paths this document
permits, from seventeen configurations. That is a real number about a real
family, and it is not the historical PBO this section says is unreconstructable
— a family declared today is a lower bound on this lineage's search, not a
reconstruction of it. `family_deflated_sharpe` correspondingly returns
`NOT_ESTIMABLE` for every member, and will while the trial count is
unrecoverable. An earlier revision returned `ESTIMATED` at ≈0.9999 for all
seventeen, which cleared the 95% gate for every configuration including the
worst; a statistic that passes everything is not a gate.

**Out-of-sample records exist, and none of them is a holdout.** Three, with
their limitations attached, because the limitations are what make them citable:

| Record | Artifacts | Span | Out of sample with respect to |
|---|---|---|---|
| 2015–2016 futures subset | `outputs/holdout/` | 522 sessions, 12 of 59 roots | bytes the canonical source manifest proves the pipeline never read — but trend sleeve only, no basis sleeve, no equity/FX/energy/ags exposure, and far too short to resolve a Sharpe difference of the size the gates below require. The append-only ledger refuses a second scoring |
| Stitched futures walk-forward | `outputs/validation/` | 1995-01-02 → 2014-12-31, 5,218 sessions (20.7 252-session-equivalent years; 20.0 elapsed calendar years), 20 pairwise-disjoint folds, zero double counting | the **selector only**. The specification being replayed was written with this window already read, so the segments are not out of sample with respect to specification choice |
| ETF regime-allocation sleeve | `outputs/etf/` | 2009-01-02 → 2018-12-31, 2,516 sessions, 10 complete calendar years, including a contiguous sealed block 2014-01-02 → 2018-12-31 | the selector throughout, and the development replay's custody over the sealed block (byte-identical replay, maximum absolute difference 0.0 over 1,990 sessions); later sealed folds fit on earlier sealed sessions. A survivors-only panel; the sealed block holds no full equity bear market, so full-bear conditional performance is not estimable, although the candidates' own defensive gates did operate; the universe rule and cost model were written after the whole panel had been read |

None of the three is post-freeze, independently custodied, or prospective.
"Genuine holdout and forward evidence" below is unaffected by all of them.

## Pre-registration and append-only trial registry

Before calculating candidate returns, an independent reviewer must timestamp
and approve a content-addressed protocol containing:

- economic mechanism, expected sign and exact timestamped inputs;
- formula, transformations, missing-data rules and finite parameter grid;
- point-in-time universe and effective-dated contract specifications;
- incumbent, risk budget, execution timing, costs and capacity limits;
- fold plan, purge/embargo, primary statistic, multiplicity method and seeds;
- promotion/rejection rules and code, configuration, source and environment
  hashes; and
- the permitted process for correcting a genuine implementation defect.

Each batch is limited to three economically distinct challengers. Every
parameter, combination, ablation, universe change, retry and post-result fix is
a separate trial. Failed and unfavorable trials remain in an append-only
registry with their complete synchronized net-return path hash.

`delta1_strategy.research.registry.ResearchTrialRegistry` implements the local
hash-linked, append-only record and defaults to the three-candidate batch
limit. It freezes candidate formulas, configuration/source fingerprints,
windows, costs, risk budgets and planned metrics before linking one immutable
result. This is enforcement plumbing only: independent timestamping and
custody are still required, and the registry cannot reconstruct omitted legacy
trials or turn reused history into a holdout.

## Causality and state

At decision time, every input must have a publication or venue timestamp no
later than that decision. Scaling, winsorization, covariance, liquidity and
universe membership are fitted only on information available then. Prefix and
truncation-invariance tests are mandatory, and
`tests/test_strategy.py::test_full_pipeline_is_truncation_invariant` is the one
that enforces it. [`causality.md`](causality.md) states the conventions and the
proofs that back them.

The strategy has long, path-dependent state: signal windows, volatility state,
the no-trade buffer, integer positions and NAV. A validation fold must replay
chronologically from an earlier warm-up and carry the actual book state.
Restarted fold NAVs or independently spliced optimized paths are invalid.

That rule is now executable rather than declarative.
`research.validation.anchored_walk_forward` uses splice-then-simulate-once: each
candidate's decision frame is built over the whole history, the selected regimes
are spliced into one frame, and `_simulate_execution` runs a single time, so
NAV, the executed book, the roll backlog and the cost ledger cross every fold
boundary intact and switching turnover is charged. It is pinned two ways — with
a constant parameter the spliced replay reproduces the single-shot backtest to
1e-12, and a test asserts a fold inherits the previous regime's book rather than
reproducing the incoming candidate's standalone path, which is precisely the
flattering error a per-fold NAV restart produces.

## Validation design

Anchored expanding walk-forward analysis is the primary diagnostic on reused
history. Its results remain retrospective. If parameters must be selected,
selection occurs only inside each training span; the stitched selector is
itself one registered strategy and includes switching turnover.

Ordinary random cross-validation is invalid for this strategy. Combinatorial
purged cross-validation is not the primary estimator because training on dates
after a test block conflicts with the long feature/state memory and would
require an impractically long embargo. CSCV/PBO may be used only as a secondary
selection-instability diagnostic on the complete matrix of synchronized,
fully causal monthly paths for every registered variant. If loser paths or the
full family are missing, PBO is **not estimable**, never PASS.

The primary endpoint is challenger-minus-incumbent net return at the same
ex-ante risk. Joint stationary-block bootstrap inference must use the entire
registered family. Promotion requires a family-wise max-statistic procedure
such as White's Reality Check or Hansen SPA, a 95% lower confidence bound above
zero for the primary improvement, and Deflated Sharpe probability of at least
95% under the conservative raw trial count when it is estimable.

### These estimators now exist in code

`delta1_strategy.research.validation` implements all three, so the gates above
are executable rather than aspirational: `anchored_walk_forward`,
`reality_check` with `hansen_spa` under all three recentrings, and `cscv_pbo`
with a mandatory refusal path. `assemble_family` is the object that feeds them,
and it validates that every member's path is synchronized on one common index
before any statistic is computed.

Four consequences of running them, recorded because they constrain what may be
claimed rather than because they are favourable:

- Across seventeen declared configurations **no procedure rejects on Sharpe at
  any block length** — White Reality Check 0.642 to 0.660, Hansen SPA consistent
  0.277 to 0.321. The Hansen SPA annualized-mean rows reject at the resolution
  floor, while White Reality Check does not (0.087 to 0.106). The SPA result is
  driven by a member raising the volatility target from 7% to 8%, lifting the
  annualized arithmetic mean by 1.89 percentage points on a path correlated
  0.994 with the incumbent — which is why the endpoint is specified at matched
  ex-ante risk.
- On the monthly paths this document permits, **PBO is 0.42**, near the 0.5
  signature of pure overfitting. The daily matrix gives 0.073 and is emitted only
  as an explicitly labelled secondary estimate; quote 0.42.
- `family_deflated_sharpe` always returns `NOT_ESTIMABLE`. A family declared
  today is a lower bound on this lineage's search, not its trial count, and an
  earlier revision that returned `ESTIMATED` cleared the 95% gate for every
  configuration including the worst. A statistic that passes everything is not a
  gate.
- The refusal path fires on data already in the repository rather than only in a
  unit test. Fed the four-variant lever sweep in `outputs/levers`, CSCV returns
  `NOT_ESTIMABLE` for all four statistics — four configurations quantise the
  out-of-sample rank in steps of 1/5, which measures the grid rather than the
  overfitting, and the floor is ten. That is this section's "not estimable,
  never PASS" enforced by code rather than by the reader.

The walk-forward's own result is the sharpest argument for the freeze rule in
the section above: selecting one parameter out of sample across twenty annual
folds cost 0.165 Sharpe and 6.3 percentage points of maximum drawdown against
leaving the specification alone — through the 15% drawdown policy. Only the
1995 fold selects a non-baseline variant; because the replay carries book and
NAV state across boundaries, its effects propagate and the full gap cannot be
assigned to that fold alone.

### An external reference point now exists

`delta1_strategy.research.benchmarks` replicates published trend-following rules
on the identical panel through the same engine and all common execution and
cost assumptions; rule-specific gross-cap departures are declared and capped
MOP is reported separately. That is the comparison a descriptive band cannot
supply. Its bearing on this document is narrow and worth stating precisely.

The incumbent's spanning alpha against four published rules plus a long-only
equal-risk control is 4.6405% a year at
HAC *t* = 6.04 with R² 0.735, and the only significant loading is
Moskowitz-Ooi-Pedersen TSMOM at 0.509. **That figure is an optimistically biased
reused-history estimate, not an unbiased estimate of edge.** The benchmark
rules ran cold and unrefitted on this panel, while the incumbent's parameters
were chosen with it visible, so the comparison is biased in the incumbent's
favour in expectation by an amount that is unmeasurable here because the trial
count is unrecoverable. It is the same
unrecoverable quantity that makes the Deflated Sharpe gate refuse.

MOP TSMOM earns *more* CAGR than the incumbent at 1.55x the volatility, so what
this replication supports is that the incumbent's advantage is risk control
rather than signal. A promotion claim needs the family-wise gates above, and
those do not reject.

Relevant primary references:

- [Time-Series Momentum](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf)
- [Basis-Momentum](https://doi.org/10.1111/jofi.12738)
- [The Stationary Bootstrap](https://doi.org/10.1080/01621459.1994.10476870)
- [The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [The Probability of Backtest Overfitting](https://scholarworks.wmich.edu/math_pubs/42/)
- [White's Reality Check](https://doi.org/10.1111/1468-0262.00152)

## Conjunctive promotion gates

A challenger must pass every gate after costs at the unchanged 7% risk budget:

- adjusted pooled improvement of at least +0.10 HAC Sharpe and +1 percentage
  point excess-return CAGR versus the incumbent;
- non-worse Sortino, positive expectancy and contribution and USD trade profit
  factors of at least 1.5;
- historical MDD no greater than 15% and no more than 2 percentage points worse
  than the incumbent, with daily block-bootstrap breach risk disclosed;
- positive economics at twice calibrated p75 costs, including both roll legs;
- no gain driven by leverage, omitted funding or continuous-roll artifacts;
- gross, participation, effective-dated margin, delivery and concentration
  controls pass; and
- deterministic replay, ledger reconciliation, artifact binding, broker
  workflow and prospective shadow evidence pass.

The leverage clause is not hypothetical. It is exactly what separates the
Hansen SPA results above: the same seventeen configurations reject on the
annualized mean and fail to reject on Sharpe, because one member simply raises
the volatility target on a path correlated 0.994 with the incumbent. White
Reality Check does not reject either endpoint. A mean-based SPA gate would have
promoted a rescale.

Two clauses are also now known to be unreachable by the position-magnitude
family. The drawdown clause cannot be satisfied by tightening
`max_risk_scalar`, `min_risk_scalar`, `signal_cap`, `risk_managed_cap` or
`shock_floor`: at matched realized volatility, 49 of 50 swept settings publish
`not_estimable` for drawdown-per-unit-volatility against a solver-jitter floor
of 0.0588, and tightening `max_risk_scalar` to 1.00 *raises* bootstrap
P(drawdown > 15%) from 4.90% to 5.95%. The Sharpe clause is not reachable by
them either: the one setting that resolves, `shock_floor` 0.75 → 0.25, is
+0.0225 with a 90% interval of [0.0034, 0.0416] — an order of magnitude below
the +0.10 gate, and a Sharpe result rather than a drawdown one. Recorded in
[`drawdown-attribution-findings.md`](drawdown-attribution-findings.md).

The next legitimate candidates require new inputs: direct front/second-expiry
curve data for executable basis momentum, publication-lagged positioning data,
or a point-in-time breadth expansion chosen for economic uniqueness and
capacity rather than realized returns.

## Genuine holdout and forward evidence

Freeze exactly one challenger before an independent custodian exposes new
post-2014 serial-contract history. That one-time evaluation is an independent
historical holdout only if the research team genuinely never saw it. Any model
change consumes the holdout and creates a new lineage.

Forward data arriving after the freeze are the prospective record. The minimum
track length must be precomputed and must cover at least 12 months and two
eligible delivery cycles per active root. No early stopping or repeated
peeking is allowed without a pre-registered alpha-spending rule. A successful
holdout still does not authorize deployment until every operational gate passes.

None of the three records in "Current evidence status" is that holdout, and each
falls short in a different way. The 2015–2016 subset is genuinely unread bytes
but is trend-only on twelve roots and has already been scored once, so it is
spent as a subset consistency diagnostic and its ledger refuses a second look —
rescoring the same continuation would be peeking, not evidence. The
stitched walk-forward is out of sample with respect to a selector, not with
respect to the specification, and no amount of stitching converts reused history
into unseen history. The ETF sealed block comes closest structurally — a
contiguous five years, custody proven by a byte-identical replay — and is still
not it: the panel is survivors-only, the fitting decisions around the block were
made by someone who had read the whole panel, and the sleeve's defining market
state never occurred inside it. Its honest use is as a demonstration that the
validation machinery returns an unflattering answer when one is deserved.

Because the two futures records here were scored on data the team can re-open,
the append-only ledger and the pre-registration rules above remain the binding
constraint, not the sample size.
