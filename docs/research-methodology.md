# Research methodology and alpha-promotion policy

## Current evidence status

Every supplied observation through 2014-12-31 is retrospective reused history.
The 1990–2004 and 2005–2014 windows are diagnostics, not independent holdouts.
Archived Git history records many earlier signal, universe, risk and leverage
trials, but complete synchronized return paths no longer exist for every trial.
The global trial count and a defensible historical Deflated Sharpe Ratio or
Probability of Backtest Overfitting therefore cannot be reconstructed.

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
truncation-invariance tests are mandatory.

The strategy has long, path-dependent state: signal windows, volatility state,
the no-trade buffer, integer positions and NAV. A validation fold must replay
chronologically from an earlier warm-up and carry the actual book state.
Restarted fold NAVs or independently spliced optimized paths are invalid.

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
