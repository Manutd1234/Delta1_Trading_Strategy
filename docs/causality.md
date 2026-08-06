# Causality Conventions

Most backtests that look too good are not fraudulent; they leak. A signal is
computed from a bar that would not have closed yet, a position is sized from a
volume that had not printed, a universe is filtered on membership nobody knew
at the time. Each leak is individually small and individually invisible in the
equity curve.

This document is the single statement of when this repository believes it knows
something. It is written for someone adapting the engine to their own data, and
the conventions matter more than the strategy they happen to serve.

## The timeline

```
   session t-1                    session t                     session t+1
   ─────────────┬───────────────────────┬───────────────────────┬──────────
   close        │                       │                       │
   ├─ prices, volumes, delivery labels complete for t-1
   ├─ forecast computed from data through t-1
   ├─ NAV known
   └─ if t-1 is a business month end:
        target sized here ────────────► order queued
                                        │
                                        ├─ order becomes eligible
                                        ├─ fill capped by:
                                        │    lagged 60-session median volume  (decided at t-1)
                                        │    this session's realized volume   (market response)
                                        ├─ fills at t's close
                                        └─ residual, if any, carries ─────────► retried
```

The rule the whole engine turns on: **a decision uses only completed bars, and
is executed no earlier than the next session.** A decision made from session
`t-1`'s close can never be filled at `t-1`'s price.

## Signal lag arithmetic

`risk_managed_forecast` ([strategy.py:570](../src/delta1_strategy/research/strategy.py#L570))
is where the shift arithmetic is justified, and it is worth reading before
changing anything.

| Fill convention | Forecast shift | Return it earns |
|---|---|---|
| Full-bar (`next_close`, no opens) | `shift(2)` | `prices.diff()`, the whole close-to-close move |
| Split (opens available) | `shift(2)` and `shift(1)` | `shift(2)` earns the overnight leg, `shift(1)` the intraday leg |

Why two shifts rather than one. A forecast computed after session `t-1`'s close
cannot earn `t-1`'s move — that already happened. With observed opens the
position is established at `t`'s open, so it earns `t`'s open-to-close move,
while the *previous* forecast is what was on the book through the overnight gap
from `t-1`'s close to `t`'s open. Collapsing that into a single shift either
credits a forecast with a move it could not have captured, or drops a leg the
book genuinely held.

The volatility estimate used for sizing is shifted identically. A position
scaled by a volatility that includes the day being traded is the same leak in a
different costume.

## What may condition on the executing session

This is the subtlest rule in the engine and the one most easily got wrong in
either direction.

**A decision may not use the executing session.** Order size is computed at the
decision date from NAV, valuation prices, and lagged volume, and is never
revised. Sizing from the session's completed volume would let the book trade
more precisely when liquidity turned out well — an *optimistic* look-ahead.
`tests/test_strategy.py` pins this: realized volume must never enlarge a fill.

**The market's response may.** The fill price is the executing session's close.
Whether a market traded at all is its volume being positive. The impact charge
divides by that session's volume. And the fill is capped at the participation
limit of that same volume, with the residual deferred.

The property that separates the two is monotonicity: filled quantity is
non-decreasing in realized volume and never exceeds what was requested. Depth
can only take away. A rule with that shape is a market-response model; a rule
without it is a look-ahead.

## Capacity, in both directions

Two bounds apply to every fill, and they are not redundant:

| Bound | Information set | What it represents |
|---|---|---|
| `rebalance_capacity` | sessions `t-60 … t-1` | the largest order the pre-session evidence justifies |
| `session_capacity` | session `t` only | what the session could actually absorb |

Roll turnover obeys the same limit. A delivery-label change obliges the book to
transfer every contract held through it — two contracts of turnover each — and
that obligation is tracked as a quantity (`roll_backlog_contracts`) rather than
a flag. The distinction matters because the position transfers whether or not
the turnover is charged, so treating the roll as skippable would hand out an
uncosted transfer. Slices are priced against the parent roll's participation:
impact grows with the square root of participation, so per-slice pricing would
make an order cheaper simply by being cut up, and this model prices neither the
permanent impact nor the delay cost that a real desk pays for that split.

## Universe and state

Within the flagship futures engine and its anchored walk-forward replay,
anything fitted — scaling, volatility, correlation, liquidity screens, universe
membership — is fitted only on information available at the decision date. Folds
replay chronologically carrying real book state; a fold that restarts NAV or
splices an optimized path is not evidence. This statement does **not** extend to
the separate ETF study: its universe and liquidity rule was written after the
whole 2006–2018 panel had been read, including the nominally sealed block, as
disclosed in [etf-regime-allocation-findings.md](etf-regime-allocation-findings.md).
See [research-methodology.md](research-methodology.md) for the governance form
of this rule.

`research.validation.anchored_walk_forward` is that rule in code. It builds each
candidate's decision frame over the whole history, splices the selected regimes
into one frame, and calls `_simulate_execution` exactly once, so NAV, the
executed book, the roll backlog and the cost ledger cross every fold boundary
intact and switching turnover is charged. The alternative — running each fold
standalone and concatenating — is the flattering error, because a fold that
starts flat never pays to get out of the previous fold's book.

## The proof

Conventions are claims. This one is tested:
`test_full_pipeline_is_truncation_invariant` re-runs the engine on history
truncated at 2004-12-31 and asserts the returns, positions and signals are
byte-identical to the corresponding prefix of the full run.

That is the property leakage breaks. If any calculation could see the future —
a full-sample normalization, a centred rolling window, a universe filtered on
end-of-sample membership — the truncated run would differ, because the future it
peeked at is no longer there.

If you adapt this engine, keep that test. It is worth more than the rest of this
document.

Three later checks are the same idea applied to seams rather than to the engine,
and each is an identity that a leak would break:

| Check | Assertion | What a failure would mean |
|---|---|---|
| Benchmark execution seam (`outputs/benchmarks/benchmark_seam_check.csv`) | replaying the incumbent's own decision frame through the benchmark execution path reproduces the canonical ledger with maximum absolute daily net-return deviation exactly **0.0** | the benchmarks are not being costed and executed on the incumbent's terms, so any comparison between them is a comparison of two ledgers |
| Walk-forward splice identity | with a constant parameter, the spliced replay reproduces the single-shot backtest to **1e-12** | fold boundaries are creating or destroying P&L, which is where a restarted-NAV artefact would hide |
| ETF custody replay (`outputs/etf/etf_holdout_custody.csv`) | the development replay, reloaded from disk with the loader's last-session guard armed, is byte-identical over 1,990 sessions with maximum absolute difference **0.0** | a sealed block was read during development, which would make "out of sample" a claim about intent rather than about execution |

None of the three is a substitute for truncation invariance. Truncation
invariance tests whether the engine can see the future at all; these test whether
a study wired around the engine has quietly changed what it is measuring. Both
failure modes exist and they are not the same failure.
