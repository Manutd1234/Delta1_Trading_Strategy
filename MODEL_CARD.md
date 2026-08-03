# DELTA1 production-candidate model card

## Status and intended use

This repository contains one frozen research strategy: a diversified global
futures time-series momentum and basis-momentum portfolio. It is suitable for
reproducible research, code review, sensitivity analysis, and paper-trading
integration work.

It is **not approved for live capital**. `production.py` defaults every item of
external evidence to absent and returns `BLOCKED` until all critical gates pass.
A strong historical statistic cannot override an operational or data blocker.

## Strategy and decision timing

- Universe: 61 equity-index, government-bond, currency, energy, metals, and
  agriculture/livestock futures markets in the supplied catalogue.
- Forecast: equal blend of a 252-session sign trend and a causal change in
  trailing roll yield, with trend-only fallback while basis is unavailable.
- Risk: causal per-market volatility sizing, a volatility-shock taper,
  portfolio EWMA volatility scaling, equal market risk budgets, and integer
  contracts.
- Schedule: month-end decisions fill on the next observed positive-volume
  session close. An optional extra-session delay is available for sensitivity
  analysis.
- Launch: zero positions and USD 1,000,000 NAV at 1990-01-01. Earlier data may
  warm causal estimates but cannot earn P&L.
- Trade definition: one continuous non-zero directional position in one market.
  Same-sign resizes and rolls stay in the episode; a sign flip closes one episode
  and opens another. Open episodes are censored.

## Costs, capacity, and risk controls

Each execution charges the configured half-spread, slippage, commission,
exchange/regulatory fees, and square-root market impact. Continuous-contract
rolls incur an approximate two-leg cost. These are transparent assumptions,
not calibrated forecasts of future fills.

Research position intents are buffered to 5x gross notional and 30% static
margin. Regular rebalances are capped at 2% of trailing median volume. The
independent live safety layer uses 6x gross, 35% margin, 2% order participation,
and a 15% drawdown gate. Missing or stale inputs, unreconciled broker state,
unhealthy monitoring, or an untested kill switch block new risk. Risk-reducing
orders may still be allowed.

There is no per-trade stop-loss or take-profit overlay. This portfolio manages
risk through volatility sizing, diversification, shock tapering, exposure
limits, and the separate live drawdown kill gate. Adding an optimized stop rule
would create a different strategy and would require a newly frozen validation
protocol.

## Reported evidence

The package reports, without return-goal verdicts:

- CAGR; daily, monthly, and HAC Sharpe; Sortino; volatility; Calmar; tail loss;
  maximum drawdown and duration;
- net directional-episode profit factor, expectancy, win rate, average win,
  average loss, payoff ratio, and holding period;
- cost drag, notional, margin, rebalance participation, roll-participation proxy,
  pending orders, and limit-binding days;
- causal expanding-threshold regime diagnostics;
- deterministic stationary monthly-block bootstrap diagnostics that are
  explicitly not selection-adjusted;
- adverse cost, delay, and capacity reruns;
- source hashes and numerical ledger reconciliations.

All historical windows are labelled `retrospective_reused_history`. The model
was developed after inspecting the available history, historical candidate
return paths are incomplete, and no historical slice is represented as a
genuine untouched holdout.

## Material limitations and launch blockers

1. The files are continuous back-adjusted research series, not tradeable serial
   contracts with executable bid/ask quotes and timestamps.
2. Contract specifications and margins are catalogue snapshots rather than
   fully dated histories. Fee schedules and market impact are not venue- and
   broker-calibrated.
3. Roll execution is inferred from vendor delivery labels. The reported roll
   participation is a diagnostic proxy and can expose implausible capacity.
4. Data provenance does not independently establish point-in-time membership,
   survivorship-bias freedom, or complete corporate-action handling upstream.
5. Foreign-currency conversion uses the supplied currency-futures observations;
   a live implementation needs synchronized FX and cash/funding accounting.
6. The close-fill convention omits intraday queue position, rejected/partial
   broker fills beyond the volume cap, exchange outages, funding interest,
   collateral return, taxes, and variation-margin cash operations.
7. No broker/OMS adapter, idempotent order-state machine, live reconciliation,
   alerts, disaster recovery, or tested kill-switch deployment is included.
8. No independent forward paper record or live shadow record exists in the
   repository.

The generated `outputs/production_readiness.csv` is the authoritative checklist.
Live deployment remains blocked until every critical item has documented,
independently reviewed evidence.

## Operational references

- [CFTC pre-trade risk controls](https://www.cftc.gov/LawRegulation/FederalRegister/finalrules/2013-22185.html)
- [CFTC system safeguards and automated trading controls](https://www.cftc.gov/PressRoom/PressReleases/6683-13)
- [NFA supervision of automated order-routing systems](https://www.nfa.futures.org/rulebooksql/rules.aspx?RuleID=9046&Section=9.)
- [CME: measuring liquidity](https://www.cmegroup.com/education/articles-and-reports/how-traders-measure-liquidity)
- [CME: futures mark-to-market](https://www.cmegroup.com/education/courses/introduction-to-futures/mark-to-market)
- [CME: futures expiration and rolls](https://www.cmegroup.com/education/courses/introduction-to-futures/understanding-futures-expiration-contract-roll)
- [Politis and Romano: stationary bootstrap](https://doi.org/10.1080/01621459.1994.10476870)
