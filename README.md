# DELTA1 Production Candidate

This repository contains one research implementation of a 61-market global
futures strategy combining 12-month time-series momentum with basis momentum.
The v2.8 engine adds realistic, parameterized trading frictions, integer
contracts, trade-episode attribution, portfolio exposure limits, and
fail-closed operational controls.

> **Status: NOT PRODUCTION READY.** The historical simulation is a
> production-candidate research package, not an executable trading system.
> `StrategyConfig(mode="production")` deliberately refuses to run. Capital
> deployment remains blocked until every external-evidence and operational
> gate described below is independently satisfied.

## Research status

The investable ledger launches on 1990-01-01 with $1 million and zero
positions. Earlier data warm causal signals and risk estimates only. The
strategy was selected retrospectively, every reported period has been reused,
and the supplied futures history ends in 2014. Consequently:

- 1990-2004 is development history;
- 2005-2014 is a reused later diagnostic, not an independent holdout;
- 1990-2014 is the complete post-launch historical simulation;
- no reported metric is evidence of forward performance; and
- block-bootstrap intervals describe sampling uncertainty but do not correct
  for specification selection.

The current intended use, evidence status, limitations, and launch blockers
are summarized in [MODEL_CARD.md](MODEL_CARD.md).

## Strategy and ledger

- 61 futures across equity indices, government bonds, FX, energy, metals, and
  agriculture/livestock.
- Equal blend of a 12-month sign-trend forecast and the year-on-year change in
  realized roll yield.
- Causal market-level risk scaling, equal pre-forecast volatility budgets,
  trailing-volume eligibility, a volatility-shock taper, and a 10% annualized
  portfolio risk objective.
- Monthly decisions with a 25% no-trade region.
- A decision made after a month-end close becomes a pending order. Contract
  quantity is fixed using decision-date NAV and can fill only on a later valid,
  positive-volume session.
- Canonical fills occur at the next eligible close. New contracts earn no P&L
  before that close; `execution_delay_sessions` can add further eligible-session
  delay.
- Positions are actual integer contracts in a self-financing USD NAV ledger.
  P&L uses back-adjusted differences, while economic notional uses unadjusted
  active-contract prices.
- Every observed vendor delivery-month change is charged as a two-leg roll
  when the book has exposure. Suspicious labels are flagged for review but do
  not suppress costs.
- Missing held settlement, FX, cost, notional, or margin inputs fail the run
  instead of silently creating zero P&L.

Continuous contracts still cannot identify the exact expiring and deferred
instruments needed for real orders. The roll model is therefore an accounting
approximation, not proof of executable roll capacity. CME describes the
underlying mechanics in its official guidance on
[daily mark-to-market](https://www.cmegroup.com/education/courses/introduction-to-futures/mark-to-market)
and [contract expiration and rolls](https://www.cmegroup.com/education/courses/introduction-to-futures/understanding-futures-expiration-contract-roll).

## Parameterized frictions and capacity

The canonical research configuration charges each one-way contract leg for:

- 0.50 tick of half-spread;
- 0.25 tick of additional slippage;
- $2.50 commission;
- $1.50 exchange and regulatory fees; and
- a square-root market-impact estimate parameterized at 10 basis points at full
  reported-volume participation.

For a traded contract quantity `q`, the engine calculates fixed cost from the
configured tick value and per-contract fees. Its impact estimate scales with
absolute unadjusted notional and `sqrt(q / reported_volume)`. These parameters
are transparent and stressable, but they have not been calibrated to
timestamped quotes, order size, venue, contract expiry, or broker fills. CME's
[liquidity guidance](https://www.cmegroup.com/education/articles-and-reports/how-traders-measure-liquidity)
explains why volume alone is not a complete execution-quality measure.

Normal rebalance orders are limited to 2% of the smaller of lagged median
volume and current reported volume. Unfinished orders remain pending. Contract
rolls are costed but cannot be capacity-capped honestly without serial-expiry
liquidity, so roll participation remains a critical readiness blocker.

The engine also applies portfolio-level limits before rounding contract
targets and checks them again after rounding:

| Control | Research intent constraint | Behavior |
|---|---:|---|
| Rebalance participation | 2% of eligible reported volume | Partial fill; remainder stays pending |
| Gross notional | 5.0x decision NAV | Scale target portfolio down |
| Static margin diagnostic | 30% of decision NAV | Scale target portfolio down |
| Additional execution delay | 0 eligible sessions | Configurable friction stress |

The margin input is a static catalogue snapshot rather than a historical or
live portfolio-margin calculation. Passing the research limit does not prove
that a clearing broker would finance the portfolio. CME notes that margin
requirements can change as market conditions change in its official
[margin guidance](https://www.cmegroup.com/education/articles-and-reports/understanding-margin-changes).

## Extended performance and trade episodes

`performance_metrics` reports return and operational diagnostics descriptively.
The report includes:

- CAGR, annualized volatility, daily/monthly/autocorrelation-robust Sharpe,
  Sortino, downside deviation, Calmar, daily profit factor, win rates, skew,
  excess kurtosis, historical VaR and CVaR;
- maximum drawdown and duration, best/worst day and month, worst rolling
  5- and 21-session returns;
- fixed, impact, and total annualized cost drag;
- risk scalar, rolling realized volatility, gross notional, static margin,
  participation, pending markets, limit-binding days, and markets held; and
- closed/open episode counts, win rate, contribution- and USD-based profit
  factor, expectancy, win/loss ratios, and holding periods.

A trade episode is a contiguous same-direction position in one market.
Same-sign resizes remain inside the episode; a sign flip closes one episode
and opens another; contract rolls contribute cost but do not create a new
alpha trade. Open episodes are retained as censored observations. Each episode
records entry/exit, direction, contract scale, holding time, gross P&L,
regular and roll costs, net P&L, contribution, MFE/MAE contribution, resize
count, and roll count.

The executed notebook displays episode summaries, asset-class attribution,
outcome concentration, duration, and best/worst closed episodes. Episode
statistics remain accounting diagnostics: overlapping markets, correlated
risk, and censored open episodes prevent interpreting episode counts as
independent observations.

`diagnostics.py` adds three deterministic report surfaces without importing or
reimplementing the strategy:

- `trade_metrics_report` summarizes closed-by-exit-date episodes overall and
  by asset class and symbol, with low-sample flags;
- `causal_regime_report` labels volatility, forecast-magnitude, cross-market
  correlation, and liquidity regimes using expanding terciles based only on
  preceding months; and
- `monthly_stationary_bootstrap_summary` simulates monthly paths across
  multiple block lengths and horizons, reporting quantiles for CAGR, monthly
  Sharpe, maximum drawdown, worst 12-month return, terminal-loss probability,
  and drawdown-exceedance probabilities.

Regime and bootstrap reports are descriptive and selection-unadjusted.
Discontiguous regime observations do not receive synthetic CAGR or drawdown
statistics.

`stress.py` exposes the canonical cost assumptions and a seven-scenario
friction suite: baseline, doubled fixed costs, doubled impact, all execution
costs doubled, one additional executable-session delay, a tighter 1%
participation cap, and a combined adverse case. Every scenario reruns the same
reused history and is labeled as retrospective sensitivity, not validation.

## Fail-closed production controls

[production.py](production.py) is deliberately independent of the research
module. It accepts plain pandas objects and exposes four control surfaces:

- `build_order_intents`: deterministic, idempotent order intents with integer
  validation, participation caps, portfolio gross/margin gates, and
  reduce-only behavior during a breach;
- `evaluate_runtime_health`: live checks for data freshness, NAV, drawdown,
  gross notional, margin, broker connection/reconciliation, monitoring, and a
  tested kill switch;
- `production_readiness_report`: historical integrity gates plus explicit
  external-evidence gates; and
- `overall_readiness_status`: returns `READY` only if every critical gate
  explicitly passes, otherwise `BLOCKED`.

Independent production defaults are intentionally conservative:

| Live/order control | Default limit |
|---|---:|
| Order participation | 2% |
| Gross notional | 6.0x NAV |
| Margin requirement | 35% of NAV |
| Drawdown | 15% from supplied peak NAV |
| Market-data age | 300 seconds |

Missing, malformed, non-finite, stale, or unreconciled inputs never approve a
risk-increasing order. A research backtest cannot set external evidence to
true by itself.

## Readiness blockers

Production remains blocked until all of the following exist and are tested:

1. Tradeable serial-contract history and an expiry-specific roll policy,
   including first-notice and last-trade safeguards.
2. Timestamped, time-zoned market data and session calendars for every venue.
3. Point-in-time contract specifications, settlement FX, margin schedules,
   commissions, exchange fees, and regulatory fees.
4. A cost and market-impact model calibrated to executable quotes and intended
   order sizes, including serial-expiry liquidity.
5. A point-in-time universe and security master without survivorship or
   present-day availability leakage.
6. Portfolio margin, variation-margin cash, collateral yield, funding, and
   liquidity controls under stressed conditions.
7. Genuinely independent post-2014 evaluation and a documented model-change
   protocol.
8. Forward paper trading with order, fill, rejection, and reconciliation logs.
9. A tested broker adapter and execution policy (for example, limit or
   participation/TWAP/VWAP logic) with idempotency, retry, duplicate-order,
   partial-fill, cancel/replace, and restart handling.
10. Broker/clearing reconciliation, monitoring and alerting, incident response,
    and a tested kill switch.
11. Compliance, market-access, position-limit, delivery, model-risk, and
    change-approval review appropriate to the trading entity and venues.

The default `ReadinessEvidence()` sets every external item to false, so the
repository's readiness result is `BLOCKED`. This is intentional. The CFTC's
[hypothetical-results warning](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/fraudadv_tradingsystem.html)
applies directly to the remaining execution, margin, and selection gaps.

## Operating and behavioral fit

This is an end-of-day, monthly-decision process, not an intraday or low-latency
strategy. An operator still needs coverage for the next eligible execution
session, contract rolls, rejected or partial orders, daily reconciliation, and
alerts. The historical episode win rate is near one half, and long drawdowns
and losing sequences remain normal even when long-run expectancy is positive.

Capital and governance therefore need to tolerate the documented drawdown and
the possibility of worse unobserved outcomes. Operators must not override
signals during stress: any discretionary intervention, parameter change, or
drawdown restart requires an incident record, approval, a new configuration
hash, and prospective evaluation. The independent 15% live drawdown gate is a
safety stop, not evidence that losses cannot exceed 15% before positions can be
reduced.

## Notebook

[DELTA1_Strategy.ipynb](DELTA1_Strategy.ipynb) is a thin client over
`strategy.py` and `production.py`. It runs the canonical pipeline once and
then presents:

1. blocked production status and scope;
2. reproducible configuration and launch checks;
3. the canonical execution and friction contract;
4. extended metrics, NAV, drawdown, and cost attribution;
5. trade episodes and contribution concentration;
6. the seven-scenario friction/capacity stress suite;
7. causal lagged regime diagnostics;
8. stationary block-bootstrap path uncertainty;
9. historical hard-control usage and breaches;
10. the fail-closed readiness checklist; and
11. reproduction and artifact notes.

The notebook imports `stress.py` and `diagnostics.py`; it does not duplicate
signal, sizing, execution, cost, regime, or bootstrap logic.

## Reproduce

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"

export DELTA1_DATA_DIR="/path/to/Round1AllData/Quant Researcher/Delta1"

# Canonical artifacts
delta1-strategy --data-dir "$DELTA1_DATA_DIR" --output-dir outputs

# Unit and supplied-data integration tests
python -m unittest discover -s tests -v

# Interactive review
jupyter lab DELTA1_Strategy.ipynb

# Deterministic clean-kernel notebook execution
jupyter nbconvert --to notebook --execute --inplace DELTA1_Strategy.ipynb \
  --ExecutePreprocessor.timeout=600
```

If `DELTA1_DATA_DIR` is unset, the notebook falls back to the supplied
repository-local `Round1AllData/Quant Researcher/Delta1` directory.

## Artifacts

The canonical CLI writes a reproducible package:

- `strategy_metrics.csv` and `strategy_daily.csv`: extended reused-history
  metrics and the portfolio ledger;
- `strategy_monthly_position_intents_per_dollar.csv`: buffered research
  intents before NAV scaling, intent constraints, rounding, and fill capacity;
- `strategy_market_daily.csv.gz` and `strategy_execution_events.csv`: the
  reconciling market-level ledger and its trade/roll event subset;
- `strategy_trade_episodes.csv` and `strategy_trade_metrics.csv`: detailed
  directional episodes and grouped episode diagnostics;
- `strategy_regime_metrics.csv` and `strategy_monte_carlo_summary.csv`: causal
  regime and stationary-bootstrap reports;
- `strategy_friction_stress.csv`: the seven-scenario retrospective stress
  suite, unless `--skip-stress` is supplied;
- `production_readiness.csv`, `strategy_ledger_checks.csv`, and
  `strategy_data_quality.csv`: fail-closed readiness, reconciliation, and
  market-level quality checks;
- `cost_model_assumptions.csv` and `source_manifest.csv`: explicit friction
  assumptions and SHA-256 hashes of every consumed source file; and
- `strategy_config.json` and `run_manifest.json`: normalized configuration,
  source/config fingerprints, engine version, and research/readiness status.

None of these artifacts proves live tradability. The notebook does not
overwrite `outputs/`.

## License

See [LICENSE](LICENSE).
