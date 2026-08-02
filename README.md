# DELTA1 Strategy

One strategy, one implementation: a cost-aware **61-market global futures
portfolio combining 12-month time-series momentum and basis momentum**.
The production code is entirely in `strategy.py`; superseded strategies and
rejected research branches have been removed from the runtime.

## Performance target

The requested threshold is **20% CAGR and 2.0 Sharpe**, net of modeled costs.
The optimized strategy clears both thresholds in the requested 1990–2004
window and in each of its two internal subperiods. The target estimator was
fixed as daily net mean/standard deviation, zero risk-free rate, annualized by
√252.

| Fixed window | CAGR | Sharpe (daily, rf=0) | Max drawdown | Both targets met |
|---|---:|---:|---:|:---:|
| 1980–1989 backward check | 26.0% | 2.02 | −15.8% | Yes |
| 1990–1997 discovery | 27.2% | 2.14 | −11.4% | Yes |
| 1998–2004 confirmation | 24.9% | 2.07 | −10.6% | Yes |
| **1990–2004 optimized window** | **26.1%** | **2.11** | **−11.4%** | **Yes** |
| 2005–2014 later stress | 16.4% | 1.37 | −14.8% | No |
| 1980–2014 full history | 23.2% | 1.87 | −15.8% | No |

CAGR uses elapsed calendar time. Sharpe is the daily estimator specified
above; the 1990–2004 monthly-return Sharpe is 1.99 and the 21-day HAC estimate
is 1.93. The 2.11 result must therefore be read as the chosen daily metric,
not a frequency-independent fact.

### What changed

A two-stage search evaluated 50 unique configurations using data truncated at
2004. Nine alpha trials tested three causal trend specifications against three
basis weights; none beat the existing 12-month trend plus 50/50 basis blend
robustly, so no third alpha was added. Forty-one additional trials varied
per-market risk management, risk budgeting, and the execution buffer.

The selected point sits on a target-clearing plateau while retaining the
existing alpha and 25% buffer:

- 63-day inverse strategy-volatility scaling per market, capped at 2×;
- equal nominal pre-forecast volatility budget per available instrument
  instead of equal budget per asset class.

Fifteen of the 50 configurations cleared both targets in both internal
subperiods, so the result is not a single numerical needle. Nevertheless,
1990–2004 has been inspected many times and this is a retrospective fit. The
later stress result is 16.4% / 1.37, not 2.0. Nothing here guarantees future
performance; a durable claim requires new full-universe futures data or live
forward validation.

The complete current-round search is retained in
[optimization_trials.csv](outputs/optimization_trials.csv). The adopted row
has the highest combined-window daily Sharpe in that ledger and ranks fifth on
the more conservative minimum-of-two-subperiods score; it was preferred over
the numerical minimum-score winner because it keeps the existing 25% buffer
and a shorter standard risk window.

### Robustness

All eight one-at-a-time parameter neighbors and a 2× modeled-cost stress still
clear both targets in both subperiods. The construction is less robust to lost
breadth: removing risk-managed sizing fails, restoring six-class budgeting
misses 2.0 in confirmation, and every leave-one-asset-class-out run fails at
least one Sharpe threshold. Dropping agriculture/livestock is the largest hit,
reducing combined Sharpe to 1.74. The full 19-row audit is in
[strategy_robustness.csv](outputs/strategy_robustness.csv).

## Optimized specification

- 61 liquid futures across equity indices, government bonds, FX, energy,
  metals, and agriculture/livestock; non-USD P&L is converted point-in-time.
- Equal blend of 12-month sign trend and the year-on-year change in realized
  roll yield (basis momentum).
- Causal 63-day strategy-volatility scaling, capped at 2×, modifies each
  market forecast; sizing then assigns an equal nominal volatility budget to
  each available instrument.
- Trailing volume gate, 20/120-day volatility-shock taper, 10% portfolio
  volatility target using EWMA λ=0.94, and a 2× portfolio leverage cap.
- Month-end decisions become active the next business day; a 25% no-trade
  region suppresses small changes.
- Net returns deduct a half-tick spread estimate and USD 2.50 commission per
  contract per one-way trade.

The immutable experiment and decision history is retained in
[RESEARCH_HISTORY.md](RESEARCH_HISTORY.md). It documents why this specification
was selected and why the alternatives were rejected; historical module names
there are archival references only.

## Run

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

delta1-strategy \
  --data-dir "/path/to/Round1AllData/Quant Researcher/Delta1" \
  --output-dir outputs
```

For the fully executed research walkthrough:

```bash
pip install -e ".[notebook]"
export DELTA1_DATA_DIR="/path/to/Round1AllData/Quant Researcher/Delta1"
jupyter lab DELTA1_Strategy.ipynb
```

The run writes four auditable artifacts:

- `outputs/strategy_metrics.csv` — fixed-window metrics and target verdicts
- `outputs/strategy_daily.csv` — gross return, cost, net return, and leverage
- `outputs/strategy_monthly_targets.csv` — executable month-end targets
- `outputs/strategy_config.json` — exact configuration with a portable data path

The two research ledgers linked above are retained audit evidence from the
search and robustness runs; the production command refreshes only the four
runtime artifacts.

Run the focused suite with the supplied dataset:

```bash
export DELTA1_DATA_DIR="/path/to/Round1AllData/Quant Researcher/Delta1"
python -m unittest discover -s tests -v
```

## Repository

```text
strategy.py                  complete strategy, accounting, reporting, and CLI
DELTA1_Strategy.ipynb        executable research walkthrough and audit
tests/test_strategy.py       causal, risk, execution, metric, and integration tests
outputs/                     current runtime outputs and research audit ledgers
RESEARCH_HISTORY.md          archived pre-registration and decision log
pyproject.toml               package metadata and the single CLI entrypoint
```

## Limits

The dataset is survivorship-biased and the full futures universe ends in 2014.
Every reported period has now been inspected repeatedly; none is an untouched
holdout. This round adds 50 unique trials. Earlier records contain at least 81
other variant evaluations plus a separate 72-configuration target search; the
cross-round overlap was not reconstructed or deduplicated. The point estimates
are not adjusted for that multiplicity. Modeled costs omit explicit roll
turnover, market impact, exchange fees, margin financing, and capacity. Results
are research estimates, not a promise of future returns or investment advice.
