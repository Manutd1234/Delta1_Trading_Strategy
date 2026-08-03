# DELTA1 Strategy

This repository contains one strategy and one implementation: a 61-market
global futures portfolio combining 12-month time-series momentum with basis
momentum. Version 2.7 corrects the execution, roll, contract, NAV, and metric
errors found in the earlier normalized backtest.

## Audit verdict

The strategy is credible as research, but it is **not production-ready** and
the repository does **not** validate a durable 20% CAGR / 2.0 Sharpe.

Under the canonical simulation launched with $1 million and zero positions on
1990-01-01, the selected 1990-2004 window has a 24.52% CAGR and 1.985 naive
daily Sharpe point estimate. The same returns produce a 1.855 monthly Sharpe
and 1.784 21-lag HAC Sharpe. The CAGR target is
met in this selected sample; the Sharpe target is not. It would be misleading
to round 1.985 into a 2.0 pass.

| Reused/selected window | CAGR | Naive daily Sharpe | Monthly Sharpe | HAC Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| 1990-1997 selected subperiod A | 23.59% | 1.907 | 1.660 | 1.642 | -14.66% |
| 1998-2004 selected subperiod B | 25.59% | 2.075 | 2.134 | 1.988 | -10.21% |
| **1990-2004 selected window** | **24.52%** | **1.985** | **1.855** | **1.784** | **-14.66%** |
| 2005-2014 reused later diagnostic | 14.84% | 1.248 | 1.248 | 1.202 | -16.41% |
| 1990-2014 full post-launch history | 20.56% | 1.686 | 1.615 | 1.551 | -16.41% |

The daily estimator is net mean divided by net standard deviation, with zero
risk-free rate and the requested sqrt(252) convention. The common calendar has
more than 252 rows in many years, so this is a convention rather than a
frequency-neutral estimate.

A deterministic six-month circular block bootstrap gives a selected-window
95% interval of 1.374-2.400 for monthly Sharpe and 17.62%-31.70% for CAGR.
Those intervals are **not adjusted for strategy selection**. The available
history records at least 50 current-round trials, 72 target-search
configurations, and 81 earlier variants, with unknown overlap and without the
full trial return paths needed for an honest multiplicity correction.
[Backtest-overfitting](https://escholarship.org/content/qt4hn4t174/qt4hn4t174.pdf)
and [deflated-Sharpe](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
methods require that missing experiment history rather than a guessed trial
count.

## Corrected implementation

- A target calculated from a month-end close becomes a pending order. It can
  fill only on that market's next raw session with finite prices and positive
  reported volume.
- The canonical fill is the next observed session close, so a target formed
  after a month-end close earns no part of the following close-to-close bar.
  The data have dates but no exchange timestamps; across a global universe,
  some nominal next-day opens may occur before all inputs bearing the prior
  date are known. The 25.63% / 2.052 next-open result is therefore retained
  only as an unverified sensitivity and cannot establish the target.
- The ledger carries actual integer contracts and dollar NAV. Contracts are
  sized from decision-date NAV, frozen through any fill delay, and changed only
  by executed orders; returns no longer imply free daily rescaling with NAV.
- The canonical ledger starts with $1 million and no inherited positions on
  1990-01-01. Earlier observations warm causal signals and risk estimates only.
- Normal rebalances are capped at 5% of trailing median contract volume and
  unfinished orders remain pending.
- Every observed vendor delivery-month switch charges the two one-way legs
  required to exit the old expiry and enter the new one, including suspicious
  reversals. Anomaly flags are diagnostic only and never suppress costs. P&L
  still uses a continuous back-adjusted series, so exact old/new-expiry fills
  remain unavailable until serial-contract histories are supplied.
- Back-adjusted price differences drive P&L, but unadjusted active-contract
  closes value economic gross notional; adjusted price levels are never used
  as contract notionals.
- P&L uses USD-converted point values. Missing held settlements, FX values,
  cost inputs, or margin inputs raise an error instead of silently booking zero.
- Costs include half a tick plus $2.50 per contract per one-way leg. At one
  tick plus $2.50, selected-window performance falls to 23.60% / 1.917.
- `risk_scalar` is correctly named: its 2x cap is a volatility-target
  multiplier, not a cap on economic gross notional.
- Sortino now uses downside root-mean-square, and CAGR includes the first
  return interval.

The corrected output records actual NAV, contracts, gross notional, the
catalogue's static margin diagnostic, order participation, ordinary turnover,
and incremental roll turnover. In 1990-2004, average gross notional is 4.75x
NAV and the static margin diagnostic averages 32.88% and peaks at 72.55%.

## Strategy specification

- 61 futures across equity indices, government bonds, FX, energy, metals, and
  agriculture/livestock.
- Equal blend of 12-month sign trend and year-on-year change in realized roll
  yield, with causal 63-day market-level strategy-volatility scaling.
- Equal pre-forecast volatility budget per available market.
- Trailing volume eligibility, 20/120-day volatility-shock taper, 10% portfolio
  volatility target, and RiskMetrics EWMA lambda 0.94 risk scaling.
- Monthly decisions and a 25% no-trade region.

The specification was selected retrospectively using 1990-2004. Neither
subperiod is a confirmation set, and 1980-1989 and 2005-2014 were also inspected
in prior research. Historical experiment context is retained in
[RESEARCH_HISTORY.md](RESEARCH_HISTORY.md), clearly marked as superseded
accounting evidence.

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

For the executed research walkthrough:

```bash
pip install -e ".[notebook]"
export DELTA1_DATA_DIR="/path/to/Round1AllData/Quant Researcher/Delta1"
jupyter lab DELTA1_Strategy.ipynb
```

The run writes five current artifacts:

- `outputs/strategy_metrics.csv`: fixed-window point estimates and validation flags
- `outputs/strategy_daily.csv`: NAV, returns, costs, exposure, margin, and execution diagnostics
- `outputs/strategy_monthly_targets_per_dollar.csv`: buffered research targets before NAV scaling and rounding
- `outputs/strategy_execution_events.csv`: actual contract changes, vendor label dates, executed rolls, and charged turnover
- `outputs/strategy_config.json`: exact assumptions and selection disclosures

The former optimization and robustness point-metric CSVs were removed because
they could not be regenerated under the corrected engine and their tests only
re-read static rows. Reproducible multiplicity analysis requires restoring all
historical trial return paths.

Run the complete suite with the supplied dataset:

```bash
export DELTA1_DATA_DIR="/path/to/Round1AllData/Quant Researcher/Delta1"
python -m unittest discover -s tests -v
```

## Real-world blockers

Production use still requires serial-contract histories and a documented roll
schedule, dated contract specifications and margin requirements, settlement FX,
exchange/clearing/regulatory fees, bid/ask and market-impact calibration,
portfolio-level margin and cash controls, delivery/first-notice safeguards, and
a point-in-time universe. The current catalogue terms are static, continuous
contracts cannot directly generate expiry-specific orders, the universe is
survivorship-biased, all available periods have been reused, and the data end in
2014. Multiple label changes before the next executable session cannot be
mapped to exact expiry transitions without serial-contract histories. Modeled
roll participation reaches 2.0 times reported daily volume in the selected
window because serial-expiry liquidity is unavailable; this alone
invalidates a production-capacity claim. Independent post-2014 data and forward
paper trading are required before capital deployment.

The implementation assumptions follow the exchange mechanics that futures are
marked to settlement daily and that a roll offsets the expiring contract while
establishing the next one. See CME's [mark-to-market](https://www.cmegroup.com/education/courses/introduction-to-futures/mark-to-market),
[contract-roll](https://www.cmegroup.com/education/courses/introduction-to-futures/understanding-futures-expiration-contract-roll),
[liquidity](https://www.cmegroup.com/education/articles-and-reports/how-traders-measure-liquidity),
and [margin-change](https://www.cmegroup.com/education/articles-and-reports/understanding-margin-changes)
explanations. The CFTC's [hypothetical-results warning](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/fraudadv_tradingsystem.html)
is directly applicable to the remaining execution, margin, and selection gaps.
