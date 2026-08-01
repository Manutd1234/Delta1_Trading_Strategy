# Machine-learning research specification

## Decision

The tested ML models do not beat the 12-month time-series momentum benchmark in the untouched 2005–2014 evaluation window. The ML code and negative result are retained because a falsified hypothesis is more informative than an optimized backtest selected after seeing the test data.

## Forecast panel

The panel is monthly by instrument. It pools 22 futures so the estimator can learn common relationships while the `asset_class` categorical feature permits different intercept structure. Inputs are short/medium/long trend, breakout, volatility, skew, downside share, cross-sectional rank, class trend, and lagged macro stress variables. The target is next-month direction.

Training begins in 1997. Predictions begin in 2003 to create two years of pre-evaluation model history; portfolio scoring begins in 2005 and ends with the supplied data in 2014.

## Nested annual walk-forward

For prediction year `Y`:

1. retain only labels with `label_end < January 1, Y`;
2. reserve the last 24 available months as validation;
3. fit four declared candidates on the earlier observations;
4. choose the lowest validation log-loss, breaking ties on AUC;
5. refit that candidate on every available label;
6. predict every month in year `Y` without intra-year refitting.

The candidate set contains two ridge-logistic penalties and two shallow gradient-boosted tree depths. The small grid controls researcher degrees of freedom and keeps the model explainable. A deep network is not appropriate for this small, dependent monthly panel.

## Portfolio integration

The probability forecast is

```text
ml_signal = clip((P(up) - 0.50) / 0.15, -1, 1)
hybrid = 0.50 * ml_signal + 0.50 * sign(12-month trend)
```

Both signals use the baseline portfolio's class-balanced contract sizing, lagged volatility target, next-business-day activation, and contract-specific spread/commission costs. This isolates forecast quality from accounting differences.

## Findings

- Classification ROC AUC is 0.491; the forecast does not discriminate direction.
- ML-only Sharpe is 0.15 versus 0.68 for the 12-month benchmark.
- The 50% hybrid Sharpe is 0.54 and maximum drawdown is -18.22% versus -20.19% for the benchmark.
- A 25% blend is less damaging, but selecting it now would reuse the OOS window.
- The result is stable in direction across 2005–2009 and 2010–2014: the benchmark has the strongest Sharpe in both subperiods.

## Rejected shortcuts

- The 2024 DTCC snapshot is not backfilled into 2005–2014.
- Candidate models are not selected on OOS Sharpe.
- Blend weights are not optimized on the final evaluation period.
- Permutation importance is diagnostic and not used for another round of feature selection.
- A neural network is not added merely to make the strategy sound more sophisticated.

## Valid next step

Acquire post-2014 individual-contract futures, point-in-time roll mappings, vintage macro releases, and daily DTCC histories. Freeze the current pipeline before opening those data, then treat the later period as a genuine locked holdout or paper-trading experiment.
