# Validation and model-risk review

## Evaluation window

The reported out-of-sample window is 2005-01-03 through 2014-12-31, or 10.35 trading years. All rolling estimators use only data at or before the signal date. New month-end targets are shifted one business day before earning returns.

The supplied files end in 2014. The phrase “out of sample” describes the code's temporal split and lagging convention; it does not make post-2014 claims.

## Automated invariants

The generated `outputs/institutional_invariants.csv` confirms:

- all net returns are finite;
- trading costs are never negative;
- forecasts remain inside `[-1, 1]`;
- shock multipliers remain inside configured bounds;
- held positions change only at month boundaries;
- the OOS window exceeds five years.

Unit tests also mutate future prices and assert that earlier forecasts do not change.

## Stress scenarios

`stress_test_table()` reports:

- base controls;
- zero no-trade buffer;
- a wider 40% buffer;
- disabled volatility-shock control;
- aggressive 3-month confirmation;
- 2x and 3x transaction costs.

The table is an ablation and sensitivity exercise, not a parameter optimizer. All rows are kept, including variants that weaken performance.

## Current result

| Metric | Adaptive | Baseline |
|---|---:|---:|
| CAGR | 7.04% | 6.93% |
| Annualized volatility | 10.56% | 10.69% |
| Sharpe, rf=0 | 0.70 | 0.68 |
| Maximum drawdown | -18.61% | -20.19% |
| Annual cost drag | 0.12% | 0.14% |

The differences are economically plausible but small. They should not be treated as statistically decisive.

## Known model risks

1. **Stale endpoint:** no post-2014 evidence.
2. **Continuous-contract abstraction:** no actual roll trades or expiry selection.
3. **Survivorship:** the supplied catalogue may not be point-in-time.
4. **Simplified costs:** no impact, variable spread, exchange fees, or liquidity limit.
5. **Fractional contracts:** no integer portfolio or small-capital feasibility.
6. **Funding omission:** no collateral yield, margin cash flows, or financing.
7. **Multiple testing:** reviewed sensitivity variants can overstate confidence if treated as independent discoveries.
8. **Serial dependence:** ten calendar years contain far fewer than ten independent trend regimes.

## Required work before deployment

1. Obtain contract-level histories through the present.
2. Implement and reconcile a deterministic roll schedule.
3. Add point-in-time liquidity screens and delisted contracts.
4. Calibrate spread and impact from quote/trade data.
5. Run purged walk-forward and post-2014 locked holdout tests.
6. Convert fractional targets to integer contracts under capital and margin constraints.
7. Paper trade with order, fill, fee, and broker-statement reconciliation.
8. Define live limits for gross risk, per-contract risk, margin, stale data, and daily loss.
