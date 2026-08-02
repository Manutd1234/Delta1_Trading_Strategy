# DELTA1 Quant Research

NUS Investment Society Quant Research submission: an explainable, cost-aware trend-following strategy — **same one-line forecast, 43 global futures markets instead of 22**, with a pre-registered evaluation, a walk-forward training experiment, and an audited pass over four institutional risk-engineering fixes.

The research question is simple: **does widening the traded universe — the improvement the fundamental law of active management predicts most reliably — raise CAGR and Sharpe, and does data-driven model selection ("training") add anything beyond it?**

![Breadth TSMOM performance](outputs/enhanced_performance.png)

## Result

All results are net of a half-tick spread estimate plus USD 2.50 per contract per one-way trade, at a 10% volatility target. The data end 2014-12-31; v1 already reported the 2005–2014 window, so it is labeled a *second use*, and 1990–2004 is the primary evidence window (see [PREREGISTRATION.md](PREREGISTRATION.md)).

| Strategy | Window | CAGR | Sharpe | Max drawdown |
|---|---|---:|---:|---:|
| **Breadth TSMOM (43 markets)** | 1990–2004 | **18.4%** | **1.59** | −11.9% |
| Adaptive TSMOM (22 markets, v1) | 1990–2004 | 13.7% | 1.27 | −12.2% |
| **Breadth TSMOM (43 markets)** | 2005–2014 | **11.0%** | **0.99** | −20.8% |
| + Carry extension (tested, not headline) | 2005–2014 | 10.7% | 1.00 | −11.9% |
| Adaptive TSMOM (22 markets, v1) | 2005–2014 | 7.0% | 0.70 | −18.6% |
| Walk-forward selection (training) | 2005–2014 | 7.4% | 0.70 | −24.2% |
| Ensemble of 5 candidates | 2005–2014 | 9.0% | 0.81 | −23.9% |

The breadth improvement **passed the pre-registered claim rule under the v2.0 spec of record**: primary-window 90% paired-bootstrap interval [+0.002, +0.464] and 8 of 10 winning evaluation years. Re-evaluated under the current spec (EWMA volatility targeting), the point estimates are unchanged (P(diff > 0) ≈ 95% in each window) and the evaluation-window interval [+0.004, +0.368] excludes zero, while the primary 5% quantile sits at −0.006 — both verdicts are computed live in `outputs/enhanced_claim_rule.csv` and discussed in the notebook rather than resolved by picking the friendlier spec. The gain is breadth, measured: effective independent bets rise from ~9 to ~12, and it survives 3× costs, 5× costs on the six thinnest markets, dropping any single asset class, and excluding 2008–09.

**Four institutional blueprint fixes were audited, not assumed.** RiskMetrics EWMA volatility targeting (λ = 0.94) was adopted — it improves both windows (+0.10 evaluation Sharpe). Frictional-cost minimization was already built in (cost drag ≈ 1.5% of gross profits vs the 15% ceiling). A roll-yield carry sleeve (Koijen-Moskowitz-Pedersen style, estimated from roll gaps between unadjusted and back-adjusted series) improves point estimates and halves the evaluation drawdown, but its paired Sharpe increment over trend alone is indecisive (P = 0.52), so the standing default-to-simpler rule keeps it as a shipped, tested extension rather than the headline. PCA absorption-ratio de-leveraging and ATR stops were rejected with evidence: the 70% trigger never fires on this universe (top-2 principal components explain at most 67% of variance) and stops conflict with the month-end architecture.

**The training experiment is a reported negative result.** Rolling walk-forward selection among five pre-declared forecast models — re-fit each year-end on trailing 60-month net Sharpe, applied strictly out-of-sample — underperforms the simplest model it contains (0.70 vs 0.99): selection among correlated candidates ranks noise and pays switching costs for it. The equal-weight ensemble does better (0.81) but still loses to simplicity, consistent with the forecast-combination literature.

## Method

The forecast is unchanged from v1:

```text
direction[i, t] = sign(close[i, t] - close[i, t - 252])
```

The portfolio then: sizes positions by lagged EWM dollar volatility; allocates equal ex-ante risk across six asset classes and equally within each; tapers exposure on 20d/120d volatility shocks; targets 10% portfolio volatility (RiskMetrics EWMA λ = 0.94 estimator, the one v2.1 change to the risk stack) with a 2× leverage cap; suppresses small month-end changes with a 25% no-trade region; activates targets the next business day; and deducts contract-specific costs from price-change P&L (back-adjusted levels can cross zero, so returns are never computed off price levels).

## Universe: 43 markets from a written rule

Every USD-denominated `_CCB` contract in the supplied catalogue (56) is included unless excluded for a named reason: duplicates/baskets (WBS, LSU, LRC, MWE, GD, DX), broken mechanics (VX not delta-one; ZQ near-zero volatility at the zero lower bound), or a ~$100M median daily dollar-volume liquidity rule (LBS, ZR, DC, OJ, ZO).

| Asset class | Contracts |
|---|---|
| Equity indices | ES, NQ, RTY, NKD, YM, EMD, HTW |
| US government bonds | ZT, ZF, ZN, ZB |
| FX | 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S |
| Energy | CL, BRN, HO, GAS, NG, RB |
| Metals | GC, SI, HG, PL, PA |
| Agriculture & livestock | ZC, ZW, KE, ZS, ZL, ZM, SB, KC, CC, CT, LE, HE, GF |

Two point-in-time safeguards: a market trades only once its trailing 60-session median reported volume exceeds 1,000 contracts (6N stays out until mid-2007 — its 2005–06 marks carry zero volume), and exchange closures up to 10 business days hold positions at zero P&L instead of forcing round trips (HTW's Lunar New Year closures).

## Repository

```text
DELTA1_Quant_Research.ipynb   executed submission notebook (intro, method, findings, takeaways)
PREREGISTRATION.md            frozen v2 spec, claim rule, and deviation log
delta1_cta.py                 data, accounting, sizing, baseline and metrics
institutional_strategy.py     v1 adaptive risk and no-trade controls (the baseline)
enhanced_strategy.py          the recommended strategy: universe rule, volume gate,
                              EWMA vol targeting, roll-gap carry extension,
                              walk-forward training experiment, paired bootstrap /
                              PSR / DSR / CSCV-PBO statistics, stress suite
tests/                        37 timing, leakage, bounds, costs and integration tests
outputs/                      metrics, statistics, stress tables and charts
```

## Reproduce

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
export DELTA1_DATA_DIR="/path/to/Round1AllData/Quant Researcher/Delta1"

# v2 headline strategy, training experiment, statistics and stress suite:
python enhanced_strategy.py --data-dir "$DELTA1_DATA_DIR" --output-dir outputs

# v1 baseline:
python institutional_strategy.py --data-dir "$DELTA1_DATA_DIR" --output-dir outputs

# tests and the executed notebook:
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace DELTA1_Quant_Research.ipynb
```

## Research basis

- Moskowitz, Ooi & Pedersen, [“Time Series Momentum”](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf), *JFE* (2012) — 12-month sign momentum across 58 futures.
- Hurst, Ooi & Pedersen, [“A Century of Evidence on Trend-Following Investing”](https://doi.org/10.3905/jpm.2017.44.1.015), *JPM* (2017) — 67 markets.
- Baz et al., [“Dissecting Investment Strategies in the Cross Section and Time Series”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695101) (2015) — doubly-normalized trend signals.
- Timmermann, “Forecast Combination” (2006); DeMiguel, Garlappi & Uppal, *RFS* (2009) — combination and simple rules beat selection.
- Bailey et al., [“The Probability of Backtest Overfitting”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), *JCF* (2017) — CSCV PBO, PSR/DSR.
- Koijen, Moskowitz, Pedersen & Vrugt, [“Carry”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2298565), *JFE* (2018) — the roll-yield carry extension.
- Kritzman et al., “Principal Components as a Measure of Systemic Risk”, *JPM* (2011) — the absorption ratio (tested, rejected).
- Moreira & Muir, [“Volatility-Managed Portfolios”](https://www.nber.org/papers/w22208), *JF* (2017).

## Limitations

- 2005–2014 is a second use of a previously reported window; pre-registration, paired statistics, and cross-window consistency mitigate but cannot eliminate that. **Post-2014 locked-holdout validation is required.**
- Continuous contracts hide actual rolls and are not directly tradable; the cost model omits market impact, exchange fees, margin, and collateral return; position caps vs open interest are not modeled.
- The fixed catalogue is a survivorship-biased snapshot: markets that died before 2014 are absent.
- Early-history reported volume is of uneven quality; the volume gate is only as good as the marks it reads.

Historical research only; not investment advice. MIT licensed.
