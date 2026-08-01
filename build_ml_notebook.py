"""Build the executed ML walk-forward research notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ML_WALK_FORWARD_STRATEGY.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# DELTA1: leakage-safe machine-learning extension

## Research objective

Test whether a small, explainable classifier can improve a diversified futures time-series momentum strategy. The decision standard is **unseen 2005–2014 performance after estimated costs**, not in-sample fit.

The primary benchmark is the literature-specified 12-month TSMOM CTA. The ML experiment is deliberately constrained to ridge logistic regression and shallow gradient-boosted trees; this dataset does not justify a large neural network.

This is historical research, not investment advice. It does not claim to reproduce any proprietary trading firm's models."""
    ),
    nbf.v4.new_markdown_cell(
        """## Research lineage and pre-declared hypotheses

- Moskowitz, Ooi & Pedersen, [*Time Series Momentum*](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf): benchmark the sign of the past 12-month return across diversified futures.
- Hurst, Ooi & Pedersen, [*A Century of Evidence on Trend-Following Investing*](https://doi.org/10.3905/jpm.2017.44.1.015): motivates a broad, simple trend prior rather than a narrow optimized rule.
- Moreira & Muir, [*Volatility-Managed Portfolios*](https://www.nber.org/papers/w22208): motivates lagged volatility and volatility-regime features.
- Lim, Zohren & Roberts, [*Enhancing Time Series Momentum Strategies Using Deep Neural Networks*](https://arxiv.org/abs/1904.04912): motivates nonlinear forecast combinations, but also highlights why a deep model needs a much larger panel.
- Gu, Kelly & Xiu, [*Empirical Asset Pricing via Machine Learning*](https://www.nber.org/papers/w25398): motivates nonlinear interactions and disciplined out-of-sample evaluation.
- Bailey et al., [*The Probability of Backtest Overfitting*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253): motivates the tiny candidate set, nested validation, and reporting of weak variants.

Hypothesis: short/medium/long trend, breakout, volatility regime, cross-sectional context, and lagged public macro stress variables may add information beyond the 12-month sign. Null: they do not survive strict walk-forward testing."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from data_engineer_features import save_snapshot_outputs
from delta1_cta import BacktestConfig
from ml_strategy import MLConfig, classification_metrics, run_ml_pipeline

DATA_DIR = Path(os.environ.get(
    "DELTA1_DATA_DIR", str(ROOT / "data" / "Delta1")
))
EXTERNAL_MACRO = Path(os.environ.get(
    "DELTA1_EXTERNAL_MACRO", str(ROOT / "data" / "external" / "fred_macro.csv")
))
DATA_ENGINEER_INPUT = Path(os.environ.get(
    "DELTA1_DATA_ENGINEER_INPUT",
    str(ROOT / "data" / "Data Engineer" / "CFTC_CUMULATIVE_FOREX_2024_04_08.csv"),
))
OUTPUT_DIR = ROOT / "outputs"

assert (DATA_DIR / "CATALOGUE_Delta1_Futures.csv").exists(), "Set DELTA1_DATA_DIR."
assert EXTERNAL_MACRO.exists(), "Run: python download_external_data.py"
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Data inventory and point-in-time decision

The tradable research panel is 22 USD-denominated, additive back-adjusted futures across equity indices, government bonds, G10 FX, and commodities.

The Desktop Data Engineer folder contains a DTCC/CFTC cumulative FX public-dissemination snapshot and its field guide. The snapshot is valuable as an ingestion prototype—trade count, capped USD notional, tenor, clearing, block, prime-brokerage, and venue measures—but it was published in 2024. Since the supplied futures backtest ends in 2014, using those observations as historical predictors would be direct look-ahead. The audit below therefore excludes it from model fitting."""
    ),
    nbf.v4.new_code_cell(
        """if DATA_ENGINEER_INPUT.exists():
    save_snapshot_outputs(DATA_ENGINEER_INPUT, OUTPUT_DIR)
    snapshot_audit = json.loads((OUTPUT_DIR / "data_engineer_snapshot_audit.json").read_text())
    display(pd.Series(snapshot_audit, name="DTCC snapshot audit").to_frame())
    display(pd.read_csv(OUTPUT_DIR / "data_engineer_fx_snapshot.csv").head(10))
else:
    print("Optional Data Engineer snapshot not found; set DELTA1_DATA_ENGINEER_INPUT.")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Online macro data

`download_external_data.py` retrieves three public daily FRED series and writes a timestamped, hashed source manifest:

- [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS): market volatility/stress;
- [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y): Treasury curve slope;
- [BAA10Y](https://fred.stlouisfed.org/series/BAA10Y): Baa corporate credit spread over the 10-year Treasury.

Every macro value is short-gap forward-filled and shifted by one business day before feature construction. This conservative rule avoids trading on a same-day observation whose publication time may follow the futures close. A production system should use vintage/release timestamps rather than revised historical downloads."""
    ),
    nbf.v4.new_code_cell(
        """macro_raw = pd.read_csv(EXTERNAL_MACRO, parse_dates=["Date"])
manifest = json.loads((EXTERNAL_MACRO.parent / "source_manifest.json").read_text())
coverage = pd.DataFrame({
    "first": {c: macro_raw.loc[macro_raw[c].notna(), "Date"].min() for c in macro_raw.columns[1:]},
    "last": {c: macro_raw.loc[macro_raw[c].notna(), "Date"].max() for c in macro_raw.columns[1:]},
    "observations": {c: macro_raw[c].notna().sum() for c in macro_raw.columns[1:]},
})
display(coverage)
print("Provider:", manifest["provider"])
print("Retrieved:", manifest["retrieved_at_utc"])
print("Timing policy:", manifest["timing_policy"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Model and timing

At each instrument/month observation, the feature vector contains:

- 1/3/6/12-month trends normalized by lagged 120-day volatility;
- 20-day/120-day volatility ratio, 12-month breakout location, 3-month skew, and downside-day share;
- cross-sectional 12-month trend rank and asset-class trend;
- lagged VIX, yield-curve, and Baa credit-spread levels and 1-month changes;
- the instrument's asset class.

The binary target is the sign of the next month's volatility-normalized price change. Each January:

```text
1997 ... fit history | trailing 24m validation | year Y predictions
                      select 1 of 4 models      labels end < Jan 1 Y
```

Candidate set: ridge logistic regression (`C=0.05, 0.20`) and histogram gradient boosting (depth `1, 2`). Selection minimizes validation log loss, then the winner is refit on every label ending before the test year. Monthly probabilities become a bounded signal and are blended 50/50 with the 12-month trend prior. The target activates on the next business day and enters the same volatility sizing and transaction-cost engine as the baseline."""
    ),
    nbf.v4.new_code_cell(
        """base_config = BacktestConfig(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    oos_start="2005-01-01",
    oos_end="2014-12-31",
    target_vol=0.10,
    half_spread_ticks=0.50,
    commission_per_contract=2.50,
)
ml_config = MLConfig(
    training_start="1997-01-01",
    prediction_start="2003-01-01",
    prediction_end="2014-12-31",
    validation_months=24,
    ml_blend_weight=0.50,
    probability_scale=0.15,
    random_state=17,
)

result = run_ml_pipeline(base_config, ml_config, EXTERNAL_MACRO)
display(result.metrics.style.format({
    "CAGR": "{:.2%}", "Annualized volatility": "{:.2%}",
    "Sharpe (rf=0)": "{:.2f}", "Max drawdown": "{:.2%}",
    "Annual cost drag": "{:.2%}",
}))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Primary out-of-sample result

The experiment fails the alpha hypothesis: the ML-only and 50/50 hybrid forecasts underperform the 12-month benchmark. This is the central research result, not something to conceal. The simple trend rule remains the recommended forecast under Occam's razor."""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(OUTPUT_DIR / "ml_performance.png")))
display(classification_metrics(result.walk_forward.predictions, base_config.oos_start).to_frame())"""
    ),
    nbf.v4.new_markdown_cell(
        """Classification diagnostics answer a different question from portfolio Sharpe, but here they agree: the classifier has no reliable directional discrimination. The hybrid can still reshape exposure and drawdown because its continuous probabilities interact with risk sizing, but that is not evidence of incremental alpha."""
    ),
    nbf.v4.new_markdown_cell("""## Nested selections and feature diagnostics"""),
    nbf.v4.new_code_cell(
        """selection = result.walk_forward.model_selection.copy()
selection["timing_pass"] = pd.to_datetime(selection["last_available_label"]) < pd.to_datetime(selection["year"].astype(str) + "-01-01")
display(selection)
assert selection["timing_pass"].all()

model_counts = selection["selected_model"].value_counts().sort_values()
importance = (
    result.walk_forward.feature_importance.groupby("feature")["importance"]
    .mean().sort_values()
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
model_counts.plot.barh(ax=axes[0], color="#006D77", title="Annual model selections")
importance.tail(12).plot.barh(ax=axes[1], color="#7B2CBF", title="Mean validation permutation importance")
axes[0].set_xlabel("Years selected")
axes[1].set_xlabel("Increase in validation neg-log-loss score")
plt.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_markdown_cell(
        """Permutation importance is descriptive, not a new selection layer: it is measured on the same trailing validation window used to choose the annual model. Negative or unstable values are evidence against relying on a feature, not permission to search repeatedly until it looks useful."""
    ),
    nbf.v4.new_markdown_cell("""## Robustness, uncertainty, and subperiods"""),
    nbf.v4.new_code_cell(
        """display(result.robustness.style.format({
    "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}",
    "Max drawdown": "{:.2%}", "Annual cost drag": "{:.2%}",
}))
display(result.bootstrap.style.format({
    "Sharpe 5%": "{:.2f}", "Sharpe median": "{:.2f}",
    "Sharpe 95%": "{:.2f}", "P(Sharpe > 0)": "{:.1%}",
}))
display(pd.read_csv(OUTPUT_DIR / "ml_subperiod_metrics.csv").style.format({
    "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}",
    "Max drawdown": "{:.2%}",
}))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Findings and key takeaways

1. **Keep the simple forecast.** The 12-month TSMOM benchmark produces the strongest OOS CAGR and Sharpe in this experiment. The adaptive non-ML strategy in the companion notebook remains the best implementation candidate because its small improvement comes from transparent risk/execution rules.
2. **Do not promote the ML overlay.** OOS ROC AUC is below 0.50 and ML-only portfolio performance is weak. A 25% blend is less damaging than 50%, but choosing it after seeing the test set would be data mining.
3. **Use the failed model as evidence.** Nested annual selection, lagged macro data, explicit label cutoffs, costs, subperiods, and block-bootstrap intervals make the negative result credible.
4. **The Data Engineer snapshot is useful—but not in this backtest.** It is production-oriented feature plumbing for post-2024 live research, not permissible training data for 2005–2014.
5. **Next valid experiment:** obtain point-in-time daily DTCC histories and post-2014 individual-contract futures with roll mappings, freeze the model, and conduct a genuinely untouched forward test.

### Limitations

- Historical FRED downloads can contain revisions; vintage data are preferable.
- The 22-instrument monthly panel is small and cross-sectionally dependent.
- Continuous back-adjusted contracts hide rolls and are not directly executable.
- Costs omit impact, liquidity regimes, margin, and collateral return.
- Bootstrap intervals quantify sampling variation, not model-selection uncertainty.
- Daily bars cannot support market-making, order-book, or execution-alpha claims."""
    ),
]

nbf.write(nb, OUTPUT)
print(OUTPUT)
