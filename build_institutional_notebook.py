"""Build the institutional strategy notebook; execute it with nbconvert."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "INSTITUTIONAL_STRATEGY.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Robust Adaptive TSMOM

## An institutional-style research and engineering extension

This notebook applies public quant-development principles associated with strong electronic trading teams: exact accounting, strict timing, bounded risk, cost-aware trading, ablation tests, and visible failure modes.

It **does not** reproduce or claim access to proprietary Jane Street, Optiver, Citadel, or other firms' models. Genuine market making requires quotes, order books, queue position, order flow, latency, and fills. The supplied daily bars support a medium-frequency futures portfolio strategy instead."""
    ),
    nbf.v4.new_markdown_cell(
        """## Research question

Can the literature-backed 12-month time-series momentum forecast be made more implementation-aware without hiding complexity inside an optimiser?

The adaptive model keeps the same core forecast and adds two independently testable controls:

1. **Volatility-shock taper:** reduce instrument exposure when 20-day price-change volatility rises far above its 120-day level.
2. **No-trade region:** retain the prior month-end target when a desired adjustment is smaller than 25% of the old/new position scale.

The faster 3-month trend is calculated as a diagnostic. It is disabled as a live confirmation overlay by default because the ablation result is weaker."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from delta1_cta import BacktestConfig, load_metadata, load_prices, performance_metrics, run_backtest
from institutional_strategy import (
    InstitutionalConfig,
    run_institutional_backtest,
    strategy_invariants,
    stress_test_table,
    save_institutional_outputs,
)

DATA_DIR = Path(os.environ.get(
    "DELTA1_DATA_DIR",
    str(ROOT / "data" / "Delta1"),
))
OUTPUT_DIR = ROOT / "outputs"
assert (DATA_DIR / "CATALOGUE_Delta1_Futures.csv").exists(), "Set DELTA1_DATA_DIR."
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Pipeline and timing

```text
daily CCB closes + contract catalogue
              ↓
       point-in-time forecast
              ↓
   contract dollar-risk scaling
              ↓
  lagged portfolio volatility target
              ↓
    stateful no-trade decision
              ↓
   next-business-day held position
              ↓
 contract P&L − spread − commission
```

The central timing rule is that a target formed at month-end close cannot earn that day's return. `_held_positions()` shifts it by one business row."""
    ),
    nbf.v4.new_code_cell(
        """prices = load_prices(DATA_DIR)
metadata = load_metadata(DATA_DIR)

base_config = BacktestConfig(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    oos_start="2005-01-01",
    target_vol=0.10,
    half_spread_ticks=0.50,
    commission_per_contract=2.50,
)
strategy_config = InstitutionalConfig(
    slow_horizon=252,
    confirmation_horizon=63,
    disagreement_scale=1.00,
    shock_start=1.35,
    shock_full=2.00,
    shock_floor=0.75,
    no_trade_buffer=0.25,
)

adaptive = run_institutional_backtest(
    base_config, strategy_config, prices=prices, metadata=metadata
)
baseline = run_backtest(base_config, prices=prices, metadata=metadata)
stress = stress_test_table(base_config, prices, metadata)
checks = strategy_invariants(adaptive, base_config, strategy_config)

save_institutional_outputs(
    adaptive, baseline, stress, checks, base_config, strategy_config
)

coverage = pd.DataFrame({
    "class": metadata["asset_class"],
    "first": prices.apply(pd.Series.first_valid_index),
    "last": prices.apply(pd.Series.last_valid_index),
    "observations": prices.count(),
    "tick_size": metadata["tick_size"],
    "point_value": metadata["point_value"],
})
print(f"{prices.shape[1]} contracts | {prices.index.min().date()} to {prices.index.max().date()}")
display(coverage)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Primary out-of-sample comparison

Both strategies use the same universe, risk target, contract accounting, and cost assumptions. The comparison isolates the shock taper and execution buffer."""
    ),
    nbf.v4.new_code_cell(
        """metrics = pd.concat([
    performance_metrics(adaptive.backtest, base_config.oos_start),
    performance_metrics(baseline, base_config.oos_start),
], axis=1).T
display(metrics)
display(Image(filename=str(OUTPUT_DIR / "institutional_performance.png")))"""
    ),
    nbf.v4.new_markdown_cell(
        """The improvement is modest, which is the appropriate interpretation. The no-trade region reduces estimated cost drag and changes the path of exposure; it is not a new source of economic alpha. The volatility taper slightly changes tail behavior but is not independently strong in this sample."""
    ),
    nbf.v4.new_markdown_cell(
        """## Stress tests and ablations

Every variant is retained, including weak results. This prevents a polished chart from hiding which additions did not help."""
    ),
    nbf.v4.new_code_cell(
        """display(stress.style.format({
    "CAGR": "{:.2%}",
    "Volatility": "{:.2%}",
    "Sharpe": "{:.2f}",
    "Max drawdown": "{:.2%}",
    "Annual cost drag": "{:.2%}",
}))

ax = stress["Sharpe"].sort_values().plot(
    kind="barh", figsize=(10, 4.8), color="#005F73"
)
ax.set_title("OOS Sharpe across execution and model ablations")
ax.set_xlabel("Sharpe (rf=0)")
ax.set_ylabel("")
plt.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_markdown_cell(
        """## Execution diagnostics

The next cells quantify what the controls actually do instead of describing them qualitatively."""
    ),
    nbf.v4.new_code_cell(
        """desired_change = adaptive.desired_monthly_positions.diff().abs().gt(1e-15)
buffered_change = adaptive.buffered_monthly_positions.diff().abs().gt(1e-15)

diagnostics = pd.Series({
    "Desired contract-target changes": int(desired_change.sum().sum()),
    "Executed contract-target changes": int(buffered_change.sum().sum()),
    "Changes suppressed": int(desired_change.sum().sum() - buffered_change.sum().sum()),
    "OOS estimated cost drag": adaptive.backtest.daily.loc[base_config.oos_start:, "cost"].mean() * 252,
    "Mean OOS leverage": adaptive.backtest.daily.loc[base_config.oos_start:, "leverage"].mean(),
    "Share of valid instrument-days under any shock taper": (
        adaptive.shock_multiplier.loc[base_config.oos_start:].stack() < 0.999999
    ).mean(),
})
display(diagnostics)

display(checks.to_frame())
assert checks.all(), checks[~checks].to_dict()"""
    ),
    nbf.v4.new_markdown_cell(
        """## Model-risk conclusion

What survives review:

- the 12-month direction remains the strongest and simplest forecast in this dataset;
- exact futures P&L and contract-specific costs are non-negotiable;
- a no-trade region is useful because it reduces churn without requiring a fragile optimiser;
- an aggressive 3-month confirmation overlay weakens results and stays disabled;
- 3x cost assumptions reduce but do not eliminate the historical edge;
- all named invariants pass.

What remains unresolved:

- the data stop in 2014;
- continuous contracts hide roll implementation;
- daily bars cannot estimate adverse selection, impact, queue position, or fill probability;
- fractional contracts omit capital and margin constraints;
- the result needs contract-level post-2014 and paper-trading validation before any deployment discussion.

The institutional lesson is not that more features are better. It is that every feature, risk rule, and execution assumption should have a separate test and an observable effect."""
    ),
]

nbf.write(nb, OUTPUT)
print(OUTPUT)
