"""Assemble the submission bundle and zip it.

    python scripts/build_submission_bundle.py

Produces `dist/delta1_submission_<date>.zip` containing everything the brief's
Deliverables section asks for, laid out so a reviewer can start at one file.

The bundle is self-sufficient. It carries the cleaned input panel as parquet,
so `python reproduce.py` inside the unzipped folder regenerates the headline
figures with no network access and without the licensed vendor CSVs -- which
are not redistributable and are therefore not included.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "outputs" / "submission"
DATA_DIR = ROOT / "Round1AllData" / "Quant Researcher" / "Delta1"


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "delta1_reference", ROOT / "reference" / "delta1_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPRODUCE = '''"""Reproduce the headline results offline.

    python reproduce.py

Reads the cleaned panel bundled under data/panel/ -- no network access, and no
licensed vendor CSVs required. Prints the same table the report quotes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "delta1_reference", HERE / "strategy" / "delta1_reference.py"
)
d1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1)

panel = {}
for path in sorted((HERE / "data" / "panel").glob("*.parquet")):
    frame = pd.read_parquet(path)
    # tick_size is the one Series in the panel; everything else is a frame.
    panel[path.stem] = frame.iloc[:, 0] if path.stem == "tick_size" else frame

daily = d1.run(data=panel)
table = pd.DataFrame({
    name: d1.metrics(daily, start, end)
    for name, (start, end) in {
        "1990-2004 development": ("1990-01-01", "2004-12-31"),
        "2005-2014 out-of-sample": ("2005-01-01", "2014-12-31"),
        "1990-2014 full": ("1990-01-01", "2014-12-31"),
    }.items()
})

pd.set_option("display.width", 150)
print(f"\\nDelta1 — reproduced offline from the bundled panel ({len(d1.SYMBOLS)} markets)\\n")
print(table.to_string(float_format=lambda v: f"{v:,.4f}"))

expected = 1.5895266624546427
measured = float(d1.metrics(daily, "1990-01-01", "2014-12-31")["Sharpe (rf=0)"])
status = "MATCHES" if abs(measured - expected) < 1e-12 else "DIFFERS FROM"
print(f"\\nFull-period Sharpe {measured:.10f} {status} the published {expected:.10f}")
'''


def headline_numbers() -> dict:
    """Read the headline figures from the artifacts, never from memory.

    An earlier revision hard-coded the gross column here and every one of its
    four values was wrong. Numbers in an index that a reviewer cross-checks
    against the CSVs beside it must come from those CSVs.
    """
    summary = pd.read_csv(SUBMISSION / "gross_vs_net_summary.csv")
    row = summary.loc[summary["Window"] == "1990-2014 full"].iloc[0]
    values = {
        "gross_cagr": float(row["Gross annualised return"]),
        "net_cagr": float(row["Net annualised return"]),
        "gross_sharpe": float(row["Gross Sharpe"]),
        "net_sharpe": float(row["Net Sharpe"]),
        "gross_mdd": float(row["Gross max drawdown"]),
        "net_mdd": float(row["Net max drawdown"]),
        "sessions": int(row["Sessions"]),
    }
    volatility = pd.read_csv(SUBMISSION / "required_results.csv")
    volatility = volatility.loc[
        volatility["Metric"].str.startswith("Annualised volatility")
    ].iloc[0]
    values["gross_vol"] = float(volatility["1990-2014 full (gross)"])
    values["net_vol"] = float(volatility["1990-2014 full (net)"])
    values["gross_rtd"] = values["gross_cagr"] / abs(values["gross_mdd"])
    values["net_rtd"] = values["net_cagr"] / abs(values["net_mdd"])

    fill = pd.read_csv(SUBMISSION / "robustness_no_fill.csv")
    fill = fill.loc[fill["window"] == "1990-2014 full"].iloc[0]
    values["fill_delta"] = float(fill["delta_Sharpe (rf=0)"])
    values["fill_strict_sharpe"] = float(fill["observed_only_Sharpe (rf=0)"])

    optimality = ROOT / "outputs/optimality"
    walk = pd.read_csv(optimality / "optimality_walk_forward_summary.csv").set_index("book")
    selection = pd.read_csv(optimality / "optimality_selection.csv").set_index("pool")
    manifest = json.loads((optimality / "optimality_run.json").read_text(encoding="utf-8"))
    values["opt_runs"] = int(manifest["configurations_run"])
    values["opt_draws"] = int(manifest["joint_search_trials"])
    values["opt_draws_beating"] = int(round(
        (1.0 - float(selection.at["joint random search", "frozen_full_sharpe_percentile"]))
        * float(selection.at["joint random search", "configurations"])
    ))
    values["opt_reopt_sharpe"] = float(walk.at["annually re-optimised", "sharpe"])
    values["opt_frozen_sharpe"] = float(walk.at["frozen baseline", "sharpe"])
    values["opt_reopt_cagr"] = float(walk.at["annually re-optimised", "cagr"])
    values["opt_frozen_cagr"] = float(walk.at["frozen baseline", "cagr"])
    profiles = pd.read_csv(optimality / "optimality_profile_summary.csv").set_index("parameter")
    values["opt_lookback"] = int(profiles.at["trend_lookback", "frozen_value"])
    values["opt_spike"] = float(profiles.at["trend_lookback", "neighbour_sharpe_drop"])
    return values


def start_here(files: str) -> str:
    v = headline_numbers()
    return f"""# Delta1 — Submission

**Diversified global futures: 12-month time-series momentum blended with roll-yield
momentum across 59 markets, volatility-targeted at 7%, rebalanced monthly, net of
modeled spread, slippage, commission, exchange fees, market impact and roll costs.**

Evaluation window 1990-01-01 to 2014-12-31 ({v['sessions']:,} sessions, 25 years).

| | Net | Gross |
|---|---|---|
| Annualised return | **{v['net_cagr']:.2%}** | {v['gross_cagr']:.2%} |
| Volatility | **{v['net_vol']:.2%}** | {v['gross_vol']:.2%} |
| Sharpe | **{v['net_sharpe']:.2f}** | {v['gross_sharpe']:.2f} |
| Maximum drawdown | **{v['net_mdd']:.2%}** | {v['gross_mdd']:.2%} |
| Return-to-drawdown | **{v['net_rtd']:.2f}** | {v['gross_rtd']:.2f} |

**Verdict: paper trade.** Not capital — there is no forward record, and the panel
ends 2014-12-31. Not rejection — the evidence is strong and the mechanism is
documented. See the executive conclusion at the top of `report.html`.

**Read the headline with one qualification.** The panel is forward-filled across
holidays, which the brief discourages. Rebuilding every signal from observed
prices only — the strictest reading of that rule — costs **{abs(v['fill_delta']):.3f} Sharpe**
({v['net_sharpe']:.2f} → {v['fill_strict_sharpe']:.2f}), and more over 2005-2014. The conclusion
holds, because {v['fill_strict_sharpe']:.2f} still exceeds every independent
benchmark replicated here, but the sensitivity is real
and is reported rather than buried: `results/robustness_no_fill.csv`.

---

## Start here

1. **`report.html`** — the report. Executive conclusion first, then performance,
   drawdown, rolling Sharpe, position, and a diagnostic chart, followed by the
   required results and every robustness check. Self-contained: open it in any
   browser, offline.
2. **`strategy/delta1_reference.py`** — the entire strategy in one file. No
   imports from any project package. Read it top to bottom and you have the
   complete specification.
3. **`notebook/delta1_case_research.ipynb`** — the research narrative, executed.

## Reproducing, with no internet access

```bash
pip install numpy pandas pyarrow
python reproduce.py
```

Runs in about three seconds against the cleaned panel bundled under
`data/panel/`, and prints the same figures the report quotes. **No network
access is required and no licensed vendor CSV is needed** — the parquet panel is
the cleaned, aligned input the strategy actually consumes.

To re-run against the original vendor CSVs instead:

```bash
python strategy/delta1_reference.py --data-dir "<path to>/Quant Researcher/Delta1"
```

## Where each requirement is answered

### D. Data discipline and cost assumptions

| Requirement | Where |
|---|---|
| Why adjusted or unadjusted prices | `report.html` § Data discipline; `results/data_quality_checks.csv` |
| Duplicates, missing observations, non-trading days, units, currency, stale prices | `results/data_quality_checks.csv`, `results/data_quality_by_market.csv` |
| Forward-fill policy, stated and quantified | `results/data_quality_checks.csv` — including whether a filled cell can reach a trade |
| **What the fill is worth** — every signal rebuilt on observed sessions only | `results/robustness_no_fill.csv`; declared in `report.html` § Data discipline and in the executive conclusion |
| One-way cost, applied whenever the position changes | `results/cost_assumptions.csv` |
| Cost in basis points, per asset class, against the suggested bands | `results/cost_realized_by_class.csv` |
| Gross versus net | `results/gross_vs_net_summary.csv`; diagnostic chart in `report.html` |
| Roll costs | `results/cost_assumptions.csv` — charged as two contracts of turnover per delivery transfer |

### E. Required results and robustness

| Requirement | Where |
|---|---|
| Annualised return, volatility, Sharpe, maximum drawdown, return-to-drawdown | `results/required_results.csv` |
| Hit rate, number of trades, average holding period, turnover, total costs, average exposure | `results/required_results.csv` |
| Performance versus benchmark, and what drove the result | `results/benchmark_comparison.csv`, `results/return_attribution.csv` |
| Parameter sensitivity — a small set of nearby values, not a search | `results/robustness_parameter_sensitivity.csv` |
| Chronological out-of-sample — develop early, assess unchanged rules later | `results/robustness_chronological_oos.csv` |
| Cost sensitivity | `results/cost_sensitivity.csv` |
| Performance across at least two regimes | `results/robustness_regimes.csv` — two independent regime axes |

### Was this configuration the one a search would have picked?

The parameter sensitivity above is nine pre-declared runs and searches nothing, which
leaves the fair question open. So, **after** the configuration was frozen and published,
{v['opt_runs']} configurations were run: dense one-parameter profiles, {v['opt_draws']}
independent joint draws from the same box, and seven structural alternatives.

- **{v['opt_draws_beating']} of {v['opt_draws']}** joint draws beat the frozen configuration
  in sample, and none beat it over 2005-2014.
- Choosing the best configuration on 1990-2004 and living with it afterwards **lost**
  Sharpe over 2005-2014.
- Nineteen years of annual re-optimisation returned a Sharpe of
  **{v['opt_reopt_sharpe']:.3f}** against the frozen book's **{v['opt_frozen_sharpe']:.3f}**
  — and a CAGR of {v['opt_reopt_cagr']:.2%} against {v['opt_frozen_cagr']:.2%}. The optimiser
  bought nothing and paid {(v['opt_frozen_cagr'] - v['opt_reopt_cagr']) * 100:.1f} points of
  compound return for it.
- The one real weakness it found: the {v['opt_lookback']}-session trend lookback sits on a
  spike worth {v['opt_spike']:.2f} Sharpe against its neighbours. The peak moves between
  halves of the sample, so it is noise — but it is reported, in
  `results/optimality_profile_summary.csv`.

No parameter was re-selected from any of it, and no number elsewhere in this bundle
depends on it. `report.html` § Optimality audit; `results/optimality_*.csv`.

Beyond the brief, four descriptive analyses of the same frozen configuration:

| Analysis | Where |
|---|---|
| Eight crisis windows declared from public event dates (Gulf, 1994 bonds, Asia, LTCM, dot-com, Lehman, US downgrade, taper) | `results/validation_crisis_windows.csv` |
| Cost breakeven — net CAGR survives until ~12.7x every modeled execution cost | `results/validation_cost_breakeven.csv` |
| Capacity — the binding constraint is roll completion in thin markets between $2M and $5M, not impact-cost Sharpe erosion | `results/validation_capacity.csv` |
| Leave-one-sector-out jackknife of the 59-market book | `results/validation_universe_jackknife.csv` |

### F. Deliverables

| Requirement | Where |
|---|---|
| Jupyter notebook, runnable end to end | `notebook/delta1_case_research.ipynb` |
| CSV / Parquet data; reproduces without internet | `data/panel/*.parquet` + `reproduce.py` |
| Self-contained HTML report | `report.html` |
| Source note | `data/source_note.csv` |
| Conclusion: pursue / monitor / paper trade / reject | `report.html`, first section |

## Contents

{files}

## Caveats a reviewer should hold

- **The panel ends 2014-12-31.** Nothing here observes the last decade. This is the
  binding limitation on every figure in the bundle.
- **Returns are futures excess returns.** Cash collateral earns zero; the research
  ledger excludes collateral yield and variation-margin funding.
- **No external data is used anywhere.** That is a deliberate scope choice, and the
  reason is stated in the notebook's Limitation 1 rather than left implied.
- **The raw vendor panel is not redistributed.** It was supplied with the case and is
  licensed; the cleaned derived panel is included instead so results still reproduce.
"""


def build(output_dir: Path, stage: Path) -> Path:
    d1 = load_reference()

    if stage.exists():
        shutil.rmtree(stage)
    for sub in ("strategy", "notebook", "data/panel", "results"):
        (stage / sub).mkdir(parents=True, exist_ok=True)

    # --- the model, and its equivalence proof -------------------------
    shutil.copy2(ROOT / "reference/delta1_reference.py", stage / "strategy/delta1_reference.py")
    shutil.copy2(ROOT / "tests/test_reference.py", stage / "strategy/test_reference.py")

    # --- the notebook --------------------------------------------------
    shutil.copy2(
        ROOT / "notebooks/delta1_case_research.ipynb",
        stage / "notebook/delta1_case_research.ipynb",
    )

    # --- the report ----------------------------------------------------
    shutil.copy2(SUBMISSION / "report.html", stage / "report.html")

    # --- cleaned inputs, so the bundle reproduces offline --------------
    panel = d1.load_market_data(DATA_DIR)
    for name, value in panel.items():
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        frame.to_parquet(stage / "data/panel" / f"{name}.parquet", compression="zstd")

    # --- results -------------------------------------------------------
    for name in sorted(p.name for p in SUBMISSION.glob("*.csv")):
        shutil.copy2(SUBMISSION / name, stage / "results" / name)
    # Cost sensitivity already exists canonically; give it its spec name.
    stress = pd.read_csv(ROOT / "outputs/strategy_friction_stress.csv")
    stress.to_csv(stage / "results/cost_sensitivity.csv", index=False)
    # Robustness beyond the brief: descriptive analyses of the frozen baseline.
    for name in (
        "validation_crisis_windows.csv",
        "validation_cost_breakeven.csv",
        "validation_capacity.csv",
        "validation_universe_jackknife.csv",
    ):
        shutil.copy2(ROOT / "outputs/validation" / name, stage / "results" / name)
    # The optimality audit: the one explicit search in the bundle, shipped
    # whole -- every configuration it ran, not just the ones that flatter the
    # frozen answer.
    for path in sorted((ROOT / "outputs/optimality").glob("optimality_*")):
        shutil.copy2(path, stage / "results" / path.name)
    # The daily ledger, so any figure in the report can be re-derived.
    shutil.copy2(ROOT / "outputs/strategy_daily.csv", stage / "results/strategy_daily.csv")
    shutil.copy2(ROOT / "outputs/strategy_metrics.csv", stage / "results/strategy_metrics.csv")
    shutil.copy2(
        ROOT / "outputs/universe/universe_audit.csv", stage / "results/universe_audit.csv"
    )
    source_note = SUBMISSION / "source_note.csv"
    if source_note.is_file():
        shutil.move(str(stage / "results/source_note.csv"), str(stage / "data/source_note.csv"))

    (stage / "reproduce.py").write_text(REPRODUCE, encoding="utf-8")

    # --- manifest, then the index that references it -------------------
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(stage / "MANIFEST.csv", index=False)

    listing = "\n".join(
        f"- `{r['path']}` — {r['bytes'] / 1024:,.0f} KB" for _, r in manifest.iterrows()
    )
    (stage / "00_START_HERE.md").write_text(start_here(listing), encoding="utf-8")

    # --- zip -----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"delta1_submission_{dt.date.today().isoformat()}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(stage.name) / path.relative_to(stage))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=str(ROOT / "dist"))
    parser.add_argument("--stage", default=str(ROOT / "dist" / "delta1_submission"))
    args = parser.parse_args()

    archive = build(Path(args.output_dir), Path(args.stage))
    size = archive.stat().st_size / 1e6
    with zipfile.ZipFile(archive) as bundle:
        count = len(bundle.namelist())
    print(f"\n{archive}")
    print(f"{count} files, {size:.1f} MB compressed")


if __name__ == "__main__":
    main()
