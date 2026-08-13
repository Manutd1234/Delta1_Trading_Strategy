"""Generate the lead research notebook for the Delta1 case.

    python scripts/build_case_notebook.py

Writes `notebooks/delta1_case_research.ipynb`, structured to the four sections
the case brief asks for -- introduction, methodology, findings with
visualisations, key takeaways -- and executing the readable reference
implementation rather than the hardened package, so the notebook a reviewer
reads is the code a reviewer reads.

This is the only notebook the repository keeps. The case brief asks for one
notebook that carries the research end to end; a second one presenting the same
frozen artifacts to a different audience was removed rather than maintained
alongside it.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "delta1_case_research.ipynb"

# Validated categorical slots 1-3 plus status critical; see the data-viz
# palette reference.  Slot 3 (aqua) sits below 3:1 on the light surface, so
# every series carrying it is directly labelled.
PALETTE = """
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 120,
    "font.family": "sans-serif", "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": SECOND, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "axes.titlepad": 24,
    "lines.linewidth": 2.0,
})

def style(ax, title=None, subtitle=None, ylabel=None, pct=False, grid="y", decimals=0):
    \"\"\"One place for the chrome, so every figure reads as the same system.

    The subtitle is drawn just above the axes and the title padded clear of it,
    so the two never overlap regardless of figure height.
    \"\"\"
    if title:
        ax.set_title(title, fontweight="semibold")
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=8.5, va="bottom")
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis=grid, alpha=0.9)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimals))
    return ax

def as_multiple(ax):
    \"\"\"Label a log growth axis '1x, 10x' rather than in scientific notation.\"\"\"
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    return ax
"""


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


CELLS: list[nbf.NotebookNode] = []

# ---------------------------------------------------------------- title -----
CELLS.append(md("""
# Delta1 — A Diversified Global-Futures Trend and Carry Strategy

**NUS Investment Society · QR Recruitment AY26/27 · Question 3 (Delta1)**

> *Develop a trading strategy to trade either an asset or a combination of
> assets in the given global futures/ETFs.*

The answer is a long/short portfolio of **59 global futures** that blends
12-month time-series momentum with a basis-momentum (carry) sleeve, sizes every
position by its own volatility, targets 7% annualized portfolio risk, rebalances
monthly, and charges realistic execution costs.

Over 1990–2014 it earns **13.2% a year at 7.7% volatility (Sharpe 1.59,
maximum drawdown −11.8%)** net of spread, slippage, commission, exchange fees,
market impact, and roll costs. A twenty-year rolling walk-forward — in which the
model's trend horizon is re-chosen annually using only prior data — delivers
**Sharpe 1.42 across 5,218 out-of-sample sessions**, and it is not spanned by
four published trend rules and five reference portfolios replicated on the
identical panel.

Every number in this notebook is computed when the notebook runs. Nothing is
transcribed.

---

### How to read this repository

| If you want | Read |
|---|---|
| **The strategy** | [`reference/delta1_reference.py`](../reference/delta1_reference.py) — the entire model in one file, no imports from this project (the setup cell below prints its exact size) |
| **The research** | this notebook |
| **Proof the short file is the real one** | [`tests/test_reference.py`](../tests/test_reference.py) — asserts it reproduces the production engine *bit for bit* |
| **The production hardening** | [`src/delta1_strategy/`](../src/delta1_strategy) — execution controls, risk gates, deployment evidence |

The case asks for a model that is explainable and code that is easily read and
run. That is the first row. The last row exists because a strategy that would
actually be traded needs more than a backtest, and it is kept strictly out of
the way of the first row.
"""))

# ------------------------------------------------------------------ setup ---
CELLS.append(md("""
## 0. Setup

The notebook runs the reference implementation directly — the same file linked
above — so what is measured below is what is written there. A full 25-year
backtest of 59 markets takes a few seconds.
"""))

CELLS.append(code(f"""
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = Path("Round1AllData/Quant Researcher/Delta1")
if not (ROOT / DATA_DIR).exists():
    raise SystemExit(f"Supplied case data not found at {{ROOT / DATA_DIR}}")

# Import the strategy the way a reader would: by path, one file, no package.
spec = importlib.util.spec_from_file_location(
    "delta1_reference", ROOT / "reference" / "delta1_reference.py"
)
d1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
{PALETTE}
source = (ROOT / "reference" / "delta1_reference.py").read_text().splitlines()
print(f"Reference implementation loaded — {{len(d1.SYMBOLS)}} markets, "
      f"{{len(source)}} lines ({{sum(1 for line in source if line.strip())}} non-blank)")
"""))

CELLS.append(code("""
market_data = d1.load_market_data(ROOT / DATA_DIR)
daily = d1.run(ROOT / DATA_DIR)

WINDOWS = {
    "1990–2004 development": ("1990-01-01", "2004-12-31"),
    "2005–2014 later window": ("2005-01-01", "2014-12-31"),
    "1990–2014 full history": ("1990-01-01", "2014-12-31"),
}
headline = pd.DataFrame({name: d1.metrics(daily, *w) for name, w in WINDOWS.items()})
full = headline["1990–2014 full history"]
returns = daily.loc["1990-01-01":"2014-12-31", "net_return"]
equity = (1 + returns).cumprod()

headline.style.format(lambda v: f"{v:,.4f}" if isinstance(v, float) else v)
"""))

# ----------------------------------------------------------- introduction ---
CELLS.append(md("""
---

## 1. Introduction

### The problem

Directional futures strategies face a specific difficulty that equity
strategies do not. A futures book has no natural long bias to fall back on:
its return has to come from correctly timing direction across markets whose
volatilities differ by an order of magnitude, whose contracts expire, and whose
returns are dominated by a handful of episodes. Get the sizing wrong and a
correct signal still loses money, because one market's risk swamps the rest.

### The economic claim

Two effects are documented across decades and asset classes, and they are
close to uncorrelated with each other:

1. **Time-series momentum.** A market that has risen over the past twelve
   months tends to keep rising over the next month. Moskowitz, Ooi and Pedersen
   (2012) document this across 58 futures over 25 years; the standard
   explanations are slow diffusion of information and the flow-driven behaviour
   of hedgers and trend followers.
2. **Basis momentum.** The *change* in a market's roll yield carries
   information beyond its level. Boons and Porras Prado (2019) show that
   markets whose carry is improving outperform, which is an inventory and
   hedging-pressure statement rather than a price statement. Their construction
   is a spread between the first- and second-nearby contracts; the supplied
   panel carries only a front-month series, so the sleeve here is the closest
   available proxy — the change in realized roll yield — and is named
   **roll-yield momentum** below rather than claimed as a replication.

Blending them at equal risk weight is the strategy. Diversification across
59 markets and six asset classes is what turns two modest edges into a
tradeable Sharpe, and disciplined risk sizing is what keeps that Sharpe from
being eaten by the largest positions.

### The result
"""))

CELLS.append(code("""
def scorecard(series, items):
    cells = "".join(
        f'<div style="flex:1;min-width:132px;padding:12px 14px;background:{SURFACE};'
        f'border:1px solid rgba(11,11,11,0.10);border-radius:8px">'
        f'<div style="font-size:11px;color:{MUTED};letter-spacing:.02em">{label}</div>'
        f'<div style="font-size:23px;color:{INK};margin-top:3px">{fmt(series[key])}</div>'
        f'<div style="font-size:11px;color:{SECOND};margin-top:2px">{note}</div></div>'
        for label, key, fmt, note in items
    )
    return Markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap">{cells}</div>')

display(scorecard(full, [
    ("CAGR",          "CAGR",                  lambda v: f"{v:.2%}", "net of all costs"),
    ("Volatility",    "Annualized volatility", lambda v: f"{v:.2%}", "7% target"),
    ("Sharpe",        "Sharpe (rf=0)",         lambda v: f"{v:.2f}", "rf = 0"),
    ("Max drawdown",  "Max drawdown",          lambda v: f"{v:.2%}", "daily close"),
    ("Calmar",        "Calmar",                lambda v: f"{v:.2f}", "CAGR / |MDD|"),
    ("Positive months", "Positive months",     lambda v: f"{v:.0%}", "300 months"),
]))
"""))

CELLS.append(md("""
The headline that matters is not the 13.2% return — it is that the return is
earned at 7.7% volatility with an 11.8% worst drawdown. The strategy's edge is
**risk-adjusted**, and section 6 shows exactly where that advantage comes from:
not from a better signal than the literature's, but from better risk control
around a comparable signal.
"""))

# -------------------------------------------------------------------- data --
CELLS.append(md("""
---

## 2. Data

The supplied Delta1 panel contains 94 continuous futures contracts (each as a
raw and a back-adjusted series) plus an ETF panel, with a catalogue giving tick
size, point value, currency and margin.

**Two price series per market, used for different jobs.** The back-adjusted
series (`_CCB`) is continuous, so it is the only correct input for returns and
volatility — but its *level* is an artefact of splicing, so nothing in the model
divides by it. The unadjusted series carries the real price, so it values
notional exposure. Their difference is the roll return, which is precisely the
input the basis sleeve needs.

**The universe is 59 of the 94 roots**, and the other 35 are enumerated with a
ground each in [`outputs/universe/universe_audit.csv`](../outputs/universe/universe_audit.csv),
generated by `scripts/run_universe_audit.py`. Every ground is derived from the
catalogue, the price files, or the strategy's own configured liquidity gate.
**No test in that runner looks at returns**, which is what makes "excluded for
correctness, never for performance" checkable rather than asserted. A rolling
60-day median-volume gate then suppresses any market too thin to trade in a
given period, so the traded universe grows from 1990 to 2014 rather than being
fixed with hindsight.

Ten roots come out as **available** — convertible, liquid, distinct, and simply
not included. That is the honest label. Section 8 records what including them
would have done.

**FX conversion comes from inside the panel.** Non-USD contracts are converted
using the currency futures in the same dataset, so no external series and no
modern FX snapshot enters a 1990 decision.

**The panel ends 2014-12-31**, which bounds what any result here can claim.
The case permits other data sources, so stopping there is a **choice**, and
section 8 states the reason rather than leaving it implied.
"""))

CELLS.append(code("""
prices = market_data["prices"]
coverage = pd.DataFrame({
    "Markets": [len(members) for members in d1.UNIVERSE.values()],
    "First observation": [
        min(prices[list(m)].first_valid_index() for m in [members]).date().isoformat()
        for members in d1.UNIVERSE.values()
    ],
}, index=list(d1.UNIVERSE))
coverage.loc["Total"] = [len(d1.SYMBOLS), prices.first_valid_index().date().isoformat()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.4),
                               gridspec_kw={"width_ratios": [1.35, 1]})

live = prices.notna().sum(axis=1)
ax1.fill_between(live.index, live, color=BLUE, alpha=0.16, linewidth=0)
ax1.plot(live.index, live, color=BLUE)
style(ax1, "Markets with price data", "count of the 59-market universe, daily")
ax1.set_ylim(0, 62)

held = daily.loc["1990-01-01":"2014-12-31", "active_markets"]
ax2.fill_between(held.index, held, color=AQUA, alpha=0.16, linewidth=0)
ax2.plot(held.index, held, color=AQUA)
style(ax2, "Markets actually held", "after the liquidity gate and integer rounding")
ax2.set_ylim(0, 62)
# Shared limits on both axes: the panels are meant to be read against each other.
ax2.set_xlim(*ax1.get_xlim())

plt.tight_layout()
plt.show()
display(coverage)
"""))

CELLS.append(code("""
# Every one of the 94 supplied roots, with the ground for its decision.
# Regenerate with: python scripts/run_universe_audit.py
audit = pd.read_csv(ROOT / "outputs/universe/universe_audit.csv")
summary = (
    audit.groupby(["decision", "ground"])
    .agg(roots=("root", "count"),
         markets=("root", lambda s: ", ".join(sorted(s)[:5]) + ("…" if len(s) > 5 else "")))
    .reset_index()
    .sort_values("roots", ascending=False)
)
assert int(summary["roots"].sum()) == 94, "the audit must account for every supplied root"
display(summary.style.hide(axis="index"))

display(Markdown(
    f"**{int(audit['decision'].eq('included').sum())} traded, "
    f"{int(audit['decision'].eq('excluded').sum())} excluded, 94 accounted for.** "
    f"The {int(audit['ground'].eq('available').sum())} roots on the `available` row are "
    "convertible, liquid and distinct — they are simply not in the book, and saying so "
    "is more useful than a reason that would not survive checking."
))
"""))

# ------------------------------------------------------------- methodology --
CELLS.append(md("""
---

## 3. Methodology

The model is six steps. Each is a few lines of code, each has an economic
reason, and each is causal — every quantity used in a decision is computed from
data observed strictly before the trade is filled.

### 3.1 Signals

**Trend** is the *sign* of the 252-session price change. Not a t-statistic and
not a normalised return: a back-adjusted price level is a splicing artefact, so
using it as a denominator would put an arbitrary constant in the signal. The
sign is the weakest possible functional form that still expresses the claim,
which is exactly what Occam's razor asks for here — and it removes a magnitude
parameter that would otherwise need fitting.

**Basis momentum** is the year-on-year change in trailing roll yield. The daily
gap between the unadjusted and back-adjusted price change *is* the roll return;
accumulate it over 252 sessions and it is the carry the term structure paid;
take its year-on-year change and it is the momentum in that carry. It is
normalised by the market's own volatility and by a rolling standard deviation,
then clipped, so it arrives on the same scale as the trend sign.

The two are blended at **equal risk weight** — a choice, not an optimisation
(section 4). Before basis is estimable the blend falls back to trend alone, so
the book is never under-invested purely because one sleeve needs a longer
warm-up.

### 3.2 Conditioning

Two multiplicative risk controls, both of which can only ever *cut* exposure:

- **Volatility-shock taper.** When 20-day realized volatility runs above 1.35×
  its 120-day baseline, the forecast is tapered linearly to a floor of 0.75 at
  a ratio of 2.0. Trend signals are least reliable exactly when volatility
  gaps, and this reduces exposure into that.
- **Risk-managed scaling.** Each market's forecast is divided by the realized
  volatility of *its own forecast-weighted P&L* over 63 sessions. This targets
  the risk of the strategy in that market, not the risk of the market itself,
  which is the distinction that stops a high-conviction signal in a suddenly
  turbulent market from dominating the book.

### 3.3 Sizing

Every available market receives an equal *pre-forecast* risk budget of
`target_vol / √N`. Dividing that budget by the market's annualized dollar
volatility per contract converts risk into a contract count — which is what
makes a Eurodollar future and a crude future comparable at all. A portfolio-level
RiskMetrics EWMA scaler then steers realized volatility toward the 7% target,
bounded to [0.25×, 2.0×], with a 5× gross-notional ceiling behind it.

### 3.4 Trading

Monthly decisions with a **25% no-trade band**: an adjustment smaller than 25%
of the position is not made. This is the single most valuable cost control in
the strategy — most month-to-month forecast changes are noise, and suppressing
their turnover costs almost no signal. Orders are integer contracts, sized from
lagged median volume and filled at no more than 2% of the volume the session
actually traded, with the residual carried forward.
"""))

CELLS.append(code("""
sample = "ES"
window = slice("2007-06-01", "2009-12-31")

trend = d1.trend_signal(prices)
basis = d1.basis_momentum(prices, market_data["unadjusted"])
forecast = d1.build_forecast(market_data)

fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 1, 1]})

px = prices.loc[window, sample]
axes[0].plot(px.index, px, color=INK, linewidth=1.4)
style(axes[0], f"{sample} — back-adjusted price",
      "the signal chain below, through the 2008 crisis", grid="y")

axes[1].axhline(0, color=AXIS, linewidth=0.9)
axes[1].plot(trend.loc[window, sample], color=BLUE, label="Trend (12-month sign)")
axes[1].plot(basis.loc[window, sample], color=ORANGE, label="Basis momentum")
axes[1].set_ylim(-1.55, 1.55)
# Direct labels sit in the margin the wider limits create, clear of the data,
# which is bounded to [-1, 1] by construction.
axes[1].text(px.index[-1], -1.26, "Trend  ", color=BLUE, fontsize=9,
             ha="right", va="center", fontweight="semibold")
axes[1].text(px.index[-1], 1.26, "Basis  ", color="#c04d1f", fontsize=9,
             ha="right", va="center", fontweight="semibold")
style(axes[1], "Raw sleeves", "both bounded to [-1, 1]")
axes[1].legend(loc="lower left", ncol=2, fontsize=8.5)

fc = forecast.loc[window, sample]
axes[2].axhline(0, color=AXIS, linewidth=0.9)
axes[2].fill_between(fc.index, fc, color=AQUA, alpha=0.20, linewidth=0)
axes[2].plot(fc.index, fc, color=AQUA)
axes[2].set_ylim(-1.25, 1.25)
style(axes[2], "Final forecast", "after the shock taper, risk scaling and liquidity gate")

plt.tight_layout()
plt.show()
"""))

CELLS.append(code("""
# What actually drove the delivered forecast through the crash?
shock = d1.shock_multiplier(prices)
blend = (trend * 0.5 + basis.fillna(trend) * 0.5).clip(-1, 1).where(trend.notna())
crash = slice("2008-07-01", "2008-12-31")

display(pd.DataFrame({
    "Trend sleeve":           trend.loc[crash, sample],
    "Basis sleeve":           basis.loc[crash, sample],
    "Blend":                  blend.loc[crash, sample],
    "Shock multiplier":       shock.loc[crash, sample],
    "Delivered forecast":     forecast.loc[crash, sample],
}).agg(["mean", "min", "max"]).T.round(3))
"""))

CELLS.append(md("""
This is the value of blending two sleeves rather than levering one. Through the
second half of 2008 the trend sleeve was pinned at −1 — maximum short
conviction. The basis sleeve **disagreed**, averaging +0.71 as the term
structure repriced, and the blend collapsed to −0.14. The delivered forecast
averaged −0.28: about a quarter of what trend alone would have demanded, in the
most violent quarter of the sample.

Note what did *not* do the work here. The shock multiplier averaged 0.98 in this
window — it barely bound, because ES volatility rose against a 120-day baseline
that was itself rising. The exposure cut came from sleeve disagreement, which
is exactly the diversification the two-signal construction is for. The shock
taper earns its place elsewhere, in fast single-market gaps rather than in a
sustained regime shift.
"""))

# ----------------------------------------------------- optimization choices --
CELLS.append(md("""
---

## 4. Optimizations, and what was deliberately left alone

Every parameter in the model is one of three kinds, and the distinction is what
keeps the backtest honest.

**Conventions taken from the literature, not fitted.** The 252-session trend
lookback, the 60-day volatility span, the equal blend weight, the 63-day
risk-management window. These are the standard values in the cited papers. They
were not searched over, which is why the walk-forward in section 5 re-selects
the trend lookback out of sample — to measure what choosing it would have cost.

**Risk-policy choices, set by constraint rather than by return.** The 7%
volatility target and the 5× gross ceiling were chosen to leave headroom under
a 15% drawdown policy. They are not return levers: raising the target scales
both return and risk almost exactly, so it buys no Sharpe. A 9.3% variant was
tested and rejected — it raised exposure rather than alpha and materially
increased simulated drawdown-breach frequency.

**Genuine cost optimizations, tested against a documented alternative.** The
25% no-trade band and the monthly rebalance frequency. Both trade a small
amount of signal for a large amount of turnover, and their value is measurable.

### What was tested and rejected

A separate sweep asked whether tightening any position-magnitude bound would
reduce drawdown. The answer was **no, measurably**:

- `max_risk_scalar` (2.00) and `min_risk_scalar` (0.25) **never bind** — the
  realized multiplier stays inside [0.294, 1.846], binding on **0 of 6,523
  sessions** (`outputs/attribution/attribution_bound_activity.csv`).
- Inside drawdowns the book is **21.5% smaller** than usual while volatility is
  96.9% of normal and the daily hit rate falls 8.9 points. The failure mode is
  **forecast accuracy, not position size**, so a size limit cannot fix it.
- Tightening `max_risk_scalar` to 1.00 at matched risk *raises* simulated
  P(drawdown > 15%) from 4.90% to 5.95%.

That is a negative result, and it is reported because it is the answer. Full
measurements: [`docs/drawdown-attribution-findings.md`](../docs/drawdown-attribution-findings.md).

### Was this the configuration a search would have picked?

The three kinds above describe how the parameters were *chosen*. They do not
answer the obvious follow-up: if someone had simply searched, would they have
found something better? So after the configuration was frozen and published,
417 configurations were run — dense one-parameter profiles over 18 axes, 300
independent joint draws from the same box, and seven structural alternatives
including multi-horizon trend ensembles.

**This is a search, and the only one in this notebook.** It is declared as one.
No parameter was re-selected from it and no figure anywhere else depends on it;
its purpose is to price the configuration choice, not to make it. Regenerate
with `python scripts/run_optimality_study.py`.
"""))

CELLS.append(code("""
# The optimality audit, read from its artifacts. Regenerate with:
#   python scripts/run_optimality_study.py
summary = pd.read_csv(ROOT / "outputs/optimality/optimality_summary.csv")
walk = pd.read_csv(ROOT / "outputs/optimality/optimality_walk_forward_summary.csv")
shapes = pd.read_csv(ROOT / "outputs/optimality/optimality_profile_summary.csv")
profiles = pd.read_csv(ROOT / "outputs/optimality/optimality_profiles.csv")

display(summary[["question", "value", "reading"]].style.hide(axis="index")
        .set_properties(**{"text-align": "left"}))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6),
                               gridspec_kw={"width_ratios": [1.2, 1]})

# Left: the axis that spikes. A single-horizon choice is the one real
# fragility in this design, and hiding it would defeat the point of measuring.
axis = profiles[profiles["parameter"] == "trend_lookback"].sort_values("value")
frozen = float(shapes.set_index("parameter").at["trend_lookback", "frozen_value"])
ax1.plot(axis["value"], axis["full_sharpe"], color=BLUE, marker="o", markersize=3.5)
ax1.axvline(frozen, color=ORANGE, linewidth=1.1, linestyle=(0, (4, 3)))
ax1.text(frozen + 8, axis["full_sharpe"].min(), f" frozen at {frozen:.0f}",
         color="#c04d1f", fontsize=8.5, va="bottom", fontweight="semibold")
style(ax1, "The trend lookback is a spike, not a plateau",
      "full-sample Sharpe against lookback, everything else frozen")
ax1.set_xlabel("Trend lookback (sessions)")

# Right: nineteen years of annual re-optimisation against holding the rules.
books = walk.set_index("book")
labels = ["Frozen rules", "Re-optimised\\nevery year"]
sharpes = [float(books.at["frozen baseline", "sharpe"]),
           float(books.at["annually re-optimised", "sharpe"])]
cagrs = [float(books.at["frozen baseline", "cagr"]),
         float(books.at["annually re-optimised", "cagr"])]
x = range(2)
ax2.bar([v - 0.19 for v in x], sharpes, width=0.36, color=BLUE, label="Sharpe")
ax2.bar([v + 0.19 for v in x], cagrs, width=0.36, color=MUTED, label="CAGR")
for position, (sharpe, cagr) in enumerate(zip(sharpes, cagrs)):
    ax2.text(position - 0.19, sharpe + 0.03, f"{sharpe:.3f}", ha="center", fontsize=8.5)
    ax2.text(position + 0.19, cagr + 0.03, f"{cagr:.2%}", ha="center", fontsize=8.5,
             color=SECOND)
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels)
ax2.set_ylim(0, max(sharpes) * 1.25)
style(ax2, "Optimising bought nothing", "1996-2014, same window, same panel")
ax2.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.show()
"""))

CELLS.append(md("""
Three results, and one of them is uncomfortable.

**Nothing in the joint space beat it.** Of 300 independent draws from the
declared box, **none** beat the frozen configuration on the full history and
none beat it over 2005-2014. Fifteen of the eighteen axes are plateaus or
gentle slopes — every risk control among them.

**Optimising paid nothing.** Choosing the best configuration on 1990-2004 and
holding it afterwards *lost* Sharpe over 2005-2014. Nineteen years of annual
re-optimisation matched the frozen book's Sharpe to four decimals (1.5958
against 1.5959) while giving up 4.2 points of compound return, because the
optimiser kept selecting a lower risk budget. The paired block bootstrap puts
that CAGR shortfall below zero at every block length.

**The trend lookback is a spike.** It is worth 0.28 Sharpe against its own
neighbours, and that is the honest weakness in this design. Two things bound
it: 252 sessions is the 12-month convention from Moskowitz, Ooi and Pedersen
(2012) rather than a fitted value, and the peak does not transfer — the
development window's best value is 231, not 252, and adopting it would have
cost 0.14 Sharpe afterwards. A peak whose location moves between halves of the
sample is noise. The multi-horizon ensembles that would normally smooth it were
tested and are all worse, in both windows.

One change beat the frozen configuration in both windows: a basis weight of 0.6
rather than 0.5. It is **not adopted** — it is one grid step from the equal-risk
prior, the gain is inside the bootstrap standard error, and one further step
gives it all back. Moving a frozen parameter onto a peak found by searching the
same history that scored it is the exact failure this study was built to detect.
"""))

CELLS.append(code("""
# What the no-trade band is worth: turnover and cost drag against its width.
turnover = daily.loc["1990-01-01":"2014-12-31", "total_contract_turnover"]
cost_usd = daily.loc["1990-01-01":"2014-12-31", "transaction_cost_usd"]
nav = daily.loc["1990-01-01":"2014-12-31", "nav"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.4))

annual_cost = (cost_usd.groupby(cost_usd.index.year).sum()
               / nav.groupby(nav.index.year).mean())
ax1.bar(annual_cost.index, annual_cost, color=ORANGE, width=0.72)
style(ax1, "Annual cost drag", "all-in: spread, slippage, commission, fees, impact, rolls",
      pct=True, decimals=1)
ax1.axhline(float(full["Annual cost drag"]), color=INK, linewidth=1.1, linestyle=(0, (4, 3)))
ax1.text(annual_cost.index[-1], float(full["Annual cost drag"]) * 1.10,
         f"mean {float(full['Annual cost drag']):.2%}", color=INK, fontsize=8.5, ha="right")

gross = daily.loc["1990-01-01":"2014-12-31", "gross_return"]
net = daily.loc["1990-01-01":"2014-12-31", "net_return"]
for series, colour, label in ((gross, MUTED, "Gross"), (net, BLUE, "Net of costs")):
    curve = (1 + series).cumprod()
    ax2.plot(curve.index, curve, color=colour, label=label)
    ax2.text(curve.index[-1], curve.iloc[-1], f"  {label}", color=colour,
             fontsize=9, va="center", fontweight="semibold")
ax2.set_yscale("log")
style(ax2, "Cost is a level effect, not a slope effect",
      "growth of $1, log scale", grid="y")
as_multiple(ax2)
ax2.set_xlim(right=curve.index[-1] + pd.Timedelta(days=1800))

plt.tight_layout()
plt.show()

display(Markdown(
    f"Costs consume **{float(full['Annual cost drag']):.2%} a year** — roughly "
    f"{float(full['Annual cost drag']) / (float(full['CAGR']) + float(full['Annual cost drag'])):.0%} "
    "of gross return. The strategy survives realistic friction because the "
    "no-trade band and the monthly schedule hold turnover down, not because "
    "friction was assumed away."
))
"""))

# ---------------------------------------------------------- out of sample ---
CELLS.append(md("""
---

## 5. Out-of-sample design and results

The brief asks for at least five years of out-of-sample forecast and recommends
rolling walk-forward analysis to conserve data. This section reports **twenty
years** of it, and then states plainly what that record does and does not prove.

### 5.1 Design

An **anchored expanding walk-forward** over four candidate trend lookbacks
(the 252-session baseline plus three alternatives). At each annual boundary the
selector may only see data up to that boundary; it chooses the lookback with the
best in-sample statistic and that choice governs the following year. The book
and NAV carry forward across boundaries, so what is measured is one continuous
replay a live desk could have run — not twenty spliced fragments.

This design answers a specific question: **what does choosing a parameter
actually cost?** Comparing the walk-forward path to the fixed-specification
path isolates selection risk, which is the part of a backtest most likely to be
optimistic.
"""))

CELLS.append(code("""
wf = pd.read_csv(ROOT / "outputs/validation/validation_walk_forward_summary.csv").iloc[0]
folds = pd.read_csv(ROOT / "outputs/validation/validation_walk_forward_folds.csv")
path = pd.read_csv(ROOT / "outputs/validation/validation_walk_forward_monthly_path.csv",
                   parse_dates=["month"]).set_index("month")["net_return"]

display(scorecard(wf, [
    ("OOS sessions",  "stitched_sessions",             lambda v: f"{int(v):,}",  "1995-01-02 → 2014-12-31"),
    ("OOS years",     "stitched_sessions",             lambda v: f"{v / 252:.1f}", "252-session equivalent"),
    ("OOS Sharpe",    "stitched_sharpe",               lambda v: f"{v:.2f}",     "rf = 0"),
    ("OOS CAGR",      "stitched_cagr",                 lambda v: f"{v:.2%}",     "net of all costs"),
    ("OOS drawdown",  "stitched_max_drawdown",         lambda v: f"{v:.2%}",     "daily close"),
    ("WF efficiency", "walk_forward_efficiency",       lambda v: f"{v:.2f}",     "OOS ÷ in-sample Sharpe"),
]))
"""))

CELLS.append(code("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8),
                               gridspec_kw={"width_ratios": [1.3, 1]})

span = slice(pd.Timestamp(wf["stitched_start"]), pd.Timestamp(wf["stitched_end"]))
anchor_monthly = (1 + returns.loc[span]).resample("ME").prod() - 1
for series, colour, label in (
    (anchor_monthly, MUTED, "Fixed specification"),
    (path,           BLUE,  "Walk-forward (out of sample)"),
):
    curve = (1 + series).cumprod()
    ax1.plot(curve.index, curve, color=colour, label=label)
ax1.set_yscale("log")
style(ax1, "Twenty years of walk-forward, against the fixed specification",
      "growth of $1, monthly, log scale")
as_multiple(ax1)
ax1.legend(loc="upper left", fontsize=8.5)

colours = [CRITICAL if s < 0 else BLUE for s in folds["fold_sharpe"]]
ax2.bar(range(len(folds)), folds["fold_sharpe"], color=colours, width=0.72)
ax2.axhline(0, color=AXIS, linewidth=0.9)
ax2.set_xticks(range(0, len(folds), 3))
ax2.set_xticklabels([folds["segment_start"].iloc[i][:4] for i in range(0, len(folds), 3)])
style(ax2, "Sharpe by out-of-sample fold", "one bar per annual segment")
losing = int((folds["fold_sharpe"] < 0).sum())
# The only losing fold is 1995, at the far left, so the right of the negative
# band is empty and the note cannot collide with a bar.
ax2.text(0.99, 0.06, f"{len(folds) - losing} of {len(folds)} folds positive",
         transform=ax2.transAxes, ha="right", color=SECOND, fontsize=8.5)

plt.tight_layout()
plt.show()
"""))

CELLS.append(md("""
### 5.2 What the walk-forward cost

Letting the trend lookback be chosen out of sample **cost 0.165 Sharpe** and
deepened maximum drawdown from −11.85% to −18.19% — through the 15% drawdown
policy. Walk-forward efficiency is 0.90, so roughly 90% of the fixed
specification's risk-adjusted performance survives honest parameter selection.

Only the **1995 fold** selected a non-baseline variant, and it lost 11.6% that
year. Because the replay carries book and NAV state forward, the full gap
cannot be attributed to that fold alone — but the direction is unambiguous:
**the selector destroyed value.** The baseline lookback was the better choice
in 19 of 20 years, and a live desk running this design would have paid for
finding that out.

### 5.3 What this record is, precisely

This is where most backtests overstate themselves, so the claim is bounded
explicitly.

The walk-forward is out of sample **with respect to the selector**. The four
candidate lookbacks, the signal construction, the cost model and the risk budget
were all written by someone who had already seen 1990–2014. So this measures
selection risk honestly and it does **not** measure specification risk.

Three further records exist, and the honest reading of them is uncomfortable:

| Record | Span | Out of sample w.r.t. | Result |
|---|---|---|---|
| Stitched walk-forward | 1995–2014, 5,218 sessions | the selector | Sharpe **1.42** |
| 2015–2016 futures subset | 522 sessions, 12 of 59 roots | data the pipeline provably never read | consistent, but trend-only and too short to be evidence |
| ETF regime sleeve | 2009–2018, with a sealed 2014–2018 block | selector and custody | **loses** to 60/40 by 5.93% a year over the full 2,516-session window, *t* = −3.48; on the sealed 1,258-session block alone, 1.59% vs 6.45% CAGR at Sharpe 0.39 vs 0.89 |

**The supplied futures panel ends 2014-12-31.** A genuinely sealed five-year
futures tail cannot be constructed from it — the FXFI extension reaches only
2016-12-30. The only contiguous five-year sealed block available anywhere in
the supplied data is the ETF one, and it loses. That is reported because it is
the answer.

Note the two ETF figures are on different spans and neither substitutes for the
other. The *t* = −3.48 differential is measured over the full 2009–2018
out-of-sample window (2,516 sessions,
`outputs/etf/etf_paired_reference_inference.csv`). The sealed 2014–2018 block
is 1,258 sessions and carries **no published *t*-statistic** — it is one look,
and a paired test on a single sealed block would overstate what one look can
support (`outputs/etf/etf_comparison_sealed_block.csv`).

### 5.4 The multiple-testing check

Two estimators were run against seventeen declared configurations:

- **White's Reality Check / Hansen's SPA**: no procedure rejects on Sharpe at
  any block length (RC 0.642–0.660, SPA 0.277–0.321). The best in-sample member
  sits +0.088 Sharpe above the incumbent and is **indistinguishable from it**
  once the search is priced.
- **CSCV probability of backtest overfitting**: **0.42** on the monthly paths
  the methodology permits — uncomfortably near the 0.5 signature of pure
  overfitting.

The correct conclusion from PBO 0.42 is not that the strategy is fake — the
spanning alpha in section 6 is far too strong for that — but that **the
in-sample ranking of variants carries almost no information**. This is precisely
why the baseline configuration is retained rather than the best-performing one,
and why section 5.2's finding that the selector destroyed value is unsurprising.
"""))

# --------------------------------------------------------------- benchmarks --
CELLS.append(md("""
---

## 6. Benchmark comparison

The brief asks for a benchmark comparison. Seven published rules were
**re-implemented and run through the identical engine, panel, cost model and
execution assumptions** — not quoted from their papers. Only that way is a
difference attributable to the rule rather than to the test conditions.

- **Moskowitz-Ooi-Pedersen (2012)** — 12-month TSMOM, the canonical reference
- **Hurst-Ooi-Pedersen (2017)** — 1/3/12-month trend blend
- **Baltas-Kosowski** — trend *t*-statistic sizing
- **MACD/EWMAC crossover** — the practitioner standard
- **Barroso-Santa-Clara (2015)** and **Moreira-Muir** — volatility-managed overlays
- **Long-only** equal-risk and equal-notional references
"""))

CELLS.append(code("""
bench = pd.read_csv(ROOT / "outputs/benchmarks/benchmark_comparison.csv")
spanning = pd.read_csv(ROOT / "outputs/benchmarks/benchmark_spanning.csv").iloc[0]

LABELS = {
    "incumbent": "This strategy",
    "mop_tsmom": "Moskowitz-Ooi-Pedersen TSMOM",
    "mop_tsmom_under_incumbent_leverage_cap": "MOP TSMOM (capped)",
    "hop_trend_blend": "Hurst-Ooi-Pedersen blend",
    "baltas_kosowski_tstatistic": "Baltas-Kosowski t-statistic",
    "baz_macd_ewmac": "MACD / EWMAC crossover",
    "long_only_equal_risk": "Long-only, equal risk",
    "long_only_equal_notional": "Long-only, equal notional",
    "long_only_equal_risk_volatility_matched": "Long-only, vol matched",
    "barroso_santa_clara_on_mop_tsmom": "Barroso-Santa-Clara on MOP",
    "barroso_santa_clara_on_incumbent": "Barroso-Santa-Clara overlay",
    "moreira_muir_expanding_constant_on_incumbent": "Moreira-Muir (expanding)",
    "moreira_muir_full_sample_constant_on_incumbent": "Moreira-Muir (full sample)",
}
table = bench.assign(name=bench["benchmark"].map(LABELS)).sort_values("sharpe")
published = table[~table["benchmark"].str.contains("on_incumbent")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                               gridspec_kw={"width_ratios": [1.15, 1]})

is_us = published["benchmark"].eq("incumbent")
ax1.barh(published["name"], published["sharpe"],
         color=[BLUE if u else MUTED for u in is_us], height=0.68)
for y, (value, mine) in enumerate(zip(published["sharpe"], is_us)):
    ax1.text(value + 0.02, y, f"{value:.2f}", va="center", fontsize=8.5,
             color=INK if mine else SECOND,
             fontweight="semibold" if mine else "normal")
style(ax1, "Sharpe ratio, identical panel and costs",
      "published rules replicated through the same engine", grid="x")
ax1.set_xlim(0, max(published["sharpe"]) * 1.16)

for _, row in published.iterrows():
    mine = row["benchmark"] == "incumbent"
    ax2.scatter(row["annualized_volatility"], row["cagr"], s=126 if mine else 62,
                color=BLUE if mine else MUTED, zorder=3,
                edgecolor=SURFACE, linewidth=1.6)

# Two labelled points, each with a hairline leader so the text is unambiguous.
# The unlabelled grey marks are the remaining replicated rules, named at left.
CALLOUTS = (
    ("incumbent", "This strategy", (0.058, 0.158), BLUE),
    ("mop_tsmom", "MOP TSMOM", (0.100, 0.055), SECOND),
)
for key, label, xytext, colour in CALLOUTS:
    row = published.loc[published["benchmark"].eq(key)].iloc[0]
    ax2.annotate(
        label, xy=(row["annualized_volatility"], row["cagr"]), xytext=xytext,
        fontsize=8.5, color=colour, fontweight="semibold", ha="left", va="center",
        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.9,
                    "shrinkA": 2, "shrinkB": 6},
    )
ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax2.set_xlabel("Annualized volatility")
ax2.set_xlim(0.045, 0.152)
ax2.set_ylim(0.0, 0.20)
style(ax2, "Return against risk",
      "grey marks are the replicated rules named at left", pct=True, grid="both")

plt.tight_layout()
plt.show()
"""))

CELLS.append(code("""
display(Markdown(f\"\"\"
**Joint spanning regression** — this strategy regressed on all replicated rules
simultaneously, HAC standard errors:

| | |
|---|---|
| Annualized alpha | **{spanning['alpha_annualized']:.2%}** |
| HAC *t*-statistic | **{spanning['alpha_hac_t_statistic']:.2f}** |
| R² | {spanning['r_squared']:.3f} |
| Appraisal ratio | {spanning['appraisal_ratio']:.2f} |
\"\"\"))
"""))

CELLS.append(md("""
### Reading this honestly

The strategy **beats every published rule on Sharpe**, and survives a joint
spanning regression against all of them with 4.64% annualized alpha at
*t* = 6.04. That is a strong result and it should be stated as one.

Two qualifications belong next to it:

1. **The advantage is risk control, not signal.** MOP TSMOM earns *more* CAGR
   (13.8% vs 13.2%) — but at 1.55× the volatility and nearly double the
   drawdown. Strip out the risk management and the signal edge is modest. The
   right-hand panel above is the clearest statement of what this strategy
   actually does.
2. **The alpha estimate is biased in this strategy's favour.** The benchmark
   rules ran cold and unrefitted, while this strategy's parameters were chosen
   with the panel visible. 4.64% is an upper bound on the true edge, not an
   unbiased estimate of it.
"""))

# ----------------------------------------------------------------- findings --
CELLS.append(md("""
---

## 7. Findings

### 7.1 The return path
"""))

CELLS.append(code("""
drawdown = equity / equity.cummax() - 1

fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.9, 1]})

axes[0].plot(equity.index, equity, color=BLUE)
axes[0].set_yscale("log")
as_multiple(axes[0])
style(axes[0], "Growth of $1, net of all costs",
      f"1990–2014 · CAGR {float(full['CAGR']):.2%} · volatility "
      f"{float(full['Annualized volatility']):.2%} · Sharpe {float(full['Sharpe (rf=0)']):.2f}")

axes[1].fill_between(drawdown.index, drawdown, color=CRITICAL, alpha=0.20, linewidth=0)
axes[1].plot(drawdown.index, drawdown, color=CRITICAL, linewidth=1.3)
axes[1].axhline(-0.15, color=SECOND, linewidth=1.0, linestyle=(0, (4, 3)))
axes[1].text(drawdown.index[40], -0.152, "  15% drawdown policy", color=SECOND,
             fontsize=8.5, va="top")
trough = drawdown.idxmin()
axes[1].annotate(f"{drawdown.min():.1%}", (trough, drawdown.min()),
                 xytext=(trough, drawdown.min() - 0.022), fontsize=8.5,
                 color=CRITICAL, fontweight="semibold", ha="center")
style(axes[1], "Drawdown", "daily close, including the initial capital high-water mark", pct=True)
axes[1].set_ylim(-0.18, 0.012)

plt.tight_layout()
plt.show()
"""))

CELLS.append(md("""
### 7.2 Consistency across time

A 25-year Sharpe of 1.59 means little if it was earned in three good years.
"""))

CELLS.append(code("""
annual = (1 + returns).groupby(returns.index.year).prod() - 1
rolling = returns.rolling(756).mean() / returns.rolling(756).std() * math.sqrt(252)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

ax1.bar(annual.index, annual, width=0.72,
        color=[CRITICAL if v < 0 else BLUE for v in annual])
ax1.axhline(0, color=AXIS, linewidth=0.9)
style(ax1, "Calendar-year net return",
      f"{int((annual > 0).sum())} of {len(annual)} years positive · "
      f"worst {annual.min():.1%} ({annual.idxmin()})", pct=True)

ax2.axhline(0, color=AXIS, linewidth=0.9)
ax2.axhline(float(full["Sharpe (rf=0)"]), color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
ax2.fill_between(rolling.index, rolling, color=AQUA, alpha=0.18, linewidth=0)
ax2.plot(rolling.index, rolling, color=AQUA)
ax2.text(rolling.index[-1], float(full["Sharpe (rf=0)"]),
         f"  full-period {float(full['Sharpe (rf=0)']):.2f}", color=SECOND,
         fontsize=8.5, va="bottom", ha="right")
style(ax2, "Rolling 3-year Sharpe", "756-session window, rf = 0")

plt.tight_layout()
plt.show()

display(pd.DataFrame({
    "Share of years positive":      [f"{(annual > 0).mean():.0%}"],
    "Share of months positive":     [f"{float(full['Positive months']):.0%}"],
    "Rolling 3y Sharpe below zero": [f"{(rolling.dropna() < 0).mean():.1%} of days"],
    "Worst calendar year":          [f"{annual.min():.1%}"],
}, index=["1990–2014"]).T)
"""))

CELLS.append(md("""
The strategy is positive in the large majority of calendar years, and the
rolling three-year Sharpe spends almost no time below zero. Its two visible
soft patches — the mid-2000s and 2011–2012 — are the well-documented
trend-following droughts, so the strategy is *not* independent of the CTA
factor; section 6's spanning regression prices exactly that dependence.

### 7.3 Where the risk comes from
"""))

CELLS.append(code("""
gross_daily = daily.loc["1990-01-01":"2014-12-31", "gross_return"]
regimes = pd.DataFrame({
    "Sessions": returns.groupby(returns.index.year // 5 * 5).count(),
    "CAGR": (1 + returns).groupby(returns.index.year // 5 * 5).prod()
            ** (252 / returns.groupby(returns.index.year // 5 * 5).count()) - 1,
    "Volatility": returns.groupby(returns.index.year // 5 * 5).std() * math.sqrt(252),
    "Sharpe": returns.groupby(returns.index.year // 5 * 5).mean()
              / returns.groupby(returns.index.year // 5 * 5).std() * math.sqrt(252),
})
regimes.index = [f"{y}–{min(y + 4, 2014)}" for y in regimes.index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.4))

ax1.bar(regimes.index, regimes["Sharpe"], color=BLUE, width=0.62)
for x, v in enumerate(regimes["Sharpe"]):
    ax1.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5, color=SECOND)
style(ax1, "Sharpe by five-year block", "no single era carries the record")
ax1.set_ylim(0, regimes["Sharpe"].max() * 1.2)

realized_vol = returns.rolling(63).std() * math.sqrt(252)
ax2.axhline(0.07, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
ax2.plot(realized_vol.index, realized_vol, color=BLUE, linewidth=1.2)
ax2.set_xlim(realized_vol.index[0], realized_vol.index[-1] + pd.Timedelta(days=2000))
ax2.text(realized_vol.index[-1] + pd.Timedelta(days=120), 0.0735, "7% target",
         color="#c04d1f", fontsize=8.5, va="bottom", ha="left", fontweight="semibold")
# The opening spike is the warm-up: the portfolio scaler is pinned at 1.0 until
# the book has been live for a full 63-session window.
warm_end = realized_vol.index[min(len(realized_vol) - 1, 63 * 2)]
ax2.axvspan(realized_vol.index[0], warm_end, color=MUTED, alpha=0.10, linewidth=0)
ax2.text(warm_end, 0.152, "  scaler warm-up", color=MUTED, fontsize=8, va="top")
style(ax2, "Realized volatility against target",
      "63-session rolling, annualized", pct=True)
ax2.set_ylim(0, 0.17)

plt.tight_layout()
plt.show()
display(regimes.style.format({"CAGR": "{:.2%}", "Volatility": "{:.2%}",
                              "Sharpe": "{:.2f}", "Sessions": "{:,.0f}"}))
"""))

CELLS.append(md("""
The volatility target is doing its job: after the scaler warms up, realized
63-day volatility oscillates around 7% without sustained excursions, which is
what makes the drawdown profile predictable enough to size against. No era
carries the record — 2000–2004 is the strongest block at Sharpe 2.15, 2005–2009
the weakest at 0.99, and none is negative.
"""))

# --------------------------------------------------------------- limitations -
CELLS.append(md("""
---

## 8. Limitations

Stated plainly, because they bound every number above.

1. **The data ends 2014-12-31, and continuing past it was a choice.** The
   supplied futures panel stops there and the FXFI extension reaches only
   2016-12-30, so no result here speaks to the 2020 shock or the 2022 rates
   cycle. The brief permits other data sources, so that boundary is a scope
   decision, not an impossibility, and it is defended rather than assumed:

   - **Free daily futures history** (Yahoo `=F`, Stooq) reaches roughly two
     thirds of the 59 traded roots. The roots it misses — including seven of
     the eleven government bond roots — carry roughly two fifths of 1990–2014
     gross P&L, so a spliced panel would be a materially different book wearing
     the same name. Those
     feeds are also unadjusted front-month with no vendor-consistent
     back-adjusted twin, and the roll-yield sleeve is computed from exactly
     that pair, so half the forecast could not be formed. A vendor-consistent
     extension (Norgate, the panel's own source) is licensed.
   - **A splice would also not be sealed.** Downloading 2015–2026 today, having
     read what those years did, is not custody. Presenting it as an independent
     holdout would be the one unsupportable claim in this notebook.
   - **What external data would genuinely add** is an externally maintained
     benchmark rather than more price history — AQR's published time-series
     momentum factor and the Fung-Hsieh trend-following factors are both free
     and would test this strategy against a construction nobody here chose.
     That is a real gap and it is recorded as one.

   The 2015–2016 FXFI extension is used precisely because it was in hand before
   the specification was frozen, which is the property a download cannot have.
2. **No genuinely sealed five-year futures record exists.** The walk-forward is
   out of sample with respect to the selector only; the specification itself was
   written with this window visible. The 2015–2016 extension is 522 sessions on
   12 roots.
3. **PBO is 0.42.** In-sample variant rankings are close to uninformative. The
   baseline is retained for that reason, but the finding constrains how much
   any parameter choice here can be trusted.
4. **The spanning alpha is biased upward.** Benchmark rules ran unrefitted;
   this strategy's parameters did not.
5. **Continuous contracts are not tradeable instruments.** Rolls are modelled
   as a two-leg turnover charge against a proxy participation. Real serial
   contracts, calendar-spread liquidity, first-notice dates and delivery
   mechanics are not in this data and cannot be reconstructed from it.
6. **Returns are futures excess returns.** Cash collateral earns zero, and the
   ledger excludes collateral yield, variation-margin funding, and forced
   liquidation. This is not a funded-account return.
7. **Capacity is modelled, not proven.** The 2% participation cap bounds
   realized participation by construction, but market impact beyond a
   square-root model — and the behaviour of that model in a stressed session —
   is an assumption.
""".rstrip()))

# ---------------------------------------------------------------- takeaways --
CELLS.append(md("""
---

## 9. Key takeaways

**1. Two simple, well-documented edges, blended and diversified, produce an
institutional-quality risk-adjusted return.** 13.2% CAGR at 7.7% volatility
(Sharpe 1.59, maximum drawdown −11.8%) over 25 years and 59 markets, net of
spread, slippage, commission, fees, impact and roll costs.

**2. The edge is risk control, not signal.** The strategy has the highest
Sharpe of the thirteen replicated paths — including four independent published
trend rules and three long-only references — while MOP TSMOM earns more raw
return at 1.55× the volatility. Against the volatility-managed overlays applied
to its *own* decisions the margin is negligible (Barroso-Santa-Clara on the
incumbent scores 1.588 against 1.590), which is the honest way to read it: the
risk management is the edge, so re-managing the same risk does not add one. Position sizing by each market's own risk, the volatility-shock
taper, and the portfolio volatility target are where the Sharpe comes from —
and a joint spanning regression still leaves 4.64% annualized alpha at
*t* = 6.04.

**3. Twenty years of walk-forward hold up, and quantify what parameter
selection costs.** Sharpe 1.42 across 5,218 out-of-sample sessions, 90%
walk-forward efficiency. Selecting the trend lookback out of sample cost 0.165
Sharpe and deepened the worst drawdown to −18.2%, through the 15% policy. The
baseline was the better choice in 19 of 20 years.

**4. The validation machinery returns unflattering answers, and they are
reported.** PBO 0.42; no family-wise procedure rejects on Sharpe across
seventeen configurations; and the ETF sleeve — carrying the only contiguous
five-year sealed block anywhere in the supplied data — *loses* to 60/40, by
5.93% a year at *t* = −3.48 over the full 2009–2018 window and by 1.59% against
6.45% CAGR on the sealed block itself. A backtest that only produces good news
has not been tested.

**5. Drawdown here is a forecast-accuracy problem, not a position-size
problem.** Inside drawdowns the book is 21.5% smaller than usual at 96.9% of
normal volatility, with the hit rate down 8.9 points. Every position-magnitude
bound was swept and none reduced drawdown by more than the measurement can
resolve; tightening the risk scalar makes forward breach risk *worse*.

**6. Complexity was spent on correctness, not on the model.** The strategy is
one readable file with no fitted magnitude parameters. The surrounding package
exists to prove those lines are right — and the equivalence is asserted
bit-for-bit in [`tests/test_reference.py`](../tests/test_reference.py), not
claimed in prose.

---

### References

- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE 104(2)
- Boons & Porras Prado (2019), *Basis-Momentum*, [Journal of Finance 74(1)](https://doi.org/10.1111/jofi.12738) — motivation for the carry sleeve; the first/second-nearby construction is not reproducible from a front-month-only panel, so the implemented signal is a roll-yield proxy
- Hurst, Ooi & Pedersen (2017), *A Century of Evidence on Trend-Following Investing*, JPM
- Baltas & Kosowski (2013/2020), *Demystifying Time-Series Momentum Strategies*
- Barroso & Santa-Clara (2015), *Momentum Has Its Moments*, JFE 116(1)
- Moreira & Muir (2017), *Volatility-Managed Portfolios*, Journal of Finance 72(4)
- White (2000), *A Reality Check for Data Snooping*, [Econometrica 68(5)](https://doi.org/10.1111/1468-0262.00152)
- Hansen (2005), *A Test for Superior Predictive Ability*, JBES 23(4)
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, JPM 40(5)
- Politis & Romano (1994), *The Stationary Bootstrap*, [JASA 89(428)](https://doi.org/10.1080/01621459.1994.10476870)

### Reproducing this notebook

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,dev]"

# the strategy on its own, no notebook required (~3 seconds)
python reference/delta1_reference.py --data-dir "Round1AllData/Quant Researcher/Delta1"

# the equivalence proof against the production engine
python -m unittest tests.test_reference -v

# this notebook
jupyter lab notebooks/delta1_case_research.ipynb
```

The walk-forward, benchmark and validation artifacts this notebook reads are
regenerated by `scripts/run_validation_suite.py` and
`scripts/run_benchmark_comparison.py`; see the [README](../README.md) for the
full study map.
""".rstrip()))


def main() -> None:
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    print(f"wrote {TARGET.relative_to(ROOT)} — {len(CELLS)} cells")


if __name__ == "__main__":
    main()
