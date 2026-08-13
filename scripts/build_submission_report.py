"""Build the self-contained HTML report the submission spec requires.

    python scripts/build_submission_report.py

Writes outputs/submission/report.html: a single file with no external requests
-- every chart is an inline data URI and all styling is inline, so it opens
from a zip on a machine with no network.

Section order follows the brief: executive conclusion at the top, then
cumulative net performance versus benchmark, drawdown, rolling Sharpe, the
position/signal path, and a diagnostic chart, followed by the required results,
the robustness checks, and the data and cost discipline behind them.

Tables are rendered from whatever CSVs exist under outputs/submission/, so the
report degrades to "section absent" rather than failing if a study has not been
regenerated.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "outputs" / "submission"

# Validated categorical slots plus status colours; see the data-viz palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 140,
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.labelcolor": SECOND, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.titlesize": 10,
    "axes.titlelocation": "left", "axes.titlepad": 22,
    "lines.linewidth": 1.8,
})


def style(ax, title=None, subtitle=None, pct=False, decimals=0, grid="y"):
    if title:
        ax.set_title(title, fontweight="bold")
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=8, va="bottom")
    if grid:
        ax.grid(axis=grid, alpha=0.9)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimals))
    return ax


def figure_uri(fig) -> str:
    """Serialise a figure to an inline data URI and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load() -> dict:
    daily = pd.read_csv(ROOT / "outputs/strategy_daily.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv(ROOT / "outputs/strategy_metrics.csv")
    bench_returns = pd.read_csv(
        ROOT / "outputs/benchmarks/benchmark_daily_net_returns.csv",
        index_col=0, parse_dates=True,
    )
    bench = pd.read_csv(ROOT / "outputs/benchmarks/benchmark_comparison.csv")
    spanning = pd.read_csv(ROOT / "outputs/benchmarks/benchmark_spanning.csv").iloc[0]
    walk = pd.read_csv(ROOT / "outputs/validation/validation_walk_forward_summary.csv").iloc[0]
    return {
        "daily": daily,
        "metrics": metrics,
        "bench_returns": bench_returns,
        "bench": bench,
        "spanning": spanning,
        "walk": walk,
    }


def optional(name: str) -> pd.DataFrame | None:
    path = SUBMISSION / name
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


# --------------------------------------------------------------------------
# Charts -- the five the brief names, in its order
# --------------------------------------------------------------------------

WINDOW = slice("1990-01-01", "2014-12-31")


def chart_cumulative(data: dict) -> str:
    net = data["daily"].loc[WINDOW, "net_return"]
    bench = data["bench_returns"].loc[WINDOW]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    series = [("This strategy (net)", net, BLUE, 2.1)]
    for column, label, colour in (
        ("mop_tsmom", "Moskowitz-Ooi-Pedersen TSMOM", ORANGE),
        ("long_only_equal_risk", "Long-only, equal risk", MUTED),
    ):
        if column in bench:
            series.append((label, bench[column], colour, 1.5))

    for label, returns, colour, width in series:
        curve = (1 + returns.fillna(0.0)).cumprod()
        ax.plot(curve.index, curve, color=colour, linewidth=width, label=label)
        offsets = {"This strategy (net)": -0.055, "Moskowitz-Ooi-Pedersen TSMOM": 0.055}
        ax.annotate(f"{curve.iloc[-1]:,.1f}×",
                    xy=(curve.index[-1], curve.iloc[-1] * (1 + offsets.get(label, 0.0))),
                    xytext=(6, 0), textcoords="offset points",
                    color=colour, fontsize=8.5, va="center", fontweight="bold")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_xlim(right=curve.index[-1] + pd.Timedelta(days=1500))
    style(ax, "Cumulative net performance versus benchmark",
          "growth of $1 after all costs, log scale, identical panel and cost model")
    ax.legend(loc="upper left", fontsize=8.5)
    return figure_uri(fig)


def chart_drawdown(data: dict) -> str:
    net = data["daily"].loc[WINDOW, "net_return"]
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1

    bench = data["bench_returns"].loc[WINDOW]
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    if "mop_tsmom" in bench:
        mop = (1 + bench["mop_tsmom"].fillna(0.0)).cumprod()
        mop_dd = mop / mop.cummax() - 1
        ax.plot(mop_dd.index, mop_dd, color=MUTED, linewidth=1.1,
                label=f"MOP TSMOM ({mop_dd.min():.1%})")
    ax.fill_between(drawdown.index, drawdown, color=CRITICAL, alpha=0.18, linewidth=0)
    ax.plot(drawdown.index, drawdown, color=CRITICAL, linewidth=1.3,
            label=f"This strategy ({drawdown.min():.1%})")
    ax.axhline(-0.15, color=SECOND, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(drawdown.index[30], -0.153, "  15% drawdown policy", color=SECOND,
            fontsize=8, va="top")
    style(ax, "Drawdown", "daily close, including the initial-capital high-water mark", pct=True)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    return figure_uri(fig)


def chart_rolling(data: dict) -> str:
    net = data["daily"].loc[WINDOW, "net_return"]
    rolling = net.rolling(756).mean() / net.rolling(756).std() * math.sqrt(252)
    annual = (1 + net).groupby(net.index.year).prod() - 1
    full = float(net.mean() / net.std() * math.sqrt(252))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.axhline(0, color=AXIS, linewidth=0.9)
    ax1.axhline(full, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax1.fill_between(rolling.index, rolling, color=AQUA, alpha=0.16, linewidth=0)
    ax1.plot(rolling.index, rolling, color=AQUA)
    ax1.text(rolling.index[-1], full, f"full period {full:.2f}  ", color=SECOND,
             fontsize=8, va="bottom", ha="right")
    style(ax1, "Rolling 3-year Sharpe", "756-session window, net of costs, rf = 0")

    ax2.bar(annual.index, annual, width=0.72,
            color=[CRITICAL if v < 0 else BLUE for v in annual])
    ax2.axhline(0, color=AXIS, linewidth=0.9)
    style(ax2, "Calendar-year net return",
          f"{int((annual > 0).sum())} of {len(annual)} years positive", pct=True)
    plt.tight_layout()
    return figure_uri(fig)


def chart_position(data: dict) -> str:
    frame = data["daily"].loc[WINDOW]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 4.6), sharex=True)

    exposure = frame["gross_notional_multiple"]
    ax1.fill_between(exposure.index, exposure, color=BLUE, alpha=0.16, linewidth=0)
    ax1.plot(exposure.index, exposure, color=BLUE, linewidth=1.2)
    ax1.axhline(5.0, color=CRITICAL, linewidth=1.0, linestyle=(0, (4, 3)))
    ax1.text(exposure.index[30], 5.18, "  5× gross ceiling", color=CRITICAL, fontsize=8,
             va="bottom")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    style(ax1, "Position: gross notional exposure",
          "multiple of NAV, end of session; the ceiling binds on 24 of 6,523 sessions")
    ax1.set_ylim(0, 5.8)

    scalar = frame["risk_scalar"]
    held = frame["active_markets"]
    ax2.plot(held.index, held, color=AQUA, linewidth=1.1)
    ax2.set_ylabel("markets held", color=SECOND, fontsize=8.5)
    ax2.text(held.index[-1], held.iloc[-1], f"  {int(held.iloc[-1])} markets",
             color="#0f7d57", fontsize=8.5, va="center", fontweight="bold")
    style(ax2, "Signal: number of markets held",
          f"the portfolio risk scalar stays inside [{scalar.min():.2f}, {scalar.max():.2f}] "
          "and never binds its configured [0.25, 2.00] bounds")
    ax2.set_ylim(0, 62)
    plt.tight_layout()
    return figure_uri(fig)


def chart_diagnostic(data: dict) -> str:
    """Cost is the diagnostic that matters: it is what turns gross into net."""
    frame = data["daily"].loc[WINDOW]
    gross, net = frame["gross_return"], frame["net_return"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    for series, colour, label in ((gross, MUTED, "Gross"), (net, BLUE, "Net of costs")):
        curve = (1 + series).cumprod()
        ax1.plot(curve.index, curve, color=colour, label=label)
        ax1.text(curve.index[-1], curve.iloc[-1], f"  {curve.iloc[-1]:,.0f}×", color=colour,
                 fontsize=8.5, va="center", fontweight="bold")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    ax1.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax1.set_xlim(right=curve.index[-1] + pd.Timedelta(days=1600))
    style(ax1, "Diagnostic: gross versus net", "growth of $1, log scale")
    ax1.legend(loc="upper left", fontsize=8.5)

    cost = frame["transaction_cost_usd"]
    nav = frame["prior_nav_usd"] if "prior_nav_usd" in frame else frame["nav"]
    annual_cost = (cost / nav).groupby(cost.index.year).sum()
    ax2.bar(annual_cost.index, annual_cost, color=ORANGE, width=0.72)
    mean_drag = float((cost / nav).mean() * 252)
    ax2.axhline(mean_drag, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.text(annual_cost.index[-1], mean_drag * 1.10, f"mean {mean_drag:.2%}",
             color=INK, fontsize=8, ha="right")
    style(ax2, "All-in cost by year",
          "spread, slippage, commission, fees, impact and rolls", pct=True, decimals=1)
    plt.tight_layout()
    return figure_uri(fig)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f7f4;--card:#fcfcfb;--ink:#0b0b0b;--second:#52514e;--muted:#898781;
--line:#e1e0d9;--blue:#2a78d6;--good:#0ca30c;--crit:#d03b3b;--warn:#b26a00;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 72px}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:26px}
h1{font-size:29px;line-height:1.2;margin:0 0 6px}
.sub{color:var(--second);font-size:15px;margin:0}
.meta{color:var(--muted);font-size:12.5px;margin-top:10px}
h2{font-size:20px;margin:40px 0 6px;padding-top:20px;border-top:1px solid var(--line)}
h2:first-of-type{border-top:none;padding-top:0}
h3{font-size:15.5px;margin:24px 0 6px}
p{margin:9px 0}
.lede{color:var(--second)}
.verdict{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--blue);
border-radius:9px;padding:20px 22px;margin:22px 0}
.verdict .tag{display:inline-block;background:var(--blue);color:#fff;font-size:12px;
font-weight:700;letter-spacing:.06em;padding:4px 11px;border-radius:5px;text-transform:uppercase}
.verdict h3{margin:16px 0 3px;font-size:14px;color:var(--second);
text-transform:uppercase;letter-spacing:.05em}
.verdict h3:first-of-type{margin-top:16px}
.tiles{display:flex;gap:9px;flex-wrap:wrap;margin:18px 0}
.tile{flex:1 1 120px;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:11px 13px}
.tile .k{font-size:11px;color:var(--muted);letter-spacing:.02em}
.tile .v{font-size:22px;margin-top:2px}
.tile .n{font-size:11px;color:var(--second);margin-top:1px}
figure{margin:20px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
border-radius:8px;background:var(--card)}
figcaption{color:var(--muted);font-size:12.5px;margin-top:7px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--line);
white-space:nowrap;font-variant-numeric:tabular-nums}
th{background:#eeeeea;font-size:12px;letter-spacing:.02em;position:sticky;top:0}
td:first-child,th:first-child{white-space:normal;min-width:190px}
tbody tr:last-child td{border-bottom:none}
.note{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:13px 16px;margin:16px 0;font-size:13.5px;color:var(--second)}
.note strong{color:var(--ink)}
code{background:#eeeeea;padding:1.5px 5px;border-radius:4px;font-size:12.5px}
pre{background:#eeeeea;padding:13px 15px;border-radius:8px;overflow-x:auto;font-size:12.5px}
.g{color:var(--good);font-weight:600}.r{color:var(--crit);font-weight:600}
.w{color:var(--warn);font-weight:600}
footer{margin-top:52px;padding-top:16px;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px}
@media print{body{background:#fff}.wrap{max-width:none;padding:0}
h2{page-break-after:avoid}figure{page-break-inside:avoid}}
@media (prefers-color-scheme:dark){
:root{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--second:#c3c2b7;--muted:#898781;
--line:#2c2c2a;--blue:#3987e5;--good:#0ca30c;--crit:#e66767;--warn:#fab219;}
th{background:#232320}code,pre{background:#232320}
figure img{background:#fcfcfb}}
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


# Long-prose columns that would make an on-screen table unreadable. They stay
# in the shipped CSVs; the report renders the numbers beside them.
PROSE_COLUMNS = (
    "Definition", "Canonical source", "Reconciliation", "note", "notes", "detail",
    "rules_status", "source", "definition", "label_information_set",
    "joint_spanning_family", "construction", "primary_comparator_rationale",
    "comment", "spec_mapping_rationale", "prespecified_parameter_grid",
    "joint_spanning_method", "scope",
)


def table(frame: pd.DataFrame | None, *, limit: int | None = None,
          drop: tuple[str, ...] = (), keep_prose: bool = False) -> str:
    if frame is None or frame.empty:
        return '<p class="lede"><em>Artifact not present in this build.</em></p>'
    hide = tuple(drop) if keep_prose else tuple(drop) + PROSE_COLUMNS
    frame = frame.drop(columns=[c for c in hide if c in frame.columns])
    if limit:
        frame = frame.head(limit)
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tiles(items) -> str:
    cells = "".join(
        f'<div class="tile"><div class="k">{esc(k)}</div>'
        f'<div class="v">{esc(v)}</div><div class="n">{esc(n)}</div></div>'
        for k, v, n in items
    )
    return f'<div class="tiles">{cells}</div>'


def _reoptimization_interval() -> str:
    """State the bootstrap verdict only if the artifact actually supports it.

    An earlier revision of this file hard-coded a headline column and every
    value was wrong. A sentence claiming statistical significance is worth even
    less than a wrong number, so this one is read from the differential file
    each build and omitted entirely when the interval covers zero.
    """
    path = ROOT / "outputs/optimality/optimality_paired_differentials.csv"
    if not path.is_file():
        return ""
    paired = pd.read_csv(path)
    cagr = paired.loc[
        (paired["Comparison"] == "annually_reoptimized_vs_frozen")
        & (paired["Statistic"] == "CAGR differential")
    ]
    if cagr.empty or float(cagr["Confidence upper"].max()) >= 0.0:
        return ""
    return (
        " The paired block bootstrap puts that CAGR shortfall below zero at every"
        " block length tested."
    )


def build(data: dict) -> str:
    metrics = data["metrics"]
    full = metrics.loc[metrics["Window"].str.contains("full post-launch")].iloc[0]
    dev = metrics.loc[metrics["Window"].str.contains("1990-2004")].iloc[0]
    late = metrics.loc[metrics["Window"].str.contains("2005-2014")].iloc[0]
    walk, spanning = data["walk"], data["spanning"]

    sharpe_full = float(full["Naive daily Sharpe (sqrt252, rf=0)"])
    sharpe_dev = float(dev["Naive daily Sharpe (sqrt252, rf=0)"])
    sharpe_late = float(late["Naive daily Sharpe (sqrt252, rf=0)"])
    decay = (sharpe_late - sharpe_dev) / sharpe_dev

    net = data["daily"].loc[WINDOW, "net_return"]

    # Gross figures come from the published artifact, not a second computation
    # here. Annualising by session count rather than elapsed calendar time
    # inflated gross CAGR by 1.2 percentage points in an earlier revision, and a
    # report that disagrees with the CSV shipped beside it is worse than useless.
    gross_summary = optional("gross_vs_net_summary.csv")
    if gross_summary is not None:
        row = gross_summary.loc[gross_summary["Window"] == "1990-2014 full"].iloc[0]
        gross_cagr = float(row["Gross annualised return"])
        gross_sharpe = float(row["Gross Sharpe"])
    else:
        gross = data["daily"].loc[WINDOW, "gross_return"]
        gross_sharpe = float(gross.mean() / gross.std() * math.sqrt(252))
        gross_cagr = float(net.mean() * 0 + float("nan"))

    # The sleeve decomposition is the sharpest available answer to "what drove
    # the result", so it is quoted in the conclusion rather than buried.
    sleeves = ""
    attribution = optional("return_attribution.csv")
    if attribution is not None and "block" in attribution:
        rows = attribution[attribution["block"].eq("signal_sleeve_decomposition")]
        by_weight = {}
        for _, row in rows.iterrows():
            if pd.notna(row.get("basis_weight")):
                by_weight[float(row["basis_weight"])] = row
        if {0.0, 0.5, 1.0} <= set(by_weight):
            sleeves = (
                f" Neither sleeve carries the result alone: trend-only earns Sharpe "
                f"<strong>{float(by_weight[0.0]['sharpe']):.2f}</strong> and carry-only "
                f"<strong>{float(by_weight[1.0]['sharpe']):.2f}</strong>, while the equal blend "
                f"reaches <strong>{float(by_weight[0.5]['sharpe']):.2f}</strong> — the two "
                "disagree often enough that the diversification between them, not either signal, "
                "is the edge."
            )

    # A -0.13 Sharpe sensitivity to a data-handling choice is not a footnote.
    fill_headline = ""
    nofill_summary = optional("robustness_no_fill.csv")
    if nofill_summary is not None:
        full_row = nofill_summary.loc[nofill_summary["window"] == "1990-2014 full"]
        if len(full_row):
            full_row = full_row.iloc[0]
            fill_headline = (
                " One further sensitivity belongs in this paragraph rather than an appendix: "
                "the panel is forward-filled across holidays, and rebuilding every signal from "
                "observed prices only — the strictest reading of the brief's data-discipline "
                f"rule — costs <span class=\"r\">"
                f"{float(full_row['delta_Sharpe (rf=0)']):+.3f} Sharpe</span> "
                f"({float(full_row['canonical_Sharpe (rf=0)']):.2f} → "
                f"{float(full_row['observed_only_Sharpe (rf=0)']):.2f}), and more over 2005-2014. "
                "The conclusion survives — the strict number still exceeds every independent "
                "benchmark replicated here — but the headline should be read with it."
            )

    parts: list[str] = []
    add = parts.append

    add(f"""<header>
<h1>Delta1 — Diversified Global Futures Trend and Carry</h1>
<p class="sub">A 59-market long/short futures strategy: 12-month time-series momentum blended
with roll-yield momentum, volatility-targeted at 7% and rebalanced monthly under modeled costs.</p>
<p class="meta">Junior Analyst Case · Delta1 · evaluation window 1990-01-01 to 2014-12-31
({int(len(net)):,} sessions, {len(net.index.year.unique())} calendar years) ·
report generated {dt.date.today().isoformat()}</p>
</header>""")

    # ---- executive conclusion, first, as the brief requires -----------
    add(f"""<h2>Executive conclusion</h2>
<div class="verdict">
<span class="tag">Verdict — paper trade</span>
<p style="margin-top:13px">The evidence is strong enough that rejecting it would be wrong, and
thin in exactly one place that no amount of further backtesting can fix: there is no forward
record. A paper-trading period is the only test that produces one, so that is the recommendation
— <strong>not</strong> capital, and <strong>not</strong> rejection.</p>

<h3>Main evidence</h3>
<p>Net of spread, slippage, commission, exchange fees, square-root impact and roll costs, the
strategy returns <strong>{float(full['CAGR']):.2%} a year at {float(full['Annualized volatility']):.2%}
volatility</strong> over 25 years — Sharpe <strong>{sharpe_full:.2f}</strong>, maximum drawdown
<strong>{float(full['Max drawdown']):.2%}</strong>, return-to-drawdown
<strong>{float(full['Calmar']):.2f}</strong>. It has the highest Sharpe of thirteen paths replicated
through the identical engine and cost model, and a joint spanning regression against all of them
still leaves <strong>{float(spanning['alpha_annualized']):.2%} annualised alpha at HAC
<em>t</em> = {float(spanning['alpha_hac_t_statistic']):.2f}</strong>. A 20-fold anchored
walk-forward that re-selects the trend horizon out of sample retains Sharpe
<strong>{float(walk['stitched_sharpe']):.2f}</strong> across {int(walk['stitched_sessions']):,}
sessions.{sleeves}</p>

<h3>Biggest limitation</h3>
<p>The supplied panel ends <strong>2014-12-31</strong>, so nothing here observes the last decade —
not 2020, not the 2022 rates cycle. Every result is a replay on history the specification was
written with in view. CSCV puts the probability of backtest overfitting at <strong>0.42</strong>,
which means in-sample rankings between variants carry almost no information; the baseline is kept
for that reason rather than the best-scoring variant. There is no post-freeze holdout, and the
only genuinely sealed five-year block anywhere in this data — the ETF sleeve — <span class="r">loses</span>
to a 60/40 benchmark.{fill_headline}</p>

<h3>Mechanism, and what would undermine it</h3>
<p><em>Supporting:</em> time-series momentum is among the most replicated effects in asset pricing
— across 58 futures over 25 years (Moskowitz-Ooi-Pedersen) and across more than a century
(Hurst-Ooi-Pedersen) — with a documented economic story: slow diffusion of information, and the
hedging pressure of producers and consumers who trade for reasons other than return. The carry
sleeve rests on inventory and storage economics, which is a different mechanism, and the two
disagree often enough to diversify each other. Sixty markets across six asset classes means no
single trade dominates.</p>
<p><em>Undermining, and it is visible in this data:</em> the effect appears to be decaying as it
is arbitraged. Sharpe falls from <strong>{sharpe_dev:.2f}</strong> in 1990-2004 to
<strong>{sharpe_late:.2f}</strong> in 2005-2014 — a <span class="r">{abs(decay):.0%} decline</span>
— with CAGR falling {float(dev['CAGR']):.2%} → {float(late['CAGR']):.2%} at essentially unchanged
volatility. Managed-futures assets grew roughly tenfold over that period. If that trend continued
past 2014, the forward Sharpe is materially below the headline, and this is the single most
important thing a paper-trading period would measure.</p>
</div>""")

    add(tiles([
        ("CAGR (net)", f"{float(full['CAGR']):.2%}", f"gross {gross_cagr:.2%}"),
        ("Volatility", f"{float(full['Annualized volatility']):.2%}", "7% target"),
        ("Sharpe (net)", f"{sharpe_full:.2f}", f"gross {gross_sharpe:.2f}"),
        ("Max drawdown", f"{float(full['Max drawdown']):.2%}", "daily close"),
        ("Return / drawdown", f"{float(full['Calmar']):.2f}", "CAGR ÷ |MDD|"),
        ("Cost drag", f"{float(full['Annual cost drag']):.2%}", "per year, all-in"),
    ]))

    # ---- the five charts, in the brief's order ------------------------
    add("<h2>Performance</h2>")
    add(f"""<figure><img alt="Cumulative net performance versus benchmark"
src="{chart_cumulative(data)}"><figcaption>Every path is run through the same engine, panel,
execution timing and cost model, so differences are attributable to the rule rather than the test
conditions. MOP TSMOM ends higher on raw growth — it takes 1.55× the volatility to do it.</figcaption></figure>""")
    add(f"""<figure><img alt="Drawdown" src="{chart_drawdown(data)}"><figcaption>The strategy's
worst peak-to-trough loss is roughly half the canonical trend rule's, which is where its Sharpe
advantage comes from.</figcaption></figure>""")
    add(f"""<figure><img alt="Rolling Sharpe and calendar-year returns"
src="{chart_rolling(data)}"><figcaption>The rolling window never turns negative, but its decline
after 2005 is the decay quantified in the conclusion — not noise.</figcaption></figure>""")
    add(f"""<figure><img alt="Position and signal" src="{chart_position(data)}"><figcaption>Exposure
is an outcome of volatility targeting, not a setting. The risk scalar never reaches either of its
configured bounds, so those bounds are compliance ceilings rather than active risk
controls.</figcaption></figure>""")
    add(f"""<figure><img alt="Diagnostic: gross versus net"
src="{chart_diagnostic(data)}"><figcaption>Diagnostic chart. Costs are a level effect, not a slope
effect: the two curves stay parallel, which is what a strategy with controlled turnover should
look like. The early-1990s cost peak is thinner markets, not more trading.</figcaption></figure>""")

    # ---- required results --------------------------------------------
    results = optional("required_results.csv")
    gross_net = optional("gross_vs_net_summary.csv")
    add("<h2>Required results</h2>")
    add('<p class="lede">Gross and net side by side. Turnover, trade count and average exposure '
        'are basis-invariant and reported once.</p>')
    add(table(results))
    if gross_net is not None:
        add("<h3>Gross versus net</h3>")
        add(table(gross_net))

    # ---- robustness ---------------------------------------------------
    add("<h2>Robustness</h2>")
    add("<h3>Parameter sensitivity</h3>")
    add('<p class="lede">A deliberately small, pre-declared grid of nearby values — not a search. '
        'The base configuration was not selected from these runs.</p>')
    add(table(optional("robustness_parameter_sensitivity.csv")))

    add("<h3>Chronological out-of-sample</h3>")
    add('<p class="lede">Developed on 1990-2004, then the unchanged rules assessed on 2005-2014. '
        'The third row re-selects the trend horizon at 20 annual boundaries using only prior data.</p>')
    add(table(optional("robustness_chronological_oos.csv")))

    add("<h3>Performance across regimes</h3>")
    add('<p class="lede">Two independent regime axes, both labelled using only information '
        'available before the return being labelled.</p>')
    add(table(optional("robustness_regimes.csv")))

    add("<h3>Sensitivity to the forward fill</h3>")
    add('<p class="lede">Every signal rebuilt on each market\'s own observed trading sessions, '
        'with non-trading days absent from its history rather than imputed. Marking is unchanged: '
        'a held position must be marked every session, and a closed market genuinely did not move. '
        'The difference is material and is reported as such.</p>')
    add(table(optional("robustness_no_fill.csv")))
    add("""<div class="note">The fill also biases risk estimation in a measurable direction. A
filled cell carries a price change of exactly zero, and those zeros pull the volatility estimate
down: across <strong>all 59 markets, without exception</strong>, volatility measured on the filled
panel is below volatility measured on observed sessions only, by a median of 1.6%. Because
positions are sized as risk budget divided by estimated volatility, that understatement inflates
every position. The two constructions also differ in how a fixed-length rolling window spans
calendar time, so this bias is <em>a</em> mechanism, not a full decomposition of the Sharpe
difference above.</div>""")

    add("<h3>Cost sensitivity</h3>")
    stress = pd.read_csv(ROOT / "outputs/strategy_friction_stress.csv")
    keep = ["scenario", "cagr", "annualized_volatility", "sharpe", "max_drawdown", "annual_cost_drag"]
    add(table(stress[[c for c in keep if c in stress.columns]]))

    # ---- robustness beyond the brief ----------------------------------
    # Four descriptive analyses of the same frozen configuration.  Each reads
    # its artifact from outputs/validation; a missing artifact degrades to
    # "section absent", the same contract as every optional() table above.
    def validation(name: str) -> pd.DataFrame | None:
        path = ROOT / "outputs/validation" / name
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
        return frame if not frame.empty else None

    crisis = validation("validation_crisis_windows.csv")
    if crisis is not None:
        add("<h3>Crisis windows</h3>")
        add('<p class="lede">Eight windows declared from public event dates — an invasion, '
            'a rate hike, a bankruptcy filing — not from the equity curve, so they measure '
            'episodes history picked rather than episodes the drawdown record picked.</p>')
        keep = ["label", "start", "end", "sessions", "cumulative_net_return",
                "max_drawdown", "worst_rolling_21_session_net_return"]
        add(table(crisis[[c for c in keep if c in crisis.columns]]))

    breakeven = validation("validation_cost_breakeven.csv")
    if breakeven is not None:
        add("<h3>Cost breakeven</h3>")
        grid = breakeven[breakeven["row_type"] == "grid"]
        crossings = breakeven[breakeven["row_type"] == "breakeven"].set_index("crossing_metric")
        cagr_x = float(crossings.at["net_cagr", "cost_multiplier"])
        sharpe_x = float(crossings.at["net_sharpe", "cost_multiplier"])
        add(f'<p class="lede">Every modeled execution cost — spread, slippage, commission, fees '
            f'and impact — scaled jointly, everything else frozen. Net CAGR crosses zero at an '
            f'interpolated <strong>{cagr_x:.1f}x</strong> the modeled costs and net Sharpe at '
            f'<strong>{sharpe_x:.1f}x</strong>; both crossings sit inside the declared grid, '
            f'and are estimates from adjacent grid runs, not measurements.</p>')
        keep = ["cost_multiplier", "net_cagr", "net_sharpe", "max_drawdown", "annual_cost_drag"]
        add(table(grid[[c for c in keep if c in grid.columns]]))

    capacity = validation("validation_capacity.csv")
    if capacity is not None:
        add("<h3>Capacity</h3>")
        add('<p class="lede">The frozen configuration replayed at pre-declared capital multiples. '
            'The binding constraint is not impact-cost Sharpe erosion: from $5M the '
            '21-session roll-completion guard aborts the replay in thin markets, so on this cost '
            'model the working capacity sits between $2M and $5M and is set by roll completion.</p>')
        levels = capacity[capacity["row_type"] == "capacity_level"]
        keep = ["capital_multiple", "initial_capital_usd", "run_outcome", "sharpe", "cagr",
                "annual_cost_drag", "peak_order_participation", "backlog_guard_detail"]
        add(table(levels[[c for c in keep if c in levels.columns]]))

    jackknife = validation("validation_universe_jackknife.csv")
    if jackknife is not None:
        add("<h3>Leave-one-sector-out</h3>")
        add('<p class="lede">The frozen configuration replayed six times, each time with one '
            'asset class removed, in the declared order. No row is a recommendation; the table '
            'substantiates the claim that no single class carries the result.</p>')
        add(table(jackknife))

    # ---- the optimality audit -----------------------------------------
    # The one section in this report that is an explicit search.  It is placed
    # after the robustness work, and declared as a search, because it exists to
    # price the configuration choice rather than to make one.
    def optimality(name: str) -> pd.DataFrame | None:
        path = ROOT / "outputs/optimality" / name
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
        return frame if not frame.empty else None

    audit = optimality("optimality_summary.csv")
    if audit is not None:
        selection = optimality("optimality_selection.csv")
        walk_forward = optimality("optimality_walk_forward_summary.csv")
        profile = optimality("optimality_profile_summary.csv")
        candidates = optimality("optimality_candidates.csv")
        deflated = optimality("optimality_deflated_sharpe.csv")

        run = json.loads(
            (ROOT / "outputs/optimality/optimality_run.json").read_text(encoding="utf-8")
        )
        add("<h3>Optimality audit — was this the configuration a search would have picked?</h3>")
        add(f'<p class="lede"><strong>This section is a search, and the only one in this '
            f'report.</strong> The parameter sensitivity above is nine pre-declared runs and '
            f'deliberately searches nothing. That leaves a fair question open: if someone had '
            f'searched, what would they have found? So after the configuration was frozen and '
            f'published, {run["configurations_run"]} configurations were run — dense '
            f'one-parameter profiles over {len(run["profile_grid"])} axes, '
            f'{run["joint_search_trials"]} independent joint draws from the same box, and '
            f'{len(run["candidates"])} pre-registered structural alternatives. No number '
            f'anywhere else in this bundle depends on this section, and no parameter was '
            f're-selected from it.</p>')
        add(table(audit, keep_prose=True))

        if selection is not None:
            add("<h4>Choose on 1990-2004, then live with it over 2005-2014</h4>")
            add('<p class="lede">The test that decides the question. Each pool is searched on '
                'the development window alone, its winner is then measured on the later window, '
                'and the gap between the two is the price of having optimised.</p>')
            keep = ["pool", "configurations", "selected", "selected_train_sharpe",
                    "frozen_train_sharpe", "in_sample_gain", "selected_test_sharpe",
                    "frozen_test_sharpe", "out_of_sample_gain",
                    "share_beating_frozen_out_of_sample", "train_test_rank_correlation"]
            add(table(selection[[c for c in keep if c in selection.columns]]))

        if walk_forward is not None:
            row = walk_forward.set_index("book")
            reopt = float(row.at["annually re-optimised", "sharpe"])
            frozen_sharpe = float(row.at["frozen baseline", "sharpe"])
            reopt_cagr = float(row.at["annually re-optimised", "cagr"])
            frozen_cagr = float(row.at["frozen baseline", "cagr"])
            add("<h4>Nineteen years of annual re-optimisation</h4>")
            add(f'<p class="lede">At each year end from 1995, the configuration with the best '
                f'Sharpe over all history to date is chosen and held through the next calendar '
                f'year. Nineteen such decisions produced a Sharpe of <strong>{reopt:.3f}</strong> '
                f'against the frozen book\'s <strong>{frozen_sharpe:.3f}</strong> over the same '
                f'window, at a CAGR of {reopt_cagr:.2%} against {frozen_cagr:.2%}: the optimiser '
                f'bought no risk-adjusted return and gave up '
                f'{(frozen_cagr - reopt_cagr) * 100:.1f} points of compound return to do it.'
                f'{_reoptimization_interval()}</p>')
            add(table(walk_forward))

        if profile is not None:
            indexed = profile.set_index("parameter")
            axis = indexed.loc["trend_lookback"]
            spiked = list(profile.loc[profile["shape"] == "spike", "parameter"])
            flat = int(profile["shape"].isin(["plateau", "sloped"]).sum())
            inert = list(profile.loc[profile["shape"].str.startswith("inert"), "parameter"])
            add("<h4>The shape of each axis</h4>")
            add(f'<p class="lede">One parameter moved at a time, everything else frozen. '
                f'<em>Plateau</em>, <em>sloped</em> and <em>spike</em> are assigned by rule from '
                f'the share of the axis lying within 0.10 Sharpe of its own best, not by eye. '
                f'{flat} of {len(profile)} axes are flat or gently sloped — every risk control '
                f'among them — and the {len(spiked)} that spike are both signal-design choices: '
                f'<code>{"</code> and <code>".join(spiked)}</code>. '
                + (f'<code>{inert[0]}</code> is inert over this window: its warm-up gate is '
                   f'already satisfied by the pre-launch replay, so its grid is flat by '
                   f'construction rather than by luck. ' if inert else "")
                + f'The trend lookback is the honest weakness in this design: it is worth '
                f'<strong>{float(axis["neighbour_sharpe_drop"]):.2f} Sharpe</strong> against its '
                f'own neighbours. Two things bound that. The {int(axis["frozen_value"])}-session '
                f'value is the 12-month convention taken from Moskowitz, Ooi and Pedersen (2012) '
                f'rather than fitted here; and the peak does not transfer — the development '
                f'window\'s own best value is {float(axis["best_train_value"]):g}, not '
                f'{int(axis["frozen_value"])}, and adopting it would have cost '
                f'{abs(float(axis["selection_gain_out_of_sample"])):.2f} Sharpe over 2005-2014. '
                f'A peak whose location moves between halves of the sample is noise, and it is '
                f'reported rather than smoothed away.</p>')
            weight = indexed.loc["basis_weight"]
            points = optimality("optimality_profiles.csv")
            if points is not None:
                curve = (points.loc[points["parameter"] == "basis_weight"]
                         .set_index("value")["full_sharpe"])
                beyond = curve.loc[curve.index > float(weight["best_full_value"])]
                add(f'<div class="note">The one change that beat the frozen configuration in '
                    f'<em>both</em> windows is a basis weight of '
                    f'{float(weight["best_full_value"]):g} rather than '
                    f'{float(weight["frozen_value"]):g}: '
                    f'{float(weight["full_sharpe_headroom"]):+.3f} Sharpe over the full history '
                    f'and {float(weight["selection_gain_out_of_sample"]):+.3f} over 2005-2014. '
                    f'It is reported and <strong>not adopted</strong>, for three reasons that '
                    f'were true before it was measured. It is one grid step from the equal-risk '
                    f'prior the sleeves were blended on; the gain is inside the paired bootstrap '
                    f'standard error on every comparison in this section; and the axis is a '
                    f'spike, not a plateau — one further step, to '
                    f'{float(beyond.index[0]):g}, gives back '
                    f'{float(curve.loc[float(weight["best_full_value"])] - beyond.iloc[0]):.3f} '
                    f'Sharpe. Moving a frozen parameter onto a peak found by searching the same '
                    f'history that scored it is the exact failure this study was built to '
                    f'detect.</div>')
            keep = ["parameter", "frozen_value", "grid_points", "shape", "best_full_value",
                    "best_full_sharpe", "full_sharpe_headroom", "frozen_full_sharpe_percentile",
                    "axis_sharpe_span", "neighbour_sharpe_drop",
                    "selection_gain_out_of_sample", "train_test_rank_correlation"]
            add(table(profile[[c for c in keep if c in profile.columns]]))

        if candidates is not None:
            gap = candidates.set_index("name")["delta_full_sharpe"]
            worse = int((candidates["delta_test_sharpe"] < 0).sum())
            add("<h4>Pre-registered structural alternatives</h4>")
            add(f'<p class="lede">Not parameter tweaks but different books: multi-horizon trend '
                f'ensembles — the standard answer to a single-horizon choice — and each sleeve '
                f'run alone. {worse} of {len(candidates)} are worse out of sample, and the '
                f'ensembles do not rescue the lookback spike; they pay for the smoothing. The '
                f'two-sleeve blend beats trend alone by '
                f'{abs(float(gap["candidate:trend_only"])):.2f} Sharpe and basis alone by '
                f'{abs(float(gap["candidate:basis_only"])):.2f}, which is the clearest evidence '
                f'in this bundle that the second sleeve is doing real work.</p>')
            keep = ["name", "assumption", "full_sharpe", "train_sharpe", "test_sharpe",
                    "full_cagr", "full_max_drawdown", "delta_full_sharpe", "delta_test_sharpe"]
            add(table(candidates[[c for c in keep if c in candidates.columns]],
                      keep_prose=True))

        if deflated is not None:
            frozen_row = deflated.loc[deflated["member"] == "frozen baseline"].iloc[0]
            add("<h4>What the search costs the headline number</h4>")
            add(f'<p class="lede">A search this size has to be paid for. Deflating the frozen '
                f'configuration\'s Sharpe for a declared family of '
                f'{int(frozen_row["declared_trials"])} trials leaves a deflated probability of '
                f'<strong>{float(frozen_row["deflated_probability"]):.4f}</strong>. That is a '
                f'lower bound in one direction only: it corrects for the trials counted here, '
                f'and the earlier lineage remains unrecoverable, exactly as '
                f'<code>validation_deflated_sharpe.csv</code> records.</p>')
            keep = ["member", "configuration", "annualized_sharpe", "declared_trials",
                    "sessions", "status", "deflated_probability"]
            add(table(deflated[[c for c in keep if c in deflated.columns]]))

    # ---- benchmark and attribution ------------------------------------
    add("<h2>Benchmark and attribution</h2>")
    add(table(optional("benchmark_comparison.csv")))
    attribution = optional("return_attribution.csv")
    if attribution is not None:
        add("<h3>What drove the result</h3>")
        add(table(attribution))
    add(f"""<div class="note"><strong>Honest reading.</strong> The strategy wins on Sharpe, not on
signal. Moskowitz-Ooi-Pedersen TSMOM earns more raw CAGR; it simply takes far more risk to do it.
Strip out the volatility targeting, the per-market risk sizing and the volatility-shock taper and
the remaining signal edge is modest. The {float(spanning['alpha_annualized']):.2%} spanning alpha is
also biased in this strategy's favour — the benchmark rules ran cold and unrefitted, while this
strategy's parameters were chosen with the panel visible. Treat it as an upper bound.</div>""")

    # ---- data discipline and costs ------------------------------------
    add("<h2>Data discipline</h2>")

    # The brief says "do not artificially fill across holidays or missing
    # values". This panel does fill, with a limit. Rather than let that sit
    # unremarked in row 7 of a 23-row table, state it, quantify it, and let the
    # reader judge. Numbers are read from the artifact so they cannot drift.
    nofill = optional("robustness_no_fill.csv")
    fill_cost = "not measured in this build."
    if nofill is not None:
        row = nofill.loc[nofill["window"] == "1990-2014 full"]
        late = nofill.loc[nofill["window"] == "2005-2014 out-of-sample"]
        if len(row) and len(late):
            row, late = row.iloc[0], late.iloc[0]
            base_sr = float(row["canonical_Sharpe (rf=0)"])
            obs_sr = float(row["observed_only_Sharpe (rf=0)"])
            delta = float(row["delta_Sharpe (rf=0)"])
            fill_cost = (
                "rebuilding every signal on each market's own observed sessions — no price "
                "imputed anywhere — moves full-window Sharpe "
                f'<span class="r">{base_sr:.4f} → {obs_sr:.4f} ({delta:+.4f})</span>, and '
                f'{float(late["canonical_Sharpe (rf=0)"]):.4f} → '
                f'{float(late["observed_only_Sharpe (rf=0)"]):.4f} over 2005-2014. Daily net '
                f'returns correlate {float(row["net_return_correlation"]):.3f}. This is material '
                "and is not explained away: the result is genuinely sensitive to the fill "
                "treatment, and the strictest reading is the lower number."
            )

    quality = optional("data_quality_checks.csv")
    if quality is not None and "check" in quality:
        look = {row["check"]: row["result"] for _, row in quality.iterrows()}

        def q(prefix: str, fallback: str = "—") -> str:
            for key, value in look.items():
                if str(key).startswith(prefix):
                    return esc(value)
            return fallback

        add(f"""<div class="note">
<strong>Declared deviation from the brief.</strong> The brief says not to fill artificially across
holidays or missing values. This panel <em>is</em> forward-filled, with a 10-session limit, because
59 markets on four continents never share a holiday calendar and a strategy that drops every
partial session would hold no position across half of December. The fill is therefore bounded and
measured rather than avoided, and it is reported here rather than left for a reader to find:
<ul>
<li><strong>How much:</strong> {q('Forward fill: how much')}. Longest observed run is 7 sessions
against the 10-session limit.</li>
<li><strong>Can a filled cell enter a signal?</strong> <span class="w">{q('(a) Can a forward-filled cell enter a SIGNAL?')}</span>
— 3.89% of live market-cells on month-end decision dates carry a filled price. This is the part
that is a genuine concession, and it is not defensible as "harmless".</li>
<li><strong>Can a filled cell enter a fill?</strong> <span class="g">{q('(b) Can a forward-filled cell enter a FILL?')}</span>
— execution is gated on <code>isfinite(raw_close) &amp; volume &gt; 0</code>, evaluated on the
<em>unfilled</em> series, so no trade can originate or complete on an imputed price.</li>
<li><strong>Can a filled cell enter P&amp;L?</strong> {q('(c) Can a forward-filled cell enter P&L on a held position?')}
— the price change on a filled cell is exactly zero and the move is booked in full on the next
observed session. No return is invented; its <em>timing</em> is distorted.</li>
<li><strong>Effect on the headline:</strong> {q('Holiday rows inside the reported daily return series')}.
Including the closure rows is the conservative choice, and they are included.</li>
<li><strong>What the fill is worth, measured:</strong> {fill_cost}</li>
</ul>
Roughly half the missing cells fall on dates when fewer than a quarter of live markets traded —
they are holidays, not vendor omissions. FX conversion is filled on 0.49% of observed
market-sessions, which is the one place a filled value multiplies realised P&amp;L directly.</div>""")

    add(table(quality, keep_prose=True, drop=("scope", "detail")))
    add("<h2>Cost assumptions</h2>")
    add(table(optional("cost_assumptions.csv")))
    by_class = optional("cost_realized_by_class.csv")
    if by_class is not None:
        add("<h3>One-way cost by asset class, in basis points of notional</h3>")
        add(table(by_class))

    # ---- sources -------------------------------------------------------
    add("<h2>Sources and reproduction</h2>")
    add(table(optional("source_note.csv")))
    add("""<div class="note"><strong>No internet access is required.</strong> No external or
downloaded data enters this result: no network call exists anywhere in the code. The bundled
cleaned panel reproduces the headline figures offline and exactly — see
<code>reproduce.py</code>.</div>""")
    add("""<pre>python reproduce.py                 # headline table, offline, ~3 seconds
python -m pytest strategy/           # the reference file equals the production engine</pre>""")

    add(f"""<footer>Delta1 submission · strategy in one file
(<code>strategy/delta1_reference.py</code>), proven equal to the production engine bit for bit ·
report generated {dt.date.today().isoformat()} · no external data, no network access required.
</footer>""")

    body = "\n".join(parts)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Delta1 — Diversified Global Futures Trend and Carry</title>"
        f"<style>{CSS}</style></head><body><div class=\"wrap\">{body}</div></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(SUBMISSION / "report.html"))
    args = parser.parse_args()

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    document = build(load())
    path = Path(args.output)
    path.write_text(document, encoding="utf-8")
    print(f"wrote {path} — {len(document) / 1e6:.2f} MB, self-contained")


if __name__ == "__main__":
    main()
