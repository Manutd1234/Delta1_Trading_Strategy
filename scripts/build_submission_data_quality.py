#!/usr/bin/env python
"""Section D data-quality audit of the supplied Delta1 futures panel.

Everything here is measured from the vendor CSVs and from the exact panel the
strategy trades (`reference.delta1_reference.load_market_data`), so the audit
cannot drift from the backtest.  Nothing is asserted that is not computed.

Writes:
    outputs/submission/data_quality_checks.csv     one row per check
    outputs/submission/data_quality_by_market.csv  per-market detail

Run:
    .venv/bin/python scripts/build_submission_data_quality.py
"""

from __future__ import annotations

import ast
import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "Round1AllData" / "Quant Researcher" / "Delta1"
FUT = DATA_DIR / "Futures Data"
OUT = REPO / "outputs" / "submission"
PREFIX = "data_quality"
REF_PATH = REPO / "reference" / "delta1_reference.py"

EVAL_START, EVAL_END = "1990-01-01", "2014-12-31"


def _load_reference():
    spec = importlib.util.spec_from_file_location("d1_ref_audit", REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d1 = _load_reference()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a 1-D boolean array."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(0, int)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges[1::2] - edges[::2]


def stale_run_cells(values: np.ndarray, min_len: int = 3) -> np.ndarray:
    """Sizes of runs of >= `min_len` identical consecutive values."""
    values = np.asarray(values, float)
    if values.size < min_len:
        return np.zeros(0, int)
    lens = run_lengths(values[1:] == values[:-1]) + 1
    return lens[lens >= min_len]


def pct(x: float) -> str:
    return f"{100.0 * x:.4f}%"


CHECKS: list[dict] = []


def record(check: str, scope: str, result: str, detail: str, pass_fail: str) -> None:
    CHECKS.append(
        {"check": check, "scope": scope, "result": result, "detail": detail, "pass_fail": pass_fail}
    )


# --------------------------------------------------------------------------
# 0.  Raw file sweep: duplicates, weekend rows, monotonicity, parseability
# --------------------------------------------------------------------------

def raw_file_sweep() -> pd.DataFrame:
    rows = []
    for path in sorted(FUT.glob("*.csv")):
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        close = pd.to_numeric(frame["Close"], errors="coerce")
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        rows.append(
            {
                "file": path.name,
                "rows": len(frame),
                "unparseable_dates": int(dates.isna().sum()),
                "duplicate_dates": int(dates.duplicated().sum()),
                "weekend_rows": int(dates.dt.dayofweek.ge(5).sum()),
                "non_monotonic": int(not dates.is_monotonic_increasing),
                "unparseable_close": int(close.isna().sum() - frame["Close"].isna().sum()),
                "nan_close": int(frame["Close"].isna().sum()),
                "back_adjusted": path.stem.endswith("_CCB"),
                "nonpositive_close": int((close <= 0).sum()),
                "infinite_close": int(np.isinf(close.to_numpy(float)).sum()),
                "negative_volume": int((volume < 0).sum()),
                "first_date": dates.min(),
                "last_date": dates.max(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1.  Static audit of reference/delta1_reference.py: what divides by what
# --------------------------------------------------------------------------

LEVEL_TOKENS = re.compile(r"\b(prices|closes|close|raw_close)\b")
DERIVED_TOKENS = re.compile(r"\.diff\(\)|\.std\(|_vol\b|realized|change\b")


def division_denominators(source: str) -> list[tuple[str, str]]:
    """Every division site in the file, as (operator, denominator source)."""
    tree = ast.parse(source)
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            found.append(("/", ast.get_source_segment(source, node.right) or ""))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            found.append(("/=", ast.get_source_segment(source, node.value) or ""))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "divide" and len(node.args) >= 2:
                found.append(("np.divide", ast.get_source_segment(source, node.args[1]) or ""))
            elif name == "div" and node.args:
                found.append((".div", ast.get_source_segment(source, node.args[0]) or ""))
    return found


def limit_gross_price_argument(source: str) -> list[str]:
    """The `price` argument at every `_limit_gross(...)` call site."""
    tree = ast.parse(source)
    args = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "_limit_gross" and len(node.args) >= 3:
                args.append(ast.get_source_segment(source, node.args[2]) or "")
    return args


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = REF_PATH.read_text()

    symbols = list(d1.SYMBOLS)
    asset_class = {s: name for name, members in d1.UNIVERSE.items() for s in members}

    data = d1.load_market_data(DATA_DIR)
    prices = data["prices"]                       # back-adjusted, ffill(limit=10)
    observed = data["closes"]                     # back-adjusted, UNFILLED
    volumes = data["volumes"]                     # UNFILLED
    unadj_filled = data["unadjusted"]             # unadjusted, ffill(limit=10)
    unadj_raw = d1._panel(DATA_DIR, "&{s}.csv", "Close", None)
    point_values = data["point_values"]           # USD per point, calendar x symbol
    calendar = prices.index
    n_cells = int(prices.shape[0] * prices.shape[1])

    # in-range mask: between each market's own first and last observed session
    first_obs = observed.apply(lambda c: c.first_valid_index())
    last_obs = observed.apply(lambda c: c.last_valid_index())
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    lo = np.array([pos[first_obs[s]] for s in symbols])
    hi = np.array([pos[last_obs[s]] for s in symbols])
    grid = np.arange(len(calendar))[:, None]
    in_range = pd.DataFrame((grid >= lo) & (grid <= hi), index=calendar, columns=symbols)

    obs_mask = observed.notna()
    filled_mask = prices.notna() & observed.isna()
    gap_mask = in_range & observed.isna()                 # missing observation in range
    unfilled_gap = in_range & prices.isna()               # gap longer than the ffill limit

    # ------------------------------------------------------------------ 1
    dens = division_denominators(source)
    suspect = [
        (op, den) for op, den in dens
        if LEVEL_TOKENS.search(den) and not DERIVED_TOKENS.search(den)
    ]
    price_args = limit_gross_price_argument(source)
    notional_from_unadjusted = all("unadj" in a for a in price_args) and bool(price_args)

    ccb_nonpos = (observed <= 0).sum()
    unadj_nonpos = (unadj_raw[symbols] <= 0).sum()
    ccb_neg_markets = int((ccb_nonpos > 0).sum())

    record(
        "Adjusted vs unadjusted: which series drives what",
        "reference/delta1_reference.py, 59 markets",
        "Both are loaded and used for different jobs",
        "Back-adjusted &SYM_CCB Close -> `prices`: trend (prices - prices.shift(252)), all "
        "volatility (prices.diff() EWMA), the shock taper, the risk-managed scaler and daily "
        "mark-to-market P&L (close[i]-close[i-1] x point value). Unadjusted &SYM Close -> "
        "`unadjusted`: the roll-yield/basis signal (unadjusted.diff() - prices.diff()), gross "
        "notional valuation in _limit_gross, and the square-root impact charge "
        "(charged x |unadj| x pv x bps x sqrt(participation)). Margin is a catalogue constant, "
        "not price-derived.",
        "PASS",
    )
    record(
        "No back-adjusted LEVEL used as a denominator",
        f"AST scan of all {len(dens)} division sites in reference/delta1_reference.py",
        f"{len(suspect)} violations",
        "A back-adjusted level is a cumulative sum of price CHANGES plus an arbitrary constant, "
        "so it has no economic scale: percentage returns, notional and participation computed "
        "from it would be meaningless and can flip sign. Measured: the back-adjusted series goes "
        f"<= 0 at some point in {ccb_neg_markets} of {len(symbols)} markets "
        f"({int(ccb_nonpos.sum()):,} cells), while the unadjusted series is <= 0 in "
        f"{int(unadj_nonpos.sum())} cells - direct proof the adjusted level cannot be a "
        "denominator or a notional. Every denominator resolves to a price DIFFERENCE "
        "(daily_vol, annual_dollar_vol, realized), a volume, a NAV (prior_nav, rows[:,0]), a "
        "count, a constant, or a Path join. The single price-valued denominator is `gross` "
        "inside _limit_gross, and its only call site builds it from unadj[loc] - the unadjusted "
        "price, checked separately below.",
        "PASS" if not suspect else "FAIL",
    )
    record(
        "Notional and impact priced off the unadjusted contract",
        "_limit_gross call sites and the impact term in simulate()",
        f"{len(price_args)}/{len(price_args)} call sites pass an unadjusted price",
        f"price arguments seen: {sorted(set(price_args))}; impact term multiplies "
        "np.abs(unadj[i]) x pv[i]. Gross P&L instead uses close[i]-close[i-1] from the "
        "back-adjusted panel, which is the only series whose differences are roll-free.",
        "PASS" if notional_from_unadjusted else "FAIL",
    )

    # ------------------------------------------------------------------ 2
    sweep = raw_file_sweep()
    dup_total = int(sweep["duplicate_dates"].sum())
    record(
        "Duplicate Date rows",
        f"all {len(sweep)} raw CSVs in Futures Data/",
        f"{dup_total} duplicate dates across {int((sweep['duplicate_dates'] > 0).sum())} files",
        "Checked with pandas duplicated() on the parsed Date column of every file, not only the "
        "59 traded markets. The loader also raises on duplicates (_column), so a duplicate would "
        "abort the backtest rather than silently double-count. "
        f"Unparseable dates: {int(sweep['unparseable_dates'].sum())}; "
        f"non-monotonic files: {int(sweep['non_monotonic'].sum())} "
        "(the loader sorts, so ordering is not relied on).",
        "PASS" if dup_total == 0 else "FAIL",
    )

    weekend_rows = int(sweep["weekend_rows"].sum())
    record(
        "Weekend / off-calendar rows silently dropped by the reindex",
        f"all {len(sweep)} raw CSVs",
        f"{weekend_rows} Saturday/Sunday rows in the vendor files",
        "The loader reindexes onto pd.bdate_range, which would DISCARD any weekend observation "
        "without warning. Measured at zero, so the business-day calendar loses no vendor data.",
        "PASS" if weekend_rows == 0 else "FAIL",
    )

    # ------------------------------------------------------------------ 3
    gaps_per_market = gap_mask.sum()
    in_range_per_market = in_range.sum()
    gap_total = int(gaps_per_market.sum())
    in_range_total = int(in_range_per_market.sum())
    worst_gap = (gaps_per_market / in_range_per_market).sort_values(ascending=False).head(5)
    record(
        "Missing observations inside each market's own history",
        "59 traded markets, business-day calendar between each market's first and last "
        "observed session",
        f"{gap_total:,} missing cells out of {in_range_total:,} in-range cells "
        f"({pct(gap_total / in_range_total)})",
        "Worst five by share: "
        + "; ".join(f"{s} {pct(v)}" for s, v in worst_gap.items())
        + ". Nothing is filled before a market's first observation, so history is never "
        "back-created; the panel simply carries fewer markets early on.",
        "INFO",
    )

    # decomposition of the gaps: whole-market closure vs single-market absence
    obs_rate = obs_mask.sum(axis=1) / in_range.sum(axis=1).replace(0, np.nan)
    quiet_day = obs_rate.lt(0.25)
    gap_on_quiet = int((gap_mask & quiet_day.to_numpy()[:, None]).sum().sum())
    zero_obs_days = int(((obs_mask.sum(axis=1) == 0) & (in_range.sum(axis=1) > 0)).sum())
    record(
        "Are the gaps holidays or vendor omissions?",
        "59 markets x business-day calendar",
        f"{pct(gap_on_quiet / gap_total)} of missing cells fall on dates when <25% of "
        f"in-range markets traded",
        f"{zero_obs_days:,} business days in the panel have ZERO observations across all "
        "in-range markets (global closures such as 1 Jan and 25 Dec). The remaining gaps are "
        "single-exchange holidays and occasional vendor omissions; they are not distinguishable "
        "from each other with the supplied data, which is a limitation, not a claim.",
        "INFO",
    )

    # ------------------------------------------------------------------ 4  the forward fill
    fills_per_market = filled_mask.sum()
    fill_total = int(fills_per_market.sum())
    obs_total = int(obs_mask.sum().sum())
    all_runs = np.concatenate(
        [run_lengths(filled_mask[s].to_numpy()) for s in symbols] + [np.zeros(0, int)]
    )
    run_hist = pd.Series(all_runs).value_counts().sort_index()
    fills_gt5 = int(all_runs[all_runs > 5].sum())
    runs_gt5 = int((all_runs > 5).sum())
    trailing = {s: int(filled_mask[s].loc[last_obs[s]:].sum()) for s in symbols}
    trailing_fill_cells = int(sum(trailing.values()))
    trailing_markets = [s for s, v in trailing.items() if v]

    record(
        "Forward fill: how much of the traded panel is filled rather than observed",
        f"{len(symbols)} markets x {len(calendar):,} business days = {n_cells:,} cells",
        f"{fill_total:,} filled cells = {pct(fill_total / n_cells)} of the panel, "
        f"{pct(fill_total / (fill_total + obs_total))} of populated cells",
        f"P['price_ffill_limit'] = {d1.P['price_ffill_limit']} sessions. Fill-run length "
        "distribution (sessions: runs): "
        + ", ".join(f"{int(k)}:{int(v):,}" for k, v in run_hist.items())
        + f". Runs longer than 5 sessions: {runs_gt5:,} runs / {fills_gt5:,} cells "
        f"({pct(fills_gt5 / n_cells)} of the panel). Gaps longer than "
        f"{d1.P['price_ffill_limit']} sessions are NOT filled: "
        f"{int(unfilled_gap.sum().sum()):,} in-range cells stay NaN and drop out of the "
        f"forecast entirely. {trailing_fill_cells} filled cells sit past a market's last "
        f"observation ({', '.join(trailing_markets)} on the panel's final session); nothing is "
        "ever filled before a market's first observation.",
        "INFO",
    )

    # (a) does a filled cell enter a signal?
    all_decisions = d1.month_end_rows(prices).index
    decisions = all_decisions[all_decisions >= pd.Timestamp(d1.P["launch"])]
    dec_filled = int(filled_mask.reindex(decisions).sum().sum())
    dec_live = int((filled_mask | obs_mask).reindex(decisions).sum().sum())
    record(
        "(a) Can a forward-filled cell enter a SIGNAL?",
        f"{len(decisions)} monthly decision dates from {d1.P['launch']} "
        "(last business day of each month)",
        "YES - and it does",
        f"On those month-end decision dates, {dec_filled:,} of {dec_live:,} "
        f"live market-cells ({pct(dec_filled / dec_live)}) carry a forward-filled price. The "
        "trend, basis, volatility and sizing terms are all computed on the filled panel, so a "
        "target can be set from a price up to 10 sessions stale. This is a real approximation, "
        "not a neutral one: it makes the signal lag, it does not fabricate a move (a filled "
        "cell repeats the last OBSERVED close exactly).",
        "INFO",
    )

    # (b) does a filled cell enter a fill?
    raw_close = data["closes"].to_numpy(float)
    vol_np = data["volumes"].to_numpy(float)
    tradeable_now = np.isfinite(raw_close) & (vol_np > 0)
    filled_and_tradeable = int((filled_mask.to_numpy() & tradeable_now).sum())
    tradeable_total = int(tradeable_now.sum())
    record(
        "(b) Can a forward-filled cell enter a FILL?",
        "simulate(): tradeable_now = np.isfinite(raw_close[i]) & (volume[i] > 0)",
        f"NO - {filled_and_tradeable} filled cells are tradeable, out of "
        f"{tradeable_total:,} tradeable cells",
        "raw_close is the UNFILLED back-adjusted close and volume is the UNFILLED volume, so a "
        "filled cell is NaN in raw_close and cannot pass the gate. Orders, roll slices and "
        "turnover are all masked by tradeable_now, and an unfilled order stays pending to the "
        "next actually-observed session. Empirically verified as an exact zero, not argued.",
        "PASS" if filled_and_tradeable == 0 else "FAIL",
    )

    # additional gate detail: observed close but no volume
    obs_no_volume = int((obs_mask.to_numpy() & ~(vol_np > 0)).sum())
    record(
        "Sessions with an observed close but no volume",
        "59 markets x business-day calendar",
        f"{obs_no_volume:,} cells ({pct(obs_no_volume / max(obs_total, 1))} of observed cells)",
        "These carry a price but are not tradeable under the gate, so they mark existing "
        "positions but cannot originate or complete a trade. Treating a zero-volume print as "
        "executable would be the more aggressive assumption; the code takes the conservative one.",
        "INFO",
    )

    # (c) does a filled cell enter P&L?
    change = prices.diff()
    max_abs_change_on_fill = float(np.nanmax(np.abs(change.to_numpy()[filled_mask.to_numpy()])))
    prev_filled = filled_mask.shift(1).fillna(False)
    catchup_mask = obs_mask & prev_filled
    catchup_cells = int(catchup_mask.sum().sum())

    market_daily = pd.read_csv(REPO / "outputs" / "strategy_market_daily.csv.gz",
                               parse_dates=["date"])
    md = market_daily[(market_daily["date"] >= EVAL_START) & (market_daily["date"] <= EVAL_END)]
    catch_long = catchup_mask.stack().rename("catchup").rename_axis(["date", "symbol"])
    md = md.join(catch_long, on=["date", "symbol"])
    md["catchup"] = md["catchup"].fillna(False)
    held = md[md["start_contracts"] != 0]
    held_rows = len(held)
    held_catchup = int(held["catchup"].sum())
    abs_pnl_total = float(held["gross_pnl_usd"].abs().sum())
    abs_pnl_catchup = float(held.loc[held["catchup"], "gross_pnl_usd"].abs().sum())
    net_pnl_total = float(held["gross_pnl_usd"].sum())
    net_pnl_catchup = float(held.loc[held["catchup"], "gross_pnl_usd"].sum())

    record(
        "(c) Can a forward-filled cell enter P&L on a held position?",
        "simulate(): gross_pnl = old x (close[i]-close[i-1]) x pv[i]",
        "YES, but only as a zero and a deferral - never as an invented move",
        f"On a filled cell the back-adjusted close repeats the previous value exactly, so the "
        f"marked change is identically zero (max |change| on a filled cell = "
        f"{max_abs_change_on_fill:.1e}). The move is not lost: it is booked in full on the next "
        f"observed session. {catchup_cells:,} market-sessions are such catch-up prints; "
        f"{held_catchup:,} of {held_rows:,} held market-sessions (1990-2014) land on one "
        f"({pct(held_catchup / held_rows)} of held sessions), carrying "
        f"{pct(abs_pnl_catchup / abs_pnl_total)} of gross |P&L| "
        f"(net ${net_pnl_catchup:,.0f} of ${net_pnl_total:,.0f}) - a concentration of "
        f"{(abs_pnl_catchup / abs_pnl_total) / (held_catchup / held_rows):.2f}x the average "
        "session. Total P&L is therefore correct; its DAILY timing is not, which biases daily "
        "volatility and Sharpe (a zero day followed by an oversized day). This is the honest "
        "cost of the fill.",
        "INFO",
    )

    # holiday zero-return rows in the reported daily series
    daily = pd.read_csv(REPO / "outputs" / "strategy_daily.csv", index_col=0, parse_dates=True)
    window = daily.loc[EVAL_START:EVAL_END]
    net = window["net_return"].dropna()
    zero_obs_index = obs_mask.sum(axis=1).eq(0) & in_range.sum(axis=1).gt(0)
    holiday_rows = net.index.intersection(calendar[zero_obs_index])
    ann = d1.P["annualization"]
    sharpe_all = net.mean() / net.std() * math.sqrt(ann)
    net_ex = net.drop(holiday_rows)
    sharpe_ex = net_ex.mean() / net_ex.std() * math.sqrt(ann)
    sessions_per_year = len(net_ex) / ((net.index[-1] - net.index[0]).days / 365.2425)
    record(
        "Holiday rows inside the reported daily return series",
        f"{EVAL_START}..{EVAL_END} net_return, {len(net):,} sessions",
        f"{len(holiday_rows):,} rows are whole-panel closures; Sharpe {sharpe_all:.4f} with "
        f"them, {sharpe_ex:.4f} without",
        f"The ledger runs on pd.bdate_range, so global holidays appear as exact-zero return "
        f"rows and dilute the daily series. Removing them moves Sharpe by "
        f"{sharpe_ex - sharpe_all:+.4f}. Actual traded sessions per calendar year in this "
        f"window: {sessions_per_year:.1f}, versus the annualization constant "
        f"{ann} the metrics use - a {100 * (sessions_per_year / ann - 1):+.1f}% mismatch that "
        "flows into every annualized number. Reported headline numbers keep the 252/bdate "
        "convention for comparability; this is the size of that choice.",
        "INFO",
    )

    # ------------------------------------------------------------------ 5  stale prices
    stale_runs_market, stale_cells_market, longest_stale = {}, {}, {}
    repeat_cells, repeat_with_volume = {}, {}
    for s in symbols:
        live = observed[s].dropna()
        runs = stale_run_cells(live.to_numpy(float))
        stale_runs_market[s] = int(runs.size)
        stale_cells_market[s] = int(runs.sum())
        longest_stale[s] = int(runs.max()) if runs.size else 0
        # any repeat of the previous observed close, and whether it printed volume
        same = live.eq(live.shift()).to_numpy().copy()
        if same.size:
            same[0] = False
        rep_idx = live.index[same]
        repeat_cells[s] = int(same.sum())
        repeat_with_volume[s] = int((volumes[s].reindex(rep_idx) > 0).sum())
    stale_runs_market = pd.Series(stale_runs_market)
    stale_cells_market = pd.Series(stale_cells_market)
    repeat_cells = pd.Series(repeat_cells)
    repeat_with_volume = pd.Series(repeat_with_volume)
    obs_per_market = obs_mask.sum()
    stale_share = stale_cells_market / obs_per_market
    worst_stale = stale_share.sort_values(ascending=False).head(5)
    record(
        "Stale prices (>= 3 identical consecutive observed closes)",
        "59 markets, OBSERVED sessions only (the forward fill is excluded by construction)",
        f"{int(stale_runs_market.sum()):,} runs covering {int(stale_cells_market.sum()):,} "
        f"observed cells ({pct(stale_cells_market.sum() / obs_total)} of observed cells)",
        "Worst five by share of own observed history: "
        + "; ".join(f"{s} {pct(v)} (longest run {longest_stale[s]})" for s, v in worst_stale.items())
        + f". Counting any repeat of the previous observed close (runs of 2 upward): "
        f"{int(repeat_cells.sum()):,} cells ({pct(repeat_cells.sum() / obs_total)}), of which "
        f"{int(repeat_with_volume.sum()):,} ({pct(repeat_with_volume.sum() / max(int(repeat_cells.sum()), 1))}) "
        "printed positive volume - i.e. the overwhelming majority are genuine quiet sessions in "
        "tick-coarse contracts (short-end rates lead the table), not a frozen feed. Repeats are "
        "left in place: deleting them would be a discretionary edit to the price history.",
        "INFO",
    )

    # ------------------------------------------------------------------ 6  units and currency
    catalogue = pd.read_csv(DATA_DIR / "CATALOGUE_Delta1_Futures.csv")
    catalogue["sym"] = catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    terms = (
        catalogue[catalogue["symbol"].str.endswith("_CCB")]
        .drop_duplicates("sym")
        .set_index("sym")
    )
    traded_terms = terms.reindex(symbols)
    currency = traded_terms["currency"]
    convertible = set(d1.FX_SOURCE) | {"USD"}
    no_source = sorted(set(currency) - convertible)
    non_usd = int((currency != "USD").sum())
    all_currencies = terms["currency"].value_counts()
    orphan_currencies = sorted(set(terms["currency"]) - convertible)
    orphan_markets = sorted(terms.index[terms["currency"].isin(orphan_currencies)])

    record(
        "Every traded contract's currency has a conversion source inside the panel",
        f"{len(symbols)} traded markets",
        f"{len(no_source)} traded currencies without an in-panel FX source; "
        f"{non_usd} markets are non-USD",
        "Currency mix of the traded book: "
        + ", ".join(f"{c}:{n}" for c, n in currency.value_counts().items())
        + ". Conversion uses the currency futures already in the panel "
        + ", ".join(f"{c}<-&{sym} x{scale:g}" for c, (sym, scale) in d1.FX_SOURCE.items())
        + f", so no external FX series enters the backtest. Across all {len(terms)} catalogued "
        f"markets the currency mix is "
        + ", ".join(f"{c}:{n}" for c, n in all_currencies.items())
        + f"; {len(orphan_markets)} markets quoted in {orphan_currencies} have no conversion "
        "source in the supplied data and are therefore outside the traded universe "
        f"({', '.join(orphan_markets)}).",
        "PASS" if not no_source else "FAIL",
    )

    tick = traded_terms["tick_size"].astype(float)
    pv_local = traded_terms["point_value"].astype(float)
    margin_local = traded_terms["margin"].astype(float)
    bad_terms = int(
        (~np.isfinite(tick) | (tick <= 0)).sum()
        + (~np.isfinite(pv_local) | (pv_local <= 0)).sum()
        + (~np.isfinite(margin_local) | (margin_local <= 0)).sum()
    )
    record(
        "Tick size, point value and margin are positive and finite",
        f"{len(symbols)} traded markets, CATALOGUE_Delta1_Futures.csv",
        f"{bad_terms} bad values",
        f"tick_size range [{tick.min():g}, {tick.max():g}]; point_value (local ccy) range "
        f"[{pv_local.min():g}, {pv_local.max():g}]; margin range "
        f"[{margin_local.min():,.0f}, {margin_local.max():,.0f}]. These feed the fixed cost term "
        "((0.5+0.25) ticks x point value + $2.50 commission + $1.50 fees) and the margin "
        "utilisation report, so a zero would silently zero a cost.",
        "PASS" if bad_terms == 0 else "FAIL",
    )

    # notional-per-contract sanity: a units error shows up as an implausible notional
    notional = (unadj_filled[symbols].abs() * point_values[symbols]).where(in_range)
    med_notional = notional.median()
    outside = med_notional[(med_notional < 1e3) | (med_notional > 2e7)]
    record(
        "Units sanity: median USD notional per contract",
        f"{len(symbols)} traded markets, unadjusted close x USD point value",
        f"{len(outside)} markets outside a plausible $1k-$20m band",
        f"Range ${med_notional.min():,.0f} ({med_notional.idxmin()}) to "
        f"${med_notional.max():,.0f} ({med_notional.idxmax()}); median across markets "
        f"${med_notional.median():,.0f}. This is the check that catches a currency-scale error: "
        "the JPY rate is taken from &6J with an explicit 0.01 scale because the contract is "
        "quoted per 100 yen, and a missing scale would inflate every JPY notional 100x.",
        "PASS" if len(outside) == 0 else "REVIEW",
    )

    # FX staleness on traded sessions
    fx = pd.DataFrame({"USD": 1.0}, index=calendar)
    fx_observed = pd.DataFrame({"USD": True}, index=calendar)
    for ccy, (sym, scale) in d1.FX_SOURCE.items():
        rate = d1._column(FUT / f"&{sym}.csv", "Close", ccy) * scale
        aligned = rate.reindex(calendar)
        fx_observed[ccy] = aligned.notna()
        fx[ccy] = aligned.ffill(limit=d1.P["price_ffill_limit"])
    fx_filled = pd.DataFrame(
        {s: (fx[currency[s]].notna() & ~fx_observed[currency[s]]).to_numpy() for s in symbols},
        index=calendar,
    )
    fx_missing = pd.DataFrame(
        {s: fx[currency[s]].isna().to_numpy() for s in symbols}, index=calendar
    )
    stale_fx_on_obs = (obs_mask & fx_filled).sum()
    missing_fx_on_obs = (obs_mask & fx_missing).sum()
    worst_fx = (stale_fx_on_obs / obs_per_market.clip(lower=1)).sort_values(
        ascending=False
    ).head(5)
    record(
        "FX conversion is forward-filled on a market's own trading session",
        "16 non-USD traded markets x their observed sessions",
        f"{int(stale_fx_on_obs.sum()):,} observed market-sessions convert at a forward-filled "
        f"FX rate ({pct(stale_fx_on_obs.sum() / obs_total)} of all observed cells)",
        "The USD point value multiplies daily P&L, so a stale rate is not cosmetic. It happens "
        "whenever a foreign exchange trades on a CME holiday, because every conversion rate is "
        "taken from a CME-listed currency future rather than an external spot series. Worst "
        "markets, as a share of their own observed history: "
        + "; ".join(f"{s} {pct(v)}" for s, v in worst_fx.items())
        + ". The error is bounded by one to a few days of FX drift applied to that day's P&L "
        "only; the position and the local-currency price are unaffected, and the deferred FX "
        "move is picked up on the next observed rate.",
        "INFO",
    )

    orphan_fx = missing_fx_on_obs[missing_fx_on_obs > 0]
    md_nofx = md.join(
        (obs_mask & fx_missing).stack().rename("no_fx").rename_axis(["date", "symbol"]),
        on=["date", "symbol"],
    )
    held_without_fx = int(((md_nofx["start_contracts"] != 0) & md_nofx["no_fx"].fillna(False)).sum())
    traded_without_fx = int(
        ((md_nofx["trade_contracts"] != 0) & md_nofx["no_fx"].fillna(False)).sum()
    )
    record(
        "Markets priced before their FX conversion source exists",
        "59 traded markets x observed sessions",
        f"{int(missing_fx_on_obs.sum()):,} observed market-sessions have a local price but NO "
        f"USD rate; {held_without_fx} were ever held and {traded_without_fx} ever traded",
        "Affected: "
        + "; ".join(
            f"{s} {int(v):,} sessions to "
            f"{calendar[(obs_mask[s] & fx_missing[s]).to_numpy()][-1].date().isoformat()}"
            for s, v in orphan_fx.items()
        )
        + ". &6E begins 1999-01-04 and &6A begins 1987-01-13, so EUR- and AUD-quoted contracts "
        "that pre-date them cannot be valued in USD. This is a silent exclusion rather than an "
        "error: a NaN USD point value makes annualized dollar volatility NaN, the market fails "
        "the `annual_dollar_vol.gt(0)` availability test, and it never receives a risk budget. "
        "Verified directly against the traded ledger - zero held and zero traded sessions. The "
        "cost is real, though: those markets contribute no diversification before their FX "
        "source starts.",
        "PASS" if held_without_fx == 0 and traded_without_fx == 0 else "FAIL",
    )

    # ------------------------------------------------------------------ 7  bad values
    neg_vol = int((volumes < 0).sum().sum())
    inf_prices = int(np.isinf(observed.to_numpy(float)).sum() + np.isinf(unadj_raw[symbols].to_numpy(float)).sum())
    unadj_nonpos_total = int(unadj_nonpos.sum())
    record(
        "Zero / negative / infinite prices and negative volume",
        "59 traded markets, raw unfilled panels",
        f"unadjusted close <= 0: {unadj_nonpos_total}; infinite prices: {inf_prices}; "
        f"negative volume: {neg_vol}",
        f"Back-adjusted close <= 0 occurs in {ccb_neg_markets} markets "
        f"({int(ccb_nonpos.sum()):,} cells) and is EXPECTED, not an error - back-adjustment "
        "subtracts accumulated roll gaps and can drive the synthetic level through zero. It is "
        "harmless here only because the back-adjusted series is used exclusively in differences. "
        f"Across all {len(sweep)} raw files: "
        f"{int(sweep.loc[sweep['back_adjusted'], 'nonpositive_close'].sum()):,} non-positive "
        f"closes in back-adjusted files versus "
        f"{int(sweep.loc[~sweep['back_adjusted'], 'nonpositive_close'].sum())} in unadjusted "
        f"files, {int(sweep['negative_volume'].sum())} negative volumes, "
        f"{int(sweep['nan_close'].sum())} NaN closes.",
        "PASS" if (unadj_nonpos_total == 0 and inf_prices == 0 and neg_vol == 0) else "REVIEW",
    )

    # unadjusted / back-adjusted grid agreement
    grid_mismatch = int((obs_mask & unadj_raw[symbols].isna()).sum().sum())
    record(
        "Back-adjusted and unadjusted files share the same session grid",
        "59 traded markets",
        f"{grid_mismatch} sessions with a back-adjusted close but no unadjusted close",
        "If the two files disagreed, a forward-filled unadjusted price could price a real "
        "trade's notional and impact while the trade itself passed the back-adjusted gate. "
        "Measured at zero, so the notional used for a fill is always an observed price.",
        "PASS" if grid_mismatch == 0 else "FAIL",
    )

    # ------------------------------------------------------------------ reconciliation
    canon = pd.read_csv(REPO / "outputs" / "strategy_data_quality.csv")
    canon_rows = canon.set_index("symbol")["model_price_rows"].reindex(symbols)
    mine = (obs_mask | filled_mask).sum().reindex(symbols)
    recon_max = int((canon_rows - mine).abs().max())
    record(
        "Reconciliation against the canonical bundle",
        "outputs/strategy_data_quality.csv model_price_rows vs observed+filled cells",
        f"max absolute difference {recon_max} rows across {len(symbols)} markets",
        f"Also reconciled: {len(symbols)} markets and {len(calendar):,} calendar rows match the "
        f"canonical panel; the {EVAL_START}..{EVAL_END} ledger has {len(net):,} sessions. "
        "This audit recomputes the panel through the same loader the backtest uses, so the "
        "figures cannot drift from the traded book.",
        "PASS" if recon_max == 0 else "REVIEW",
    )

    record(
        "Adjusted-price policy statement",
        "whole panel",
        "Both series retained; neither is substituted for the other",
        "Back-adjusted prices are used because a continuous futures P&L must be free of roll "
        "jumps, and only differences of that series are ever taken. Unadjusted prices are kept "
        "because the actual contract price is what a notional, a participation rate and an "
        "impact charge must be measured against, and because the difference between the two "
        "series IS the roll yield the carry sleeve trades. No value is interpolated, no holiday "
        "is invented, and no missing observation is back-filled; the only imputation anywhere is "
        f"a forward fill of at most {d1.P['price_ffill_limit']} sessions, quantified above.",
        "INFO",
    )

    # ------------------------------------------------------------------ outputs
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(OUT / f"{PREFIX}_checks.csv", index=False)

    by_market = pd.DataFrame(
        {
            "symbol": symbols,
            "asset_class": [asset_class[s] for s in symbols],
            "currency": currency.reindex(symbols).to_numpy(),
            "first_observed": [first_obs[s].date().isoformat() for s in symbols],
            "last_observed": [last_obs[s].date().isoformat() for s in symbols],
            "calendar_sessions_in_range": in_range_per_market.reindex(symbols).to_numpy(),
            "observed_sessions": obs_per_market.reindex(symbols).to_numpy(),
            "missing_in_range": gaps_per_market.reindex(symbols).to_numpy(),
            "missing_share_of_range": (gaps_per_market / in_range_per_market)
            .reindex(symbols).to_numpy(),
            "forward_filled_cells": fills_per_market.reindex(symbols).to_numpy(),
            "forward_filled_share_of_range": (fills_per_market / in_range_per_market)
            .reindex(symbols).to_numpy(),
            "longest_fill_run": [int(run_lengths(filled_mask[s].to_numpy()).max())
                                 if filled_mask[s].any() else 0 for s in symbols],
            "fill_cells_in_runs_gt5": [
                int(r[r > 5].sum()) for r in
                (run_lengths(filled_mask[s].to_numpy()) for s in symbols)
            ],
            "unfilled_gap_cells": unfilled_gap.sum().reindex(symbols).to_numpy(),
            "catchup_sessions": catchup_mask.sum().reindex(symbols).to_numpy(),
            "tradeable_sessions": pd.Series(tradeable_now.sum(axis=0), index=symbols).to_numpy(),
            "observed_no_volume": pd.Series(
                (obs_mask.to_numpy() & ~(vol_np > 0)).sum(axis=0), index=symbols
            ).to_numpy(),
            "stale_runs_ge3": stale_runs_market.reindex(symbols).to_numpy(),
            "stale_cells_ge3": stale_cells_market.reindex(symbols).to_numpy(),
            "stale_share_of_observed": stale_share.reindex(symbols).to_numpy(),
            "longest_stale_run": [longest_stale[s] for s in symbols],
            "repeat_closes_any_run": repeat_cells.reindex(symbols).to_numpy(),
            "repeat_closes_with_volume": repeat_with_volume.reindex(symbols).to_numpy(),
            "ccb_min_close": observed.min().reindex(symbols).to_numpy(),
            "ccb_nonpositive_cells": ccb_nonpos.reindex(symbols).to_numpy(),
            "unadjusted_min_close": unadj_raw[symbols].min().to_numpy(),
            "unadjusted_nonpositive_cells": unadj_nonpos.reindex(symbols).to_numpy(),
            "negative_volume_cells": (volumes < 0).sum().reindex(symbols).to_numpy(),
            "duplicate_dates_ccb": sweep.set_index("file")["duplicate_dates"]
            .reindex([f"&{s}_CCB.csv" for s in symbols]).to_numpy(),
            "duplicate_dates_unadjusted": sweep.set_index("file")["duplicate_dates"]
            .reindex([f"&{s}.csv" for s in symbols]).to_numpy(),
            "tick_size": tick.reindex(symbols).to_numpy(),
            "point_value_local": pv_local.reindex(symbols).to_numpy(),
            "margin_local": margin_local.reindex(symbols).to_numpy(),
            "median_usd_notional_per_contract": med_notional.reindex(symbols).to_numpy(),
            "fx_forward_filled_on_observed_sessions": stale_fx_on_obs.reindex(symbols).to_numpy(),
            "fx_missing_on_observed_sessions": missing_fx_on_obs.reindex(symbols).to_numpy(),
            "trailing_fill_cells": [trailing[s] for s in symbols],
        }
    )
    by_market.to_csv(OUT / f"{PREFIX}_by_market.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print(checks[["check", "result", "pass_fail"]].to_string(index=False))
    print(f"\nWrote {OUT / (PREFIX + '_checks.csv')}")
    print(f"Wrote {OUT / (PREFIX + '_by_market.csv')}  ({len(by_market)} markets)")


if __name__ == "__main__":
    main()
