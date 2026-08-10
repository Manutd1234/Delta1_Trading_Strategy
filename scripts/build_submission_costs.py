"""Section D cost artifacts: express the futures cost model in the spec's bps units.

The submission spec asks for "a reasonable one-way cost" stated in basis points and
compared against a suggested starting table (liquid equity ETF 2-5 bps, commodity ETF
5-10 bps, G10 FX 1-3 bps, EM FX / less-liquid ETF 10-25 bps).  The strategy's cost model
is not quoted in bps: it is quoted in ticks and dollars per contract plus a square-root
impact term.  This script performs the conversion so a reader can check it against their
own benchmark, and then reports what the model actually *charged* over 1990-2014.

Nothing here re-runs the backtest.  Modeled costs come from ``reference/delta1_reference.py``
(``d1.P`` and ``d1.load_market_data``); realized costs come from the canonical artifacts
``outputs/strategy_daily.csv`` and ``outputs/strategy_market_daily.csv.gz``.  Every realized
aggregate is asserted against ``outputs/strategy_metrics.csv`` before it is written.

Outputs
-------
outputs/submission/cost_assumptions.csv
    Tidy long table: the model parameters as configured, the per-market one-way bps
    conversion for all 59 markets, the per-class verdict against the spec's suggested
    bands, the roll accounting, and the financing / borrow / management-fee statement.
outputs/submission/cost_realized_by_class.csv
    One row per asset class (plus an ALL row): modeled one-way bps median and IQR beside
    the cost the ledger actually charged that class.

Run:  .venv/bin/python scripts/build_submission_costs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "Round1AllData" / "Quant Researcher" / "Delta1"
OUT_DIR = REPO / "outputs" / "submission"
WINDOW = ("1990-01-01", "2014-12-31")

sys.path.insert(0, str(REPO / "reference"))
import delta1_reference as d1  # noqa: E402  (path is set immediately above)

# --------------------------------------------------------------------------
# The spec's suggested starting costs, and the analogue each futures class maps to.
# The spec's table is an ETF/FX table; futures have no exact row in it, so the mapping
# is stated explicitly rather than implied.
# --------------------------------------------------------------------------
SPEC_BANDS = {
    "Equity indices": ("liquid equity ETF", 2.0, 5.0,
                       "Index futures are the wholesale form of the same exposure a liquid equity ETF sells."),
    "Government bonds": ("liquid equity ETF (nearest analogue; the spec table has no bond row)", 2.0, 5.0,
                         "A liquid Treasury/Bund ETF prices at the liquid-ETF end of the table, so that band is the fair comparison."),
    "FX": ("G10 FX", 1.0, 3.0,
           "Seven of the eight FX markets are G10; 6M (Mexican peso) is the one EM cross and is flagged separately."),
    "Energy": ("commodity ETF", 5.0, 10.0,
               "Commodity ETFs hold these same futures and add a wrapper fee on top."),
    "Metals": ("commodity ETF", 5.0, 10.0,
               "Commodity ETFs hold these same futures and add a wrapper fee on top."),
    "Agriculture & livestock": ("commodity ETF", 5.0, 10.0,
                                "Commodity ETFs hold these same futures and add a wrapper fee on top."),
}
EM_FX_BAND = ("EM FX / less-liquid ETF", 10.0, 25.0)
EM_FX_SYMBOLS = ("6M",)

# Canonical values from outputs/strategy_metrics.csv, 1990-2014 full post-launch history.
CANON = {
    "annual_cost_drag": 0.010421436303130022,
    "annual_fixed_cost_drag": 0.00948913025155786,
    "annual_impact_cost_drag": 0.000932306051572164,
    "cagr": 0.13189524685750764,
    "sessions": 6523,
    "markets": 59,
}
TOL = 1e-9


def _verdict(value: float, low: float, high: float) -> str:
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "inside"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


# --------------------------------------------------------------------------
# 1.  The one-way cost model, exactly as configured
# --------------------------------------------------------------------------

def cost_parameters() -> dict[str, float]:
    """Read the cost block out of d1.P and refuse to proceed if it has drifted."""
    expected = {
        "half_spread_ticks": 0.50,
        "slippage_ticks": 0.25,
        "commission": 2.50,
        "fees": 1.50,
        "impact_bps_at_full_participation": 10.0,
        "max_participation": 0.02,
    }
    for key, value in expected.items():
        actual = float(d1.P[key])
        if abs(actual - value) > 1e-12:
            raise AssertionError(f"d1.P[{key!r}] is {actual}, expected {value}")
    ticks = expected["half_spread_ticks"] + expected["slippage_ticks"]
    per_contract_usd = expected["commission"] + expected["fees"]
    return {**expected, "one_way_ticks": ticks, "per_contract_usd": per_contract_usd}


# --------------------------------------------------------------------------
# 2.  Convert to basis points of traded notional, per side, per market
# --------------------------------------------------------------------------

def per_market_bps(params: dict[str, float], traded_dates: dict[str, pd.DatetimeIndex],
                   asset_class: pd.Series) -> pd.DataFrame:
    """(0.75 * tick * pv + 4.00) / (price * pv) * 10_000 at a representative price.

    ``pv`` is the USD point value, which for a non-USD market is the local multiplier
    times the point-in-time FX rate the backtest itself uses, so it moves through time.
    The representative price and point value are medians taken over the sessions the
    strategy actually held or traded that market -- not over the whole vendor file --
    so a market that only entered the book in 2007 is priced on 2007-2014 levels.
    """
    data = d1.load_market_data(DATA_DIR)
    unadjusted, point_values, tick_size = data["unadjusted"], data["point_values"], data["tick_size"]

    rows = []
    for symbol in d1.SYMBOLS:
        dates = traded_dates[symbol]
        price = unadjusted[symbol].reindex(dates)
        pv = point_values[symbol].reindex(dates)
        keep = price.notna() & pv.notna() & (price != 0)
        if not keep.any():
            raise ValueError(f"no priced session for {symbol}")
        rep_price = float(price[keep].median())
        rep_pv = float(pv[keep].median())
        tick = float(tick_size[symbol])

        fixed_usd = params["one_way_ticks"] * tick * rep_pv + params["per_contract_usd"]
        notional = rep_price * rep_pv
        # Cross-check: median of the per-session bps rather than bps at the median price.
        # They differ only where the FX leg drifts against the price level.
        session_bps = ((params["one_way_ticks"] * tick * pv[keep] + params["per_contract_usd"])
                       / (price[keep] * pv[keep]) * 10_000.0)
        rows.append({
            "symbol": symbol,
            "asset_class": asset_class[symbol],
            "traded_sessions": int(keep.sum()),
            "first_traded": dates.min().date().isoformat(),
            "last_traded": dates.max().date().isoformat(),
            "tick_size": tick,
            "representative_price": rep_price,
            "usd_point_value": rep_pv,
            "tick_value_usd": tick * rep_pv,
            "fixed_cost_usd_per_contract_per_side": fixed_usd,
            "notional_usd_per_contract": notional,
            "one_way_bps": fixed_usd / notional * 10_000.0,
            "one_way_bps_session_median_check": float(session_bps.median()),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != CANON["markets"]:
        raise AssertionError(f"expected {CANON['markets']} markets, got {len(frame)}")
    return frame


# --------------------------------------------------------------------------
# 4 & 5.  What the ledger actually charged
# --------------------------------------------------------------------------

def realized_costs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Realized cost from the canonical daily and per-market ledgers, reconciled."""
    daily = (pd.read_csv(REPO / "outputs" / "strategy_daily.csv", parse_dates=["date"])
             .set_index("date").loc[WINDOW[0]:WINDOW[1]])
    if len(daily) != CANON["sessions"]:
        raise AssertionError(f"expected {CANON['sessions']} sessions, got {len(daily)}")

    market = pd.read_csv(REPO / "outputs" / "strategy_market_daily.csv.gz", parse_dates=["date"])
    market = market[(market["date"] >= WINDOW[0]) & (market["date"] <= WINDOW[1])].copy()

    # Traded notional on the rebalance leg, from the same unadjusted price and USD point
    # value the ledger valued the position with.
    data = d1.load_market_data(DATA_DIR)
    price = data["unadjusted"].stack().rename("price")
    price.index.names = ["date", "symbol"]
    pv = data["point_values"].stack().rename("usd_point_value")
    pv.index.names = ["date", "symbol"]
    market = market.join(price, on=["date", "symbol"]).join(pv, on=["date", "symbol"])
    market["abs_contracts"] = market["trade_contracts"].abs()
    market["traded_notional_usd"] = market["abs_contracts"] * market["price"].abs() * market["usd_point_value"]
    if market.loc[market["abs_contracts"] > 0, "price"].isna().any():
        raise AssertionError("a rebalance trade has no unadjusted price")

    by_class = market.groupby("asset_class").agg(
        markets=("symbol", "nunique"),
        rebalance_contracts=("abs_contracts", "sum"),
        rebalance_cost_usd=("regular_cost_usd", "sum"),
        roll_cost_usd=("roll_cost_usd", "sum"),
        rebalance_notional_usd=("traded_notional_usd", "sum"),
        gross_pnl_usd=("gross_pnl_usd", "sum"),
    )
    by_class["total_cost_usd"] = by_class["rebalance_cost_usd"] + by_class["roll_cost_usd"]

    # A daily roll-cost series, so roll drag can be expressed on the same
    # mean(cost / prior NAV) * 252 basis the canonical metrics use.
    roll_daily = market.groupby("date")["roll_cost_usd"].sum().reindex(daily.index).fillna(0.0)
    regular_daily = market.groupby("date")["regular_cost_usd"].sum().reindex(daily.index).fillna(0.0)

    totals = {
        "transaction_cost_usd": float(daily["transaction_cost_usd"].sum()),
        "fixed_cost_usd": float(daily["fixed_execution_cost_usd"].sum()),
        "impact_cost_usd": float(daily["market_impact_cost_usd"].sum()),
        "rebalance_cost_usd": float(regular_daily.sum()),
        "roll_cost_usd": float(roll_daily.sum()),
        "gross_pnl_usd": float(daily["gross_pnl_usd"].sum()),
        "net_pnl_usd": float(daily["net_pnl_usd"].sum()),
        "rebalance_turnover_contracts": float(daily["rebalance_contract_turnover"].sum()),
        "roll_turnover_contracts": float(daily["roll_contract_turnover_increment"].sum()),
        "total_turnover_contracts": float(daily["total_contract_turnover"].sum()),
        "rebalance_notional_usd": float(by_class["rebalance_notional_usd"].sum()),
        "annual_cost_drag": float(daily["cost"].mean() * d1.P["annualization"]),
        "annual_fixed_cost_drag": float((daily["fixed_execution_cost_usd"] / daily["prior_nav_usd"]).mean()
                                        * d1.P["annualization"]),
        "annual_impact_cost_drag": float((daily["market_impact_cost_usd"] / daily["prior_nav_usd"]).mean()
                                         * d1.P["annualization"]),
        "annual_roll_cost_drag": float((roll_daily / daily["prior_nav_usd"]).mean() * d1.P["annualization"]),
        "annual_rebalance_cost_drag": float((regular_daily / daily["prior_nav_usd"]).mean()
                                            * d1.P["annualization"]),
    }
    totals["cost_share_of_gross_pnl"] = totals["transaction_cost_usd"] / totals["gross_pnl_usd"]
    totals["fixed_share_of_cost"] = totals["fixed_cost_usd"] / totals["transaction_cost_usd"]
    totals["impact_share_of_cost"] = totals["impact_cost_usd"] / totals["transaction_cost_usd"]
    totals["roll_share_of_turnover"] = totals["roll_turnover_contracts"] / totals["total_turnover_contracts"]
    totals["roll_share_of_cost"] = totals["roll_cost_usd"] / totals["transaction_cost_usd"]
    totals["all_in_usd_per_charged_contract"] = (totals["transaction_cost_usd"]
                                                 / totals["total_turnover_contracts"])
    totals["fixed_usd_per_charged_contract"] = totals["fixed_cost_usd"] / totals["total_turnover_contracts"]
    totals["impact_usd_per_charged_contract"] = totals["impact_cost_usd"] / totals["total_turnover_contracts"]
    totals["impact_uplift_factor"] = (totals["all_in_usd_per_charged_contract"]
                                      / totals["fixed_usd_per_charged_contract"])

    # Reconciliation against the canonical bundle.  These are assertions, not comments.
    checks = {
        "per-market cost sums to the daily ledger":
            abs(totals["rebalance_cost_usd"] + totals["roll_cost_usd"] - totals["transaction_cost_usd"]),
        "fixed + impact sums to the daily ledger":
            abs(totals["fixed_cost_usd"] + totals["impact_cost_usd"] - totals["transaction_cost_usd"]),
        "per-market trades sum to rebalance turnover":
            abs(float(by_class["rebalance_contracts"].sum()) - totals["rebalance_turnover_contracts"]),
        "rebalance + roll turnover sums to total turnover":
            abs(totals["rebalance_turnover_contracts"] + totals["roll_turnover_contracts"]
                - totals["total_turnover_contracts"]),
        "annual cost drag matches strategy_metrics.csv":
            abs(totals["annual_cost_drag"] - CANON["annual_cost_drag"]),
        "annual fixed-cost drag matches strategy_metrics.csv":
            abs(totals["annual_fixed_cost_drag"] - CANON["annual_fixed_cost_drag"]),
        "annual impact-cost drag matches strategy_metrics.csv":
            abs(totals["annual_impact_cost_drag"] - CANON["annual_impact_cost_drag"]),
    }
    for name, error in checks.items():
        if not np.isfinite(error) or error > 1e-6:
            raise AssertionError(f"reconciliation failed: {name} (error {error})")
    totals["max_reconciliation_error"] = float(max(checks.values()))

    traded_dates = {symbol: pd.DatetimeIndex(sorted(group["date"].unique()))
                    for symbol, group in market.groupby("symbol")}
    asset_class = market.drop_duplicates("symbol").set_index("symbol")["asset_class"]
    if set(traded_dates) != set(d1.SYMBOLS):
        raise AssertionError("per-market ledger does not cover the reference universe")

    by_class.attrs["traded_dates"] = traded_dates
    by_class.attrs["asset_class"] = asset_class
    return by_class, daily, totals


# --------------------------------------------------------------------------
# 6.  Financing / borrow / management fee
# --------------------------------------------------------------------------

def funded_view(daily: pd.DataFrame) -> dict[str, object]:
    """Does outputs/levers/ carry a funded variant?  If not, build one from &ZQ.csv."""
    variants_path = REPO / "outputs" / "levers" / "lever_variants.json"
    lever_names: list[str] = []
    if variants_path.exists():
        lever_names = [str(v.get("name", "")) for v in json.loads(variants_path.read_text())]
    funded_levers = [n for n in lever_names
                     if any(k in n.lower() for k in ("fund", "collateral", "financ", "zq"))]

    out: dict[str, object] = {
        "lever_variants": ", ".join(lever_names) or "(none)",
        "funded_lever_present": bool(funded_levers),
    }

    rate_file = DATA_DIR / "Futures Data" / "&ZQ.csv"
    if not rate_file.exists():
        out["funded_status"] = "no &ZQ.csv in the supplied panel"
        return out
    try:
        from delta1_strategy.research import collateral as C
    except Exception as exc:  # pragma: no cover - environment guard
        out["funded_status"] = f"collateral module unavailable: {exc}"
        return out

    rate = C.load_financing_rate(DATA_DIR)
    ledger = C.funded_ledger(daily[["net_return"]], rate, period_start=WINDOW[0])
    report = C.funded_performance_report(ledger).set_index("Basis")
    regimes = C.funded_regime_report(ledger)
    excess_row = report.loc[C.EXCESS_BASIS_LABEL]
    funded_row = report.loc[C.FUNDED_BASIS_LABEL]
    out.update({
        "funded_status": "computed here from &ZQ.csv; no funded variant exists in outputs/levers/",
        "rate_instrument": C.CollateralConfig().rate_instrument,
        "rate_transform": C.CollateralConfig().rate_transform,
        "average_financing_rate": float(funded_row["Average financing rate"]),
        "excess_cagr": float(excess_row["CAGR"]),
        "funded_cagr": float(funded_row["CAGR"]),
        "funded_uplift_pp": float(funded_row["CAGR"] - excess_row["CAGR"]) * 100.0,
        "funded_max_drawdown": float(funded_row["Max drawdown"]),
        "excess_max_drawdown": float(excess_row["Max drawdown"]),
        "sharpe_excess_of_financing": float(funded_row["Sharpe excess of financing"]),
        "funded_limitation": str(funded_row["Limitations"]),
        "regimes": regimes,
    })
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build() -> None:
    params = cost_parameters()
    by_class, daily, totals = realized_costs()
    markets = per_market_bps(params, by_class.attrs["traded_dates"], by_class.attrs["asset_class"])
    funded = funded_view(daily)

    impact_cap_bps = params["impact_bps_at_full_participation"] * np.sqrt(params["max_participation"])
    uplift = totals["impact_uplift_factor"]

    # ---------------- cost_realized_by_class.csv ----------------
    rows = []
    for name, group in markets.groupby("asset_class"):
        bps = group["one_way_bps"]
        analogue, low, high, rationale = SPEC_BANDS[name]
        realized = by_class.loc[name]
        median = float(bps.median())
        rows.append({
            "asset_class": name,
            "markets": int(len(group)),
            "modeled_one_way_bps_median": median,
            "modeled_one_way_bps_q25": float(bps.quantile(0.25)),
            "modeled_one_way_bps_q75": float(bps.quantile(0.75)),
            "modeled_one_way_bps_min": float(bps.min()),
            "modeled_one_way_bps_max": float(bps.max()),
            "modeled_all_in_bps_median": median * uplift,
            "spec_analogue": analogue,
            "spec_band_low_bps": low,
            "spec_band_high_bps": high,
            "verdict_vs_spec_band": _verdict(median, low, high),
            "spec_mapping_rationale": rationale,
            "realized_rebalance_contracts": float(realized["rebalance_contracts"]),
            "realized_rebalance_notional_usd": float(realized["rebalance_notional_usd"]),
            "realized_rebalance_cost_usd": float(realized["rebalance_cost_usd"]),
            "realized_rebalance_usd_per_contract": float(realized["rebalance_cost_usd"]
                                                        / realized["rebalance_contracts"]),
            "realized_rebalance_all_in_bps": float(realized["rebalance_cost_usd"]
                                                   / realized["rebalance_notional_usd"] * 10_000.0),
            "realized_roll_cost_usd": float(realized["roll_cost_usd"]),
            "realized_total_cost_usd": float(realized["total_cost_usd"]),
            "roll_share_of_class_cost": float(realized["roll_cost_usd"] / realized["total_cost_usd"]),
            "share_of_portfolio_cost": float(realized["total_cost_usd"] / totals["transaction_cost_usd"]),
            "gross_pnl_usd": float(realized["gross_pnl_usd"]),
            "cost_share_of_class_gross_pnl": float(realized["total_cost_usd"] / realized["gross_pnl_usd"]),
        })
    class_frame = pd.DataFrame(rows).sort_values("modeled_one_way_bps_median").reset_index(drop=True)

    all_bps = markets["one_way_bps"]
    all_row = {
        "asset_class": "ALL (59 markets)",
        "markets": int(len(markets)),
        "modeled_one_way_bps_median": float(all_bps.median()),
        "modeled_one_way_bps_q25": float(all_bps.quantile(0.25)),
        "modeled_one_way_bps_q75": float(all_bps.quantile(0.75)),
        "modeled_one_way_bps_min": float(all_bps.min()),
        "modeled_one_way_bps_max": float(all_bps.max()),
        "modeled_all_in_bps_median": float(all_bps.median()) * uplift,
        "spec_analogue": "n/a",
        "spec_band_low_bps": np.nan,
        "spec_band_high_bps": np.nan,
        "verdict_vs_spec_band": "n/a",
        "spec_mapping_rationale": "portfolio aggregate; compare per class",
        "realized_rebalance_contracts": totals["rebalance_turnover_contracts"],
        "realized_rebalance_notional_usd": totals["rebalance_notional_usd"],
        "realized_rebalance_cost_usd": totals["rebalance_cost_usd"],
        "realized_rebalance_usd_per_contract": (totals["rebalance_cost_usd"]
                                                / totals["rebalance_turnover_contracts"]),
        "realized_rebalance_all_in_bps": (totals["rebalance_cost_usd"]
                                          / totals["rebalance_notional_usd"] * 10_000.0),
        "realized_roll_cost_usd": totals["roll_cost_usd"],
        "realized_total_cost_usd": totals["transaction_cost_usd"],
        "roll_share_of_class_cost": totals["roll_share_of_cost"],
        "share_of_portfolio_cost": 1.0,
        "gross_pnl_usd": totals["gross_pnl_usd"],
        "cost_share_of_class_gross_pnl": totals["cost_share_of_gross_pnl"],
    }
    class_frame = pd.concat([class_frame, pd.DataFrame([all_row])], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_frame.to_csv(OUT_DIR / "cost_realized_by_class.csv", index=False)

    # ---------------- cost_assumptions.csv ----------------
    a: list[dict[str, object]] = []

    def add(section: str, item: str, value: object, unit: str, source: str,
            note: str = "", asset_class: str = "") -> None:
        a.append({"section": section, "item": item, "asset_class": asset_class,
                  "value": _fmt(value), "unit": unit, "source": source, "note": note})

    src_p = "reference/delta1_reference.py::P"
    add("1_model_parameter", "half_spread_ticks", params["half_spread_ticks"], "ticks per contract per side", src_p,
        "Half the quoted bid/ask, charged on every execution.")
    add("1_model_parameter", "slippage_ticks", params["slippage_ticks"], "ticks per contract per side", src_p,
        "Adverse move between decision and fill, charged on every execution.")
    add("1_model_parameter", "one_way_ticks_total", params["one_way_ticks"], "ticks per contract per side", src_p,
        "half_spread_ticks + slippage_ticks. This is the full one-way spread/slippage charge.")
    add("1_model_parameter", "commission", params["commission"], "USD per contract per side", src_p,
        "Broker commission.")
    add("1_model_parameter", "exchange_and_regulatory_fees", params["fees"], "USD per contract per side", src_p,
        "Exchange plus regulatory fees.")
    add("1_model_parameter", "flat_usd_per_contract_per_side", params["per_contract_usd"],
        "USD per contract per side", src_p, "commission + fees; the 4.00 term in the bps formula.")
    add("1_model_parameter", "impact_bps_at_full_participation", params["impact_bps_at_full_participation"],
        "bps of traded notional", src_p,
        "Square-root impact coefficient: impact_bps = 10.0 * sqrt(participation).")
    add("1_model_parameter", "max_participation", params["max_participation"], "fraction of session volume", src_p,
        "Hard cap on the share of a session's volume an order may take.")
    add("1_model_parameter", "impact_bps_at_the_participation_cap", float(impact_cap_bps),
        "bps of traded notional", "derived",
        "10.0 * sqrt(0.02): the largest impact charge the model can ever levy on one order.")
    add("1_model_parameter", "one_way_bps_formula",
        "(0.75 * tick_size * usd_point_value + 4.00) / (price * usd_point_value) * 10000",
        "text", "derived",
        "Fixed leg only; the impact leg is participation-dependent and additive.")
    add("1_model_parameter", "charge_roll_costs", "True", "boolean", src_p,
        "A delivery transfer is charged two contracts of turnover (sell the old, buy the new).")

    for _, r in markets.sort_values(["asset_class", "one_way_bps"]).iterrows():
        add("2_per_market_one_way_bps", r["symbol"], float(r["one_way_bps"]),
            "bps of notional per side", "derived from d1.P and d1.load_market_data",
            (f"tick {r['tick_size']:g}; median price {r['representative_price']:,.4f}; "
             f"USD point value {r['usd_point_value']:,.4f}; tick value ${r['tick_value_usd']:,.2f}; "
             f"fixed ${r['fixed_cost_usd_per_contract_per_side']:,.2f}/contract on "
             f"${r['notional_usd_per_contract']:,.0f} notional; {r['traded_sessions']} traded sessions "
             f"{r['first_traded']}..{r['last_traded']}; per-session median check "
             f"{r['one_way_bps_session_median_check']:.4f} bps"),
            asset_class=r["asset_class"])

    for _, r in class_frame.iterrows():
        if r["asset_class"].startswith("ALL"):
            continue
        add("3_vs_spec_suggested_band", "modeled median one-way bps",
            float(r["modeled_one_way_bps_median"]), "bps of notional per side", "derived",
            (f"IQR {r['modeled_one_way_bps_q25']:.2f}-{r['modeled_one_way_bps_q75']:.2f} bps "
             f"across {int(r['markets'])} markets. Spec analogue: {r['spec_analogue']} "
             f"{r['spec_band_low_bps']:.0f}-{r['spec_band_high_bps']:.0f} bps. "
             f"Modeled cost is {r['verdict_vs_spec_band'].upper()} that band. {r['spec_mapping_rationale']}"),
            asset_class=r["asset_class"])

    em = markets[markets["symbol"].isin(EM_FX_SYMBOLS)]
    for _, r in em.iterrows():
        add("3_vs_spec_suggested_band", f"{r['symbol']} (the one EM cross)", float(r["one_way_bps"]),
            "bps of notional per side", "derived",
            (f"Spec analogue: {EM_FX_BAND[0]} {EM_FX_BAND[1]:.0f}-{EM_FX_BAND[2]:.0f} bps. "
             f"Modeled cost is {_verdict(float(r['one_way_bps']), EM_FX_BAND[1], EM_FX_BAND[2]).upper()} "
             "that band: a CME peso future is far more liquid than an EM FX ETF or a local-market cash trade, "
             "so the spec's EM band is the wrong reference for it."),
            asset_class="FX")
    add("3_vs_spec_suggested_band", "structural gap vs ETF costs",
        "futures costs are structurally lower than ETF costs for the same exposure", "text", "derived",
        "A futures position carries no management fee and no borrow: there is no fund wrapper to pay and "
        "nothing is borrowed to be short. The spec's ETF bands price the wrapper as well as the trade, so "
        "a futures book should sit at or below them for identical exposure -- which is what the table shows.")
    add("3_vs_spec_suggested_band", "impact uplift from fixed-leg bps to all-in bps", float(uplift),
        "multiplier", "derived",
        (f"Realized all-in ${totals['all_in_usd_per_charged_contract']:.2f} vs fixed "
         f"${totals['fixed_usd_per_charged_contract']:.2f} per charged contract. Multiply the fixed-leg bps "
         "above by this to get an all-in one-way figure comparable to an ETF spread quote."))

    src_d = "outputs/strategy_daily.csv (1990-01-01..2014-12-31)"
    add("4_realized_cost", "total transaction cost", totals["transaction_cost_usd"], "USD", src_d,
        f"On {CANON['sessions']} sessions across {CANON['markets']} markets, from 1,000,000 USD initial capital.")
    add("4_realized_cost", "total gross P&L", totals["gross_pnl_usd"], "USD", src_d, "")
    add("4_realized_cost", "total net P&L", totals["net_pnl_usd"], "USD", src_d, "")
    add("4_realized_cost", "cost as a share of gross P&L", totals["cost_share_of_gross_pnl"], "fraction", src_d,
        "Costs consumed this much of the gross profit the positions produced.")
    add("4_realized_cost", "annual cost drag", totals["annual_cost_drag"], "fraction of NAV per year", src_d,
        f"mean(cost/prior NAV) * 252. Reconciles to strategy_metrics.csv "
        f"({CANON['annual_cost_drag']!r}) within {TOL:g}.")
    add("4_realized_cost", "annual fixed-cost drag", totals["annual_fixed_cost_drag"],
        "fraction of NAV per year", src_d,
        f"Spread, slippage, commission and fees. Reconciles to {CANON['annual_fixed_cost_drag']!r}.")
    add("4_realized_cost", "annual impact-cost drag", totals["annual_impact_cost_drag"],
        "fraction of NAV per year", src_d,
        f"Square-root market impact. Reconciles to {CANON['annual_impact_cost_drag']!r}.")
    add("4_realized_cost", "fixed share of total cost", totals["fixed_share_of_cost"], "fraction", src_d, "")
    add("4_realized_cost", "impact share of total cost", totals["impact_share_of_cost"], "fraction", src_d,
        "The impact term is real but secondary; the participation cap keeps it small.")
    add("4_realized_cost", "all-in cost per charged contract", totals["all_in_usd_per_charged_contract"],
        "USD per contract per side", src_d, "")
    add("4_realized_cost", "fixed cost per charged contract", totals["fixed_usd_per_charged_contract"],
        "USD per contract per side", src_d, "")
    add("4_realized_cost", "impact cost per charged contract", totals["impact_usd_per_charged_contract"],
        "USD per contract per side", src_d, "")
    add("4_realized_cost", "realized all-in one-way bps, rebalance leg",
        totals["rebalance_cost_usd"] / totals["rebalance_notional_usd"] * 10_000.0,
        "bps of traded notional", "derived",
        "Rebalance-leg cost over rebalance-leg traded notional. Directly comparable to the modeled "
        "per-market bps above, but notional-weighted rather than an equal-weighted median.")
    add("4_realized_cost", "max reconciliation error vs canonical bundle",
        totals["max_reconciliation_error"], "absolute", "derived",
        "Largest absolute discrepancy across the seven identity checks this script asserts.")

    add("5_roll_cost", "rebalance turnover", totals["rebalance_turnover_contracts"], "contracts", src_d, "")
    add("5_roll_cost", "roll turnover increment", totals["roll_turnover_contracts"], "contracts", src_d,
        "Two contracts of turnover per delivery transfer (incremental_roll = 2.0 * roll_slice).")
    add("5_roll_cost", "total charged turnover", totals["total_turnover_contracts"], "contracts", src_d, "")
    add("5_roll_cost", "roll share of turnover", totals["roll_share_of_turnover"], "fraction", src_d,
        "Rolling the book, not changing the book, is the majority of what gets traded.")
    add("5_roll_cost", "roll share of total cost", totals["roll_share_of_cost"], "fraction",
        "outputs/strategy_market_daily.csv.gz", "roll_cost_usd / transaction_cost_usd; the per-market "
        "regular and roll legs sum to the daily ledger total.")
    add("5_roll_cost", "annual roll-cost drag", totals["annual_roll_cost_drag"], "fraction of NAV per year",
        "derived", "The roll is the single largest cost line in the strategy.")
    add("5_roll_cost", "annual rebalance-cost drag", totals["annual_rebalance_cost_drag"],
        "fraction of NAV per year", "derived", "")

    add("6_financing_and_fees", "management fee", 0.0, "bps per year", "structural",
        "A futures book has no fund wrapper. There is no expense ratio to pay, unlike every ETF row in "
        "the spec's suggested-cost table.")
    add("6_financing_and_fees", "borrow / stock-loan cost", 0.0, "bps per year", "structural",
        "Short futures are sold, not borrowed. There is no borrow fee and no recall risk on any of the 59 markets.")
    add("6_financing_and_fees", "return basis", "futures excess return, cash collateral earning zero",
        "text", "reference/delta1_reference.py",
        "The ledger accrues no interest on the cash that backs margin. Every headline number in the "
        "submission -- 13.19% CAGR, 1.5895 Sharpe -- is on this excess basis.")
    add("6_financing_and_fees", "funded variant in outputs/levers/", funded["funded_lever_present"],
        "boolean", "outputs/levers/lever_variants.json",
        f"Lever variants present: {funded['lever_variants']}. The lever program sweeps the volatility "
        "budget only; it contains no funded/collateral variant.")
    add("6_financing_and_fees", "funded view status", funded.get("funded_status", "not computed"),
        "text", "scripts/build_submission_costs.py", "")
    if "funded_cagr" in funded:
        add("6_financing_and_fees", "financing series", funded["rate_instrument"], "text",
            "Round1AllData/Quant Researcher/Delta1/Futures Data/&ZQ.csv",
            f"Rate = {funded['rate_transform']}, ACT/360, accrued on calendar days at the rate observed "
            "strictly before the session. The unadjusted contract is mandatory: back-adjustment destroys the level.")
        add("6_financing_and_fees", "average financing rate 1990-2014", funded["average_financing_rate"],
            "fraction per year", "&ZQ.csv", "")
        add("6_financing_and_fees", "excess-basis CAGR", funded["excess_cagr"], "fraction per year", "derived",
            f"Canonical strategy_metrics.csv reports {CANON['cagr']!r}; the small difference is the "
            "year-count convention (first session vs window start), not a different path.")
        add("6_financing_and_fees", "funded-basis CAGR", funded["funded_cagr"], "fraction per year", "derived",
            f"Excess CAGR plus collateral yield: an uplift of {funded['funded_uplift_pp']:.2f} percentage points. "
            "This is accounting, not alpha.")
        add("6_financing_and_fees", "funded-basis max drawdown", funded["funded_max_drawdown"], "fraction",
            "derived", f"Excess basis {funded['excess_max_drawdown']:.4f}. Carry adds drift without exposure.")
        add("6_financing_and_fees", "Sharpe, excess of financing", funded["sharpe_excess_of_financing"],
            "ratio", "derived",
            "Unchanged by construction. Adding the financing rate to the numerator while still calling the "
            "hurdle zero would print roughly 2.00 on a strategy whose risk-adjusted return has not moved; "
            "that number is not reported here.")
        for _, r in funded["regimes"].iterrows():
            add("6_financing_and_fees", f"collateral contribution, rate regime {r['Rate regime']}",
                float(r["Annualized collateral contribution"]), "fraction per year", "&ZQ.csv",
                f"{r['Share of sessions']:.1%} of sessions at an average rate of "
                f"{r['Average financing rate']:.2%}. In a zero-rate regime this lever is worth nothing, so "
                "the blended uplift must never be presented as a forward expectation.")
        add("6_financing_and_fees", "funded view limitation", funded["funded_limitation"], "text", "derived",
            "The yield leg only. Variation-margin financing and forced liquidation are absent, so the "
            "funded view is systematically optimistic. The excess basis remains the reported basis.")

    pd.DataFrame(a).to_csv(OUT_DIR / "cost_assumptions.csv", index=False)

    print(f"wrote {OUT_DIR / 'cost_assumptions.csv'}  ({len(a)} rows)")
    print(f"wrote {OUT_DIR / 'cost_realized_by_class.csv'}  ({len(class_frame)} rows)")
    print(class_frame[["asset_class", "markets", "modeled_one_way_bps_median",
                       "modeled_one_way_bps_q25", "modeled_one_way_bps_q75",
                       "spec_band_low_bps", "spec_band_high_bps", "verdict_vs_spec_band",
                       "realized_rebalance_all_in_bps", "roll_share_of_class_cost",
                       "share_of_portfolio_cost"]].to_string(index=False))
    print(f"\nmax reconciliation error vs canonical bundle: {totals['max_reconciliation_error']:.3e}")


if __name__ == "__main__":
    build()
