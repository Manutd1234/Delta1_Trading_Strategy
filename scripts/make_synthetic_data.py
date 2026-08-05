#!/usr/bin/env python3
"""Generate a synthetic dataset in the vendor layout the loaders expect.

The licensed history this project was built on cannot be redistributed, which
left the repository in an awkward state: the end-to-end tests — including
``test_full_pipeline_is_truncation_invariant``, the one that actually proves
the engine has no look-ahead — silently skip whenever that data is absent. In
continuous integration that is always. The suite was green while the guarantee
it exists to protect went unexercised.

This script closes that hole. It writes deterministic pseudo-random series in
exactly the vendor's file layout, so the whole pipeline runs against them and
the leakage proof executes on every commit.

What it is not: these prices carry no trend, no basis structure and no
cross-market correlation worth the name. Performance measured on them is
meaningless and any number produced from this data should be treated as an
arithmetic check, never as evidence. What it does establish is that the code
paths run, the ledger reconciles, and the causality invariants hold — which are
properties of the engine rather than of any particular market.

The generator is committed instead of its output: a few hundred lines of source
that anyone can read and re-run beats ten megabytes of CSV nobody can audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research.strategy import (
    FX_SOURCE,
    GLOBAL_ASSET_CLASSES,
    strategy_symbols,
)

# Contract terms are plausible but invented. They exist so the notional, margin
# and tick arithmetic has something well-formed to operate on.
CLASS_TERMS: dict[str, dict[str, float]] = {
    "Equity indices": {"tick_size": 0.25, "point_value": 50.0, "margin": 6_000.0},
    "Government bonds": {"tick_size": 0.015625, "point_value": 1_000.0, "margin": 2_000.0},
    "FX": {"tick_size": 0.0001, "point_value": 125_000.0, "margin": 2_500.0},
    "Energy": {"tick_size": 0.01, "point_value": 1_000.0, "margin": 5_000.0},
    "Metals": {"tick_size": 0.1, "point_value": 100.0, "margin": 5_500.0},
    "Agriculture & livestock": {"tick_size": 0.25, "point_value": 50.0, "margin": 2_000.0},
}

CLASS_LEVEL: dict[str, float] = {
    "Equity indices": 2_000.0,
    "Government bonds": 120.0,
    "FX": 1.10,
    "Energy": 70.0,
    "Metals": 1_200.0,
    "Agriculture & livestock": 400.0,
}

# A handful of non-USD markets so the FX conversion path is exercised rather
# than trivially bypassed.
CURRENCY_BY_SYMBOL: dict[str, str] = {
    "FDAX": "EUR", "FESX": "EUR", "FGBL": "EUR", "FGBM": "EUR",
    "FGBS": "EUR", "FGBX": "EUR", "LFT": "GBP", "LLG": "GBP",
    "NKD": "JPY", "SJB": "JPY", "FSMI": "CHF", "SXF": "CAD",
    "CGB": "CAD", "YAP": "AUD", "HTW": "AUD",
}

# Raw contract quotes, not the rates themselves: the loader multiplies each by
# its own scale, so the yen contract quotes at a hundred times the USD/JPY rate.
FX_QUOTE: dict[str, float] = {
    "EUR": 1.15, "GBP": 1.55, "JPY": 0.95, "CHF": 1.05, "CAD": 0.85, "AUD": 0.80,
}


def _asset_class(symbol: str) -> str:
    for name, members in GLOBAL_ASSET_CLASSES.items():
        if symbol in members:
            return name
    raise KeyError(symbol)


def _price_path(
    rng: np.random.Generator,
    sessions: int,
    level: float,
    annual_vol: float,
) -> np.ndarray:
    """A driftless geometric random walk.

    Deliberately driftless: a synthetic series with a built-in trend would make
    a trend-following strategy look profitable on data that contains no
    information, which is exactly the false comfort this fixture must not give.
    """

    daily = annual_vol / np.sqrt(252.0)
    shocks = rng.normal(0.0, daily, sessions)
    return level * np.exp(np.cumsum(shocks) - 0.5 * daily**2 * np.arange(sessions))


def _delivery_months(index: pd.DatetimeIndex) -> np.ndarray:
    """Quarterly delivery labels, rolling in the month before expiry."""

    labels = []
    for stamp in index:
        quarter_month = ((stamp.month - 1) // 3 + 1) * 3
        year = stamp.year
        if stamp.month == quarter_month:
            quarter_month += 3
            if quarter_month > 12:
                quarter_month -= 12
                year += 1
        labels.append(year * 100 + quarter_month)
    return np.asarray(labels, dtype=float)


def _write_market(
    directory: Path,
    symbol: str,
    index: pd.DatetimeIndex,
    close: np.ndarray,
    volume: np.ndarray,
    delivery: np.ndarray,
    rng: np.random.Generator,
) -> None:
    noise = np.abs(rng.normal(0.0, 0.002, len(index))) * close
    frame = pd.DataFrame(
        {
            "Date": index.strftime("%Y-%m-%d"),
            "Open": np.round(close - noise * 0.5, 5),
            "High": np.round(close + noise, 5),
            "Low": np.round(close - noise, 5),
            "Close": np.round(close, 5),
            "Volume": np.round(volume, 0),
            "Delivery Month": delivery,
            "Open Interest": np.round(volume * 8.0, 0),
        }
    )
    frame.to_csv(directory / f"&{symbol}.csv", index=False)

    # The back-adjusted companion differs by the accumulated roll gaps.  Giving
    # it a genuine offset matters: the basis sleeve is built from the difference
    # between the two series, so identical files would silently zero that signal
    # rather than exercise it.
    rolls = np.concatenate([[False], delivery[1:] != delivery[:-1]])
    gaps = np.where(rolls, rng.normal(0.0, 0.0015, len(index)) * close, 0.0)
    adjusted = close - np.cumsum(gaps[::-1])[::-1]
    adjusted_frame = frame.copy()
    for column in ("Open", "High", "Low", "Close"):
        adjusted_frame[column] = np.round(
            frame[column].to_numpy() + (adjusted - close), 5
        )
    adjusted_frame.to_csv(directory / f"&{symbol}_CCB.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/data/synthetic")
    )
    parser.add_argument("--start", type=str, default="2000-01-03")
    # Long enough for the ten-year daily drawdown bootstrap to have something
    # to resample, which is what the canonical-artifact test checks.
    parser.add_argument("--sessions", type=int, default=3_200)
    parser.add_argument("--seed", type=int, default=20_260_806)
    args = parser.parse_args()

    directory = args.output_dir / "Futures Data"
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range(args.start, periods=args.sessions)
    root = np.random.SeedSequence(args.seed)

    rows: list[dict[str, object]] = []
    symbols = strategy_symbols()
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(root.spawn(1)[0] if offset == 0 else
                                    np.random.SeedSequence([args.seed, offset]))
        asset_class = _asset_class(symbol)
        close = _price_path(
            rng, len(index), CLASS_LEVEL[asset_class], annual_vol=0.18
        )
        volume = np.clip(
            rng.lognormal(np.log(40_000.0), 0.35, len(index)), 5_000.0, None
        )
        delivery = _delivery_months(index)
        _write_market(directory, symbol, index, close, volume, delivery, rng)
        terms = CLASS_TERMS[asset_class]
        rows.append(
            {
                "symbol": f"&{symbol}_CCB",
                "securityname": f"Synthetic {symbol} continuous contract",
                "currency": CURRENCY_BY_SYMBOL.get(symbol, "USD"),
                "exchange": "SYNTH",
                "tick_size": terms["tick_size"],
                "point_value": terms["point_value"],
                "margin": terms["margin"],
            }
        )

    # Currency futures, so the USD conversion path is exercised.  These are
    # written unadjusted only; that is the file the FX loader reads.
    for position, (currency, (symbol, _scale)) in enumerate(FX_SOURCE.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, 9_000 + position])
        )
        # Modest volatility so a multi-year random walk stays inside the
        # loader's plausibility bounds; a synthetic currency wandering to twice
        # its starting level would fail a check that exists to catch real
        # scaling mistakes.
        close = _price_path(rng, len(index), FX_QUOTE[currency], annual_vol=0.06)
        volume = np.clip(
            rng.lognormal(np.log(30_000.0), 0.3, len(index)), 5_000.0, None
        )
        _write_market(
            directory, symbol, index, close, volume, _delivery_months(index), rng
        )

    pd.DataFrame(rows).to_csv(
        args.output_dir / "CATALOGUE_Delta1_Futures.csv", index=False
    )
    print(
        f"wrote {len(symbols)} markets plus {len(FX_SOURCE)} currency futures "
        f"({len(index)} sessions from {index[0].date()}) to {args.output_dir}"
    )
    print(
        "Synthetic prices are driftless and independent. Any performance "
        "measured on them is an arithmetic check, not evidence."
    )


if __name__ == "__main__":
    main()
