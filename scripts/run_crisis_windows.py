#!/usr/bin/env python3
"""Describe the frozen ledger through eight historical crisis windows.

Every window in ``CRISIS_WINDOWS`` below is declared from an external public
event date - an invasion, a rate hike, a bankruptcy filing - before this
script reads a single file.  None of the boundaries were chosen by looking at
the strategy's performance, so the resulting rows measure how the frozen
baseline behaved through episodes history picked, not through episodes the
strategy's equity curve picked.

What the script refuses to do.

It does not re-simulate.  The only input is ``outputs/strategy_daily.csv``,
the frozen daily ledger, and the only output is one descriptive CSV under
``outputs/validation``.  Nothing in the canonical bundle is touched.

It does not rank or conclude.  Rows are emitted in the order the manifest
declares them, carry no verdict column, and no window is compared against
another or against the full-sample numbers.  Whether a drawdown in one of
these windows is acceptable is a judgement this script has no standing to
make.

It does not adjust the windows after seeing a result.  The manifest is the
module-level constant, it is written verbatim into the artifact, and the test
suite pins the emitted rows back to it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# The crisis manifest: (label, start, end, external event anchor).  Each
# anchor is a public event date; each start is either that date or the first
# calendar day of the month it disrupted.  Declared here, before any file is
# read, and never edited against the ledger.
CRISIS_WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    ("1990 Gulf oil shock", "1990-08-02", "1990-12-31", "1990-08-02 Iraq invades Kuwait"),
    ("1994 bond selloff", "1994-02-04", "1994-12-30", "1994-02-04 first Fed hike of the cycle"),
    ("Asia crisis", "1997-07-02", "1997-12-31", "1997-07-02 Thai baht floated"),
    ("LTCM/Russia", "1998-08-01", "1998-10-30", "1998-08-17 Russian default and devaluation"),
    ("dot-com unwind", "2000-03-24", "2001-09-28", "2000-03-24 S&P 500 closing peak"),
    ("Lehman", "2008-09-15", "2008-12-31", "2008-09-15 Lehman Brothers bankruptcy filing"),
    ("US downgrade", "2011-08-01", "2011-09-30", "2011-08-05 S&P downgrades the US to AA+"),
    ("taper tantrum", "2013-05-22", "2013-09-30", "2013-05-22 Bernanke taper testimony"),
)

VALIDATION_STATUS = (
    "descriptive_fixed_calendar; declared from external event dates; "
    "not selection adjusted"
)

ANNUALIZATION = 252
ROLLING_SESSIONS = 21


def window_report(daily: pd.DataFrame, label: str, start: str, end: str, anchor: str) -> dict:
    """One descriptive row for a declared window of the frozen ledger."""
    window = daily.loc[start:end]
    if window.empty:
        raise ValueError(f"The ledger has no sessions in the declared window {label}")
    returns = window["net_return"]
    equity = (1.0 + returns).cumprod()
    # Wealth restarts at 1.0 at the window boundary, so the drawdown is the
    # loss from a peak inside the window, matching performance_metrics.
    running_peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = equity / running_peak - 1.0
    rolling = (1.0 + returns).rolling(ROLLING_SESSIONS).apply(np.prod, raw=True) - 1.0
    return {
        "label": label,
        "event_anchor": anchor,
        "start": start,
        "end": end,
        "sessions": int(len(window)),
        "cumulative_net_return": float(equity.iloc[-1] - 1.0),
        "annualized_volatility": float(returns.std() * math.sqrt(ANNUALIZATION)),
        "max_drawdown": float(drawdown.min()),
        "worst_rolling_21_session_net_return": float(rolling.min()),
        "cumulative_cost_drag": float(window["cost"].sum()),
        "mean_gross_notional_multiple": float(window["gross_notional_multiple"].mean()),
        "validation_status": VALIDATION_STATUS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("outputs/strategy_daily.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/validation_crisis_windows.csv"),
    )
    args = parser.parse_args()

    daily = pd.read_csv(args.ledger, index_col="date", parse_dates=True)
    rows = pd.DataFrame(
        window_report(daily, label, start, end, anchor)
        for label, start, end, anchor in CRISIS_WINDOWS
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)

    printable = rows.drop(columns=["event_anchor", "validation_status"])
    print(printable.to_string(index=False))
    print(f"\nwrote {len(rows)} declared windows to {args.output}")


if __name__ == "__main__":
    main()
