"""Enumerate every supplied futures root and record why each is in or out.

    python scripts/run_universe_audit.py \
      --data-dir "Round1AllData/Quant Researcher/Delta1" \
      --output-dir outputs/universe

The repository hashes what it read and, until this runner existed, never
enumerated what it declined.  That asymmetry is the whole point: a universe of
59 markets drawn from a panel of 94 is a selection, and a selection with no
published denominator cannot be checked.

Every ground below is derived from the catalogue, the price files, or the
strategy's own configured gate -- never from returns.  No test in this file
looks at whether a market would have made money, which is what makes the
claim "excluded for correctness, never for performance" auditable rather than
asserted.

The grounds, applied in this order (the first that matches wins):

  yield_quoted        A yield- or rate-quoted contract whose P&L and notional
                      cannot be recovered from `price x point_value`.
  rate_index_quoted   An IMM-style contract quoted as 100 - rate.  Same defect:
                      `price x point_value` is not the economic notional, so
                      the gross-exposure and margin ledgers would be wrong.
  no_usd_conversion   Priced in a currency with no conversion source inside the
                      supplied panel, so a USD ledger cannot value it without
                      importing an external FX series.
  duplicate_underlying  A second listing of an underlying already carried --
                      another tenor, venue, or contract size.
  volume_gate         Fails the strategy's own configured liquidity gate
                      (60-session median volume > 1,000 contracts) on the
                      majority of its own sessions.
  available           None of the above.  Convertible, liquid, distinct, and
                      simply not included -- reported as such.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

# Rate- and yield-quoted contracts.  For each, the quoted price is an index
# (100 - rate) or a yield, so the catalogue's scalar point value reproduces
# neither the contract's economic notional nor, for the ASX bonds, its P&L.
# Valuing them needs effective-dated, rate-dependent contract functions that
# the supplied continuous panel does not carry.
YIELD_QUOTED = {"YXT", "YYT"}
RATE_INDEX_QUOTED = {"BAX", "LEU", "YIB", "YIR", "ZQ"}

# Second listings of an underlying the book already holds.  Value is the root
# that is carried instead.
DUPLICATE_OF = {
    "FDAX9": "FDAX",
    "FESX9": "FESX",
    "YAP4": "YAP",
    "YAP10": "YAP",
    "NIY": "NKD",
    "WBS": "CL",
    "MHI": "HSI",
}


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "delta1_reference", ROOT_DIR / "reference" / "delta1_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(data_dir: Path) -> pd.DataFrame:
    d1 = load_reference()
    futures = Path(data_dir) / "Futures Data"

    roots = sorted(
        {
            path.name[1:].removesuffix(".csv").removesuffix("_CCB")
            for path in futures.glob("*.csv")
        }
    )
    included = set(d1.SYMBOLS)
    convertible = {"USD"} | set(d1.FX_SOURCE)

    catalogue = pd.read_csv(Path(data_dir) / "CATALOGUE_Delta1_Futures.csv")
    catalogue["sym"] = (
        catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    )
    catalogue = catalogue[catalogue["symbol"].str.endswith("_CCB")].set_index("sym")

    records = []
    for root in roots:
        path = futures / f"&{root}_CCB.csv"
        frame = pd.read_csv(path, usecols=["Date", "Volume"])
        currency = str(catalogue.at[root, "currency"]) if root in catalogue.index else ""
        name = str(catalogue.at[root, "securityname"]) if root in catalogue.index else ""
        name = name.replace(" Continuous Contract Backadjusted", "")

        if frame.empty:
            gate_share = 0.0
        else:
            median = frame["Volume"].fillna(0.0).rolling(60, min_periods=60).median()
            gate_share = float(median.gt(d1.P["min_median_contracts"]).mean())

        if root in included:
            decision, ground, detail = "included", "traded", ""
        elif root in YIELD_QUOTED:
            decision, ground = "excluded", "yield_quoted"
            detail = "yield-quoted; price x point_value reproduces neither P&L nor notional"
        elif root in RATE_INDEX_QUOTED:
            decision, ground = "excluded", "rate_index_quoted"
            detail = "quoted as 100 - rate; price x point_value is not the economic notional"
        elif currency not in convertible:
            decision, ground = "excluded", "no_usd_conversion"
            detail = f"{currency} has no conversion source in the supplied panel"
        elif root in DUPLICATE_OF:
            decision, ground = "excluded", "duplicate_underlying"
            detail = f"second listing of {DUPLICATE_OF[root]}, which is carried"
        elif gate_share < 0.50:
            decision, ground = "excluded", "volume_gate"
            detail = (
                f"passes the 60-session median > "
                f"{int(d1.P['min_median_contracts'])} contracts gate on only "
                f"{gate_share:.1%} of its sessions"
            )
        else:
            decision, ground = "excluded", "available"
            detail = "convertible, liquid and distinct; not included in the traded universe"

        records.append(
            {
                "root": root,
                "security_name": name,
                "currency": currency,
                "asset_class": str(catalogue.at[root, "Class"]) if root in catalogue.index else "",
                "sessions": len(frame),
                "volume_gate_pass_share": round(gate_share, 4),
                "decision": decision,
                "ground": ground,
                "detail": detail,
            }
        )

    return pd.DataFrame(records).sort_values(["decision", "ground", "root"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="Round1AllData/Quant Researcher/Delta1")
    parser.add_argument("--output-dir", default="outputs/universe")
    args = parser.parse_args()

    frame = audit(Path(args.data_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "universe_audit.csv", index=False)

    summary = (
        frame.groupby(["decision", "ground"])
        .agg(roots=("root", "count"), examples=("root", lambda s: ", ".join(sorted(s)[:6])))
        .reset_index()
    )
    summary.to_csv(output_dir / "universe_audit_summary.csv", index=False)

    print(f"\n{len(frame)} supplied futures roots\n")
    print(summary.to_string(index=False))
    print(f"\nwritten to {output_dir}/universe_audit.csv")


if __name__ == "__main__":
    main()
