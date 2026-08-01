"""Ingest and audit the supplied Data Engineer DTCC/CFTC FX snapshot.

The Desktop file is a *single cumulative public-price-dissemination slice*, not
a point-in-time history spanning the futures backtest. It is therefore used as
an engineering and live-feature prototype, never backfilled into 2005-2014.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Dissemination Identifier",
    "Action type",
    "Event type",
    "Event timestamp",
    "Effective Date",
    "Expiration Date",
    "Notional amount-Leg 1",
    "Notional amount-Leg 2",
    "Notional currency-Leg 1",
    "Notional currency-Leg 2",
    "Block trade election indicator",
    "Prime brokerage transaction indicator",
    "Cleared",
    "Platform identifier",
    "UPI Underlier Name",
}


@dataclass(frozen=True)
class SnapshotAudit:
    rows: int
    columns: int
    unique_dissemination_ids: int
    event_timestamp_min: str
    event_timestamp_max: str
    new_trade_rows: int
    usd_notional_coverage: float
    capped_notional_share: float
    usable_in_2005_2014_backtest: bool
    exclusion_reason: str


def _numeric_notional(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = series.astype("string")
    capped = text.str.contains(r"\+", regex=True, na=False)
    clean = text.str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(clean, errors="coerce"), capped


def load_dtcc_fx_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing DTCC fields: {sorted(missing)}")

    frame["event_time"] = pd.to_datetime(frame["Event timestamp"], utc=True, errors="coerce")
    frame["effective_date"] = pd.to_datetime(
        frame["Effective Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    frame["expiration_date"] = pd.to_datetime(
        frame["Expiration Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    frame["tenor_days"] = (frame["expiration_date"] - frame["effective_date"]).dt.days

    leg1, capped1 = _numeric_notional(frame["Notional amount-Leg 1"])
    leg2, capped2 = _numeric_notional(frame["Notional amount-Leg 2"])
    leg1_usd = frame["Notional currency-Leg 1"].eq("USD")
    leg2_usd = frame["Notional currency-Leg 2"].eq("USD")
    frame["usd_notional"] = np.select(
        [leg1_usd & leg1.notna(), leg2_usd & leg2.notna()],
        [leg1, leg2],
        default=np.nan,
    )
    frame["notional_is_capped"] = (leg1_usd & capped1) | (leg2_usd & capped2)
    frame["pair"] = frame["UPI Underlier Name"].astype("string").str.strip()
    frame["is_new_trade"] = frame["Action type"].eq("NEWT") & frame["Event type"].eq("TRAD")
    return frame


def snapshot_audit(frame: pd.DataFrame) -> SnapshotAudit:
    new = frame.loc[frame["is_new_trade"]]
    valid_times = frame["event_time"].dropna()
    return SnapshotAudit(
        rows=len(frame),
        columns=len(frame.columns),
        unique_dissemination_ids=frame["Dissemination Identifier"].nunique(),
        event_timestamp_min=valid_times.min().isoformat(),
        event_timestamp_max=valid_times.max().isoformat(),
        new_trade_rows=len(new),
        usd_notional_coverage=float(new["usd_notional"].notna().mean()),
        capped_notional_share=float(new["notional_is_capped"].mean()),
        usable_in_2005_2014_backtest=False,
        exclusion_reason=(
            "The cumulative slice is a 2024 publication snapshot, after the 2014 "
            "backtest endpoint; injecting it into historical training would be look-ahead bias."
        ),
    )


def aggregate_fx_snapshot(frame: pd.DataFrame, min_trades: int = 20) -> pd.DataFrame:
    new = frame.loc[frame["is_new_trade"] & frame["pair"].notna()].copy()
    summary = new.groupby("pair", observed=True).agg(
        trade_count=("Dissemination Identifier", "size"),
        usd_notional=("usd_notional", "sum"),
        median_tenor_days=("tenor_days", "median"),
        block_share=("Block trade election indicator", lambda x: x.eq("TRUE").mean()),
        prime_broker_share=("Prime brokerage transaction indicator", lambda x: x.eq("TRUE").mean()),
        cleared_share=("Cleared", lambda x: x.eq("Y").mean()),
        platform_count=("Platform identifier", "nunique"),
        capped_notional_share=("notional_is_capped", "mean"),
    )
    return summary.loc[summary["trade_count"] >= min_trades].sort_values(
        ["usd_notional", "trade_count"], ascending=False
    )


def save_snapshot_outputs(path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dtcc_fx_snapshot(path)
    audit = snapshot_audit(frame)
    summary_path = output_dir / "data_engineer_fx_snapshot.csv"
    audit_path = output_dir / "data_engineer_snapshot_audit.json"
    aggregate_fx_snapshot(frame).to_csv(summary_path, index_label="pair")
    audit_path.write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")
    return summary_path, audit_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("\n".join(str(path) for path in save_snapshot_outputs(args.input, args.output_dir)))


if __name__ == "__main__":
    main()
