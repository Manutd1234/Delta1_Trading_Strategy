"""Download auditable public macro features used by the walk-forward model.

No API key is required. Each series is downloaded separately from FRED's CSV
endpoint, merged by observation date, and accompanied by a source manifest and
SHA-256 hashes. Raw external data remain git-ignored and are reproducible by
rerunning this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


FRED_SERIES = {
    "VIXCLS": {
        "name": "CBOE Volatility Index: VIX",
        "page": "https://fred.stlouisfed.org/series/VIXCLS",
    },
    "T10Y2Y": {
        "name": "10-Year Treasury Minus 2-Year Treasury",
        "page": "https://fred.stlouisfed.org/series/T10Y2Y",
    },
    "BAA10Y": {
        "name": "Moody's Baa Corporate Bond Yield Minus 10-Year Treasury",
        "page": "https://fred.stlouisfed.org/series/BAA10Y",
    },
}


def _download_series(series_id: str, timeout: int = 60) -> tuple[pd.DataFrame, dict[str, str]]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    frame = pd.read_csv(StringIO(response.text))
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame[["Date", series_id]], {"download_url": url, "sha256": digest}


def download_external_data(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged: pd.DataFrame | None = None
    manifest: dict[str, object] = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Federal Reserve Bank of St. Louis (FRED)",
        "series": {},
        "timing_policy": (
            "Backtest features are forward-filled only over short publication gaps "
            "and lagged one business day before model use."
        ),
    }
    for series_id, description in FRED_SERIES.items():
        frame, audit = _download_series(series_id)
        merged = frame if merged is None else merged.merge(frame, on="Date", how="outer")
        manifest["series"][series_id] = {**description, **audit}

    assert merged is not None
    merged = merged.sort_values("Date")
    csv_path = output_dir / "fred_macro.csv"
    manifest_path = output_dir / "source_manifest.json"
    merged.to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external"))
    return parser.parse_args()


def main() -> None:
    paths = download_external_data(parse_args().output_dir)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
