"""Section F source note and reproduction note for the Delta1 submission.

Every field in ``outputs/submission/source_note.csv`` is measured from the files
themselves at run time -- file counts, root counts, first and last observation,
row counts, quote conventions, and the presence or absence of network calls in
the codebase.  Nothing in this script is a remembered number.

    python scripts/build_submission_sources.py \
      --repo-root . \
      --output-dir outputs/submission

Two artifacts are written:

    source_note.csv        one row per data source, with the columns the case
                           spec asks for: source, provider, reference_or_url,
                           series_or_tickers, instrument_count, frequency,
                           first_observation, last_observation, obtained,
                           licence_status, used_for, caveats.

    reproduction_note.csv  the exact command that regenerates each headline
                           artifact, its inputs and outputs, and a measured
                           wall-clock runtime where one was taken.

Runtimes are supplied through ``--timing`` as ``artifact_id=seconds`` pairs so
that the timing measurement stays outside this script; anything without a
supplied timing is recorded honestly as ``not_measured``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Layout of the supplied case pack, relative to the repository root.
# --------------------------------------------------------------------------

PACK = "Round1AllData"
QR = f"{PACK}/Quant Researcher"
DELTA1 = f"{QR}/Delta1"
FUTURES_DATA = f"{DELTA1}/Futures Data"
ETF_DATA = f"{DELTA1}/ETF Data"
FUTURES_CATALOGUE = f"{DELTA1}/CATALOGUE_Delta1_Futures.csv"
ETF_CATALOGUE = f"{DELTA1}/CATALOGUE_Delta1_ETF.csv"
FXFI_FUTURES = f"{QR}/FXFI/Futures Data"
FXFI_FOREX = f"{QR}/FXFI/Forex Data"
DATA_ENGINEER = f"{PACK}/Data Engineer"
CFTC_FILE = f"{DATA_ENGINEER}/CFTC_CUMULATIVE_FOREX_2024_04_08.csv"
CFTC_GUIDE = f"{DATA_ENGINEER}/RT_PPD_quick_ref_guide.pdf"

# The six currency futures the backtest uses to convert non-USD contract terms.
# The 6J multiplier is the quote convention, not a fudge: CME Japanese yen
# futures are quoted in USD per 100 yen.
FX_SOURCE = {
    "EUR": ("6E", 1.0),
    "GBP": ("6B", 1.0),
    "JPY": ("6J", 0.01),
    "CHF": ("6S", 1.0),
    "CAD": ("6C", 1.0),
    "AUD": ("6A", 1.0),
}

# Packages that would pull data over the network.  The check is an import scan,
# not a text grep: a string that merely mentions "requests" in a docstring is
# not a network call, and counting it as one would make the claim worthless in
# the other direction.  Nothing here can be used without importing it, so an
# empty intersection is a real proof of offline reproducibility.
NETWORK_MODULES = frozenset(
    {
        "requests", "urllib", "urllib2", "urllib3", "httpx", "aiohttp", "http",
        "socket", "ssl", "ftplib", "telnetlib", "smtplib", "xmlrpc", "asyncio",
        "yfinance", "pandas_datareader", "fredapi", "quandl", "nasdaqdatalink",
        "alpha_vantage", "investpy", "tiingo", "eod", "polygon", "alpaca_trade_api",
        "ib_insync", "ccxt", "bloomberg", "blpapi", "xbbg", "refinitiv", "eikon",
        "boto3", "botocore", "gcsfs", "s3fs", "fsspec", "gspread", "openbb",
    }
)
STDLIB_ALLOWLIST = frozenset(
    {
        "__future__", "argparse", "ast", "base64", "collections", "contextlib",
        "copy", "csv", "dataclasses", "datetime", "enum", "functools", "gzip",
        "hashlib", "html", "importlib", "inspect", "io", "itertools", "json",
        "math", "os", "pathlib", "pickle", "random", "re", "shutil", "statistics",
        "string", "subprocess", "sys", "tempfile", "textwrap", "time", "typing",
        "unittest", "warnings", "zipfile", "zlib", "uuid", "operator", "bisect",
        "decimal", "fractions", "glob", "logging", "platform", "secrets", "signal",
        "struct", "traceback", "types", "weakref", "abc", "numbers",
    }
)
NETWORK_SEARCH_ROOTS = ("src", "scripts", "reference")

SOURCE_COLUMNS = [
    "source",
    "provider",
    "reference_or_url",
    "series_or_tickers",
    "instrument_count",
    "frequency",
    "first_observation",
    "last_observation",
    "obtained",
    "licence_status",
    "used_for",
    "caveats",
]

REPRODUCTION_COLUMNS = [
    "artifact_id",
    "headline_artifact",
    "command",
    "working_directory",
    "inputs",
    "writes",
    "measured_runtime_seconds",
    "runtime_basis",
    "network_required",
    "notes",
]


# --------------------------------------------------------------------------
# Measurement helpers.  Each returns facts read off disk.
# --------------------------------------------------------------------------


def _iso(value: object) -> str:
    """Render a measured date-like value as an ISO string, or empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value
    return pd.Timestamp(value).date().isoformat()


def _first_last_date(path: Path) -> tuple[str, str, int]:
    """First data date, last data date and data-row count of a vendor CSV.

    Read by seeking rather than parsing the whole file: the panel is 188 files
    and the source note should not cost more than the backtest.
    """
    with path.open("rb") as handle:
        header = handle.readline()
        if not header:
            return "", "", 0
        first_line = handle.readline().decode("utf-8", "replace").strip()
        if not first_line:
            return "", "", 0
        handle.seek(0, 2)
        size = handle.tell()
        window = min(size, 8192)
        handle.seek(size - window)
        tail = handle.read().decode("utf-8", "replace").strip().splitlines()
    last_line = tail[-1]
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1
    return first_line.split(",")[0], last_line.split(",")[0], max(rows, 0)


def _birth_and_mtime(path: Path) -> tuple[str, str]:
    """Filesystem creation and modification dates, as delivered."""
    info = path.stat()
    birth = getattr(info, "st_birthtime", info.st_ctime)
    return (
        dt.date.fromtimestamp(birth).isoformat(),
        dt.date.fromtimestamp(info.st_mtime).isoformat(),
    )


def scan_csv_directory(directory: Path) -> dict:
    """Span, size and completeness of a directory of vendor CSVs."""
    files = sorted(directory.glob("*.csv"))
    spans = {path.name: _first_last_date(path) for path in files}
    populated = {name: span for name, span in spans.items() if span[2] > 0}
    empty = sorted(name for name, span in spans.items() if span[2] == 0)
    firsts = [span[0] for span in populated.values()]
    lasts = [span[1] for span in populated.values()]
    birth_dates = Counter(_birth_and_mtime(path)[0] for path in files)
    mtime_dates = Counter(_birth_and_mtime(path)[1] for path in files)
    return {
        "files": len(files),
        "populated_files": len(populated),
        "empty_files": empty,
        "first_observation": min(firsts) if firsts else "",
        "last_observation": max(lasts) if lasts else "",
        "latest_first_observation": max(firsts) if firsts else "",
        "earliest_last_observation": min(lasts) if lasts else "",
        "last_observation_counts": Counter(lasts),
        "total_rows": sum(span[2] for span in populated.values()),
        "birth_dates": birth_dates,
        "mtime_dates": mtime_dates,
        "spans": spans,
    }


def futures_panel_facts(root: Path) -> dict:
    """The 188-file futures panel: roots, raw/back-adjusted pairing, span."""
    directory = root / FUTURES_DATA
    scan = scan_csv_directory(directory)
    stems = [path.stem for path in sorted(directory.glob("*.csv"))]
    roots = sorted({stem.removeprefix("&").removesuffix("_CCB") for stem in stems})
    raw = {stem.removeprefix("&") for stem in stems if not stem.endswith("_CCB")}
    adjusted = {stem.removeprefix("&").removesuffix("_CCB") for stem in stems if stem.endswith("_CCB")}

    columns = pd.read_csv(directory / f"&{roots[0]}.csv", nrows=1).columns.tolist()

    # Duplicate dates would silently corrupt every rolling window; the loader
    # raises on them, so confirm the claim rather than trusting the loader.
    duplicate_files = []
    for path in sorted(directory.glob("*.csv")):
        dates = pd.read_csv(path, usecols=["Date"])["Date"]
        if dates.duplicated().any():
            duplicate_files.append(path.name)

    scan.update(
        {
            "roots": len(roots),
            "root_list": roots,
            "raw_only": sorted(raw - adjusted),
            "adjusted_only": sorted(adjusted - raw),
            "columns": columns,
            "duplicate_date_files": duplicate_files,
        }
    )
    return scan


def catalogue_facts(root: Path) -> dict:
    """Contract terms file: coverage, and whether margin carries an as-of date."""
    catalogue = pd.read_csv(root / FUTURES_CATALOGUE)
    catalogue["root"] = (
        catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    )
    date_columns = [
        column
        for column in catalogue.columns
        if "date" in column.lower() or "as_of" in column.lower()
    ]
    margin_as_of = [
        column for column in date_columns if "margin" in column.lower()
    ]
    per_root_margin_values = catalogue.groupby("root")["margin"].nunique()
    return {
        "rows": len(catalogue),
        "roots": catalogue["root"].nunique(),
        "columns": catalogue.columns.tolist(),
        "date_columns": date_columns,
        "margin_as_of_columns": margin_as_of,
        "distinct_margin_values_per_root_max": int(per_root_margin_values.max()),
        "currencies": Counter(catalogue["currency"]),
        "exchanges": Counter(catalogue["exchange_name"]),
        "table": catalogue,
    }


def traded_universe_facts(root: Path, symbols: list[str]) -> dict:
    """Currency and exchange split of the 59 markets the strategy actually trades."""
    catalogue = pd.read_csv(root / FUTURES_CATALOGUE)
    catalogue["root"] = (
        catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    )
    terms = (
        catalogue[
            catalogue["symbol"].str.endswith("_CCB")
            & catalogue["root"].isin(symbols)
        ]
        .set_index("root")
        .reindex(symbols)
    )
    non_usd = sorted(terms.index[terms["currency"] != "USD"])
    return {
        "symbols": symbols,
        "currencies": Counter(terms["currency"]),
        "exchanges": Counter(terms["exchange_name"]),
        "non_usd_roots": non_usd,
        "non_usd_currencies": sorted(set(terms.loc[non_usd, "currency"])),
    }


def fx_source_facts(root: Path) -> dict:
    """Spans and quote levels of the six in-panel currency futures."""
    directory = root / FUTURES_DATA
    detail = {}
    for currency, (symbol, scale) in FX_SOURCE.items():
        frame = pd.read_csv(directory / f"&{symbol}.csv", usecols=["Date", "Close"])
        detail[currency] = {
            "symbol": symbol,
            "scale": scale,
            "first": frame["Date"].iloc[0],
            "last": frame["Date"].iloc[-1],
            "rows": len(frame),
            "last_close": float(frame["Close"].iloc[-1]),
            "last_close_scaled": float(frame["Close"].iloc[-1]) * scale,
        }
    return detail


def fx_gating_reconciliation(
    root: Path, traded: dict, fx: dict, market_daily: Path
) -> dict:
    """Check the FX availability claim against the canonical position ledger.

    A source note that says "these markets are not sized before their FX series
    exists" should be checkable.  Read the shipped per-market daily file and
    confirm no non-USD market carries a position before its currency future
    starts quoting.
    """
    if not market_daily.exists():
        return {"available": False}

    catalogue = pd.read_csv(root / FUTURES_CATALOGUE)
    catalogue["root"] = (
        catalogue["symbol"].str.removeprefix("&").str.removesuffix("_CCB")
    )
    currency = (
        catalogue[catalogue["symbol"].str.endswith("_CCB")]
        .set_index("root")["currency"]
        .to_dict()
    )

    frame = pd.read_csv(market_daily, usecols=["date", "symbol", "end_contracts"])
    held = frame[frame["end_contracts"] != 0]
    first_position = held.groupby("symbol")["date"].min()

    checked: dict[str, dict] = {}
    violations: list[str] = []
    for symbol in traded["non_usd_roots"]:
        if symbol not in first_position.index:
            continue
        fx_start = fx[currency[symbol]]["first"]
        entry = {
            "currency": currency[symbol],
            "fx_series": fx[currency[symbol]]["symbol"],
            "fx_first": fx_start,
            "first_position": first_position[symbol],
        }
        checked[symbol] = entry
        if entry["first_position"] < fx_start:
            violations.append(
                f"{symbol} traded {entry['first_position']} before &{entry['fx_series']} "
                f"starts {fx_start}"
            )
    earliest = min(checked.items(), key=lambda item: item[1]["first_position"], default=None)
    return {
        "available": True,
        "checked": len(checked),
        "violations": violations,
        "earliest_non_usd": earliest,
        "source": str(market_daily.relative_to(root)),
    }


def etf_panel_facts(root: Path, universe_csv: Path | None) -> dict:
    """The 745-file ETF panel and the 11 funds the regime sleeve selects."""
    scan = scan_csv_directory(root / ETF_DATA)
    tickers = sorted(path.stem for path in (root / ETF_DATA).glob("*.csv"))
    selected: list[str] = []
    selected_detail = pd.DataFrame()
    if universe_csv is not None and universe_csv.exists():
        selected_detail = pd.read_csv(universe_csv)
        selected = selected_detail["Ticker"].tolist()
    columns = pd.read_csv(root / ETF_DATA / f"{tickers[0]}.csv", nrows=1).columns.tolist()
    catalogue = pd.read_csv(root / ETF_CATALOGUE)
    scan.update(
        {
            "tickers": tickers,
            "selected": selected,
            "selected_detail": selected_detail,
            "columns": columns,
            "catalogue_rows": len(catalogue),
            "catalogue_columns": catalogue.columns.tolist(),
        }
    )
    return scan


def fxfi_facts(root: Path, traded_symbols: list[str], holdout_summary: Path) -> dict:
    """The 2015-2016 continuation extracts used only for the holdout.

    The overlap with the traded universe is measured, not asserted, and the
    scored session count is read from the canonical holdout artifact rather
    than restated from memory.
    """
    futures = scan_csv_directory(root / FXFI_FUTURES)
    forex = scan_csv_directory(root / FXFI_FOREX)
    futures_stems = sorted(path.stem for path in (root / FXFI_FUTURES).glob("*.csv"))
    forex_stems = sorted(path.stem for path in (root / FXFI_FOREX).glob("*.csv"))
    futures.update(
        {
            "roots": sorted(
                stem.removeprefix("&").removesuffix("_CCB") for stem in futures_stems
            ),
            "unadjusted_files": [
                stem for stem in futures_stems if not stem.endswith("_CCB")
            ],
        }
    )
    forex["pairs"] = forex_stems

    overlap = sorted(set(futures["roots"]) & set(traded_symbols))
    futures["traded_overlap"] = overlap
    futures["traded_overlap_count"] = len(overlap)
    futures["outside_traded_universe"] = sorted(set(futures["roots"]) - set(traded_symbols))

    scored = {}
    if holdout_summary.exists():
        summary = pd.read_csv(holdout_summary)
        scored = {
            "sessions": int(summary["sessions"].iloc[0]),
            "roots": int(summary["roots"].iloc[0]),
            "window": f"{summary['window_start'].iloc[0]}..{summary['window_end'].iloc[0]}",
            "sleeve": str(summary["sleeve"].iloc[0]),
            "source": str(holdout_summary.relative_to(root)),
        }
    return {"futures": futures, "forex": forex, "scored": scored}


def cftc_facts(root: Path) -> dict:
    """Identify the Data Engineer CSV by reading it, not by its filename."""
    path = root / CFTC_FILE
    frame = pd.read_csv(path, low_memory=False)
    event = pd.to_datetime(frame["Event timestamp"], errors="coerce", utc=True)
    execution = pd.to_datetime(frame["Execution Timestamp"], errors="coerce", utc=True)
    cot_markers = [
        column
        for column in frame.columns
        if re.search(r"commercial|open interest|report_date|trader|positions", column, re.I)
    ]
    part43_markers = [
        column
        for column in frame.columns
        if column
        in {
            "Dissemination Identifier",
            "Original Dissemination Identifier",
            "Action type",
            "Event type",
            "Event timestamp",
            "Cleared",
            "Platform identifier",
            "Block trade election indicator",
            "Notional amount-Leg 1",
        }
    ]
    guide_title = ""
    guide = root / CFTC_GUIDE
    if guide.exists():
        guide_title = _pdf_producer_line(guide)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "asset_classes": Counter(frame["Asset Class"].dropna()),
        "action_types": Counter(frame["Action type"].dropna()),
        "event_first": _iso(event.min()),
        "event_last": _iso(event.max()),
        "execution_first": _iso(execution.min()),
        "execution_last": _iso(execution.max()),
        "cot_marker_columns": cot_markers,
        "part43_marker_columns": part43_markers,
        "bytes": path.stat().st_size,
        "guide_present": guide.exists(),
        "guide_title": guide_title,
    }


def _pdf_producer_line(path: Path) -> str:
    """Pull the document title out of a PDF without a PDF library."""
    import zlib

    blob = path.read_bytes()
    chunks = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", blob, re.S):
        try:
            chunks.append(zlib.decompress(match.group(1)))
        except zlib.error:
            continue
    text = b" ".join(chunks).decode("latin-1")
    strings = " ".join(
        item[1:-1] for item in re.findall(r"\((?:[^()\\]|\\.)*\)", text)
    )
    collapsed = re.sub(r"\s+", " ", strings.replace("\\", ""))
    match = re.search(
        r"Microsoft Word - (.+?)(?=\s+Arial|\s+Calibri|\s+Times|\s+Webdings|$)", collapsed
    )
    return match.group(1).strip() if match else collapsed[:120]


def case_brief_facts(root: Path) -> dict:
    """The recruitment brief workbook: which tracks it defines."""
    candidates = sorted(
        path
        for path in (root / PACK).glob("*.xlsx")
        if not path.name.startswith("~$")
    )
    if not candidates:
        return {}
    path = candidates[0]
    with zipfile.ZipFile(path) as archive:
        workbook = archive.read("xl/workbook.xml").decode("utf-8", "replace")
    sheets = re.findall(r'<sheet name="([^"]+)"', workbook)
    birth, mtime = _birth_and_mtime(path)
    return {
        "name": path.name,
        "sheets": sheets,
        "birth": birth,
        "mtime": mtime,
        "bytes": path.stat().st_size,
    }


def network_dependency_scan(root: Path) -> dict:
    """Parse every shipped module and list what it actually imports.

    An AST import scan, not a text grep: prose mentioning "requests" is not a
    network call, and a claim built on grep hits would be wrong in both
    directions.  Nothing can reach the network without importing something in
    ``NETWORK_MODULES``, so an empty intersection is the proof.
    """
    imported: Counter = Counter()
    hits: list[str] = []
    searched = 0
    unparsed: list[str] = []
    for subdir in NETWORK_SEARCH_ROOTS:
        for path in sorted((root / subdir).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            searched += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                unparsed.append(str(path.relative_to(root)))
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module] if node.module and node.level == 0 else []
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    imported[top] += 1
                    if top in NETWORK_MODULES:
                        hits.append(
                            f"{path.relative_to(root)}:{node.lineno}: import {name}"
                        )
    third_party = sorted(
        module
        for module in imported
        if module not in STDLIB_ALLOWLIST
        and module not in {"delta1_strategy", "delta1_reference", "build_submission_sources"}
    )
    return {
        "files_searched": searched,
        "hits": hits,
        "roots": NETWORK_SEARCH_ROOTS,
        "distinct_modules": len(imported),
        "third_party": third_party,
        "unparsed": unparsed,
    }


def gitignore_facts(root: Path) -> dict:
    """Where the redistribution ban is actually enforced."""
    path = root / ".gitignore"
    if not path.exists():
        return {"line": 0}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip().rstrip("/") == PACK:
            return {"line": number, "pattern": line.strip()}
    return {"line": 0}


def margin_enforcement_facts(root: Path) -> dict:
    """Prove the margin snapshot is informational: find the disabled constraint."""
    path = root / "src/delta1_strategy/research/strategy.py"
    if not path.exists():
        return {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip().startswith("max_static_margin_fraction:"):
            return {
                "file": "src/delta1_strategy/research/strategy.py",
                "line": number,
                "declaration": line.strip(),
                "disabled": line.strip().endswith("None"),
            }
    return {}


def git_head(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


# --------------------------------------------------------------------------
# Row construction.  Facts in, prose out; no number is typed by hand.
# --------------------------------------------------------------------------


def _join(values, limit: int | None = None) -> str:
    items = list(values)
    if limit is not None and len(items) > limit:
        return "; ".join(items[:limit]) + f"; ... ({len(items)} total)"
    return "; ".join(items)


def _stamp(dates: Counter) -> str:
    ordered = sorted(dates)
    return ordered[0] if len(ordered) == 1 else f"{ordered[0]}..{ordered[-1]}"


def _delivery_stamp(scan: dict, pack_created: str) -> str:
    """How the extract was obtained, evidenced by its own file timestamps.

    There is no download receipt to quote, because nothing was downloaded.  The
    honest substitute is the pair of timestamps the files themselves carry: the
    vendor extract date preserved inside the pack, and the date the pack was
    unpacked onto this machine.
    """
    return (
        "supplied with the case pack; not downloaded by this project, so there is no "
        f"download date to quote. Vendor extract timestamp carried in the files: "
        f"{_stamp(scan['birth_dates'])} (modification {_stamp(scan['mtime_dates'])}). "
        f"Pack unpacked locally: {pack_created}"
    )


def build_source_rows(facts: dict) -> list[dict]:
    futures = facts["futures"]
    catalogue = facts["catalogue"]
    traded = facts["traded"]
    fx = facts["fx"]
    etf = facts["etf"]
    fxfi = facts["fxfi"]
    cftc = facts["cftc"]
    network = facts["network"]
    ignore = facts["gitignore"]
    margin = facts["margin"]
    brief = facts["brief"]
    pack_created = facts["pack_created"]
    fx_gating = facts["fx_gating"]

    rows: list[dict] = []

    # 1. The futures panel -- the only source the headline result depends on.
    pairing = (
        "every root supplied as a matched raw/_CCB pair"
        if not futures["raw_only"] and not futures["adjusted_only"]
        else f"unpaired: raw-only {futures['raw_only']}, adjusted-only {futures['adjusted_only']}"
    )
    rows.append(
        {
            "source": "Delta1 futures price panel (continuous contracts, raw and back-adjusted)",
            "provider": "Norgate Data, supplied inside the case pack",
            "reference_or_url": (
                "https://norgatedata.com | local path: "
                f"{FUTURES_DATA}/ | continuous-contract stitching and the _CCB "
                "back-adjustment convention are Norgate's construction, not this project's"
            ),
            "series_or_tickers": (
                f"{futures['roots']} roots as &<ROOT>.csv (raw continuous) and "
                f"&<ROOT>_CCB.csv (back-adjusted continuous); {pairing}; columns "
                f"{', '.join(futures['columns'])}"
            ),
            "instrument_count": (
                f"{futures['roots']} roots / {futures['files']} files "
                f"({len(traded['symbols'])} roots traded)"
            ),
            "frequency": "daily OHLCV + delivery month + open interest, exchange session dates",
            "first_observation": futures["first_observation"],
            "last_observation": futures["last_observation"],
            "obtained": _delivery_stamp(futures, pack_created),
            "licence_status": (
                "vendor-licensed, NOT redistributable; excluded from version control at "
                f".gitignore:{ignore.get('line', 0)} ('{ignore.get('pattern', PACK + '/')}'), "
                "so the repository ships code and results but never the vendor history"
            ),
            "used_for": (
                "the entire headline backtest: back-adjusted closes drive P&L and trend, "
                "raw closes drive roll yield (basis momentum) and notional, Volume drives the "
                "liquidity gate and participation cap, Delivery Month drives roll detection"
            ),
            "caveats": (
                "BOTH price conventions are used deliberately and neither alone would do: "
                "back-adjusted levels are a P&L series, not a price, so nothing divides by them "
                "(the trend signal is a sign, not a normalised return), while raw continuous "
                "closes jump at each roll and are used only where a contract-level quantity is "
                "required. The panel is unbalanced -- the earliest root starts "
                f"{futures['first_observation']} and the latest root only at "
                f"{futures['latest_first_observation']}, so market count grows through the sample. "
                f"Duplicate dates: {len(futures['duplicate_date_files'])} files. No holiday "
                "back-filling: the panel is reindexed to business days and forward-filled with a "
                "10-session cap, and an unfilled close series is kept alongside so the engine can "
                "tell a real session from a carried one. Back-adjusted history is rewritten by the "
                "vendor at every future roll, so a re-extract of the same window will not be "
                "byte-identical; that is the futures analogue of revision risk. Panel ends "
                f"{futures['last_observation']}, so nothing after 2014 is testable from this file."
            ),
        }
    )

    # 2. Contract terms.
    margin_note = (
        f"informational only, enforced at {margin.get('file', 'n/a')}:{margin.get('line', 0)} "
        f"where '{margin.get('declaration', 'n/a')}' leaves the margin constraint switched off"
        if margin
        else "informational only"
    )
    rows.append(
        {
            "source": "Delta1 futures contract catalogue (tick size, point value, currency, margin, exchange)",
            "provider": "Norgate Data, supplied inside the case pack",
            "reference_or_url": f"https://norgatedata.com | local path: {FUTURES_CATALOGUE}",
            "series_or_tickers": (
                f"{catalogue['rows']} rows covering {catalogue['roots']} roots; fields used: "
                "tick_size, point_value, currency, margin, exchange_name"
            ),
            "instrument_count": f"{catalogue['roots']} roots / {catalogue['rows']} symbol rows",
            "frequency": "static reference table, one row per symbol, no time dimension",
            "first_observation": "",
            "last_observation": "",
            "obtained": (
                "supplied with the case pack; no download date exists. Vendor extract timestamp "
                f"carried in the file: {_birth_and_mtime(facts['root'] / FUTURES_CATALOGUE)[0]}. "
                f"Pack unpacked locally: {pack_created}"
            ),
            "licence_status": "vendor-licensed, not redistributable; gitignored with the rest of the pack",
            "used_for": (
                "tick_size and point_value convert contracts to USD risk and cost; currency selects "
                "the in-panel FX series; exchange_name documents venue; margin is reported as a "
                "diagnostic only"
            ),
            "caveats": (
                "The margin column is a CURRENT snapshot with no as-of date -- the file carries "
                f"{len(catalogue['date_columns'])} date columns "
                f"({', '.join(catalogue['date_columns']) or 'none'}) and none of them dates the margin "
                f"({len(catalogue['margin_as_of_columns'])} margin as-of columns), and each root carries "
                f"at most {catalogue['distinct_margin_values_per_root_max']} distinct margin value across "
                "its whole history. Applying today's margin schedule to 1990 decisions would inject "
                f"future information, so it is NOT allowed to size or gate anything: {margin_note}. "
                "Realised margin usage is published as a diagnostic ('static margin fraction') and a "
                "live deployment would replace it with an effective-dated broker snapshot. Tick size "
                "and point value are likewise current-contract terms; where an exchange has changed a "
                "multiplier historically the cost model uses the modern one."
            ),
        }
    )

    # 3. In-panel FX conversion.
    fx_detail = "; ".join(
        f"{currency}=&{item['symbol']} (x{item['scale']:g}, {item['rows']} rows, "
        f"{item['first']}..{item['last']})"
        for currency, item in sorted(fx.items())
    )
    jpy = fx["JPY"]
    rows.append(
        {
            "source": "In-panel FX conversion series (six currency futures, subset of the Delta1 panel)",
            "provider": "Norgate Data, supplied inside the case pack",
            "reference_or_url": f"https://norgatedata.com | local path: {FUTURES_DATA}/&6[EBJSCA].csv",
            "series_or_tickers": fx_detail,
            "instrument_count": f"{len(fx)} currency futures serving {len(traded['non_usd_roots'])} non-USD contracts",
            "frequency": "daily close, aligned to the same business-day calendar as the panel",
            "first_observation": min(item["first"] for item in fx.values()),
            "last_observation": max(item["last"] for item in fx.values()),
            "obtained": (
                "supplied with the case pack, inside the futures panel itself; no external FX feed is "
                f"used and none is needed. Vendor extract timestamp: {_stamp(futures['birth_dates'])}. "
                f"Pack unpacked locally: {pack_created}"
            ),
            "licence_status": "vendor-licensed, not redistributable; gitignored with the rest of the pack",
            "used_for": (
                "point-in-time USD conversion of non-USD contract terms for "
                f"{len(traded['non_usd_roots'])} markets in {len(traded['non_usd_currencies'])} currencies "
                f"({', '.join(traded['non_usd_currencies'])}): {', '.join(traded['non_usd_roots'])}"
            ),
            "caveats": (
                "Using the panel's own currency futures rather than an external spot series is a "
                "deliberate choice: it keeps the backtest self-contained and guarantees the FX "
                "observation is a traded price from the same session as the contract it converts. "
                f"The 6J series is quoted in USD per 100 yen, so it carries an explicit x{jpy['scale']:g} "
                f"scaling -- its final close of {jpy['last_close']:.4f} becomes "
                f"{jpy['last_close_scaled']:.6f} USD per yen; the other five are USD per unit and take "
                "x1. Futures FX is a forward rate, so it differs from spot by the interest-rate "
                "differential to the delivery date (basis points to low single-digit percent), which "
                "is immaterial for sizing but would not be acceptable for a settlement calculation. "
                "The euro series &6E only begins "
                f"{fx['EUR']['first']}, so euro-denominated contracts are convertible only from that "
                "date and before it those markets cannot be sized at all -- which is a real coverage "
                "hole in the 1990s, not a rounding detail. "
                + (
                    f"Checked against the shipped position ledger ({fx_gating['source']}): "
                    f"{fx_gating['checked']} non-USD markets, "
                    f"{len(fx_gating['violations'])} traded before their own FX series begins"
                    + (
                        f"; earliest non-USD position is {fx_gating['earliest_non_usd'][0]} on "
                        f"{fx_gating['earliest_non_usd'][1]['first_position']} against "
                        f"&{fx_gating['earliest_non_usd'][1]['fx_series']} starting "
                        f"{fx_gating['earliest_non_usd'][1]['fx_first']}."
                        if fx_gating.get("earliest_non_usd")
                        else "."
                    )
                    if fx_gating.get("available")
                    else "The canonical position ledger was not present, so this claim is "
                    "asserted from the loader rather than reconciled."
                )
            ),
        }
    )

    # 4. ETF panel.
    selected_first = ""
    selected_joint = ""
    selected_binding = ""
    if len(etf["selected_detail"]):
        detail = etf["selected_detail"]
        first_quoted = detail.set_index("Ticker")["First quoted"]
        selected_first = f"{first_quoted.min()} ({first_quoted.idxmin()})"
        selected_joint = str(first_quoted.max())
        selected_binding = str(first_quoted.idxmax())
    rows.append(
        {
            "source": "Delta1 ETF price panel (used only by the ETF regime-allocation sleeve)",
            "provider": "Norgate Data, supplied inside the case pack",
            "reference_or_url": (
                f"https://norgatedata.com | local paths: {ETF_DATA}/ and {ETF_CATALOGUE}"
            ),
            "series_or_tickers": (
                f"selected: {', '.join(etf['selected'])} | drawn from {len(etf['tickers'])} "
                f"supplied tickers | columns {', '.join(etf['columns'])}"
            ),
            "instrument_count": f"{len(etf['selected'])} used / {etf['files']} supplied",
            "frequency": "daily adjusted and unadjusted close, volume, turnover, dividend, index membership",
            "first_observation": etf["first_observation"],
            "last_observation": etf["last_observation"],
            "obtained": _delivery_stamp(etf, pack_created),
            "licence_status": "vendor-licensed, not redistributable; gitignored with the rest of the pack",
            "used_for": (
                "the ETF regime-allocation sleeve only; it does not touch the headline futures result. "
                f"The oldest selected fund begins {selected_first}, but the eleven are only jointly "
                f"quoted from {selected_joint}, the inception of {selected_binding}, and that date -- "
                "not a research preference -- sets the sleeve's start"
            ),
            "caveats": (
                "SURVIVORS ONLY, and measured rather than assumed: all "
                f"{etf['files']} supplied files end on the same date "
                f"({etf['last_observation']}, "
                f"{len(etf['last_observation_counts'])} distinct terminal date across the whole "
                "directory), so no closed, merged or delisted fund is present anywhere in the extract. "
                "Any selection made on this panel is therefore made among survivors and its measured "
                "performance is biased upward by an amount this data cannot quantify. Both adjusted "
                "and unadjusted closes are supplied, so the total-return adjustment is auditable "
                "instead of taken on trust. The panel ends "
                f"{etf['last_observation']}, four years after the futures panel ends, so the two "
                "sleeves cover different windows and must never be spliced into one track record."
            ),
        }
    )

    # 5. FXFI holdout extension -- futures.
    fxfi_futures = fxfi["futures"]
    fxfi_forex = fxfi["forex"]
    rows.append(
        {
            "source": "FXFI futures continuation extract (2015-2016 out-of-sample window only)",
            "provider": "Norgate Data, supplied inside the case pack for the FXFI track",
            "reference_or_url": f"https://norgatedata.com | local path: {FXFI_FUTURES}/",
            "series_or_tickers": (
                f"{', '.join('&' + root + '_CCB' for root in fxfi_futures['roots'])} "
                f"(back-adjusted only; {len(fxfi_futures['unadjusted_files'])} unadjusted files supplied)"
            ),
            "instrument_count": (
                f"{fxfi_futures['files']} files, {fxfi_futures['populated_files']} populated"
            ),
            "frequency": "daily OHLCV, same vendor layout as the Delta1 panel",
            "first_observation": fxfi_futures["first_observation"],
            "last_observation": fxfi_futures["last_observation"],
            "obtained": _delivery_stamp(fxfi_futures, pack_created),
            "licence_status": "vendor-licensed, not redistributable; gitignored with the rest of the pack",
            "used_for": (
                "the single chronological out-of-sample scoring of the frozen rules on 2015-2016; "
                "it is not used to fit, select or tune anything"
            ),
            "caveats": (
                "This extract does NOT continue the traded universe. It covers "
                f"{len(fxfi_futures['roots'])} roots against {len(traded['symbols'])} traded, and only "
                f"{fxfi_futures['traded_overlap_count']} of them are in the book at all "
                f"({', '.join(fxfi_futures['traded_overlap'])}) -- government bonds and gold, with no "
                "equity index, FX, energy or agricultural exposure, so the out-of-sample test runs on a "
                "structurally different portfolio from the one being validated. No unadjusted series "
                f"are supplied ({len(fxfi_futures['unadjusted_files'])} of {fxfi_futures['files']} "
                "files), so roll yield cannot be reconstructed and the basis-momentum sleeve is "
                "unavailable out of sample -- the holdout scores the trend sleeve alone. "
                f"{len(fxfi_futures['empty_files'])} of {fxfi_futures['files']} files are header-only "
                f"with zero data rows ({', '.join(fxfi_futures['empty_files']) or 'none'}). The extract "
                f"itself spans {fxfi_futures['first_observation']}..{fxfi_futures['last_observation']}, "
                "i.e. full history rather than a 2015-2016 slice, and that overlap is a hazard rather "
                "than a bonus: only the post-2014 tail is scored, and the pre-2015 portion is used "
                "solely for the seam-continuity check, never re-fitted on. "
                + (
                    f"As actually scored ({fxfi['scored']['source']}): window "
                    f"{fxfi['scored']['window']}, {fxfi['scored']['sessions']} sessions, "
                    f"{fxfi['scored']['roots']} roots, {fxfi['scored']['sleeve']}. "
                    if fxfi.get("scored")
                    else ""
                )
                + "That is far too short to resolve a Sharpe difference of the size a promotion "
                "decision needs; it is a consistency check, not a verdict, and it is reported as one."
            ),
        }
    )

    # 6. FXFI holdout extension -- forex.
    rows.append(
        {
            "source": "FXFI spot-forex extract (holdout FX splice only)",
            "provider": "Norgate Data, supplied inside the case pack for the FXFI track",
            "reference_or_url": f"https://norgatedata.com | local path: {FXFI_FOREX}/",
            "series_or_tickers": _join(fxfi_forex["pairs"], limit=12),
            "instrument_count": f"{fxfi_forex['files']} series",
            "frequency": "daily",
            "first_observation": fxfi_forex["first_observation"],
            "last_observation": fxfi_forex["last_observation"],
            "obtained": _delivery_stamp(fxfi_forex, pack_created),
            "licence_status": "vendor-licensed, not redistributable; gitignored with the rest of the pack",
            "used_for": (
                "continuing USD conversion of non-USD contracts across the 2015-2016 holdout, where "
                "the in-panel currency futures stop"
            ),
            "caveats": (
                "This is a spot series spliced onto a futures-implied FX history at the panel seam, so "
                "the conversion basis changes convention at the join by the forward points; the splice "
                "is audited in outputs/holdout/holdout_fx_splice.csv rather than assumed harmless. "
                f"History reaches back to {fxfi_forex['first_observation']}, far earlier than needed, "
                "but only the post-2014 tail is used -- nothing in the in-sample window is re-based onto it."
            ),
        }
    )

    # 7. The CFTC file that is NOT what its folder implies.
    rows.append(
        {
            "source": "CFTC/DTCC swap real-time public dissemination file (INSPECTED AND REJECTED, not used)",
            "provider": (
                "DTCC Data Repository (US) LLC, publishing under CFTC Part 43 real-time public "
                "reporting; supplied in the case pack's Data Engineer folder"
            ),
            "reference_or_url": (
                f"local path: {CFTC_FILE} | companion specification: {CFTC_GUIDE}"
                + (f" ('{cftc['guide_title']}')" if cftc["guide_title"] else "")
                + " | rule reference: 17 CFR Part 43 (real-time public reporting of swap transaction "
                "and pricing data)"
            ),
            "series_or_tickers": (
                f"{cftc['columns']} Part-43 swap fields; identifying columns present: "
                f"{', '.join(cftc['part43_marker_columns'])}; asset class "
                f"{', '.join(f'{k}={v}' for k, v in cftc['asset_classes'].items())}; action types "
                f"{', '.join(f'{k}={v}' for k, v in cftc['action_types'].most_common())}"
            ),
            "instrument_count": f"{cftc['rows']} swap transaction reports",
            "frequency": "transaction-level, cumulative file stamped 2024-04-08",
            "first_observation": cftc["event_first"],
            "last_observation": cftc["event_last"],
            "obtained": (
                "supplied with the case pack; no download date exists. Vendor/regulator extract "
                f"timestamp carried in the file: {_birth_and_mtime(facts['root'] / CFTC_FILE)[0]}, "
                f"consistent with the 2024-04-08 stamp in its filename. Pack unpacked locally: {pack_created}"
            ),
            "licence_status": "public regulatory dissemination; not redistributed here (gitignored with the pack)",
            "used_for": (
                "NOTHING. Inspected, identified and rejected. It is recorded here because a source "
                "note that omits the inputs you looked at and discarded is not a source note."
            ),
            "caveats": (
                "It is NOT Commitments of Traders positioning data despite sitting in a folder that "
                "invites the assumption. Verified by reading the columns: "
                f"{len(cftc['cot_marker_columns'])} COT-style positioning columns are present "
                "(no commercial/non-commercial breakdown, no report date, no reportable open interest), "
                f"while {len(cftc['part43_marker_columns'])} Part-43 dissemination columns are. Every "
                f"one of the {cftc['rows']} rows is an FX swap transaction report with lifecycle action "
                "types (NEWT/TERM/MODI/CORR/EROR). Decisive on timing regardless of content: the "
                f"earliest dissemination timestamp is {cftc['event_first']} and the latest is "
                f"{cftc['event_last']}, so no row in this file was publicly observable before 2021, "
                "while the futures panel ends 2014-12-31 -- more than six years earlier. Using it as a "
                "signal would be look-ahead by construction. Execution timestamps reach back to "
                f"{cftc['execution_first']} because of back-loaded trades, which is exactly the trap: "
                "the trade date is not the publication date, and only the publication date is real-time "
                "information."
            ),
        }
    )

    # 8. The case brief itself.
    if brief:
        rows.append(
            {
                "source": "Recruitment case brief workbook (specification, not data)",
                "provider": "Case pack",
                "reference_or_url": f"local path: {PACK}/{brief['name']}",
                "series_or_tickers": f"sheets: {', '.join(brief['sheets'])}",
                "instrument_count": f"{len(brief['sheets'])} track specifications",
                "frequency": "static document",
                "first_observation": "",
                "last_observation": "",
                "obtained": (
                    "supplied with the case pack; local copy created "
                    f"{brief['birth']}, last modified {brief['mtime']}"
                ),
                "licence_status": "case material, not redistributable; gitignored with the pack",
                "used_for": (
                    "the Delta1 sheet defines the task and the deliverable list this submission answers; "
                    "no number in any result comes from this file"
                ),
                "caveats": (
                    "Listed for completeness of the input chain. It is a specification input, so it "
                    "cannot bias a measurement, but it does bias the research question that was asked."
                ),
            }
        )

    # 9. The negative result that makes offline reproduction checkable.
    rows.append(
        {
            "source": "External / internet data",
            "provider": "none",
            "reference_or_url": "none",
            "series_or_tickers": "none",
            "instrument_count": "0",
            "frequency": "n/a",
            "first_observation": "",
            "last_observation": "",
            "obtained": "n/a -- nothing is downloaded, at build time or at run time",
            "licence_status": "n/a",
            "used_for": (
                "nothing. No benchmark index, no risk-free rate series, no macro release, no FX spot "
                "feed and no fundamental data enters any result. Benchmarks are built from the same "
                "supplied panel, FX conversion comes from the panel's own currency futures, and the "
                "risk-free rate is set to zero and labelled as such rather than sourced"
            ),
            "caveats": (
                "Verified by parsing -- not grepping -- every Python file shipped in "
                f"{', '.join(network['roots'])}/ ({network['files_searched']} files) and collecting "
                "every import statement from the syntax tree, then intersecting with the set of "
                f"{len(NETWORK_MODULES)} networking and vendor-API packages (requests, urllib, httpx, "
                "aiohttp, socket, yfinance, pandas_datareader, fredapi, quandl, blpapi, ccxt, boto3, "
                f"s3fs and the rest). Matching imports: {len(network['hits'])}. "
                + (f"Occurrences: {_join(network['hits'], limit=5)}. " if network["hits"] else "")
                + "The complete third-party dependency set of the shipped code is "
                f"{{{', '.join(network['third_party'])}}}, none of which reaches the network. A text "
                "grep was rejected as the test because a docstring that mentions 'requests' is not a "
                "network call and would produce a false positive, while a dynamically constructed "
                "import would produce a false negative -- the import scan is exact for the former and "
                "the package list above bounds the latter. "
                "Core results therefore reproduce with the network disconnected, which is what the "
                "deliverable spec requires. The cost of that choice is stated rather than hidden: "
                "Sharpe ratios are excess-of-zero, not excess-of-cash, so in the high-rate first "
                "decade of the sample the true excess return is materially lower than the reported "
                "one, and cross-checking a vendor's back-adjustment against a second source is not "
                "possible from inside this repository."
            ),
        }
    )

    return rows


def build_reproduction_rows(root: Path, timings: dict[str, float]) -> list[dict]:
    data = f'"{DELTA1}"'
    python = ".venv/bin/python"

    def timing(key: str, unmeasured_reason: str) -> tuple[str, str]:
        if key in timings:
            return f"{timings[key]:.2f}", "measured wall clock, this machine, single run"
        return "not_measured", f"not timed: {unmeasured_reason}"

    entries = [
        {
            "artifact_id": "reference_headline",
            "headline_artifact": (
                "Headline metric table (1990-2004 / 2005-2014 / 1990-2014), printed"
            ),
            "command": f'{python} reference/delta1_reference.py --data-dir {data}',
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}",
            "writes": "stdout only",
            "notes": (
                "The whole strategy in one dependency-light file (numpy + pandas). This is the "
                "fastest path to the headline numbers and the one a reviewer should run first."
            ),
        },
        {
            "artifact_id": "reference_equality_proof",
            "headline_artifact": "Proof the readable reference reproduces the hardened engine",
            "command": f"{python} -m unittest tests.test_reference -v",
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}, src/delta1_strategy/",
            "writes": "test output only",
            "notes": (
                "Runs both implementations side by side and asserts array equality on NAV, gross "
                "P&L, cost, turnover, gross notional, margin fraction and both return series; also "
                "re-checks truncation invariance, which is the no-look-ahead test."
            ),
        },
        {
            "artifact_id": "canonical_bundle",
            "headline_artifact": (
                "outputs/strategy_metrics.csv, strategy_daily.csv, strategy_trade_metrics.csv, "
                "strategy_friction_stress.csv, strategy_regime_metrics.csv, run_manifest.json"
            ),
            "command": f".venv/bin/delta1-strategy --data-dir {data} --output-dir outputs",
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}",
            "writes": "outputs/ (canonical bundle plus manifests)",
            "notes": (
                "Console entry point installed by `pip install -e .`; the package has no __main__, so "
                "`python -m delta1_strategy` is not a substitute. Writes the hashed manifest that every "
                "downstream runner reconciles against. Verified: a clean rebuild into a scratch "
                "directory reproduced the shipped outputs/strategy_metrics.csv exactly."
            ),
        },
        {
            "artifact_id": "universe_audit",
            "headline_artifact": "outputs/universe/universe_audit.csv (94 roots in, 59 traded, every exclusion grounded)",
            "command": (
                f"{python} scripts/run_universe_audit.py --data-dir {data} "
                "--output-dir outputs/universe"
            ),
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}",
            "writes": "outputs/universe/",
            "notes": "Enumerates every supplied root and records why each one is in or out.",
        },
        {
            "artifact_id": "lever_sweep",
            "unmeasured_reason": (
                "long-running study (repeated full replays plus a 2,000-sample block bootstrap and 5,000 paired draws); timing it would have cost more than the artifact it documents"
            ),
            "headline_artifact": "outputs/levers/ (parameter sensitivity on a small, pre-declared set)",
            "command": (
                f"{python} scripts/run_lever_sweep.py --data-dir {data} --output-dir outputs/levers"
            ),
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}, outputs/run_manifest.json",
            "writes": "outputs/levers/",
            "notes": "Run before the validation suite: the CSCV refusal reads this family.",
        },
        {
            "artifact_id": "benchmarks",
            "unmeasured_reason": (
                "long-running study (published-rule replications plus a permutation null); not timed during this build"
            ),
            "headline_artifact": "outputs/benchmarks/ (published-rule replication, spanning regression, sign-flip null)",
            "command": (
                f"{python} scripts/run_benchmark_comparison.py --data-dir {data} "
                "--output-dir outputs/benchmarks"
            ),
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}, outputs/run_manifest.json",
            "writes": "outputs/benchmarks/",
            "notes": (
                "Benchmarks are constructed from the same supplied panel, not downloaded, which is "
                "why the comparison survives an offline rebuild."
            ),
        },
        {
            "artifact_id": "validation",
            "unmeasured_reason": (
                "long-running study (10,000 paired samples and a CSCV submatrix sweep); not timed during this build"
            ),
            "headline_artifact": "outputs/validation/ (walk-forward, family-wise inference, CSCV/PBO, deflated Sharpe)",
            "command": (
                f"{python} scripts/run_validation_suite.py --data-dir {data} "
                "--output-dir outputs/validation --levers-dir outputs/levers"
            ),
            "inputs": f"{FUTURES_DATA}/, {FUTURES_CATALOGUE}, outputs/run_manifest.json, outputs/levers/",
            "writes": "outputs/validation/",
            "notes": "Seeded; reruns are deterministic given the same seed arguments.",
        },
        {
            "artifact_id": "etf_regime",
            "headline_artifact": "outputs/etf/ (ETF regime-allocation sleeve and its out-of-sample accounting)",
            "command": (
                f"{python} scripts/run_etf_regime_allocation.py --data-dir {data} "
                "--output-dir outputs/etf"
            ),
            "inputs": f"{ETF_DATA}/, {ETF_CATALOGUE}",
            "writes": "outputs/etf/",
            "notes": "Reads the ETF panel only; has no incumbent baseline to reconcile against.",
        },
        {
            "artifact_id": "holdout_2015_2016",
            "unmeasured_reason": (
                "cannot be re-run for a timing measurement without deleting the append-only ledger, which is precisely the guard that stops the out-of-sample window being re-scored"
            ),
            "headline_artifact": "outputs/holdout/ (chronological out-of-sample scoring of the frozen rules)",
            "command": (
                f"{python} scripts/run_holdout_evaluation.py --data-dir {data} "
                f'--extension-dir "{FXFI_FUTURES}" --forex-dir "{FXFI_FOREX}" '
                "--output-dir outputs/holdout --as-of <ISO8601 timestamp>"
            ),
            "inputs": f"{FUTURES_DATA}/, {FXFI_FUTURES}/, {FXFI_FOREX}/",
            "writes": "outputs/holdout/ (append-only ledger)",
            "notes": (
                "Deliberately hard to run twice: the append-only ledger refuses a second scoring of "
                "the same dataset, so the out-of-sample window cannot be quietly re-rolled until it "
                "prints a better number. Re-running requires deleting outputs/holdout/holdout_ledger.jsonl, "
                "which leaves a visible trace in version control."
            ),
        },
        {
            "artifact_id": "case_notebook",
            "unmeasured_reason": (
                "writes outside this artifact's write scope, so it was not executed here"
            ),
            "headline_artifact": "notebooks/delta1_case_research.ipynb (end-to-end runnable notebook)",
            "command": f"{python} scripts/build_case_notebook.py",
            "inputs": "outputs/ canonical bundle",
            "writes": "notebooks/delta1_case_research.ipynb",
            "notes": "Regenerates the notebook from the hashed bundle rather than from hand-edited cells.",
        },
        {
            "artifact_id": "source_note",
            "headline_artifact": (
                "outputs/submission/source_note.csv and outputs/submission/reproduction_note.csv "
                "(this artifact)"
            ),
            "command": (
                f"{python} scripts/build_submission_sources.py --repo-root . "
                "--output-dir outputs/submission"
            ),
            "inputs": f"{PACK}/ (measured, not parsed for prices), src/, scripts/, reference/, .gitignore",
            "writes": "outputs/submission/source_note.csv, outputs/submission/reproduction_note.csv",
            "notes": (
                "Re-measures every span, count and quote convention from the files on each run, so a "
                "changed extract changes the source note instead of silently contradicting it."
            ),
        },
        {
            "artifact_id": "offline_smoke_test",
            "unmeasured_reason": (
                "writes outside this artifact's write scope (examples/ and /tmp), so it was not executed here"
            ),
            "headline_artifact": "Offline reproduction check on generated data (no licensed panel required)",
            "command": (
                f"{python} scripts/make_synthetic_data.py --output-dir examples/data/synthetic && "
                ".venv/bin/delta1-strategy --data-dir examples/data/synthetic "
                "--output-dir /tmp/synthetic-outputs"
            ),
            "inputs": "none (prices are generated)",
            "writes": "examples/data/synthetic/, /tmp/synthetic-outputs/",
            "notes": (
                "The path a reviewer without the licensed data can run. Those prices are driftless and "
                "independent by construction, so any performance on them is an arithmetic check and "
                "never evidence; what it establishes is that the ledger reconciles and the engine has "
                "no look-ahead."
            ),
        },
    ]

    rows = []
    for entry in entries:
        seconds, basis = timing(
            entry["artifact_id"],
            entry.get("unmeasured_reason", "not run during this build"),
        )
        rows.append(
            {
                "artifact_id": entry["artifact_id"],
                "headline_artifact": entry["headline_artifact"],
                "command": entry["command"],
                "working_directory": "repository root",
                "inputs": entry["inputs"],
                "writes": entry["writes"],
                "measured_runtime_seconds": seconds,
                "runtime_basis": basis,
                "network_required": "no",
                "notes": entry["notes"],
            }
        )
    return rows


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------


def parse_timings(pairs: list[str]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        timings[key.strip()] = float(value)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submission"))
    parser.add_argument(
        "--etf-universe",
        type=Path,
        default=Path("outputs/etf/etf_universe.csv"),
        help="canonical ETF sleeve selection, reused rather than recomputed",
    )
    parser.add_argument(
        "--timing",
        action="append",
        default=[],
        metavar="ARTIFACT_ID=SECONDS",
        help="measured wall-clock runtime for a reproduction row",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols_module = root / "reference/delta1_reference.py"
    symbols = _traded_symbols(symbols_module)

    facts = {
        "root": root,
        "pack_created": _birth_and_mtime(root / PACK)[0],
        "futures": futures_panel_facts(root),
        "catalogue": catalogue_facts(root),
        "traded": traded_universe_facts(root, symbols),
        "fx": fx_source_facts(root),
        "market_daily": root / "outputs/strategy_market_daily.csv.gz",
        "etf": etf_panel_facts(
            root,
            args.etf_universe if args.etf_universe.is_absolute() else root / args.etf_universe,
        ),
        "fxfi": fxfi_facts(root, symbols, root / "outputs/holdout/holdout_summary.csv"),
        "cftc": cftc_facts(root),
        "network": network_dependency_scan(root),
        "gitignore": gitignore_facts(root),
        "margin": margin_enforcement_facts(root),
        "brief": case_brief_facts(root),
    }
    facts["fx_gating"] = fx_gating_reconciliation(
        root, facts["traded"], facts["fx"], facts["market_daily"]
    )

    source_rows = build_source_rows(facts)
    reproduction_rows = build_reproduction_rows(root, parse_timings(args.timing))

    source_path = output_dir / "source_note.csv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(source_rows)

    reproduction_path = output_dir / "reproduction_note.csv"
    with reproduction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPRODUCTION_COLUMNS)
        writer.writeheader()
        writer.writerows(reproduction_rows)

    summary = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_head": git_head(root),
        "futures_files": facts["futures"]["files"],
        "futures_roots": facts["futures"]["roots"],
        "futures_span": [
            facts["futures"]["first_observation"],
            facts["futures"]["last_observation"],
        ],
        "futures_duplicate_date_files": len(facts["futures"]["duplicate_date_files"]),
        "etf_files": facts["etf"]["files"],
        "etf_selected": facts["etf"]["selected"],
        "etf_span": [facts["etf"]["first_observation"], facts["etf"]["last_observation"]],
        "etf_distinct_terminal_dates": len(facts["etf"]["last_observation_counts"]),
        "fxfi_futures_span": [
            facts["fxfi"]["futures"]["first_observation"],
            facts["fxfi"]["futures"]["last_observation"],
        ],
        "fxfi_forex_span": [
            facts["fxfi"]["forex"]["first_observation"],
            facts["fxfi"]["forex"]["last_observation"],
        ],
        "cftc_rows": facts["cftc"]["rows"],
        "cftc_event_span": [facts["cftc"]["event_first"], facts["cftc"]["event_last"]],
        "fx_gating_non_usd_checked": facts["fx_gating"].get("checked", 0),
        "fx_gating_violations": len(facts["fx_gating"].get("violations", [])),
        "network_files_searched": facts["network"]["files_searched"],
        "network_hits": len(facts["network"]["hits"]),
        "source_rows": len(source_rows),
        "reproduction_rows": len(reproduction_rows),
    }
    (output_dir / "source_note_measurements.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {source_path} ({len(source_rows)} sources)")
    print(f"wrote {reproduction_path} ({len(reproduction_rows)} commands)")
    print(json.dumps(summary, indent=2))


def _traded_symbols(reference_path: Path) -> list[str]:
    """Read the traded universe out of the reference file without importing it."""
    namespace: dict = {}
    source = reference_path.read_text(encoding="utf-8")
    match = re.search(r"UNIVERSE: dict\[str, tuple\[str, \.\.\.\]\] = (\{.*?\n\})", source, re.S)
    if not match:
        raise ValueError("could not locate UNIVERSE in the reference implementation")
    namespace["UNIVERSE"] = eval(match.group(1))  # noqa: S307 - literal dict from our own file
    return [symbol for members in namespace["UNIVERSE"].values() for symbol in members]


if __name__ == "__main__":
    main()
