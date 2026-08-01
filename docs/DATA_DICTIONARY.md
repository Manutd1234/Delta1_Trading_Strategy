# Data dictionary

The market data are intentionally not committed to this repository. Set `DELTA1_DATA_DIR` to the supplied directory.

## Expected layout

```text
Delta1/
├── CATALOGUE_Delta1_Futures.csv
└── Futures Data/
    ├── &ES_CCB.csv
    ├── &ZN_CCB.csv
    └── ...
```

## Catalogue fields used

| Field | Meaning | Use |
|---|---|---|
| `symbol` | Norgate continuous symbol | Select `_CCB` rows and map files. |
| `securityname` | Human-readable contract name | Coverage and audit output. |
| `currency` | Contract P&L currency | Enforce the USD-only universe. |
| `Class` | Vendor asset classification | Reference only; strategy uses an explicit mapping. |
| `tick_size` | Minimum quoted price increment in vendor units | Convert the half-spread assumption to dollars. |
| `point_value` | Currency P&L per one-point move | Convert price changes to contract P&L. |

## Price-file fields used

| Field | Type | Use |
|---|---|---|
| `Date` | ISO date | Sorted unique time index. |
| `Close` | Floating-point back-adjusted close | Forecasts, price-change volatility, and P&L. |

The files also contain OHLC, volume, delivery month, and open interest, but the current model does not consume them.

## Cleaning policy

- duplicate dates retain the last row;
- all series are sorted ascending;
- a common business-day calendar is created;
- at most five missing business days are forward-filled for exchange holidays;
- longer gaps remain missing;
- calculations require valid lookback and volatility history;
- percentage returns are never computed from additive back-adjusted price levels.

## Universe policy

Only explicit, USD-denominated contracts are selected. The selection avoids hidden FX translation but does not prove point-in-time liquidity or eliminate survivorship bias.

## External macro file

Run `python download_external_data.py` to create the ignored local file `data/external/fred_macro.csv` and its SHA-256 source manifest. The feature set uses:

| FRED series | Interpretation | Backtest treatment |
|---|---|---|
| `VIXCLS` | CBOE VIX close | `log1p` level and 21-business-day change |
| `T10Y2Y` | 10-year less 2-year Treasury rate | level and 21-business-day change |
| `BAA10Y` | Moody's Baa yield less 10-year Treasury | level and 21-business-day change |

All three series are aligned to the futures calendar, forward-filled for no more than ten business days, and lagged one business day. Historical FRED files may contain revisions; a production study should use vintage/release-time data.

## Data Engineer DTCC snapshot

`data_engineer_features.py` validates and aggregates the supplied `CFTC_CUMULATIVE_FOREX_2024_04_08.csv`. It reads dissemination ID, action/event type, event/effective/expiration timestamps, two-leg notionals and currencies, block/prime/cleared flags, platform, and UPI underlier.

The derived audit and pair-level summary are committed under `outputs/`. The snapshot is not an input to 2005–2014 model training because it is a 2024 cumulative publication slice; backfilling it would introduce look-ahead bias.
