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
