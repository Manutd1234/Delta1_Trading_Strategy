# Architecture

## Design objective

The code is structured like a small research-to-production system: raw data, forecast, risk, execution, accounting, analytics, and controls are separate layers. This makes timing errors and silent assumptions easier to review.

```text
catalogue + daily CCB files
          |
          v
  load + validate data
          |
          v
  point-in-time forecast -------- diagnostics
          |
          v
  contract risk sizing
          |
          v
  portfolio volatility target
          |
          v
  no-trade execution state machine
          |
          v
  next-business-day held positions
          |
          v
  contract P&L - spread/commission
          |
          +------> metrics / stress tests / invariants / plots
```

The optional ML branch adds a point-in-time feature panel before the forecast layer:

```text
daily futures + lagged FRED macro
              |
              v
 monthly instrument/market features
              |
              v
annual nested walk-forward model selection
              |
              v
bounded probability signal + 12m trend prior
              |
              +------> existing risk / execution / accounting layers
```

## Layer ownership

### Data layer

`load_metadata()` and `load_prices()` own schema checks, contract metadata, date sorting, duplicate removal, and calendar alignment. No downstream function reads CSV files directly.

### Forecast layer

`trend_signal()` implements the baseline. `build_institutional_forecast()` returns both its bounded forecast and the components needed to explain it. Forecast code never performs portfolio construction.

`build_feature_panel()` creates monthly point-in-time observations. `walk_forward_predict()` owns the annual fit/validation/test boundary. `predictions_to_signal_matrix()` is the only bridge from probabilities to the portfolio engine.

### Risk layer

`_base_target_positions()` maps forecasts to contracts per dollar using annualized dollar price-change risk. Class balancing avoids giving commodities more risk merely because the universe contains more commodity contracts.

`_portfolio_leverage()` is a second-pass portfolio volatility overlay. Its month-end value becomes active on the next business day.

### Execution layer

`apply_no_trade_buffer()` is a state machine over month-end desired positions. It is deterministic, path-dependent, and separately unit-tested. It is deliberately simple because daily data cannot support a fill simulator.

### Accounting layer

`_gross_returns()` uses held contracts, daily price changes, and catalogue point values. `_cost_series()` applies contract-specific tick economics and commissions to changes in held positions.

### Controls and analytics

`strategy_invariants()` returns named booleans suitable for CI logs or a monitoring surface. Performance and stress outputs are CSVs rather than notebook-only objects.

## Timing convention

1. Month-end close at `t` is observed.
2. Forecast, volatility, desired contracts, and buffer decision are formed using data through `t`.
3. The new position becomes active on the next business day.
4. P&L at `t+1` uses that held position and the price change from `t` to `t+1`.

The one-row shift in `_held_positions()` encodes this boundary centrally.

## Extension points

- Replace `_CCB` files with a contract-level roll engine behind `load_prices()`.
- Add FX conversion at the accounting layer for non-USD contracts.
- Replace the no-trade band with an optimizer using forecast benefit versus expected cost.
- Add integer contract sizing after fractional research positions are generated.
- Stream invariant results to monitoring without changing the alpha model.
- Replace revised FRED histories with vintage-aware release data behind `load_external_macro()`.
- Accumulate daily DTCC snapshots into a dated feature store for a post-2024 forward experiment.
