# Strategy specification

## 1. Core forecast

For back-adjusted close `P(i,t)` and slow horizon `H=252`:

```text
core(i,t) = sign(P(i,t) - P(i,t-H))
```

The forecast is bounded to `[-1, 1]`. A 63-day direction is calculated as a diagnostic and can reduce conviction when `disagreement_scale < 1`; the default is `1`, so the faster signal cannot alter the live forecast.

## 2. Volatility-shock multiplier

Daily price-change volatility is estimated with 20-day and 120-day exponentially weighted standard deviations:

```text
ratio(i,t) = fast_vol(i,t) / slow_vol(i,t)
```

The multiplier equals 1 below a ratio of 1.35, declines linearly, and reaches a floor of 0.75 at a ratio of 2.0. The taper avoids a discontinuous exposure jump.

```text
forecast(i,t) = core(i,t) * shock_multiplier(i,t)
```

## 3. Instrument risk sizing

Annualized dollar risk per contract is:

```text
annual_dollar_vol(i,t)
    = EWM_STD(delta_price(i), span=60)
    * point_value(i)
    * sqrt(252)
```

Each of four asset classes receives equal ex-ante risk. Within a class, currently available contracts split risk equally. The fractional contract target per dollar is the class risk budget divided by annual dollar volatility, multiplied by the forecast.

## 4. Portfolio volatility target

The unlevered forecast portfolio is marked historically. A lagged 63-day realized volatility estimate produces:

```text
leverage(t) = clip(10% / realized_portfolio_vol(t), 0.25, 2.0)
```

The multiplier is sampled at month-end and applies from the next business day.

## 5. Cost-aware no-trade region

Let `q*` be the desired position and `q_prev` the existing month-end target. Trade only if:

```text
abs(q* - q_prev) > 25% * max(abs(q*), abs(q_prev))
```

Otherwise retain `q_prev`. A sign reversal always clears the threshold. Missing desired positions flatten rather than persist silently.

This is not an optimal execution model. It is a transparent proxy for the principle that small forecast improvements may not pay for spread and fees.

## 6. P&L and costs

```text
gross_return(t)
    = sum_i held_contracts(i,t)
      * delta_price(i,t)
      * point_value(i)

one_way_cost(i)
    = 0.5 * tick_size(i) * point_value(i)
      + USD 2.50

net_return(t) = gross_return(t) - trading_cost(t)
```

Positions are contracts per dollar, so the computed P&L is a portfolio return.

## 7. What is “institutional” here?

The label refers to engineering discipline, not secret alpha:

- every feature is observable before the traded return;
- accounting uses contract economics instead of synthetic percentage returns;
- forecasts and multipliers have hard bounds;
- execution is stateful and cost-aware;
- weak model additions are retained as ablations, not hidden;
- stress scenarios and invariant checks are committed with the result;
- the code states what the data cannot identify.

An actual market maker would additionally require tick data, two-sided quotes, queue models, inventory skew, adverse-selection estimates, exchange-specific fees, latency measurements, and fill reconciliation. None can be inferred from these daily files.
