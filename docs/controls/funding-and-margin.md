# Cash, collateral, variation-margin and funding control plane

The research NAV is an excess-return ledger. It deliberately does not add cash
collateral yield or claim to simulate funded-account survival. A credible live
deployment requires a separate broker-backed treasury ledger; historical
returns must not be retrofitted using present-day margin or invented funding.

Required point-in-time objects include official settlement marks and USD FX,
broker portfolio-margin quotes, collateral lots with haircuts and effective
intervals, multi-currency account snapshots, and immutable double-entry cash
events. The ledger must attribute strategy P&L, variation margin, collateral
income, funding expense, fees and FX translation separately.

Core invariants are:

- positions, cash, collateral, pending variation margin, fees and broker equity
  reconcile to timestamped source records;
- deposits and withdrawals never count as P&L, and VM and fees are recognized
  exactly once;
- collateral is not double-counted and post-haircut value never exceeds market
  value or its effective eligibility interval;
- missing or stale settlement, FX, margin, collateral or account data blocks
  every risk-increasing order;
- route-time assessment includes open-order reservations and every plausible
  partial-fill state of multi-leg rolls; and
- stronger margin, VM, haircut, concentration or FX stress cannot improve the
  calculated liquidity buffer.

Runtime gates must require a fresh broker portfolio-margin quote, zero current
margin call, positive projected available funds, positive stressed liquidity,
and full statement reconciliation. Exchange and broker margin schedules can
change with volatility and portfolio composition; see CME's
[margin model](https://www.cmegroup.com/solutions/risk-management/performance-bonds-margins/futures-and-options-margin-model.html)
and [margin FAQ](https://www.cmegroup.com/solutions/risk-management/performance-bonds-margins/faq-performance-bonds-margins.html).

`delta1_strategy.controls.treasury` now supplies immutable record types, a
read-only provider protocol, journal-chain checks and fail-closed bundle
validation for this boundary. It does not implement the broker/clearing
provider, post cash, simulate liquidation, or create external evidence. The
live gate must remain BLOCKED until verified broker/clearing data and a
prospective shadow record exist.

The ETF regime-allocation sleeve does not supply any of the objects above and
must not be read as a funded-account record. Its book is fully funded, long-only
and unlevered, and its "cash" position is a short-duration Treasury ETF that
carries duration — an allocation sleeve, not a cash balance, and it can lose
money. The ETF panel contains no risk-free series and no financing series, so
that record is neither excess-of-financing nor funded-with-yield; it is a
total-return path on eleven funds with no interest leg at all, which is a
different and non-comparable convention from the futures ledger's
excess-return NAV. Its costs are a pre-declared assumption rather than a
calibration,
and its artifacts carry those statements as inline limitations. Nothing in
`outputs/etf/` bears on cash, variation margin, collateral, funding or
liquidation, and it cannot be filed against `cash_margin_funding_controls`.
