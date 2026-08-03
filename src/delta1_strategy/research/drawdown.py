"""Causal drawdown-risk overlay prototype for the DELTA1 research ledger.

This module is deliberately standalone and diagnostic.  It scales an already
simulated portfolio uniformly; it does not create executable futures orders,
round contracts, enforce market capacity, or reproduce the main ledger after
the scaling trades.  A favorable result here is therefore a hypothesis for a
full engine integration, not production-readiness evidence.

The controller observes only the overlay NAV available at a session close.  A
decision enters a FIFO delay queue before becoming effective, so future
returns never affect today's exposure.  The default two-session lag matches a
conservative close-data convention: a decision after close *t* fills at close
*t+1* and first affects close-to-close P&L ending at *t+2*.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownOverlayConfig:
    """Parameters for a transparent, uniformly scaled drawdown controller.

    The 7%/11% thresholds intentionally leave a buffer below the independent
    15% live drawdown gate.  They are design choices, not fitted signal
    parameters, and still require prospective validation.
    """

    soft_drawdown_fraction: float = 0.07
    hard_drawdown_fraction: float = 0.11
    minimum_exposure_multiplier: float = 0.25
    max_risk_on_step: float = 0.05
    execution_lag_sessions: int = 2
    overlay_turnover_cost_bps: float = 2.0
    annualization: int = 252

    def __post_init__(self) -> None:
        if not 0 < self.soft_drawdown_fraction < self.hard_drawdown_fraction < 1:
            raise ValueError(
                "drawdown thresholds must satisfy 0 < soft < hard < 1"
            )
        if not 0 <= self.minimum_exposure_multiplier < 1:
            raise ValueError("minimum_exposure_multiplier must be in [0, 1)")
        if not 0 < self.max_risk_on_step <= 1:
            raise ValueError("max_risk_on_step must be in (0, 1]")
        if (
            isinstance(self.execution_lag_sessions, bool)
            or not isinstance(self.execution_lag_sessions, int)
            or self.execution_lag_sessions < 1
        ):
            raise ValueError("execution_lag_sessions must be a positive integer")
        if (
            not np.isfinite(self.overlay_turnover_cost_bps)
            or self.overlay_turnover_cost_bps < 0
        ):
            raise ValueError("overlay_turnover_cost_bps must be finite and nonnegative")
        if (
            isinstance(self.annualization, bool)
            or not isinstance(self.annualization, int)
            or self.annualization <= 0
        ):
            raise ValueError("annualization must be a positive integer")


REQUIRED_DAILY_COLUMNS = (
    "gross_return",
    "cost",
    "gross_notional_multiple",
)


def _desired_multiplier(
    drawdown_fraction: float,
    config: DrawdownOverlayConfig,
) -> float:
    """Map the known drawdown to a piecewise-linear exposure multiplier."""

    if drawdown_fraction <= config.soft_drawdown_fraction:
        return 1.0
    if drawdown_fraction >= config.hard_drawdown_fraction:
        return config.minimum_exposure_multiplier
    progress = (
        drawdown_fraction - config.soft_drawdown_fraction
    ) / (config.hard_drawdown_fraction - config.soft_drawdown_fraction)
    return 1.0 - progress * (1.0 - config.minimum_exposure_multiplier)


def _validated_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    missing = [column for column in REQUIRED_DAILY_COLUMNS if column not in daily]
    if missing:
        raise ValueError(f"daily is missing required columns: {missing}")
    if daily.empty:
        raise ValueError("daily cannot be empty")
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be unique and increasing")

    validated = daily.loc[:, REQUIRED_DAILY_COLUMNS].astype(float).copy()
    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("overlay inputs must all be finite")
    if (validated["gross_notional_multiple"] < 0).any():
        raise ValueError("gross_notional_multiple cannot be negative")
    base_net_return = validated["gross_return"] - validated["cost"]
    if (base_net_return <= -1).any():
        raise ValueError("base net returns must be greater than -100%")
    if (validated["cost"] < 0).any():
        raise ValueError("cost cannot be negative")
    return validated


def simulate_drawdown_overlay(
    daily: pd.DataFrame,
    config: DrawdownOverlayConfig = DrawdownOverlayConfig(),
) -> pd.DataFrame:
    """Apply a lagged drawdown controller to an existing daily research path.

    Base gross returns and base costs are scaled by the effective multiplier.
    Changes in that multiplier incur an additional cost equal to configured
    basis points times the prior session's gross-notional multiple.  This is a
    transparent approximation: actual contract rounding, spreads, impact,
    capacity, roll mechanics, and delayed partial fills require a full ledger
    re-run before the overlay can be considered executable.
    """

    if not isinstance(config, DrawdownOverlayConfig):
        raise TypeError("config must be a DrawdownOverlayConfig")
    source = _validated_daily(daily)

    pending: deque[float] = deque(
        [1.0] * config.execution_lag_sessions,
        maxlen=config.execution_lag_sessions,
    )
    nav = 1.0
    peak_nav = 1.0
    previous_effective = 1.0
    previous_decision = 1.0
    previous_gross_notional = 0.0
    records: list[dict[str, float]] = []

    for row in source.itertuples(index=False):
        prior_nav = nav
        prior_peak = peak_nav
        prior_drawdown = 1.0 - prior_nav / prior_peak
        effective = pending.popleft()

        adjustment_turnover = (
            abs(effective - previous_effective) * previous_gross_notional
        )
        adjustment_cost = (
            adjustment_turnover * config.overlay_turnover_cost_bps / 10_000.0
        )
        scaled_gross_return = effective * float(row.gross_return)
        scaled_base_cost = effective * float(row.cost)
        overlay_net_return = (
            scaled_gross_return - scaled_base_cost - adjustment_cost
        )
        if not np.isfinite(overlay_net_return) or overlay_net_return <= -1:
            raise ValueError("overlay produced an invalid daily return")

        nav = prior_nav * (1.0 + overlay_net_return)
        peak_nav = max(prior_peak, nav)
        observed_drawdown = 1.0 - nav / peak_nav
        desired = _desired_multiplier(observed_drawdown, config)
        if desired > previous_decision:
            decision = min(desired, previous_decision + config.max_risk_on_step)
        else:
            decision = desired
        pending.append(decision)

        records.append(
            {
                "base_gross_return": float(row.gross_return),
                "base_cost_return": float(row.cost),
                "base_net_return": float(row.gross_return - row.cost),
                "prior_overlay_nav": prior_nav,
                "prior_overlay_peak_nav": prior_peak,
                "prior_overlay_drawdown_fraction": prior_drawdown,
                "exposure_multiplier": effective,
                "desired_multiplier_after_close": desired,
                "decision_multiplier_after_close": decision,
                "overlay_adjustment_turnover_multiple": adjustment_turnover,
                "overlay_adjustment_cost_return": adjustment_cost,
                "scaled_base_cost_return": scaled_base_cost,
                "overlay_net_return": overlay_net_return,
                "overlay_nav": nav,
                "overlay_drawdown_fraction": 1.0 - nav / peak_nav,
            }
        )
        previous_effective = effective
        previous_decision = decision
        previous_gross_notional = float(row.gross_notional_multiple)

    return pd.DataFrame(records, index=source.index)


def _path_metrics(
    returns: pd.Series,
    annualization: int,
    *,
    period_start: pd.Timestamp | None = None,
) -> dict[str, float]:
    wealth = (1.0 + returns).cumprod()
    if period_start is None:
        period_start = returns.index[0] - pd.offsets.BDay(1)
    period_start = pd.Timestamp(period_start)
    if period_start > returns.index[0]:
        raise ValueError("period_start cannot follow the first return")
    elapsed_years = (
        (returns.index[-1] - period_start).total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )
    if elapsed_years <= 0:
        cagr = np.nan
    else:
        cagr = float(wealth.iloc[-1] ** (1.0 / elapsed_years) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(annualization))
        if returns.std(ddof=1) > 0
        else np.nan
    )
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    drawdown = wealth / running_peak - 1.0
    return {
        "CAGR": cagr,
        "Annualized volatility": volatility,
        "Naive daily Sharpe": sharpe,
        "Max drawdown": float(drawdown.min()),
    }


def overlay_performance_report(
    overlay: pd.DataFrame,
    config: DrawdownOverlayConfig = DrawdownOverlayConfig(),
    *,
    period_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return directly comparable base and diagnostic-overlay path metrics."""

    required = {
        "base_net_return",
        "overlay_net_return",
        "exposure_multiplier",
        "overlay_adjustment_cost_return",
    }
    missing = sorted(required.difference(overlay.columns))
    if missing:
        raise ValueError(f"overlay is missing required columns: {missing}")
    if overlay.empty:
        raise ValueError("overlay cannot be empty")

    rows = []
    for name, column in (
        ("base", "base_net_return"),
        ("drawdown_overlay_diagnostic", "overlay_net_return"),
    ):
        metrics = _path_metrics(
            overlay[column].astype(float),
            config.annualization,
            period_start=period_start,
        )
        metrics["Path"] = name
        metrics["Average exposure multiplier"] = (
            1.0
            if name == "base"
            else float(overlay["exposure_multiplier"].mean())
        )
        metrics["Annual overlay adjustment-cost drag"] = (
            0.0
            if name == "base"
            else float(
                overlay["overlay_adjustment_cost_return"].mean()
                * config.annualization
            )
        )
        rows.append(metrics)
    columns = [
        "Path",
        "CAGR",
        "Annualized volatility",
        "Naive daily Sharpe",
        "Max drawdown",
        "Average exposure multiplier",
        "Annual overlay adjustment-cost drag",
    ]
    return pd.DataFrame(rows).loc[:, columns]
