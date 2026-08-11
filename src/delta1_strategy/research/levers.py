"""Config-gated lever comparison for the DELTA1 research engine.

This is the structural sibling of ``friction``: it reruns the full 1990-2014
history under a family of configurations and reports them side by side.  The
difference is intent.  ``friction`` varies assumptions the engine cannot
control (costs, delays, capacity) to show the result survives them.  This
module varies choices the engine *does* control, to measure what changing them
would do.

Every row is reused history.  No row carries a verdict, and the module emits no
ranking, because selecting a configuration on these numbers is the
retrospective fitting that ``docs/research-methodology.md`` prohibits.  The
comparison is evidence for a committee, not a decision.

Two mechanical safeguards are load-bearing.

``run_backtest`` substitutes permissive defaults for omitted input frames: with
``delivery_months`` absent, no roll is ever detected and the strategy pays no
roll costs at all, which by itself moves the reported result by roughly +0.8
percentage points of CAGR and +0.09 Sharpe.  That is close enough to the
promotion gate to manufacture a passing lever out of a wiring mistake, so
``run_lever_comparison`` verifies the baseline reproduces the frozen run
manifest's daily fingerprint before any variant is allowed to run.

A variant that breaks a ledger identity must not abort the sweep, and it must
not silently disappear either.  The harness reconciles through the non-raising
``ledger_reconciliation_report`` and records which identity failed, so a broken
lever is debuggable from the report instead of from a traceback.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RETROSPECTIVE_START = "1990-01-01"
RETROSPECTIVE_END = "2014-12-31"
VALIDATION_STATUS = "retrospective_lever_sensitivity"

RUN_COMPLETED = "completed"
RUN_BLOCKED = "blocked"
RUN_ERRORED = "errored"

_FRAME_KEYS = (
    "prices",
    "observed_opens",
    "observed_closes",
    "unadjusted",
    "metadata",
    "volumes",
    "delivery_months",
    "fx_rates",
)

# Column names deliberately avoid the substrings "target", "pass" and "fail":
# the repository's report tests reject them so that a descriptive artifact can
# never be mistaken for a promotion decision.  In particular the engine's
# ``target_portfolio_limit_scale`` is surfaced here as a binding-day count.
LEVER_REPORT_COLUMNS = [
    "variant",
    "lever",
    "evaluation_mode",
    "assumption",
    "validation_status",
    "run_status",
    "run_detail",
    "ledger_reconciliation",
    "start",
    "end",
    "years",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "hac_sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "max_drawdown_duration_sessions",
    "episode_profit_factor_usd",
    "episode_expectancy_bps_nav",
    "annual_cost_drag",
    "annual_fixed_cost_drag",
    "annual_impact_cost_drag",
    "annual_rebalance_turnover_contracts",
    "annual_roll_turnover_contracts",
    "roll_share_of_turnover",
    "average_risk_scalar",
    "average_markets_held",
    "peak_gross_notional_multiple",
    "peak_static_margin_fraction",
    "portfolio_limit_binding_decision_days",
    "peak_order_participation",
    "peak_rebalance_participation",
    "p999_rebalance_participation",
    "delta_cagr",
    "delta_sharpe",
    "delta_annualized_volatility",
    "delta_max_drawdown",
    "delta_annual_cost_drag",
    "delta_peak_gross_notional_multiple",
]

_METRIC_SOURCES: tuple[tuple[str, str], ...] = (
    ("start", "Start"),
    ("end", "End"),
    ("years", "Years"),
    ("cagr", "CAGR"),
    ("annualized_volatility", "Annualized volatility"),
    ("sharpe", "Naive daily Sharpe (sqrt252, rf=0)"),
    ("hac_sharpe", "HAC Sharpe (21 lags, rf=0)"),
    ("sortino", "Sortino (rf=0)"),
    ("calmar", "Calmar"),
    ("max_drawdown", "Max drawdown"),
    ("max_drawdown_duration_sessions", "Max drawdown duration (sessions)"),
    ("episode_profit_factor_usd", "Trade profit factor (USD)"),
    ("episode_expectancy_bps_nav", "Trade expectancy (bps NAV)"),
    ("annual_cost_drag", "Annual cost drag"),
    ("annual_fixed_cost_drag", "Annual fixed-cost drag"),
    ("annual_impact_cost_drag", "Annual impact-cost drag"),
    ("average_risk_scalar", "Average risk scalar"),
    ("average_markets_held", "Average markets held"),
    ("peak_gross_notional_multiple", "Peak gross notional multiple"),
    ("peak_static_margin_fraction", "Peak static margin fraction"),
    (
        "portfolio_limit_binding_decision_days",
        "Portfolio-limit binding decision days",
    ),
    ("peak_order_participation", "Peak order participation"),
    ("peak_rebalance_participation", "Peak rebalance participation"),
)

_DELTA_SOURCES: tuple[tuple[str, str], ...] = (
    ("delta_cagr", "cagr"),
    ("delta_sharpe", "sharpe"),
    ("delta_annualized_volatility", "annualized_volatility"),
    ("delta_max_drawdown", "max_drawdown"),
    ("delta_annual_cost_drag", "annual_cost_drag"),
    ("delta_peak_gross_notional_multiple", "peak_gross_notional_multiple"),
)

_FRAME_CACHE: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}


@dataclass(frozen=True)
class LeverVariant:
    """One configuration change, with its economic assumption stated."""

    name: str
    lever: str
    evaluation_mode: str
    assumption: str
    overrides: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for field in ("name", "lever", "evaluation_mode", "assumption"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty string")
        seen = [key for key, _ in self.overrides]
        if len(seen) != len(set(seen)):
            raise ValueError("override keys must be unique")

    def apply(self, config: Any) -> Any:
        return replace(config, **dict(self.overrides))

    def is_supported(self, config: Any) -> bool:
        """Whether this variant's fields exist on the supplied config.

        Lets the harness land before the engine changes it measures, and lets a
        sweep skip variants for levers that are not implemented yet rather than
        reporting them as errors.
        """

        return all(hasattr(config, key) for key, _ in self.overrides)


def load_shared_frames(
    data_dir: Path,
    *,
    ffill_limit: int = 10,
    symbols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every ``run_backtest`` input frame once, for reuse across variants.

    Sharing one set of frames costs about a second per sweep, but more
    importantly it guarantees every variant sees byte-identical inputs, so a
    difference in the report can only come from the configuration.
    """

    from .strategy import (
        load_delivery_months,
        load_fx_rates,
        load_metadata,
        load_observed_prices,
        load_prices,
        load_unadjusted_prices,
        load_volumes,
    )

    key = (str(Path(data_dir).resolve()), int(ffill_limit))
    cached = _FRAME_CACHE.get(key)
    if cached is not None and symbols is None:
        return cached

    prices = load_prices(data_dir, symbols=symbols, ffill_limit=ffill_limit)
    frames = {
        "prices": prices,
        "observed_opens": load_observed_prices(data_dir, "Open", symbols=symbols),
        "observed_closes": load_observed_prices(data_dir, "Close", symbols=symbols),
        "unadjusted": load_unadjusted_prices(
            data_dir, symbols=symbols, ffill_limit=ffill_limit
        ),
        "metadata": load_metadata(data_dir, symbols=symbols),
        "volumes": load_volumes(data_dir, symbols=symbols),
        "delivery_months": load_delivery_months(data_dir, symbols=symbols),
        "fx_rates": load_fx_rates(data_dir, prices.index, ffill_limit),
    }
    missing = [key_ for key_ in _FRAME_KEYS if key_ not in frames]
    if missing:
        raise RuntimeError(f"shared frames are incomplete: {missing}")
    if symbols is None:
        _FRAME_CACHE[key] = frames
    return frames


def _ledger_status(result: Any) -> tuple[str, str]:
    """Reconcile without raising, naming any identity that failed.

    ``validate_backtest_result`` raises, which would abort a sweep on the first
    broken variant and discard the evidence needed to diagnose it.  The report
    behind it is non-raising and names each check, so the harness uses that and
    keeps going.
    """

    from .strategy import ledger_reconciliation_report

    daily = getattr(result, "daily", None)
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return RUN_BLOCKED, "daily ledger is missing or empty"
    try:
        report = ledger_reconciliation_report(result)
    except Exception as error:  # pragma: no cover - defensive
        return RUN_BLOCKED, f"reconciliation raised {type(error).__name__}: {error}"
    broken = [
        str(row["check"])
        for _, row in report.iterrows()
        if str(row["status"]) != "PASS"
    ]
    numeric = daily.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        broken.append("finite_daily_columns")
    if "nav" in daily and (daily["nav"] <= 0).any():
        broken.append("positive_nav")
    if broken:
        return RUN_BLOCKED, ", ".join(broken)
    return RUN_COMPLETED, ""


def _turnover_diagnostics(result: Any, start: str, end: str) -> dict[str, float]:
    """Turnover and participation detail the standard metrics do not expose.

    Computed after the fact from existing columns, so nothing is added to the
    daily ledger and its fingerprint is untouched.
    """

    empty = {
        "annual_rebalance_turnover_contracts": float("nan"),
        "annual_roll_turnover_contracts": float("nan"),
        "roll_share_of_turnover": float("nan"),
        "p999_rebalance_participation": float("nan"),
    }
    daily = getattr(result, "daily", None)
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return empty
    window = daily.loc[start:end]
    if window.empty:
        return empty
    years = max((window.index[-1] - window.index[0]).days / 365.2425, 1e-9)
    rebalance = float(window.get("rebalance_contract_turnover", pd.Series(dtype=float)).sum())
    roll = float(
        window.get("roll_contract_turnover_increment", pd.Series(dtype=float)).sum()
    )
    total = rebalance + roll
    participation = window.get("max_rebalance_participation")
    return {
        "annual_rebalance_turnover_contracts": rebalance / years,
        "annual_roll_turnover_contracts": roll / years,
        "roll_share_of_turnover": (roll / total) if total > 0 else float("nan"),
        "p999_rebalance_participation": (
            float(np.nanpercentile(participation.to_numpy(dtype=float), 99.9))
            if participation is not None and not participation.empty
            else float("nan")
        ),
    }


def _coerce_metrics(run_output: Any, start: str, end: str, annualization: int) -> pd.Series:
    """Accept a result, a metrics Series, a Mapping, or a ``(result, metrics)`` pair.

    The same injection seam ``friction`` uses, so the harness can be unit
    tested without running a backtest.
    """

    from .strategy import performance_metrics

    if isinstance(run_output, tuple) and len(run_output) == 2:
        run_output = run_output[1]
    if isinstance(run_output, pd.Series):
        return run_output
    if isinstance(run_output, Mapping):
        return pd.Series(dict(run_output))
    return performance_metrics(run_output, start, end, annualization)


def _metric_row(metrics: pd.Series) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column, name in _METRIC_SOURCES:
        value = metrics.get(name, np.nan)
        row[column] = value
    return row


def _nan_metric_row() -> dict[str, Any]:
    return {column: np.nan for column, _ in _METRIC_SOURCES}


def assert_baseline_fingerprint(result: Any, manifest_path: Path) -> str:
    """Verify the baseline reproduces the frozen run manifest's daily ledger.

    This is the guard against partially wired input frames.  Omitting
    ``delivery_months`` alone removes every roll cost and flatters the result by
    roughly the size of the promotion gate, so a sweep that cannot reproduce the
    frozen fingerprint is not measuring what it claims to measure.
    """

    from ..controls.production import daily_frame_fingerprint

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = manifest.get("daily_fingerprint_sha256")
    if not expected:
        raise ValueError(f"{manifest_path} has no daily_fingerprint_sha256")
    observed = daily_frame_fingerprint(result.daily)
    if observed != expected:
        raise ValueError(
            "baseline daily ledger does not reproduce the frozen run manifest: "
            f"expected {expected}, observed {observed}. The most common cause "
            "is an incompletely injected set of input frames, which silently "
            "disables roll costs."
        )
    return observed


def run_lever_comparison(
    config: Any,
    *,
    variants: Sequence[LeverVariant],
    run_fn: Callable[[Any], Any] | None = None,
    frames: Mapping[str, pd.DataFrame] | None = None,
    start: str = RETROSPECTIVE_START,
    end: str = RETROSPECTIVE_END,
    manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every variant and report it beside the baseline.

    Returns ``(comparison, monthly_net_returns)``.  The monthly paths are not
    optional detail: family-wise inference needs the losing variants' return
    paths as much as the winners', and they cannot be reconstructed from
    summary statistics later.
    """

    from .strategy import run_backtest

    annualization = getattr(config, "annualization", 252)
    if not variants:
        raise ValueError("at least one variant is required")
    if variants[0].evaluation_mode != "baseline":
        raise ValueError("the first variant must be the baseline")

    frame_kwargs = dict(frames) if frames is not None else {}
    if frame_kwargs:
        missing = [key for key in _FRAME_KEYS if key not in frame_kwargs]
        if missing:
            raise ValueError(
                "shared frames must supply every input; missing: "
                f"{missing}. Partial injection silently disables roll costs."
            )

    def default_runner(variant_config: Any) -> Any:
        return run_backtest(variant_config, **frame_kwargs)

    runner = default_runner if run_fn is None else run_fn

    cache: dict[Any, tuple[pd.Series, dict[str, Any], str, str, pd.Series]] = {}
    rows: list[dict[str, Any]] = []
    monthly: dict[str, pd.Series] = {}

    for position, variant in enumerate(variants):
        if not variant.is_supported(config):
            unsupported = [key for key, _ in variant.overrides if not hasattr(config, key)]
            rows.append(
                {
                    "variant": variant.name,
                    "lever": variant.lever,
                    "evaluation_mode": variant.evaluation_mode,
                    "assumption": variant.assumption,
                    "validation_status": VALIDATION_STATUS,
                    "run_status": RUN_ERRORED,
                    "run_detail": f"config does not define {unsupported}",
                    "ledger_reconciliation": "",
                    **_nan_metric_row(),
                    **_turnover_diagnostics(None, start, end),
                }
            )
            continue

        variant_config = variant.apply(config)
        cached = cache.get(variant_config)
        if cached is not None:
            metrics, extra, status, detail, path = cached
        else:
            try:
                output = runner(variant_config)
            except Exception as error:
                rows.append(
                    {
                        "variant": variant.name,
                        "lever": variant.lever,
                        "evaluation_mode": variant.evaluation_mode,
                        "assumption": variant.assumption,
                        "validation_status": VALIDATION_STATUS,
                        "run_status": RUN_ERRORED,
                        "run_detail": f"{type(error).__name__}: {error}"[:240],
                        "ledger_reconciliation": "",
                        **_nan_metric_row(),
                        **_turnover_diagnostics(None, start, end),
                    }
                )
                continue

            result = output[0] if isinstance(output, tuple) else output
            if position == 0 and manifest_path is not None:
                assert_baseline_fingerprint(result, manifest_path)
            metrics = _coerce_metrics(output, start, end, annualization)
            extra = _turnover_diagnostics(result, start, end)
            status, detail = _ledger_status(result)
            daily = getattr(result, "daily", None)
            if isinstance(daily, pd.DataFrame) and "net_return" in daily:
                returns = daily.loc[start:end, "net_return"].dropna()
                path = (1.0 + returns).resample("ME").prod() - 1.0
            else:
                path = pd.Series(dtype=float)
            cache[variant_config] = (metrics, extra, status, detail, path)

        rows.append(
            {
                "variant": variant.name,
                "lever": variant.lever,
                "evaluation_mode": variant.evaluation_mode,
                "assumption": variant.assumption,
                "validation_status": VALIDATION_STATUS,
                "run_status": status,
                "run_detail": detail,
                "ledger_reconciliation": "reconciled" if status == RUN_COMPLETED else detail,
                **_metric_row(metrics),
                **extra,
            }
        )
        if not path.empty:
            monthly[variant.name] = path

    comparison = pd.DataFrame(rows)
    baseline = comparison.iloc[0]
    for delta_column, source in _DELTA_SOURCES:
        comparison[delta_column] = pd.to_numeric(
            comparison[source], errors="coerce"
        ) - pd.to_numeric(pd.Series([baseline[source]] * len(comparison)), errors="coerce").to_numpy()

    for column in LEVER_REPORT_COLUMNS:
        if column not in comparison:
            comparison[column] = np.nan
    comparison = comparison.loc[:, LEVER_REPORT_COLUMNS]

    monthly_frame = (
        pd.DataFrame(monthly) if monthly else pd.DataFrame(index=pd.DatetimeIndex([]))
    )
    return comparison, monthly_frame


__all__ = [
    "LEVER_REPORT_COLUMNS",
    "LeverVariant",
    "RETROSPECTIVE_END",
    "RETROSPECTIVE_START",
    "RUN_BLOCKED",
    "RUN_COMPLETED",
    "RUN_ERRORED",
    "VALIDATION_STATUS",
    "assert_baseline_fingerprint",
    "load_shared_frames",
    "run_lever_comparison",
]
