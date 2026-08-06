"""Position-size bound sensitivity for the DELTA1 research engine.

The question this module exists to answer is whether tightening a
magnitude bound — the clip on the EWMA portfolio volatility-target multiplier
(``max_risk_scalar`` / ``min_risk_scalar``), the two forecast caps, the
volatility-shock floor, or the gross-notional multiple — reduces drawdown.

Only two of the six are leverage ceilings.  ``max_risk_scalar`` and
``max_gross_notional_multiple`` cap position magnitude directly, so tightening
either far enough lowers realised volatility and takes drawdown and CAGR down
with it in rough proportion.  That is de-levering: it is available for free by
lowering ``target_vol``, it costs return one-for-one, and it is not a risk
improvement.

The other three do not act that way, and the emitted rows say so.
``signal_cap`` is a normalisation divisor rather than a ceiling —
``basis_momentum`` returns ``(raw / scale).clip(-cap, cap) / cap``, so lowering
the cap renormalises every unsaturated cell *upward* and makes the sleeve more
aggressive.  ``shock_floor`` sets both the depth of the volatility-shock taper
and the slope of the ramp above it.  ``risk_managed_cap`` reweights markets
against one another rather than scaling the book.  On 1990-2014 five of the
eight one-at-a-time settings raise realised volatility instead of lowering it.
So the sign of the volatility response is not known a priori for anything but
the two ceilings, and the reader must not assume an as-is row is de-levering
merely because the bound was tightened.

Every bound variant is therefore reported twice.  The ``as_is`` row is the raw
effect of the setting.  The ``risk_matched`` row rescales ``target_vol`` until
realised annualised volatility returns to the incumbent's.  Both rows carry
``annualized_volatility_gap_to_reference``: on an as-is row it is the
volatility the setting moved on its own, and on a matched row it is the
solver's residual, so the match is auditable rather than asserted.

The risk match is NOT a constant rescale of a return series, and the module
does not claim the invariance that would follow if it were.  Changing
``target_vol`` re-runs a discrete-execution engine: integer contract counts,
no-trade buffer crossings, participation clipping and the gross-notional limit
all resolve differently, so the output is a different path rather than a scaled
one.  Measured on 1990-2014, moving the budget by 0.36% — well inside the match
tolerance — produces a path correlated 0.9991 with the incumbent's at 33 basis
points of annualised tracking error.  Every matched loss-shape number is
therefore read against two measured noise floors rather than against an
analytic argument.

The first floor is solver jitter.  ``risk_match_control_budgets`` re-runs the
UNCHANGED incumbent configuration at a ladder of budgets that all land inside
the match tolerance, and ``match_jitter`` reports the spread each loss-shape
statistic shows across that ladder.  On 1990-2014 the incumbent's own maximum
drawdown over annualised volatility ranges from 1.515 to 1.574 across that
band — a spread of 0.059 on a level of 1.535, wider than every matched delta
the risk-scalar axis produces.  ``baseline_risk_matched`` is not this control:
the first-order rescale of an already-matched configuration is exactly the
identity in IEEE arithmetic, so that row is an exact-arithmetic plumbing check
and nothing more.

The second floor is sampling error.  ``loss_shape_resolution`` runs a paired
stationary block bootstrap of the maximum-drawdown, drawdown-over-volatility
and Ulcer differentials against the incumbent, on one shared index draw, and
publishes the 95% one-sided minimum detectable effect at every declared block
length.  On 1990-2014 that floor runs from 0.02 to 0.37 on the drawdown ratio
across the matched panel, and no matched setting on any swept axis clears the
larger of its two floors.  A delta that does not clear that larger floor is
emitted as ``not_estimable`` with the floor beside it rather than as a number,
following ``deflated_sharpe_ratio``'s shape; the underlying level is still
reported, so nothing is hidden.

Two further properties are load-bearing.

The binding-frequency diagnostic is the explanation, not decoration.  A bound
that never touches the realised path cannot change anything, and a sweep that
reports a difference without reporting whether the constraint was ever active
is reporting simulation noise.  ``bound_binding_diagnostic`` re-derives the
signal stage from the shared frames and refuses to report a binding share for
any constraint whose reconstruction does not reproduce the engine's own
delivered forecast exactly.

The drawdown-risk bootstrap draws its stationary-bootstrap index matrix once
and applies it to every variant.  Without those common random numbers a
difference of a few tenths of a percentage point in breach probability is
indistinguishable from the resampling noise between two independent draws.

Nothing here ranks, scores or recommends.  These are reused 1990-2014
numbers, and ``docs/research-methodology.md`` prohibits selecting a
configuration on reused history.  The paired-inference panel is fixed by
``declared_paired_comparisons`` from the grid alone, before any result exists,
so the subset that receives an interval is not itself a selection; a declared
comparison whose variant was never built is reported as refused rather than
dropped, so the panel that was declared and the panel that was reported cannot
drift apart silently.

What this module does not establish.  The risk match is a one-dimensional
rescale of ``target_vol`` verified ex post to a stated tolerance; it is not an
ex-ante risk equalisation, and a matched variant is a different execution
problem rather than the same book at a different size.  The solver-jitter band
is measured on the incumbent configuration and applied to every matched row, so
it is a shared control rather than a per-variant estimate.  The bootstrap
resamples the realised path and therefore assumes future dependence resembles
1990-2014's; it says nothing about a regime absent from that window.  And a
favourable matched row is a hypothesis for prospective evaluation, not evidence
that a bound should be changed.

``levers.run_lever_comparison`` supplies the standard per-variant metric,
turnover and ledger-reconciliation columns and is used unchanged for them.  It
is driven in phases rather than in one call because the risk-matched
configurations cannot be constructed until the as-is realised volatilities
exist, and because a ``BacktestResult`` costs roughly 35 MB: streaming the
variants through the harness keeps at most the pinned baseline and the running
variant alive at once.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import inference, levers


BASELINE_MODE = "baseline"
AS_IS_MODE = "as_is"
RISK_MATCHED_MODE = "risk_matched"
MATCH_CONTROL_MODE = "risk_match_control"

BOUNDS_VALIDATION_STATUS = "retrospective_bound_sensitivity"

RISK_SCALAR_LEVER = "risk_scalar_bounds"
MATCH_CONTROL_LEVER = "risk_match_control"

ESTIMATED = "estimated"
NOT_ESTIMABLE = "not_estimable"
REFERENCE_ROW = "reference"

# The loss-shape statistics whose matched deltas are gated on a measured
# resolution floor.  They are exactly the statistics ``drawdown_shape_metrics``
# computes itself, so the bootstrap estimator and the reported level are the
# same function of the same path by construction rather than by inspection.
RESOLUTION_STATISTICS = (
    "max_drawdown",
    "max_drawdown_over_annualized_volatility",
    "ulcer_index",
)

_PERMITTED_USE = (
    "descriptive bound sensitivity on reused history; not selection adjusted "
    "and does not license a configuration change"
)

# The relaxed cap used to detect whether ``risk_managed_cap`` changed the
# forecast the engine actually delivered.  Any value large enough that the
# volatility weight is never truncated works; the constant is finite so the
# arithmetic stays in floating point.
_RELAXED_CAP = 1.0e12

_BINDING_TOLERANCE = 1e-12

BOUNDS_SHAPE_COLUMNS = [
    "variant",
    "lever",
    "evaluation_mode",
    "bound_setting",
    "validation_status",
    "risk_budget_volatility",
    "reference_annualized_volatility",
    "annualized_volatility_gap_to_reference",
    "solver_engine_runs",
    "start",
    "end",
    "years",
    "annualized_volatility",
    "cagr",
    "sharpe",
    "hac_sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "max_drawdown_duration_sessions",
    "max_drawdown_over_annualized_volatility",
    "ulcer_index",
    "pain_index",
    "worst_rolling_21_sessions",
    "worst_rolling_63_sessions",
    "conditional_shortfall_95",
    "conditional_shortfall_99",
    "annualized_downside_deviation",
    "daily_return_skew",
    "daily_return_excess_kurtosis",
    "average_risk_scalar",
    "peak_risk_scalar",
    "trough_risk_scalar",
    "average_gross_notional_multiple",
    "peak_gross_notional_multiple",
    "portfolio_limit_binding_decision_days",
    "annual_cost_drag",
    "annual_rebalance_turnover_contracts",
    "annual_roll_turnover_contracts",
    "delta_annualized_volatility",
    "delta_cagr",
    "delta_sharpe",
    "delta_max_drawdown",
    "max_drawdown_resolution_floor",
    "delta_max_drawdown_status",
    "delta_max_drawdown_over_annualized_volatility",
    "max_drawdown_over_annualized_volatility_resolution_floor",
    "delta_max_drawdown_over_annualized_volatility_status",
    "delta_ulcer_index",
    "ulcer_index_resolution_floor",
    "delta_ulcer_index_status",
    "delta_annual_cost_drag",
]

BOUNDS_RESOLUTION_COLUMNS = [
    "comparison",
    "evaluation_mode",
    "statistic",
    "method",
    "expected_block_sessions",
    "samples",
    "seed",
    "sessions",
    "common_random_numbers",
    "return_correlation_with_incumbent",
    "point_estimate",
    "bootstrap_standard_error",
    "confidence_lower",
    "confidence_upper",
    "confidence_method",
    "minimum_detectable_effect_95_one_sided",
    "selection_adjusted",
    "permitted_use",
]

BOUNDS_MATCH_JITTER_COLUMNS = [
    "statistic",
    "method",
    "band_status",
    "band_basis",
    "probe_variants",
    "probes_inside_tolerance",
    "lowest_risk_budget_volatility",
    "highest_risk_budget_volatility",
    "lowest_achieved_annualized_volatility",
    "highest_achieved_annualized_volatility",
    "widest_volatility_match_error",
    "tolerance",
    "lowest_observed",
    "highest_observed",
    "solver_jitter_band",
    "selection_adjusted",
    "permitted_use",
]

BOUNDS_BINDING_COLUMNS = [
    "variant",
    "evaluation_mode",
    "bound",
    "bound_value",
    "binding_status",
    "observation_unit",
    "observations",
    "binding_observations",
    "binding_share",
    "extreme_observed",
    "binding_basis",
]


def drawdown_risk_columns(
    config: DrawdownRiskConfig | None = None,
) -> list[str]:
    """Column order for the drawdown-risk report at the given thresholds.

    The two breach columns name their thresholds, so the schema depends on the
    configuration rather than being a constant.  Building it from the config
    keeps the name and the number it reports from drifting apart.
    """

    lower, upper = (config or DrawdownRiskConfig()).breach_thresholds
    return [
        "variant",
        "evaluation_mode",
        "method",
        "samples",
        "expected_block_sessions",
        "horizon_sessions",
        "seed",
        "source_sessions",
        "common_random_numbers",
        f"bootstrap_p_drawdown_breach_{int(round(lower * 100))}pct",
        f"bootstrap_p_drawdown_breach_{int(round(upper * 100))}pct",
        "bootstrap_p01_max_drawdown",
        "bootstrap_p05_max_drawdown",
        "bootstrap_median_max_drawdown",
        "bootstrap_mean_max_drawdown",
        "selection_adjusted",
        "permitted_use",
    ]

BOUNDS_RISK_MATCH_COLUMNS = [
    "variant",
    "source_variant",
    "iteration",
    "method",
    "risk_budget_volatility",
    "achieved_annualized_volatility",
    "reference_annualized_volatility",
    "volatility_match_error",
    "relative_volatility_match_error",
    "within_tolerance",
    "tolerance",
]

BOUNDS_POWER_COLUMNS = [
    "comparison",
    "power_status",
    "power_basis",
    "tracking_error_annualized",
    "standard_error_of_sharpe_differential",
    "standard_error_of_cagr_differential",
    "minimum_detectable_sharpe_effect_95_one_sided",
    "minimum_detectable_cagr_effect_95_one_sided",
    "selection_adjusted",
    "permitted_use",
]

POWER_ESTIMATED = ESTIMATED
POWER_NOT_ESTIMABLE = NOT_ESTIMABLE


def _finite_positive_tuple(
    name: str,
    values: Any,
    *,
    minimum: float,
    maximum: float | None,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} entries must be real numbers")
        if not np.isfinite(value) or value < minimum:
            raise ValueError(f"{name} entries must be finite and at least {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} entries must not exceed {maximum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")


@dataclass(frozen=True)
class BoundGrid:
    """The magnitude bounds to sweep, and the settings to sweep them over.

    The risk-scalar pair is swept as a full grid because the user's question
    names both ends of it and because the two interact: the lower bound can
    only matter in the regimes the upper bound has already left alone.  The
    remaining four bounds are swept one at a time off the canonical setting,
    which prices each in isolation and keeps the engine budget bounded.
    """

    max_risk_scalars: tuple[float, ...] = (1.00, 1.25, 1.50, 1.75, 2.00)
    min_risk_scalars: tuple[float, ...] = (0.00, 0.10, 0.25)
    signal_caps: tuple[float, ...] = (1.0, 1.5, 2.0)
    risk_managed_caps: tuple[float, ...] = (1.0, 1.5, 2.0)
    shock_floors: tuple[float, ...] = (0.25, 0.50, 0.75)
    max_gross_notional_multiples: tuple[float, ...] = (3.0, 4.0, 5.0)

    def __post_init__(self) -> None:
        _finite_positive_tuple(
            "max_risk_scalars", self.max_risk_scalars, minimum=1e-9, maximum=None
        )
        _finite_positive_tuple(
            "min_risk_scalars", self.min_risk_scalars, minimum=0.0, maximum=None
        )
        _finite_positive_tuple("signal_caps", self.signal_caps, minimum=1e-9, maximum=None)
        _finite_positive_tuple(
            "risk_managed_caps", self.risk_managed_caps, minimum=1e-9, maximum=None
        )
        _finite_positive_tuple("shock_floors", self.shock_floors, minimum=0.0, maximum=1.0)
        _finite_positive_tuple(
            "max_gross_notional_multiples",
            self.max_gross_notional_multiples,
            minimum=1e-9,
            maximum=None,
        )
        if max(self.min_risk_scalars) > min(self.max_risk_scalars):
            raise ValueError(
                "every risk-scalar pair must satisfy min <= max; the grid as "
                f"given produces the infeasible pair "
                f"min={max(self.min_risk_scalars)} > max={min(self.max_risk_scalars)}"
            )


@dataclass(frozen=True)
class RiskMatchConfig:
    """How a bound variant is returned to the incumbent's realised volatility.

    ``reference_volatility`` defaults to ``None``, meaning the volatility the
    baseline actually realised in this run rather than a number transcribed
    from a frozen artifact.  A transcribed reference would silently become
    wrong the moment the engine changed.

    The tolerance is 5 basis points of annualised volatility.  A first-order
    rescale lands inside it on this engine, so the usual cost of a match is one
    extra backtest, and the residual is reported on every row regardless.

    Tightening the tolerance is not by itself a fix for the dispersion the
    tolerance admits.  Changing ``target_vol`` re-runs a discrete-execution
    engine rather than rescaling a return series, and on 1990-2014 a residual of
    2.7 basis points already moves maximum drawdown over annualised volatility
    by 0.032.  ``control_probe_fractions`` is the answer instead: it names the
    positions inside the tolerance band, in units of the tolerance, at which the
    UNCHANGED incumbent configuration is re-run so the dispersion the band
    admits is measured and published rather than assumed away.  Set it to an
    empty tuple to skip the control, at the cost of having no measured solver
    floor to read the matched rows against.
    """

    reference_volatility: float | None = None
    tolerance: float = 5.0e-4
    max_iterations: int = 3
    min_multiplier: float = 0.2
    max_multiplier: float = 5.0
    control_probe_fractions: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)

    def __post_init__(self) -> None:
        if self.reference_volatility is not None and (
            not np.isfinite(self.reference_volatility) or self.reference_volatility <= 0
        ):
            raise ValueError("reference_volatility must be finite and positive")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not 0 < self.min_multiplier < 1 < self.max_multiplier:
            raise ValueError("multipliers must satisfy 0 < min < 1 < max")
        if not isinstance(self.control_probe_fractions, tuple):
            raise ValueError("control_probe_fractions must be a tuple")
        for fraction in self.control_probe_fractions:
            if isinstance(fraction, bool) or not isinstance(fraction, int | float):
                raise ValueError("control_probe_fractions entries must be real numbers")
            if not np.isfinite(fraction) or not 0 < abs(fraction) <= 1:
                raise ValueError(
                    "control_probe_fractions entries must be non-zero and no "
                    "larger than one tolerance in magnitude; a probe outside "
                    "the band would not measure what the band admits"
                )
        if len(set(self.control_probe_fractions)) != len(self.control_probe_fractions):
            raise ValueError("control_probe_fractions entries must be unique")


@dataclass(frozen=True)
class LossShapeResolutionConfig:
    """Sampling floor for the loss-shape differentials, by paired bootstrap.

    The block lengths are the repository's declared grid rather than a chosen
    one, so the friendliest dependence assumption cannot be picked after the
    fact; the reported floor is the widest across the grid.  The draw is shared
    across every comparison, which is what makes the floors comparable to one
    another and to the drawdown-risk report.
    """

    block_lengths: tuple[int, ...] = inference.DEFAULT_BLOCK_LENGTHS
    samples: int = 2_000
    seed: int = inference.DEFAULT_PAIRED_SEED
    batch_size: int = 250
    confidence_z: float = 1.645

    def __post_init__(self) -> None:
        if not isinstance(self.block_lengths, tuple) or not self.block_lengths:
            raise ValueError("block_lengths must be a non-empty tuple")
        for block in self.block_lengths:
            if isinstance(block, bool) or not isinstance(block, int) or block < 1:
                raise ValueError("block_lengths entries must be positive integers")
        if len(set(self.block_lengths)) != len(self.block_lengths):
            raise ValueError("block_lengths entries must be unique")
        for name in ("samples", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not np.isfinite(self.confidence_z) or self.confidence_z <= 0:
            raise ValueError("confidence_z must be finite and positive")


@dataclass(frozen=True)
class DrawdownRiskConfig:
    """Stationary-bootstrap forward drawdown risk, on one shared index draw.

    The horizon is ten years of sessions because the drawdown policy the
    numbers are compared against is a live policy, and a policy breach is a
    property of a forward path rather than of the single realised one.
    """

    horizon_sessions: int = 2_520
    block_length_sessions: int = 63
    samples: int = 2_000
    seed: int = inference.DEFAULT_PAIRED_SEED
    breach_thresholds: tuple[float, float] = (0.15, 0.20)
    batch_size: int = 250

    def __post_init__(self) -> None:
        for name in ("horizon_sessions", "block_length_sessions", "samples", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.breach_thresholds, tuple) or len(self.breach_thresholds) != 2:
            raise ValueError("breach_thresholds must be a pair")
        first, second = self.breach_thresholds
        if not 0 < first < second < 1:
            raise ValueError("breach_thresholds must satisfy 0 < first < second < 1")


@dataclass(frozen=True)
class BoundsSweepArtifacts:
    """The frames a bound sweep produces, one per question it answers."""

    comparison: pd.DataFrame
    shape: pd.DataFrame
    binding: pd.DataFrame
    drawdown_risk: pd.DataFrame
    risk_match: pd.DataFrame
    match_jitter: pd.DataFrame
    resolution: pd.DataFrame
    paired: pd.DataFrame
    power: pd.DataFrame
    monthly_net_returns: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "comparison",
            "shape",
            "binding",
            "drawdown_risk",
            "risk_match",
            "match_jitter",
            "resolution",
            "paired",
            "power",
            "monthly_net_returns",
        ):
            if not isinstance(getattr(self, name), pd.DataFrame):
                raise ValueError(f"{name} must be a pandas DataFrame")


def _format_bound(value: float) -> str:
    return f"{float(value):.2f}"


def build_bound_variants(
    config: Any,
    grid: BoundGrid = BoundGrid(),
) -> tuple[levers.LeverVariant, ...]:
    """Baseline first, then every distinct bound setting in the grid.

    The baseline cell is removed from each axis rather than reported twice, so
    a reader cannot mistake a duplicate run for an independent observation.
    """

    if not isinstance(grid, BoundGrid):
        raise TypeError("grid must be a BoundGrid")
    required = (
        "max_risk_scalar",
        "min_risk_scalar",
        "signal_cap",
        "risk_managed_cap",
        "shock_floor",
        "max_gross_notional_multiple",
        "target_vol",
    )
    missing = [field for field in required if not hasattr(config, field)]
    if missing:
        raise ValueError(f"config does not define the swept bounds: {missing}")

    variants: list[levers.LeverVariant] = [
        levers.LeverVariant(
            name="baseline",
            lever="baseline",
            evaluation_mode=BASELINE_MODE,
            assumption=(
                "Canonical configuration: risk-scalar clip "
                f"[{config.min_risk_scalar:g}, {config.max_risk_scalar:g}], "
                f"signal_cap {config.signal_cap:g}, risk_managed_cap "
                f"{config.risk_managed_cap:g}, shock_floor {config.shock_floor:g}, "
                f"gross-notional multiple {config.max_gross_notional_multiple}."
            ),
        )
    ]

    for upper in grid.max_risk_scalars:
        for lower in grid.min_risk_scalars:
            if lower > upper:
                raise ValueError(f"infeasible risk-scalar pair min={lower} > max={upper}")
            if upper == config.max_risk_scalar and lower == config.min_risk_scalar:
                continue
            variants.append(
                levers.LeverVariant(
                    name=(
                        f"risk_scalar_max{_format_bound(upper)}"
                        f"_min{_format_bound(lower)}"
                    ),
                    lever=RISK_SCALAR_LEVER,
                    evaluation_mode=AS_IS_MODE,
                    assumption=(
                        "EWMA volatility-target multiplier clipped to "
                        f"[{lower:g}, {upper:g}] with every other control "
                        "unchanged."
                    ),
                    overrides=(
                        ("max_risk_scalar", float(upper)),
                        ("min_risk_scalar", float(lower)),
                    ),
                )
            )

    one_at_a_time: tuple[tuple[str, tuple[float, ...], str], ...] = (
        (
            "signal_cap",
            grid.signal_caps,
            "Basis-momentum normalisation cap set to {value:g}.",
        ),
        (
            "risk_managed_cap",
            grid.risk_managed_caps,
            "Forecast volatility-weight cap set to {value:g}.",
        ),
        (
            "shock_floor",
            grid.shock_floors,
            "Volatility-shock taper floor set to {value:g}.",
        ),
        (
            "max_gross_notional_multiple",
            grid.max_gross_notional_multiples,
            "Gross-notional multiple limited to {value:g} of NAV.",
        ),
    )
    for field, values, template in one_at_a_time:
        for value in values:
            if value == getattr(config, field):
                continue
            variants.append(
                levers.LeverVariant(
                    name=f"{field}_{_format_bound(value)}",
                    lever=field,
                    evaluation_mode=AS_IS_MODE,
                    assumption=(
                        template.format(value=value)
                        + " Every other control unchanged."
                    ),
                    overrides=((field, float(value)),),
                )
            )

    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError(
            "bound variant names must be unique; two grid settings on the same "
            "axis round to the same two-decimal label"
        )
    return tuple(variants)


def risk_matched_variant(
    variant: levers.LeverVariant,
    *,
    risk_budget_volatility: float,
) -> levers.LeverVariant:
    """The same bound setting with the risk budget rescaled to match."""

    if not np.isfinite(risk_budget_volatility) or risk_budget_volatility <= 0:
        raise ValueError("risk_budget_volatility must be finite and positive")
    overrides = tuple(
        (key, value) for key, value in variant.overrides if key != "target_vol"
    )
    return levers.LeverVariant(
        name=f"{variant.name}_risk_matched",
        lever=variant.lever,
        evaluation_mode=RISK_MATCHED_MODE,
        assumption=(
            f"{variant.assumption} Risk budget rescaled to "
            f"{risk_budget_volatility:.6f} so realised annualised volatility "
            "returns to the incumbent's."
        ),
        overrides=(*overrides, ("target_vol", float(risk_budget_volatility))),
    )


def declared_paired_comparisons(
    config: Any,
    grid: BoundGrid = BoundGrid(),
) -> tuple[str, ...]:
    """Risk-matched variant names that receive a paired interval.

    Fixed by the grid, before any number exists.  The panel is the whole
    primary risk-scalar axis at the canonical opposite bound — which is the
    pair the question actually names — plus the setting furthest from the
    canonical one on each remaining axis.  Choosing it from the grid rather
    than from the results is what keeps it from being a selection.

    Every name is built by holding the *canonical* setting fixed on the other
    axis, so a grid that omits the canonical setting would declare comparisons
    the sweep never builds.  That is refused rather than silently truncated: a
    declared panel that is not a subset of the built panel is not a control at
    all, because the rows that went missing are unobservable to the reader.
    """

    for field, values in (
        ("max_risk_scalar", grid.max_risk_scalars),
        ("min_risk_scalar", grid.min_risk_scalars),
        ("signal_cap", grid.signal_caps),
        ("risk_managed_cap", grid.risk_managed_caps),
        ("shock_floor", grid.shock_floors),
        ("max_gross_notional_multiple", grid.max_gross_notional_multiples),
    ):
        canonical = getattr(config, field, None)
        if canonical is None or not any(value == canonical for value in values):
            raise ValueError(
                f"the {field} axis must contain the canonical setting "
                f"{canonical!r} so the declared comparison panel is a subset "
                f"of the built variants; the grid supplies {values}"
            )

    names: list[str] = []
    for upper in grid.max_risk_scalars:
        if upper == config.max_risk_scalar:
            continue
        names.append(
            f"risk_scalar_max{_format_bound(upper)}"
            f"_min{_format_bound(config.min_risk_scalar)}_risk_matched"
        )
    for lower in grid.min_risk_scalars:
        if lower == config.min_risk_scalar:
            continue
        names.append(
            f"risk_scalar_max{_format_bound(config.max_risk_scalar)}"
            f"_min{_format_bound(lower)}_risk_matched"
        )
    for lever, values in (
        ("signal_cap", grid.signal_caps),
        ("risk_managed_cap", grid.risk_managed_caps),
        ("shock_floor", grid.shock_floors),
        ("max_gross_notional_multiple", grid.max_gross_notional_multiples),
    ):
        candidates = [value for value in values if value != getattr(config, lever)]
        if not candidates:
            continue
        names.append(f"{lever}_{_format_bound(min(candidates))}_risk_matched")
    return tuple(dict.fromkeys(names))


def _loss_shape_kernel(
    block: np.ndarray,
    annualization: int,
) -> dict[str, np.ndarray]:
    """Vectorised loss-shape statistics for a stack of return paths.

    One implementation serves both the realised path and every bootstrap
    resample, so the point estimate whose delta is reported and the sampling
    distribution that delta is read against are the same function of the same
    definition rather than two transcriptions of it.
    """

    equity = np.cumprod(1.0 + block, axis=-1)
    peaks = np.maximum.accumulate(np.maximum(equity, 1.0), axis=-1)
    drawdown = equity / peaks - 1.0
    max_drawdown = drawdown.min(axis=-1)
    volatility = block.std(axis=-1, ddof=1) * math.sqrt(annualization)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(volatility > 0, np.abs(max_drawdown) / volatility, np.nan)
    return {
        "annualized_volatility": volatility,
        "max_drawdown": max_drawdown,
        "max_drawdown_over_annualized_volatility": ratio,
        "ulcer_index": np.sqrt(np.mean(np.square(drawdown), axis=-1)),
        "pain_index": np.mean(np.abs(drawdown), axis=-1),
    }


def drawdown_shape_metrics(
    returns: pd.Series,
    *,
    annualization: int = 252,
) -> dict[str, float]:
    """Loss-shape statistics ``performance_metrics`` does not already report.

    ``max_drawdown_over_annualized_volatility`` is the statistic the matched
    rows are read on, because a *constant rescale of a return series* multiplies
    numerator and denominator by very nearly the same factor and so cannot move
    it.  That property is arithmetic and it is real, but it is not what licenses
    the comparison here, and the module does not lean on it.  The risk match is
    not a constant rescale: it re-runs a discrete-execution engine at a
    different ``target_vol``, and integer contracts, no-trade buffer crossings,
    participation clipping and the gross-notional limit all resolve differently,
    so the matched variant walks a genuinely different path.

    The margin this ratio is read against is therefore measured, not derived.
    ``loss_shape_resolution`` supplies the sampling floor by paired bootstrap
    and the risk-match control ladder supplies the solver floor; on 1990-2014
    those are roughly 0.02-0.37 and 0.06 respectively, against a level of about
    1.53.  Both dwarf the ~1% residual that compounding leaves in the
    scale-invariance argument, which is why that argument is not the one the
    report relies on.

    The volatility convention matches ``performance_metrics`` exactly (sample
    standard deviation, ``sqrt(annualization)``) so the two are directly
    comparable.
    """

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    values = returns.dropna()
    if values.empty:
        raise ValueError("returns cannot be empty")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("returns must be finite")
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, int)
        or annualization < 1
    ):
        raise ValueError("annualization must be a positive integer")

    kernel = _loss_shape_kernel(values.to_numpy(dtype=float), annualization)

    rolling_21 = (1.0 + values).rolling(21).apply(np.prod, raw=True) - 1.0
    rolling_63 = (1.0 + values).rolling(63).apply(np.prod, raw=True) - 1.0

    def shortfall(level: float) -> float:
        quantile = float(values.quantile(level))
        tail = values[values <= quantile]
        return -float(tail.mean()) if len(tail) else float("nan")

    return {
        "annualized_volatility": float(kernel["annualized_volatility"]),
        "max_drawdown": float(kernel["max_drawdown"]),
        "max_drawdown_over_annualized_volatility": float(
            kernel["max_drawdown_over_annualized_volatility"]
        ),
        "ulcer_index": float(kernel["ulcer_index"]),
        "pain_index": float(kernel["pain_index"]),
        "worst_rolling_21_sessions": float(rolling_21.min()),
        "worst_rolling_63_sessions": float(rolling_63.min()),
        "conditional_shortfall_95": shortfall(0.05),
        "conditional_shortfall_99": shortfall(0.01),
    }


def _signal_stage_frames(
    config: Any,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Re-derive the pre-sizing signal stage the engine runs internally.

    The three signal-stage bounds are applied inside ``run_backtest`` and are
    not recoverable from a ``BacktestResult``: the delivered forecast has
    already been clipped to [-1, 1], which hides whether an earlier cap was the
    active constraint.  Reconstructing the stage is the only way to measure
    them, and the reconstruction is checked against the engine's own output
    before any number derived from it is reported.
    """

    from .strategy import (
        basis_momentum,
        blend_signals,
        risk_managed_forecast,
        shock_multiplier,
        tradeable_mask,
        trend_signal,
    )

    prices = frames["prices"]
    unadjusted = frames["unadjusted"]
    volumes = frames["volumes"]
    observed_opens = frames["observed_opens"]
    observed_closes = frames["observed_closes"]

    trend = trend_signal(prices, config.trend_lookback)
    basis = basis_momentum(
        prices,
        unadjusted,
        vol_span=config.vol_span,
        cap=config.signal_cap,
        roll_window=config.basis_roll_window,
        lookback=config.basis_lookback,
        normalization_window=config.signal_normalization_window,
    )
    blended = blend_signals(trend, basis, config.basis_weight)
    shock = shock_multiplier(prices, config)
    forecast = (blended * shock).clip(-1, 1)

    delivered = forecast
    relaxed = forecast
    if config.risk_managed_window is not None:
        execution_opens = observed_opens.where(
            observed_closes.notna() & volumes.gt(0)
        )
        opens = execution_opens if config.execution_timing == "next_open" else None
        delivered = risk_managed_forecast(
            forecast,
            prices,
            config.vol_span,
            config.risk_managed_window,
            config.risk_managed_cap,
            opens,
        )
        relaxed = risk_managed_forecast(
            forecast,
            prices,
            config.vol_span,
            config.risk_managed_window,
            _RELAXED_CAP,
            opens,
        )
    tradeable = (
        tradeable_mask(volumes, config.volume_gate_window, config.min_median_contracts)
        .reindex(prices.index)
        .fillna(False)
    )
    return {
        "basis": basis,
        "shock": shock,
        "risk_managed_delivered": delivered.where(
            tradeable.reindex_like(delivered), np.nan
        ),
        "risk_managed_relaxed": relaxed.where(tradeable.reindex_like(relaxed), np.nan),
    }


def _binding_row(
    *,
    variant: str,
    evaluation_mode: str,
    bound: str,
    bound_value: Any,
    status: str,
    unit: str,
    observations: int,
    binding: int,
    extreme: float,
    basis: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "evaluation_mode": evaluation_mode,
        "bound": bound,
        "bound_value": bound_value,
        "binding_status": status,
        "observation_unit": unit,
        "observations": observations,
        "binding_observations": binding,
        "binding_share": (binding / observations) if observations else float("nan"),
        "extreme_observed": extreme,
        "binding_basis": basis,
    }


def _frame_binding(
    frame: pd.DataFrame,
    mask: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[int, int]:
    window = frame.loc[start:end]
    active = window.notna().to_numpy(dtype=bool)
    hits = mask.loc[start:end].to_numpy(dtype=bool) & active
    return int(active.sum()), int(hits.sum())


def _extreme(frame: pd.DataFrame, start: str, end: str, *, largest: bool) -> float:
    """Largest or smallest finite cell, or NaN when the window has none."""

    values = frame.loc[start:end].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max() if largest else finite.min())


def bound_binding_diagnostic(
    result: Any,
    config: Any,
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
    variant: str = "baseline",
    evaluation_mode: str = BASELINE_MODE,
    start: str = levers.RETROSPECTIVE_START,
    end: str = levers.RETROSPECTIVE_END,
) -> pd.DataFrame:
    """How often each magnitude bound was the active constraint.

    Each bound gets the definition that is meaningful for it, named in
    ``binding_basis`` rather than left implicit.  The risk-scalar clip and the
    basis-momentum cap are exact equalities with the bound.  The
    ``risk_managed_cap`` row is functional: it counts market-sessions where the
    cap changed the forecast the engine delivered, because a capped and an
    uncapped weight are indistinguishable once the final [-1, 1] clip has
    fired.  ``shock_floor`` counts market-sessions sitting exactly on the
    floor, and the row records that the floor also reshapes the ramp above it,
    so it is not a pure clip.

    A row whose constraint cannot be re-derived is emitted with
    ``binding_status`` ``not_estimable`` and a reason, never with a zero.
    """

    from .strategy import _month_end_rows

    rows: list[dict[str, Any]] = []
    daily = getattr(result, "daily", None)
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("result must carry a non-empty daily ledger")
    window = daily.loc[start:end]
    if window.empty:
        raise ValueError("the evaluation window selects no sessions")

    if "risk_scalar" in window:
        scalar = window["risk_scalar"].dropna()
        decision_values = _month_end_rows(scalar).to_numpy(dtype=float)
        decision_count = int(decision_values.size)
        for bound, value, extreme_of in (
            ("max_risk_scalar", float(config.max_risk_scalar), np.max),
            ("min_risk_scalar", float(config.min_risk_scalar), np.min),
        ):
            hits = int(
                np.isclose(
                    decision_values, value, rtol=0.0, atol=_BINDING_TOLERANCE
                ).sum()
            )
            rows.append(
                _binding_row(
                    variant=variant,
                    evaluation_mode=evaluation_mode,
                    bound=bound,
                    bound_value=value,
                    status="measured",
                    unit="monthly decision",
                    observations=decision_count,
                    binding=hits,
                    extreme=(
                        float(extreme_of(decision_values))
                        if decision_count
                        else float("nan")
                    ),
                    basis=(
                        "applied EWMA volatility-target multiplier exactly equal "
                        "to the clip, read at each completed monthly decision"
                    ),
                )
            )
    else:
        for bound, value in (
            ("max_risk_scalar", float(config.max_risk_scalar)),
            ("min_risk_scalar", float(config.min_risk_scalar)),
        ):
            rows.append(
                _binding_row(
                    variant=variant,
                    evaluation_mode=evaluation_mode,
                    bound=bound,
                    bound_value=value,
                    status="not_estimable",
                    unit="monthly decision",
                    observations=0,
                    binding=0,
                    extreme=float("nan"),
                    basis="the daily ledger carries no risk_scalar column",
                )
            )

    if "target_portfolio_limit_scale" in window:
        limit_scale = window["target_portfolio_limit_scale"].to_numpy(dtype=float)
        rows.append(
            _binding_row(
                variant=variant,
                evaluation_mode=evaluation_mode,
                bound="max_gross_notional_multiple",
                bound_value=config.max_gross_notional_multiple,
                status="measured",
                unit="session",
                observations=int(limit_scale.size),
                binding=int((limit_scale < 1.0 - _BINDING_TOLERANCE).sum()),
                extreme=(
                    float(window["gross_notional_multiple"].max())
                    if "gross_notional_multiple" in window
                    else float("nan")
                ),
                basis=(
                    "sessions on which the engine scaled the decision book down "
                    "to respect the gross-notional multiple"
                ),
            )
        )
    else:
        rows.append(
            _binding_row(
                variant=variant,
                evaluation_mode=evaluation_mode,
                bound="max_gross_notional_multiple",
                bound_value=config.max_gross_notional_multiple,
                status="not_estimable",
                unit="session",
                observations=0,
                binding=0,
                extreme=float("nan"),
                basis="the daily ledger carries no portfolio-limit scale column",
            )
        )

    signal_bounds = (
        ("signal_cap", float(config.signal_cap)),
        ("risk_managed_cap", float(config.risk_managed_cap)),
        ("shock_floor", float(config.shock_floor)),
    )
    if frames is None:
        for bound, value in signal_bounds:
            rows.append(
                _binding_row(
                    variant=variant,
                    evaluation_mode=evaluation_mode,
                    bound=bound,
                    bound_value=value,
                    status="not_estimable",
                    unit="market-session",
                    observations=0,
                    binding=0,
                    extreme=float("nan"),
                    basis=(
                        "signal-stage frames were not supplied, so the "
                        "constraint cannot be re-derived from the ledger alone"
                    ),
                )
            )
        return pd.DataFrame(rows, columns=BOUNDS_BINDING_COLUMNS)

    stage = _signal_stage_frames(config, frames)
    signals = getattr(result, "signals", None)
    reproduced = (
        isinstance(signals, pd.DataFrame)
        and stage["risk_managed_delivered"].shape == signals.shape
        and stage["risk_managed_delivered"].equals(signals)
    )
    if not reproduced:
        for bound, value in signal_bounds:
            rows.append(
                _binding_row(
                    variant=variant,
                    evaluation_mode=evaluation_mode,
                    bound=bound,
                    bound_value=value,
                    status="not_estimable",
                    unit="market-session",
                    observations=0,
                    binding=0,
                    extreme=float("nan"),
                    basis=(
                        "the signal-stage reconstruction did not reproduce the "
                        "engine's delivered forecast, so any binding share "
                        "derived from it would describe a different pipeline"
                    ),
                )
            )
        return pd.DataFrame(rows, columns=BOUNDS_BINDING_COLUMNS)

    basis_frame = stage["basis"]
    observations, hits = _frame_binding(
        basis_frame,
        basis_frame.abs() >= 1.0 - _BINDING_TOLERANCE,
        start,
        end,
    )
    rows.append(
        _binding_row(
            variant=variant,
            evaluation_mode=evaluation_mode,
            bound="signal_cap",
            bound_value=float(config.signal_cap),
            status="measured",
            unit="market-session",
            observations=observations,
            binding=hits,
            extreme=_extreme(basis_frame.abs(), start, end, largest=True),
            basis=(
                "normalised basis-momentum cells saturated at the cap before "
                "the sleeve blend"
            ),
        )
    )

    delivered = stage["risk_managed_delivered"]
    relaxed = stage["risk_managed_relaxed"]
    observations, hits = _frame_binding(
        delivered,
        (delivered - relaxed).abs() > _BINDING_TOLERANCE,
        start,
        end,
    )
    rows.append(
        _binding_row(
            variant=variant,
            evaluation_mode=evaluation_mode,
            bound="risk_managed_cap",
            bound_value=float(config.risk_managed_cap),
            status="measured",
            unit="market-session",
            observations=observations,
            binding=hits,
            extreme=_extreme(delivered.abs(), start, end, largest=True),
            basis=(
                "market-sessions where removing the cap would have changed the "
                "delivered forecast; a structural test on the weight would "
                "overcount cells the final clip makes identical anyway"
            ),
        )
    )

    shock = stage["shock"]
    observations, hits = _frame_binding(
        shock,
        (shock - float(config.shock_floor)).abs() <= _BINDING_TOLERANCE,
        start,
        end,
    )
    rows.append(
        _binding_row(
            variant=variant,
            evaluation_mode=evaluation_mode,
            bound="shock_floor",
            bound_value=float(config.shock_floor),
            status="measured",
            unit="market-session",
            observations=observations,
            binding=hits,
            extreme=_extreme(shock, start, end, largest=False),
            basis=(
                "market-sessions resting exactly on the taper floor; the floor "
                "also sets the slope of the ramp above it, so this share is a "
                "lower bound on the parameter's influence"
            ),
        )
    )
    return pd.DataFrame(rows, columns=BOUNDS_BINDING_COLUMNS)


def stationary_drawdown_indices(
    source_length: int,
    config: DrawdownRiskConfig = DrawdownRiskConfig(),
) -> np.ndarray:
    """The one index matrix every variant's drawdown bootstrap is read through.

    Seeded from the block length and horizon only.  No variant label enters
    the seed, which is the whole point: two variants must be resampled on
    identical paths or their breach probabilities differ by draw noise before
    they differ by anything else.
    """

    if isinstance(source_length, bool) or not isinstance(source_length, int):
        raise ValueError("source_length must be an integer")
    if source_length < 2:
        raise ValueError("source_length must be at least 2")
    if not isinstance(config, DrawdownRiskConfig):
        raise TypeError("config must be a DrawdownRiskConfig")
    method_seed = (
        zlib.crc32(b"common random number stationary bootstrap drawdown") & 0xFFFFFFFF
    )
    entropy = np.random.SeedSequence(
        [
            int(config.seed) & 0xFFFFFFFF,
            method_seed,
            int(config.block_length_sessions),
            int(config.horizon_sessions),
        ]
    )
    rng = np.random.default_rng(entropy)
    return inference._stationary_indices(
        rng,
        source_length,
        config.samples,
        config.horizon_sessions,
        config.block_length_sessions,
    )


def common_random_number_drawdown_risk(
    paths: Mapping[str, pd.Series],
    *,
    config: DrawdownRiskConfig = DrawdownRiskConfig(),
    evaluation_modes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Forward drawdown risk for every variant, on one shared index draw.

    Fails closed on any index mismatch rather than intersecting, for the same
    reason ``inference._aligned_pair`` does: a shared draw is only meaningful
    if every series is indexed identically, and a silent intersection would
    leave the report looking paired when it is not.
    """

    if not paths:
        raise ValueError("at least one return path is required")
    if not isinstance(config, DrawdownRiskConfig):
        raise TypeError("config must be a DrawdownRiskConfig")
    modes = dict(evaluation_modes or {})

    reference_index: pd.Index = pd.Index([])
    cleaned: dict[str, np.ndarray] = {}
    for position, (name, series) in enumerate(paths.items()):
        if not isinstance(series, pd.Series):
            raise TypeError(f"path {name!r} must be a pandas Series")
        values = series.dropna()
        if values.empty:
            raise ValueError(f"path {name!r} is empty")
        if position == 0:
            reference_index = values.index
        elif not values.index.equals(reference_index):
            raise ValueError(
                "common random numbers require identical indexes; "
                f"{name!r} has {len(values)} sessions against "
                f"{len(reference_index)}"
            )
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"path {name!r} must be finite")
        cleaned[name] = array

    source_length = int(len(reference_index))
    indices = stationary_drawdown_indices(source_length, config)
    lower, upper = config.breach_thresholds

    rows: list[dict[str, Any]] = []
    for name, array in cleaned.items():
        worst = np.empty(config.samples, dtype=float)
        for begin in range(0, config.samples, config.batch_size):
            stop = min(begin + config.batch_size, config.samples)
            block = np.cumprod(1.0 + array[indices[begin:stop]], axis=1)
            peaks = np.maximum.accumulate(np.maximum(block, 1.0), axis=1)
            worst[begin:stop] = (block / peaks - 1.0).min(axis=1)
        rows.append(
            {
                "variant": name,
                "evaluation_mode": modes.get(name, ""),
                "method": (
                    "stationary block bootstrap of daily net returns; one "
                    "shared index draw across variants"
                ),
                "samples": int(config.samples),
                "expected_block_sessions": int(config.block_length_sessions),
                "horizon_sessions": int(config.horizon_sessions),
                "seed": int(config.seed),
                "source_sessions": source_length,
                "common_random_numbers": True,
                f"bootstrap_p_drawdown_breach_{int(round(lower * 100))}pct": float(
                    (worst < -lower).mean()
                ),
                f"bootstrap_p_drawdown_breach_{int(round(upper * 100))}pct": float(
                    (worst < -upper).mean()
                ),
                "bootstrap_p01_max_drawdown": float(np.percentile(worst, 1)),
                "bootstrap_p05_max_drawdown": float(np.percentile(worst, 5)),
                "bootstrap_median_max_drawdown": float(np.median(worst)),
                "bootstrap_mean_max_drawdown": float(worst.mean()),
                "selection_adjusted": False,
                "permitted_use": _PERMITTED_USE,
            }
        )
    return pd.DataFrame(rows, columns=drawdown_risk_columns(config))


def next_risk_budget_volatility(
    history: Sequence[tuple[float, float]],
    reference: float,
    match: RiskMatchConfig,
    *,
    anchor: float,
) -> float:
    """The next risk budget to try, given every (budget, volatility) so far.

    One observation gives the first-order rescale, which assumes realised
    volatility is proportional to the budget.  It is close but not exact,
    because the risk-scalar clip, integer contracts and the gross-notional
    limit all respond to the level.  Two or more observations switch to a
    secant step, which does not make that assumption.  The proposal is clamped
    to a band around the incumbent's budget so a degenerate slope cannot ask
    the engine for an absurd risk level.
    """

    if not history:
        raise ValueError("history must contain at least one observation")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference must be finite and positive")
    if not np.isfinite(anchor) or anchor <= 0:
        raise ValueError("anchor must be finite and positive")

    last_budget, last_volatility = history[-1]
    if not np.isfinite(last_volatility) or last_volatility <= 0:
        raise ValueError("the most recent realised volatility must be positive")

    proposal = last_budget * reference / last_volatility
    if len(history) >= 2:
        previous_budget, previous_volatility = history[-2]
        slope_denominator = last_budget - previous_budget
        if abs(slope_denominator) > 1e-12:
            slope = (last_volatility - previous_volatility) / slope_denominator
            if np.isfinite(slope) and slope > 1e-9:
                proposal = last_budget + (reference - last_volatility) / slope
    return float(
        min(
            max(proposal, anchor * match.min_multiplier),
            anchor * match.max_multiplier,
        )
    )


def _probe_label(fraction: float) -> str:
    sign = "minus" if fraction < 0 else "plus"
    return f"{sign}{abs(float(fraction)):.2f}tol"


def risk_match_control_budgets(
    *,
    anchor_budget: float,
    anchor_volatility: float,
    reference: float,
    match: RiskMatchConfig,
) -> tuple[tuple[str, float], ...]:
    """Risk budgets that land at named positions inside the match tolerance.

    The solver stops at the first budget whose realised volatility is within
    ``tolerance`` of the reference, so every budget in that band is an equally
    admissible answer to "match the incumbent's risk".  These are the budgets
    at the band's named positions, obtained from the same first-order relation
    the solver's opening step uses.  Re-running the UNCHANGED incumbent
    configuration at them measures how far the matched loss-shape statistics
    move for no reason other than where the solver happened to stop.

    Returns ``(label, budget)`` pairs, empty when the control is disabled.
    """

    for name, value in (
        ("anchor_budget", anchor_budget),
        ("anchor_volatility", anchor_volatility),
        ("reference", reference),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(match, RiskMatchConfig):
        raise TypeError("match must be a RiskMatchConfig")

    probes: list[tuple[str, float]] = []
    for fraction in match.control_probe_fractions:
        wanted = reference + float(fraction) * match.tolerance
        if wanted <= 0:
            raise ValueError(
                "a control probe asks for a non-positive volatility; the "
                "tolerance is not smaller than the reference volatility"
            )
        budget = anchor_budget * wanted / anchor_volatility
        probes.append((_probe_label(fraction), float(budget)))
    return tuple(probes)


def risk_match_control_variant(
    label: str,
    *,
    risk_budget_volatility: float,
) -> levers.LeverVariant:
    """The incumbent configuration re-run at one in-tolerance risk budget."""

    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if not np.isfinite(risk_budget_volatility) or risk_budget_volatility <= 0:
        raise ValueError("risk_budget_volatility must be finite and positive")
    return levers.LeverVariant(
        name=f"baseline_match_control_{label}",
        lever=MATCH_CONTROL_LEVER,
        evaluation_mode=MATCH_CONTROL_MODE,
        assumption=(
            "No bound changed.  The canonical configuration re-run at risk "
            f"budget {risk_budget_volatility:.6f}, which the match tolerance "
            "accepts, so this row's loss-shape delta is solver jitter and "
            "nothing else."
        ),
        overrides=(("target_vol", float(risk_budget_volatility)),),
    )


def match_jitter_report(
    observations: Sequence[tuple[str, float, float, Mapping[str, float]]],
    *,
    tolerance: float,
) -> pd.DataFrame:
    """Spread of each loss-shape statistic across the in-tolerance ladder.

    ``observations`` are ``(variant, risk budget, volatility gap, loss-shape
    metrics)`` for the anchor and every control probe.  Probes whose realised
    gap fell outside the tolerance are excluded, because a band measured over a
    wider spread of volatilities than the solver would have accepted would
    overstate the jitter; the exclusion is recorded rather than assumed.

    The band is a range across a handful of runs, not a standard error.  It
    says how far the statistic moves between admissible matches; it does not
    describe the sampling distribution, which ``loss_shape_resolution``
    measures separately.
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    retained = [
        record
        for record in observations
        if np.isfinite(record[2]) and abs(record[2]) <= tolerance
    ]
    covered = max((abs(record[2]) for record in retained), default=float("nan"))
    method = (
        "range across re-runs of the unchanged incumbent configuration at "
        "risk budgets the match tolerance accepts"
    )
    rows: list[dict[str, Any]] = []
    for statistic in RESOLUTION_STATISTICS:
        levels = [
            float(record[3][statistic])
            for record in retained
            if np.isfinite(record[3].get(statistic, np.nan))
        ]
        estimable = len(retained) >= 2 and len(levels) == len(retained)
        rows.append(
            {
                "statistic": statistic,
                "method": method,
                "band_status": ESTIMATED if estimable else NOT_ESTIMABLE,
                "band_basis": (
                    (
                        f"{len(retained)} in-tolerance runs of the canonical "
                        f"configuration spanning residuals up to {covered:.2e} "
                        f"of the {float(tolerance):.2e} the solver accepts; a "
                        "span narrower than the tolerance understates the band"
                    )
                    if estimable
                    else (
                        "fewer than two in-tolerance runs of the canonical "
                        "configuration were available, so the spread the "
                        "tolerance admits was not measured"
                    )
                ),
                "probe_variants": "; ".join(record[0] for record in retained),
                "probes_inside_tolerance": len(retained),
                "lowest_risk_budget_volatility": (
                    min(record[1] for record in retained) if retained else np.nan
                ),
                "highest_risk_budget_volatility": (
                    max(record[1] for record in retained) if retained else np.nan
                ),
                "lowest_achieved_annualized_volatility": (
                    min(
                        float(record[3]["annualized_volatility"])
                        for record in retained
                    )
                    if retained
                    else np.nan
                ),
                "highest_achieved_annualized_volatility": (
                    max(
                        float(record[3]["annualized_volatility"])
                        for record in retained
                    )
                    if retained
                    else np.nan
                ),
                "widest_volatility_match_error": (
                    max(abs(record[2]) for record in retained) if retained else np.nan
                ),
                "tolerance": float(tolerance),
                "lowest_observed": min(levels) if estimable else np.nan,
                "highest_observed": max(levels) if estimable else np.nan,
                "solver_jitter_band": (
                    max(levels) - min(levels) if estimable else np.nan
                ),
                "selection_adjusted": False,
                "permitted_use": _PERMITTED_USE,
            }
        )
    return pd.DataFrame(rows, columns=BOUNDS_MATCH_JITTER_COLUMNS)


def loss_shape_resolution(
    incumbent: pd.Series,
    challengers: Mapping[str, pd.Series],
    *,
    config: LossShapeResolutionConfig = LossShapeResolutionConfig(),
    annualization: int = 252,
    evaluation_modes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Sampling distribution of the loss-shape differentials, paired and shared.

    The three statistics ``performance_metrics`` never gave an interval to are
    exactly the ones the matched rows are read on, so without this the report
    would publish a drawdown-shape improvement with no way to tell it from
    noise while publishing a Sharpe differential with one.  One stationary
    block-bootstrap index path is drawn per replicate and applied to the
    incumbent and to every challenger, so the shared market history cancels in
    the differential and the floors are comparable across variants.

    The index draw is seeded from the block length and the path length only.
    No comparison label enters it, which is deliberate and is the difference
    from ``inference.paired_block_bootstrap_differential``: there the label
    decorrelates independent comparisons, here every comparison must see the
    same resampled histories or the floors could not be compared.
    """

    if not isinstance(config, LossShapeResolutionConfig):
        raise TypeError("config must be a LossShapeResolutionConfig")
    if not isinstance(incumbent, pd.Series):
        raise TypeError("incumbent must be a pandas Series")
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, int)
        or annualization < 1
    ):
        raise ValueError("annualization must be a positive integer")

    base_series = incumbent.dropna()
    if base_series.empty:
        raise ValueError("the incumbent path is empty")
    base = base_series.to_numpy(dtype=float)
    if not np.isfinite(base).all():
        raise ValueError("the incumbent path must be finite")
    count = int(base.size)
    if count < 2:
        raise ValueError("the incumbent path must contain at least two sessions")

    cleaned: dict[str, np.ndarray] = {}
    for name, series in challengers.items():
        if not isinstance(series, pd.Series):
            raise TypeError(f"path {name!r} must be a pandas Series")
        values = series.dropna()
        if not values.index.equals(base_series.index):
            raise ValueError(
                "a paired resolution estimate requires identical indexes; "
                f"{name!r} has {len(values)} sessions against {count}"
            )
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"path {name!r} must be finite")
        cleaned[name] = array

    modes = dict(evaluation_modes or {})
    observed_base = _loss_shape_kernel(base, annualization)
    method_seed = (
        zlib.crc32(b"paired loss-shape stationary block bootstrap") & 0xFFFFFFFF
    )

    rows: list[dict[str, Any]] = []
    for block_length in config.block_lengths:
        entropy = np.random.SeedSequence(
            [
                int(config.seed) & 0xFFFFFFFF,
                method_seed,
                int(block_length),
                count,
            ]
        )
        rng = np.random.default_rng(entropy)
        indices = inference._stationary_indices(
            rng, count, config.samples, count, int(block_length)
        )
        base_draws = _bootstrap_loss_shape(base, indices, config, annualization)
        for name, values in cleaned.items():
            draws = _bootstrap_loss_shape(values, indices, config, annualization)
            observed = _loss_shape_kernel(values, annualization)
            correlation = float(np.corrcoef(base, values)[0, 1])
            for statistic in RESOLUTION_STATISTICS:
                distribution = draws[statistic] - base_draws[statistic]
                distribution = distribution[np.isfinite(distribution)]
                estimate = float(observed[statistic]) - float(
                    observed_base[statistic]
                )
                standard_error = (
                    float(distribution.std(ddof=1))
                    if distribution.size >= 2
                    else float("nan")
                )
                # A zero standard error is what an inert variant really produces,
                # and it is exactly the number that must not be published: as a
                # minimum detectable effect it reads as infinite power, and as a
                # floor it would let a delta of zero count as resolved.
                if not np.isfinite(standard_error) or standard_error <= 0.0:
                    rows.append(
                        _resolution_row(
                            comparison=name,
                            evaluation_mode=modes.get(name, ""),
                            statistic=statistic,
                            block_length=int(block_length),
                            config=config,
                            sessions=count,
                            correlation=correlation,
                            estimate=estimate,
                            standard_error=float("nan"),
                            lower=float("nan"),
                            upper=float("nan"),
                            confidence_method=(
                                "not estimable: the resampled differential has "
                                "no sampling variation"
                                if distribution.size >= 2
                                else "not estimable: the resampled differential "
                                "produced fewer than two finite draws"
                            ),
                        )
                    )
                    continue
                percentile_lower = float(np.percentile(distribution, 5.0))
                percentile_upper = float(np.percentile(distribution, 95.0))
                rows.append(
                    _resolution_row(
                        comparison=name,
                        evaluation_mode=modes.get(name, ""),
                        statistic=statistic,
                        block_length=int(block_length),
                        config=config,
                        sessions=count,
                        correlation=correlation,
                        estimate=estimate,
                        standard_error=standard_error,
                        lower=min(
                            percentile_lower, 2.0 * estimate - percentile_upper
                        ),
                        upper=max(
                            percentile_upper, 2.0 * estimate - percentile_lower
                        ),
                        confidence_method=(
                            "minimum of percentile and basic bootstrap bounds"
                        ),
                    )
                )
    return pd.DataFrame(rows, columns=BOUNDS_RESOLUTION_COLUMNS)


def _bootstrap_loss_shape(
    values: np.ndarray,
    indices: np.ndarray,
    config: LossShapeResolutionConfig,
    annualization: int,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        statistic: [] for statistic in RESOLUTION_STATISTICS
    }
    for begin in range(0, indices.shape[0], config.batch_size):
        block = values[indices[begin : begin + config.batch_size]]
        kernel = _loss_shape_kernel(block, annualization)
        for statistic in RESOLUTION_STATISTICS:
            collected[statistic].append(kernel[statistic])
    return {
        statistic: np.concatenate(batches)
        for statistic, batches in collected.items()
    }


def _resolution_row(
    *,
    comparison: str,
    evaluation_mode: str,
    statistic: str,
    block_length: int,
    config: LossShapeResolutionConfig,
    sessions: int,
    correlation: float,
    estimate: float,
    standard_error: float,
    lower: float,
    upper: float,
    confidence_method: str,
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "evaluation_mode": evaluation_mode,
        "statistic": statistic,
        "method": (
            "paired stationary block bootstrap of the loss-shape "
            "differential; one shared index draw across comparisons"
        ),
        "expected_block_sessions": int(block_length),
        "samples": int(config.samples),
        "seed": int(config.seed),
        "sessions": int(sessions),
        "common_random_numbers": True,
        "return_correlation_with_incumbent": correlation,
        "point_estimate": estimate,
        "bootstrap_standard_error": standard_error,
        "confidence_lower": lower,
        "confidence_upper": upper,
        "confidence_method": confidence_method,
        "minimum_detectable_effect_95_one_sided": (
            config.confidence_z * standard_error
        ),
        "selection_adjusted": False,
        "permitted_use": _PERMITTED_USE,
    }


def resolution_floors(
    resolution: pd.DataFrame,
    jitter: pd.DataFrame,
    *,
    solver_exposed_modes: Sequence[str] = (RISK_MATCHED_MODE,),
) -> pd.DataFrame:
    """The floor each variant's loss-shape delta has to clear, and why.

    The sampling floor is the widest minimum detectable effect across the
    declared block-length grid, so the friendliest dependence assumption cannot
    be chosen after the fact.  The solver floor applies only to rows whose risk
    budget was moved by the match, because only those rows inherit the
    dispersion the tolerance admits.  The reported floor is the larger of the
    two: they are different sources of error and neither excuses the other.
    """

    bands = {
        str(row["statistic"]): float(row["solver_jitter_band"])
        for _, row in jitter.iterrows()
    }
    rows: list[dict[str, Any]] = []
    if resolution.empty:
        return pd.DataFrame(
            rows,
            columns=[
                "comparison",
                "evaluation_mode",
                "statistic",
                "sampling_floor",
                "solver_jitter_band",
                "resolution_floor",
            ],
        )
    grouped = resolution.groupby(["comparison", "evaluation_mode", "statistic"])
    for (comparison, mode, statistic), block in grouped:
        detectable = pd.to_numeric(
            block["minimum_detectable_effect_95_one_sided"], errors="coerce"
        )
        # One non-estimable block length makes the whole sampling floor
        # non-estimable: taking the max over the blocks that did resolve would
        # quietly substitute the friendliest dependence assumption for a
        # missing one.
        sampling = (
            float(detectable.max()) if detectable.notna().all() else float("nan")
        )
        band = bands.get(str(statistic), float("nan"))
        applicable = band if str(mode) in tuple(solver_exposed_modes) else float("nan")
        candidates = [
            value for value in (sampling, applicable) if np.isfinite(value)
        ]
        rows.append(
            {
                "comparison": comparison,
                "evaluation_mode": mode,
                "statistic": statistic,
                "sampling_floor": sampling,
                "solver_jitter_band": applicable,
                "resolution_floor": max(candidates) if candidates else float("nan"),
            }
        )
    return pd.DataFrame(rows)


class _EngineRunner:
    """Runs each configuration once, pinning only the baseline result.

    A ``BacktestResult`` is roughly 35 MB, so caching the whole sweep would
    cost more than a gigabyte for no benefit: ``run_lever_comparison`` reduces
    each result to a handful of summary rows and drops it.  The observer hook
    extracts what this module needs on the way past, which keeps peak memory
    at the pinned baseline plus the variant in flight.
    """

    def __init__(
        self,
        run: Callable[[Any], Any],
        *,
        pinned_config: Any,
        observer: Callable[[Any, Any], None],
    ) -> None:
        self._run = run
        self._pinned_config = pinned_config
        self._observer = observer
        self._pinned: Any = None

    def __call__(self, config: Any) -> Any:
        if self._pinned is not None and config == self._pinned_config:
            return self._pinned
        output = self._run(config)
        self._observer(config, output)
        if config == self._pinned_config:
            self._pinned = output
        return output


def _split_output(output: Any) -> tuple[Any, pd.Series | None]:
    if isinstance(output, tuple) and len(output) == 2:
        return output[0], output[1]
    return output, None


def _refused_paired_report(
    *,
    comparison: str,
    reason: str,
    block_lengths: Sequence[int],
    samples: int,
    sessions: int,
    correlation: float,
    tracking_error: float,
    point_estimate: float,
) -> pd.DataFrame:
    """A paired-differential frame that states why it has no interval.

    Same schema as ``inference.paired_block_bootstrap_differential`` so the two
    concatenate, but every sampling column is blank.  A degenerate pair really
    does have a zero standard error and a zero-width interval, and publishing
    those numbers would make the least informative comparison in the panel read
    as the best powered one.  What is observable — the correlation, the tracking
    error and the point estimate — is still reported.
    """

    rows: list[dict[str, Any]] = []
    for block_length in block_lengths:
        for statistic in (
            "Annualized mean differential",
            "Sharpe differential",
            "CAGR differential",
        ):
            rows.append(
                {
                    "Comparison": comparison,
                    "Statistic": statistic,
                    "Method": f"paired bootstrap not estimable: {reason}",
                    "Expected block sessions": int(block_length),
                    "Samples": int(samples),
                    "Seed": int(inference.DEFAULT_PAIRED_SEED),
                    "Sessions": int(sessions),
                    "Return correlation with incumbent": correlation,
                    "Tracking error (annualized)": tracking_error,
                    "Point estimate": point_estimate,
                    "Bootstrap standard error": np.nan,
                    "Confidence lower": np.nan,
                    "Confidence upper": np.nan,
                    "Confidence method": "not estimable",
                    "Minimum detectable effect (95% one-sided)": np.nan,
                    "Selection adjusted": False,
                    "Permitted use": _PERMITTED_USE,
                }
            )
    return pd.DataFrame(rows, columns=inference.PAIRED_REPORT_COLUMNS)


def _metric(metrics: pd.Series, name: str) -> float:
    if name not in metrics:
        raise KeyError(f"metrics are missing {name!r}")
    return float(metrics[name])


def _shape_row(
    *,
    variant: levers.LeverVariant,
    observation: dict[str, Any],
    reference: float,
    solver: dict[str, Any],
) -> dict[str, Any]:
    metrics = observation["metrics"]
    shape = observation["shape"]
    row: dict[str, Any] = {
        "variant": variant.name,
        "lever": variant.lever,
        "evaluation_mode": variant.evaluation_mode,
        "bound_setting": (
            "; ".join(f"{key}={value:g}" for key, value in variant.overrides)
            or "canonical"
        ),
        "validation_status": BOUNDS_VALIDATION_STATUS,
        "risk_budget_volatility": observation["risk_budget_volatility"],
        "reference_annualized_volatility": reference,
        "annualized_volatility_gap_to_reference": (
            shape["annualized_volatility"] - reference
        ),
        "solver_engine_runs": solver.get("solver_engine_runs", 0),
        "start": metrics.get("Start"),
        "end": metrics.get("End"),
        "years": _metric(metrics, "Years"),
        "annualized_volatility": shape["annualized_volatility"],
        "cagr": _metric(metrics, "CAGR"),
        "sharpe": _metric(metrics, "Naive daily Sharpe (sqrt252, rf=0)"),
        "hac_sharpe": _metric(metrics, "HAC Sharpe (21 lags, rf=0)"),
        "sortino": _metric(metrics, "Sortino (rf=0)"),
        "calmar": _metric(metrics, "Calmar"),
        "max_drawdown": shape["max_drawdown"],
        "max_drawdown_duration_sessions": _metric(
            metrics, "Max drawdown duration (sessions)"
        ),
        "max_drawdown_over_annualized_volatility": shape[
            "max_drawdown_over_annualized_volatility"
        ],
        "ulcer_index": shape["ulcer_index"],
        "pain_index": shape["pain_index"],
        "worst_rolling_21_sessions": shape["worst_rolling_21_sessions"],
        "worst_rolling_63_sessions": shape["worst_rolling_63_sessions"],
        "conditional_shortfall_95": shape["conditional_shortfall_95"],
        "conditional_shortfall_99": shape["conditional_shortfall_99"],
        "annualized_downside_deviation": _metric(
            metrics, "Annualized downside deviation"
        ),
        "daily_return_skew": _metric(metrics, "Daily return skew"),
        "daily_return_excess_kurtosis": _metric(metrics, "Daily return excess kurtosis"),
        "average_risk_scalar": _metric(metrics, "Average risk scalar"),
        "peak_risk_scalar": observation["peak_risk_scalar"],
        "trough_risk_scalar": observation["trough_risk_scalar"],
        "average_gross_notional_multiple": _metric(
            metrics, "Average gross notional multiple"
        ),
        "peak_gross_notional_multiple": _metric(metrics, "Peak gross notional multiple"),
        "portfolio_limit_binding_decision_days": _metric(
            metrics, "Portfolio-limit binding decision days"
        ),
        "annual_cost_drag": _metric(metrics, "Annual cost drag"),
        "annual_rebalance_turnover_contracts": observation["turnover"][
            "annual_rebalance_turnover_contracts"
        ],
        "annual_roll_turnover_contracts": observation["turnover"][
            "annual_roll_turnover_contracts"
        ],
    }
    return row


def _attach_shape_deltas(
    shape: pd.DataFrame,
    *,
    floors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Differences against the baseline row, refused below the measured floor.

    A loss-shape delta smaller than the resolution floor is emitted as
    ``not_estimable`` with the floor beside it rather than as a number, which is
    ``deflated_sharpe_ratio``'s shape applied to a difference: reporting the
    figure would invite a reading the data cannot support.  The *level* the
    delta was formed from stays in the row, so the refusal withholds a claim,
    not the evidence.
    """

    baseline = shape.iloc[0]
    for delta, source in (
        ("delta_annualized_volatility", "annualized_volatility"),
        ("delta_cagr", "cagr"),
        ("delta_sharpe", "sharpe"),
        ("delta_max_drawdown", "max_drawdown"),
        (
            "delta_max_drawdown_over_annualized_volatility",
            "max_drawdown_over_annualized_volatility",
        ),
        ("delta_ulcer_index", "ulcer_index"),
        ("delta_annual_cost_drag", "annual_cost_drag"),
    ):
        shape[delta] = pd.to_numeric(shape[source], errors="coerce") - float(
            baseline[source]
        )

    lookup: dict[tuple[str, str], float] = {}
    if floors is not None and not floors.empty:
        lookup = {
            (str(row["comparison"]), str(row["statistic"])): float(
                row["resolution_floor"]
            )
            for _, row in floors.iterrows()
        }
    baseline_name = str(baseline["variant"])
    for statistic in RESOLUTION_STATISTICS:
        floor_column = f"{statistic}_resolution_floor"
        status_column = f"delta_{statistic}_status"
        delta_column = f"delta_{statistic}"
        floor_values: list[float] = []
        statuses: list[str] = []
        deltas: list[float] = []
        for _, row in shape.iterrows():
            name = str(row["variant"])
            delta = float(pd.to_numeric(row[delta_column], errors="coerce"))
            if name == baseline_name:
                floor_values.append(float("nan"))
                statuses.append(REFERENCE_ROW)
                deltas.append(delta)
                continue
            floor = lookup.get((name, statistic), float("nan"))
            floor_values.append(floor)
            if not np.isfinite(floor) or abs(delta) <= floor:
                statuses.append(NOT_ESTIMABLE)
                deltas.append(float("nan"))
            else:
                statuses.append(ESTIMATED)
                deltas.append(delta)
        shape[floor_column] = floor_values
        shape[status_column] = statuses
        shape[delta_column] = deltas

    for column in BOUNDS_SHAPE_COLUMNS:
        if column not in shape:
            shape[column] = np.nan
    return shape.loc[:, BOUNDS_SHAPE_COLUMNS]


def run_bounds_sweep(
    config: Any,
    *,
    grid: BoundGrid = BoundGrid(),
    frames: Mapping[str, pd.DataFrame] | None = None,
    run_fn: Callable[[Any], Any] | None = None,
    match: RiskMatchConfig = RiskMatchConfig(),
    risk: DrawdownRiskConfig = DrawdownRiskConfig(),
    start: str = levers.RETROSPECTIVE_START,
    end: str = levers.RETROSPECTIVE_END,
    manifest_path: Path | None = None,
    paired_samples: int = 2_000,
    include_risk_matched: bool = True,
    resolution: LossShapeResolutionConfig = LossShapeResolutionConfig(),
) -> BoundsSweepArtifacts:
    """Sweep the magnitude bounds as-is and at matched realised volatility.

    The as-is pass runs first because the risk-matched configurations cannot be
    written down until the as-is realised volatilities exist.  Each subsequent
    pass re-proposes a risk budget only for the variants still outside the
    match tolerance, so a variant that lands on the first rescale costs one
    extra backtest rather than ``max_iterations``.

    Between the two passes the control ladder runs: the canonical configuration
    re-executed at the tolerance band's edges, which is what turns the matched
    deltas from unqualified numbers into numbers with a measured floor beneath
    them.  It costs one engine run per declared probe.
    """

    from .strategy import performance_metrics, run_backtest

    if not isinstance(grid, BoundGrid):
        raise TypeError("grid must be a BoundGrid")
    if not isinstance(match, RiskMatchConfig):
        raise TypeError("match must be a RiskMatchConfig")
    if not isinstance(risk, DrawdownRiskConfig):
        raise TypeError("risk must be a DrawdownRiskConfig")
    if not isinstance(resolution, LossShapeResolutionConfig):
        raise TypeError("resolution must be a LossShapeResolutionConfig")
    if paired_samples < 1:
        raise ValueError("paired_samples must be positive")

    # Fixed before a single engine run, and validated here so a grid that
    # cannot support the declared panel fails in a second rather than after an
    # hour of backtests.
    declared = declared_paired_comparisons(config, grid)

    annualization = getattr(config, "annualization", 252)
    frame_kwargs = dict(frames) if frames is not None else {}

    def default_run(variant_config: Any) -> Any:
        return run_backtest(variant_config, **frame_kwargs)

    base_run = default_run if run_fn is None else run_fn
    observations: dict[Any, dict[str, Any]] = {}

    def observe(variant_config: Any, output: Any) -> None:
        result, supplied = _split_output(output)
        metrics = (
            supplied
            if isinstance(supplied, pd.Series)
            else performance_metrics(result, start, end, annualization)
        )
        daily = getattr(result, "daily", None)
        if not isinstance(daily, pd.DataFrame) or "net_return" not in daily:
            raise ValueError("a bound sweep requires a daily net-return ledger")
        path = daily.loc[start:end, "net_return"].dropna()
        scalar = daily.loc[start:end, "risk_scalar"] if "risk_scalar" in daily else None
        observations[variant_config] = {
            "metrics": metrics,
            "path": path,
            "shape": drawdown_shape_metrics(path, annualization=annualization),
            "turnover": levers._turnover_diagnostics(result, start, end),
            "risk_budget_volatility": float(variant_config.target_vol),
            "peak_risk_scalar": (
                float(scalar.max()) if scalar is not None else float("nan")
            ),
            "trough_risk_scalar": (
                float(scalar.min()) if scalar is not None else float("nan")
            ),
            "binding": bound_binding_diagnostic(
                result,
                variant_config,
                frames=frames,
                start=start,
                end=end,
            ),
        }

    runner = _EngineRunner(base_run, pinned_config=config, observer=observe)
    variants = build_bound_variants(config, grid)

    comparison, monthly = levers.run_lever_comparison(
        config,
        variants=variants,
        run_fn=runner,
        start=start,
        end=end,
        manifest_path=manifest_path,
    )

    baseline_observation = observations[config]
    reference = (
        float(match.reference_volatility)
        if match.reference_volatility is not None
        else baseline_observation["shape"]["annualized_volatility"]
    )

    ordered: list[levers.LeverVariant] = list(variants)
    # Engine runs spent solving the match, counted against the risk-matched row
    # only: the as-is row costs one run and owes the solver nothing.
    solver_runs: dict[str, int] = {}
    match_rows: list[dict[str, Any]] = []
    matched_variants: dict[str, levers.LeverVariant] = {}
    control_variants: list[levers.LeverVariant] = []
    jitter_observations: list[tuple[str, float, float, Mapping[str, float]]] = [
        (
            variants[0].name,
            baseline_observation["risk_budget_volatility"],
            baseline_observation["shape"]["annualized_volatility"] - reference,
            baseline_observation["shape"],
        )
    ]

    if include_risk_matched and match.control_probe_fractions:
        probes = risk_match_control_budgets(
            anchor_budget=float(config.target_vol),
            anchor_volatility=baseline_observation["shape"][
                "annualized_volatility"
            ],
            reference=reference,
            match=match,
        )
        control_variants = [
            risk_match_control_variant(label, risk_budget_volatility=budget)
            for label, budget in probes
        ]
        control_comparison, control_monthly = levers.run_lever_comparison(
            config,
            variants=[variants[0], *control_variants],
            run_fn=runner,
            start=start,
            end=end,
        )
        comparison = pd.concat(
            [comparison, control_comparison.iloc[1:]], ignore_index=True
        )
        monthly = pd.concat(
            [monthly, control_monthly.drop(columns=[variants[0].name], errors="ignore")],
            axis=1,
        )
        for control in control_variants:
            observation = observations[control.apply(config)]
            jitter_observations.append(
                (
                    control.name,
                    observation["risk_budget_volatility"],
                    observation["shape"]["annualized_volatility"] - reference,
                    observation["shape"],
                )
            )

    match_jitter = match_jitter_report(
        jitter_observations, tolerance=match.tolerance
    )

    if include_risk_matched:
        histories: dict[str, list[tuple[float, float]]] = {}
        pending: list[levers.LeverVariant] = []
        for variant in variants:
            variant_config = variant.apply(config)
            observation = observations[variant_config]
            histories[variant.name] = [
                (
                    observation["risk_budget_volatility"],
                    observation["shape"]["annualized_volatility"],
                )
            ]
            pending.append(variant)

        for iteration in range(1, match.max_iterations + 1):
            proposals = {
                variant.name: next_risk_budget_volatility(
                    histories[variant.name],
                    reference,
                    match,
                    anchor=float(config.target_vol),
                )
                for variant in pending
            }
            candidates = [
                risk_matched_variant(
                    variant, risk_budget_volatility=proposals[variant.name]
                )
                for variant in pending
            ]
            matched_comparison, matched_monthly = levers.run_lever_comparison(
                config,
                variants=[variants[0], *candidates],
                run_fn=runner,
                start=start,
                end=end,
            )
            matched_comparison = matched_comparison.iloc[1:]
            matched_monthly = matched_monthly.drop(
                columns=[variants[0].name], errors="ignore"
            )

            still_pending: list[levers.LeverVariant] = []
            for variant, candidate in zip(pending, candidates):
                candidate_config = candidate.apply(config)
                observation = observations[candidate_config]
                achieved = observation["shape"]["annualized_volatility"]
                error = achieved - reference
                histories[variant.name].append(
                    (proposals[variant.name], achieved)
                )
                converged = abs(error) <= match.tolerance
                solver_runs[candidate.name] = (
                    solver_runs.get(candidate.name, 0) + 1
                )
                match_rows.append(
                    {
                        "variant": candidate.name,
                        "source_variant": variant.name,
                        "iteration": iteration,
                        "method": (
                            "first-order rescale of the risk budget"
                            if iteration == 1
                            else "secant step on realised annualised volatility"
                        ),
                        "risk_budget_volatility": proposals[variant.name],
                        "achieved_annualized_volatility": achieved,
                        "reference_annualized_volatility": reference,
                        "volatility_match_error": error,
                        "relative_volatility_match_error": error / reference,
                        "within_tolerance": bool(converged),
                        "tolerance": match.tolerance,
                    }
                )
                matched_variants[variant.name] = candidate
                if not converged:
                    still_pending.append(variant)

            keep = {matched_variants[variant.name].name for variant in pending}
            comparison = pd.concat(
                [
                    comparison[~comparison["variant"].isin(keep)],
                    matched_comparison,
                ],
                ignore_index=True,
            )
            monthly = pd.concat(
                [monthly.drop(columns=list(keep), errors="ignore"), matched_monthly],
                axis=1,
            )
            pending = still_pending
            if not pending:
                break

        ordered = [
            *variants,
            *control_variants,
            *(matched_variants[v.name] for v in variants),
        ]

    order = {variant.name: position for position, variant in enumerate(ordered)}
    comparison = (
        comparison.assign(_order=comparison["variant"].map(order))
        .sort_values("_order", kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    shape_rows: list[dict[str, Any]] = []
    binding_frames: list[pd.DataFrame] = []
    paths: dict[str, pd.Series] = {}
    for variant in ordered:
        variant_config = variant.apply(config)
        observation = observations[variant_config]
        shape_rows.append(
            _shape_row(
                variant=variant,
                observation=observation,
                reference=reference,
                solver={"solver_engine_runs": solver_runs.get(variant.name, 0)},
            )
        )
        binding = observation["binding"].copy()
        binding["variant"] = variant.name
        binding["evaluation_mode"] = variant.evaluation_mode
        binding_frames.append(binding)
        paths[variant.name] = observation["path"]

    modes = {variant.name: variant.evaluation_mode for variant in ordered}
    baseline_path = paths[variants[0].name]
    resolution_report = loss_shape_resolution(
        baseline_path,
        {
            name: path
            for name, path in paths.items()
            if name != variants[0].name
        },
        config=resolution,
        annualization=annualization,
        evaluation_modes=modes,
    )
    floors = resolution_floors(resolution_report, match_jitter)
    shape = _attach_shape_deltas(pd.DataFrame(shape_rows), floors=floors)
    binding_report = (
        pd.concat(binding_frames, ignore_index=True)
        if binding_frames
        else pd.DataFrame(columns=BOUNDS_BINDING_COLUMNS)
    )
    drawdown_risk = common_random_number_drawdown_risk(
        paths,
        config=risk,
        evaluation_modes=modes,
    )

    baseline_volatility = baseline_observation["shape"]["annualized_volatility"]
    baseline_years = _metric(baseline_observation["metrics"], "Years")
    paired_frames: list[pd.DataFrame] = []
    power_rows: list[dict[str, Any]] = []
    for name in declared:
        comparison_label = f"{name}_vs_baseline"
        path = paths.get(name)
        aligned = path is not None and path.index.equals(baseline_path.index)
        # The refusal decisions are taken BEFORE the bootstrap runs.  Calling it
        # first and guarding afterwards is what let the power table refuse a
        # comparison while the paired table published a zero standard error and
        # a zero-width interval for the same comparison — the very reading the
        # refusal exists to prevent.
        if not aligned:
            reason = (
                "the declared comparison was not built by this sweep, so no "
                "path exists to compare"
                if path is None
                else "the declared variant's path does not share the "
                "incumbent's session index"
            )
            paired_frames.append(
                _refused_paired_report(
                    comparison=comparison_label,
                    reason=reason,
                    block_lengths=inference.DEFAULT_BLOCK_LENGTHS,
                    samples=paired_samples,
                    sessions=int(len(baseline_path)),
                    correlation=np.nan,
                    tracking_error=np.nan,
                    point_estimate=np.nan,
                )
            )
            power_rows.append(
                {
                    "comparison": comparison_label,
                    "tracking_error_annualized": np.nan,
                    "selection_adjusted": False,
                    "permitted_use": _PERMITTED_USE,
                    "power_status": POWER_NOT_ESTIMABLE,
                    "power_basis": reason,
                    "standard_error_of_sharpe_differential": np.nan,
                    "standard_error_of_cagr_differential": np.nan,
                    "minimum_detectable_sharpe_effect_95_one_sided": np.nan,
                    "minimum_detectable_cagr_effect_95_one_sided": np.nan,
                }
            )
            continue

        differential = path.to_numpy(dtype=float) - baseline_path.to_numpy(
            dtype=float
        )
        tracking_error = float(
            differential.std(ddof=0) * math.sqrt(annualization)
        )
        common: dict[str, Any] = {
            "comparison": comparison_label,
            "tracking_error_annualized": tracking_error,
            "selection_adjusted": False,
            "permitted_use": _PERMITTED_USE,
        }
        if tracking_error <= 0:
            # An inert bound leaves the path untouched, so there is no
            # differential to detect and no power to report.  Emitting a zero
            # here would read as a perfectly powered comparison, which is the
            # opposite of the truth, so both tables refuse instead of one
            # refusing and the other publishing zeros.
            reason = (
                "the bound left the daily path identical to the incumbent's, "
                "so the differential has no sampling variation and a "
                "detectable effect is undefined"
            )
            paired_frames.append(
                _refused_paired_report(
                    comparison=comparison_label,
                    reason=reason,
                    block_lengths=inference.DEFAULT_BLOCK_LENGTHS,
                    samples=paired_samples,
                    sessions=int(len(baseline_path)),
                    correlation=1.0,
                    tracking_error=0.0,
                    point_estimate=0.0,
                )
            )
            power_rows.append(
                {
                    **common,
                    "power_status": POWER_NOT_ESTIMABLE,
                    "power_basis": reason,
                    "standard_error_of_sharpe_differential": np.nan,
                    "standard_error_of_cagr_differential": np.nan,
                    "minimum_detectable_sharpe_effect_95_one_sided": np.nan,
                    "minimum_detectable_cagr_effect_95_one_sided": np.nan,
                }
            )
            continue
        paired_frames.append(
            inference.paired_block_bootstrap_differential(
                baseline_path,
                path,
                comparison=comparison_label,
                samples=paired_samples,
                annualization=annualization,
            )
        )
        preflight = inference.minimum_detectable_effect(
            tracking_error, baseline_volatility, baseline_years
        )
        power_rows.append(
            {
                **common,
                "power_status": POWER_ESTIMATED,
                "power_basis": (
                    "closed-form standard error of a paired differential over "
                    f"{baseline_years:.2f} years of reused history"
                ),
                "standard_error_of_sharpe_differential": float(
                    preflight["Standard error of Sharpe differential"]
                ),
                "standard_error_of_cagr_differential": float(
                    preflight["Standard error of CAGR differential"]
                ),
                "minimum_detectable_sharpe_effect_95_one_sided": float(
                    preflight["Minimum detectable effect (95% one-sided)"]
                ),
                "minimum_detectable_cagr_effect_95_one_sided": float(
                    preflight["Minimum detectable CAGR effect (95% one-sided)"]
                ),
            }
        )
    paired = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame(columns=inference.PAIRED_REPORT_COLUMNS)
    )
    power = pd.DataFrame(power_rows, columns=BOUNDS_POWER_COLUMNS)
    risk_match = pd.DataFrame(match_rows, columns=BOUNDS_RISK_MATCH_COLUMNS)

    return BoundsSweepArtifacts(
        comparison=comparison,
        shape=shape,
        binding=binding_report,
        drawdown_risk=drawdown_risk,
        risk_match=risk_match,
        match_jitter=match_jitter,
        resolution=resolution_report,
        paired=paired,
        power=power,
        monthly_net_returns=monthly,
    )


__all__ = [
    "AS_IS_MODE",
    "BASELINE_MODE",
    "BOUNDS_BINDING_COLUMNS",
    "BOUNDS_MATCH_JITTER_COLUMNS",
    "BOUNDS_POWER_COLUMNS",
    "BOUNDS_RESOLUTION_COLUMNS",
    "BOUNDS_RISK_MATCH_COLUMNS",
    "BOUNDS_SHAPE_COLUMNS",
    "BOUNDS_VALIDATION_STATUS",
    "ESTIMATED",
    "MATCH_CONTROL_LEVER",
    "MATCH_CONTROL_MODE",
    "NOT_ESTIMABLE",
    "POWER_ESTIMATED",
    "POWER_NOT_ESTIMABLE",
    "REFERENCE_ROW",
    "RESOLUTION_STATISTICS",
    "RISK_MATCHED_MODE",
    "RISK_SCALAR_LEVER",
    "BoundGrid",
    "BoundsSweepArtifacts",
    "DrawdownRiskConfig",
    "LossShapeResolutionConfig",
    "RiskMatchConfig",
    "bound_binding_diagnostic",
    "build_bound_variants",
    "common_random_number_drawdown_risk",
    "declared_paired_comparisons",
    "drawdown_risk_columns",
    "drawdown_shape_metrics",
    "loss_shape_resolution",
    "match_jitter_report",
    "next_risk_budget_volatility",
    "resolution_floors",
    "risk_match_control_budgets",
    "risk_match_control_variant",
    "risk_matched_variant",
    "run_bounds_sweep",
    "stationary_drawdown_indices",
]
