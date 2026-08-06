"""Family-wise validation estimators for the DELTA1 research ledger.

``docs/research-methodology.md`` names three estimators as required and the
package contained none of them: an anchored expanding walk-forward as the
declared primary diagnostic, a family-wise max-statistic procedure (White's
Reality Check or Hansen's SPA) as a promotion gate, and CSCV/PBO as a
secondary selection-instability diagnostic that must read *not estimable*
rather than PASS when the family is incomplete.  This module implements those
three, plus the family object all three consume.

What the module does not do is more important than what it does.

None of these estimators turns reused history into out-of-sample evidence.
The walk-forward replays 1990-2014, which every parameter in the frozen
specification has already seen; its out-of-sample segments are out of sample
only with respect to the *selector*, never with respect to the researcher who
wrote the strategy.  With no selection dimension supplied the procedure has no
selector at all, so it degenerates to a stability report of one frozen
specification across expanding anchors and is labelled as such.  A stability
report is a description of a single path, not validation.

The Reality Check and the SPA correct for multiplicity *within a declared
family* and for nothing else.  They cannot see trials that were run and not
recorded, and a max-statistic computed over a subset of the searched family
understates the correction, so both refuse when a declared member's path is
missing.  Their p-values are also resolution-bound: nothing finer than
``1 / (samples + 1)`` is reportable.

CSCV assumes its columns are the configurations actually searched over.  When
they are not, the in-sample argmax is taken over the wrong set and PBO is
systematically understated, so ``cscv_pbo`` returns NOT_ESTIMABLE rather than
a number whenever the family is incomplete, the paths are unsynchronized, the
number of submatrices is odd or too small, the configuration count is too
small for the rank to be interpretable, or a submatrix is too short to carry a
Sharpe ratio.  That refusal is the mechanical encoding of the methodology
document's "not estimable, never PASS", and it is the point of the estimator
here rather than an edge case in it.

CSCV's per-submatrix length floor follows the family's own frequency, one
calendar year of it, rather than a single session count.  The methodology
document permits this estimator only on monthly paths, and a daily floor
applied to a monthly family refuses every permitted input with a message about
252 monthly observations.  The floor governs the temporal resolution of the
partition, not the estimation sample: the Sharpe ratio CSCV computes is
measured on a concatenation of ``S / 2`` submatrices.

The rest of the implementation choices below are load-bearing.

The walk-forward is a *splice-then-simulate-once* construction.  The engine
carries live book state - NAV compounds, the no-trade buffer holds the
previous executed intent, the participation cap defers residual contracts into
a bounded roll backlog, and the portfolio risk scalar warms up over sixty-three
sessions - so calling the engine once per fold with a fresh initial capital
would reset all four and start every fold over-levered relative to the true
path.  Instead each candidate's monthly decision frame is built over the whole
history, the folds' regimes are spliced into one frame, and the execution
ledger is simulated once.  One NAV path, one executed book, one roll backlog,
one cost ledger, and the turnover of switching regimes at a boundary is charged
because the ledger sees the jump.  The construction is verified by an
elementwise identity: with a constant parameter the spliced replay reproduces
the single-shot backtest exactly.

The splice does not carry everything, and the part it does not carry is stated
rather than glossed.  Execution state crosses a boundary intact because the
ledger is one simulation - but for the same reason the ledger is built once,
from one configuration, so a candidate cannot vary anything the ledger reads.
Cost parameters, participation and capacity caps, leverage and margin limits,
fill timing, integer rounding, initial capital and the launch date are all
refused as selection dimensions rather than accepted and quietly dropped; only
the fields that shape the monthly intent frame are spliceable, and the
allow-list is explicit so a field added to ``StrategyConfig`` later is refused
until someone decides which side of the split it falls on.

State internal to *intent construction* is the second exception:
an incoming specification's no-trade-buffer hysteresis and its EWMA portfolio
risk scalar are computed from that specification's own full-history path, so
the first decision after a switch is buffered against the incoming
specification's counterfactual previous intent rather than against the book the
outgoing regime actually left.  Both quantities live inside ``run_backtest``
and the engine exposes neither, so carrying them would require an engine
change rather than a harness change.  The residual matters in proportion to
how often the selector switches, which the fold table reports.

The Reality Check and the SPA share one set of stationary-bootstrap index
paths, drawn by ``inference._stationary_indices`` under this repository's
``SeedSequence`` plus ``zlib.crc32`` seeding discipline.  Sharing is not a
convenience: the two procedures differ only in studentisation and recentring,
and that claim is only checkable if the resampling underneath them is
identical.

They also refuse together, and in the same direction.  A declared member whose
statistic is undefined - a flat configuration has no Sharpe ratio, and a nearly
flat one has none on some resamples - would otherwise poison the family maximum
with a NaN that compares False against every draw, so the p-value would land on
``1 / (samples + 1)``: an undefined statistic reported as the strongest
rejection the procedure can emit, with a status of ESTIMATED beside it.  Both
procedures therefore refuse and name the offender.  Masking the member out
instead would answer a different question, because a maximum over a silently
shrunk family is exactly what these procedures refuse an incomplete family for.
A member that merely reproduces the benchmark *exactly* is a separate and
benign case: its differential is a finite zero, the Reality Check keeps it, and
only Hansen's studentisation drops it, which is neutral because the studentised
statistic is already truncated at zero.  The count of such members is reported
on every row.

The deflated Sharpe wrapper refuses.  Its status is always NOT_ESTIMABLE,
because the deflation needs the number of trials that were *searched* and the
declared family is the number that were *registered* - a lower bound on this
lineage's search, not an estimate of it.  ``inference.promotion_support`` reads
an estimated deflated Sharpe as clearing the promotion gate, so returning a
number here would clear that gate for every member of a family on the strength
of a trial count nobody can defend.  The arithmetic is still performed and
reported under a name the promotion machinery cannot consume, together with the
Gumbel threshold it is testing against, because the size of the deflation is
informative even when its status is not.  Two conventions inside it are silent
when wrong and are handled rather than delegated: the statistic is defined on
the *per-period* Sharpe, and the published higher-moment coefficient is
``(gamma_4 - 1) / 4`` on non-excess kurtosis.  ``inference`` now forms that
coefficient itself from the excess figure as ``(excess + 2) / 4``, so the
excess kurtosis is passed straight through; the ``gamma_4 - 1`` compensation
this module once applied would double-correct against the corrected function.

Every CAGR here is annualized on elapsed calendar time rather than on
``observations / annualization``.  This session grid holds 260.95 sessions per
calendar year, not 252, so the session-count convention would inflate the
elapsed period by 3.5% and understate the reported CAGR by roughly the same
proportion against ``outputs/strategy_metrics.csv`` for no reason visible in
the artifact.  The convention is named on every row.
"""

from __future__ import annotations

import itertools
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from . import inference
from .inference import (
    DEFAULT_BLOCK_LENGTHS,
    DEFAULT_PAIRED_SEED,
    DEFAULT_SAMPLES,
    ESTIMATED,
    NOT_ESTIMABLE,
    InferenceResult,
)


RETROSPECTIVE_START = "1990-01-01"
RETROSPECTIVE_END = "2014-12-31"

DAILY_FREQUENCY = "daily"
MONTHLY_FREQUENCY = "monthly"

MEAN_STATISTIC = "Annualized mean differential"
SHARPE_STATISTIC = "Sharpe differential"

REALITY_CHECK = "White Reality Check"
SPA_LOWER = "Hansen SPA lower"
SPA_CONSISTENT = "Hansen SPA consistent"
SPA_UPPER = "Hansen SPA upper"
SPA_PROCEDURES = (SPA_LOWER, SPA_CONSISTENT, SPA_UPPER)
ALL_PROCEDURES = (REALITY_CHECK, *SPA_PROCEDURES)

# Bailey, Borwein, Lopez de Prado and Zhu recommend sixteen submatrices and
# report stability roughly over eight to sixteen.  Eight yields only C(8,4)=70
# combinations, so the finest achievable PBO resolution there is 1/70; the
# default is the recommended value and the floor is the paper's.
DEFAULT_SUBMATRICES = 16
MINIMUM_SUBMATRICES = 8
# Below roughly ten configurations the out-of-sample rank is quantised in steps
# of 1/(N+1) coarse enough that PBO measures the grid, not the overfitting.
DEFAULT_MINIMUM_CONFIGURATIONS = 10
# One calendar year of observations per submatrix, at whatever frequency the
# family carries.  The floor is on temporal resolution rather than on
# statistical precision: the Sharpe ratio CSCV actually computes is measured on
# a concatenation of S/2 submatrices, so at S = 16 the estimation sample is
# eight times a submatrix.  A submatrix shorter than a year cannot straddle a
# full seasonal cycle, and below roughly a quarter the per-submatrix path stops
# describing a regime at all.
#
# The frequency-aware form matters because research-methodology.md permits
# CSCV/PBO only on monthly paths.  A single hard-coded 252 refuses every
# monthly family with a message about 252 monthly observations, which is both
# wrong on its face and a refusal of the only input the document allows.
DEFAULT_MINIMUM_ROWS_PER_SUBMATRIX = 252
MINIMUM_ROWS_PER_SUBMATRIX_BY_FREQUENCY = {
    DAILY_FREQUENCY: DEFAULT_MINIMUM_ROWS_PER_SUBMATRIX,
    MONTHLY_FREQUENCY: 12,
}
# The law-of-iterated-logarithm threshold is only meaningful once the sample is
# in the hundreds; log(log(n)) is not even positive below e.
MINIMUM_SPA_OBSERVATIONS = 30

# Replicates per index draw.  Fixed rather than derived from the family size so
# the resampled histories depend only on the seed, the block length and the
# history itself.
_INDEX_CHUNK = 500

WALK_FORWARD_SELECTION_MODE = "anchored_expanding_walk_forward"
WALK_FORWARD_STABILITY_MODE = "frozen_specification_stability"

_HAC_CONVENTION = "inference.hac_standard_error; Bartlett kernel, 1/n scaling"
_CAGR_CONVENTION = (
    "elapsed calendar time from the session before the first evaluated return, "
    "matching strategy.performance_metrics; not sessions/annualization"
)

_WALK_FORWARD_PERMITTED_USE = (
    "chronological replay on reused 1990-2014 history; out of sample with "
    "respect to the selector only, and not with respect to specification "
    "choices already informed by this window"
)
_STABILITY_PERMITTED_USE = (
    "stability of one frozen specification across expanding anchors on reused "
    "history; no selection occurs, so this is description and not validation"
)
_FAMILY_WISE_PERMITTED_USE = (
    "multiplicity correction within one declared family on reused history; "
    "corrects for nothing outside that family and does not license promotion"
)
_CSCV_PERMITTED_USE = (
    "selection-instability diagnostic over the declared configuration matrix; "
    "secondary to the walk-forward and never a promotion decision"
)

WALK_FORWARD_COLUMNS = [
    "label",
    "mode",
    "fold",
    "anchor",
    "training_end",
    "training_sessions",
    "segment_start",
    "segment_end",
    "fold_sessions",
    "fold_decisions",
    "selected_variant",
    "selection_statistic",
    "selection_score",
    "selection_switched",
    "fold_cagr",
    "fold_annualized_volatility",
    "fold_sharpe",
    "fold_hac_sharpe",
    "fold_max_drawdown",
    "fold_hit_rate",
    "hac_convention",
    "cagr_convention",
    "selection_adjusted",
    "permitted_use",
]

FAMILY_WISE_COLUMNS = [
    "Family",
    "Benchmark",
    "Statistic",
    "Procedure",
    "Method",
    "Status",
    "p value",
    "p value resolution",
    "Observed max statistic",
    "Expected block sessions",
    "Samples",
    "Seed",
    "Observations",
    "Family size",
    "Family completeness",
    "Members without bootstrap variance",
    "Reason",
    "Selection adjusted",
    "Permitted use",
]

CSCV_COLUMNS = [
    "Family",
    "Statistic",
    "Method",
    "Status",
    "Value",
    "Submatrices",
    "Combinations",
    "Rows per submatrix",
    "Rows dropped from the front",
    "Configurations",
    "Family completeness",
    "In-sample selection ties",
    "Reason",
    "Selection adjusted",
    "Permitted use",
]

CSCV_SPLIT_COLUMNS = [
    "combination",
    "in_sample_selection",
    "in_sample_sharpe",
    "out_of_sample_sharpe",
    "out_of_sample_rank",
    "relative_rank",
    "logit",
]

# The configuration fields a spliced replay can carry faithfully.
#
# The splice works by taking each candidate's *intent* frame - per-dollar-of-NAV
# target contracts on the monthly decision calendar - and simulating one
# execution ledger over the result.  A field that only shapes those intents is
# therefore carried exactly.  A field the ledger reads is not: the ledger is
# built once, from the base configuration, so a candidate declaring different
# costs, a different participation cap, different leverage limits or different
# fill timing would be *scored* under its own specification and then *replayed*
# under the base one.  Nothing about the resulting fold table would say so, and
# the error is not signed - a cheap candidate replayed at expensive costs reads
# conservative, an expensive one replayed cheap reads flattering.
#
# So the list is an allow-list rather than a deny-list: a field added to
# StrategyConfig later is refused until someone has decided which side of the
# split it falls on.  The complement is enumerated in
# ``_EXECUTION_LEDGER_FIELD_NOTE`` for the error message, and the split is
# checked against the engine source in the tests rather than trusted here.
SPLICEABLE_CONFIG_FIELDS = frozenset(
    {
        # Signal construction.
        "trend_lookback",
        "vol_span",
        "basis_roll_window",
        "basis_lookback",
        "signal_normalization_window",
        "signal_cap",
        "basis_weight",
        "min_median_contracts",
        # The causal volatility-shock taper.
        "fast_vol_span",
        "slow_vol_span",
        "shock_start",
        "shock_full",
        "shock_floor",
        # Position sizing and the portfolio risk scalar.
        "target_vol",
        "risk_budget",
        "vol_decay",
        "vol_estimator_min_periods",
        "portfolio_vol_window",
        "min_risk_scalar",
        "max_risk_scalar",
        "risk_managed_window",
        "risk_managed_cap",
        # The no-trade band.
        "no_trade_buffer",
        "cost_aware_buffer",
        "cost_aware_buffer_floor",
        "cost_aware_buffer_ceiling",
        "buffer_suppresses_risk_scalar",
        "annualization",
    }
)

_EXECUTION_LEDGER_FIELD_NOTE = (
    "cost fields, participation and capacity caps, leverage and margin limits, "
    "fill timing, integer rounding, initial capital and the launch date"
)

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


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """An empty frame that still carries the report's full column contract."""

    return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})


def _seeded_generator(
    seed: int,
    method: str,
    label: str,
    *extra: int,
) -> np.random.Generator:
    """Reproducible generator under the repository's seeding discipline.

    Python's builtin ``hash`` is randomized per process, so a label is folded
    in through ``zlib.crc32`` exactly as ``inference`` and ``diagnostics`` do.
    Every draw in this module is therefore reproducible from one integer.
    """

    method_seed = zlib.crc32(method.encode("utf-8")) & 0xFFFFFFFF
    label_seed = zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF
    entropy = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, method_seed, label_seed, *(int(v) for v in extra)]
    )
    return np.random.default_rng(entropy)


def _max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough fraction, negative by convention.

    Initial capital is the first high-water mark, matching
    ``diagnostics._path_drawdown_statistics`` so drawdowns quoted by the two
    modules are the same quantity.
    """

    if returns.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    return float((equity / peak - 1.0).min())


def _path_statistics(
    returns: pd.Series,
    *,
    annualization: int,
    hac_lags: int,
    period_start: pd.Timestamp | None = None,
) -> dict[str, float]:
    """CAGR, volatility, Sharpe, HAC Sharpe, max drawdown and hit rate.

    CAGR is annualized on **elapsed calendar time**, matching
    ``strategy.performance_metrics``.  ``inference._cagr`` divides by
    ``observations / annualization`` instead, which assumes the path carries
    exactly ``annualization`` observations per year.  This session grid does
    not: 1990-2014 holds 6,523 sessions over 24.997 calendar years, or 260.95
    per year, because the price index is every weekday rather than every
    exchange trading day.  Using the session count would inflate the elapsed
    period by 3.5% and understate every CAGR here by roughly the same
    proportion - 0.1271 against the canonical 0.1319 on the full window - for
    no reason a reader of the artifact could reconstruct.

    ``period_start`` is the timestamp elapsed time is measured *from*, and
    defaults to the first evaluated session.  ``performance_metrics`` anchors
    on the session immediately before the first in-window return, so passing
    that session reproduces its CAGR exactly; passing the previous fold's last
    session additionally makes the folds' elapsed periods tile the stitched
    span without gaps.

    The HAC Sharpe divides by the long-run standard deviation implied by
    ``inference.hac_standard_error``.  That estimator scales autocovariances by
    ``n`` where ``strategy.performance_metrics`` scales by ``n - 1``, so the two
    HAC Sharpes differ in the last digits; the convention is named on every row
    this module emits rather than left for the reader to infer.
    """

    values = returns.to_numpy(dtype=float)
    count = values.size
    if count == 0:
        return dict.fromkeys(
            (
                "cagr",
                "annualized_volatility",
                "sharpe",
                "hac_sharpe",
                "max_drawdown",
                "hit_rate",
            ),
            float("nan"),
        )
    sharpe = float(inference._sharpe(values, annualization))
    anchor = returns.index[0] if period_start is None else pd.Timestamp(period_start)
    years = max(
        (returns.index[-1] - anchor).total_seconds() / (365.2425 * 24 * 60 * 60),
        1.0 / 365.2425,
    )
    cagr = float(np.expm1(np.log1p(values).sum() / years))
    volatility = float(values.std(ddof=0) * math.sqrt(annualization))
    hac_sharpe = float("nan")
    if count > hac_lags:
        standard_error = inference.hac_standard_error(values, hac_lags)
        long_run_sigma = standard_error * math.sqrt(count)
        if np.isfinite(long_run_sigma) and long_run_sigma > 0:
            hac_sharpe = float(
                values.mean() / long_run_sigma * math.sqrt(annualization)
            )
    return {
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "hac_sharpe": hac_sharpe,
        "max_drawdown": _max_drawdown(values),
        "hit_rate": float((values > 0).mean()),
    }


# ---------------------------------------------------------------------------
# The family object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyFamily:
    """A declared family of synchronized causal return paths.

    ``declared_members`` is the family as registered *before* the estimators
    ran.  A max-statistic or an in-sample argmax over anything else is not the
    statistic it claims to be, so the assembler refuses columns that were never
    declared and records declared members whose path is absent instead of
    quietly shrinking the family.
    """

    label: str
    paths: pd.DataFrame
    benchmark: str
    frequency: str
    annualization: int
    declared_members: tuple[str, ...]
    missing_members: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("provenance must be a non-empty string")
        if self.frequency not in {DAILY_FREQUENCY, MONTHLY_FREQUENCY}:
            raise ValueError("frequency must be 'daily' or 'monthly'")
        if (
            isinstance(self.annualization, bool)
            or not isinstance(self.annualization, int)
            or self.annualization < 1
        ):
            raise ValueError("annualization must be a positive integer")
        if self.benchmark not in self.paths.columns:
            raise ValueError("benchmark must be one of the supplied paths")

    @property
    def is_complete(self) -> bool:
        return not self.missing_members

    @property
    def completeness(self) -> str:
        return "complete" if self.is_complete else "incomplete"

    @property
    def size(self) -> int:
        return int(self.paths.shape[1])

    @property
    def observations(self) -> int:
        return int(self.paths.shape[0])

    @property
    def challengers(self) -> tuple[str, ...]:
        return tuple(
            column for column in self.paths.columns if column != self.benchmark
        )

    def incompleteness_reason(self) -> str:
        missing = ", ".join(self.missing_members)
        return (
            f"{len(self.missing_members)} declared member(s) have no "
            f"synchronized path: {missing}"
        )


def assemble_family(
    paths: Mapping[str, pd.Series] | pd.DataFrame,
    *,
    label: str,
    benchmark: str,
    declared_members: Sequence[str],
    provenance: str,
    frequency: str = DAILY_FREQUENCY,
    annualization: int = 252,
) -> StrategyFamily:
    """Validate a set of return paths into one family the estimators can share.

    Synchronization is checked by equality of the whole index rather than by
    intersection.  An intersection would silently answer a different question
    on a family whose members cover different spans, and every estimator
    downstream - the paired resample, the max statistic, the CSCV submatrix
    partition - assumes one common clock.

    A column that was never declared raises.  Adding a model to the family
    after seeing its result invalidates the max-statistic p-value outright, so
    the family manifest is an input rather than something inferred from what
    happened to be produced.
    """

    if isinstance(paths, pd.DataFrame):
        supplied: dict[str, pd.Series] = {
            str(name): paths[name] for name in paths.columns
        }
    elif isinstance(paths, Mapping):
        supplied = {str(name): series for name, series in paths.items()}
    else:
        raise TypeError("paths must be a mapping of name to Series, or a DataFrame")
    if not supplied:
        raise ValueError("a family needs at least one path")

    declared = tuple(str(name) for name in declared_members)
    if not declared:
        raise ValueError("declared_members must not be empty")
    if len(set(declared)) != len(declared):
        raise ValueError("declared_members must be unique")
    if benchmark not in declared:
        raise ValueError("benchmark must appear in declared_members")

    undeclared = sorted(set(supplied) - set(declared))
    if undeclared:
        raise ValueError(
            "these paths were not in the declared family manifest, so a "
            f"family-wise statistic over them is not defined: {undeclared}"
        )
    if benchmark not in supplied:
        raise ValueError("the benchmark path is missing from the supplied family")

    reference_index: pd.DatetimeIndex | None = None
    cleaned: dict[str, pd.Series] = {}
    for name in declared:
        if name not in supplied:
            continue
        series = supplied[name]
        if not isinstance(series, pd.Series):
            raise TypeError(f"path {name!r} must be a pandas Series")
        series = series.astype(float)
        if series.empty:
            raise ValueError(f"path {name!r} is empty")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"path {name!r} must be indexed by dates")
        if not series.index.is_monotonic_increasing or series.index.has_duplicates:
            raise ValueError(f"path {name!r} must have a sorted, unique index")
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError(f"path {name!r} contains non-finite returns")
        if reference_index is None:
            reference_index = series.index
        elif not series.index.equals(reference_index):
            raise ValueError(
                "family paths are not synchronized on one index; "
                f"{name!r} has {len(series)} observations against "
                f"{len(reference_index)} for the first member"
            )
        cleaned[name] = series

    frame = pd.DataFrame(cleaned)
    missing = tuple(name for name in declared if name not in cleaned)
    return StrategyFamily(
        label=label,
        paths=frame,
        benchmark=benchmark,
        frequency=frequency,
        annualization=annualization,
        declared_members=declared,
        missing_members=missing,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# White's Reality Check and Hansen's SPA
# ---------------------------------------------------------------------------


def _family_bootstrap(
    family: StrategyFamily,
    *,
    statistic: str,
    block_length: int,
    samples: int,
    seed: int,
    maximum_batch_elements: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed differentials and their recentred bootstrap draws.

    One index path per replicate is applied to *every* member, which is what
    preserves the cross-model dependence and makes a maximum over the family
    meaningful.

    Index paths are drawn in chunks of a fixed size that does not depend on how
    many members the family has, and the gather is then sub-batched to bound
    memory.  Decoupling the two matters: it makes the resampled histories a
    function of the seed, the block length and the history alone, so two
    families of different sizes evaluated on the same history see byte-identical
    draws.  Without it, adding a member would silently change every index path
    and the Reality Check's documented behaviour under family growth would not
    be checkable.
    """

    if statistic not in {MEAN_STATISTIC, SHARPE_STATISTIC}:
        raise ValueError(f"unknown family statistic: {statistic!r}")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")

    challengers = family.challengers
    if not challengers:
        raise ValueError("a family-wise statistic needs at least one challenger")
    benchmark = family.paths[family.benchmark].to_numpy(dtype=float)
    challenger = family.paths[list(challengers)].to_numpy(dtype=float)
    count = benchmark.size
    width = challenger.shape[1]
    annualization = family.annualization

    if statistic == MEAN_STATISTIC:
        differential = challenger - benchmark[:, None]
        observed = differential.mean(axis=0) * annualization
    else:
        observed = inference._sharpe(
            challenger.T, annualization
        ) - inference._sharpe(benchmark, annualization)

    rng = _seeded_generator(
        seed,
        "family-wise stationary bootstrap max statistic",
        family.label,
        int(block_length),
        int(count),
    )
    gather = max(1, int(maximum_batch_elements // max(count * width, 1)))
    draws = np.empty((samples, width), dtype=float)
    for chunk in range(0, samples, _INDEX_CHUNK):
        take = min(_INDEX_CHUNK, samples - chunk)
        indices = inference._stationary_indices(
            rng, count, take, count, int(block_length)
        )
        for offset in range(0, take, gather):
            piece = indices[offset : offset + gather]
            row = chunk + offset
            if statistic == MEAN_STATISTIC:
                draws[row : row + len(piece)] = (
                    differential[piece].mean(axis=1) * annualization
                )
            else:
                draws[row : row + len(piece)] = inference._sharpe(
                    np.moveaxis(challenger[piece], 1, 2), annualization
                ) - inference._sharpe(benchmark[piece], annualization)[:, None]
    return observed, draws


def _reality_check_p_value(
    observed: np.ndarray,
    draws: np.ndarray,
    horizon: int,
) -> tuple[float, float]:
    """White's max statistic and its finite-sample p-value.

    The bootstrap maximum is recentred on each member's own observed mean,
    which imposes the least favourable configuration ``E[f_k] = 0`` for every
    member.  That recentring is the entire test: recentring on zero, or on the
    grand mean, produces a different and generally anti-conservative number.
    """

    root = math.sqrt(horizon)
    statistic = float(root * observed.max())
    resampled = (draws - observed[None, :]).max(axis=1) * root
    exceedances = int((resampled >= statistic).sum())
    p_value = (1.0 + exceedances) / (1.0 + draws.shape[0])
    return p_value, statistic


def _spa_omega(
    observed: np.ndarray,
    draws: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Hansen's bootstrap variance estimate, one per family member."""

    centred = (draws - observed[None, :]) * math.sqrt(horizon)
    return np.sqrt((centred**2).mean(axis=0))


def spa_threshold(observations: int) -> float:
    """The law-of-iterated-logarithm constant ``sqrt(2 log log n)``.

    Exposed because it is the single most frequently mis-transcribed piece of
    the SPA and the only free constant in the consistent recentring.
    """

    if observations < MINIMUM_SPA_OBSERVATIONS:
        raise ValueError(
            "the iterated-logarithm threshold is not meaningful below "
            f"{MINIMUM_SPA_OBSERVATIONS} observations"
        )
    return math.sqrt(2.0 * math.log(math.log(observations)))


def _spa_p_values(
    observed: np.ndarray,
    draws: np.ndarray,
    horizon: int,
    omega: np.ndarray,
) -> tuple[dict[str, float], float]:
    """The three studentised SPA p-values and the observed statistic.

    The recentrings satisfy ``g_u <= g_c <= g_l`` elementwise, so the p-values
    satisfy ``p_l <= p_c <= p_u`` identically rather than approximately.  The
    width of that bracket measures how much the answer depends on how the
    demonstrably poor members of the family are handled, which is the whole
    substantive difference between Hansen and White.
    """

    root = math.sqrt(horizon)
    studentised = root * observed / omega
    statistic = float(max(0.0, studentised.max()))
    threshold = spa_threshold(horizon)
    recentrings = {
        SPA_UPPER: observed,
        SPA_CONSISTENT: np.where(studentised >= -threshold, observed, 0.0),
        SPA_LOWER: np.maximum(observed, 0.0),
    }
    p_values: dict[str, float] = {}
    for name, centre in recentrings.items():
        resampled = ((draws - centre[None, :]) * root / omega[None, :]).max(axis=1)
        resampled = np.maximum(resampled, 0.0)
        exceedances = int((resampled >= statistic).sum())
        p_values[name] = (1.0 + exceedances) / (1.0 + draws.shape[0])
    return p_values, statistic


def _family_wise_rows(
    family: StrategyFamily,
    *,
    statistic: str,
    procedures: Sequence[str],
    block_length: int,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    common = {
        "Family": family.label,
        "Benchmark": family.benchmark,
        "Statistic": statistic,
        "Expected block sessions": int(block_length),
        "Samples": int(samples),
        "Seed": int(seed),
        "Observations": family.observations,
        "Family size": family.size,
        "Family completeness": family.completeness,
        "Selection adjusted": True,
        "Permitted use": _FAMILY_WISE_PERMITTED_USE,
    }

    def refusal(reason: str) -> list[dict[str, Any]]:
        return [
            {
                **common,
                "Procedure": procedure,
                "Method": "refused before resampling",
                "Status": NOT_ESTIMABLE,
                "p value": None,
                "p value resolution": None,
                "Observed max statistic": None,
                "Members without bootstrap variance": None,
                "Reason": reason,
            }
            for procedure in procedures
        ]

    if not family.is_complete:
        return refusal(
            "a maximum over a subset of the searched family understates the "
            f"multiplicity correction; {family.incompleteness_reason()}"
        )
    if not family.challengers:
        return refusal("the family contains no challenger to compare with the benchmark")
    horizon = family.observations
    if horizon < MINIMUM_SPA_OBSERVATIONS:
        return refusal(
            f"{horizon} observations is below the {MINIMUM_SPA_OBSERVATIONS} "
            "needed for the asymptotics these procedures rely on"
        )

    observed, draws = _family_bootstrap(
        family,
        statistic=statistic,
        block_length=block_length,
        samples=samples,
        seed=seed,
    )
    # A member whose observed statistic or whose resampled statistics are
    # non-finite has no statistic at all under this measure - a path with no
    # variance has no Sharpe ratio, and a path with a handful of live sessions
    # has none on some resamples.  Leaving such a member in poisons the maximum
    # with NaN, which compares False against every draw and drives the p-value
    # onto the resolution floor: an undefined statistic silently reported as
    # the strongest rejection the procedure can emit.  Masking it out instead
    # would shrink the family the maximum runs over, which is the very thing
    # this function refuses an incomplete family for.  So it refuses, names the
    # offenders, and leaves the caller to redeclare the family or the statistic.
    finite = np.isfinite(observed) & np.isfinite(draws).all(axis=0)
    undefined = [name for name, keep in zip(family.challengers, finite) if not keep]
    if undefined:
        return refusal(
            f"the {statistic.lower()} is undefined for "
            f"{len(undefined)} declared member(s) - {', '.join(undefined)} - so "
            "the family maximum is undefined; a maximum over the remaining "
            "members would be a maximum over a smaller family than the one "
            "that was searched, which understates the multiplicity correction"
        )

    # Distinct from the above, and benign: a member that reproduces the
    # benchmark exactly has a finite differential of zero whose bootstrap
    # variance is also zero.  The Reality Check handles it correctly, because
    # its recentred draws are identically zero and its observed contribution to
    # the maximum is a genuine zero.  Only Hansen's studentisation is undefined
    # there, and dropping such a member from the SPA is exactly neutral: its
    # studentised contribution would be 0/0, and the statistic is already
    # truncated at zero.  Both facts are recorded on every row rather than
    # asserted in one place.
    omega = _spa_omega(observed, draws, horizon)
    positive_variance = np.isfinite(omega) & (omega > 0)
    zero_variance = [
        name
        for name, keep in zip(family.challengers, positive_variance)
        if not keep
    ]

    resolution = 1.0 / (samples + 1.0)
    rows: list[dict[str, Any]] = []

    if REALITY_CHECK in procedures:
        p_value, statistic_value = _reality_check_p_value(observed, draws, horizon)
        kept = (
            "estimated over the complete declared family under a "
            f"{block_length}-session expected block"
        )
        if zero_variance:
            kept += (
                f"; {len(zero_variance)} member(s) reproduce the benchmark "
                f"exactly ({', '.join(zero_variance)}) and are retained, their "
                "recentred draws being identically zero"
            )
        rows.append(
            {
                **common,
                "Procedure": REALITY_CHECK,
                "Method": (
                    "stationary bootstrap maximum recentred on each member's "
                    "observed differential (White 2000)"
                ),
                "Status": ESTIMATED,
                "p value": p_value,
                "p value resolution": resolution,
                "Observed max statistic": statistic_value,
                "Members without bootstrap variance": len(zero_variance),
                "Reason": kept,
            }
        )

    wanted_spa = [name for name in procedures if name in SPA_PROCEDURES]
    if wanted_spa:
        estimable = positive_variance
        dropped = zero_variance
        if not estimable.any():
            rows.extend(
                {
                    **common,
                    "Procedure": procedure,
                    "Method": "refused after resampling",
                    "Status": NOT_ESTIMABLE,
                    "p value": None,
                    "p value resolution": None,
                    "Observed max statistic": None,
                    "Members without bootstrap variance": len(dropped),
                    "Reason": (
                        "every member has zero bootstrap variance, so the "
                        "studentised statistic is undefined; a variance floor "
                        "would convert an undefined statistic into an "
                        "optimistic one"
                    ),
                }
                for procedure in wanted_spa
            )
        else:
            p_values, statistic_value = _spa_p_values(
                observed[estimable],
                draws[:, estimable],
                horizon,
                omega[estimable],
            )
            for procedure in wanted_spa:
                rows.append(
                    {
                        **common,
                        "Procedure": procedure,
                        "Method": (
                            "studentised stationary bootstrap maximum, "
                            f"{procedure.split()[-1]} recentring (Hansen 2005)"
                        ),
                        "Status": ESTIMATED,
                        "p value": p_values[procedure],
                        "p value resolution": resolution,
                        "Observed max statistic": statistic_value,
                        "Members without bootstrap variance": len(dropped),
                        "Reason": (
                            "estimated over "
                            f"{int(estimable.sum())} of {len(family.challengers)} "
                            "challengers with a positive bootstrap variance"
                        ),
                    }
                )
    return rows


def reality_check(
    family: StrategyFamily,
    *,
    statistics: Sequence[str] = (MEAN_STATISTIC, SHARPE_STATISTIC),
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_PAIRED_SEED,
) -> pd.DataFrame:
    """White's Reality Check across a declared grid of block lengths.

    Reported at every block length in the grid rather than at one chosen
    afterwards.  The Reality Check is known to lose power as demonstrably poor
    members are added to the family, so a p-value near one is evidence about
    the family as declared and not only about the best member in it.
    """

    rows: list[dict[str, Any]] = []
    for statistic in statistics:
        for block_length in block_lengths:
            rows.extend(
                _family_wise_rows(
                    family,
                    statistic=statistic,
                    procedures=(REALITY_CHECK,),
                    block_length=int(block_length),
                    samples=int(samples),
                    seed=int(seed),
                )
            )
    return pd.DataFrame(rows, columns=FAMILY_WISE_COLUMNS)


def hansen_spa(
    family: StrategyFamily,
    *,
    statistics: Sequence[str] = (MEAN_STATISTIC, SHARPE_STATISTIC),
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_PAIRED_SEED,
) -> pd.DataFrame:
    """Hansen's SPA under all three recentrings, on the Reality Check's draws.

    The consistent recentring is the headline; the lower and upper recentrings
    bracket it, and the width of that bracket is itself the finding, because it
    measures how much of the answer is the treatment of the poor members rather
    than the evidence for the good ones.
    """

    rows: list[dict[str, Any]] = []
    for statistic in statistics:
        for block_length in block_lengths:
            rows.extend(
                _family_wise_rows(
                    family,
                    statistic=statistic,
                    procedures=SPA_PROCEDURES,
                    block_length=int(block_length),
                    samples=int(samples),
                    seed=int(seed),
                )
            )
    return pd.DataFrame(rows, columns=FAMILY_WISE_COLUMNS)


def family_wise_report(
    family: StrategyFamily,
    *,
    statistics: Sequence[str] = (MEAN_STATISTIC, SHARPE_STATISTIC),
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_PAIRED_SEED,
) -> pd.DataFrame:
    """Reality Check and SPA together, computed from one set of index paths.

    Running them separately with the same seed gives the same numbers; running
    them together makes it structural rather than incidental, which is the only
    way the claim that they differ solely in studentisation and recentring is
    checkable.
    """

    rows: list[dict[str, Any]] = []
    for statistic in statistics:
        for block_length in block_lengths:
            rows.extend(
                _family_wise_rows(
                    family,
                    statistic=statistic,
                    procedures=ALL_PROCEDURES,
                    block_length=int(block_length),
                    samples=int(samples),
                    seed=int(seed),
                )
            )
    return pd.DataFrame(rows, columns=FAMILY_WISE_COLUMNS)


def _gumbel_expected_maximum(trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe under the null over ``trials`` draws.

    Recomputed here only so the threshold the deflated Sharpe is testing
    against can be *reported*.  It is the same Gumbel expression
    ``inference.deflated_sharpe_ratio`` uses internally, and the two are
    checked against each other in the tests rather than assumed equal.
    """

    if trials <= 1:
        return 0.0
    from statistics import NormalDist

    euler = 0.577_215_664_901_532_9
    normal = NormalDist()
    count = float(trials)
    return math.sqrt(trial_sharpe_variance) * (
        (1.0 - euler) * normal.inv_cdf(1.0 - 1.0 / count)
        + euler * normal.inv_cdf(1.0 - 1.0 / (count * math.e))
    )


def family_deflated_sharpe(
    family: StrategyFamily,
    *,
    member: str | None = None,
) -> pd.Series:
    """The deflated Sharpe refusal for one family member, and why it refuses.

    The status of this row is always NOT_ESTIMABLE, and that is the finding
    rather than a gap in the implementation.  The deflated Sharpe deflates by
    the number of trials that were *searched*; the declared family is the
    number of trials that were *registered*, which for this lineage is a lower
    bound on the search and not an estimate of it.  ``research-methodology.md``
    records the global trial count as unrecoverable, ``inference.
    deflated_sharpe_ratio`` refuses rather than assume one, and
    ``inference.promotion_support`` reads an estimated deflated Sharpe as
    clearing the promotion gate.  Returning a status of ESTIMATED here would
    therefore hand every member of a seventeen-configuration family a cleared
    gate on the strength of a trial count nobody can defend.

    The arithmetic is still performed and reported, under a name the promotion
    machinery cannot consume, because the *size of the deflation* is itself
    informative.  Two conventions inside it are silent when wrong and are
    handled here rather than left to the caller.  The statistic is defined on
    the *per-period* Sharpe, so feeding an annualized Sharpe with a daily
    observation count saturates it at exactly 1.0; the conversion happens here.
    The published higher-moment coefficient is ``(gamma_4 - 1) / 4`` on
    non-excess kurtosis; ``inference.deflated_sharpe_ratio`` forms it from the
    excess figure as ``(excess + 2) / 4``, so excess kurtosis is passed
    through unmodified.

    What makes the illustrative number uninformative *within* a family is not
    floating point.  It is the deflation threshold: the Gumbel expected maximum
    over a declared family of seventeen is a per-period Sharpe of 0.0214, or
    0.34 annualized, so the statistic is asking whether each member beats an
    annualized Sharpe of about a third - which every trend configuration on
    this history clears by construction.  The threshold is reported beside the
    value so a reader can see what was actually tested.
    """

    name = member if member is not None else family.benchmark
    if name not in family.paths.columns:
        raise ValueError(f"{name!r} is not a member of this family")
    per_period_sharpe: float | None = None
    trial_variance: float | None = None
    threshold: float | None = None
    illustrative: float | None = None
    if not family.is_complete:
        reason = (
            "the declared family is incomplete, so the trial count understates "
            f"the search even as a lower bound; {family.incompleteness_reason()}"
        )
    else:
        per_period = family.paths.mean(axis=0) / family.paths.std(axis=0, ddof=0)
        trial_variance = float(per_period.var(ddof=1)) if family.size > 1 else None
        per_period_sharpe = float(per_period[name])
        returns = family.paths[name]
        skewness = float(returns.skew())
        excess_kurtosis = float(returns.kurt())
        lower_bound = inference.deflated_sharpe_ratio(
            per_period_sharpe,
            trial_count=family.size,
            trial_sharpe_variance=trial_variance,
            sessions=family.observations,
            skewness=skewness,
            # ``inference.deflated_sharpe_ratio`` now forms the published
            # (gamma_4 - 1)/4 coefficient itself from the excess figure, so the
            # excess kurtosis is passed straight through.  It previously
            # divided this argument by four unchanged, and the compensating
            # ``kurtosis - 1`` that used to sit here would now double-correct.
            excess_kurtosis=excess_kurtosis,
        )
        illustrative = lower_bound.value
        if trial_variance is not None:
            threshold = _gumbel_expected_maximum(family.size, trial_variance)
        reason = (
            f"the declared family of {family.size} is a lower bound on this "
            "lineage's search, not its trial count, which "
            "research-methodology.md records as unrecoverable; the deflated "
            "probability computed against a lower bound is itself only a lower "
            "bound and cannot support the promotion gate"
        )
    result = InferenceResult(
        statistic="deflated_sharpe_ratio",
        status=NOT_ESTIMABLE,
        value=None,
        reason=reason,
    )
    return pd.Series(
        {
            "Family": family.label,
            "Member": name,
            "Statistic": result.statistic,
            "Method": (
                "Bailey and Lopez de Prado deflated Sharpe on per-period "
                "returns; higher-moment coefficient (kurtosis - 1)/4 with "
                "non-excess kurtosis; reported as a lower bound only, because "
                "the deflation uses the registered family size in place of an "
                "unrecoverable search count"
            ),
            "Status": result.status,
            "Value": result.value,
            "Reason": result.reason,
            "Lower-bound deflated probability (illustrative)": illustrative,
            "Deflation threshold (per period)": threshold,
            "Deflation threshold (annualized)": (
                None
                if threshold is None
                else threshold * math.sqrt(family.annualization)
            ),
            "Per-period Sharpe": per_period_sharpe,
            "Declared trials": family.size,
            "Trial Sharpe variance (per period)": trial_variance,
            "Observations": family.observations,
            "Selection adjusted": True,
            "Permitted use": _FAMILY_WISE_PERMITTED_USE,
        }
    )


# ---------------------------------------------------------------------------
# CSCV / probability of backtest overfitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CscvResult:
    """CSCV outputs, each either estimated or explicitly not estimable.

    ``splits`` is empty whenever the procedure refused, so a caller cannot
    accidentally summarise a partial combination set into a number the refusal
    was meant to withhold.
    """

    family: str
    probability_of_backtest_overfitting: InferenceResult
    performance_degradation_slope: InferenceResult
    probability_of_loss: InferenceResult
    probability_selected_beats_median: InferenceResult
    splits: pd.DataFrame
    diagnostics: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))

    def to_frame(self) -> pd.DataFrame:
        """One descriptive row per statistic, with status and reason attached."""

        common = {
            "Family": self.family,
            "Submatrices": self.diagnostics.get("submatrices"),
            "Combinations": self.diagnostics.get("combinations"),
            "Rows per submatrix": self.diagnostics.get("rows_per_submatrix"),
            "Rows dropped from the front": self.diagnostics.get("rows_dropped"),
            "Configurations": self.diagnostics.get("configurations"),
            "Family completeness": self.diagnostics.get("family_completeness"),
            "In-sample selection ties": self.diagnostics.get("selection_ties"),
            "Selection adjusted": True,
            "Permitted use": _CSCV_PERMITTED_USE,
        }
        methods = {
            "probability_of_backtest_overfitting": (
                "CSCV: share of C(S, S/2) splits whose in-sample best lands in "
                "the out-of-sample bottom half"
            ),
            "performance_degradation_slope": (
                "OLS slope of the selected configuration's out-of-sample Sharpe "
                "on its in-sample Sharpe; a negative slope is the overfitting "
                "signature, but the two halves are complements of one fixed "
                "history, so the slope is mechanically negative whenever a "
                "single configuration wins most splits and is not on its own "
                "evidence of overfitting"
            ),
            "probability_of_loss": (
                "share of splits whose selected configuration has a negative "
                "out-of-sample Sharpe"
            ),
            "probability_selected_beats_median": (
                "share of splits whose selected configuration beats the median "
                "configuration out of sample"
            ),
        }
        rows = []
        for attribute, method in methods.items():
            result: InferenceResult = getattr(self, attribute)
            rows.append(
                {
                    **common,
                    "Statistic": attribute,
                    "Method": method,
                    "Status": result.status,
                    "Value": result.value,
                    "Reason": result.reason,
                }
            )
        return pd.DataFrame(rows, columns=CSCV_COLUMNS)


def _cscv_refusal(family_label: str, reason: str, diagnostics: pd.Series) -> CscvResult:
    def refuse(name: str) -> InferenceResult:
        return InferenceResult(
            statistic=name, status=NOT_ESTIMABLE, value=None, reason=reason
        )

    return CscvResult(
        family=family_label,
        probability_of_backtest_overfitting=refuse(
            "probability_of_backtest_overfitting"
        ),
        performance_degradation_slope=refuse("performance_degradation_slope"),
        probability_of_loss=refuse("probability_of_loss"),
        probability_selected_beats_median=refuse(
            "probability_selected_beats_median"
        ),
        splits=_empty_frame(CSCV_SPLIT_COLUMNS),
        diagnostics=diagnostics,
    )


def cscv_pbo(
    family: StrategyFamily,
    *,
    submatrices: int = DEFAULT_SUBMATRICES,
    minimum_configurations: int = DEFAULT_MINIMUM_CONFIGURATIONS,
    minimum_rows_per_submatrix: int | None = None,
) -> CscvResult:
    """Combinatorially symmetric cross-validation and the probability of
    backtest overfitting.

    The refusal path is the reason this estimator is here.  CSCV's in-sample
    argmax assumes its columns are the configurations that were actually
    searched; when they are not, the argmax is over the wrong set and PBO comes
    out too low.  So an incomplete family, an odd or too-small submatrix count,
    too few configurations for the out-of-sample rank to carry information, or
    a submatrix too short to support a Sharpe ratio each return NOT_ESTIMABLE
    with the reason attached, and never a number.

    Rows that do not divide evenly into submatrices are dropped from the front
    of the history, never the back: the most recent regime is the one a reader
    cares about, and silently discarding it would make the diagnostic answer a
    question about the 1990s.

    ``minimum_rows_per_submatrix`` defaults to one calendar year of the
    family's own frequency - 252 daily, 12 monthly - rather than to a single
    session count.  ``research-methodology.md`` permits this estimator only on
    monthly paths, and a daily floor applied to a monthly family refuses every
    such input with a message about 252 monthly observations.
    """

    if (
        isinstance(submatrices, bool)
        or not isinstance(submatrices, int)
        or submatrices < 2
    ):
        raise ValueError("submatrices must be an integer of at least 2")
    if minimum_rows_per_submatrix is None:
        minimum_rows_per_submatrix = MINIMUM_ROWS_PER_SUBMATRIX_BY_FREQUENCY[
            family.frequency
        ]
    if minimum_configurations < 2 or minimum_rows_per_submatrix < 2:
        raise ValueError(
            "minimum_configurations and minimum_rows_per_submatrix must be at least 2"
        )

    observations = family.observations
    configurations = family.size
    rows_per_submatrix = observations // submatrices
    diagnostics = pd.Series(
        {
            "family": family.label,
            "frequency": family.frequency,
            "submatrices": int(submatrices),
            "combinations": None,
            "rows_per_submatrix": int(rows_per_submatrix),
            "minimum_rows_per_submatrix": int(minimum_rows_per_submatrix),
            "rows_dropped": int(observations - rows_per_submatrix * submatrices),
            "configurations": int(configurations),
            "observations": int(observations),
            "family_completeness": family.completeness,
            "selection_ties": None,
            "provenance": family.provenance,
        }
    )

    if not family.is_complete:
        return _cscv_refusal(
            family.label,
            (
                "the configuration matrix is incomplete, so the in-sample "
                "argmax is taken over the wrong set and PBO would be "
                f"systematically understated; {family.incompleteness_reason()}"
            ),
            diagnostics,
        )
    if submatrices % 2 != 0:
        return _cscv_refusal(
            family.label,
            "CSCV partitions each split into equal halves, which requires an "
            f"even number of submatrices; {submatrices} is odd",
            diagnostics,
        )
    if submatrices < MINIMUM_SUBMATRICES:
        return _cscv_refusal(
            family.label,
            f"{submatrices} submatrices give only "
            f"{math.comb(submatrices, submatrices // 2)} combinations, too few "
            f"for a stable PBO; the floor is {MINIMUM_SUBMATRICES}",
            diagnostics,
        )
    if configurations < minimum_configurations:
        return _cscv_refusal(
            family.label,
            f"{configurations} configurations quantise the out-of-sample rank "
            f"in steps of 1/{configurations + 1}, which measures the grid "
            f"rather than the overfitting; the floor is {minimum_configurations}",
            diagnostics,
        )
    if rows_per_submatrix < minimum_rows_per_submatrix:
        return _cscv_refusal(
            family.label,
            f"{rows_per_submatrix} {family.frequency} observations per "
            "submatrix cannot support a Sharpe ratio; the floor is "
            f"{minimum_rows_per_submatrix}",
            diagnostics,
        )

    values = family.paths.to_numpy(dtype=float)
    kept = rows_per_submatrix * submatrices
    trimmed = values[observations - kept :]
    blocks = trimmed.reshape(submatrices, rows_per_submatrix, configurations)
    sums = blocks.sum(axis=1)
    squares = (blocks**2).sum(axis=1)

    half = submatrices // 2
    combinations = np.array(
        list(itertools.combinations(range(submatrices), half)), dtype=np.int64
    )
    membership = np.zeros((combinations.shape[0], submatrices), dtype=float)
    np.put_along_axis(membership, combinations, 1.0, axis=1)

    half_rows = rows_per_submatrix * half
    root = math.sqrt(family.annualization)

    def sharpe(total: np.ndarray, total_squares: np.ndarray) -> np.ndarray:
        mean = total / half_rows
        variance = total_squares / half_rows - mean**2
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(variance > 0, mean / np.sqrt(variance) * root, np.nan)

    in_sample_sum = membership @ sums
    in_sample_squares = membership @ squares
    out_sample_sum = sums.sum(axis=0)[None, :] - in_sample_sum
    out_sample_squares = squares.sum(axis=0)[None, :] - in_sample_squares

    in_sample = sharpe(in_sample_sum, in_sample_squares)
    out_sample = sharpe(out_sample_sum, out_sample_squares)
    if not np.isfinite(in_sample).all() or not np.isfinite(out_sample).all():
        return _cscv_refusal(
            family.label,
            "at least one submatrix combination produced a zero-variance "
            "return path, so its Sharpe ratio is undefined",
            diagnostics,
        )

    selected = np.argmax(in_sample, axis=1)
    maxima = (in_sample == in_sample.max(axis=1, keepdims=True)).sum(axis=1)
    ties = int((maxima > 1).sum())
    diagnostics["combinations"] = int(combinations.shape[0])
    diagnostics["selection_ties"] = ties

    split_rows = np.arange(combinations.shape[0])
    selected_out_sample = out_sample[split_rows, selected]
    selected_in_sample = in_sample[split_rows, selected]
    below = (out_sample < selected_out_sample[:, None]).sum(axis=1)
    equal = (out_sample == selected_out_sample[:, None]).sum(axis=1)
    rank = below + (equal + 1.0) / 2.0
    relative_rank = rank / (configurations + 1.0)
    logit = np.log(relative_rank / (1.0 - relative_rank))

    pbo = float((logit <= 0.0).mean())
    probability_of_loss = float((selected_out_sample < 0.0).mean())
    median_out_sample = np.median(out_sample, axis=1)
    beats_median = float((selected_out_sample > median_out_sample).mean())

    spread = float(selected_in_sample.std(ddof=0))
    if spread > 0:
        slope = float(
            np.polyfit(selected_in_sample, selected_out_sample, 1)[0]
        )
        degradation = InferenceResult(
            statistic="performance_degradation_slope",
            status=ESTIMATED,
            value=slope,
            reason=(
                f"ordinary least squares over {combinations.shape[0]} splits of "
                f"the selected configuration's out-of-sample Sharpe on its "
                "in-sample Sharpe"
            ),
        )
    else:
        degradation = InferenceResult(
            statistic="performance_degradation_slope",
            status=NOT_ESTIMABLE,
            value=None,
            reason=(
                "the selected configuration's in-sample Sharpe is constant "
                "across splits, so the regression has no variation to fit"
            ),
        )

    names = list(family.paths.columns)
    splits = pd.DataFrame(
        {
            "combination": split_rows,
            "in_sample_selection": [names[index] for index in selected],
            "in_sample_sharpe": selected_in_sample,
            "out_of_sample_sharpe": selected_out_sample,
            "out_of_sample_rank": rank,
            "relative_rank": relative_rank,
            "logit": logit,
        },
        columns=CSCV_SPLIT_COLUMNS,
    )
    reason = (
        f"estimated over {combinations.shape[0]} combinations of "
        f"{submatrices} submatrices of {rows_per_submatrix} "
        f"{family.frequency} observations across {configurations} "
        "declared configurations"
    )
    return CscvResult(
        family=family.label,
        probability_of_backtest_overfitting=InferenceResult(
            statistic="probability_of_backtest_overfitting",
            status=ESTIMATED,
            value=pbo,
            reason=reason,
        ),
        performance_degradation_slope=degradation,
        probability_of_loss=InferenceResult(
            statistic="probability_of_loss",
            status=ESTIMATED,
            value=probability_of_loss,
            reason=reason,
        ),
        probability_selected_beats_median=InferenceResult(
            statistic="probability_selected_beats_median",
            status=ESTIMATED,
            value=beats_median,
            reason=reason,
        ),
        splits=splits,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Anchored expanding walk-forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardCandidate:
    """One selectable specification, with the choice it represents stated.

    ``overrides`` is a tuple of pairs rather than a mapping so the candidate
    stays hashable and a run cache keyed on it cannot be mutated between folds.
    """

    name: str
    assumption: str
    overrides: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for attribute in ("name", "assumption"):
            value = getattr(self, attribute)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{attribute} must be a non-empty string")
        keys = [key for key, _ in self.overrides]
        if len(keys) != len(set(keys)):
            raise ValueError("override keys must be unique")

    def apply(self, config: Any) -> Any:
        return replace(config, **dict(self.overrides))


@dataclass(frozen=True)
class WalkForwardConfig:
    """Anchor, boundary schedule and the minimum history the anchor must hold.

    ``minimum_training_sessions`` defaults to 315 because the frozen
    specification's longest lookback is 252 sessions and the portfolio risk
    scalar warms up over a further 63; below that sum the first boundary is not
    a walk-forward origin but a truncated backtest with extra vocabulary.
    """

    boundaries: tuple[str, ...]
    anchor: str = RETROSPECTIVE_START
    end: str | None = RETROSPECTIVE_END
    minimum_training_sessions: int = 315
    annualization: int = 252
    hac_lags: int = 21

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, str) or not self.anchor:
            raise ValueError("anchor must be a non-empty date string")
        if not isinstance(self.boundaries, tuple):
            raise ValueError("boundaries must be a tuple of date strings")
        if len(self.boundaries) < 1:
            raise ValueError("a walk-forward needs at least one boundary")
        stamps = [pd.Timestamp(value) for value in self.boundaries]
        if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
            raise ValueError("boundaries must be sorted and unique")
        if stamps[0] <= pd.Timestamp(self.anchor):
            raise ValueError("the first boundary must fall after the anchor")
        if self.end is not None and pd.Timestamp(self.end) <= stamps[-1]:
            raise ValueError("end must fall after the last boundary")
        for attribute in ("minimum_training_sessions", "annualization", "hac_lags"):
            value = getattr(self, attribute)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{attribute} must be a positive integer")

    @property
    def boundary_stamps(self) -> tuple[pd.Timestamp, ...]:
        return tuple(pd.Timestamp(value) for value in self.boundaries)


@dataclass
class WalkForwardReport:
    """Fold table, stitched path and the ledger the stitched path came from."""

    label: str
    mode: str
    folds: pd.DataFrame
    stitched_path: pd.Series
    daily: pd.DataFrame
    decision_intents: pd.DataFrame
    summary: pd.Series


def _select_highest_sharpe(training: pd.DataFrame) -> tuple[str, float]:
    """Pick the training-span Sharpe maximiser, ties broken by column order.

    Deliberately naive.  A selector regularised toward the incumbent would
    make the walk-forward look better while measuring less: the diagnostic
    exists to expose what naive in-sample selection costs out of sample, so
    hardening the selector would remove the thing being measured.
    """

    if training.empty:
        raise ValueError("selection requires a non-empty training span")
    scores = training.mean(axis=0) / training.std(axis=0, ddof=0)
    scores = scores.replace([np.inf, -np.inf], np.nan)
    if scores.isna().all():
        raise ValueError("no candidate has a finite training-span Sharpe")
    name = str(scores.idxmax())
    return name, float(scores.max())


def _execution_inputs(config: Any, frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Point values, one-way costs, margins and roll flags, built once.

    Identical to the arithmetic inside ``run_backtest`` so a spliced replay is
    charged the same costs as the canonical run by construction rather than by
    inspection.
    """

    from .strategy import roll_event_mask, usd_margin_values, usd_point_values

    prices = frames["prices"]
    metadata = frames["metadata"]
    point_values = usd_point_values(metadata, frames["fx_rates"], prices.index)
    margin_values = usd_margin_values(metadata, frames["fx_rates"], prices.index)
    one_way_costs = (
        (config.half_spread_ticks + config.slippage_ticks)
        * point_values.mul(metadata["tick_size"], axis=1)
        + config.commission_per_contract
        + config.exchange_and_regulatory_fees_per_contract
    )
    rolls, _ = roll_event_mask(frames["delivery_months"])
    return {
        "point_values": point_values,
        "margin_values": margin_values,
        "one_way_costs": one_way_costs,
        "rolls": rolls,
    }


def anchored_walk_forward(
    config: Any,
    *,
    frames: Mapping[str, pd.DataFrame],
    walk_forward: WalkForwardConfig,
    candidates: Sequence[WalkForwardCandidate] | None = None,
    label: str = "delta1_anchored_walk_forward",
    run_fn: Callable[[Any], Any] | None = None,
    select_fn: Callable[[pd.DataFrame], tuple[str, float]] | None = None,
) -> WalkForwardReport:
    """Replay the strategy once, switching regimes only on fold boundaries.

    Each candidate's monthly decision frame is built over the whole history,
    the folds' selected regimes are spliced into a single frame, and the
    execution ledger is simulated once over the spliced result.  That is the
    only construction available here that preserves the engine's live execution
    state: NAV compounds across boundaries, the executed book is the one the
    previous regime left, the roll backlog stays open, the participation and
    cost ledgers are continuous, and the turnover of switching regimes is
    charged because the ledger sees the jump in intents.  A per-fold engine
    call with fresh initial capital would reset all of that and would report a
    better path than the strategy could have produced.

    Because the ledger is built once from ``config``, a candidate may only
    override fields that shape the intent frame.  ``SPLICEABLE_CONFIG_FIELDS``
    is the allow-list and anything outside it raises, rather than being scored
    under the candidate's specification and then replayed under the base one
    with nothing in the fold table to say so.

    Intent-construction state is the remaining exception, and it is a genuine
    approximation rather than an omission: the incoming specification's
    no-trade band and portfolio risk scalar were warmed on its own full-history
    path, not on the spliced one, so the first decision after a switch is
    buffered against a counterfactual previous intent.  The module docstring
    explains why the engine cannot currently do better, and ``selection_switches``
    in the summary is the count that bounds how much it can matter - counted
    against the warm-up regime, which is the first declared candidate, so a
    fold-zero switch away from it is counted as the discontinuity it is.

    The selector never sees a date at or after the fold it selects for: it is
    handed a slice ending strictly before the boundary, so the restriction is
    structural rather than a convention the caller has to honour.

    With one candidate there is no selector, and the procedure measures the
    stability of a frozen specification across expanding anchors.  That is
    reported under its own mode label because it is a description of one path
    and not evidence about selection.

    What it is not: the specification being replayed was written with this
    window already read, so the out-of-sample segments are out of sample with
    respect to the selector alone.  A favourable walk-forward here is a
    consistency result on reused history, not prospective evidence.
    """

    from .strategy import _simulate_execution

    missing_frames = [key for key in _FRAME_KEYS if key not in frames]
    if missing_frames:
        raise ValueError(
            "run_backtest substitutes permissive defaults for omitted frames, "
            "so a partial set silently changes the result; missing "
            f"{missing_frames}"
        )
    members = tuple(candidates) if candidates else (
        WalkForwardCandidate(
            name="frozen_specification",
            assumption="the canonical configuration, unchanged across every fold",
        ),
    )
    names = [candidate.name for candidate in members]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    unspliceable = sorted(
        {
            key
            for candidate in members
            for key, _ in candidate.overrides
            if key not in SPLICEABLE_CONFIG_FIELDS
        }
    )
    if unspliceable:
        raise ValueError(
            "the spliced replay builds one execution ledger from the base "
            "configuration, so a candidate can only vary fields that shape the "
            "monthly intent frame; these override the ledger itself "
            f"({_EXECUTION_LEDGER_FIELD_NOTE}) and would be scored under the "
            "candidate but replayed under the base configuration without any "
            f"record of the substitution: {unspliceable}"
        )
    mode = (
        WALK_FORWARD_SELECTION_MODE
        if len(members) > 1
        else WALK_FORWARD_STABILITY_MODE
    )
    permitted_use = (
        _WALK_FORWARD_PERMITTED_USE
        if mode == WALK_FORWARD_SELECTION_MODE
        else _STABILITY_PERMITTED_USE
    )

    if run_fn is None:
        from .strategy import run_backtest

        def default_runner(candidate_config: Any) -> Any:
            return run_backtest(candidate_config, **frames)

        runner: Callable[[Any], Any] = default_runner
    else:
        runner = run_fn

    runs = {
        candidate.name: runner(candidate.apply(config)) for candidate in members
    }

    intents: dict[str, pd.DataFrame] = {}
    daily_paths: dict[str, pd.Series] = {}
    reference: pd.DataFrame | None = None
    for name in names:
        result = runs[name]
        frame = result.target_positions
        if reference is None:
            reference = frame
        elif not frame.index.equals(reference.index) or not frame.columns.equals(
            reference.columns
        ):
            raise ValueError(
                f"candidate {name!r} produced a different decision calendar; "
                "a spliced replay requires one shared schedule"
            )
        intents[name] = frame
        daily_paths[name] = result.daily["net_return"].astype(float)

    if reference is None:
        raise ValueError("no candidate produced a decision frame")
    decision_index = reference.index
    prices = frames["prices"]
    session_index = prices.index
    anchor = pd.Timestamp(walk_forward.anchor)
    stamps = walk_forward.boundary_stamps
    end = (
        pd.Timestamp(walk_forward.end)
        if walk_forward.end is not None
        else session_index[-1]
    )
    if stamps[-1] >= end:
        raise ValueError("the last boundary must fall before the evaluation end")

    training_sessions_available = int(
        ((session_index >= anchor) & (session_index < stamps[0])).sum()
    )
    if training_sessions_available < walk_forward.minimum_training_sessions:
        raise ValueError(
            f"the anchor holds {training_sessions_available} sessions before the "
            f"first boundary, below the {walk_forward.minimum_training_sessions} "
            "the frozen specification needs to warm its longest lookback and its "
            "portfolio risk scalar"
        )

    candidate_daily = pd.DataFrame(daily_paths)
    selector = select_fn or _select_highest_sharpe

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for position, start in enumerate(stamps):
        stop = stamps[position + 1] if position + 1 < len(stamps) else None
        segments.append((start, stop if stop is not None else end))

    # Every decision before the first boundary is warm-up and comes from the
    # first declared candidate, so that candidate - not "nothing" - is the
    # regime in force when fold zero makes its choice.  Seeding the comparison
    # with it is what makes ``selection_switches`` count the discontinuities
    # the spliced intent frame actually contains: leaving it None reports a
    # fold-zero switch away from the warm-up regime as no switch at all, and
    # that first switch is exactly the case the intent-state caveat is about.
    spliced = intents[names[0]].copy()
    fold_records: list[dict[str, Any]] = []
    previous_choice: str = names[0]
    for position, (start, stop) in enumerate(segments):
        final = position + 1 == len(segments)
        window = (session_index >= start) & (
            (session_index <= stop) if final else (session_index < stop)
        )
        if not window.any():
            raise ValueError(
                f"fold {position} spans no sessions between {start.date()} and "
                f"{stop.date()}"
            )
        training = candidate_daily.loc[
            (candidate_daily.index >= anchor) & (candidate_daily.index < start)
        ]
        if mode == WALK_FORWARD_SELECTION_MODE:
            choice, score = selector(training)
            if choice not in intents:
                raise ValueError(f"the selector returned an unknown candidate: {choice!r}")
        else:
            choice, score = names[0], float("nan")
        decisions = (decision_index >= start) & (
            (decision_index <= stop) if final else (decision_index < stop)
        )
        spliced.loc[decisions] = intents[choice].loc[decisions]
        fold_records.append(
            {
                "label": label,
                "mode": mode,
                "fold": position,
                "anchor": anchor.date().isoformat(),
                "training_end": start.date().isoformat(),
                "training_sessions": int(len(training)),
                "segment_start": start.date().isoformat(),
                "segment_end": stop.date().isoformat(),
                "fold_decisions": int(decisions.sum()),
                "selected_variant": choice,
                "selection_statistic": (
                    "training-span Sharpe maximiser"
                    if mode == WALK_FORWARD_SELECTION_MODE
                    else "no selection; frozen specification"
                ),
                "selection_score": score,
                "selection_switched": bool(choice != previous_choice),
                "hac_convention": _HAC_CONVENTION,
                "cagr_convention": _CAGR_CONVENTION,
                "selection_adjusted": False,
                "permitted_use": permitted_use,
                "_window": window,
            }
        )
        previous_choice = choice

    execution = _execution_inputs(config, frames)
    ledger = _simulate_execution(
        spliced,
        prices,
        frames["unadjusted"],
        frames["observed_opens"],
        frames["observed_closes"],
        frames["volumes"],
        execution["point_values"],
        execution["one_way_costs"],
        execution["margin_values"],
        execution["rolls"],
        config,
        initial_capital=config.initial_capital,
        integer_contracts=config.integer_contracts,
        charge_costs=True,
        launch_date=config.launch_date,
    )
    daily = ledger.daily
    net_return = daily["net_return"].astype(float)

    def _previous_session(stamp: pd.Timestamp) -> pd.Timestamp:
        """The session before ``stamp``, or ``stamp`` itself at the left edge.

        CAGR is annualized on elapsed calendar time from this point, which is
        ``performance_metrics``' convention and additionally makes consecutive
        folds' elapsed periods abut instead of each losing their first session.
        """

        earlier = net_return.index[net_return.index < stamp]
        return earlier[-1] if len(earlier) else stamp

    for record in fold_records:
        window = record.pop("_window")
        segment = net_return.loc[window]
        record["fold_sessions"] = int(len(segment))
        statistics = _path_statistics(
            segment,
            annualization=walk_forward.annualization,
            hac_lags=walk_forward.hac_lags,
            period_start=(
                _previous_session(segment.index[0]) if len(segment) else None
            ),
        )
        record["fold_cagr"] = statistics["cagr"]
        record["fold_annualized_volatility"] = statistics["annualized_volatility"]
        record["fold_sharpe"] = statistics["sharpe"]
        record["fold_hac_sharpe"] = statistics["hac_sharpe"]
        record["fold_max_drawdown"] = statistics["max_drawdown"]
        record["fold_hit_rate"] = statistics["hit_rate"]

    folds = pd.DataFrame(fold_records, columns=WALK_FORWARD_COLUMNS)

    stitched_window = (net_return.index >= stamps[0]) & (net_return.index <= end)
    stitched = net_return.loc[stitched_window]
    stitched_statistics = _path_statistics(
        stitched,
        annualization=walk_forward.annualization,
        hac_lags=walk_forward.hac_lags,
        period_start=_previous_session(stitched.index[0]),
    )
    anchor_path = daily_paths[names[0]]
    anchor_window = (anchor_path.index >= anchor) & (anchor_path.index <= end)
    anchor_full = anchor_path.loc[anchor_window]
    anchor_full_statistics = _path_statistics(
        anchor_full,
        annualization=walk_forward.annualization,
        hac_lags=walk_forward.hac_lags,
        period_start=(
            _previous_session(anchor_full.index[0]) if len(anchor_full) else None
        ),
    )
    # The efficiency denominator is measured on the *stitched span*, not on the
    # anchor window.  Dividing an out-of-sample Sharpe measured from the first
    # boundary by an in-sample Sharpe measured from the anchor compares two
    # different histories: the training years appear only in the denominator,
    # so the ratio departs from one even when no selection happens at all and
    # a frozen specification silently books that artefact as inefficiency.
    # Matched, the stability mode returns exactly 1.0 by construction and every
    # departure is attributable to the selector.  The full-window figures stay,
    # under names that say which span they cover.
    anchor_matched = anchor_path.reindex(stitched.index)
    anchor_matched_statistics = _path_statistics(
        anchor_matched,
        annualization=walk_forward.annualization,
        hac_lags=walk_forward.hac_lags,
        period_start=_previous_session(stitched.index[0]),
    )
    matched_sharpe = anchor_matched_statistics["sharpe"]
    efficiency = float("nan")
    if np.isfinite(matched_sharpe) and matched_sharpe != 0:
        efficiency = stitched_statistics["sharpe"] / matched_sharpe

    stitched_frame = daily.loc[stitched_window]
    anchor_frame = runs[names[0]].daily.loc[anchor_window]
    years = max((stitched.index[-1] - stitched.index[0]).days / 365.2425, 1e-9)
    anchor_years = max(
        (anchor_frame.index[-1] - anchor_frame.index[0]).days / 365.2425, 1e-9
    )
    summary = pd.Series(
        {
            "label": label,
            "mode": mode,
            "anchor": anchor.date().isoformat(),
            "folds": int(len(folds)),
            "candidates": int(len(members)),
            "stitched_start": stitched.index[0].date().isoformat(),
            "stitched_end": stitched.index[-1].date().isoformat(),
            "stitched_sessions": int(len(stitched)),
            "stitched_cagr": stitched_statistics["cagr"],
            "stitched_annualized_volatility": stitched_statistics[
                "annualized_volatility"
            ],
            "stitched_sharpe": stitched_statistics["sharpe"],
            "stitched_hac_sharpe": stitched_statistics["hac_sharpe"],
            "stitched_max_drawdown": stitched_statistics["max_drawdown"],
            "stitched_hit_rate": stitched_statistics["hit_rate"],
            "anchor_specification_full_window_sharpe": anchor_full_statistics[
                "sharpe"
            ],
            "anchor_specification_full_window_cagr": anchor_full_statistics["cagr"],
            "anchor_specification_stitched_span_sharpe": matched_sharpe,
            "anchor_specification_stitched_span_cagr": anchor_matched_statistics[
                "cagr"
            ],
            "walk_forward_efficiency": efficiency,
            "walk_forward_efficiency_denominator_span": "stitched span",
            "warm_up_variant": names[0],
            "selection_switches": int(folds["selection_switched"].sum()),
            "distinct_variants_selected": int(folds["selected_variant"].nunique()),
            "stitched_annual_rebalance_turnover_contracts": float(
                stitched_frame["rebalance_contract_turnover"].sum() / years
            ),
            "anchor_annual_rebalance_turnover_contracts": float(
                anchor_frame["rebalance_contract_turnover"].sum() / anchor_years
            ),
            "stitched_annual_cost_drag": float(
                stitched_frame["cost"].mean() * walk_forward.annualization
            ),
            "anchor_annual_cost_drag": float(
                anchor_frame["cost"].mean() * walk_forward.annualization
            ),
            "hac_convention": _HAC_CONVENTION,
            "cagr_convention": _CAGR_CONVENTION,
            "selection_adjusted": False,
            "permitted_use": permitted_use,
        }
    )
    return WalkForwardReport(
        label=label,
        mode=mode,
        folds=folds,
        stitched_path=stitched,
        daily=daily,
        decision_intents=spliced,
        summary=summary,
    )


def calendar_boundaries(
    start: str,
    end: str,
    *,
    step_years: int = 1,
) -> tuple[str, ...]:
    """Year-end boundaries strictly inside ``(start, end)``.

    A convenience so a caller cannot accidentally place a boundary on or after
    the evaluation end, which would produce a fold with no sessions in it.
    """

    if step_years < 1:
        raise ValueError("step_years must be positive")
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    stamps = pd.date_range(first, last, freq=pd.offsets.YearBegin())
    chosen = [stamp for stamp in stamps if first < stamp < last]
    return tuple(
        stamp.date().isoformat() for stamp in chosen[step_years - 1 :: step_years]
    )


__all__ = [
    "ALL_PROCEDURES",
    "CSCV_COLUMNS",
    "CSCV_SPLIT_COLUMNS",
    "DAILY_FREQUENCY",
    "DEFAULT_MINIMUM_CONFIGURATIONS",
    "DEFAULT_MINIMUM_ROWS_PER_SUBMATRIX",
    "DEFAULT_SUBMATRICES",
    "FAMILY_WISE_COLUMNS",
    "MEAN_STATISTIC",
    "MINIMUM_ROWS_PER_SUBMATRIX_BY_FREQUENCY",
    "MINIMUM_SPA_OBSERVATIONS",
    "MINIMUM_SUBMATRICES",
    "MONTHLY_FREQUENCY",
    "REALITY_CHECK",
    "RETROSPECTIVE_END",
    "RETROSPECTIVE_START",
    "SHARPE_STATISTIC",
    "SPA_CONSISTENT",
    "SPA_LOWER",
    "SPA_PROCEDURES",
    "SPA_UPPER",
    "SPLICEABLE_CONFIG_FIELDS",
    "WALK_FORWARD_COLUMNS",
    "WALK_FORWARD_SELECTION_MODE",
    "WALK_FORWARD_STABILITY_MODE",
    "CscvResult",
    "StrategyFamily",
    "WalkForwardCandidate",
    "WalkForwardConfig",
    "WalkForwardReport",
    "anchored_walk_forward",
    "assemble_family",
    "calendar_boundaries",
    "cscv_pbo",
    "family_deflated_sharpe",
    "family_wise_report",
    "hansen_spa",
    "reality_check",
    "spa_threshold",
]
