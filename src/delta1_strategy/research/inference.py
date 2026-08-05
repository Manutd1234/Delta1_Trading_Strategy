"""Paired inferential statistics for DELTA1 challenger-versus-incumbent work.

Comparing two headline Sharpe ratios is not a weak test of an improvement; it
is not a test at all.  Resampling the incumbent's own 1990-2014 daily returns
gives a standard deviation of roughly 0.20 Sharpe, so the +0.10 promotion gate
in ``docs/research-methodology.md`` sits at about half of one standard
deviation of ordinary backtest noise.

The estimator here removes that noise the only way it can be removed without
new data: by resampling the *differential* between two strategies under a
shared index draw, so the common market history cancels.  A challenger
correlated at 0.99 with the incumbent has a differential standard deviation
near 0.03 Sharpe, at which point the promotion gate becomes a three-sigma
statement rather than a coin flip.

Two properties of this module are deliberate and load-bearing.

First, every interval is a *sampling-uncertainty* interval.  It says nothing
about selection.  ``Selection adjusted`` is False on every row emitted here,
exactly as it is throughout ``diagnostics``.

Second, the Deflated Sharpe Ratio is wired to fail closed.  The incumbent's
historical trial count is unrecoverable, so the statistic is not estimable for
this lineage, and ``research-methodology.md`` makes a non-estimable
multiplicity statistic blocking rather than waivable.  ``deflated_sharpe_ratio``
therefore takes the trial count as a mandatory argument with no default and
returns a result object that cannot carry a number alongside a NOT_ESTIMABLE
status.  There is no code path that quietly produces a value.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_PAIRED_SEED = 20_260_805
DEFAULT_BLOCK_LENGTHS = (21, 63, 126)
DEFAULT_SAMPLES = 10_000
DEFAULT_HAC_LAGS = 21

ESTIMATED = "ESTIMATED"
NOT_ESTIMABLE = "NOT_ESTIMABLE"

PAIRED_REPORT_COLUMNS = [
    "Comparison",
    "Statistic",
    "Method",
    "Expected block sessions",
    "Samples",
    "Seed",
    "Sessions",
    "Return correlation with incumbent",
    "Tracking error (annualized)",
    "Point estimate",
    "Bootstrap standard error",
    "Confidence lower",
    "Confidence upper",
    "Confidence method",
    "Minimum detectable effect (95% one-sided)",
    "Selection adjusted",
    "Permitted use",
]

_PERMITTED_USE = (
    "sampling uncertainty of a paired differential on reused history; "
    "not selection adjusted and does not license promotion"
)


@dataclass(frozen=True)
class InferenceResult:
    """A statistic that is either estimated or explicitly not estimable.

    ``value`` must be ``None`` whenever ``status`` is ``NOT_ESTIMABLE``.  The
    invariant is enforced in ``__post_init__`` rather than by convention so a
    downstream formatter cannot render a sentinel float as a real number.
    """

    statistic: str
    status: str
    value: float | None
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {ESTIMATED, NOT_ESTIMABLE}:
            raise ValueError(f"unknown inference status: {self.status!r}")
        if self.status == NOT_ESTIMABLE and self.value is not None:
            raise ValueError("a NOT_ESTIMABLE result cannot carry a value")
        if self.status == ESTIMATED and (
            self.value is None or not np.isfinite(self.value)
        ):
            raise ValueError("an ESTIMATED result requires a finite value")
        if not self.statistic or not self.reason:
            raise ValueError("statistic and reason must both be non-empty")


def _stationary_indices(
    rng: np.random.Generator,
    source_length: int,
    samples: int,
    horizon: int,
    expected_block_length: int,
) -> np.ndarray:
    """Return stationary-bootstrap indices with circular continuation.

    Mirrors ``diagnostics._stationary_indices`` deliberately: the paired
    estimator must resample on exactly the same dependence model as the
    existing drawdown and CAGR bootstraps, or the two families of numbers in
    the bundle would not be comparable.
    """

    if source_length <= 0:
        raise ValueError("source_length must be positive")
    if expected_block_length <= 0:
        raise ValueError("expected_block_length must be positive")
    indices = np.empty((samples, horizon), dtype=np.int64)
    indices[:, 0] = rng.integers(0, source_length, size=samples)
    restart_probability = 1.0 / expected_block_length
    for offset in range(1, horizon):
        restart = rng.random(samples) < restart_probability
        continuation = (indices[:, offset - 1] + 1) % source_length
        fresh = rng.integers(0, source_length, size=samples)
        indices[:, offset] = np.where(restart, fresh, continuation)
    return indices


def _aligned_pair(
    incumbent: pd.Series,
    challenger: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Validate that two return series share one index, and return them.

    Pairing is the entire source of statistical power.  A single date present
    in one series and not the other silently breaks it, so this fails closed
    rather than taking an intersection.
    """

    if not isinstance(incumbent, pd.Series) or not isinstance(challenger, pd.Series):
        raise TypeError("incumbent and challenger must both be pandas Series")
    incumbent = incumbent.dropna()
    challenger = challenger.dropna()
    if incumbent.empty or challenger.empty:
        raise ValueError("incumbent and challenger must both be non-empty")
    if not incumbent.index.equals(challenger.index):
        raise ValueError(
            "paired inference requires identical indexes; "
            f"incumbent has {len(incumbent)} sessions and challenger has "
            f"{len(challenger)}"
        )
    left = incumbent.to_numpy(dtype=float)
    right = challenger.to_numpy(dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired inference requires finite returns")
    return left, right, incumbent.index


def _sharpe(values: np.ndarray, annualization: int) -> np.ndarray:
    """Annualized Sharpe along the last axis, with rf = 0 by convention."""

    mean = values.mean(axis=-1)
    deviation = values.std(axis=-1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(deviation > 0, mean / deviation, np.nan)
    return ratio * math.sqrt(annualization)


def _cagr(values: np.ndarray, annualization: int) -> np.ndarray:
    growth = np.log1p(values).sum(axis=-1)
    years = values.shape[-1] / annualization
    return np.expm1(growth / years)


def hac_standard_error(differential: np.ndarray, lags: int = DEFAULT_HAC_LAGS) -> float:
    """Newey-West standard error of the mean with a Bartlett kernel.

    Independent of the bootstrap, so two methods that agree are much stronger
    evidence than either alone.  On material disagreement the caller should
    fail closed to the wider interval.
    """

    values = np.asarray(differential, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("differential must be a one-dimensional series")
    if lags < 0:
        raise ValueError("lags cannot be negative")
    count = values.size
    centered = values - values.mean()
    variance = float(centered @ centered) / count
    for lag in range(1, min(lags, count - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(centered[lag:] @ centered[:-lag]) / count
        variance += 2.0 * weight * covariance
    if variance <= 0:
        return float("inf")
    return math.sqrt(variance / count)


def paired_block_bootstrap_differential(
    incumbent: pd.Series,
    challenger: pd.Series,
    *,
    comparison: str,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_PAIRED_SEED,
    annualization: int = 252,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Sampling distribution of challenger-minus-incumbent statistics.

    One index vector is drawn per replicate and applied to *both* series.
    Those common random numbers are what make the estimator powerful: the
    shared market history cancels in the differential, leaving only the part
    of the variation that the change itself caused.

    Returns one row per (statistic, block length).  ``Confidence lower`` is the
    more conservative of the percentile and basic-bootstrap bounds; callers
    should additionally report the minimum across block lengths so the
    friendliest dependence assumption cannot be chosen after the fact.
    """

    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")
    left, right, index = _aligned_pair(incumbent, challenger)
    count = left.size

    observed_correlation = float(np.corrcoef(left, right)[0, 1])
    differential = right - left
    tracking_error = float(differential.std(ddof=0) * math.sqrt(annualization))

    observed = {
        "Annualized mean differential": float(
            differential.mean() * annualization
        ),
        "Sharpe differential": float(
            _sharpe(right, annualization) - _sharpe(left, annualization)
        ),
        "CAGR differential": float(
            _cagr(right, annualization) - _cagr(left, annualization)
        ),
    }

    method_seed = zlib.crc32(b"paired stationary block bootstrap") & 0xFFFFFFFF
    comparison_seed = zlib.crc32(comparison.encode("utf-8")) & 0xFFFFFFFF

    rows: list[dict[str, object]] = []
    for block_length in block_lengths:
        entropy = np.random.SeedSequence(
            [seed, method_seed, comparison_seed, int(block_length)]
        )
        rng = np.random.default_rng(entropy)
        draws: dict[str, list[np.ndarray]] = {name: [] for name in observed}
        remaining = samples
        while remaining > 0:
            take = min(batch_size, remaining)
            indices = _stationary_indices(rng, count, take, count, int(block_length))
            resampled_left = left[indices]
            resampled_right = right[indices]
            resampled_differential = resampled_right - resampled_left
            draws["Annualized mean differential"].append(
                resampled_differential.mean(axis=1) * annualization
            )
            draws["Sharpe differential"].append(
                _sharpe(resampled_right, annualization)
                - _sharpe(resampled_left, annualization)
            )
            draws["CAGR differential"].append(
                _cagr(resampled_right, annualization)
                - _cagr(resampled_left, annualization)
            )
            remaining -= take

        for statistic, batches in draws.items():
            distribution = np.concatenate(batches)
            distribution = distribution[np.isfinite(distribution)]
            if distribution.size == 0:
                raise ValueError(
                    f"bootstrap produced no finite draws for {statistic!r}"
                )
            estimate = observed[statistic]
            standard_error = float(distribution.std(ddof=1))
            percentile_lower = float(np.percentile(distribution, 5.0))
            percentile_upper = float(np.percentile(distribution, 95.0))
            # Basic-bootstrap bounds reflect the draw distribution around the
            # observed estimate rather than around itself; reporting the more
            # conservative of the two costs nothing and removes a choice.
            basic_lower = 2.0 * estimate - percentile_upper
            basic_upper = 2.0 * estimate - percentile_lower
            rows.append(
                {
                    "Comparison": comparison,
                    "Statistic": statistic,
                    "Method": (
                        "paired stationary block bootstrap; shared index draw"
                    ),
                    "Expected block sessions": int(block_length),
                    "Samples": int(samples),
                    "Seed": int(seed),
                    "Sessions": int(count),
                    "Return correlation with incumbent": observed_correlation,
                    "Tracking error (annualized)": tracking_error,
                    "Point estimate": estimate,
                    "Bootstrap standard error": standard_error,
                    "Confidence lower": min(percentile_lower, basic_lower),
                    "Confidence upper": max(percentile_upper, basic_upper),
                    "Confidence method": (
                        "minimum of percentile and basic bootstrap bounds"
                    ),
                    "Minimum detectable effect (95% one-sided)": (
                        1.645 * standard_error
                    ),
                    "Selection adjusted": False,
                    "Permitted use": _PERMITTED_USE,
                }
            )

    return pd.DataFrame(rows, columns=PAIRED_REPORT_COLUMNS)


def paired_hac_report(
    incumbent: pd.Series,
    challenger: pd.Series,
    *,
    comparison: str,
    lags: int = DEFAULT_HAC_LAGS,
    annualization: int = 252,
) -> pd.Series:
    """Newey-West cross-check on the mean differential.

    Deliberately a separate estimator from the bootstrap.  Agreement between
    the two is the evidence; disagreement is a signal to widen, not to pick.
    """

    left, right, _ = _aligned_pair(incumbent, challenger)
    differential = right - left
    standard_error = hac_standard_error(differential, lags)
    mean = float(differential.mean())
    t_statistic = mean / standard_error if standard_error > 0 else float("nan")
    return pd.Series(
        {
            "Comparison": comparison,
            "Statistic": "Annualized mean differential",
            "Method": f"Newey-West HAC ({lags} lags)",
            "Point estimate": mean * annualization,
            "Standard error (annualized)": standard_error * annualization,
            "t statistic": t_statistic,
            "Confidence lower": (mean - 1.645 * standard_error) * annualization,
            "Selection adjusted": False,
            "Permitted use": _PERMITTED_USE,
        }
    )


def minimum_detectable_effect(
    tracking_error_annual: float,
    volatility_annual: float,
    years: float,
    *,
    confidence_z: float = 1.645,
) -> pd.Series:
    """Closed-form power pre-flight, computable before any return is compared.

    ``SE(dSharpe) ~ TE / (sigma * sqrt(years))`` matches the paired bootstrap
    to a few percent and costs nothing, so a comparison can be declared
    underpowered *before* its point estimate exists.  That ordering is the
    control which keeps a power calculation from becoming a rationalization.
    """

    for name, value in (
        ("tracking_error_annual", tracking_error_annual),
        ("volatility_annual", volatility_annual),
        ("years", years),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    root_years = math.sqrt(years)
    sharpe_error = tracking_error_annual / (volatility_annual * root_years)
    cagr_error = tracking_error_annual / root_years
    return pd.Series(
        {
            "Tracking error (annualized)": tracking_error_annual,
            "Standard error of Sharpe differential": sharpe_error,
            "Standard error of CAGR differential": cagr_error,
            "Minimum detectable effect (95% one-sided)": confidence_z * sharpe_error,
            "Minimum detectable CAGR effect (95% one-sided)": (
                confidence_z * cagr_error
            ),
        }
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    trial_count: int | None,
    trial_sharpe_variance: float | None,
    sessions: int,
    skewness: float,
    excess_kurtosis: float,
) -> InferenceResult:
    """Bailey and Lopez de Prado deflated Sharpe, or an explicit refusal.

    ``trial_count`` and ``trial_sharpe_variance`` are mandatory and have no
    defaults on purpose.  For the incumbent lineage both are unrecoverable, so
    the honest output is a refusal rather than an optimistic number computed
    from an assumed family size.
    """

    if trial_count is None or trial_sharpe_variance is None:
        return InferenceResult(
            statistic="deflated_sharpe_ratio",
            status=NOT_ESTIMABLE,
            value=None,
            reason=(
                "the trial family is not fully recoverable, so the "
                "multiplicity correction cannot be computed"
            ),
        )
    if trial_count < 1:
        raise ValueError("trial_count must be at least 1")
    if trial_sharpe_variance <= 0:
        raise ValueError("trial_sharpe_variance must be positive")
    if sessions < 2:
        raise ValueError("sessions must be at least 2")

    # Expected maximum of `trial_count` draws under the null, via the standard
    # Gumbel approximation used in the original paper.
    euler = 0.577_215_664_901_532_9
    trials = float(trial_count)
    if trials <= 1.0:
        expected_maximum = 0.0
    else:
        from statistics import NormalDist

        normal = NormalDist()
        expected_maximum = math.sqrt(trial_sharpe_variance) * (
            (1.0 - euler) * normal.inv_cdf(1.0 - 1.0 / trials)
            + euler * normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
        )

    denominator = 1.0 - skewness * observed_sharpe + (
        (excess_kurtosis) / 4.0
    ) * observed_sharpe**2
    if denominator <= 0:
        return InferenceResult(
            statistic="deflated_sharpe_ratio",
            status=NOT_ESTIMABLE,
            value=None,
            reason=(
                "the higher-moment adjustment is non-positive, so the "
                "deflated statistic is undefined for this return distribution"
            ),
        )
    from statistics import NormalDist

    statistic = (
        (observed_sharpe - expected_maximum)
        * math.sqrt(sessions - 1.0)
        / math.sqrt(denominator)
    )
    return InferenceResult(
        statistic="deflated_sharpe_ratio",
        status=ESTIMATED,
        value=float(NormalDist().cdf(statistic)),
        reason=f"estimated over a declared family of {trial_count} trials",
    )


def promotion_support(
    deflated: InferenceResult,
    *,
    window_label: str,
) -> pd.Series:
    """Whether the evidence can support promotion, decided mechanically.

    For any challenger evaluated on the reused 1990-2014 history this returns
    False regardless of its point estimates, because the deflated Sharpe is not
    estimable for the lineage and ``research-methodology.md`` treats a
    non-estimable multiplicity statistic as blocking.  Encoding that as a
    return value rather than a committee discussion is the point.
    """

    supported = deflated.status == ESTIMATED
    return pd.Series(
        {
            "Window": window_label,
            "Deflated Sharpe status": deflated.status,
            "Deflated Sharpe reason": deflated.reason,
            "Promotion supported": supported,
            "Promotion reason": (
                "multiplicity correction estimated"
                if supported
                else "deflated_sharpe_not_estimable_for_this_lineage"
            ),
            "Selection adjusted": False,
        }
    )


__all__ = [
    "DEFAULT_BLOCK_LENGTHS",
    "DEFAULT_HAC_LAGS",
    "DEFAULT_PAIRED_SEED",
    "DEFAULT_SAMPLES",
    "ESTIMATED",
    "NOT_ESTIMABLE",
    "InferenceResult",
    "PAIRED_REPORT_COLUMNS",
    "deflated_sharpe_ratio",
    "hac_standard_error",
    "minimum_detectable_effect",
    "paired_block_bootstrap_differential",
    "paired_hac_report",
    "promotion_support",
]
