"""Causal regime states for the ETF dynamic regime allocation sleeve.

Every state in this module is a *label on a session*, and the label on session
``t`` is computable from data published strictly before ``t``.  That is one
displacement.  The allocation layer adds one further session of execution lag,
which reproduces the repository's full-bar ``shift(2)`` convention from
``docs/causality.md``: a decision taken from session ``t-1``'s close fills at
session ``t``'s close and first earns the close-to-close move ending at ``t+1``.

What this module does NOT prove
-------------------------------
**It does not prove any of these states forecasts anything.**  Nothing here is
scored, ranked or compared.  A regime label is a description of the information
set at a point in time; whether conditioning on it improves an allocation is a
separate question, answered by the walk-forward and not by this file.

**It does not correct survivorship.**  Every state is computed from a panel in
which no closed or delisted fund appears.  Frames carry
``etfs.SURVIVORSHIP_LIMITATION`` for that reason and downstream artifacts must
keep it.

**It does not make a fitted state safe by being causal in shape.**  Two objects
here are estimated rather than defined --- the Moreira-Muir scaling constant and
the Markov-switching parameter vector --- and both take a mandatory
``fit_window``.  Fitting either on the full sample and applying it historically
is look-ahead of exactly the kind ``docs/causality.md`` forbids, because the
likelihood at every date then depends on every observation including the future.
``fit_markov_switching`` refuses a window that reaches the last observation
unless the caller explicitly asks for a descriptive in-sample fit, and the
result of such a fit must never drive a weight.

**It does not read a smoothed probability anywhere.**  The Markov state is the
*filtered* probability ``Pr(s_t = k | y_1..y_t)``.  The smoothed probability
``Pr(s_t = k | y_1..y_T)`` is what most library examples hand you and it
conditions on the whole sample; a series built from it fails the truncation test
in ``tests/test_regimes.py`` immediately and loudly, which is why that test
exists separately from the parameter test.

**It does not resolve the Forbes-Rigobon bias by ignoring it.**  A correlation
estimated on a subsample selected for high volatility is mechanically inflated.
``forbes_rigobon_adjusted_correlation`` is provided so any artifact that
compares correlation across volatility states can report the adjusted figure, or
state plainly that the comparison is unadjusted and therefore overstates the
increase.

Thin windows produce no label
-----------------------------
Where a state cannot be honestly computed it is ``pd.NA``, never a default.
A trend sleeve with fewer than ten complete monthly closes after its own
inception is excluded from the risk-on set rather than defaulted to on;
defaulting to on would backfill a bullish prior into every young sleeve's first
ten months.  Scalar statistics that cannot be computed return
``InferenceResult(status=NOT_ESTIMABLE)`` from ``research.inference`` rather than
an optimistic number.

References
----------
Faber (2007) for the ten-month moving-average trend state; Moskowitz, Ooi and
Pedersen (2012) for time-series momentum and the exponentially weighted
volatility convention; Barroso and Santa-Clara (2015) and Moreira and Muir
(2017) for the two volatility-scaling forms, with Harvey, Hoyle, Rattray,
Sargaison and Van Hemert (2018) for the caveat that the benefit is concentrated
in equity-like assets; Pollet and Wilson (2010) and Kritzman, Li, Page and
Rigobon (2011) for the diversification states; Longin and Solnik (2001), Ang and
Chen (2002) and Forbes and Rigobon (2002) for the conditioning bias; Daniel and
Moskowitz (2016) and Cooper, Gutierrez and Hameed (2004) for the bear and market
states; Hamilton (1989) and Ang and Bekaert (2002) for the switching model.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from delta1_strategy.marketdata.etfs import SURVIVORSHIP_LIMITATION
from delta1_strategy.research.attribution import trailing_mean_pairwise_correlation
from delta1_strategy.research.inference import (
    DEFAULT_PAIRED_SEED,
    ESTIMATED,
    NOT_ESTIMABLE,
    InferenceResult,
)


RISK_ON = "Risk on"
RISK_OFF = "Risk off"

LOW = "Low"
MIDDLE = "Middle"
HIGH = "High"

BEAR = "Bear"
NOT_BEAR = "Not bear"

PANIC = "Panic"
NOT_PANIC = "Not panic"

STRESS = "Systemic stress"
NOT_STRESS = "No systemic stress"

#: Ascending fitted mean anchors the Markov labels, so fold-to-fold state
#: numbering is comparable.  Without the anchor, EM's label permutation makes a
#: fold table meaningless.
MARKOV_STATE_LABELS: tuple[str, ...] = ("Low mean", "High mean")

REGIME_STATE_COLUMNS: tuple[str, ...] = (
    "Regime",
    "State",
    "Sessions",
    "Share of labelled sessions",
    "Share of all sessions",
    "Episodes",
    "Mean episode duration (sessions)",
    "Median episode duration (sessions)",
    "Longest episode (sessions)",
    "First labelled session",
    "Last labelled session",
    "Limitation",
)

REGIME_TRANSITION_COLUMNS: tuple[str, ...] = (
    "Regime",
    "From state",
    "To state",
    "Transitions",
    "Transition probability",
    "Limitation",
)

_FULL_SAMPLE_FIT_MESSAGE = (
    "the fit window reaches the last observation, so the parameters would be "
    "estimated from data the state is then applied to; fit inside a training "
    "fold, or pass full_sample_descriptive_fit=True for a description whose "
    "output must never drive a weight"
)


# --------------------------------------------------------------------------
# shared causal machinery
# --------------------------------------------------------------------------


def _validated_lag(lag: int) -> int:
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive whole number of sessions")
    return lag


def _validated_window(window: int, label: str, minimum: int = 2) -> int:
    if isinstance(window, bool) or not isinstance(window, int) or window < minimum:
        raise ValueError(f"{label} must be a whole number of at least {minimum}")
    return window


def _validated_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must be indexed by session date")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be unique and increasing")
    return frame


def _validated_series(series: pd.Series, label: str) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError(f"{label} must be a non-empty Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must be indexed by session date")
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be unique and increasing")
    return series


def month_end_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last session of each calendar month present in ``sessions``.

    These are the decision dates.  The final group may be an incomplete month,
    which is correct: a decision taken on the last session available is a real
    decision, and it can only affect sessions after it.
    """

    if not isinstance(sessions, pd.DatetimeIndex) or len(sessions) == 0:
        raise ValueError("sessions must be a non-empty DatetimeIndex")
    positions = pd.Series(np.arange(len(sessions)), index=sessions)
    last = positions.groupby([sessions.year, sessions.month]).max().to_numpy()
    return sessions[np.sort(last)]


def _project(decisions: pd.Series | pd.DataFrame, sessions: pd.DatetimeIndex, lag: int):
    """Carry a decision-date label forward onto the session calendar, then lag.

    The lag is what makes the projection causal: the label on session ``t`` is
    the one decided at the last decision date strictly before ``t``.
    """

    return decisions.reindex(sessions).ffill().shift(lag)


def _as_boolean_array(flags: pd.Series) -> np.ndarray:
    """A plain boolean array, with a missing flag read as False.

    Comparisons against a nullable string column return ``pd.NA`` rather than
    ``False``, and a numpy expression on that raises.  Missing means unlabelled,
    which is never a positive condition, so it collapses to False here and the
    separate ``usable`` mask is what preserves the distinction.
    """

    return flags.fillna(False).to_numpy(dtype=bool, na_value=False)


def _label_from_condition(
    condition: pd.Series,
    usable: pd.Series,
    true_label: str,
    false_label: str,
) -> pd.Series:
    raw = pd.Series(
        np.where(_as_boolean_array(condition), true_label, false_label),
        index=condition.index,
        dtype="string",
    )
    return raw.where(_as_boolean_array(usable), other=pd.NA)


def _trailing_tercile_labels(
    feature: pd.Series,
    *,
    history: int | None,
    minimum_history: int,
) -> pd.Series:
    """Low/Middle/High against a window of strictly prior values of the feature.

    This lifts the shape of ``diagnostics._causal_tercile_labels`` --- quantiles
    from prior history only, with a minimum-history guard and no label before it
    is met --- and generalises the expanding window to a trailing one.  A
    full-sample percentile is the most common way a volatility state leaks, and
    it is exactly what the truncation test catches.
    """

    if isinstance(minimum_history, bool) or not isinstance(minimum_history, int):
        raise ValueError("minimum_history must be a whole number")
    if minimum_history < 2:
        raise ValueError("minimum_history must be at least two observations")
    if history is not None:
        history = _validated_window(history, "history", minimum=minimum_history)

    values = feature.to_numpy(dtype=float)
    labels = pd.Series(pd.NA, index=feature.index, dtype="string")
    finite = np.isfinite(values)
    for offset in range(len(values)):
        if not finite[offset]:
            continue
        start = 0 if history is None else max(0, offset - history)
        window = values[start:offset]
        window = window[np.isfinite(window)]
        if window.size < minimum_history:
            continue
        lower, upper = np.quantile(window, [1.0 / 3.0, 2.0 / 3.0])
        value = values[offset]
        if value <= lower:
            labels.iloc[offset] = LOW
        elif value >= upper:
            labels.iloc[offset] = HIGH
        else:
            labels.iloc[offset] = MIDDLE
    return labels


# --------------------------------------------------------------------------
# trend and momentum states
# --------------------------------------------------------------------------


def faber_trend_state(
    adjusted_close: pd.DataFrame,
    *,
    months: int = 10,
    lag: int = 1,
) -> pd.DataFrame:
    """Faber's ten-month moving-average trend state, one label per sleeve.

    The moving average is over ``months`` *monthly closes inclusive of the
    decision month*, which is complete at the decision instant; that is Faber's
    own construction and only the monthly version reproduces the reference.  A
    210-session daily average is a different rule and gives different states.

    Warm-up is deliberately one month longer than the average itself: the
    sleeve's inception month is a partial month, so a sleeve is labelled only
    once ``months`` *complete* monthly closes exist after inception.  Before
    that the state is ``pd.NA`` and the sleeve is excluded from the risk-on set
    rather than defaulted to on.

    Zero fitted parameters, so there is nothing to fit inside a fold and this
    state is identical in every fold of a walk-forward.
    """

    frame = _validated_frame(adjusted_close, "adjusted_close")
    months = _validated_window(months, "months", minimum=2)
    lag = _validated_lag(lag)

    decisions = month_end_sessions(frame.index)
    monthly = frame.loc[decisions]
    average = monthly.rolling(months, min_periods=months).mean()
    positions = np.arange(len(monthly))

    labels = pd.DataFrame(
        pd.NA, index=monthly.index, columns=monthly.columns, dtype="string"
    )
    for column in monthly.columns:
        present = monthly[column].notna().to_numpy()
        if not present.any():
            continue
        first = int(np.argmax(present))
        seasoned = positions >= first + months
        usable = pd.Series(
            seasoned & average[column].notna().to_numpy() & present,
            index=monthly.index,
        )
        labels[column] = _label_from_condition(
            monthly[column] > average[column], usable, RISK_ON, RISK_OFF
        )
    return _project(labels, frame.index, lag)


def time_series_momentum_state(
    adjusted_close: pd.DataFrame,
    *,
    lookback: int = 252,
    lag: int = 1,
) -> pd.DataFrame:
    """Sign of the trailing twelve-month total return, evaluated monthly.

    The continuous magnitude of the Moskowitz-Ooi-Pedersen signal is *not*
    returned here.  A long-only unlevered book cannot express a short, so the
    state enters as a gate; the magnitude, if used at all, belongs in the
    sleeve-sizing step where the 40%/sigma convention lives.

    A cumulative return of exactly zero is labelled risk-off.  That is a stated
    tie-break, not an estimate.
    """

    frame = _validated_frame(adjusted_close, "adjusted_close")
    lookback = _validated_window(lookback, "lookback", minimum=2)
    lag = _validated_lag(lag)

    decisions = month_end_sessions(frame.index)
    cumulative = frame.div(frame.shift(lookback)).sub(1.0).loc[decisions]
    labels = pd.DataFrame(
        pd.NA, index=cumulative.index, columns=cumulative.columns, dtype="string"
    )
    for column in cumulative.columns:
        labels[column] = _label_from_condition(
            cumulative[column] > 0.0, cumulative[column].notna(), RISK_ON, RISK_OFF
        )
    return _project(labels, frame.index, lag)


# --------------------------------------------------------------------------
# volatility states and volatility scaling
# --------------------------------------------------------------------------


def ewma_volatility(
    returns: pd.DataFrame | pd.Series,
    *,
    centre_of_mass: int = 60,
    min_periods: int = 63,
    annualization: int = 252,
    lag: int = 1,
) -> pd.DataFrame | pd.Series:
    """Exponentially weighted annualized volatility, lagged.

    ``adjust=False`` is pinned so the estimate is a genuine recursion and
    byte-reproducible; both settings are strictly trailing, so the choice is
    about reproducibility rather than causality.  ``min_periods`` must be set or
    the first observations return a volatility estimated from two points, and
    any sizing rule that divides by it explodes.
    """

    _validated_window(centre_of_mass, "centre_of_mass", minimum=1)
    _validated_window(min_periods, "min_periods", minimum=2)
    _validated_window(annualization, "annualization", minimum=1)
    lag = _validated_lag(lag)
    estimate = returns.ewm(
        com=centre_of_mass, min_periods=min_periods, adjust=False
    ).std() * math.sqrt(annualization)
    return estimate.shift(lag)


def realized_volatility(
    returns: pd.DataFrame | pd.Series,
    *,
    window: int = 126,
    annualization: int = 252,
    demean: bool = False,
    lag: int = 1,
) -> pd.DataFrame | pd.Series:
    """Trailing realized volatility, annualized and lagged.

    ``demean`` names which estimator is in use and the choice must be stated in
    any artifact: Moreira and Muir take squared deviations from the within-window
    mean, Barroso and Santa-Clara take raw squared returns.  The default is the
    raw form, matching the scaling rule this module recommends.

    With ``lag=1`` the window ends at ``t-1`` and excludes the session being
    labelled.  A variance that includes the day being traded is the same leak in
    a different costume.
    """

    _validated_window(window, "window", minimum=2)
    _validated_window(annualization, "annualization", minimum=1)
    if not isinstance(demean, bool):
        raise ValueError("demean must be a bool")
    lag = _validated_lag(lag)
    if demean:
        variance = returns.rolling(window, min_periods=window).var(ddof=0)
    else:
        variance = returns.pow(2.0).rolling(window, min_periods=window).mean()
    return (variance * annualization).pow(0.5).shift(lag)


def volatility_state(
    returns: pd.Series,
    *,
    window: int = 21,
    history: int | None = 1260,
    minimum_history: int = 504,
    annualization: int = 252,
    demean: bool = False,
    lag: int = 1,
) -> pd.Series:
    """Low/Middle/High realized-volatility state from strictly prior history.

    Two pre-declared window choices exist in the literature and only one may be
    used in a given artifact: 21 sessions (Moreira and Muir's one month) or 126
    (Barroso and Santa-Clara's six months).  The default here is 21; whichever
    is chosen must be declared before any return is computed.

    Below ``minimum_history`` prior estimates the state is ``pd.NA`` and the
    caller falls back to its pre-declared neutral state.
    """

    series = _validated_series(returns, "returns")
    estimate = realized_volatility(
        series,
        window=window,
        annualization=annualization,
        demean=demean,
        lag=lag,
    )
    return _trailing_tercile_labels(
        estimate, history=history, minimum_history=minimum_history
    )


def barroso_santa_clara_scale(
    returns: pd.Series,
    *,
    window: int = 126,
    volatility_budget_annualized: float = 0.12,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    lag: int = 1,
) -> pd.Series:
    """Volatility scaling with no fitted constant.

    ``sigmahat_t`` is the annualized root mean squared daily return over the
    previous ``window`` sessions and the weight is the budget divided by it,
    clipped.  Nothing is estimated on the sample the scale is then applied to,
    which is why this is the safer default of the two scaling forms.

    ``max_scale`` defaults to 1.0 because this book is unlevered and long only;
    any value above one is a leverage decision requiring a stated margin policy,
    not a research result.  The undeployed fraction is the caller's to route to
    the short-duration Treasury sleeve.

    Harvey et al. (2018) find volatility scaling raises risk-adjusted return
    mainly for equity-like assets and much less for bonds and commodities, so a
    multi-asset book is not entitled to expect the single-factor improvements
    the source papers report.
    """

    series = _validated_series(returns, "returns")
    for name, value in (
        ("volatility_budget_annualized", volatility_budget_annualized),
        ("min_scale", min_scale),
        ("max_scale", max_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(min_scale) >= float(max_scale):
        raise ValueError("min_scale must be below max_scale")
    if float(volatility_budget_annualized) <= 0.0:
        raise ValueError("volatility_budget_annualized must be positive")

    sigma = realized_volatility(series, window=window, demean=False, lag=lag)
    scale = float(volatility_budget_annualized) / sigma.replace(0.0, np.nan)
    return scale.clip(float(min_scale), float(max_scale))


@dataclass(frozen=True)
class VolatilityScaling:
    """A scaling series and the constant that was fitted to produce it."""

    scale: pd.Series
    constant: InferenceResult
    fit_start: str
    fit_end: str
    window: int
    max_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.scale, pd.Series):
            raise ValueError("scale must be a pandas Series")
        if not isinstance(self.constant, InferenceResult):
            raise ValueError("constant must be an InferenceResult")
        for name in ("fit_start", "fit_end"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty ISO date string")
        if isinstance(self.window, bool) or not isinstance(self.window, int):
            raise ValueError("window must be a whole number")
        if self.window < 2:
            raise ValueError("window must be at least two sessions")
        if (
            isinstance(self.max_scale, bool)
            or not isinstance(self.max_scale, (int, float, np.floating))
            or not np.isfinite(float(self.max_scale))
            or float(self.max_scale) <= 0.0
        ):
            raise ValueError("max_scale must be finite and positive")


def _fit_slice(series: pd.Series, fit_window: tuple[object, object]) -> pd.Series:
    if not isinstance(fit_window, tuple) or len(fit_window) != 2:
        raise ValueError("fit_window must be a (start, end) tuple of dates")
    start, end = (pd.Timestamp(fit_window[0]), pd.Timestamp(fit_window[1]))
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("fit_window start must precede fit_window end")
    return series.loc[(series.index >= start) & (series.index <= end)]


def moreira_muir_scale(
    returns: pd.Series,
    *,
    fit_window: tuple[object, object],
    window: int = 21,
    max_scale: float = 1.0,
    min_scale: float = 0.0,
    lag: int = 1,
    full_sample_descriptive_fit: bool = False,
) -> VolatilityScaling:
    """Moreira-Muir inverse-variance scaling, with the constant fitted in fold.

    ``c`` is chosen so the scaled series carries the same unconditional standard
    deviation as the unscaled one.  In the source paper it is a full-sample
    constant, and using it that way is a full-sample normalisation: the scale
    applied in 2008 would then depend on what happened in 2017.  Here ``c`` is
    estimated on ``fit_window`` only and frozen through everything after it, and
    the fitted value is reported so a fold table can carry it as a column.

    The variance window ends at ``t - lag`` and excludes the session being
    scaled.  If the fit window is too short or degenerate the constant is
    ``NOT_ESTIMABLE`` and the scale series is empty of values rather than
    silently falling back to one.
    """

    series = _validated_series(returns, "returns")
    window = _validated_window(window, "window", minimum=2)
    lag = _validated_lag(lag)
    if not isinstance(full_sample_descriptive_fit, bool):
        raise ValueError("full_sample_descriptive_fit must be a bool")
    for name, value in (("max_scale", max_scale), ("min_scale", min_scale)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.floating))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if float(min_scale) >= float(max_scale):
        raise ValueError("min_scale must be below max_scale")

    training = _fit_slice(series, fit_window)
    if not full_sample_descriptive_fit and len(training) and (
        training.index[-1] >= series.index[-1]
    ):
        raise ValueError(_FULL_SAMPLE_FIT_MESSAGE)

    variance = (
        series.pow(2.0).rolling(window, min_periods=window).mean().shift(lag)
    ).replace(0.0, np.nan)
    unscaled_managed = series / variance

    fitted = unscaled_managed.loc[training.index].dropna()
    paired = training.reindex(fitted.index).dropna()
    fitted = fitted.reindex(paired.index)
    if len(paired) < window + 2 or float(fitted.std(ddof=1)) <= 0.0:
        constant = InferenceResult(
            statistic="Moreira-Muir volatility scaling constant",
            status=NOT_ESTIMABLE,
            value=None,
            reason=(
                f"the fit window yields {len(paired)} paired observations, which "
                "cannot identify a scaling constant"
            ),
        )
        scale = pd.Series(np.nan, index=series.index, dtype=float)
    else:
        value = float(paired.std(ddof=1) / fitted.std(ddof=1))
        constant = InferenceResult(
            statistic="Moreira-Muir volatility scaling constant",
            status=ESTIMATED,
            value=value,
            reason=(
                f"variance-matched on {len(paired)} sessions inside the fit window "
                f"{training.index[0].date()}..{training.index[-1].date()}"
            ),
        )
        scale = (value / variance).clip(float(min_scale), float(max_scale))

    return VolatilityScaling(
        scale=scale,
        constant=constant,
        fit_start=str(pd.Timestamp(fit_window[0]).date()),
        fit_end=str(pd.Timestamp(fit_window[1]).date()),
        window=window,
        max_scale=float(max_scale),
    )


# --------------------------------------------------------------------------
# diversification states
# --------------------------------------------------------------------------


def mean_pairwise_correlation(
    returns: pd.DataFrame,
    *,
    window: int = 63,
    minimum_markets: int = 5,
    minimum_coverage: float = 0.8,
    lag: int = 1,
    drop_zero_observations: bool = True,
) -> pd.Series:
    """Mean off-diagonal correlation of asset returns over a trailing window.

    With ``drop_zero_observations=True`` this delegates verbatim to
    ``attribution.trailing_mean_pairwise_correlation``, so the ETF and futures
    sleeves report a numerically comparable statistic.  That function's
    endogeneity caveat --- that the estimate embeds position sizes and is partly
    endogenous to the strategy's own positioning --- applies to its *P&L* usage
    and NOT to this one.  Fed asset returns it is an exogenous market-correlation
    estimate, and the caveat must not be copied across.

    One property of the house function does need stating, because it was written
    for P&L: it maps an exact zero to a missing observation, on the reasoning
    that a zero P&L means no position.  A zero *asset return* is a real
    observation, and this panel has many of them --- a short-duration Treasury
    fund quoted to the cent prints an exactly flat close on roughly a tenth of
    its sessions.  ``drop_zero_observations=False`` computes the identical
    statistic with exact zeros retained, under the same double-displacement
    convention, so the size of that effect can be measured rather than assumed
    away.
    """

    frame = _validated_frame(returns, "returns")
    if not isinstance(drop_zero_observations, bool):
        raise ValueError("drop_zero_observations must be a bool")
    if drop_zero_observations:
        return trailing_mean_pairwise_correlation(
            frame,
            window=window,
            minimum_markets=minimum_markets,
            minimum_coverage=minimum_coverage,
            lag=lag,
        )

    window = _validated_window(window, "window", minimum=2)
    if isinstance(minimum_markets, bool) or not isinstance(minimum_markets, int):
        raise ValueError("minimum_markets must be a whole number")
    if minimum_markets < 2:
        raise ValueError("minimum_markets must be at least two")
    if not 0 < float(minimum_coverage) <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    lag = _validated_lag(lag)

    threshold = int(float(minimum_coverage) * window)
    values = np.full(len(frame.index), np.nan)
    for position in range(window, len(frame.index)):
        block = frame.iloc[position - window : position]
        block = block.dropna(axis=1, thresh=threshold)
        if block.shape[1] < minimum_markets:
            continue
        matrix = block.corr().to_numpy()
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size:
            values[position] = float(upper.mean())
    return pd.Series(values, index=frame.index).shift(lag)


def absorption_ratio(
    returns: pd.DataFrame,
    *,
    window: int = 252,
    components: int | None = None,
    lag: int = 1,
) -> pd.Series:
    """Share of panel variance explained by its leading principal components.

    ``components`` defaults to Kritzman et al.'s fixed fraction, ``round(N/5)``
    with a floor of one.  A high ratio means the risk sources are unified: the
    diversification the book is built on has disappeared.

    Two coarseness disclosures travel with this number.  The panel has eleven
    columns rather than the paper's fifty-one sectors, so the ratio is far
    coarser and its leading eigenvalue is dominated by the equity sleeves.  And
    only the eigenvalue *sum* is used --- no eigenvector loading is read --- so
    the sign indeterminacy of an eigen-decomposition never enters.
    """

    frame = _validated_frame(returns, "returns")
    window = _validated_window(window, "window", minimum=3)
    lag = _validated_lag(lag)
    width = frame.shape[1]
    if width < 2:
        raise ValueError("the absorption ratio needs at least two columns")
    if components is None:
        components = max(1, int(round(width / 5)))
    if isinstance(components, bool) or not isinstance(components, int):
        raise ValueError("components must be a whole number")
    if not 1 <= components <= width:
        raise ValueError(f"components must be in [1, {width}]")

    matrix = frame.to_numpy(dtype=float)
    values = np.full(len(frame.index), np.nan)
    for position in range(window - 1, len(frame.index)):
        block = matrix[position - window + 1 : position + 1]
        if not np.isfinite(block).all():
            continue
        covariance = np.cov(block, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
        total = float(eigenvalues.sum())
        if total <= 0.0:
            continue
        values[position] = float(eigenvalues[:components].sum() / total)
    return pd.Series(values, index=frame.index).shift(lag)


def absorption_ratio_shift(
    ratio: pd.Series,
    *,
    fast: int = 15,
    slow: int = 252,
) -> pd.Series:
    """Kritzman et al.'s standardized absorption shift, on trailing moments only.

    ``(MA_fast - MA_slow) / sd_slow``.  A z-score against the full-sample mean
    and standard deviation would be a full-sample normalisation and would break
    truncation invariance; every moment here is trailing, and the input series is
    already lagged, so no further displacement is applied.
    """

    series = _validated_series(ratio, "ratio")
    fast = _validated_window(fast, "fast", minimum=2)
    slow = _validated_window(slow, "slow", minimum=3)
    if fast >= slow:
        raise ValueError("fast must be a shorter window than slow")
    quick = series.rolling(fast, min_periods=fast).mean()
    trend = series.rolling(slow, min_periods=slow).mean()
    dispersion = series.rolling(slow, min_periods=slow).std(ddof=1)
    return (quick - trend) / dispersion.replace(0.0, np.nan)


def systemic_stress_state(
    shift: pd.Series,
    *,
    threshold: float = 1.0,
) -> pd.Series:
    """Label the standardized absorption shift against a pre-declared cut.

    Kritzman et al. flag systemic stress at a shift of one standard deviation.
    The threshold is a constant taken from the paper, not a quantile of this
    sample: ``attribution.conditional_exposure_diagnostic`` documents why a cut
    point taken over the same sample the comparison is measured on is fitted in
    sample by construction.
    """

    series = _validated_series(shift, "shift")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.floating))
        or not np.isfinite(float(threshold))
    ):
        raise ValueError("threshold must be a finite real number")
    return _label_from_condition(
        series >= float(threshold), series.notna(), STRESS, NOT_STRESS
    )


def forbes_rigobon_adjusted_correlation(
    correlation: float,
    variance_ratio: float,
) -> InferenceResult:
    """Correlation corrected for conditioning on a high-volatility subsample.

    ``rho* = rho / sqrt(1 + delta (1 - rho^2))`` with
    ``delta = sigma^2_high / sigma^2_low - 1``.  Any artifact that compares
    correlation across volatility states must report this, or say plainly that
    the comparison is unadjusted and therefore overstates the increase.
    """

    for name, value in (("correlation", correlation), ("variance_ratio", variance_ratio)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError(f"{name} must be a real number")
    rho = float(correlation)
    ratio = float(variance_ratio)
    if not np.isfinite(rho) or not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must be finite and in [-1, 1]")
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("variance_ratio must be finite and positive")
    delta = ratio - 1.0
    denominator = 1.0 + delta * (1.0 - rho * rho)
    if denominator <= 0.0:
        return InferenceResult(
            statistic="Forbes-Rigobon adjusted correlation",
            status=NOT_ESTIMABLE,
            value=None,
            reason="the heteroskedasticity adjustment is undefined for this variance ratio",
        )
    return InferenceResult(
        statistic="Forbes-Rigobon adjusted correlation",
        status=ESTIMATED,
        value=float(rho / math.sqrt(denominator)),
        reason=f"variance ratio {ratio:.4f} between the conditioned and base samples",
    )


# --------------------------------------------------------------------------
# bear, crisis and panic states
# --------------------------------------------------------------------------


def daniel_moskowitz_bear_state(
    adjusted_close: pd.Series,
    *,
    months: int = 24,
    lag: int = 1,
) -> pd.Series:
    """Bear indicator: negative cumulative equity return over 24 prior months.

    Daniel and Moskowitz define it on the market index over the past 24 months
    *excluding the current month*, so the cumulation ends at the prior month-end
    rather than the decision month-end.  Getting that exclusion wrong lets the
    current month's own drawdown label the month it is happening in.
    """

    series = _validated_series(adjusted_close, "adjusted_close")
    months = _validated_window(months, "months", minimum=2)
    lag = _validated_lag(lag)

    decisions = month_end_sessions(series.index)
    monthly = series.loc[decisions]
    cumulative = monthly.shift(1).div(monthly.shift(months + 1)).sub(1.0)
    labels = _label_from_condition(
        cumulative < 0.0, cumulative.notna(), BEAR, NOT_BEAR
    )
    return _project(labels, series.index, lag)


def cooper_gutierrez_hameed_state(
    adjusted_close: pd.Series,
    *,
    months: int = 36,
    lag: int = 1,
) -> pd.Series:
    """Market state: the sign of the lagged 36-month market return.

    A longer-horizon variant of the same idea as the bear indicator, useful as a
    robustness read on the 24-month choice.  It is only a robustness read if it
    does not change the traded weights; if it does, it is a separate registered
    trial rather than a second view of the same path.
    """

    series = _validated_series(adjusted_close, "adjusted_close")
    months = _validated_window(months, "months", minimum=2)
    lag = _validated_lag(lag)

    decisions = month_end_sessions(series.index)
    monthly = series.loc[decisions]
    cumulative = monthly.shift(1).div(monthly.shift(months + 1)).sub(1.0)
    labels = _label_from_condition(
        cumulative > 0.0, cumulative.notna(), RISK_ON, RISK_OFF
    )
    return _project(labels, series.index, lag)


def high_volatility_state(
    returns: pd.Series,
    *,
    window: int = 126,
    history: int | None = 1260,
    minimum_history: int = 504,
    lag: int = 1,
) -> pd.Series:
    """Whether trailing equity volatility sits in the top tercile of its own past."""

    return volatility_state(
        returns,
        window=window,
        history=history,
        minimum_history=minimum_history,
        demean=False,
        lag=lag,
    )


def panic_state(bear: pd.Series, high_volatility: pd.Series) -> pd.Series:
    """Daniel-Moskowitz panic: a bear market with high realized volatility.

    The label is missing wherever either input is missing.  Treating an
    unlabelled component as "not panic" would manufacture a calm state out of a
    warm-up window.
    """

    left = _validated_series(bear, "bear")
    right = _validated_series(high_volatility, "high_volatility")
    if not left.index.equals(right.index):
        raise ValueError("bear and high_volatility must share one index")
    known = left.notna() & right.notna()
    condition = left.eq(BEAR) & right.eq(HIGH)
    return _label_from_condition(condition, known, PANIC, NOT_PANIC)


# --------------------------------------------------------------------------
# Markov switching, fitted inside a fold or not at all
# --------------------------------------------------------------------------


def weekly_equal_weight_log_returns(returns: pd.DataFrame) -> pd.Series:
    """Friday-anchored weekly log return of the equal-weight universe portfolio.

    Each observation is labelled at the last session of its week and a week is
    emitted only once the next week has begun, so the series never contains a
    partially-observed week.  That is what makes it truncation-invariant: a run
    truncated mid-week simply omits the open week rather than reporting a short
    one.
    """

    frame = _validated_frame(returns, "returns")
    portfolio = frame.mean(axis=1, skipna=True)
    log_returns = np.log1p(portfolio.astype(float))
    weeks = frame.index.to_period("W-FRI")
    grouped = log_returns.groupby(weeks)
    totals = grouped.sum()
    stamps = pd.Series(frame.index, index=frame.index).groupby(weeks).last()
    series = pd.Series(totals.to_numpy(), index=pd.DatetimeIndex(stamps.to_numpy()))
    return series.iloc[:-1] if len(series) else series


@dataclass(frozen=True)
class MarkovSwitchingFit:
    """Parameters of a Gaussian hidden Markov model fitted inside one window.

    The states are ordered by ascending fitted mean, so state zero is always the
    low-mean state and fold-to-fold labels are comparable.  Without that anchor
    EM's label permutation makes a fold table meaningless.
    """

    status: str
    reason: str
    states: int
    observations: int
    fit_start: str | None
    fit_end: str | None
    means: tuple[float, ...]
    variances: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    initial: tuple[float, ...]
    log_likelihood: float | None
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        if self.status not in {ESTIMATED, NOT_ESTIMABLE}:
            raise ValueError(f"unknown fit status: {self.status!r}")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if isinstance(self.states, bool) or not isinstance(self.states, int):
            raise ValueError("states must be a whole number")
        if self.states < 2:
            raise ValueError("states must be at least two")
        if isinstance(self.observations, bool) or not isinstance(self.observations, int):
            raise ValueError("observations must be a whole number")
        if self.observations < 0:
            raise ValueError("observations cannot be negative")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise ValueError("iterations must be a whole number")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be a bool")
        if self.status == NOT_ESTIMABLE:
            if self.log_likelihood is not None:
                raise ValueError("a NOT_ESTIMABLE fit cannot carry a log likelihood")
            if self.means or self.variances or self.transition or self.initial:
                raise ValueError("a NOT_ESTIMABLE fit cannot carry parameters")
            return
        if self.log_likelihood is None or not np.isfinite(self.log_likelihood):
            raise ValueError("an ESTIMATED fit requires a finite log likelihood")
        for name in ("means", "variances", "initial"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != self.states:
                raise ValueError(f"{name} must hold one value per state")
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f"{name} must be finite")
        if any(v <= 0.0 for v in self.variances):
            raise ValueError("variances must be positive")
        if any(v < 0.0 for v in self.initial):
            raise ValueError("initial probabilities cannot be negative")
        if abs(sum(self.initial) - 1.0) > 1e-8:
            raise ValueError("initial probabilities must sum to one")
        if not isinstance(self.transition, tuple) or len(self.transition) != self.states:
            raise ValueError("transition must be a square tuple of tuples")
        for row in self.transition:
            if not isinstance(row, tuple) or len(row) != self.states:
                raise ValueError("transition must be a square tuple of tuples")
            if any(v < 0.0 for v in row) or abs(sum(row) - 1.0) > 1e-8:
                raise ValueError("each transition row must be a probability vector")
        if list(self.means) != sorted(self.means):
            raise ValueError("states must be ordered by ascending fitted mean")
        for name in ("fit_start", "fit_end"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty ISO date string")

    def log_likelihood_result(self) -> InferenceResult:
        if self.status == NOT_ESTIMABLE:
            return InferenceResult(
                statistic="Markov switching log likelihood",
                status=NOT_ESTIMABLE,
                value=None,
                reason=self.reason,
            )
        return InferenceResult(
            statistic="Markov switching log likelihood",
            status=ESTIMATED,
            value=float(self.log_likelihood),
            reason=self.reason,
        )


def _emission_density(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    deviation = observations[:, None] - means[None, :]
    return np.exp(-0.5 * deviation * deviation / variances[None, :]) / np.sqrt(
        2.0 * np.pi * variances[None, :]
    )


def _forward(
    density: np.ndarray, initial: np.ndarray, transition: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward recursion.  Row ``t`` is the filtered probability at ``t``."""

    length, states = density.shape
    filtered = np.empty((length, states))
    scale = np.empty(length)
    current = initial * density[0]
    scale[0] = current.sum()
    if scale[0] <= 0.0:
        scale[0] = np.finfo(float).tiny
    filtered[0] = current / scale[0]
    for position in range(1, length):
        current = (filtered[position - 1] @ transition) * density[position]
        total = current.sum()
        if total <= 0.0:
            total = np.finfo(float).tiny
        scale[position] = total
        filtered[position] = current / total
    return filtered, scale


def _backward(
    density: np.ndarray, transition: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    length, states = density.shape
    backward = np.ones((length, states))
    for position in range(length - 2, -1, -1):
        backward[position] = (
            transition @ (density[position + 1] * backward[position + 1])
        ) / scale[position + 1]
    return backward


def _expectation_maximisation(
    observations: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    previous = -np.inf
    iterations = 0
    converged = False
    for step in range(1, max_iterations + 1):
        iterations = step
        density = _emission_density(observations, means, variances)
        filtered, scale = _forward(density, initial, transition)
        backward = _backward(density, transition, scale)
        log_likelihood = float(np.log(scale).sum())

        posterior = filtered * backward
        totals = posterior.sum(axis=1, keepdims=True)
        totals[totals <= 0.0] = np.finfo(float).tiny
        posterior = posterior / totals

        joint = (
            filtered[:-1, :, None]
            * transition[None, :, :]
            * (density[1:] * backward[1:])[:, None, :]
            / scale[1:, None, None]
        )
        joint_totals = joint.sum(axis=(1, 2), keepdims=True)
        joint_totals[joint_totals <= 0.0] = np.finfo(float).tiny
        joint = joint / joint_totals

        initial = posterior[0] / posterior[0].sum()
        numerator = joint.sum(axis=0)
        denominator = numerator.sum(axis=1, keepdims=True)
        denominator[denominator <= 0.0] = np.finfo(float).tiny
        transition = numerator / denominator

        weights = posterior.sum(axis=0)
        weights[weights <= 0.0] = np.finfo(float).tiny
        means = (posterior * observations[:, None]).sum(axis=0) / weights
        deviation = observations[:, None] - means[None, :]
        variances = np.maximum(
            (posterior * deviation * deviation).sum(axis=0) / weights, variance_floor
        )

        if np.isfinite(previous) and abs(log_likelihood - previous) < tolerance:
            previous = log_likelihood
            converged = True
            break
        previous = log_likelihood

    return means, variances, transition, initial, float(previous), iterations, converged


def fit_markov_switching(
    observations: pd.Series,
    *,
    fit_window: tuple[object, object],
    states: int = 2,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    restarts: int = 20,
    fold_index: int = 0,
    variance_floor: float = 1e-12,
    seed: int = DEFAULT_PAIRED_SEED,
    full_sample_descriptive_fit: bool = False,
) -> MarkovSwitchingFit:
    """Fit a Gaussian hidden Markov model on ``fit_window`` and nothing else.

    ``fit_window`` is mandatory.  A model fitted on the whole sample and applied
    historically is look-ahead: the EM likelihood at every date depends on every
    observation, including the future, so the resulting state series is not one
    a trader could have computed.  The function refuses a window reaching the
    last observation unless ``full_sample_descriptive_fit`` is set, and the
    output of such a fit is a description, never an input to a weight.

    Determinism follows the repository's seeding discipline: restarts are drawn
    from ``SeedSequence([seed, crc32(method), fold_index])``, so a fold's fit is
    byte-reproducible and two folds do not share a stream.

    Even fitted correctly the persistence parameters are identified off the
    handful of regime transitions inside the training span, so their standard
    errors are large and fold-to-fold instability is expected rather than
    anomalous.  This is the reason the method scout recommends this model as a
    declared challenger and not as the primary regime engine.
    """

    series = _validated_series(observations, "observations")
    if isinstance(states, bool) or not isinstance(states, int) or states < 2:
        raise ValueError("states must be a whole number of at least two")
    for name, value in (
        ("max_iterations", max_iterations),
        ("restarts", restarts),
        ("fold_index", fold_index),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a whole number")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if restarts < 1:
        raise ValueError("restarts must be positive")
    if fold_index < 0:
        raise ValueError("fold_index cannot be negative")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    if not isinstance(full_sample_descriptive_fit, bool):
        raise ValueError("full_sample_descriptive_fit must be a bool")

    training = _fit_slice(series, fit_window).dropna()
    fit_start = str(pd.Timestamp(fit_window[0]).date())
    fit_end = str(pd.Timestamp(fit_window[1]).date())
    if (
        len(training)
        and not full_sample_descriptive_fit
        and training.index[-1] >= series.index[-1]
    ):
        raise ValueError(_FULL_SAMPLE_FIT_MESSAGE)

    minimum = 20 * states
    values = training.to_numpy(dtype=float)
    if values.size < minimum or float(values.std(ddof=1) if values.size > 1 else 0.0) <= 0.0:
        return MarkovSwitchingFit(
            status=NOT_ESTIMABLE,
            reason=(
                f"the fit window holds {values.size} observations against a minimum "
                f"of {minimum} for a {states}-state model"
            ),
            states=states,
            observations=int(values.size),
            fit_start=fit_start,
            fit_end=fit_end,
            means=(),
            variances=(),
            transition=(),
            initial=(),
            log_likelihood=None,
            iterations=0,
            converged=False,
        )

    method_seed = zlib.crc32(b"markov switching regime") & 0xFFFFFFFF
    entropy = np.random.SeedSequence([int(seed), method_seed, int(fold_index)])
    rng = np.random.default_rng(entropy)

    quantiles = np.quantile(values, np.linspace(0.0, 1.0, states + 1)[1:-1])
    assignment = np.digitize(values, quantiles)
    base_means = np.array(
        [
            values[assignment == k].mean() if np.any(assignment == k) else values.mean()
            for k in range(states)
        ]
    )
    base_variance = max(float(values.var(ddof=1)), variance_floor)

    best: tuple[float, tuple[np.ndarray, ...], int, bool] | None = None
    for attempt in range(restarts):
        if attempt == 0:
            means = base_means.copy()
            variances = np.full(states, base_variance)
        else:
            means = base_means + rng.normal(0.0, math.sqrt(base_variance), states)
            variances = base_variance * np.exp(rng.normal(0.0, 0.5, states))
        variances = np.maximum(variances, variance_floor)
        transition = np.full((states, states), 0.1 / max(states - 1, 1))
        np.fill_diagonal(transition, 0.9)
        transition = transition / transition.sum(axis=1, keepdims=True)
        initial = np.full(states, 1.0 / states)

        try:
            fitted = _expectation_maximisation(
                values,
                means,
                variances,
                transition,
                initial,
                max_iterations=max_iterations,
                tolerance=tolerance,
                variance_floor=variance_floor,
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        *parameters, log_likelihood, iterations, converged = fitted
        if not np.isfinite(log_likelihood):
            continue
        if best is None or log_likelihood > best[0]:
            best = (log_likelihood, tuple(parameters), iterations, converged)

    if best is None:
        return MarkovSwitchingFit(
            status=NOT_ESTIMABLE,
            reason="no restart produced a finite likelihood on the fit window",
            states=states,
            observations=int(values.size),
            fit_start=fit_start,
            fit_end=fit_end,
            means=(),
            variances=(),
            transition=(),
            initial=(),
            log_likelihood=None,
            iterations=0,
            converged=False,
        )

    log_likelihood, (means, variances, transition, initial), iterations, converged = best
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    transition = transition[np.ix_(order, order)]
    transition = transition / transition.sum(axis=1, keepdims=True)
    initial = initial[order]
    initial = initial / initial.sum()

    return MarkovSwitchingFit(
        status=ESTIMATED,
        reason=(
            f"expectation maximisation on {values.size} observations in "
            f"{fit_start}..{fit_end}, {restarts} restarts, best of {iterations} iterations"
        ),
        states=states,
        observations=int(values.size),
        fit_start=fit_start,
        fit_end=fit_end,
        means=tuple(float(v) for v in means),
        variances=tuple(float(v) for v in variances),
        transition=tuple(tuple(float(v) for v in row) for row in transition),
        initial=tuple(float(v) for v in initial),
        log_likelihood=float(log_likelihood),
        iterations=int(iterations),
        converged=bool(converged),
    )


def markov_filtered_probabilities(
    observations: pd.Series,
    fit: MarkovSwitchingFit,
) -> pd.DataFrame:
    """``Pr(s_t = k | y_1..y_t)`` under frozen parameters.

    The forward recursion only.  Reading the smoothed probability
    ``Pr(s_t = k | y_1..y_T)`` instead --- which is what most library examples
    expose by default --- conditions every historical label on the whole sample
    and breaks truncation invariance immediately.
    """

    series = _validated_series(observations, "observations")
    if not isinstance(fit, MarkovSwitchingFit):
        raise ValueError("fit must be a MarkovSwitchingFit")
    columns = _state_names(fit.states)
    if fit.status == NOT_ESTIMABLE:
        return pd.DataFrame(np.nan, index=series.index, columns=columns)

    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite to run the filter")
    density = _emission_density(
        values, np.asarray(fit.means), np.asarray(fit.variances)
    )
    filtered, _ = _forward(
        density,
        np.asarray(fit.initial),
        np.asarray([list(row) for row in fit.transition]),
    )
    return pd.DataFrame(filtered, index=series.index, columns=columns)


def _state_names(states: int) -> list[str]:
    if states == len(MARKOV_STATE_LABELS):
        return list(MARKOV_STATE_LABELS)
    return [f"State {k}" for k in range(states)]


def markov_switching_state(
    observations: pd.Series,
    fit: MarkovSwitchingFit,
    *,
    sessions: pd.DatetimeIndex | None = None,
    lag: int = 1,
) -> pd.Series:
    """The most probable filtered state, optionally projected onto a calendar."""

    lag = _validated_lag(lag)
    probabilities = markov_filtered_probabilities(observations, fit)
    names = list(probabilities.columns)
    if fit.status == NOT_ESTIMABLE:
        labels = pd.Series(pd.NA, index=probabilities.index, dtype="string")
    else:
        chosen = probabilities.to_numpy().argmax(axis=1)
        labels = pd.Series(
            [names[k] for k in chosen], index=probabilities.index, dtype="string"
        )
    if sessions is None:
        return labels.shift(lag)
    return _project(labels, sessions, lag)


# --------------------------------------------------------------------------
# combined report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeConfig:
    """Pre-declared window lengths for the combined state builder.

    Every value here is a constant taken from the source paper, not a swept
    parameter.  ``docs/research-methodology.md`` counts every value of every
    parameter as a separate trial, so these are set once and left.
    """

    faber_months: int = 10
    momentum_lookback: int = 252
    book_volatility_window: int = 21
    equity_volatility_window: int = 126
    volatility_history: int = 1260
    volatility_minimum_history: int = 504
    correlation_window: int = 63
    correlation_minimum_markets: int = 5
    correlation_minimum_coverage: float = 0.8
    correlation_history: int = 1260
    absorption_window: int = 252
    absorption_fast: int = 15
    absorption_slow: int = 252
    absorption_stress_threshold: float = 1.0
    bear_months: int = 24
    market_state_months: int = 36
    annualization: int = 252
    lag: int = 1

    def __post_init__(self) -> None:
        whole = (
            "faber_months",
            "momentum_lookback",
            "book_volatility_window",
            "equity_volatility_window",
            "volatility_history",
            "volatility_minimum_history",
            "correlation_window",
            "correlation_minimum_markets",
            "correlation_history",
            "absorption_window",
            "absorption_fast",
            "absorption_slow",
            "bear_months",
            "market_state_months",
            "annualization",
            "lag",
        )
        for name in whole:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a whole number")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.correlation_minimum_markets < 2:
            raise ValueError("correlation_minimum_markets must be at least two")
        if self.volatility_minimum_history < 2:
            raise ValueError("volatility_minimum_history must be at least two")
        if self.volatility_history < self.volatility_minimum_history:
            raise ValueError("volatility_history must be at least the minimum history")
        if self.correlation_history < self.volatility_minimum_history:
            raise ValueError("correlation_history must be at least the minimum history")
        if self.absorption_fast >= self.absorption_slow:
            raise ValueError("absorption_fast must be shorter than absorption_slow")
        for name in ("correlation_minimum_coverage", "absorption_stress_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.floating)
            ):
                raise ValueError(f"{name} must be a real number")
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 < float(self.correlation_minimum_coverage) <= 1:
            raise ValueError("correlation_minimum_coverage must be in (0, 1]")


def combined_regime_states(
    adjusted_close: pd.DataFrame,
    *,
    equity_ticker: str = "SPY",
    config: RegimeConfig = RegimeConfig(),
    include_sleeve_states: bool = False,
    drop_zero_observations: bool = True,
) -> pd.DataFrame:
    """One labelled state per regime definition, on the panel session calendar.

    Book-level states are always present.  Per-sleeve trend and momentum states
    are eleven columns each and are appended only on request, because a report
    a reader cannot scan is a report that does not get checked.
    """

    frame = _validated_frame(adjusted_close, "adjusted_close")
    if not isinstance(config, RegimeConfig):
        raise ValueError("config must be a RegimeConfig")
    if not isinstance(include_sleeve_states, bool):
        raise ValueError("include_sleeve_states must be a bool")
    if equity_ticker not in frame.columns:
        raise ValueError(f"{equity_ticker!r} is not a column of adjusted_close")

    sessions = frame.index
    returns = frame.div(frame.shift(1)).sub(1.0)
    equity_returns = returns[equity_ticker]
    book_returns = returns.mean(axis=1, skipna=True)

    trend = faber_trend_state(frame, months=config.faber_months, lag=config.lag)
    momentum = time_series_momentum_state(
        frame, lookback=config.momentum_lookback, lag=config.lag
    )
    book_volatility = volatility_state(
        book_returns,
        window=config.book_volatility_window,
        history=config.volatility_history,
        minimum_history=config.volatility_minimum_history,
        annualization=config.annualization,
        lag=config.lag,
    )
    equity_volatility = high_volatility_state(
        equity_returns,
        window=config.equity_volatility_window,
        history=config.volatility_history,
        minimum_history=config.volatility_minimum_history,
        lag=config.lag,
    )
    correlation = mean_pairwise_correlation(
        returns.iloc[1:],
        window=config.correlation_window,
        minimum_markets=config.correlation_minimum_markets,
        minimum_coverage=config.correlation_minimum_coverage,
        lag=config.lag,
        drop_zero_observations=drop_zero_observations,
    ).reindex(sessions)
    correlation_labels = _trailing_tercile_labels(
        correlation,
        history=config.correlation_history,
        minimum_history=config.volatility_minimum_history,
    )
    ratio = absorption_ratio(
        returns.iloc[1:], window=config.absorption_window, lag=config.lag
    ).reindex(sessions)
    absorption_labels = _trailing_tercile_labels(
        ratio,
        history=config.correlation_history,
        minimum_history=config.volatility_minimum_history,
    )
    stress = systemic_stress_state(
        absorption_ratio_shift(
            ratio, fast=config.absorption_fast, slow=config.absorption_slow
        ),
        threshold=config.absorption_stress_threshold,
    )
    bear = daniel_moskowitz_bear_state(
        frame[equity_ticker], months=config.bear_months, lag=config.lag
    )
    market = cooper_gutierrez_hameed_state(
        frame[equity_ticker], months=config.market_state_months, lag=config.lag
    )

    states = pd.DataFrame(
        {
            "Equity trend state": trend[equity_ticker],
            "Equity momentum state": momentum[equity_ticker],
            "Book volatility state": book_volatility,
            "Equity volatility state": equity_volatility,
            "Diversification state": correlation_labels,
            "Absorption state": absorption_labels,
            "Systemic stress state": stress,
            "Equity bear state": bear,
            "Long market state": market,
        },
        index=sessions,
    )
    states["Panic state"] = panic_state(bear, equity_volatility)
    if include_sleeve_states:
        for ticker in frame.columns:
            states[f"Faber trend {ticker}"] = trend[ticker]
            states[f"Momentum {ticker}"] = momentum[ticker]
    return states.astype("string")


def _episode_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    run = 0
    for flag in mask:
        if flag:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def regime_state_report(states: pd.DataFrame) -> pd.DataFrame:
    """Occupancy and persistence of every state in every regime.

    An episode is a maximal run of consecutive labelled sessions in one state.
    Runs are counted on the session calendar, so a run interrupted by an
    unlabelled session counts as two episodes; that is the honest reading,
    because the strategy would have had no state to act on in between.
    """

    frame = _validated_frame(states, "states")
    rows: list[dict[str, object]] = []
    for regime in frame.columns:
        column = frame[regime]
        labelled = column.notna()
        denominator = int(labelled.sum())
        if denominator == 0:
            continue
        for state in sorted(column.dropna().unique()):
            mask = _as_boolean_array(column.eq(state))
            sessions = int(mask.sum())
            if sessions == 0:
                continue
            lengths = _episode_lengths(mask)
            stamps = column.index[mask]
            rows.append(
                {
                    "Regime": regime,
                    "State": str(state),
                    "Sessions": sessions,
                    "Share of labelled sessions": sessions / denominator,
                    "Share of all sessions": sessions / len(frame),
                    "Episodes": len(lengths),
                    "Mean episode duration (sessions)": float(np.mean(lengths)),
                    "Median episode duration (sessions)": float(np.median(lengths)),
                    "Longest episode (sessions)": int(max(lengths)),
                    "First labelled session": stamps.min().date().isoformat(),
                    "Last labelled session": stamps.max().date().isoformat(),
                    "Limitation": SURVIVORSHIP_LIMITATION,
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(REGIME_STATE_COLUMNS))
    return pd.DataFrame(rows, columns=list(REGIME_STATE_COLUMNS))


def regime_transition_matrix(
    state: pd.Series,
    *,
    as_probabilities: bool = True,
) -> pd.DataFrame:
    """Session-to-session transition matrix, rows from and columns to.

    Only consecutive sessions where both labels exist contribute.  Self
    transitions are counted, so the diagonal is the persistence of each state
    and the rows sum to one when normalized.
    """

    series = _validated_series(state, "state")
    if not isinstance(as_probabilities, bool):
        raise ValueError("as_probabilities must be a bool")
    names = sorted(series.dropna().unique())
    counts = pd.DataFrame(
        0.0, index=[str(n) for n in names], columns=[str(n) for n in names]
    )
    current = series.to_numpy(dtype=object)
    for position in range(1, len(current)):
        source = current[position - 1]
        destination = current[position]
        if pd.isna(source) or pd.isna(destination):
            continue
        counts.loc[str(source), str(destination)] += 1.0
    if not as_probabilities:
        return counts.astype(int)
    totals = counts.sum(axis=1)
    normalized = counts.div(totals.replace(0.0, np.nan), axis=0)
    return normalized


def regime_transition_report(states: pd.DataFrame) -> pd.DataFrame:
    """Every regime's transitions in long form, counts beside probabilities."""

    frame = _validated_frame(states, "states")
    rows: list[dict[str, object]] = []
    for regime in frame.columns:
        counts = regime_transition_matrix(frame[regime], as_probabilities=False)
        if counts.empty:
            continue
        probabilities = regime_transition_matrix(frame[regime])
        for source in counts.index:
            for destination in counts.columns:
                rows.append(
                    {
                        "Regime": regime,
                        "From state": source,
                        "To state": destination,
                        "Transitions": int(counts.loc[source, destination]),
                        "Transition probability": float(
                            probabilities.loc[source, destination]
                        ),
                        "Limitation": SURVIVORSHIP_LIMITATION,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(REGIME_TRANSITION_COLUMNS))
    return pd.DataFrame(rows, columns=list(REGIME_TRANSITION_COLUMNS))


__all__ = [
    "BEAR",
    "HIGH",
    "LOW",
    "MARKOV_STATE_LABELS",
    "MIDDLE",
    "NOT_BEAR",
    "NOT_PANIC",
    "NOT_STRESS",
    "PANIC",
    "REGIME_STATE_COLUMNS",
    "REGIME_TRANSITION_COLUMNS",
    "RISK_OFF",
    "RISK_ON",
    "STRESS",
    "MarkovSwitchingFit",
    "RegimeConfig",
    "VolatilityScaling",
    "absorption_ratio",
    "absorption_ratio_shift",
    "barroso_santa_clara_scale",
    "combined_regime_states",
    "cooper_gutierrez_hameed_state",
    "daniel_moskowitz_bear_state",
    "ewma_volatility",
    "faber_trend_state",
    "fit_markov_switching",
    "forbes_rigobon_adjusted_correlation",
    "high_volatility_state",
    "markov_filtered_probabilities",
    "markov_switching_state",
    "mean_pairwise_correlation",
    "month_end_sessions",
    "moreira_muir_scale",
    "panic_state",
    "realized_volatility",
    "regime_state_report",
    "regime_transition_matrix",
    "regime_transition_report",
    "systemic_stress_state",
    "time_series_momentum_state",
    "volatility_state",
    "weekly_equal_weight_log_returns",
]
