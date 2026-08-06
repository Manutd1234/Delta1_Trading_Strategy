"""Published-strategy benchmarks replicated on the DELTA1 panel and ledger.

The repository previously had no strategy benchmark at all, so the question
"is a 1.59 Sharpe good?" could not be answered from inside it.  This module
answers it the only way that is defensible without a licensed index: it
rebuilds the published academic rules on the *identical* 59-market panel and
pushes them through the *identical* execution ledger, so the only thing that
differs between a benchmark row and the incumbent row is the signal.

Every benchmark here builds a monthly target frame and hands it to
``strategy._simulate_execution`` with the incumbent's own ``StrategyConfig``.
Costs, integer contracts, the 2% participation cap, the realized-capacity fill
truncation, roll turnover, the roll backlog and FX are therefore identical *by
construction* rather than by inspection.  No parallel simulator exists in this
file, and ``strategy.py`` is not modified: the seam the engine already exposes
was sufficient, so the canonical daily fingerprint is untouched by definition.

What this module does not prove
-------------------------------
It does not establish that the incumbent is better than published trend
following.  Every number here is measured on the same reused 1990-2014 history
the incumbent was constructed against, so a favourable comparison is a
description of that history and not out-of-sample evidence.  The benchmarks
themselves were *not* refit here, which cuts the other way: they are honest
published rules run cold on this panel, while the incumbent's parameters were
chosen with this panel visible.  A positive alpha against them is therefore an
upper bound on the incumbent's edge, not an estimate of it.

It also does not reproduce the published papers' own reported results.  A
different universe, a different sample, a different cost model and a different
fee load all sit between this panel and theirs; the replication is of the
*rule*, never of the result.  Their headline figures are recorded in each
builder's docstring so the gap is visible rather than implied.

Two further limits are worth naming.  No external index (SG Trend, BTOP50 and
the like) is used: those require a licence and a network, and they carry a
different universe, cost base and management/performance fee load, so a
comparison against them would not be like-for-like even if it were available.
And the regressions below are ordinary reused-history regressions: the
intervals are sampling uncertainty only, ``Selection adjusted`` is False on
every row, and nothing here licenses promoting or demoting a configuration,
which ``docs/research-methodology.md`` prohibits on reused history regardless
of the numbers.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from . import inference
from .strategy import (
    BacktestResult,
    StrategyConfig,
    _month_end_rows,
    _simulate_execution,
    build_trade_episodes,
    performance_metrics,
    roll_event_mask,
    tradeable_mask,
    usd_margin_values,
    usd_point_values,
)


RETROSPECTIVE_START = "1990-01-01"
RETROSPECTIVE_END = "2014-12-31"
VALIDATION_STATUS = "retrospective_published_rule_replication"

RUN_COMPLETED = "completed"
RUN_ERRORED = "errored"

DEFAULT_PERMUTATIONS = 1_000
DEFAULT_PERMUTATION_SEED = 20_260_805

BLOCK_SIGN_FLIP = "block_sign_flip"
IID_SIGN_FLIP = "iid_sign_flip"

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

_PERMITTED_USE = (
    "descriptive replication of published rules on reused 1990-2014 history; "
    "not selection adjusted and does not license promotion"
)

# Column names deliberately avoid the substrings "target", "pass", "fail" and
# "verdict".  The repository's report tests reject them so that a descriptive
# artifact can never be read as a promotion decision, which is exactly the
# failure mode a benchmark table invites.
BENCHMARK_COMPARISON_COLUMNS = [
    "benchmark",
    "family",
    "citation",
    "construction",
    "leverage_cap_applied",
    "validation_status",
    "run_status",
    "run_detail",
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
    "max_drawdown_over_volatility",
    "annual_cost_drag",
    "annual_rebalance_notional_turnover_over_nav",
    "annual_rebalance_turnover_contracts",
    "annual_roll_turnover_contracts",
    "annual_total_turnover_contracts",
    "average_gross_notional_multiple",
    "peak_gross_notional_multiple",
    "average_markets_held",
    "return_correlation_with_incumbent",
]

ALPHA_REPORT_COLUMNS = [
    "regression",
    "explanatory_series",
    "method",
    "sessions",
    "alpha_annualized",
    "alpha_hac_standard_error_annualized",
    "alpha_hac_t_statistic",
    "alpha_hedged_mean_t_statistic",
    "beta",
    "beta_hac_t_statistic",
    "r_squared",
    "residual_volatility_annualized",
    "appraisal_ratio",
    "tracking_error_annualized",
    "information_ratio",
    "return_correlation",
    "hac_lags",
    "selection_adjusted",
    "permitted_use",
]

SPANNING_REPORT_COLUMNS = [
    "regression",
    "method",
    "sessions",
    "explanatory_series_count",
    "explanatory_series",
    "alpha_annualized",
    "alpha_hac_standard_error_annualized",
    "alpha_hac_t_statistic",
    "r_squared",
    "adjusted_r_squared",
    "residual_volatility_annualized",
    "appraisal_ratio",
    "coefficient_gross_exposure",
    "design_condition_number",
    "hac_lags",
    "selection_adjusted",
    "permitted_use",
]

SPANNING_COEFFICIENT_COLUMNS = [
    "regression",
    "explanatory_series",
    "coefficient",
    "hac_standard_error",
    "hac_t_statistic",
]

PERMUTATION_NULL_COLUMNS = [
    "null",
    "construction",
    "permutations",
    "permutations_completed",
    "seed",
    "resolution",
    "incumbent_sharpe",
    "null_sharpe_mean",
    "null_sharpe_standard_deviation",
    "null_sharpe_percentile_01",
    "null_sharpe_percentile_05",
    "null_sharpe_percentile_50",
    "null_sharpe_percentile_95",
    "null_sharpe_percentile_99",
    "incumbent_percentile_in_null",
    "empirical_one_sided_p_value",
    "cost_only_sharpe_reference",
    "null_annual_cost_drag_mean",
    "incumbent_annual_cost_drag",
    "null_annual_turnover_mean",
    "incumbent_annual_turnover",
    "incumbent_median_run_decisions",
    "null_median_run_decisions",
    "selection_adjusted",
    "permitted_use",
]

OVERLAY_WEIGHT_COLUMNS = [
    "overlay",
    "underlying",
    "causality",
    "decisions",
    "first_active_decision",
    "weight_minimum",
    "weight_percentile_05",
    "weight_median",
    "weight_mean",
    "weight_percentile_95",
    "weight_percentile_99",
    "weight_maximum",
    "incumbent_maximum_risk_scalar_reference",
    "selection_adjusted",
    "permitted_use",
]

SIGNAL_DIAGNOSTIC_COLUMNS = [
    "signal",
    "citation",
    "decisions",
    "average_live_markets",
    "mean_absolute_exposure",
    "maximum_absolute_exposure",
    "sign_agreement_with_mop",
    "average_gross_notional_multiple",
    "maximum_gross_notional_multiple",
    "note",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Published-rule parameters, each pinned to the paper that fixes it.

    Nothing here is fitted.  Every default is transcribed from a citation:
    ``instrument_volatility_budget`` is Moskowitz, Ooi & Pedersen's 40%,
    ``volatility_center_of_mass`` and ``volatility_annualization`` are their
    equation (1) verbatim (a centre of mass of 60 days, i.e. delta = 60/61,
    and a 261-day variance annualisation), ``portfolio_volatility_budget`` and
    ``covariance_months`` are Hurst, Ooi & Pedersen's 10% target and footnote
    5's rolling three-year equally weighted monthly covariance, the MACD
    time-scales are Baz et al.'s equation (8), and the Barroso-Santa-Clara and
    Moreira-Muir windows are those papers' own.

    The 261 in ``volatility_annualization`` is deliberately *not* the
    repository's 252.  It lives inside the benchmark's own ex-ante volatility
    estimator, where the paper puts it; reported Sharpe ratios still annualise
    at 252 so they stay comparable to the incumbent's.  Unifying the two
    constants would silently change the benchmark's sizing.

    What is deliberately *absent* matters as much.  The tradeable-universe gate
    is not here: no paper fixes it, it is an engine parameter, and the
    incumbent reads it from its own ``StrategyConfig``.  Holding a second copy
    of it on this dataclass would let a caller vary the engine's gate while
    every benchmark silently kept trading the default universe, which is the
    one divergence the shared-inputs design cannot detect after the fact
    because the gate is applied upstream of ``_simulate_execution``.
    ``prepare_engine_inputs`` therefore reads it from the ``StrategyConfig``,
    so the benchmark universe equals the incumbent's by construction.
    """

    instrument_volatility_budget: float = 0.40
    volatility_center_of_mass: int = 60
    volatility_annualization: int = 261
    volatility_min_periods: int = 60

    trend_lookback_sessions: int = 252
    min_return_observations: int = 200

    hop_horizon_sessions: tuple[int, ...] = (21, 63, 252)
    hop_min_observation_fraction: float = 0.80
    portfolio_volatility_budget: float = 0.10
    covariance_months: int = 36

    macd_time_scales: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
    macd_price_volatility_window: int = 63
    macd_signal_volatility_window: int = 252
    macd_warmup_sessions: int = 350
    macd_normalizing_constant: float = 0.89

    barroso_window_sessions: int = 126
    barroso_volatility_budget: float = 0.12
    barroso_sessions_per_month: int = 21

    moreira_min_month_sessions: int = 5
    moreira_min_history_months: int = 24

    def __post_init__(self) -> None:
        for name in (
            "instrument_volatility_budget",
            "portfolio_volatility_budget",
            "barroso_volatility_budget",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.macd_normalizing_constant)
            or self.macd_normalizing_constant <= 0
        ):
            raise ValueError("macd_normalizing_constant must be finite and positive")
        if not 0 < self.hop_min_observation_fraction <= 1:
            raise ValueError("hop_min_observation_fraction must be in (0, 1]")
        for name in (
            "volatility_center_of_mass",
            "volatility_annualization",
            "volatility_min_periods",
            "trend_lookback_sessions",
            "min_return_observations",
            "covariance_months",
            "macd_price_volatility_window",
            "macd_signal_volatility_window",
            "macd_warmup_sessions",
            "barroso_window_sessions",
            "barroso_sessions_per_month",
            "moreira_min_month_sessions",
            "moreira_min_history_months",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_return_observations > self.trend_lookback_sessions:
            raise ValueError(
                "min_return_observations cannot exceed trend_lookback_sessions"
            )
        if self.covariance_months < 2:
            raise ValueError("covariance_months must be at least two months")
        if not self.hop_horizon_sessions:
            raise ValueError("hop_horizon_sessions must not be empty")
        for horizon in self.hop_horizon_sessions:
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                raise ValueError("every HOP horizon must be a positive integer")
        if not self.macd_time_scales:
            raise ValueError("macd_time_scales must not be empty")
        for pair in self.macd_time_scales:
            if len(pair) != 2:
                raise ValueError("every MACD time-scale must be a (short, long) pair")
            short, long = pair
            for scale in (short, long):
                if isinstance(scale, bool) or not isinstance(scale, int) or scale < 2:
                    raise ValueError("MACD time-scales must be integers of at least two")
            if short >= long:
                raise ValueError("a MACD short time-scale must be below its long one")


@dataclass(frozen=True)
class EngineInputs:
    """Every derived frame ``_simulate_execution`` needs, built once.

    Sharing one set guarantees that a difference between two benchmark rows can
    only come from the target frame, which is the entire claim this module
    makes.  ``valuation_prices`` is the unadjusted panel because a back-adjusted
    level is not a notional; that is the same frame ``run_backtest`` passes.
    """

    prices: pd.DataFrame
    valuation_prices: pd.DataFrame
    observed_opens: pd.DataFrame
    observed_closes: pd.DataFrame
    volumes: pd.DataFrame
    point_values: pd.DataFrame
    one_way_costs: pd.DataFrame
    margin_values: pd.DataFrame
    rolls: pd.DataFrame
    delivery_anomalies: pd.DataFrame
    excess_returns: pd.DataFrame
    eligible: pd.DataFrame
    # The four fee parameters baked into ``one_way_costs`` when this object was
    # built.  ``simulate_monthly_targets`` accepts a modified config — releasing
    # the gross-notional cap needs one — and those four are the fields a
    # modified config could change without the pre-derived cost frame noticing.
    fee_parameters: tuple[float, float, float, float]


def _fee_parameters(config: StrategyConfig) -> tuple[float, float, float, float]:
    return (
        float(config.half_spread_ticks),
        float(config.slippage_ticks),
        float(config.commission_per_contract),
        float(config.exchange_and_regulatory_fees_per_contract),
    )


def prepare_engine_inputs(
    config: StrategyConfig,
    frames: Mapping[str, pd.DataFrame],
) -> EngineInputs:
    """Derive the engine's cost, valuation and eligibility frames from the panel.

    Every line here is copied from ``run_backtest``'s own preamble so the cost
    model a benchmark pays is the incumbent's, not a re-derivation of it.  A
    partially supplied frame set is rejected rather than silently defaulted:
    omitting ``delivery_months`` alone removes every roll cost and would flatter
    each benchmark by roughly the size of the promotion gate.

    The tradeable-universe gate is read from the ``StrategyConfig``, exactly as
    ``run_backtest`` reads it, so a benchmark can never trade a market the
    incumbent is forbidden from trading.  ``BenchmarkConfig`` deliberately does
    not carry its own copy: the gate is applied here, upstream of
    ``_simulate_execution``, so no downstream identity — not
    ``_require_matching_fees``, not ``assert_seam_reproduces_incumbent`` — can
    see a divergence in it, and the only safe design is for the divergence to
    be unrepresentable.
    """

    missing = [key for key in _FRAME_KEYS if key not in frames]
    if missing:
        raise ValueError(
            "benchmark replication requires every input frame; missing: "
            f"{missing}. Partial injection silently disables roll costs."
        )
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
    rolls, delivery_anomalies = roll_event_mask(frames["delivery_months"])
    returns = excess_returns(prices, frames["unadjusted"])
    eligible = (
        tradeable_mask(
            frames["volumes"],
            config.volume_gate_window,
            config.min_median_contracts,
        )
        .reindex(prices.index)
        .fillna(False)
    )
    return EngineInputs(
        prices=prices,
        valuation_prices=frames["unadjusted"],
        observed_opens=frames["observed_opens"],
        observed_closes=frames["observed_closes"],
        volumes=frames["volumes"],
        point_values=point_values,
        one_way_costs=one_way_costs,
        margin_values=margin_values,
        rolls=rolls,
        delivery_anomalies=delivery_anomalies,
        excess_returns=returns,
        eligible=eligible,
        fee_parameters=_fee_parameters(config),
    )


# ---------------------------------------------------------------------------
# Return and volatility primitives shared by every benchmark
# ---------------------------------------------------------------------------


def excess_returns(prices: pd.DataFrame, unadjusted: pd.DataFrame) -> pd.DataFrame:
    """Fully collateralised per-market excess return of a futures position.

    ``r[i, t] = (P_adj[i, t] - P_adj[i, t-1]) / P_unadj[i, t-1]``.

    The division by the *unadjusted* previous price is mandatory rather than
    stylistic.  Back-adjustment is a cumulative additive shift, so the
    back-adjusted panel goes non-positive in 48,001 cells on this data; a
    percentage change taken there is undefined or sign-flipped and would poison
    both the trend signal and the volatility estimator without raising.  The
    unadjusted panel has no non-positive cells, and the numerator must stay the
    back-adjusted difference because that is the economic price move.
    """

    previous_unadjusted = unadjusted.reindex_like(prices).shift(1)
    denominator = previous_unadjusted.where(previous_unadjusted > 0)
    returns = prices.diff() / denominator
    return returns.where(prices.notna() & denominator.notna())


def ex_ante_volatility(
    returns: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.DataFrame:
    """Moskowitz, Ooi & Pedersen equation (1), then lagged one session.

    ``sigma^2_t = 261 * sum_i (1-delta) delta^i (r_{t-1-i} - rbar_t)^2`` with
    the weights' centre of mass at 60 days, so ``delta = 60/61`` and pandas'
    ``alpha = 1/61``.  The paper's estimator is a mean of squared deviations,
    not a reliability-corrected variance, so ``.std()`` and ``.var()`` are both
    wrong here: they default to ``ddof=1``.

    The sum already begins at ``r_{t-1}``.  On top of that the paper states
    plainly that "we use the volatility estimates at time t-1 applied to time-t
    returns", so the whole frame is shifted one further session before it can
    enter sizing.  The returned frame is that shifted frame: no caller has to
    remember to lag it, and no caller can forget.

    Both recursions run from the first observation and the warm-up is blanked
    afterwards.  Putting ``min_periods`` on the *mean* instead would leave the
    deviations undefined through the warm-up and silently restart the variance
    recursion at the first valid deviation, which is a different estimator with
    a much shorter effective memory at the start of every market's life.
    """

    alpha = 1.0 / (config.volatility_center_of_mass + 1.0)
    mean = returns.ewm(alpha=alpha, adjust=False).mean()
    variance = ((returns - mean) ** 2).ewm(alpha=alpha, adjust=False).mean()
    volatility = np.sqrt(config.volatility_annualization * variance)
    observed = returns.notna().astype(float).cumsum()
    warm = volatility.where(observed >= config.volatility_min_periods)
    return warm.shift(1)


def _valid_observation_count(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing count of *observed* returns, not of index rows.

    ``DataFrame.notna().rolling(w).count()`` counts non-null cells of a boolean
    frame, which is the window length everywhere.  Summing the booleans is what
    actually screens a market that has not been quoted for the whole window,
    and that screen is the survivorship control.
    """

    return returns.notna().astype(float).rolling(window, min_periods=window).sum()


def _cumulative_return(
    returns: pd.DataFrame,
    window: int,
    min_observations: int,
) -> pd.DataFrame:
    """Trailing cumulative excess return over ``window`` sessions."""

    if bool((returns <= -1.0).any().any()):
        raise ValueError(
            "an excess return at or below -100% cannot be compounded; the "
            "return definition or the panel is wrong"
        )
    total = np.log1p(returns).fillna(0.0).rolling(window, min_periods=window).sum()
    observed = _valid_observation_count(returns, window)
    return np.expm1(total).where(observed >= min_observations)


def monthly_excess_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-market monthly excess returns, indexed by calendar month period.

    A month's return is known at the close of that month's last observed
    session, which is exactly the decision date ``_month_end_rows`` selects, so
    a decision may use its own month.  A truncated panel's partial terminal
    month is never used, because ``_month_end_rows`` does not emit a decision in
    it unless the data end on a business month end.
    """

    periods = returns.index.to_period("M")
    observed = returns.notna().astype(float).groupby(periods).sum()
    compounded = (1.0 + returns.fillna(0.0)).groupby(periods).prod() - 1.0
    return compounded.where(observed > 0)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def mop_tsmom_signal(
    returns: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.DataFrame:
    """Moskowitz, Ooi & Pedersen (2012) 12-month time-series momentum sign.

    ``X[i, t] = sign(prod_{s=t-251..t}(1 + r[i, s]) - 1)``, inclusive of the
    current month.  Cross-sectional momentum skips the most recent month; time
    series momentum does not, and their equation (5) is ``sign(r^s_{t-12,t})``.

    ``sign(0)`` is 0, which is a flat position and reduces the diversification
    count for that month.  It is left as 0 rather than coerced to +1: coercing
    it would silently add a long tilt.
    """

    cumulative = _cumulative_return(
        returns, config.trend_lookback_sessions, config.min_return_observations
    )
    return np.sign(cumulative).where(cumulative.notna())


def hop_blend_signal(
    returns: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.DataFrame:
    """Hurst, Ooi & Pedersen (2017) equal-weighted 1/3/12-month sign blend.

    The paper's transform is the plain sign: "A positive past return is
    considered an 'up trend' and leads to a long position".  There is no smooth
    or capped transform anywhere in it, so adding one implements Baz et al.
    instead.

    Each sleeve needs 80% of its window observed.  If any sleeve is missing the
    market is dropped for that month rather than having the remaining sleeve
    weights renormalised, because renormalising turns a three-horizon blend
    into a shorter-horizon strategy exactly during warm-up.
    """

    sleeves = []
    for horizon in config.hop_horizon_sessions:
        minimum = int(math.ceil(config.hop_min_observation_fraction * horizon))
        cumulative = _cumulative_return(returns, horizon, minimum)
        sleeves.append(np.sign(cumulative).where(cumulative.notna()))
    blend = sleeves[0]
    for sleeve in sleeves[1:]:
        blend = blend + sleeve
    return blend / float(len(sleeves))


def time_trend_tstatistic(
    returns: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.DataFrame:
    """OLS t-statistic of a time trend fitted to the cumulative log return.

    The regressand is the cumulative log excess return, not a log price.  The
    back-adjusted panel is non-positive in 48,001 cells so ``log(P_adj)`` is
    undefined there, and ``log(P_unadj)`` carries roll discontinuities that are
    not economic moves.  The cumulative log excess return is the continuous
    log-price analogue of a rolled futures position and is always defined.

    Closed form, with ``s = 0..n-1`` fixed so the design is a constant:
    ``S_xx = n(n^2-1)/12``, ``b = sum_s (s - sbar) c_s / S_xx``,
    ``RSS = sum_s (c_s - cbar)^2 - b^2 S_xx``, and
    ``t = b / sqrt((RSS/(n-2)) / S_xx)``.  Three rolling aggregations, no
    per-market loop.

    The standard error is the plain OLS one.  The residuals of a random walk
    regressed on time *are* strongly autocorrelated, but Baltas and Kosowski use
    this t as a signal transform rather than as a test, so replacing it with a
    HAC standard error would change the signal itself and not merely its
    inference.  No p-value is reported for the same reason.
    """

    window = config.trend_lookback_sessions
    if window < 3:
        raise ValueError("a time-trend regression needs at least three sessions")
    cumulative_log = np.log1p(returns).fillna(0.0).cumsum()
    positions = pd.Series(
        np.arange(len(cumulative_log), dtype=float), index=cumulative_log.index
    )
    sum_c = cumulative_log.rolling(window, min_periods=window).sum()
    sum_sc = cumulative_log.mul(positions, axis=0).rolling(
        window, min_periods=window
    ).sum()
    sum_c2 = (cumulative_log**2).rolling(window, min_periods=window).sum()

    sum_squares = float(window) * (float(window) ** 2 - 1.0) / 12.0
    centered_positions = positions - (window - 1.0) / 2.0
    slope = sum_sc.sub(sum_c.mul(centered_positions, axis=0)) / sum_squares
    total_squares = sum_c2 - sum_c**2 / float(window)
    residual_squares = (total_squares - slope**2 * sum_squares).clip(lower=0.0)
    standard_error = np.sqrt(
        (residual_squares / (window - 2.0)) / sum_squares
    ).replace(0.0, np.nan)
    statistic = slope / standard_error
    observed = _valid_observation_count(returns, window)
    return statistic.where(observed >= config.min_return_observations)


def baltas_kosowski_signal(
    returns: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
    *,
    hard_threshold: bool = False,
) -> pd.DataFrame:
    """Baltas-Kosowski trend estimator: the time-trend t-statistic as a position.

    The literature is read two ways and they are different strategies, so both
    are exposed and the caller must name which one it reported.  The default is
    the proportional reading, ``X = clip(t, -1, 1)``; ``hard_threshold=True`` is
    the strict reading, ``+1`` above ``+1``, ``-1`` below ``-1``, flat between.

    Published claim, for contrast: Lim, Zohren & Roberts report that
    t-statistic trend estimation gave Baltas and Kosowski a 66% reduction in
    portfolio turnover at comparable Sharpe ratios.  Whether the clip does any
    of that work on *this* panel is an empirical question the signal
    diagnostics answer directly through the mean absolute exposure.  A
    cumulative log return over 252 sessions is close to a random walk, and the
    OLS t of a random walk on time is routinely far outside +/-1; where that
    holds the clip is nearly inert, and any turnover reduction comes instead
    from the slope being a smoother estimator than the endpoint return.

    Baltas and Kosowski also recommend a range-based Yang-Zhang volatility
    estimator over close-to-close.  This panel carries opens and closes but no
    highs or lows, so that variant is not computable here at all; it is left
    unimplemented rather than substituted with a different estimator under the
    same name.
    """

    statistic = time_trend_tstatistic(returns, config)
    if hard_threshold:
        strong = statistic.abs() > 1.0
        return np.sign(statistic).where(strong, 0.0).where(statistic.notna())
    return statistic.clip(-1.0, 1.0)


def baz_macd_signal(
    prices: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.DataFrame:
    """Baz et al. (2015) volatility-normalised MACD, averaged over three scales.

    ``MACD = m(S) - m(L)``; ``q = MACD / std(p_{t-63:t})``;
    ``Y = q / std(q_{t-252:t})``; ``X = phi(Y)`` with
    ``phi(y) = y exp(-y^2/4) / 0.89``.  The published normalising constant is
    0.89 rather than the exact peak-normalising ``sqrt(2) e^{-1/2} = 0.857766``;
    0.89 is kept so the transform matches the literature.

    A time-scale ``n`` maps to ``alpha = 1/n``, i.e. a half-life of
    ``log(0.5)/log(1 - 1/n)``.  It does *not* map to pandas' ``span=n``: at
    ``n = 8`` that would be ``alpha = 0.222`` instead of ``0.125``, a different
    filter entirely.

    Aggregation across the three scale pairs is the *average of the transformed*
    signals, ``X = (1/3) sum_k phi(Y_k)``.  Lim, Zohren & Roberts' printed
    equation (8) sums the untransformed ``Y``, while their surrounding prose
    says the signals are averaged.  Averaging is chosen deliberately: it matches
    the prose, it keeps ``X`` bounded in the same +/-0.9638 range as the
    single-pair rule, and summing three ``phi`` values would reach +/-2.89 and
    so near-triple this benchmark's leverage against every other rule in the
    family, which is a scale artefact rather than an edge.  ``phi`` is
    non-monotone, so ``phi(sum Y)`` and ``mean(phi(Y))`` are genuinely different
    functions and not a rescale of one another.

    The back-adjusted panel is the right input here and nowhere else in this
    module.  A standard deviation is translation invariant and back-adjustment
    is a piecewise translation, so within a roll-free window the two panels give
    the same ``std(p)``; across a roll the unadjusted series contains a jump
    that is not a price move.  The numerator is likewise a difference of two
    moving averages of the level, so it too belongs on the continuous panel.
    """

    transformed = []
    for short, long in config.macd_time_scales:
        fast = prices.ewm(alpha=1.0 / short, adjust=False).mean()
        slow = prices.ewm(alpha=1.0 / long, adjust=False).mean()
        price_volatility = prices.rolling(
            config.macd_price_volatility_window,
            min_periods=config.macd_price_volatility_window,
        ).std()
        normalized = (fast - slow) / price_volatility.replace(0.0, np.nan)
        signal_volatility = normalized.rolling(
            config.macd_signal_volatility_window,
            min_periods=config.macd_signal_volatility_window,
        ).std()
        scaled = normalized / signal_volatility.replace(0.0, np.nan)
        transformed.append(
            scaled * np.exp(-(scaled**2) / 4.0) / config.macd_normalizing_constant
        )
    averaged = transformed[0]
    for value in transformed[1:]:
        averaged = averaged + value
    averaged = averaged / float(len(transformed))
    warm = (
        prices.notna()
        .astype(float)
        .rolling(config.macd_warmup_sessions, min_periods=config.macd_warmup_sessions)
        .sum()
        >= config.macd_warmup_sessions
    )
    return averaged.where(warm & prices.notna())


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _live_mask(
    signal: pd.DataFrame,
    volatility: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """Markets with a signal, a positive lagged volatility, and volume."""

    return (
        signal.notna()
        & volatility.notna()
        & volatility.gt(0)
        & eligible.reindex_like(signal).fillna(False)
    )


def inverse_volatility_weights(
    signal: pd.DataFrame,
    volatility: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    volatility_budget: float,
) -> pd.DataFrame:
    """The MOP allocator: ``w[i,t] = X[i,t] * (1/S_t) * (tau / sigma[i,t])``.

    Deliberately not ``strategy._base_target_positions``, which allocates
    ``target_vol * sqrt(class_weight / count)`` — a diversification-aware
    allocator with a portfolio-level volatility budget.  MOP have no
    portfolio-level control at all: the 40% is per instrument and is the entire
    risk model, and realised portfolio volatility is an output.  The weight is
    ``1/S_t``, never ``1/sqrt(S_t)``; with about 50 live markets the latter
    would over-lever the book roughly sevenfold.
    """

    live = _live_mask(signal, volatility, eligible)
    count = live.sum(axis=1).replace(0, np.nan)
    weights = (
        signal.where(live)
        .mul(volatility_budget)
        .div(volatility.where(live))
        .div(count, axis=0)
    )
    return weights.where(live, 0.0).replace([np.inf, -np.inf], 0.0)


def notional_weights_to_contracts_per_dollar(
    weights: pd.DataFrame,
    valuation_prices: pd.DataFrame,
    point_values: pd.DataFrame,
) -> pd.DataFrame:
    """Convert notional weights per dollar of NAV into contracts per dollar.

    ``n[i,t] = w[i,t] * NAV_t / (P_unadj[i,t] * point_value[i,t])``.  The
    unadjusted price is the notional; a back-adjusted level is not one, and
    using it would misstate exposure by the whole accumulated roll adjustment.
    A market whose valuation input is missing is zeroed rather than left NaN,
    because ``_limit_target_contracts`` refuses a nonzero target it cannot
    value.
    """

    denominator = (
        valuation_prices.reindex_like(weights) * point_values.reindex_like(weights)
    ).replace(0.0, np.nan)
    contracts = weights / denominator
    return contracts.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def hop_portfolio_weights(
    signal: pd.DataFrame,
    volatility: pd.DataFrame,
    eligible: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    decisions: pd.DatetimeIndex,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hurst, Ooi & Pedersen's portfolio volatility scaling, footnote 5 verbatim.

    "The positions across the three strategies are aggregated each month, and
    scaled such that the combined portfolio has an annualized ex-ante volatility
    target of 10%", using "a simple covariance matrix estimated using rolling
    3-year equally weighted monthly returns".

    ``Sigma_t`` is therefore the equally weighted sample covariance of the
    trailing 36 *monthly* market returns, annualised by 12, over months at or
    before the decision month.  With 59 markets and 36 observations it is rank
    deficient.  That is fine and is not repaired: the quadratic form
    ``w' Sigma w`` is well defined and non-negative on a positive semi-definite
    matrix, the inverse is never needed, and shrinking it would be a deviation
    from the paper rather than a fix.

    ``Sigma_t``, ``w`` and the ``1/S_t`` count are all restricted to the same
    set: markets with a live signal *and* a complete 36-month history.

    Returns ``(weights, diagnostics)``; the diagnostics frame carries the
    ex-ante volatility and the scaling constant per decision so the reader can
    see the estimator working rather than take the realised volatility on trust.
    """

    live = _live_mask(signal, volatility, eligible)
    rows: dict[pd.Timestamp, pd.Series] = {}
    diagnostics: list[dict[str, Any]] = []
    columns = signal.columns
    for decision in decisions:
        period = decision.to_period("M")
        history = monthly_returns.loc[:period]
        if len(history) < config.covariance_months:
            continue
        block = history.iloc[-config.covariance_months :]
        complete = block.notna().all(axis=0)
        selected = columns[live.loc[decision].to_numpy() & complete.to_numpy()]
        if len(selected) < 2:
            continue
        count = float(len(selected))
        raw = (
            signal.loc[decision, selected]
            * (config.instrument_volatility_budget / volatility.loc[decision, selected])
            / count
        )
        covariance = (
            np.cov(block[selected].to_numpy(dtype=float), rowvar=False, ddof=1) * 12.0
        )
        values = raw.to_numpy(dtype=float)
        variance = float(values @ covariance @ values)
        ex_ante = math.sqrt(max(variance, 0.0))
        if not np.isfinite(ex_ante) or ex_ante <= 0:
            continue
        scale = config.portfolio_volatility_budget / ex_ante
        rows[decision] = raw * scale
        diagnostics.append(
            {
                "decision": decision,
                "live_markets": int(count),
                "ex_ante_portfolio_volatility": ex_ante,
                "portfolio_scaling_constant": scale,
                "gross_notional_multiple": float(np.abs(values * scale).sum()),
            }
        )
    if not rows:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=columns, dtype=float)
        return empty, pd.DataFrame(
            columns=[
                "decision",
                "live_markets",
                "ex_ante_portfolio_volatility",
                "portfolio_scaling_constant",
                "gross_notional_multiple",
            ]
        )
    weights = pd.DataFrame(rows).T.reindex(columns=columns).fillna(0.0)
    weights.index = pd.DatetimeIndex(weights.index)
    return weights, pd.DataFrame(diagnostics)


# ---------------------------------------------------------------------------
# Volatility-management overlays
# ---------------------------------------------------------------------------


def barroso_santa_clara_weights(
    daily_returns: pd.Series,
    decisions: pd.DatetimeIndex,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.Series:
    """Barroso & Santa-Clara (2015) constant-volatility weight, uncapped.

    ``sigmahat^2_t = 21 * (1/126) * sum_{j=1..126} r^2_{t-j}`` — a *monthly*
    variance, and not demeaned, because it is a realised-variance estimator.
    The target must therefore also be monthly: ``0.12 / sqrt(12) = 0.034641``.
    Pairing the 21-scaled estimator with an annual 12% is the standard
    transcription error and produces a weight wrong by ``sqrt(12) = 3.46``.

    The weight is uncapped, as in the paper.  Capping it would be a third
    strategy, and the uncapped distribution is the object of interest: it is
    what makes this overlay informative against the incumbent's
    ``max_risk_scalar = 2.00``.

    The window ends at the decision session inclusive and scales the following
    month, so nothing after the decision enters it.
    """

    squares = daily_returns.astype(float) ** 2
    realized = (
        squares.rolling(
            config.barroso_window_sessions,
            min_periods=config.barroso_window_sessions,
        ).mean()
        * config.barroso_sessions_per_month
    )
    monthly_volatility = np.sqrt(realized)
    monthly_budget = config.barroso_volatility_budget / math.sqrt(12.0)
    weights = monthly_budget / monthly_volatility.replace(0.0, np.nan)
    return weights.reindex(decisions)


def moreira_muir_weights(
    daily_returns: pd.Series,
    decisions: pd.DatetimeIndex,
    config: BenchmarkConfig = BenchmarkConfig(),
    *,
    expanding_constant: bool,
) -> pd.Series:
    """Moreira & Muir (2017) inverse previous-month realised variance weight.

    ``f^sigma_{t+1} = (c / RV^2_t) f_{t+1}`` with ``RV^2_t`` the sum of squared
    *demeaned* daily returns within month ``t``.  Inverse variance, not inverse
    volatility, and demeaned — both differ from Barroso-Santa-Clara on purpose,
    and conflating them destroys the contrast the pair exists to draw.

    The month a decision reads is the one thing here that is easy to get wrong
    by one.  A decision is taken on the *last* session of month ``m`` and
    governs the book through month ``m+1``, and the rule scales month ``m+1``
    by ``RV`` of month ``m``.  So the weight at that decision is ``c / RV[m]``:
    its own month, complete at that session's close, not the month before it.
    Reading the prior month instead would still look like a volatility overlay
    — it is finite, it is causal, and its distribution is plausible — while
    carrying almost none of the rule's conditioning information, and it would
    apply a constant fitted on one series to a series lagged a month against
    it.  ``barroso_santa_clara_weights`` uses the same convention: its window
    ends at the decision session inclusive.

    ``c`` is where the look-ahead lives.  As the paper defines it, ``c`` makes
    the managed series match the unmanaged series' *full-sample* unconditional
    standard deviation.  For a spanning-regression alpha that is harmless
    because the statistic is scale invariant; for a backtest it sets today's
    leverage from the whole future's volatility, which ``docs/causality.md``
    forbids.  Both are therefore available:

    ``expanding_constant=False`` is the paper's own definition and is
    **NON-CAUSAL**.  It exists so the size of the look-ahead can be measured,
    and must never enter a comparison that could support a decision.

    ``expanding_constant=True`` recomputes ``c_t`` from data through the
    decision month only, and is the implementable version.  The difference
    between the two rows is itself the reportable statistic: it measures how
    much of the published effect is the constant, which is the failure Cederburg
    et al. (2020) document across 103 equity strategies.

    Published claim, for contrast: MM report significantly positive alphas in
    spanning regressions against the unmanaged factor, across the market,
    momentum, value, profitability, ROE and currency carry.  A spanning alpha is
    scale invariant, so it says nothing directly about whether the *level* of
    drawdown improves — a positive alpha is fully compatible with a larger
    maximum drawdown, which is why the comparison table reports drawdown over
    volatility rather than drawdown alone.  Before
    ``moreira_min_history_months`` of history the constant is not estimable and
    the weight is 1.0, i.e. the book is left unmanaged; the first genuinely
    scaled decision is reported alongside the weight distribution.
    """

    values = daily_returns.astype(float)
    periods = values.index.to_period("M")
    counts = values.groupby(periods).count()
    demeaned = values - values.groupby(periods).transform("mean")
    realized_variance = (demeaned**2).groupby(periods).sum()
    realized_variance = realized_variance.where(
        counts >= config.moreira_min_month_sessions
    )
    realized_variance = realized_variance.where(realized_variance > 0)

    # The proxy series ``c`` is fitted on: a return dated inside month ``k`` is
    # scaled by the variance of month ``k-1``, which is the rule stated at the
    # return level.  It is the same mapping the decision weights below use,
    # seen from the other side of the decision.
    previous_variance = realized_variance.shift(1)
    unscaled = previous_variance.reindex(periods).to_numpy(dtype=float)
    multiplier = pd.Series(1.0 / unscaled, index=values.index)
    managed = multiplier * values

    if not expanding_constant:
        unmanaged_deviation = float(values.std(ddof=1))
        managed_deviation = float(managed.dropna().std(ddof=1))
        if not np.isfinite(managed_deviation) or managed_deviation <= 0:
            raise ValueError("the managed proxy series has no dispersion")
        constant = pd.Series(
            unmanaged_deviation / managed_deviation, index=decisions, dtype=float
        )
    else:
        month_ends = pd.Series(periods, index=values.index)
        constants: dict[pd.Timestamp, float] = {}
        for decision in decisions:
            window = values.index <= decision
            observed_months = month_ends[window].nunique()
            if observed_months < config.moreira_min_history_months:
                continue
            left = values[window]
            right = managed[window].dropna()
            if len(right) < 2 or len(left) < 2:
                continue
            right_deviation = float(right.std(ddof=1))
            if not np.isfinite(right_deviation) or right_deviation <= 0:
                continue
            constants[decision] = float(left.std(ddof=1)) / right_deviation
        constant = pd.Series(constants, dtype=float).reindex(decisions)

    decision_variance = realized_variance.reindex(
        pd.PeriodIndex(decisions, freq="M")
    ).to_numpy(dtype=float)
    weights = pd.Series(
        constant.to_numpy(dtype=float) / decision_variance, index=decisions
    )
    return weights


def overlay_weight_report(
    weights: pd.Series,
    *,
    overlay: str,
    underlying: str,
    causality: str,
    maximum_risk_scalar_reference: float,
) -> pd.DataFrame:
    """Distribution of an overlay's realised weight, including its extremes.

    The extremes are the point.  An uncapped inverse-variance rule asks for
    leverage that a clipped ``[0.25, 2.00]`` band would refuse, and the honest
    way to show what the band costs is to report what the unclipped rule
    actually demanded rather than to describe it.
    """

    finite = weights.dropna()
    finite = finite[np.isfinite(finite.to_numpy(dtype=float))]
    if finite.empty:
        row: dict[str, Any] = dict.fromkeys(OVERLAY_WEIGHT_COLUMNS, np.nan)
        row.update(
            {
                "overlay": overlay,
                "underlying": underlying,
                "causality": causality,
                "decisions": 0,
                "first_active_decision": "",
                "incumbent_maximum_risk_scalar_reference": (
                    maximum_risk_scalar_reference
                ),
                "selection_adjusted": False,
                "permitted_use": _PERMITTED_USE,
            }
        )
        return pd.DataFrame([row], columns=OVERLAY_WEIGHT_COLUMNS)
    active = finite[finite.ne(1.0)]
    values = finite.to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "overlay": overlay,
                "underlying": underlying,
                "causality": causality,
                "decisions": int(finite.size),
                "first_active_decision": (
                    active.index[0].date().isoformat() if not active.empty else ""
                ),
                "weight_minimum": float(values.min()),
                "weight_percentile_05": float(np.percentile(values, 5.0)),
                "weight_median": float(np.median(values)),
                "weight_mean": float(values.mean()),
                "weight_percentile_95": float(np.percentile(values, 95.0)),
                "weight_percentile_99": float(np.percentile(values, 99.0)),
                "weight_maximum": float(values.max()),
                "incumbent_maximum_risk_scalar_reference": float(
                    maximum_risk_scalar_reference
                ),
                "selection_adjusted": False,
                "permitted_use": _PERMITTED_USE,
            }
        ],
        columns=OVERLAY_WEIGHT_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Execution through the incumbent's own ledger
# ---------------------------------------------------------------------------


def _require_matching_fees(inputs: EngineInputs, config: StrategyConfig) -> None:
    """Refuse a config whose fees differ from the pre-derived cost frame.

    ``one_way_costs`` is built once, from one config, and then reused by every
    benchmark.  A caller may legitimately vary the config afterwards — releasing
    the gross-notional cap for a faithful MOP replication requires it — but a
    varied *fee* would be silently ignored, because the fee is already baked
    into the frame.  A benchmark quietly paying the wrong commission is exactly
    the class of error this module exists to rule out, so it raises.
    """

    observed = _fee_parameters(config)
    if observed != inputs.fee_parameters:
        raise ValueError(
            "the supplied config's fee parameters "
            f"{observed} differ from the ones baked into one_way_costs "
            f"{inputs.fee_parameters}; rebuild the engine inputs from the same "
            "config rather than varying fees after the fact"
        )


def simulate_monthly_targets(
    monthly_contracts_per_dollar: pd.DataFrame,
    inputs: EngineInputs,
    config: StrategyConfig,
    *,
    name: str,
    risk_scalar: pd.Series | None = None,
) -> BacktestResult:
    """Run a benchmark's monthly decision frame through the incumbent's ledger.

    Only the decision frame differs from ``run_backtest``.  The cost model,
    integer rounding, the lagged-volume participation cap, the realized-capacity
    fill truncation, the roll backlog, portfolio limits and the FX conversion
    are the same code operating on the same inputs, which is why a difference
    between two rows of the comparison is attributable to the signal.

    ``risk_scalar`` records an overlay's applied weight for reporting.  It is
    *not* applied here — an overlay must already be inside the supplied targets
    so it pays real transaction costs — and defaults to 1.0, which is the honest
    value for a benchmark that has no portfolio-level volatility control.
    """

    _require_matching_fees(inputs, config)
    ledger = _simulate_execution(
        monthly_contracts_per_dollar,
        inputs.prices,
        inputs.valuation_prices,
        inputs.observed_opens,
        inputs.observed_closes,
        inputs.volumes,
        inputs.point_values,
        inputs.one_way_costs,
        inputs.margin_values,
        inputs.rolls,
        config,
        initial_capital=config.initial_capital,
        integer_contracts=config.integer_contracts,
        charge_costs=True,
        launch_date=config.launch_date,
    )
    daily = ledger.daily.copy()
    if risk_scalar is None:
        daily["risk_scalar"] = 1.0
    else:
        daily["risk_scalar"] = (
            risk_scalar.reindex(inputs.prices.index).ffill().shift(1).fillna(1.0)
        )
    daily["active_markets"] = ledger.positions.ne(0).sum(axis=1)
    daily["delivery_label_anomalies"] = inputs.delivery_anomalies.sum(axis=1)
    result = BacktestResult(
        name=name,
        daily=daily,
        positions=ledger.positions,
        target_positions=monthly_contracts_per_dollar,
        signals=pd.DataFrame(index=inputs.prices.index, columns=inputs.prices.columns),
        prices=inputs.prices,
        metadata=pd.DataFrame(),
        trades=ledger.trades,
        roll_events=inputs.rolls,
        executed_rolls=ledger.executed_rolls,
        gross_pnl_by_market=ledger.gross_pnl_by_market,
        regular_cost_by_market=ledger.regular_cost_by_market,
        roll_cost_by_market=ledger.roll_cost_by_market,
        trade_episodes=pd.DataFrame(),
        execution_timing=config.execution_timing,
    )
    result.trade_episodes = build_trade_episodes(result)
    return result


def assert_seam_reproduces_incumbent(
    incumbent_result: Any,
    inputs: EngineInputs,
    config: StrategyConfig,
    *,
    tolerance: float = 0.0,
) -> float:
    """Replay the incumbent's own decision frame and require the same ledger.

    This is the proof behind the module's central claim.  ``run_backtest``
    computes a monthly decision frame and hands it to ``_simulate_execution``;
    every benchmark here computes a *different* decision frame and hands it to
    the same function with the same inputs.  Feeding the incumbent's own frame
    back through this module's call must therefore reproduce the incumbent's
    daily ledger exactly, and if it does not then some argument differs and
    every benchmark row is being executed under a subtly different model.

    The default tolerance is zero: this is an identity, not an approximation.
    Returns the observed maximum absolute deviation so a caller can record it.
    """

    _require_matching_fees(inputs, config)
    replayed = _simulate_execution(
        incumbent_result.target_positions,
        inputs.prices,
        inputs.valuation_prices,
        inputs.observed_opens,
        inputs.observed_closes,
        inputs.volumes,
        inputs.point_values,
        inputs.one_way_costs,
        inputs.margin_values,
        inputs.rolls,
        config,
        initial_capital=config.initial_capital,
        integer_contracts=config.integer_contracts,
        charge_costs=True,
        launch_date=config.launch_date,
    )
    deviation = float(
        (replayed.daily["net_return"] - incumbent_result.daily["net_return"])
        .abs()
        .max()
    )
    if not (deviation <= tolerance):
        raise ValueError(
            "the benchmark execution seam does not reproduce the incumbent's "
            f"own ledger: maximum absolute net-return deviation {deviation:.3e} "
            f"exceeds {tolerance:.3e}. Every benchmark row would be running "
            "under a different execution model than the incumbent."
        )
    return deviation


def solve_volatility_matched_budget(
    build: Callable[[float], pd.DataFrame],
    inputs: EngineInputs,
    config: StrategyConfig,
    *,
    volatility: float,
    start: str = RETROSPECTIVE_START,
    end: str = RETROSPECTIVE_END,
    lower: float = 0.005,
    upper: float = 1.0,
    tolerance: float = 1e-4,
    max_iterations: int = 24,
) -> tuple[float, BacktestResult]:
    """Bisect the per-instrument volatility budget to a realised volatility.

    Used only for the matched-volatility long-only row, and it is an
    **ex-post, full-sample** calibration in exactly Harvey et al.'s sense: the
    constant is chosen so the finished series realises a stated volatility over
    the whole window.  It is not causal and any row built from it must say so.
    Realised volatility is monotone in the budget only approximately — integer
    contracts, the participation cap and the gross-notional cap all intervene —
    so the bisection is bounded and returns its best bracket rather than
    iterating to machine precision.

    A budget large enough for the engine to refuse the book outright (a roll it
    cannot clear inside ``max_roll_backlog_sessions``, or a non-positive NAV) is
    treated as too large and lowers the upper bound.  That is the same monotone
    assumption the search already makes about volatility, and without it a
    single refusal in the middle of the bracket would abandon the whole row.

    The search itself calls the execution ledger directly and skips the trade
    episode builder, which the realised volatility does not depend on; the full
    result object is assembled once, for the budget the search settles on.
    """

    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("volatility must be finite and positive")
    if not 0 < lower < upper:
        raise ValueError("bisection bounds must satisfy 0 < lower < upper")
    _require_matching_fees(inputs, config)

    best: tuple[float, float] | None = None
    low, high = float(lower), float(upper)
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        try:
            ledger = _simulate_execution(
                build(middle),
                inputs.prices,
                inputs.valuation_prices,
                inputs.observed_opens,
                inputs.observed_closes,
                inputs.volumes,
                inputs.point_values,
                inputs.one_way_costs,
                inputs.margin_values,
                inputs.rolls,
                config,
                initial_capital=config.initial_capital,
                integer_contracts=config.integer_contracts,
                charge_costs=True,
                launch_date=config.launch_date,
            )
        except Exception:
            high = middle
            continue
        realized = float(
            ledger.daily.loc[start:end, "net_return"].std()
            * math.sqrt(config.annualization)
        )
        error = abs(realized - volatility)
        if best is None or error < best[1]:
            best = (middle, error)
        if error <= tolerance:
            break
        if realized < volatility:
            low = middle
        else:
            high = middle
    if best is None:
        raise ValueError(
            "the engine refused every candidate budget in "
            f"[{lower}, {upper}]; no volatility-matched row is available"
        )
    budget = best[0]
    return budget, simulate_monthly_targets(
        build(budget), inputs, config, name=f"volatility_matched_{budget:.6f}"
    )


# ---------------------------------------------------------------------------
# The no-skill sign-permutation null
# ---------------------------------------------------------------------------


def _run_identifiers(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Label each market's contiguous constant-sign runs with a unique integer.

    Flat positions get identifier ``-1`` and are never flipped: flipping a zero
    is a no-op that would consume randomisation entropy and overstate the
    number of independent draws behind the null.

    Column-major accumulation is what keeps a run at the top of one market from
    continuing the run at the bottom of the previous one; the first observed
    row of every market always opens a fresh run.
    """

    signs = np.sign(np.asarray(values, dtype=float))
    live = signs != 0.0
    change = np.empty(signs.shape, dtype=bool)
    change[0] = live[0]
    change[1:] = live[1:] & (signs[1:] != signs[:-1])
    flat_change = change.ravel(order="F")
    flat_identifiers = np.cumsum(flat_change) - 1
    flat_identifiers[~live.ravel(order="F")] = -1
    identifiers = flat_identifiers.reshape(signs.shape, order="F").astype(np.int64)
    return identifiers, int(flat_change.sum())


def _median_run_length(identifiers: np.ndarray, total: int) -> float:
    if total == 0:
        return float("nan")
    flat = identifiers.ravel()
    flat = flat[flat >= 0]
    if flat.size == 0:
        return float("nan")
    lengths = np.bincount(flat, minlength=total)
    return float(np.median(lengths[lengths > 0]))


def sign_permutation_null(
    monthly_contracts_per_dollar: pd.DataFrame,
    inputs: EngineInputs,
    config: StrategyConfig,
    *,
    incumbent_daily: pd.Series,
    mode: str = BLOCK_SIGN_FLIP,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    start: str = RETROSPECTIVE_START,
    end: str = RETROSPECTIVE_END,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No-skill null built by flipping the signs of the incumbent's own book.

    The design holds sizing, execution, costs, the participation cap, integer
    rounding, roll obligations and market exposure fixed, and destroys exactly
    one thing: direction.  Every permutation is pushed through
    ``_simulate_execution`` rather than applied to the daily return series,
    because a sign flip is not a sign flip in P&L — ``_limit_target_contracts``,
    the fill cap at realised session volume, the roll backlog and integer
    rounding are all nonlinear in the target, and that nonlinearity is precisely
    what a return-level flip would hide.

    ``mode`` selects the construction, and the distinction is load-bearing:

    ``block_sign_flip`` (the headline) draws one sign per contiguous constant-
    sign run per market, so the within-run magnitude path is preserved.  Two
    honest caveats follow, and both are reported rather than asserted away.
    First, run *boundaries* are not preserved: two adjacent runs that draw the
    same sign merge, so the permuted book makes strictly fewer position changes
    than the incumbent, and ``null_median_run_decisions`` measures by how much.
    Second, a boundary that was a reversal can become a resize, so trade
    magnitudes at boundaries differ.

    Contract turnover is reported for both books but is *not* the right
    comparison between them: a contract count scales with NAV, and the
    incumbent compounds many times over this window while a no-skill book does
    not, so the incumbent's contract turnover is inflated relative to the
    null's for a reason that has nothing to do with trading intensity.  The
    scale-free comparison is the annual cost drag, which is a fraction of NAV
    by construction, and it is reported for both.

    ``iid_sign_flip`` draws independently per market-month.  It shreds position
    persistence, so its turnover and cost drag are far above the incumbent's and
    its Sharpe null is biased down by a cost artefact rather than only by the
    loss of skill.  It is a sanity bound, not a headline.

    The validation check is the *centre* of the null, not its tails.  Each
    permutation pays the full cost model against an expected gross return of
    zero, so the null Sharpe distribution must sit near
    ``-(annual cost drag)/(annual volatility)``.  A null centred at zero means
    costs are not being charged.

    Returns ``(draws, summary)``.  The p-value is
    ``(1 + #{b : S*_b >= S_obs}) / (1 + B)``; the plain proportion can return
    exactly zero, which is never a legitimate p-value.
    """

    if mode not in {BLOCK_SIGN_FLIP, IID_SIGN_FLIP}:
        raise ValueError(f"unknown permutation mode: {mode!r}")
    if isinstance(permutations, bool) or not isinstance(permutations, int):
        raise ValueError("permutations must be an integer")
    if permutations < 1:
        raise ValueError("permutations must be at least one")
    if monthly_contracts_per_dollar.empty:
        raise ValueError("the permutation null needs a non-empty decision frame")
    _require_matching_fees(inputs, config)

    values = monthly_contracts_per_dollar.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    identifiers, run_total = _run_identifiers(values)
    nonzero = values != 0.0

    incumbent = incumbent_daily.loc[start:end].dropna()
    incumbent_sharpe = float(
        incumbent.mean() / incumbent.std() * math.sqrt(config.annualization)
    )

    method_seed = zlib.crc32(b"benchmark sign permutation null") & 0xFFFFFFFF
    label_seed = zlib.crc32(mode.encode("utf-8")) & 0xFFFFFFFF

    draws: list[dict[str, Any]] = []
    for index in range(permutations):
        entropy = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, method_seed, label_seed, index])
        rng = np.random.default_rng(entropy)
        if mode == BLOCK_SIGN_FLIP:
            run_signs = rng.integers(0, 2, size=max(run_total, 1)) * 2 - 1
            flips = np.where(identifiers >= 0, run_signs[identifiers], 1)
        else:
            flips = np.where(nonzero, rng.integers(0, 2, size=values.shape) * 2 - 1, 1)
        permuted = pd.DataFrame(
            values * flips,
            index=monthly_contracts_per_dollar.index,
            columns=monthly_contracts_per_dollar.columns,
        )
        try:
            ledger = _simulate_execution(
                permuted,
                inputs.prices,
                inputs.valuation_prices,
                inputs.observed_opens,
                inputs.observed_closes,
                inputs.volumes,
                inputs.point_values,
                inputs.one_way_costs,
                inputs.margin_values,
                inputs.rolls,
                config,
                initial_capital=config.initial_capital,
                integer_contracts=config.integer_contracts,
                charge_costs=True,
                launch_date=config.launch_date,
            )
        except Exception as error:  # a permuted book can breach a hard engine limit
            draws.append(
                {
                    "permutation": index,
                    "run_status": RUN_ERRORED,
                    "run_detail": f"{type(error).__name__}: {error}"[:200],
                    "sharpe": np.nan,
                    "annualized_volatility": np.nan,
                    "annual_cost_drag": np.nan,
                    "annual_total_turnover_contracts": np.nan,
                    "median_run_decisions": np.nan,
                }
            )
            if progress is not None:
                progress(index + 1, permutations)
            continue
        window = ledger.daily.loc[start:end]
        returns = window["net_return"].dropna()
        deviation = float(returns.std())
        years = max((window.index[-1] - window.index[0]).days / 365.2425, 1e-9)
        permuted_identifiers, permuted_total = _run_identifiers(permuted.to_numpy(dtype=float))
        draws.append(
            {
                "permutation": index,
                "run_status": RUN_COMPLETED,
                "run_detail": "",
                "sharpe": (
                    float(returns.mean() / deviation * math.sqrt(config.annualization))
                    if deviation > 0
                    else np.nan
                ),
                "annualized_volatility": deviation * math.sqrt(config.annualization),
                "annual_cost_drag": float(window["cost"].mean() * config.annualization),
                "annual_total_turnover_contracts": (
                    float(window["total_contract_turnover"].sum()) / years
                ),
                "median_run_decisions": _median_run_length(
                    permuted_identifiers, permuted_total
                ),
            }
        )
        if progress is not None:
            progress(index + 1, permutations)

    draws_frame = pd.DataFrame(draws)
    completed = draws_frame[draws_frame["run_status"] == RUN_COMPLETED]
    sharpes = completed["sharpe"].dropna().to_numpy(dtype=float)
    total = int(sharpes.size)

    if total == 0:
        summary_values: dict[str, Any] = dict.fromkeys(PERMUTATION_NULL_COLUMNS, np.nan)
    else:
        exceedances = int(np.sum(sharpes >= incumbent_sharpe))
        summary_values = {
            "null_sharpe_mean": float(sharpes.mean()),
            "null_sharpe_standard_deviation": float(sharpes.std(ddof=1))
            if total > 1
            else np.nan,
            "null_sharpe_percentile_01": float(np.percentile(sharpes, 1.0)),
            "null_sharpe_percentile_05": float(np.percentile(sharpes, 5.0)),
            "null_sharpe_percentile_50": float(np.percentile(sharpes, 50.0)),
            "null_sharpe_percentile_95": float(np.percentile(sharpes, 95.0)),
            "null_sharpe_percentile_99": float(np.percentile(sharpes, 99.0)),
            "incumbent_percentile_in_null": float(
                100.0 * np.mean(sharpes < incumbent_sharpe)
            ),
            "empirical_one_sided_p_value": (1.0 + exceedances) / (1.0 + total),
            "cost_only_sharpe_reference": float(
                (
                    -completed["annual_cost_drag"]
                    / completed["annualized_volatility"]
                ).mean()
            ),
            "null_annual_cost_drag_mean": float(completed["annual_cost_drag"].mean()),
            "null_annual_turnover_mean": float(
                completed["annual_total_turnover_contracts"].mean()
            ),
            "null_median_run_decisions": float(
                completed["median_run_decisions"].mean()
            ),
        }

    summary_values.update(
        {
            "null": f"{mode}_on_incumbent_decisions",
            "construction": (
                "signs of the incumbent's own monthly decision frame permuted; "
                "sizing, costs, capacity and execution held fixed"
            ),
            "permutations": int(permutations),
            "permutations_completed": total,
            "seed": int(seed),
            "resolution": 1.0 / (total + 1.0) if total else np.nan,
            "incumbent_sharpe": incumbent_sharpe,
            "incumbent_median_run_decisions": _median_run_length(identifiers, run_total),
            "selection_adjusted": False,
            "permitted_use": _PERMITTED_USE,
        }
    )
    summary = pd.DataFrame([summary_values], columns=PERMUTATION_NULL_COLUMNS)
    return draws_frame, summary


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


def _hac_ols(
    dependent: np.ndarray,
    design: np.ndarray,
    lags: int,
) -> dict[str, Any]:
    """OLS with a Bartlett-kernel Newey-West sandwich covariance.

    The kernel weight and the ``1/n`` normalisation match
    ``inference.hac_standard_error`` exactly, so with an intercept-only design
    this returns that function's number to machine precision.  A test asserts
    the identity rather than trusting the comment: two HAC estimators already
    coexist in this repository and disagree slightly, and a third silent one
    would be worse than either.
    """

    y = np.asarray(dependent, dtype=float)
    x = np.asarray(design, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("design must be (n, k) and dependent must be (n,)")
    if lags < 0:
        raise ValueError("lags cannot be negative")
    count, parameters = x.shape
    if count <= parameters:
        raise ValueError("a regression needs more observations than parameters")
    gram_inverse = np.linalg.pinv(x.T @ x)
    coefficients = gram_inverse @ (x.T @ y)
    residuals = y - x @ coefficients
    scores = x * residuals[:, None]
    spectral = scores.T @ scores
    for lag in range(1, min(lags, count - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        spectral = spectral + weight * (cross + cross.T)
    covariance = gram_inverse @ spectral @ gram_inverse
    variances = np.diag(covariance)
    standard_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
    centered = y - y.mean()
    total = float(centered @ centered)
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else np.nan
    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "residuals": residuals,
        "r_squared": r_squared,
        "observations": count,
        "parameters": parameters,
    }


def _aligned_frame(paths: Mapping[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame({name: series.astype(float) for name, series in paths.items()})
    frame = frame.dropna(how="any")
    if frame.empty:
        raise ValueError("no common sessions across the supplied return paths")
    return frame


def benchmark_alpha_report(
    incumbent: pd.Series,
    benchmarks: Mapping[str, pd.Series],
    *,
    annualization: int = 252,
    lags: int = inference.DEFAULT_HAC_LAGS,
) -> pd.DataFrame:
    """Univariate regression of the incumbent on each benchmark in turn.

    Reports the annualised intercept with a Newey-West standard error, the
    slope, the fit, and both an appraisal ratio (alpha over residual volatility)
    and an information ratio (mean active return over tracking error), which are
    different statistics and are labelled as such.

    ``alpha_hedged_mean_t_statistic`` is the same intercept tested a second way,
    as the Newey-West t of the mean of ``y - beta x`` through
    ``inference.hac_standard_error``.  It ignores the sampling variation of the
    slope, so it is an approximation — but it is an *independent* one, computed
    by the repository's existing estimator, and a material gap between the two
    columns is a wiring alarm rather than a finding.
    """

    rows: list[dict[str, Any]] = []
    for name, path in benchmarks.items():
        frame = _aligned_frame({"incumbent": incumbent, name: path})
        y = frame["incumbent"].to_numpy(dtype=float)
        x = frame[name].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        fit = _hac_ols(y, design, lags)
        alpha, beta = fit["coefficients"]
        alpha_error, beta_error = fit["standard_errors"]
        residuals = fit["residuals"]
        residual_volatility = float(np.std(residuals, ddof=1) * math.sqrt(annualization))
        active = y - x
        tracking_error = float(np.std(active, ddof=1) * math.sqrt(annualization))
        hedged = y - beta * x
        hedged_error = inference.hac_standard_error(hedged, lags)
        rows.append(
            {
                "regression": f"incumbent_on_{name}",
                "explanatory_series": name,
                "method": f"OLS with Newey-West ({lags} lags) standard errors",
                "sessions": int(fit["observations"]),
                "alpha_annualized": float(alpha) * annualization,
                "alpha_hac_standard_error_annualized": float(alpha_error) * annualization,
                "alpha_hac_t_statistic": (
                    float(alpha / alpha_error) if np.isfinite(alpha_error) else np.nan
                ),
                "alpha_hedged_mean_t_statistic": (
                    float(np.mean(hedged) / hedged_error)
                    if np.isfinite(hedged_error) and hedged_error > 0
                    else np.nan
                ),
                "beta": float(beta),
                "beta_hac_t_statistic": (
                    float(beta / beta_error) if np.isfinite(beta_error) else np.nan
                ),
                "r_squared": float(fit["r_squared"]),
                "residual_volatility_annualized": residual_volatility,
                "appraisal_ratio": (
                    float(alpha) * annualization / residual_volatility
                    if residual_volatility > 0
                    else np.nan
                ),
                "tracking_error_annualized": tracking_error,
                "information_ratio": (
                    float(active.mean()) * annualization / tracking_error
                    if tracking_error > 0
                    else np.nan
                ),
                "return_correlation": float(np.corrcoef(y, x)[0, 1]),
                "hac_lags": int(lags),
                "selection_adjusted": False,
                "permitted_use": _PERMITTED_USE,
            }
        )
    return pd.DataFrame(rows, columns=ALPHA_REPORT_COLUMNS)


def spanning_report(
    incumbent: pd.Series,
    benchmarks: Mapping[str, pd.Series],
    *,
    regression: str = "incumbent_on_published_trend_family",
    annualization: int = 252,
    lags: int = inference.DEFAULT_HAC_LAGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint regression of the incumbent on the whole published family.

    This is the question a committee actually asks: is the incumbent anything
    more than a repackaged combination of published trend factors?  The answer
    is the intercept, and the intercept is only interpretable if the family is
    declared before it is run — adding a benchmark after seeing the residual
    would make the number meaningless.

    The family must not contain any series derived from the incumbent itself
    (an overlay applied to the incumbent's own book, for example), because such
    a regressor spans the dependent variable by construction and drives the
    intercept to zero for an arithmetic reason rather than an economic one.
    Passing one raises.

    The regressors here are all trend rules on one panel and are therefore
    highly collinear.  Collinearity inflates individual coefficient standard
    errors but does not bias the intercept, which is the statistic of interest;
    the design's condition number is reported so the reader can see how much of
    the per-coefficient detail to trust.
    """

    if not benchmarks:
        raise ValueError("a spanning regression needs at least one explanatory series")
    for name in benchmarks:
        if "incumbent" in name:
            raise ValueError(
                "a spanning family may not contain a series derived from the "
                f"incumbent; {name!r} does"
            )
    frame = _aligned_frame({"incumbent": incumbent, **dict(benchmarks)})
    names = [name for name in benchmarks]
    y = frame["incumbent"].to_numpy(dtype=float)
    x = frame[names].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), x])
    fit = _hac_ols(y, design, lags)
    coefficients = fit["coefficients"]
    errors = fit["standard_errors"]
    residuals = fit["residuals"]
    residual_volatility = float(np.std(residuals, ddof=1) * math.sqrt(annualization))
    alpha, alpha_error = float(coefficients[0]), float(errors[0])
    observations = int(fit["observations"])
    parameters = int(fit["parameters"])
    adjusted = (
        1.0 - (1.0 - fit["r_squared"]) * (observations - 1) / (observations - parameters)
        if observations > parameters
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "regression": regression,
                "method": f"OLS with Newey-West ({lags} lags) standard errors",
                "sessions": observations,
                "explanatory_series_count": len(names),
                "explanatory_series": "; ".join(names),
                "alpha_annualized": alpha * annualization,
                "alpha_hac_standard_error_annualized": alpha_error * annualization,
                "alpha_hac_t_statistic": (
                    alpha / alpha_error if np.isfinite(alpha_error) else np.nan
                ),
                "r_squared": float(fit["r_squared"]),
                "adjusted_r_squared": float(adjusted),
                "residual_volatility_annualized": residual_volatility,
                "appraisal_ratio": (
                    alpha * annualization / residual_volatility
                    if residual_volatility > 0
                    else np.nan
                ),
                "coefficient_gross_exposure": float(np.abs(coefficients[1:]).sum()),
                "design_condition_number": float(np.linalg.cond(design)),
                "hac_lags": int(lags),
                "selection_adjusted": False,
                "permitted_use": _PERMITTED_USE,
            }
        ],
        columns=SPANNING_REPORT_COLUMNS,
    )
    coefficient_rows = [
        {
            "regression": regression,
            "explanatory_series": "intercept",
            "coefficient": alpha,
            "hac_standard_error": alpha_error,
            "hac_t_statistic": (
                alpha / alpha_error if np.isfinite(alpha_error) else np.nan
            ),
        }
    ]
    for position, name in enumerate(names, start=1):
        coefficient = float(coefficients[position])
        error = float(errors[position])
        coefficient_rows.append(
            {
                "regression": regression,
                "explanatory_series": name,
                "coefficient": coefficient,
                "hac_standard_error": error,
                "hac_t_statistic": (
                    coefficient / error if np.isfinite(error) else np.nan
                ),
            }
        )
    return summary, pd.DataFrame(
        coefficient_rows, columns=SPANNING_COEFFICIENT_COLUMNS
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _turnover(daily: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Annualised turnover.  Contract counts are *not* scale free.

    A contract count grows with NAV, so a book that compounds 22x over the
    window reports roughly 22x the contract turnover of an otherwise identical
    book that does not compound — the incumbent against a no-skill permutation
    is exactly that comparison.  The scale-free readings are the annual cost
    drag, which is already a fraction of NAV, and the notional turnover ratio
    added by ``_notional_turnover``.
    """

    window = daily.loc[start:end]
    if window.empty:
        return {
            "annual_rebalance_turnover_contracts": np.nan,
            "annual_roll_turnover_contracts": np.nan,
            "annual_total_turnover_contracts": np.nan,
        }
    years = max((window.index[-1] - window.index[0]).days / 365.2425, 1e-9)
    return {
        "annual_rebalance_turnover_contracts": (
            float(window["rebalance_contract_turnover"].sum()) / years
        ),
        "annual_roll_turnover_contracts": (
            float(window["roll_contract_turnover_increment"].sum()) / years
        ),
        "annual_total_turnover_contracts": (
            float(window["total_contract_turnover"].sum()) / years
        ),
    }


def _notional_turnover(
    result: Any,
    valuation_prices: pd.DataFrame | None,
    point_values: pd.DataFrame | None,
    start: str,
    end: str,
) -> float:
    """Rebalance turnover as multiples of NAV traded per year.

    Scale free, so it survives the NAV-compounding confound that makes raw
    contract counts incomparable across books with very different growth.  It
    covers the rebalance leg only: the ledger records roll turnover as an
    aggregate contract increment rather than per market, so a notional roll
    figure cannot be reconstructed without re-deriving it, and an incomplete
    figure presented as a total would be worse than none.
    """

    trades = getattr(result, "trades", None)
    daily = getattr(result, "daily", None)
    if (
        trades is None
        or daily is None
        or valuation_prices is None
        or point_values is None
    ):
        return float("nan")
    window = trades.loc[start:end]
    if window.empty:
        return float("nan")
    notional = (
        window.abs()
        * valuation_prices.reindex_like(window).abs()
        * point_values.reindex_like(window)
    ).sum(axis=1)
    ratio = notional / daily.loc[window.index, "prior_nav_usd"]
    years = max((window.index[-1] - window.index[0]).days / 365.2425, 1e-9)
    return float(ratio.sum() / years)


def comparison_row(
    result: Any,
    *,
    benchmark: str,
    family: str,
    citation: str,
    construction: str,
    leverage_cap_applied: Any,
    incumbent_daily: pd.Series | None = None,
    valuation_prices: pd.DataFrame | None = None,
    point_values: pd.DataFrame | None = None,
    start: str = RETROSPECTIVE_START,
    end: str = RETROSPECTIVE_END,
    annualization: int = 252,
) -> dict[str, Any]:
    """One descriptive row.  No rank, no score, no recommendation column."""

    metrics = performance_metrics(result, start, end, annualization)
    volatility = float(metrics["Annualized volatility"])
    drawdown = float(metrics["Max drawdown"])
    daily = result.daily.loc[start:end, "net_return"].dropna()
    correlation = np.nan
    if incumbent_daily is not None:
        joined = pd.DataFrame(
            {"incumbent": incumbent_daily, "benchmark": daily}
        ).dropna(how="any")
        if len(joined) > 2:
            correlation = float(
                np.corrcoef(joined["incumbent"], joined["benchmark"])[0, 1]
            )
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "family": family,
        "citation": citation,
        "construction": construction,
        "leverage_cap_applied": leverage_cap_applied,
        "validation_status": VALIDATION_STATUS,
        "run_status": RUN_COMPLETED,
        "run_detail": "",
        "start": metrics["Start"],
        "end": metrics["End"],
        "years": float(metrics["Years"]),
        "cagr": float(metrics["CAGR"]),
        "annualized_volatility": volatility,
        "sharpe": float(metrics["Naive daily Sharpe (sqrt252, rf=0)"]),
        "hac_sharpe": float(metrics["HAC Sharpe (21 lags, rf=0)"]),
        "sortino": float(metrics["Sortino (rf=0)"]),
        "calmar": float(metrics["Calmar"]),
        "max_drawdown": drawdown,
        "max_drawdown_over_volatility": (
            abs(drawdown) / volatility if volatility > 0 else np.nan
        ),
        "annual_cost_drag": float(metrics["Annual cost drag"]),
        "annual_rebalance_notional_turnover_over_nav": _notional_turnover(
            result, valuation_prices, point_values, start, end
        ),
        "average_gross_notional_multiple": float(
            metrics["Average gross notional multiple"]
        ),
        "peak_gross_notional_multiple": float(metrics["Peak gross notional multiple"]),
        "average_markets_held": float(metrics["Average markets held"]),
        "return_correlation_with_incumbent": correlation,
    }
    row.update(_turnover(result.daily, start, end))
    return row


def errored_row(
    *,
    benchmark: str,
    family: str,
    citation: str,
    construction: str,
    leverage_cap_applied: Any,
    detail: str,
) -> dict[str, Any]:
    """A benchmark the engine refused, recorded rather than dropped."""

    row: dict[str, Any] = dict.fromkeys(BENCHMARK_COMPARISON_COLUMNS, np.nan)
    row.update(
        {
            "benchmark": benchmark,
            "family": family,
            "citation": citation,
            "construction": construction,
            "leverage_cap_applied": leverage_cap_applied,
            "validation_status": VALIDATION_STATUS,
            "run_status": RUN_ERRORED,
            "run_detail": detail[:240],
            "start": "",
            "end": "",
        }
    )
    return row


def comparison_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Assemble rows in declaration order.  Never sorted by performance."""

    return pd.DataFrame(list(rows), columns=BENCHMARK_COMPARISON_COLUMNS)


# ---------------------------------------------------------------------------
# The declared benchmark family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSpecification:
    """One declared benchmark: its key, its paper, and how it is built here."""

    key: str
    family: str
    citation: str
    construction: str
    leverage_cap_applied: str
    in_spanning_family: bool


BENCHMARK_FAMILY: tuple[BenchmarkSpecification, ...] = (
    BenchmarkSpecification(
        key="mop_tsmom",
        family="time-series momentum",
        citation=(
            "Moskowitz, Ooi & Pedersen (2012), Journal of Financial Economics "
            "104(2): 228-250, eq. (1) and eq. (5)"
        ),
        construction=(
            "12-month sign, no skip month; 40% per-instrument ex-ante volatility "
            "with a 60-day centre of mass and 261-day annualisation, lagged one "
            "session; 1/S_t diversification; no portfolio volatility control"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=True,
    ),
    BenchmarkSpecification(
        key="mop_tsmom_under_incumbent_leverage_cap",
        family="time-series momentum",
        citation="Moskowitz, Ooi & Pedersen (2012), as above",
        construction=(
            "identical to mop_tsmom, but subject to the incumbent's "
            "max_gross_notional_multiple = 5.0; a capped MOP is not MOP, so the "
            "two rows are reported separately"
        ),
        leverage_cap_applied="5.0x gross notional (incumbent's own limit)",
        in_spanning_family=False,
    ),
    BenchmarkSpecification(
        key="hop_trend_blend",
        family="multi-horizon trend",
        citation=(
            "Hurst, Ooi & Pedersen (2017), Journal of Portfolio Management "
            "44(1): 15-29, incl. footnote 5"
        ),
        construction=(
            "equal-weight sign blend over 1/3/12-month horizons; per-market "
            "inverse-volatility sizing; portfolio scaled to a 10% ex-ante "
            "volatility budget using a 36-month equally weighted covariance"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=True,
    ),
    BenchmarkSpecification(
        key="baltas_kosowski_tstatistic",
        family="alternative trend estimator",
        citation=(
            "Baltas & Kosowski (2013/2020), SSRN 2140091; transform as described "
            "in Lim, Zohren & Roberts (2019), arXiv:1904.04912 sec. II-A"
        ),
        construction=(
            "OLS t-statistic of a 252-session time trend fitted to the cumulative "
            "log excess return, clipped to [-1, 1]; MOP sizing unchanged so the "
            "estimator is the only difference"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=True,
    ),
    BenchmarkSpecification(
        key="baz_macd_ewmac",
        family="alternative trend estimator",
        citation=(
            "Baz, Granger, Harvey, Le Roux & Rattray (2015), SSRN 2695101; "
            "equations as reproduced in Lim, Zohren & Roberts (2019), eqs (4)-(8)"
        ),
        construction=(
            "volatility-normalised MACD over (8,24), (16,48), (32,96) with "
            "alpha = 1/n, response function phi(y) = y exp(-y^2/4)/0.89, averaged "
            "across the three scales; MOP sizing unchanged"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=True,
    ),
    BenchmarkSpecification(
        key="long_only_equal_risk",
        family="no-timing control",
        citation=(
            "no single paper; the zero-timing control implied by Moskowitz, Ooi & "
            "Pedersen (2012) sec. 4.1's always-long comparison"
        ),
        construction=(
            "X = +1 everywhere; MOP inverse-volatility sizing at a 40% "
            "per-instrument budget. A passive long-only equal-risk futures panel, "
            "not a market portfolio and not a CAPM beta reference"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=True,
    ),
    BenchmarkSpecification(
        key="long_only_equal_notional",
        family="no-timing control",
        citation="no single paper; equal-notional companion to the above",
        construction=(
            "X = +1 everywhere with w = 1/S_t and no volatility scaling. At this "
            "capital base the 1/S_t book is not delivered: about half its "
            "intended markets round to zero under integer contracts and the "
            "realised gross notional is roughly 0.60x against an intended 1.0x, so the gap "
            "against long_only_equal_risk is NOT a clean read of the weighting "
            "rule - it also carries a universe and an investment-level "
            "difference. Read average_markets_held and "
            "average_gross_notional_multiple on both rows before attributing it"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=False,
    ),
    BenchmarkSpecification(
        key="long_only_equal_risk_volatility_matched",
        family="no-timing control",
        citation="no single paper; Harvey et al. (2018) matched-volatility protocol",
        construction=(
            "long_only_equal_risk with the per-instrument budget bisected so the "
            "realised 1990-2014 volatility matches the incumbent's. The constant "
            "is an EX-POST full-sample calibration and is not causal"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=False,
    ),
    BenchmarkSpecification(
        key="barroso_santa_clara_on_mop_tsmom",
        family="volatility-management overlay",
        citation="Barroso & Santa-Clara (2015), Journal of Financial Economics 116(1): 111-120",
        construction=(
            "uncapped 12%-annualised constant-volatility weight from the "
            "underlying's own trailing 126 daily returns, not demeaned, applied "
            "to the monthly decision so it pays real transaction costs"
        ),
        leverage_cap_applied="none (max_gross_notional_multiple released)",
        in_spanning_family=False,
    ),
    BenchmarkSpecification(
        key="barroso_santa_clara_on_incumbent",
        family="volatility-management overlay",
        citation="Barroso & Santa-Clara (2015), as above",
        construction=(
            "the same overlay on the incumbent's own decision frame. This is a "
            "SECOND, SLOWER volatility-target layer on top of the incumbent's "
            "EWMA risk scalar, not Barroso-Santa-Clara in the paper's sense, "
            "which applies it to an unscaled portfolio"
        ),
        leverage_cap_applied="incumbent default (5.0x gross notional)",
        in_spanning_family=False,
    ),
    BenchmarkSpecification(
        key="moreira_muir_expanding_constant_on_incumbent",
        family="volatility-management overlay",
        citation="Moreira & Muir (2017), Journal of Finance 72(4): 1611-1644",
        construction=(
            "inverse previous-month realised variance, demeaned within the month, "
            "with the scaling constant recomputed on an expanding window through "
            "the decision month only. The implementable version"
        ),
        leverage_cap_applied="incumbent default (5.0x gross notional)",
        in_spanning_family=False,
    ),
    BenchmarkSpecification(
        key="moreira_muir_full_sample_constant_on_incumbent",
        family="volatility-management overlay",
        citation="Moreira & Muir (2017), as above",
        construction=(
            "the paper's own definition, with the scaling constant fitted to the "
            "FULL SAMPLE. NON-CAUSAL: it sets today's leverage using the whole "
            "future's volatility and exists only so the size of that look-ahead "
            "can be measured against the expanding-window row"
        ),
        leverage_cap_applied="incumbent default (5.0x gross notional)",
        in_spanning_family=False,
    ),
)


def spanning_family_keys() -> tuple[str, ...]:
    """Keys declared for the joint spanning regression, fixed before it runs."""

    return tuple(
        specification.key
        for specification in BENCHMARK_FAMILY
        if specification.in_spanning_family
    )


def signal_diagnostic_row(
    signal: pd.DataFrame,
    weights: pd.DataFrame,
    decisions: pd.DatetimeIndex,
    *,
    name: str,
    citation: str,
    reference_signal: pd.DataFrame | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Exposure and agreement diagnostics for one signal, at decision dates.

    ``mean_absolute_exposure`` is the check that catches an inert transform: a
    clipped statistic that never leaves its bounds is a sign rule wearing a
    different name, and the number says so immediately.
    """

    at_decisions = signal.reindex(decisions)
    live = at_decisions.notna() & weights.reindex(decisions).ne(0.0)
    values = at_decisions.where(live).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    agreement = np.nan
    if reference_signal is not None:
        reference = reference_signal.reindex(decisions).where(live)
        both = np.isfinite(values) & np.isfinite(reference.to_numpy(dtype=float))
        if both.any():
            agreement = float(
                (
                    np.sign(values[both])
                    == np.sign(reference.to_numpy(dtype=float)[both])
                ).mean()
            )
    gross = weights.reindex(decisions).abs().sum(axis=1)
    return {
        "signal": name,
        "citation": citation,
        "decisions": int(len(decisions)),
        "average_live_markets": float(live.sum(axis=1).mean()),
        "mean_absolute_exposure": float(np.abs(finite).mean()) if finite.size else np.nan,
        "maximum_absolute_exposure": float(np.abs(finite).max()) if finite.size else np.nan,
        "sign_agreement_with_mop": agreement,
        "average_gross_notional_multiple": float(gross.mean()),
        "maximum_gross_notional_multiple": float(gross.max()),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Decision-frame builders
# ---------------------------------------------------------------------------


def signal_decision_frame(
    signal: pd.DataFrame,
    inputs: EngineInputs,
    config: BenchmarkConfig = BenchmarkConfig(),
    *,
    volatility_budget: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MOP-sized monthly decisions for any signal frame.

    Returns ``(monthly contracts per dollar of NAV, daily notional weights)``.
    Every trend benchmark other than the HOP blend shares this sizing exactly,
    which is what makes a difference between those rows attributable to the
    estimator and to nothing else.
    """

    volatility = ex_ante_volatility(inputs.excess_returns, config)
    budget = (
        config.instrument_volatility_budget
        if volatility_budget is None
        else float(volatility_budget)
    )
    weights = inverse_volatility_weights(
        signal, volatility, inputs.eligible, volatility_budget=budget
    )
    contracts = notional_weights_to_contracts_per_dollar(
        weights, inputs.valuation_prices, inputs.point_values
    )
    return _month_end_rows(contracts), weights


def long_only_signal(inputs: EngineInputs) -> pd.DataFrame:
    """``X = +1`` everywhere.  Not a market portfolio and not a CAPM beta.

    On a 59-market panel of bonds, equity indices, FX and commodities, "long
    everything" is a specific and arguable bet: long bonds over 1990-2014 was a
    spectacular trade and long commodities was not.  Any alpha computed against
    this row inherits that arbitrariness.

    Universe membership is driven by the trailing volume gate and by the
    volatility estimator's own 60-observation warm-up, never by a fixed
    59-market list and never by the catalogue's ``last_quoted_date``.  A market
    that has not started trading fails the gate, and one that has stopped fails
    it again; screening on ``last_quoted_date`` would be worse than a fixed
    universe, because knowing when a contract will stop quoting is future
    information at the decision date.

    No cash leg is added.  A fully collateralised futures position earns the
    excess return, which is what this ledger measures; adding a Treasury-bill
    return here would inflate this row by the entire 1990-2014 bill path while
    leaving the trend rows alone.
    """

    return pd.DataFrame(
        1.0, index=inputs.prices.index, columns=inputs.prices.columns
    )


def equal_notional_decision_frame(
    inputs: EngineInputs,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-only with ``w = 1/S_t`` and no volatility scaling.

    ``S_t`` is deliberately the same live set the equal-risk row uses, including
    its requirement of an estimable lagged volatility, so the *intended*
    universe is identical and only the weighting rule differs.

    The *delivered* universe is not, and the row must not be read as though it
    were.  ``w = 1/S_t`` with about fifty live markets asks for roughly 2% of
    NAV per market; at a one-million-dollar NAV that is an order for a fraction
    of a contract in half the panel, and ``integer_contracts`` rounds those to
    zero.  Measured on this panel over 1990-2014 both rows intend 48.6 markets
    per decision; the equal-notional book loses 24.4 of them to rounding and
    the equal-risk book loses 0.5, and the equal-notional row's realised gross
    notional multiple is 0.60 against an intended 1.0.  So the
    difference between the two rows is the weighting rule *plus* a granularity
    artefact of comparable size, and the artefact is a property of the capital
    base rather than of either rule.  ``average_markets_held`` and
    ``average_gross_notional_multiple`` are on the comparison row so a reader
    can see it without rerunning anything.
    """

    signal = long_only_signal(inputs)
    volatility = ex_ante_volatility(inputs.excess_returns, config)
    live = _live_mask(signal, volatility, inputs.eligible)
    count = live.sum(axis=1).replace(0, np.nan)
    weights = signal.where(live).div(count, axis=0).where(live, 0.0)
    contracts = notional_weights_to_contracts_per_dollar(
        weights, inputs.valuation_prices, inputs.point_values
    )
    return _month_end_rows(contracts), weights


def hop_decision_frame(
    inputs: EngineInputs,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly decisions for the HOP blend under its own 10% portfolio budget.

    Returns ``(monthly contracts, decision-date notional weights, scaling
    diagnostics)``.  Unlike every other row here this benchmark has a genuine
    portfolio-level volatility control, which is why it is the right foil for
    the incumbent's own volatility targeting while MOP is the right foil for
    having no portfolio overlay at all.
    """

    signal = hop_blend_signal(inputs.excess_returns, config)
    volatility = ex_ante_volatility(inputs.excess_returns, config)
    monthly = monthly_excess_returns(inputs.excess_returns)
    decisions = pd.DatetimeIndex(_month_end_rows(inputs.prices).index)
    weights, diagnostics = hop_portfolio_weights(
        signal, volatility, inputs.eligible, monthly, decisions, config
    )
    contracts = notional_weights_to_contracts_per_dollar(
        weights,
        inputs.valuation_prices.reindex(weights.index),
        inputs.point_values.reindex(weights.index),
    )
    return contracts, weights, diagnostics


def apply_overlay_to_decisions(
    monthly_contracts_per_dollar: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """Multiply a decision frame by an overlay weight, decision by decision.

    Applied to the *positions*, never to the finished return series.  A
    post-hoc multiplier on daily net returns gets the leverage for free: it
    skips transaction costs, integer rounding, the participation cap and roll
    costs, all of which the overlay's extra turnover would really pay.

    A decision with no estimable weight is left unmanaged at 1.0 rather than
    dropped, so the underlying's own history is not truncated by the overlay's
    warm-up.
    """

    aligned = weights.reindex(monthly_contracts_per_dollar.index).astype(float)
    aligned = aligned.where(np.isfinite(aligned), 1.0)
    return monthly_contracts_per_dollar.mul(aligned, axis=0)


def release_leverage_cap(config: StrategyConfig) -> StrategyConfig:
    """The incumbent's config with the gross-notional cap removed.

    A faithful MOP replication reaches a gross notional multiple above eight on
    this panel, so the incumbent's 5.0x limit binds on it.  Silently leaving the
    cap on would report a capped MOP as MOP; silently removing it would hide
    that the incumbent operates under a constraint the benchmark does not.  Both
    rows are produced, and this helper makes which is which explicit at the call
    site.
    """

    return replace(config, max_gross_notional_multiple=None)


def _specification(key: str) -> BenchmarkSpecification:
    for specification in BENCHMARK_FAMILY:
        if specification.key == key:
            return specification
    raise KeyError(f"{key!r} is not a declared benchmark")


def run_benchmark_suite(
    config: StrategyConfig,
    frames: Mapping[str, pd.DataFrame],
    *,
    incumbent_result: Any,
    benchmark_config: BenchmarkConfig = BenchmarkConfig(),
    start: str = RETROSPECTIVE_START,
    end: str = RETROSPECTIVE_END,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    include_permutation_null: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Replicate the declared family and compare it with the incumbent.

    The family is declared in ``BENCHMARK_FAMILY`` above this function, before
    any of it runs, and the spanning regression takes its regressors from that
    declaration.  Choosing which benchmarks to include after seeing the residual
    would make the intercept meaningless.

    A benchmark the engine refuses — a permuted or heavily levered book can
    breach the roll-backlog limit or drive NAV non-positive — is recorded as an
    errored row rather than dropped, so a missing benchmark is visible in the
    artifact instead of only in a traceback.
    """

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    inputs = prepare_engine_inputs(config, frames)
    announce("verifying the execution seam against the incumbent's own ledger")
    seam_deviation = assert_seam_reproduces_incumbent(incumbent_result, inputs, config)
    uncapped = release_leverage_cap(config)
    annualization = int(getattr(config, "annualization", 252))

    incumbent_daily = incumbent_result.daily.loc[start:end, "net_return"].dropna()
    rows: list[dict[str, Any]] = [
        comparison_row(
            incumbent_result,
            benchmark="incumbent",
            family="incumbent",
            citation="this repository's frozen v3.2.1 specification",
            construction=(
                "252-session sign trend blended with basis momentum at equal risk "
                "weight, volatility-shock taper, EWMA portfolio risk scalar "
                "clipped to [0.25, 2.00], 0.25 no-trade buffer"
            ),
            leverage_cap_applied="incumbent default (5.0x gross notional)",
            incumbent_daily=incumbent_daily,
            valuation_prices=inputs.valuation_prices,
            point_values=inputs.point_values,
            start=start,
            end=end,
            annualization=annualization,
        )
    ]
    daily_paths: dict[str, pd.Series] = {"incumbent": incumbent_daily}
    signal_rows: list[dict[str, Any]] = []
    overlay_rows: list[pd.DataFrame] = []

    def record(
        key: str,
        decision_frame: pd.DataFrame,
        run_config: StrategyConfig,
        *,
        risk_scalar: pd.Series | None = None,
    ) -> BacktestResult | None:
        specification = _specification(key)
        announce(f"simulating {key}")
        try:
            result = simulate_monthly_targets(
                decision_frame,
                inputs,
                run_config,
                name=key,
                risk_scalar=risk_scalar,
            )
        except Exception as error:
            rows.append(
                errored_row(
                    benchmark=key,
                    family=specification.family,
                    citation=specification.citation,
                    construction=specification.construction,
                    leverage_cap_applied=specification.leverage_cap_applied,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            return None
        rows.append(
            comparison_row(
                result,
                benchmark=key,
                family=specification.family,
                citation=specification.citation,
                construction=specification.construction,
                leverage_cap_applied=specification.leverage_cap_applied,
                incumbent_daily=incumbent_daily,
                valuation_prices=inputs.valuation_prices,
                point_values=inputs.point_values,
                start=start,
                end=end,
                annualization=annualization,
            )
        )
        daily_paths[key] = result.daily.loc[start:end, "net_return"].dropna()
        return result

    # --- trend rules, all sized identically ------------------------------
    mop_signal = mop_tsmom_signal(inputs.excess_returns, benchmark_config)
    mop_frame, mop_weights = signal_decision_frame(mop_signal, inputs, benchmark_config)
    decisions = pd.DatetimeIndex(mop_frame.index)
    mop_result = record("mop_tsmom", mop_frame, uncapped)
    record("mop_tsmom_under_incumbent_leverage_cap", mop_frame, config)
    signal_rows.append(
        signal_diagnostic_row(
            mop_signal,
            mop_weights,
            decisions,
            name="mop_tsmom",
            citation="Moskowitz, Ooi & Pedersen (2012) eq. (5)",
            note="sign rule; exposure is +/-1 by construction",
        )
    )

    hop_frame, hop_weights, hop_diagnostics = hop_decision_frame(inputs, benchmark_config)
    record("hop_trend_blend", hop_frame, uncapped)
    signal_rows.append(
        signal_diagnostic_row(
            hop_blend_signal(inputs.excess_returns, benchmark_config),
            hop_weights.reindex(columns=inputs.prices.columns).fillna(0.0),
            pd.DatetimeIndex(hop_weights.index),
            name="hop_trend_blend",
            citation="Hurst, Ooi & Pedersen (2017)",
            reference_signal=mop_signal,
            note="blend of three sign sleeves; exposure lies in {+/-1/3, +/-1}",
        )
    )

    bk_signal = baltas_kosowski_signal(inputs.excess_returns, benchmark_config)
    bk_frame, bk_weights = signal_decision_frame(bk_signal, inputs, benchmark_config)
    record("baltas_kosowski_tstatistic", bk_frame, uncapped)
    signal_rows.append(
        signal_diagnostic_row(
            bk_signal,
            bk_weights,
            decisions,
            name="baltas_kosowski_tstatistic",
            citation="Baltas & Kosowski, transform per Lim, Zohren & Roberts (2019)",
            reference_signal=mop_signal,
            note=(
                "clip(t, -1, 1); mean absolute exposure near one means the clip is "
                "close to inert on this panel"
            ),
        )
    )

    macd_signal = baz_macd_signal(inputs.prices, benchmark_config)
    macd_frame, macd_weights = signal_decision_frame(macd_signal, inputs, benchmark_config)
    record("baz_macd_ewmac", macd_frame, uncapped)
    signal_rows.append(
        signal_diagnostic_row(
            macd_signal,
            macd_weights,
            decisions,
            name="baz_macd_ewmac",
            citation="Baz et al. (2015), eqs (4)-(8) as reproduced by Lim et al. (2019)",
            reference_signal=mop_signal,
            note="phi de-weights extreme trends, so exposure is bounded by 0.9638",
        )
    )

    # --- no-timing controls ----------------------------------------------
    passive_signal = long_only_signal(inputs)
    passive_frame, passive_weights = signal_decision_frame(
        passive_signal, inputs, benchmark_config
    )
    record("long_only_equal_risk", passive_frame, uncapped)
    signal_rows.append(
        signal_diagnostic_row(
            passive_signal,
            passive_weights,
            decisions,
            name="long_only_equal_risk",
            citation="no single paper; the zero-timing control",
            note="X is literally +1 everywhere; no timing information enters",
        )
    )
    notional_frame, _ = equal_notional_decision_frame(inputs, benchmark_config)
    record("long_only_equal_notional", notional_frame, uncapped)

    announce("bisecting the volatility-matched long-only budget")
    incumbent_volatility = float(
        incumbent_daily.std() * math.sqrt(annualization)
    )
    try:
        matched_budget, matched_result = solve_volatility_matched_budget(
            lambda budget: signal_decision_frame(
                passive_signal, inputs, benchmark_config, volatility_budget=budget
            )[0],
            inputs,
            uncapped,
            volatility=incumbent_volatility,
            start=start,
            end=end,
        )
        specification = _specification("long_only_equal_risk_volatility_matched")
        rows.append(
            comparison_row(
                matched_result,
                benchmark=specification.key,
                family=specification.family,
                citation=specification.citation,
                construction=(
                    f"{specification.construction}; solved per-instrument budget "
                    f"{matched_budget:.4f}"
                ),
                leverage_cap_applied=specification.leverage_cap_applied,
                incumbent_daily=incumbent_daily,
                valuation_prices=inputs.valuation_prices,
                point_values=inputs.point_values,
                start=start,
                end=end,
                annualization=annualization,
            )
        )
        daily_paths[specification.key] = matched_result.daily.loc[
            start:end, "net_return"
        ].dropna()
    except Exception as error:
        specification = _specification("long_only_equal_risk_volatility_matched")
        rows.append(
            errored_row(
                benchmark=specification.key,
                family=specification.family,
                citation=specification.citation,
                construction=specification.construction,
                leverage_cap_applied=specification.leverage_cap_applied,
                detail=f"{type(error).__name__}: {error}",
            )
        )

    # --- volatility-management overlays -----------------------------------
    launch = config.launch_date if config.launch_date is not None else start
    if mop_result is not None:
        mop_returns = mop_result.daily.loc[launch:, "net_return"].dropna()
        mop_overlay_weights = barroso_santa_clara_weights(
            mop_returns, decisions, benchmark_config
        )
        record(
            "barroso_santa_clara_on_mop_tsmom",
            apply_overlay_to_decisions(mop_frame, mop_overlay_weights),
            uncapped,
            risk_scalar=mop_overlay_weights,
        )
        overlay_rows.append(
            overlay_weight_report(
                mop_overlay_weights,
                overlay="barroso_santa_clara_constant_volatility",
                underlying="mop_tsmom",
                causality="causal (trailing 126 sessions through the decision date)",
                maximum_risk_scalar_reference=float(config.max_risk_scalar),
            )
        )

    incumbent_decisions = pd.DatetimeIndex(incumbent_result.target_positions.index)
    incumbent_returns = incumbent_result.daily.loc[launch:, "net_return"].dropna()
    incumbent_bsc = barroso_santa_clara_weights(
        incumbent_returns, incumbent_decisions, benchmark_config
    )
    record(
        "barroso_santa_clara_on_incumbent",
        apply_overlay_to_decisions(incumbent_result.target_positions, incumbent_bsc),
        config,
        risk_scalar=incumbent_bsc,
    )
    overlay_rows.append(
        overlay_weight_report(
            incumbent_bsc,
            overlay="barroso_santa_clara_constant_volatility",
            underlying="incumbent",
            causality="causal (trailing 126 sessions through the decision date)",
            maximum_risk_scalar_reference=float(config.max_risk_scalar),
        )
    )

    for key, expanding, causality in (
        (
            "moreira_muir_expanding_constant_on_incumbent",
            True,
            "causal (scaling constant recomputed on an expanding window)",
        ),
        (
            "moreira_muir_full_sample_constant_on_incumbent",
            False,
            "NON-CAUSAL (scaling constant fitted to the full sample)",
        ),
    ):
        try:
            weights = moreira_muir_weights(
                incumbent_returns,
                incumbent_decisions,
                benchmark_config,
                expanding_constant=expanding,
            )
        except Exception as error:
            specification = _specification(key)
            rows.append(
                errored_row(
                    benchmark=key,
                    family=specification.family,
                    citation=specification.citation,
                    construction=specification.construction,
                    leverage_cap_applied=specification.leverage_cap_applied,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            continue
        record(
            key,
            apply_overlay_to_decisions(incumbent_result.target_positions, weights),
            config,
            risk_scalar=weights,
        )
        overlay_rows.append(
            overlay_weight_report(
                weights,
                overlay=(
                    "moreira_muir_inverse_variance"
                    + ("_expanding_constant" if expanding else "_full_sample_constant")
                ),
                underlying="incumbent",
                causality=causality,
                maximum_risk_scalar_reference=float(config.max_risk_scalar),
            )
        )

    # --- regressions -------------------------------------------------------
    announce("regressing the incumbent on each benchmark")
    explanatory = {
        name: path for name, path in daily_paths.items() if name != "incumbent"
    }
    alpha = (
        benchmark_alpha_report(incumbent_daily, explanatory, annualization=annualization)
        if explanatory
        else pd.DataFrame(columns=ALPHA_REPORT_COLUMNS)
    )
    family_keys = [key for key in spanning_family_keys() if key in daily_paths]
    if family_keys:
        spanning, spanning_coefficients = spanning_report(
            incumbent_daily,
            {key: daily_paths[key] for key in family_keys},
            annualization=annualization,
        )
    else:  # pragma: no cover - only reachable if every trend row errored
        spanning = pd.DataFrame(columns=SPANNING_REPORT_COLUMNS)
        spanning_coefficients = pd.DataFrame(columns=SPANNING_COEFFICIENT_COLUMNS)

    # --- the no-skill null --------------------------------------------------
    null_summaries: list[pd.DataFrame] = []
    null_draws: list[pd.DataFrame] = []
    if include_permutation_null:
        incumbent_turnover = _turnover(incumbent_result.daily, start, end)
        for mode in (BLOCK_SIGN_FLIP, IID_SIGN_FLIP):
            announce(f"running {permutations} {mode} permutations")
            draws, summary = sign_permutation_null(
                incumbent_result.target_positions,
                inputs,
                config,
                incumbent_daily=incumbent_result.daily["net_return"],
                mode=mode,
                permutations=permutations,
                seed=seed,
                start=start,
                end=end,
            )
            summary["incumbent_annual_cost_drag"] = float(
                incumbent_result.daily.loc[start:end, "cost"].mean() * annualization
            )
            summary["incumbent_annual_turnover"] = incumbent_turnover[
                "annual_total_turnover_contracts"
            ]
            draws = draws.assign(null=f"{mode}_on_incumbent_decisions")
            null_summaries.append(summary)
            null_draws.append(draws)

    daily_frame = pd.DataFrame(daily_paths)
    monthly_frame = (1.0 + daily_frame).resample("ME").prod() - 1.0
    return {
        "comparison": comparison_frame(rows),
        "signal_diagnostics": pd.DataFrame(
            signal_rows, columns=SIGNAL_DIAGNOSTIC_COLUMNS
        ),
        "alpha": alpha,
        "spanning": spanning,
        "spanning_coefficients": spanning_coefficients,
        "overlay_weights": (
            pd.concat(overlay_rows, ignore_index=True)
            if overlay_rows
            else pd.DataFrame(columns=OVERLAY_WEIGHT_COLUMNS)
        ),
        "hop_scaling_diagnostics": hop_diagnostics,
        "permutation_null": (
            pd.concat(null_summaries, ignore_index=True)
            if null_summaries
            else pd.DataFrame(columns=PERMUTATION_NULL_COLUMNS)
        ),
        "permutation_draws": (
            pd.concat(null_draws, ignore_index=True)
            if null_draws
            else pd.DataFrame()
        ),
        "seam_check": pd.DataFrame(
            [
                {
                    "check": "incumbent_decision_frame_replayed_through_the_seam",
                    "statistic": "maximum absolute daily net-return deviation",
                    "observed": seam_deviation,
                    "required": 0.0,
                    "method": (
                        "the incumbent's own monthly decision frame fed back "
                        "through simulate_monthly_targets' execution call"
                    ),
                }
            ]
        ),
        "daily_net_returns": daily_frame,
        "monthly_net_returns": monthly_frame,
    }


__all__ = [
    "ALPHA_REPORT_COLUMNS",
    "BENCHMARK_COMPARISON_COLUMNS",
    "BENCHMARK_FAMILY",
    "BLOCK_SIGN_FLIP",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_PERMUTATION_SEED",
    "IID_SIGN_FLIP",
    "OVERLAY_WEIGHT_COLUMNS",
    "PERMUTATION_NULL_COLUMNS",
    "RETROSPECTIVE_END",
    "RETROSPECTIVE_START",
    "RUN_COMPLETED",
    "RUN_ERRORED",
    "SIGNAL_DIAGNOSTIC_COLUMNS",
    "SPANNING_COEFFICIENT_COLUMNS",
    "SPANNING_REPORT_COLUMNS",
    "VALIDATION_STATUS",
    "BenchmarkConfig",
    "BenchmarkSpecification",
    "EngineInputs",
    "apply_overlay_to_decisions",
    "assert_seam_reproduces_incumbent",
    "baltas_kosowski_signal",
    "barroso_santa_clara_weights",
    "baz_macd_signal",
    "benchmark_alpha_report",
    "comparison_frame",
    "comparison_row",
    "equal_notional_decision_frame",
    "errored_row",
    "ex_ante_volatility",
    "excess_returns",
    "hop_blend_signal",
    "hop_decision_frame",
    "hop_portfolio_weights",
    "inverse_volatility_weights",
    "long_only_signal",
    "monthly_excess_returns",
    "mop_tsmom_signal",
    "moreira_muir_weights",
    "notional_weights_to_contracts_per_dollar",
    "overlay_weight_report",
    "prepare_engine_inputs",
    "release_leverage_cap",
    "run_benchmark_suite",
    "sign_permutation_null",
    "signal_decision_frame",
    "signal_diagnostic_row",
    "simulate_monthly_targets",
    "solve_volatility_matched_budget",
    "spanning_family_keys",
    "spanning_report",
    "time_trend_tstatistic",
]
