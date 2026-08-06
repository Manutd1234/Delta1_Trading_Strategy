"""Why a drawdown happened, measured before anything is changed to prevent it.

This module answers a question that must be settled *before* a risk lever is
swept: is the proposed control an active constraint at all, and is the loss it
claims to prevent the kind of loss the strategy actually suffers?

The motivating case is the position-magnitude family --- ``max_risk_scalar``,
``min_risk_scalar``, ``signal_cap``, ``shock_floor`` and
``max_gross_notional_multiple``.  Tightening any of them is a plausible-sounding
way to reduce drawdown, and a sweep will dutifully report that it does, because
a smaller book loses less.  That report is uninformative twice over: a bound
that never binds changes nothing at all, and a bound that binds only by shrinking
exposure buys drawdown reduction at a proportional cost in return, which any
uniform de-lever would also buy.

So this module measures four things, in the order that makes the sweep
interpretable:

``bound_activity_report``
    Does the bound ever bind?  A bound outside the realized range of the
    quantity it clips is inert, and no configuration of it can matter.

``drawdown_anatomy``
    Is the loss a *size* failure or an *accuracy* failure?  If realized
    volatility inside drawdowns matches the unconditional level while the hit
    rate collapses, the book was not too big --- the forecast was wrong, and a
    magnitude ceiling addresses a failure mode the strategy does not have.

``episode_attribution``
    Is the loss *concentrated*?  A per-market ceiling can only help if a small
    number of markets produce the loss.  Breadth of participation in a drawdown
    is reported directly rather than assumed.

``uniform_rescale_frontier``
    What does pure de-levering already buy?  Any magnitude lever must beat this
    line to be a risk improvement rather than a smaller allocation, and the
    ratio of maximum drawdown to annualized volatility is the invariant that
    exposes the difference.

None of these functions ranks a configuration, and none of them promotes one.
They establish whether a lever *can* work, which is a different and prior
question to whether its measured effect is favorable.  A negative answer here
is the useful answer: it stops a sweep from reporting a real number about an
inert parameter.

Every statistic is computed from realized ledger output.  Nothing here is
causal evidence about the future, and an attribution measured on one realized
path carries the sampling error of that path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_RESCALE_MULTIPLIERS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)

# The share of a drawdown's losses carried by its three worst markets.  Three
# is a reporting choice, not a threshold: it is small enough that a genuinely
# concentrated loss would be visible in it and large enough not to be a single
# outlier.
CONCENTRATION_RANK = 3

INERT = "inert"
ACTIVE = "active"

_DAYS_PER_YEAR = 365.2425


def _elapsed_years(
    index: pd.Index,
    *,
    period_start: str | pd.Timestamp | None = None,
) -> float:
    """Canonical CAGR denominator used by ``strategy.performance_metrics``."""

    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("CAGR requires a non-empty DatetimeIndex")
    start = index[0] if period_start is None else pd.Timestamp(period_start)
    if start > index[0]:
        raise ValueError("period_start cannot follow the first return")
    elapsed_seconds = (index[-1] - start).total_seconds()
    return max(elapsed_seconds / (_DAYS_PER_YEAR * 24 * 60 * 60), 1 / _DAYS_PER_YEAR)


@dataclass(frozen=True)
class BoundSpec:
    """One configuration bound and the realized series it is supposed to clip.

    ``side`` names which end of ``series`` the bound sits on, because a ceiling
    and a floor are inert for opposite reasons and the report must not conflate
    them.
    """

    name: str
    limit: float
    side: str
    series: pd.Series
    unit: str = "sessions"
    tolerance: float = 1e-9
    binding_indicator: pd.Series | None = None
    binding_criterion: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.side not in {"upper", "lower"}:
            raise ValueError("side must be 'upper' or 'lower'")
        if not isinstance(self.limit, (int, float)) or isinstance(self.limit, bool):
            raise ValueError("limit must be a real number")
        if not np.isfinite(float(self.limit)):
            raise ValueError("limit must be finite")
        if not isinstance(self.series, pd.Series):
            raise ValueError("series must be a pandas Series")
        if self.tolerance < 0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and non-negative")
        if self.binding_indicator is not None and not isinstance(
            self.binding_indicator, pd.Series
        ):
            raise ValueError("binding_indicator must be a pandas Series")
        if self.binding_criterion is not None and (
            self.binding_indicator is None
            or not isinstance(self.binding_criterion, str)
            or not self.binding_criterion
        ):
            raise ValueError(
                "binding_criterion requires a binding indicator and non-empty text"
            )


def bound_activity_report(specs: Sequence[BoundSpec]) -> pd.DataFrame:
    """Whether each bound is an active constraint on the realized path.

    ``Binding share`` is normally the fraction of observations at the limit. An
    explicit intervention indicator takes precedence when the engine records
    the actual clipping decision separately from the realized post-trade value.
    ``Headroom`` is the signed distance from the limit to the most extreme
    realized value, so a positive headroom on an upper bound means the series
    never reached it.

    A bound reported ``inert`` cannot change any result at its current value.
    Sweeping it produces identical runs until it crosses into the realized
    range, and reporting those identical runs as a null effect would be a
    statement about the sweep rather than about the strategy.
    """

    if not specs:
        raise ValueError("at least one bound specification is required")

    rows: list[dict[str, object]] = []
    for spec in specs:
        values = spec.series.dropna().astype(float)
        if values.empty:
            raise ValueError(f"bound {spec.name!r} has no observations")
        limit = float(spec.limit)
        if spec.side == "upper":
            binding = values >= limit - spec.tolerance
            extreme = float(values.max())
            headroom = limit - extreme
        else:
            binding = values <= limit + spec.tolerance
            extreme = float(values.min())
            headroom = extreme - limit
        criterion = "realized value at limit"
        if spec.binding_indicator is not None:
            explicit = spec.binding_indicator.reindex(values.index)
            if explicit.isna().any():
                raise ValueError(
                    f"bound {spec.name!r} has missing binding indicators"
                )
            binding = explicit.astype(bool)
            criterion = spec.binding_criterion or "explicit intervention indicator"
        share = float(binding.mean())
        rows.append(
            {
                "Bound": spec.name,
                "Side": spec.side,
                "Limit": limit,
                "Observations": int(values.size),
                "Unit": spec.unit,
                "Binding criterion": criterion,
                "Binding share": share,
                "Binding observations": int(binding.sum()),
                "Realized minimum": float(values.min()),
                "Realized median": float(values.median()),
                "Realized maximum": float(values.max()),
                "Nearest realized value": extreme,
                "Headroom to limit": float(headroom),
                "Activity": INERT if share == 0.0 else ACTIVE,
            }
        )
    return pd.DataFrame(rows)


def _drawdown_path(returns: pd.Series) -> pd.Series:
    """Drawdown from a high-water mark that starts at the initial capital."""

    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("cannot measure drawdown on an empty return series")
    nav = (1.0 + clean).cumprod()
    peak = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
    return pd.Series(nav.to_numpy() / peak - 1.0, index=clean.index)


def drawdown_episodes(
    returns: pd.Series,
    *,
    minimum_depth: float = 0.05,
) -> pd.DataFrame:
    """Every peak-to-recovery episode deeper than ``minimum_depth``.

    An episode ends when the high-water mark is regained.  The final episode is
    marked ``Recovered`` false when the series ends underwater, because a
    censored episode's duration is a lower bound and averaging it with completed
    ones would understate how long recovery takes.
    """

    if not 0 < minimum_depth < 1:
        raise ValueError("minimum_depth must be in (0, 1)")

    drawdown = _drawdown_path(returns)
    underwater = drawdown < -1e-12
    if not underwater.any():
        return pd.DataFrame(
            columns=[
                "Start",
                "Trough",
                "End",
                "Depth",
                "Sessions",
                "Sessions to trough",
                "Sessions to recover",
                "Recovered",
            ]
        )

    episode_id = (~underwater).cumsum()[underwater]
    rows: list[dict[str, object]] = []
    last_id = episode_id.iloc[-1] if len(episode_id) else None
    ends_underwater = bool(underwater.iloc[-1])
    for identifier, segment in drawdown[underwater].groupby(episode_id):
        depth = float(segment.min())
        if depth > -minimum_depth:
            continue
        trough = segment.idxmin()
        recovered = not (ends_underwater and identifier == last_id)
        rows.append(
            {
                "Start": segment.index[0],
                "Trough": trough,
                "End": segment.index[-1],
                "Depth": depth,
                "Sessions": int(segment.size),
                "Sessions to trough": int(segment.index.get_loc(trough)) + 1,
                "Sessions to recover": (
                    int(segment.size - segment.index.get_loc(trough) - 1)
                    if recovered
                    else -1
                ),
                "Recovered": recovered,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("Depth").reset_index(drop=True)


def drawdown_anatomy(
    daily: pd.DataFrame,
    *,
    threshold: float = 0.05,
    annualization: int = 252,
    exposure_column: str = "gross_notional_multiple",
    scalar_column: str = "risk_scalar",
) -> pd.DataFrame:
    """Conditional statistics inside drawdowns against the unconditional path.

    The two ratios at the end carry the finding.

    ``Volatility ratio`` near one says drawdowns are not volatility events, so a
    volatility-triggered control has nothing to trigger on.  ``Exposure ratio``
    below one says the book was already smaller while it was losing, so a
    magnitude ceiling would have clipped exposure the strategy had already given
    up.  When both hold, the loss is an accuracy failure and no size bound
    reaches it.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if "net_return" not in daily.columns:
        raise ValueError("daily must provide a 'net_return' column")

    returns = daily["net_return"].dropna().astype(float)
    drawdown = _drawdown_path(returns)
    inside = drawdown < -threshold

    def block(mask: pd.Series, label: str) -> dict[str, object]:
        sample = returns[mask]
        row: dict[str, object] = {
            "Sample": label,
            "Sessions": int(sample.size),
            "Share of sessions": float(mask.mean()),
            "Annualized return": (
                float(sample.mean() * annualization) if sample.size else np.nan
            ),
            "Annualized volatility": (
                float(sample.std(ddof=1) * math.sqrt(annualization))
                if sample.size > 1
                else np.nan
            ),
            "Daily hit rate": float((sample > 0).mean()) if sample.size else np.nan,
        }
        for column, name in (
            (exposure_column, "Mean gross notional multiple"),
            (scalar_column, "Mean risk scalar"),
        ):
            if column in daily.columns:
                aligned = daily[column].reindex(returns.index).astype(float)[mask]
                row[name] = float(aligned.mean()) if aligned.notna().any() else np.nan
            else:
                row[name] = np.nan
        return row

    everything = pd.Series(True, index=returns.index)
    frame = pd.DataFrame(
        [
            block(everything, "All sessions"),
            block(inside, f"In drawdown deeper than {threshold:.0%}"),
            block(~inside, f"Outside drawdown deeper than {threshold:.0%}"),
        ]
    )
    reference = frame.iloc[0]
    conditional = frame.iloc[1]
    frame["Volatility ratio to all sessions"] = (
        frame["Annualized volatility"] / reference["Annualized volatility"]
    )
    frame["Exposure ratio to all sessions"] = (
        frame["Mean gross notional multiple"]
        / reference["Mean gross notional multiple"]
    )
    frame["Hit-rate difference to all sessions"] = (
        frame["Daily hit rate"] - reference["Daily hit rate"]
    )
    if conditional["Sessions"] == 0:
        raise ValueError(
            f"no session is deeper than {threshold:.0%}; lower the threshold"
        )
    return frame


def episode_attribution(
    market_ledger: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    rank: int = CONCENTRATION_RANK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-market and per-asset-class contribution inside each episode.

    Returns ``(episode_summary, market_detail)``.  The summary carries the
    breadth statistics that decide whether a per-market ceiling could bite:
    how many markets lost, and what share of the total loss the worst few
    carried.  A loss spread across most of the book is a correlation event, and
    clipping any single market leaves it untouched.
    """

    required = {"date", "symbol", "asset_class", "net_return_contribution"}
    missing = sorted(required - set(market_ledger.columns))
    if missing:
        raise ValueError(f"market_ledger is missing columns: {missing}")
    if rank < 1:
        raise ValueError("rank must be at least one")

    ledger = market_ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])

    summaries: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for _, episode in episodes.iterrows():
        window = ledger[
            (ledger["date"] >= pd.Timestamp(episode["Start"]))
            & (ledger["date"] <= pd.Timestamp(episode["Trough"]))
        ]
        if window.empty:
            continue
        label = (
            f"{pd.Timestamp(episode['Start']).date()} to "
            f"{pd.Timestamp(episode['Trough']).date()}"
        )
        by_market = (
            window.groupby("symbol")["net_return_contribution"].sum().sort_values()
        )
        by_class = (
            window.groupby("asset_class")["net_return_contribution"].sum().sort_values()
        )
        losses = by_market[by_market < 0]
        total_loss = float(losses.sum())
        # The worst `rank` *losses*, not the worst `rank` entries.  When fewer
        # than `rank` markets lost, slicing the sorted series would pull
        # profitable markets into the numerator and report a concentration
        # below one for a loss that was in fact carried by a single market.
        worst = float(losses.head(rank).sum())
        traded = by_market[by_market != 0.0]
        summaries.append(
            {
                "Episode": label,
                "Depth": float(episode["Depth"]),
                "Sessions to trough": int(episode["Sessions to trough"]),
                "Markets with activity": int(traded.size),
                "Markets losing": int(losses.size),
                "Breadth of loss": (
                    float(losses.size) / float(traded.size) if traded.size else np.nan
                ),
                f"Share of loss from worst {rank}": (
                    worst / total_loss if total_loss < 0 else np.nan
                ),
                "Worst market": by_market.index[0] if by_market.size else "",
                "Worst market contribution": (
                    float(by_market.iloc[0]) if by_market.size else np.nan
                ),
                "Worst asset class": by_class.index[0] if by_class.size else "",
                "Worst asset-class contribution": (
                    float(by_class.iloc[0]) if by_class.size else np.nan
                ),
                "Asset classes losing": int((by_class < 0).sum()),
                "Asset classes with activity": int(by_class.size),
            }
        )
        details.append(
            pd.DataFrame(
                {
                    "Episode": label,
                    "Symbol": by_market.index,
                    "Contribution": by_market.to_numpy(dtype=float),
                }
            )
        )

    summary = pd.DataFrame(summaries)
    detail = (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(
            columns=["Episode", "Symbol", "Contribution"]
        )
    )
    return summary, detail


def trailing_mean_pairwise_correlation(
    contributions: pd.DataFrame,
    *,
    window: int = 63,
    minimum_markets: int = 5,
    minimum_coverage: float = 0.8,
    lag: int = 1,
) -> pd.Series:
    """Causal breadth-of-agreement in the book, as a decision-usable series.

    Episode attribution shows this strategy's drawdowns are broad: most of the
    book loses at once.  That makes cross-market agreement the state variable
    the failure mode is actually indexed by, in the way that realized volatility
    --- which barely moves in these episodes --- is not.

    Each observation is the mean off-diagonal correlation of per-market P&L over
    the ``window`` sessions strictly *before* it, then shifted a further ``lag``
    sessions.  The double displacement is deliberate and matches the convention
    in ``docs/causality.md``: the window excludes the session being labelled, and
    the lag ensures a decision taken on that session could have computed the
    value from already-published data.

    The estimate is of *realized P&L* correlation, so it embeds position sizes
    as well as price co-movement, and is therefore partly endogenous to the
    strategy's own positioning.  That is a property, not a defect --- what a
    risk control needs to know is whether the book is concentrated in one bet
    --- but it means the series is not a market-correlation estimate and must
    not be described as one.
    """

    if window <= 1:
        raise ValueError("window must exceed one session")
    if minimum_markets < 2:
        raise ValueError("minimum_markets must be at least two")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive whole number of sessions")

    frame = contributions.replace(0.0, np.nan)
    values = np.full(len(frame.index), np.nan)
    threshold = int(minimum_coverage * window)
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


def conditional_exposure_diagnostic(
    returns: pd.Series,
    conditioner: pd.Series,
    *,
    quantile: float = 0.8,
    multiplier: float = 0.5,
    annualization: int = 252,
    label: str = "conditioner",
    period_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """What halving exposure in the conditioner's upper tail would have done.

    This is a *hypothesis generator*, and the threshold is its weakness.  The
    quantile is taken over the same sample the comparison is then measured on,
    so the cut point is fitted in sample by construction.  Nothing this function
    returns is promotable evidence, and ``docs/research-methodology.md`` makes
    that mechanical: a lever discovered this way is one more trial in a count
    that is already unrecoverable, so it must be pre-registered and evaluated on
    data that has not been seen before it can be believed.

    The risk-matched row is the only one worth reading.  A conditional
    de-lever reduces volatility, which reduces drawdown for free; rescaling back
    to the unconditional volatility removes that effect and leaves only the
    change in the *shape* of the loss distribution.
    """

    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    if not 0 <= multiplier < 1:
        raise ValueError("multiplier must be in [0, 1)")

    clean = returns.dropna().astype(float)
    aligned = conditioner.reindex(clean.index)
    if aligned.notna().sum() < 2:
        raise ValueError("conditioner has fewer than two aligned observations")

    threshold = float(aligned.quantile(quantile))
    exposure = pd.Series(1.0, index=clean.index)
    exposure[aligned > threshold] = multiplier
    conditioned = clean * exposure
    years = _elapsed_years(clean.index, period_start=period_start)

    def measure(series: pd.Series, name: str, note: str) -> dict[str, object]:
        nav = (1.0 + series).cumprod()
        peak = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
        max_drawdown = float((nav.to_numpy() / peak - 1.0).min())
        volatility = float(series.std(ddof=1) * math.sqrt(annualization))
        cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
        return {
            "Configuration": name,
            "Conditioner": label,
            "Sessions": int(series.size),
            "Elapsed years": years,
            "CAGR convention": "elapsed calendar time from period start",
            "CAGR": cagr,
            "Annualized volatility": volatility,
            "Sharpe (rf=0)": (
                float(series.mean() / series.std(ddof=1) * math.sqrt(annualization))
                if series.std(ddof=1) > 0
                else np.nan
            ),
            "Max drawdown": max_drawdown,
            "Calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
            "Drawdown per unit of volatility": (
                abs(max_drawdown) / volatility if volatility > 0 else np.nan
            ),
            "Sessions de-levered": int((exposure < 1.0).sum()),
            "Threshold quantile": quantile,
            "Threshold value": threshold,
            "Exposure multiplier applied": multiplier,
            "Evidence note": note,
        }

    in_sample = "threshold fitted in sample; hypothesis only, not promotable"
    base_volatility = float(clean.std(ddof=1))
    conditioned_volatility = float(conditioned.std(ddof=1))
    rows = [
        measure(clean, "unconditional", "incumbent realized path"),
        measure(conditioned, "conditioned", in_sample),
    ]
    if conditioned_volatility > 0:
        rescale = base_volatility / conditioned_volatility
        rows.append(
            measure(
                conditioned * rescale,
                f"conditioned, risk-matched (k={rescale:.4f})",
                in_sample + "; volatility matched to the unconditional path",
            )
        )
    return pd.DataFrame(rows)


def uniform_rescale_frontier(
    returns: pd.Series,
    *,
    multipliers: Sequence[float] = DEFAULT_RESCALE_MULTIPLIERS,
    annualization: int = 252,
    period_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """What a constant de-lever already buys, as the line a lever must beat.

    Scaling every net return by a constant is the crudest possible drawdown
    control.  It is exact only for a uniformly scaled book with proportional
    costs, so the frontier slightly flatters a real de-lever whose fixed costs
    do not shrink; that makes it a conservative benchmark, since a genuine lever
    has to beat a line drawn in the rescale's favour.

    ``Drawdown per unit of volatility`` is the invariant.  A constant rescale
    leaves it unchanged by construction, so a magnitude lever that improves
    maximum drawdown *without* improving this ratio has not reduced risk --- it
    has reduced the allocation, and the committee can do that with one number.
    """

    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("cannot build a frontier from an empty return series")
    values = [float(multiplier) for multiplier in multipliers]
    if not values:
        raise ValueError("at least one multiplier is required")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("multipliers must be finite and positive")
    years = _elapsed_years(clean.index, period_start=period_start)

    rows: list[dict[str, object]] = []
    for multiplier in values:
        scaled = clean * multiplier
        nav = (1.0 + scaled).cumprod()
        peak = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
        max_drawdown = float((nav.to_numpy() / peak - 1.0).min())
        cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
        volatility = float(scaled.std(ddof=1) * math.sqrt(annualization))
        rows.append(
            {
                "Exposure multiplier": multiplier,
                "Elapsed years": years,
                "CAGR convention": "elapsed calendar time from period start",
                "CAGR": cagr,
                "Annualized volatility": volatility,
                "Max drawdown": max_drawdown,
                "Calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
                "Drawdown per unit of volatility": (
                    abs(max_drawdown) / volatility if volatility > 0 else np.nan
                ),
                "Sharpe (rf=0)": (
                    float(scaled.mean() / scaled.std(ddof=1) * math.sqrt(annualization))
                    if scaled.std(ddof=1) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def magnitude_bound_specs(
    result: object,
    config: object,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[BoundSpec, ...]:
    """The position-magnitude bounds of the DELTA1 engine, ready to report.

    ``start`` and ``end`` restrict the series to the evaluated window.  Supply
    them: ``result.daily`` is indexed from the earliest warm-up session in the
    supplied history --- 1977 for the licensed panel --- and those pre-launch
    rows carry a risk scalar computed against an empty book.  Counting them
    would dilute every binding share with sessions in which no bound could have
    bound because there was no position to clip.

    ``signal_cap`` is deliberately absent.  It reads like a ceiling and is not
    one: ``basis_momentum`` divides the clipped z-score by ``cap`` before
    returning, so the sleeve is always in ``[-1, 1]`` and lowering ``signal_cap``
    *raises* average exposure by saturating sooner.  Reporting it beside genuine
    ceilings would invite exactly the wrong inference, so its behaviour is
    documented here instead of tabulated as if it clipped position size.
    """

    daily = getattr(result, "daily", None)
    if not isinstance(daily, pd.DataFrame):
        raise ValueError("result must expose a 'daily' DataFrame")
    if start is not None or end is not None:
        daily = daily.loc[start:end]
        if daily.empty:
            raise ValueError(f"no sessions between {start!r} and {end!r}")

    specs: list[BoundSpec] = []
    if "risk_scalar" in daily.columns:
        scalar = daily["risk_scalar"]
        specs.append(
            BoundSpec(
                name="max_risk_scalar",
                limit=float(config.max_risk_scalar),
                side="upper",
                series=scalar,
            )
        )
        specs.append(
            BoundSpec(
                name="min_risk_scalar",
                limit=float(config.min_risk_scalar),
                side="lower",
                series=scalar,
            )
        )
    gross_limit = getattr(config, "max_gross_notional_multiple", None)
    if gross_limit is not None and "gross_notional_multiple" in daily.columns:
        scale = daily.get("target_portfolio_limit_scale")
        specs.append(
            BoundSpec(
                name="max_gross_notional_multiple",
                limit=float(gross_limit),
                side="upper",
                series=daily["gross_notional_multiple"],
                unit="decision sessions",
                binding_indicator=(
                    scale < 1.0 - 1e-12 if scale is not None else None
                ),
                binding_criterion=(
                    "target_portfolio_limit_scale < 1 - 1e-12"
                    if scale is not None
                    else None
                ),
            )
        )
    return tuple(specs)


__all__ = [
    "ACTIVE",
    "CONCENTRATION_RANK",
    "DEFAULT_RESCALE_MULTIPLIERS",
    "INERT",
    "BoundSpec",
    "bound_activity_report",
    "conditional_exposure_diagnostic",
    "drawdown_anatomy",
    "drawdown_episodes",
    "episode_attribution",
    "magnitude_bound_specs",
    "trailing_mean_pairwise_correlation",
    "uniform_rescale_frontier",
]
