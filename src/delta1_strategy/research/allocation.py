"""Regime-conditioned allocation across the ETF panel, and the evidence about it.

This module turns the causal states in ``research.regimes`` into a traded book
on the panel in ``marketdata.etfs``, replays it once chronologically under a
rolling walk-forward, and reports what the replay is and is not evidence for.

The book is **fully funded, long only and unlevered**.  Weights are
non-negative, sum to exactly one on every session, and the residual budget sits
in the short-duration Treasury sleeve.  That convention is forced by the data
rather than chosen for convenience: the ETF panel carries no risk-free series
and no financing series, and its ``Close`` column is already total return, so
there is no honest way to construct an excess return or to price leverage.  The
consequence must be stated wherever a number from this module appears --- "cash"
is a 1-3 year Treasury fund, it carries duration, and its return can be
negative.

What this module does NOT prove
-------------------------------
**It does not correct survivorship, and no arrangement of the code could.**
Every one of the 745 files in the vendor extract ends 2018-12-31 with positive
volume on that exact session, so no closed or delisted fund is present.
Allocating across a small fixed set of large broad index-trackers is materially
less exposed to that defect than picking winners cross-sectionally would be ---
this design makes no cross-sectional pick, so the funds that died cannot flatter
it through selection --- but the inception dates and the vendor's inclusion
decision remain.  Every frame carries ``etfs.SURVIVORSHIP_LIMITATION``.

**It does not produce prospective evidence.**  The stitched walk-forward record
is out of sample *with respect to the selector only*.  The universe rule, the
candidate set, the cost model, the risk budget, the boundary schedule and the
purge and embargo lengths were all written by someone who has read the whole
panel.  The sealed block is out of sample with respect to everything sealed at
registration time, conditional on custody actually holding, and it is one look.
Neither is a forward track record.

**It does not estimate performance conditional on a full equity bear, out of
sample.**  The sealed block 2014-2018 contains no full equity bear market, and
the Daniel-Moskowitz bear indicator used as an external coverage diagnostic is
unoccupied across all of it.  That indicator is not an allocator decision
state: the candidates' lagged Faber and time-series-momentum gates did operate
and the book held material cash.  What remains untested is how those decisions
perform through a full bear.  A reader told "five years out of sample" and not
told this has been told something true and misleading, so
``SEALED_BLOCK_LIMITATION`` travels on every artifact that carries a sealed
number.

**It does not calibrate its costs.**  The half spread and commission are
pre-declared assumptions in basis points, identical for every sleeve; the only
liquidity differentiation is the square-root participation term, which reads the
panel's own ``Turnover`` column.  Because that is an assumption rather than a
measurement, cost is a swept dial and the sweep is reported, so a reader can see
how much of the result survives at two and four times the assumed level rather
than taking the assumption on trust.

**It does not rank, promote or recommend anything.**  The fold table records
which candidate the selector took, because that is a fact about the run.  No
frame in this module carries a verdict column, and the deflated Sharpe remains
not estimable for this lineage exactly as it is for the futures sleeve.

Causality, and where each displacement comes from
-------------------------------------------------
Three displacements stack, and only their sum is meaningful:

1. Every state in ``research.regimes`` labels session ``t`` from data published
   strictly before ``t``.  That is the first displacement, and it is applied
   inside the regime layer.
2. The weight vector wanted going into session ``t`` is decided from that state,
   and the order fills at the close of ``t``.  A fill at the close of ``t``
   earns the close-to-close move ending at ``t+1``.  That is the second.
3. Rebalance dates are the sessions immediately after a calendar month end, so
   a decision founded on month-end ``d``'s close fills at the close of ``d+1``
   and first earns the move ending at ``d+2``.

Together those reproduce the repository's full-bar ``shift(2)`` convention from
``docs/causality.md`` exactly, without any series in this module being shifted
twice by hand.

The order-fill rule is a market-response model, not a look-ahead.  Order size is
computed from the *lagged* sixty-session median turnover, never from the
executing session's own; realized depth can only reduce a fill, never enlarge
one, and the unfilled residual is deferred to the next session and retried.  The
filled quantity is therefore non-decreasing in liquidity and never exceeds what
was requested, which is the monotonicity property ``docs/causality.md`` names as
the line between the two.

Where the live book state lives, and why that matters for the walk-forward
--------------------------------------------------------------------------
``docs/research-methodology.md`` is explicit that a fold must replay
chronologically carrying the actual book state, and that restarted fold NAVs or
independently spliced optimised paths are invalid.  The construction here is
splice-then-simulate-once, as in ``research.validation``: each candidate's
decision frame is built over the whole history, the selected candidate's rows
are spliced into one frame fold by fold, and the execution ledger is simulated
once from the anchor.

This sleeve can carry *all* of its path-dependent state across a boundary, which
the futures harness cannot.  Every candidate's cross-sectional weight vector is
a function of asset prices and turnover alone, so it is path independent and
splices exactly.  Everything path dependent --- compounding NAV, the actually
held book, the no-trade buffer measured against that book, the deferred order
residual --- lives inside the single simulation and therefore crosses every
boundary intact.  The volatility scalar is likewise computed from the *unscaled*
candidate book's own synthetic return series, which is path independent by
construction, following Barroso and Santa-Clara, who scale by the volatility of
the unscaled strategy rather than of the scaled one.  There is no
intent-construction approximation to disclose here.

The switching cost is charged automatically and must not be added twice: because
the ledger is simulated once over the spliced frame, it sees the jump in wanted
weights on the first rebalance after a switch and charges the full round trip.

References
----------
Faber (2007) for the ten-month trend state and the cash-when-off construction;
DeMiguel, Garlappi and Uppal (2009) for 1/N as the floor an optimiser must beat;
Kirby and Ostdiek (2012) and Asness, Frazzini and Pedersen (2012) for
inverse-volatility weighting; Moskowitz, Ooi and Pedersen (2012) for time-series
momentum; Barroso and Santa-Clara (2015) for the volatility scaling, with Harvey,
Hoyle, Rattray, Sargaison and Van Hemert (2018) for the caveat that the benefit
is concentrated in equity-like assets and should not be expected in full from a
multi-asset book; Magill and Constantinides (1976), Davis and Norman (1990) and
Garleanu and Pedersen (2013) for partial adjustment toward a desired portfolio;
Lopez de Prado (2018) for the purge-and-embargo arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.marketdata import etfs
from delta1_strategy.research import regimes
from delta1_strategy.research.inference import (
    ESTIMATED,
    NOT_ESTIMABLE,
    InferenceResult,
)
from delta1_strategy.research.validation import (
    WALK_FORWARD_SELECTION_MODE,
    WALK_FORWARD_STABILITY_MODE,
    _HAC_CONVENTION,
    _path_statistics,
    _select_highest_sharpe,
)


# ---------------------------------------------------------------------------
# limitations that travel with every number this module produces
# ---------------------------------------------------------------------------

SEALED_BLOCK_LIMITATION = (
    "sealed_block_contains_no_full_equity_bear_market; "
    "the_daniel_moskowitz_bear_and_panic_states_are_unoccupied_across_it; "
    "full_equity_bear_conditional_performance_is_not_estimable_out_of_sample"
)

FUNDING_LIMITATION = (
    "fully_funded_long_only_unlevered_book; "
    "cash_is_the_short_duration_treasury_sleeve_and_carries_duration; "
    "no_risk_free_series_and_no_financing_series_exists_in_this_panel"
)

COST_LIMITATION = (
    "costs_are_a_pre_declared_assumption_and_not_a_calibration; "
    "one_half_spread_and_one_commission_in_basis_points_for_every_sleeve; "
    "liquidity_is_differentiated_only_through_the_square_root_participation_term"
)

SELECTOR_PERMITTED_USE = (
    "chronological replay on the ETF panel; out of sample with respect to the "
    "selector only, and not with respect to specification choices already "
    "informed by this panel"
)

#: What the sealed block is, stated as what was proven rather than as a process
#: claim no artifact supports.  An earlier wording said the block was "sealed
#: before any fitting decision", and two facts falsify that.  The annual
#: selector fits *inside* the block --- fold 2015's selection window closes
#: 2014-10-28, and the chosen candidate does change as a result --- which is
#: correct walk-forward behaviour and is not a leak, but it is a fitting
#: decision taken on sealed sessions.  And the universe rule, the candidate set,
#: the cost model, the risk budget and the boundary schedule were all written
#: after the whole panel had been read, as the module docstring already
#: concedes.  What ``holdout_custody_report`` proves is the narrower and
#: genuinely useful property named first below.
SEALED_PERMITTED_USE = (
    "one look at a contiguous block held out of the development replay, proven "
    "by a byte-identical custody replay; the annual selector still fits inside "
    "the block, and the universe rule, candidate set, cost model, risk budget "
    "and boundary schedule were all written after the whole panel had been "
    "read; out of sample with respect to everything sealed at registration "
    "time, conditional on custody holding, and not a prospective track record"
)

STABILITY_PERMITTED_USE = (
    "stability of one frozen specification across expanding anchors; no "
    "selection occurs, so this is description and not validation"
)

ALLOCATION_LIMITATIONS: tuple[str, ...] = (
    etfs.SURVIVORSHIP_LIMITATION,
    SEALED_BLOCK_LIMITATION,
    FUNDING_LIMITATION,
    COST_LIMITATION,
)

SELECTOR_BASIS = "selector_out_of_sample"
SEALED_BASIS = "sealed_custody_out_of_sample"
IN_SAMPLE_BASIS = "in_sample_training_span"

#: A window that straddles the first boundary is neither basis, and labelling it
#: as either is false.  The equity-bear-market episode is the case: it runs from
#: mid-2007 into mid-2009, so part of it is training span and part of it sits
#: inside the first out-of-sample fold.  The row is still worth publishing ---
#: it is the only full bear market the panel holds --- so it gets a basis that
#: says what it is instead of a basis that rounds it to the nearer claim.
MIXED_BASIS = "straddles_the_first_boundary_partly_in_sample"

#: Faber's construction is one trend signal per sleeve and cash for any sleeve
#: that is off.  Cash is therefore not itself a sleeve: the short-duration
#: Treasury fund is the residual, never a gated position, or a sleeve switched
#: off would have nowhere to go.
DEFAULT_CASH_TICKER = "SHY"

#: The reference a "do nothing" investor holds.  Declared here rather than
#: assembled at the call site so it cannot drift between artifacts.
SIXTY_FORTY_WEIGHTS: dict[str, float] = {"SPY": 0.60, "IEF": 0.40}

FABER_TREND = "faber_trend"
TIME_SERIES_MOMENTUM = "time_series_momentum"
NO_GATE = "no_gate"

EQUAL_WEIGHT = "equal_weight"
INVERSE_VOLATILITY = "inverse_volatility"

_GATES = (FABER_TREND, TIME_SERIES_MOMENTUM, NO_GATE)
_WEIGHTINGS = (EQUAL_WEIGHT, INVERSE_VOLATILITY)

LEDGER_COLUMNS: tuple[str, ...] = (
    "gross_return",
    "cost_return",
    "net_return",
    "nav",
    "turnover_fraction",
    "traded_notional_usd",
    "impact_cost_return",
    "fixed_cost_return",
    "participation_fill_fraction",
    "order_deferred",
    "rebalance_session",
    "risk_weight_sum",
    "cash_weight",
    "sleeves_held",
    "book_live",
)

PATH_REPORT_COLUMNS: tuple[str, ...] = (
    "Path",
    "Window",
    "Basis",
    "Start",
    "End",
    "Sessions",
    "Calendar years touched",
    "Complete calendar years",
    "Elapsed years",
    "CAGR",
    "Annualized volatility",
    "Sharpe (rf=0)",
    "HAC Sharpe",
    "Sortino",
    "Max drawdown",
    "Calmar",
    "Hit rate",
    "Annual one-way turnover",
    "Annual cost drag",
    "Mean risk weight",
    "HAC convention",
    "Selection adjusted",
    "Permitted use",
    "Limitation",
)

FOLD_COLUMNS: tuple[str, ...] = (
    "lineage_id",
    "mode",
    "fold",
    "boundary",
    "test_start",
    "test_end",
    "calendar_year",
    "sessions",
    "cumulative_sessions",
    "cumulative_calendar_years",
    "training_sessions",
    "selection_window_start",
    "selection_window_end",
    "selection_sessions",
    "selection_standard_error_of_annualized_sharpe",
    "purge_sessions",
    "embargo_sessions",
    "universe_members",
    "selected_candidate",
    "selection_score_annualized_sharpe",
    "selection_switched",
    "fold_cagr",
    "fold_annualized_volatility",
    "fold_sharpe",
    "fold_hac_sharpe",
    "fold_max_drawdown",
    "fold_hit_rate",
    "fold_annual_one_way_turnover",
    "fold_annual_cost_drag",
    "oos_basis",
    "hac_convention",
    "selection_adjusted",
    "permitted_use",
    "panel_limitation",
)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _require_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_whole(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class CostModel:
    """The execution-cost assumption, stated as an assumption.

    ETF share trading is cheap and it is not free.  Three components are
    charged on the notional actually traded:

    ``half_spread_bps``
        Half of the quoted bid/ask, paid on every execution.  One number for
        every sleeve.  That is wrong in opposite directions for SPY, which
        trades inside a penny, and for DBC, which does not; the default is set
        toward the thin end so the error is conservative for the sleeve that
        matters.

    ``commission_bps``
        Broker commission expressed on notional rather than per share, because
        the panel's back-adjusted ``Volume`` column is not a share count and a
        per-share charge computed from it would be wrong by the adjustment
        factor.

    ``impact_coefficient_bps``
        The square-root participation term, ``k * sqrt(order / turnover)``,
        matching the convention in ``strategy._simulate_execution``.  This is
        the only component that distinguishes a liquid sleeve from a thin one,
        and it reads the panel's own ``Turnover`` column, so it needs no
        external calibration.

    ``cost_multiplier`` is the dial the sensitivity sweep turns.  It exists
    because none of the three components above is a measurement, and a result
    that survives only at the assumed level is a result about the assumption.
    """

    half_spread_bps: float = 2.0
    commission_bps: float = 0.5
    impact_coefficient_bps: float = 10.0
    participation_limit: float = 0.05
    turnover_window: int = 60
    turnover_lag: int = 1
    cost_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name in ("half_spread_bps", "commission_bps", "impact_coefficient_bps"):
            if _require_real(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if _require_real(self.cost_multiplier, "cost_multiplier") < 0.0:
            raise ValueError("cost_multiplier cannot be negative")
        limit = _require_real(self.participation_limit, "participation_limit")
        if not 0.0 < limit <= 1.0:
            raise ValueError("participation_limit must be in (0, 1]")
        _require_whole(self.turnover_window, "turnover_window", minimum=2)
        _require_whole(self.turnover_lag, "turnover_lag", minimum=1)

    @property
    def fixed_cost_rate(self) -> float:
        """One-way spread plus commission as a fraction of traded notional."""

        return (
            (float(self.half_spread_bps) + float(self.commission_bps))
            * float(self.cost_multiplier)
            / 10_000.0
        )

    def assumption_table(self) -> pd.DataFrame:
        """The cost model as a frame, for stamping into an artifact."""

        rows = [
            {
                "Component": "half_spread_bps",
                "Value": float(self.half_spread_bps),
                "Unit": "basis points of traded notional, one way",
                "Basis": "pre-declared assumption; identical for every sleeve",
            },
            {
                "Component": "commission_bps",
                "Value": float(self.commission_bps),
                "Unit": "basis points of traded notional, one way",
                "Basis": "pre-declared assumption; charged on notional, not per share",
            },
            {
                "Component": "impact_coefficient_bps",
                "Value": float(self.impact_coefficient_bps),
                "Unit": "basis points at full participation of session turnover",
                "Basis": "square-root participation on the panel's own Turnover column",
            },
            {
                "Component": "participation_limit",
                "Value": float(self.participation_limit),
                "Unit": "fraction of lagged median daily turnover",
                "Basis": "pre-declared cap; the unfilled residual is deferred and retried",
            },
            {
                "Component": "turnover_window",
                "Value": float(self.turnover_window),
                "Unit": "sessions",
                "Basis": "median turnover window used to size an order, lagged one session",
            },
            {
                "Component": "cost_multiplier",
                "Value": float(self.cost_multiplier),
                "Unit": "multiple of the declared level",
                "Basis": "the sweep dial; 1.0 is the declared level",
            },
        ]
        frame = pd.DataFrame(rows)
        # The full stack, not only the cost clause.  A frame that carries the
        # cost caveat alone reads as though the others do not apply to it, and
        # the survivorship caveat applies to every number this sleeve produces.
        frame["Limitation"] = "; ".join(ALLOCATION_LIMITATIONS)
        return frame


@dataclass(frozen=True)
class AllocationConfig:
    """Every constant the sleeve reads, declared before any return is computed.

    Two values deserve their provenance stated rather than defaulted silently.

    ``volatility_budget_annualized`` is 7%, the incumbent futures CTA's risk
    budget, so the two sleeves are comparable at the same ex-ante risk.  It is a
    capital-policy constant: ``docs/research-methodology.md`` says a risk-budget
    change is never an alpha result, so it is set once here and never swept.

    ``no_trade_band`` is 2% of NAV per sleeve, one fifth of the 10% equal-weight
    sleeve budget.  It is derived from the budget geometry rather than from a
    sweep on realized returns: a gate flip moves a sleeve by a whole budget, so
    it always trades, while month-to-month drift of a few tenths of a percent
    never does.  A band chosen because it produced the best Sharpe would be a
    fitted parameter in an operational disguise.
    """

    volatility_budget_annualized: float = 0.07
    book_volatility_window: int = 126
    sleeve_volatility_window: int = 63
    minimum_sleeve_volatility_annualized: float = 0.02
    max_sleeve_weight: float = 0.30
    max_volatility_scale: float = 1.0
    no_trade_band: float = 0.02
    cash_ticker: str = DEFAULT_CASH_TICKER
    initial_capital_usd: float = 100_000_000.0
    faber_months: int = 10
    momentum_lookback: int = 252
    liquidity_window: int = 252
    min_median_turnover_usd: float = 10_000_000.0
    minimum_history_sessions: int = 315
    annualization: int = 252
    hac_lags: int = 21
    costs: CostModel = CostModel()

    def __post_init__(self) -> None:
        budget = _require_real(
            self.volatility_budget_annualized, "volatility_budget_annualized"
        )
        if budget <= 0.0:
            raise ValueError("volatility_budget_annualized must be positive")
        floor = _require_real(
            self.minimum_sleeve_volatility_annualized,
            "minimum_sleeve_volatility_annualized",
        )
        if floor <= 0.0:
            raise ValueError("minimum_sleeve_volatility_annualized must be positive")
        cap = _require_real(self.max_sleeve_weight, "max_sleeve_weight")
        if not 0.0 < cap <= 1.0:
            raise ValueError("max_sleeve_weight must be in (0, 1]")
        scale = _require_real(self.max_volatility_scale, "max_volatility_scale")
        if not 0.0 < scale <= 1.0:
            raise ValueError(
                "max_volatility_scale must be in (0, 1]; a value above one is a "
                "leverage decision requiring a stated margin policy, not a "
                "research result"
            )
        band = _require_real(self.no_trade_band, "no_trade_band")
        if not 0.0 <= band < 1.0:
            raise ValueError("no_trade_band must be in [0, 1)")
        capital = _require_real(self.initial_capital_usd, "initial_capital_usd")
        if capital <= 0.0:
            raise ValueError("initial_capital_usd must be positive")
        turnover = _require_real(
            self.min_median_turnover_usd, "min_median_turnover_usd"
        )
        if turnover < 0.0:
            raise ValueError("min_median_turnover_usd cannot be negative")
        for name, minimum in (
            ("book_volatility_window", 2),
            ("sleeve_volatility_window", 2),
            ("faber_months", 2),
            ("momentum_lookback", 2),
            ("liquidity_window", 2),
            ("minimum_history_sessions", 1),
            ("annualization", 1),
            ("hac_lags", 0),
        ):
            _require_whole(getattr(self, name), name, minimum=minimum)
        if not isinstance(self.cash_ticker, str) or not self.cash_ticker.strip():
            raise ValueError("cash_ticker must be a non-empty string")
        if not isinstance(self.costs, CostModel):
            raise ValueError("costs must be a CostModel")

    def with_cost_multiplier(self, multiplier: float) -> AllocationConfig:
        """The same specification at a different cost assumption."""

        return replace(self, costs=replace(self.costs, cost_multiplier=multiplier))


# ---------------------------------------------------------------------------
# the declared candidate set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationCandidate:
    """One selectable allocation rule and the economic claim it makes.

    ``docs/research-methodology.md`` limits a batch to three economically
    distinct challengers, and a selector choosing between two spellings of one
    rule measures nothing.  ``gate`` and ``weighting`` are therefore both
    recorded: two candidates that share a gate must differ in weighting, and the
    difference must be a claim about the world rather than a parameter value.
    """

    name: str
    gate: str
    weighting: str
    claim: str
    citation: str

    def __post_init__(self) -> None:
        for field in ("name", "claim", "citation"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.gate not in _GATES:
            raise ValueError(f"gate must be one of {_GATES}, got {self.gate!r}")
        if self.weighting not in _WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {_WEIGHTINGS}, got {self.weighting!r}"
            )
        if len(self.claim) < 40:
            raise ValueError(
                f"{self.name}: claim must state the economic content of the rule"
            )


#: The declared batch.  Three rules, three distinct claims, frozen at module
#: level so a fold cannot introduce a fourth after seeing how the first three
#: did.  Candidate order is the tie-break in the selector and the first member
#: is the frozen-anchor comparison, so the order is part of the declaration.
DECLARED_CANDIDATES: tuple[AllocationCandidate, ...] = (
    AllocationCandidate(
        name="faber_trend_equal_weight",
        gate=FABER_TREND,
        weighting=EQUAL_WEIGHT,
        claim=(
            "A sleeve above its own ten-month moving average earns its equal "
            "share of the budget and a sleeve below it holds cash. Zero fitted "
            "parameters anywhere in the weights, so this is the floor every "
            "other rule must beat after costs to have earned its complexity."
        ),
        citation="Faber (2007); DeMiguel, Garlappi and Uppal (2009)",
    ),
    AllocationCandidate(
        name="faber_trend_inverse_volatility",
        gate=FABER_TREND,
        weighting=INVERSE_VOLATILITY,
        claim=(
            "The same trend gate, but budget is allocated inversely to each "
            "sleeve's trailing volatility rather than equally. The claim is "
            "that volatility is the one moment forecastable at this sample "
            "size and that timing it beats naive diversification net of the "
            "turnover it adds."
        ),
        citation="Kirby and Ostdiek (2012); Asness, Frazzini and Pedersen (2012)",
    ),
    AllocationCandidate(
        name="momentum_inverse_volatility",
        gate=TIME_SERIES_MOMENTUM,
        weighting=INVERSE_VOLATILITY,
        claim=(
            "The gate is the sign of the trailing twelve-month total return "
            "rather than position against a ten-month average. That is a "
            "different economic statement about persistence horizon, not a "
            "reparameterisation: the two states disagree on roughly a fifth of "
            "sleeve-months, most of them at turning points."
        ),
        citation="Moskowitz, Ooi and Pedersen (2012); Hurst, Ooi and Pedersen (2017)",
    ),
)


def candidate_table(
    candidates: Sequence[AllocationCandidate] = DECLARED_CANDIDATES,
) -> pd.DataFrame:
    """The declared batch as a frame, for stamping into an artifact."""

    return pd.DataFrame(
        [
            {
                "Candidate": candidate.name,
                "Regime gate": candidate.gate,
                "Weighting": candidate.weighting,
                "Economic claim": candidate.claim,
                "Citation": candidate.citation,
                "Limitation": etfs.SURVIVORSHIP_LIMITATION,
            }
            for candidate in candidates
        ]
    )


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationInputs:
    """The aligned panel plus each fund's full life, which the panel truncates.

    Two frames are needed rather than one, and the reason is causal rather than
    cosmetic.  The traded panel is the *intersection* calendar from the date on
    which every member has quoted, because a union concatenation manufactures a
    missing cell for a fund that did not exist and a forward fill turns that
    cell into a fabricated zero return.  But a trader standing in February 2006
    had SPY's history back to 1993, and computing the trend state from the
    intersection window alone would throw that away and delay the book's first
    live session by roughly a year.  So states and membership are computed on
    the full-life frames and then read onto the traded calendar.

    Both frames respect ``last_session``.  The guard is checked here as well as
    inside ``etfs.load_panel``, because the second check is the one that catches
    a bug in the first.
    """

    panel: etfs.EtfPanel
    history_close: pd.DataFrame
    history_turnover: pd.DataFrame
    permitted_last_session: pd.Timestamp | None

    def __post_init__(self) -> None:
        if not isinstance(self.panel, etfs.EtfPanel):
            raise ValueError("panel must be an EtfPanel")
        for name in ("history_close", "history_turnover"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f"{name} must be a non-empty DataFrame")
            if list(frame.columns) != list(self.panel.tickers):
                raise ValueError(f"{name} columns do not match the panel tickers")
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError(f"{name} must be indexed by session date")
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise ValueError(f"{name} index must be unique and increasing")
            if not self.panel.sessions.isin(frame.index).all():
                raise ValueError(f"{name} does not cover the traded panel calendar")
        ceiling = self.permitted_last_session
        if ceiling is not None:
            if not isinstance(ceiling, pd.Timestamp):
                raise ValueError("permitted_last_session must be a Timestamp or None")
            for name in ("history_close", "history_turnover"):
                if getattr(self, name).index.max() > ceiling:
                    raise ValueError(
                        f"{name} reaches "
                        f"{getattr(self, name).index.max().date()}, beyond the "
                        f"permitted last session {ceiling.date()}"
                    )

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.panel.tickers

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.panel.sessions


def load_allocation_inputs(
    tickers: Sequence[str] = etfs.UNIVERSE_TICKERS,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    last_session: pd.Timestamp | str | None = None,
    data_dir: Path | str | None = None,
) -> AllocationInputs:
    """Load the traded panel and the full-life frames the states are built on.

    ``start`` defaults to the date on which every named fund has quoted, so the
    traded panel is balanced from its first row and no member is ever held with
    an implicit zero weight before it existed.
    """

    ordered = tuple(tickers)
    if not ordered:
        raise ValueError("at least one ticker is required")
    ceiling = pd.Timestamp(last_session) if last_session is not None else None
    if start is None:
        start = etfs.fully_quoted_date(ordered, data_dir=data_dir)
    panel = etfs.load_panel(
        ordered,
        start=start,
        end=end,
        last_session=last_session,
        data_dir=data_dir,
    )

    funds = {ticker: etfs.load_fund(ticker, data_dir) for ticker in ordered}
    union: pd.DatetimeIndex | None = None
    for frame in funds.values():
        index = frame.index if ceiling is None else frame.index[frame.index <= ceiling]
        union = index if union is None else union.union(index)
    assert union is not None
    if ceiling is not None and len(union) and union.max() > ceiling:
        raise ValueError(
            f"full-life frames reach {union.max().date()}, beyond the permitted "
            f"last session {ceiling.date()}"
        )
    history_close = pd.DataFrame(
        {t: funds[t]["adjusted_close"].reindex(union) for t in ordered}, index=union
    )
    history_turnover = pd.DataFrame(
        {t: funds[t]["turnover_usd"].reindex(union) for t in ordered}, index=union
    )
    return AllocationInputs(
        panel=panel,
        history_close=history_close,
        history_turnover=history_turnover,
        permitted_last_session=ceiling,
    )


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def rebalance_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The sessions on which the book may trade: the first of each month.

    A month-end close is the decision instant.  The order it produces fills at
    the *next* session's close and first earns the close-to-close move after
    that, which is the repository's full-bar convention.  Returning the session
    after each month end therefore encodes the second displacement in the
    calendar rather than in a hand-applied shift, where it could be applied
    twice by accident.
    """

    if not isinstance(sessions, pd.DatetimeIndex) or len(sessions) == 0:
        raise ValueError("sessions must be a non-empty DatetimeIndex")
    ends = regimes.month_end_sessions(sessions)
    positions = sessions.get_indexer(ends) + 1
    positions = positions[(positions > 0) & (positions < len(sessions))]
    return sessions[np.unique(positions)]


def _cap_and_normalise(raw: np.ndarray, cap: float) -> np.ndarray:
    """A non-negative budget vector summing to at most one, under a ceiling.

    Water-filling rather than cap-then-renormalise: renormalising after a cap
    pushes another sleeve over the same ceiling, so the naive version silently
    violates the bound it was written to enforce.  When the ceiling times the
    member count is below one the vector cannot reach one, and the shortfall is
    left for the cash sleeve rather than being forced away by a final rescale.
    """

    weights = np.asarray(raw, dtype=float).copy()
    weights[~np.isfinite(weights)] = 0.0
    weights[weights < 0.0] = 0.0
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros_like(weights)
    weights /= total
    if cap >= 1.0:
        return weights
    for _ in range(weights.size + 1):
        over = weights > cap + 1e-15
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        free = (~over) & (weights > 0.0)
        if not free.any():
            break
        weights[free] += excess * weights[free] / float(weights[free].sum())
    return np.minimum(weights, cap)


def _gate_mask(
    inputs: AllocationInputs,
    candidate: AllocationCandidate,
    config: AllocationConfig,
) -> pd.DataFrame:
    """Which sleeves are eligible on each session, from the declared gate."""

    if candidate.gate == NO_GATE:
        return pd.DataFrame(
            True, index=inputs.panel.sessions, columns=list(inputs.tickers)
        )
    if candidate.gate == FABER_TREND:
        state = regimes.faber_trend_state(
            inputs.history_close, months=config.faber_months, lag=1
        )
    else:
        state = regimes.time_series_momentum_state(
            inputs.history_close, lookback=config.momentum_lookback, lag=1
        )
    eligible = state.eq(regimes.RISK_ON)
    frame = eligible.reindex(inputs.panel.sessions)
    return frame.fillna(False).astype(bool)


def membership_mask(
    inputs: AllocationInputs,
    config: AllocationConfig,
) -> pd.DataFrame:
    """Whether each fund may be held on each traded session, decided causally.

    Delegates to ``etfs.causal_membership_mask`` on the *full-life* turnover
    frame so a fund's seasoning is counted from its own inception rather than
    from the traded window's first row.  Counting it from the window would make
    every member of an eleven-fund panel become admissible on the same day, 315
    sessions after the panel starts, which is an artifact of where the window
    was cut and not a fact about the funds.
    """

    mask = etfs.causal_membership_mask(
        inputs.history_turnover,
        liquidity_window=config.liquidity_window,
        min_median_turnover_usd=config.min_median_turnover_usd,
        min_history_sessions=config.minimum_history_sessions,
        lag=1,
    )
    return mask.reindex(inputs.panel.sessions).fillna(False).astype(bool)


def lagged_capacity(
    inputs: AllocationInputs,
    config: AllocationConfig,
) -> pd.DataFrame:
    """Lagged median daily USD turnover per sleeve, the order-sizing evidence."""

    median = etfs.lagged_median_turnover(
        inputs.history_turnover,
        window=config.costs.turnover_window,
        lag=config.costs.turnover_lag,
    )
    return median.reindex(inputs.panel.sessions)


@dataclass(frozen=True)
class DesiredWeights:
    """One candidate's wanted book, and the pieces it was assembled from."""

    candidate: str
    weights: pd.DataFrame
    risk_weights: pd.DataFrame
    volatility_scale: pd.Series
    membership: pd.DataFrame
    eligibility: pd.DataFrame
    unscaled_book_return: pd.Series
    first_live_session: pd.Timestamp | None

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.DataFrame) or self.weights.empty:
            raise ValueError("weights must be a non-empty DataFrame")
        sums = self.weights.sum(axis=1).to_numpy(dtype=float)
        if not np.allclose(sums, 1.0, atol=1e-12):
            raise ValueError(
                "the book is fully funded and unlevered, so every row of weights "
                "must sum to exactly one"
            )
        if (self.weights.to_numpy(dtype=float) < -1e-12).any():
            raise ValueError("the book is long only, so no weight may be negative")


def desired_weight_frame(
    inputs: AllocationInputs,
    candidate: AllocationCandidate,
    config: AllocationConfig,
) -> DesiredWeights:
    """The weight vector the book wants going into each session.

    Three steps, in this order, each causal on its own.

    *Membership* decides which sleeves may be held at all, from trailing
    turnover and each fund's own seasoning.  A non-member's budget is not
    redistributed among the survivors; it goes to cash, because redistributing
    it would silently concentrate the book in whichever sleeves happened to be
    admissible early.

    *The gate* decides which members carry their budget this month.  Faber's
    construction is followed exactly: a sleeve that is off holds cash rather
    than handing its share to a sleeve that is on.

    *Volatility scaling* then sizes the whole risk book.  ``sigmahat`` is the
    realized volatility of the *unscaled* candidate over the previous 126
    sessions, following Barroso and Santa-Clara, who scale by the volatility of
    the unscaled strategy.  That estimator has no fitted constant, and because
    it reads only the candidate's own synthetic returns it is path independent
    and splices exactly across a walk-forward boundary.  Where it cannot be
    computed the scale is zero and the book holds cash: the pre-declared neutral
    state is no position, never an unscaled one.

    The cash sleeve is the residual and is never gated.  If it were, a sleeve
    switched off would have nowhere to go.
    """

    if not isinstance(candidate, AllocationCandidate):
        raise ValueError("candidate must be an AllocationCandidate")
    if not isinstance(config, AllocationConfig):
        raise ValueError("config must be an AllocationConfig")
    tickers = list(inputs.tickers)
    if config.cash_ticker not in tickers:
        raise ValueError(
            f"the cash sleeve {config.cash_ticker!r} is not in the panel"
        )
    sessions = inputs.panel.sessions
    cash_position = tickers.index(config.cash_ticker)
    risk_positions = [i for i in range(len(tickers)) if i != cash_position]

    members = membership_mask(inputs, config)
    eligible = _gate_mask(inputs, candidate, config) & members

    if candidate.weighting == EQUAL_WEIGHT:
        raw = members.astype(float)
    else:
        sigma = regimes.realized_volatility(
            inputs.panel.returns,
            window=config.sleeve_volatility_window,
            annualization=config.annualization,
            demean=True,
            lag=1,
        ).reindex(sessions)
        floored = sigma.clip(lower=config.minimum_sleeve_volatility_annualized)
        raw = (1.0 / floored).where(members, 0.0).fillna(0.0)
        # Before a sleeve's volatility can be estimated it cannot be sized, so
        # it is not a member for weighting purposes either.  Falling back to an
        # equal share here would quietly reintroduce the rule this candidate
        # exists to be distinct from.
        raw = raw.where(sigma.notna(), 0.0)

    raw_values = raw.to_numpy(dtype=float).copy()
    raw_values[:, cash_position] = 0.0
    budgets = np.zeros_like(raw_values)
    for position in range(raw_values.shape[0]):
        budgets[position] = _cap_and_normalise(
            raw_values[position], float(config.max_sleeve_weight)
        )
    gated = eligible.to_numpy(dtype=bool).copy()
    gated[:, cash_position] = False
    risk_weights = pd.DataFrame(
        np.where(gated, budgets, 0.0), index=sessions, columns=tickers
    )

    lagged = risk_weights.shift(1).fillna(0.0)
    unscaled = (lagged * inputs.panel.returns.reindex(sessions).fillna(0.0)).sum(axis=1)
    unscaled = unscaled.iloc[1:]
    scale = regimes.barroso_santa_clara_scale(
        unscaled,
        window=config.book_volatility_window,
        volatility_budget_annualized=config.volatility_budget_annualized,
        min_scale=0.0,
        max_scale=config.max_volatility_scale,
        lag=1,
    ).reindex(sessions)
    scale = scale.fillna(0.0).clip(lower=0.0, upper=config.max_volatility_scale)

    scaled = risk_weights.mul(scale, axis=0)
    weights = scaled.copy()
    weights.iloc[:, cash_position] = 0.0
    weights[config.cash_ticker] = 1.0 - weights.iloc[:, risk_positions].sum(axis=1)
    weights = weights[tickers]

    live = (weights.iloc[:, risk_positions].sum(axis=1) > 0.0).to_numpy()
    first_live = sessions[int(np.argmax(live))] if live.any() else None

    return DesiredWeights(
        candidate=candidate.name,
        weights=weights,
        risk_weights=risk_weights,
        volatility_scale=scale,
        membership=members,
        eligibility=eligible,
        unscaled_book_return=unscaled,
        first_live_session=first_live,
    )


def fixed_weight_frame(
    inputs: AllocationInputs,
    config: AllocationConfig,
    weights: dict[str, float],
    *,
    label: str,
) -> DesiredWeights:
    """A constant policy portfolio on the same universe, calendar and costs.

    The reference the allocator has to beat is not zero; it is doing nothing.
    A fixed 60/40 and an equal-weight panel are both built through this function
    so they are charged the same spread, commission, impact and participation
    cap as the allocator, and any advantage the allocator shows is an advantage
    over a like-for-like implementation rather than over an uncosted line.
    """

    tickers = list(inputs.tickers)
    unknown = sorted(set(weights) - set(tickers))
    if unknown:
        raise ValueError(f"reference weights name funds outside the panel: {unknown}")
    total = float(sum(weights.values()))
    if total <= 0.0 or total > 1.0 + 1e-12:
        raise ValueError("reference weights must be positive and sum to at most one")
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("reference weights must be non-negative")

    sessions = inputs.panel.sessions
    members = membership_mask(inputs, config)
    frame = pd.DataFrame(0.0, index=sessions, columns=tickers)
    for ticker, value in weights.items():
        frame[ticker] = float(value) * members[ticker].astype(float)
    risk = frame.copy()
    risk[config.cash_ticker] = 0.0
    held = risk.copy()
    held[config.cash_ticker] = 1.0 - risk.sum(axis=1)
    held = held[tickers]

    lagged = risk.shift(1).fillna(0.0)
    unscaled = (lagged * inputs.panel.returns.reindex(sessions).fillna(0.0)).sum(axis=1)
    live = (risk.sum(axis=1) > 0.0).to_numpy()
    return DesiredWeights(
        candidate=label,
        weights=held,
        risk_weights=risk,
        volatility_scale=pd.Series(1.0, index=sessions),
        membership=members,
        eligibility=members,
        unscaled_book_return=unscaled.iloc[1:],
        first_live_session=sessions[int(np.argmax(live))] if live.any() else None,
    )


def equal_weight_reference(
    inputs: AllocationInputs, config: AllocationConfig
) -> DesiredWeights:
    """Every admissible sleeve at an equal share, including the cash sleeve."""

    tickers = list(inputs.tickers)
    sessions = inputs.panel.sessions
    members = membership_mask(inputs, config)
    counts = members.sum(axis=1).replace(0, np.nan)
    frame = members.astype(float).div(counts, axis=0).fillna(0.0)
    risk = frame.copy()
    risk[config.cash_ticker] = 0.0
    held = risk.copy()
    held[config.cash_ticker] = 1.0 - risk.sum(axis=1)
    held = held[tickers]
    lagged = risk.shift(1).fillna(0.0)
    unscaled = (lagged * inputs.panel.returns.reindex(sessions).fillna(0.0)).sum(axis=1)
    live = (risk.sum(axis=1) > 0.0).to_numpy()
    return DesiredWeights(
        candidate="equal_weight_panel",
        weights=held,
        risk_weights=risk,
        volatility_scale=pd.Series(1.0, index=sessions),
        membership=members,
        eligibility=members,
        unscaled_book_return=unscaled.iloc[1:],
        first_live_session=sessions[int(np.argmax(live))] if live.any() else None,
    )


# ---------------------------------------------------------------------------
# the execution simulator: the only place path-dependent state lives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationLedger:
    """One simulated path: daily ledger, realized weights and the label."""

    label: str
    daily: pd.DataFrame
    weights: pd.DataFrame
    config: AllocationConfig
    limitations: tuple[str, ...] = ALLOCATION_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(self.daily, pd.DataFrame) or self.daily.empty:
            raise ValueError("daily must be a non-empty DataFrame")
        missing = [c for c in LEDGER_COLUMNS if c not in self.daily.columns]
        if missing:
            raise ValueError(f"daily is missing ledger columns {missing}")
        if not self.weights.index.equals(self.daily.index):
            raise ValueError("weights and daily must share one index")
        if etfs.SURVIVORSHIP_LIMITATION not in self.limitations:
            raise ValueError("the survivorship limitation may not be dropped")

    @property
    def net_return(self) -> pd.Series:
        return self.daily["net_return"].astype(float)

    @property
    def live_sessions(self) -> pd.DatetimeIndex:
        live = self.daily["book_live"].to_numpy(dtype=bool)
        return self.daily.index[live]


def simulate_allocation(
    desired: DesiredWeights,
    inputs: AllocationInputs,
    config: AllocationConfig,
    *,
    label: str | None = None,
    trade_sessions: pd.DatetimeIndex | None = None,
) -> AllocationLedger:
    """Replay one book chronologically, carrying every piece of live state.

    The loop is not vectorised, and that is deliberate.  Four quantities are
    path dependent and each of them would be silently wrong in a vectorised
    version: NAV compounds, the no-trade band is measured against the *actually
    held* book rather than against last month's wanted book, an order capped by
    participation leaves a residual that is retried on later sessions, and the
    weights drift between rebalances with the sleeves' own returns.

    Order of operations within session ``t``:

    1. The book held from the close of ``t-1`` earns ``r_t``, and the held
       weights drift with the sleeves' relative performance.
    2. If ``t`` is a rebalance session, the wanted vector for ``t`` --- computed
       from data published strictly before ``t`` --- is passed through the
       no-trade band against the drifted book to produce a banded hold.
    3. Any outstanding banded hold is traded toward, subject to a single
       participation scalar so the vector of trades still sums to zero and the
       book stays fully invested.  Whatever is unfilled stays outstanding.
    4. Costs are charged on what actually traded, and NAV compounds on the net.

    Step 3 uses one scalar for the whole rebalance rather than a per-sleeve cap
    because a per-sleeve cap breaks the funding identity: capping one leg and
    not the other leaves the weights summing to something other than one, and
    the usual repair --- renormalising --- silently enlarges the fills that were
    not capped.  A single scalar means a thin sleeve throttles the whole
    rebalance, which is conservative and is reported as
    ``participation_fill_fraction`` so the reader can see when it bound.

    One quantity in step 3 is contemporaneous with the executing session and it
    is worth naming rather than leaving to be found: the NAV that converts a
    weight into a notional is the NAV after ``r_t``, because the fill is at
    ``t``'s close and a desk trading the close knows the day's mark.  It cannot
    flatter anything.  The *fraction* traded is NAV-free by construction, so NAV
    enters only through the participation cap and the impact charge, and in both
    it enters in the denominator of the fill and the numerator of the cost: a
    larger NAV shrinks the fill and enlarges the charge.  The size limit itself
    comes from the lagged sixty-session median turnover, never from the
    executing session's own, which is the distinction ``docs/causality.md``
    draws between a market-response model and a look-ahead.
    """

    if not isinstance(desired, DesiredWeights):
        raise ValueError("desired must be a DesiredWeights")
    if not isinstance(config, AllocationConfig):
        raise ValueError("config must be an AllocationConfig")
    tickers = list(inputs.tickers)
    returns = inputs.panel.returns
    sessions = returns.index
    cash_position = tickers.index(config.cash_ticker)

    wanted = desired.weights.reindex(sessions)[tickers].to_numpy(dtype=float)
    capacity = lagged_capacity(inputs, config).reindex(sessions)[tickers]
    median_turnover = capacity.to_numpy(dtype=float)
    return_values = returns[tickers].to_numpy(dtype=float)

    trading = (
        rebalance_sessions(inputs.panel.sessions)
        if trade_sessions is None
        else pd.DatetimeIndex(trade_sessions)
    )
    is_rebalance = np.zeros(len(sessions), dtype=bool)
    positions = sessions.get_indexer(trading)
    is_rebalance[positions[positions >= 0]] = True

    band = float(config.no_trade_band)
    participation = float(config.costs.participation_limit)
    fixed_rate = config.costs.fixed_cost_rate
    impact_bps = float(config.costs.impact_coefficient_bps) * float(
        config.costs.cost_multiplier
    )

    held = np.zeros(len(tickers), dtype=float)
    held[cash_position] = 1.0
    outstanding: np.ndarray | None = None
    nav = 1.0
    records: list[dict[str, float | bool]] = []
    weight_rows = np.empty((len(sessions), len(tickers)), dtype=float)

    for position in range(len(sessions)):
        row_return = return_values[position]
        gross = float(held @ row_return)
        growth = 1.0 + gross
        if not np.isfinite(growth) or growth <= 0.0:
            raise ValueError(
                f"the book was wiped out on {sessions[position].date()}; the "
                "simulator refuses to continue past a non-positive NAV"
            )
        held = held * (1.0 + row_return) / growth

        if is_rebalance[position]:
            want = wanted[position]
            if np.isfinite(want).all():
                outstanding = _banded_hold(want, held, band, cash_position)
            else:
                outstanding = None

        traded = np.zeros(len(tickers), dtype=float)
        fill_fraction = 1.0
        deferred = False
        if outstanding is not None:
            delta = outstanding - held
            nav_usd = nav * float(config.initial_capital_usd)
            caps = np.where(
                np.isfinite(median_turnover[position]),
                median_turnover[position] * participation / max(nav_usd, 1e-12),
                0.0,
            )
            moving = np.abs(delta) > 1e-15
            if moving.any():
                ratios = np.where(moving, caps / np.maximum(np.abs(delta), 1e-15), np.inf)
                fill_fraction = float(min(1.0, float(np.nanmin(ratios))))
                if not np.isfinite(fill_fraction) or fill_fraction < 0.0:
                    fill_fraction = 0.0
                traded = fill_fraction * delta
                held = held + traded
                deferred = fill_fraction < 1.0 - 1e-12
                if not deferred:
                    outstanding = None
            else:
                outstanding = None

        absolute = np.abs(traded)
        turnover_fraction = float(absolute.sum())
        nav_usd = nav * float(config.initial_capital_usd)
        notional = absolute * nav_usd
        fixed_cost = turnover_fraction * fixed_rate
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(
                np.isfinite(median_turnover[position])
                & (median_turnover[position] > 0.0),
                notional / np.maximum(median_turnover[position], 1e-12),
                0.0,
            )
        impact_cost = float((absolute * impact_bps * np.sqrt(share) / 10_000.0).sum())
        cost = fixed_cost + impact_cost
        net = gross - cost
        nav = nav * (1.0 + net)

        weight_rows[position] = held
        records.append(
            {
                "gross_return": gross,
                "cost_return": cost,
                "net_return": net,
                "nav": nav,
                "turnover_fraction": turnover_fraction,
                "traded_notional_usd": float(notional.sum()),
                "impact_cost_return": impact_cost,
                "fixed_cost_return": fixed_cost,
                "participation_fill_fraction": fill_fraction,
                "order_deferred": bool(deferred),
                "rebalance_session": bool(is_rebalance[position]),
                "risk_weight_sum": float(1.0 - held[cash_position]),
                "cash_weight": float(held[cash_position]),
                "sleeves_held": int((held > 1e-9).sum()),
                "book_live": bool(1.0 - held[cash_position] > 1e-9),
            }
        )

    daily = pd.DataFrame(records, index=sessions)[list(LEDGER_COLUMNS)]
    weights = pd.DataFrame(weight_rows, index=sessions, columns=tickers)
    return AllocationLedger(
        label=label or desired.candidate,
        daily=daily,
        weights=weights,
        config=config,
    )


def _banded_hold(
    want: np.ndarray,
    held: np.ndarray,
    band: float,
    cash_position: int,
) -> np.ndarray:
    """Trade only to the edge of the no-trade band, never all the way.

    Garleanu and Pedersen's aim-portfolio result says partial adjustment is the
    optimal response to trading costs and that full adjustment is the zero-cost
    special case, so stopping at the band edge is the theoretically correct
    behaviour rather than a shortcut.  The cash sleeve absorbs whatever the risk
    sleeves do not take, which keeps the funding identity exact.
    """

    banded = held.copy()
    for position in range(len(want)):
        if position == cash_position:
            continue
        gap = float(want[position] - held[position])
        if abs(gap) < band:
            banded[position] = held[position]
        else:
            banded[position] = want[position] - math.copysign(band, gap)
    banded[cash_position] = 1.0 - float(
        banded.sum() - banded[cash_position]
    )
    return banded


# ---------------------------------------------------------------------------
# path statistics
# ---------------------------------------------------------------------------


def _sortino(values: np.ndarray, annualization: int) -> float:
    downside = np.minimum(values, 0.0)
    deviation = float(np.sqrt((downside * downside).mean()))
    if deviation <= 0.0:
        return float("nan")
    return float(values.mean() / deviation * math.sqrt(annualization))


def _calendar_years(index: pd.DatetimeIndex) -> int:
    """How many calendar years the window touches, complete or not."""

    return int(pd.Index(index.year).nunique())


def _complete_calendar_years(
    index: pd.DatetimeIndex, calendar: pd.DatetimeIndex
) -> int:
    """How many calendar years the window covers end to end.

    The five-year claim is a counting argument, so the count has to be exact.
    A window running 2006-02-06 to 2008-12-31 touches three calendar years and
    completes two, and reporting three would be an overstatement of a third of
    the record.  A year is complete when the window holds every session the
    panel has in it *and* the panel's own coverage of that year runs from
    January to December --- the second clause is what stops the panel's own
    truncated first and last years from being counted as whole.
    """

    complete = 0
    for year in sorted(set(int(value) for value in index.year)):
        block = index[index.year == year]
        whole = calendar[calendar.year == year]
        if len(whole) == 0 or len(block) != len(whole):
            continue
        if whole[0].month != 1 or whole[-1].month != 12:
            continue
        complete += 1
    return complete


def path_statistics(
    returns: pd.Series,
    *,
    annualization: int = 252,
    hac_lags: int = 21,
    period_start: pd.Timestamp | None = None,
) -> dict[str, float]:
    """CAGR, volatility, Sharpe, HAC Sharpe, Sortino, drawdown, Calmar, hit rate.

    The first six delegate to ``validation._path_statistics`` unchanged, so a
    Sharpe quoted for this sleeve and a Sharpe quoted for the futures sleeve are
    the same estimator under the same HAC convention, and CAGR is annualized on
    elapsed calendar time rather than on a session count.  Only Sortino and
    Calmar are added here, and both are defined so a degenerate input yields
    ``nan`` rather than an inflated number: a path with no losing session has no
    downside deviation, and a path with no drawdown has no Calmar.

    ``period_start`` is the timestamp elapsed time is measured from.  Passing
    the session immediately before the window makes consecutive windows' elapsed
    periods tile without gaps, which is what stops a fold table's CAGRs from
    each quietly omitting their own first day.
    """

    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("cannot summarise an empty return series")
    base = _path_statistics(
        clean,
        annualization=annualization,
        hac_lags=hac_lags,
        period_start=period_start,
    )
    values = clean.to_numpy(dtype=float)
    drawdown = base["max_drawdown"]
    base["sortino"] = _sortino(values, annualization)
    base["calmar"] = (
        base["cagr"] / abs(drawdown)
        if np.isfinite(drawdown) and drawdown < 0.0
        else float("nan")
    )
    return base


def _previous_session(index: pd.DatetimeIndex, stamp: pd.Timestamp) -> pd.Timestamp:
    """The last session strictly before ``stamp``, or ``stamp`` itself.

    Elapsed-time CAGR is measured from the session before the window so that
    consecutive windows tile without gaps.  Falling back to the window's own
    first session at the very start of the path is correct rather than a fudge:
    there is no earlier session for the first return to have been earned from.
    """

    earlier = index[index < stamp]
    return earlier[-1] if len(earlier) else stamp


def path_report(
    ledger: AllocationLedger,
    *,
    window: str,
    basis: str,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    permitted_use: str = SELECTOR_PERMITTED_USE,
    label: str | None = None,
) -> pd.DataFrame:
    """One row summarising one path over one window."""

    full_index = ledger.daily.index
    daily = ledger.daily
    if start is not None:
        daily = daily.loc[daily.index >= pd.Timestamp(start)]
    if end is not None:
        daily = daily.loc[daily.index <= pd.Timestamp(end)]
    if daily.empty:
        raise ValueError(f"{ledger.label}: the requested window holds no sessions")
    annualization = ledger.config.annualization
    returns = daily["net_return"].astype(float)
    anchor = _previous_session(full_index, daily.index[0])
    statistics = path_statistics(
        returns,
        annualization=annualization,
        hac_lags=ledger.config.hac_lags,
        period_start=anchor,
    )
    elapsed = max((daily.index[-1] - anchor).days / 365.2425, 1e-9)
    years = len(daily) / annualization
    row = {
        "Path": label or ledger.label,
        "Window": window,
        "Basis": basis,
        "Start": daily.index[0].date().isoformat(),
        "End": daily.index[-1].date().isoformat(),
        "Sessions": int(len(daily)),
        "Calendar years touched": _calendar_years(daily.index),
        "Complete calendar years": _complete_calendar_years(daily.index, full_index),
        "Elapsed years": float(elapsed),
        "CAGR": statistics["cagr"],
        "Annualized volatility": statistics["annualized_volatility"],
        "Sharpe (rf=0)": statistics["sharpe"],
        "HAC Sharpe": statistics["hac_sharpe"],
        "Sortino": statistics["sortino"],
        "Max drawdown": statistics["max_drawdown"],
        "Calmar": statistics["calmar"],
        "Hit rate": statistics["hit_rate"],
        "Annual one-way turnover": float(
            daily["turnover_fraction"].sum() / max(years, 1e-9)
        ),
        "Annual cost drag": float(daily["cost_return"].mean() * annualization),
        "Mean risk weight": float(daily["risk_weight_sum"].mean()),
        "HAC convention": _HAC_CONVENTION,
        "Selection adjusted": False,
        "Permitted use": permitted_use,
        "Limitation": "; ".join(ledger.limitations),
    }
    return pd.DataFrame([row], columns=list(PATH_REPORT_COLUMNS))


# ---------------------------------------------------------------------------
# the rolling walk-forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardPlan:
    """Anchor, boundaries, stand-off and the sealed block, all pre-declared.

    ``purge_sessions`` and ``embargo_sessions`` are derived rather than
    asserted, and the derivation matters because it explains why they cost
    nothing.

    *Purge* removes the returns of trades that straddle a boundary from the
    statistic the selector reads.  A month's holding period plus two sessions of
    execution lag is 23 sessions, and that is exactly the set of training
    returns still being earned inside the test fold.  Note what is *not* purged:
    a test-fold signal on the first session after a boundary legitimately reads
    the previous 252 sessions of prices, which are training data.  That is the
    information a live trader has on that day, and purging it would model a
    trader with amnesia.

    *Embargo* removes a further month, because training returns immediately
    before a boundary remain correlated with early test returns through
    volatility clustering.  Two independent derivations agree: 23 + 21 = 44, and
    Lopez de Prado's rule of thumb of one percent of a roughly 3,250-session
    panel is 33 --- the same order, and the more conservative of the two is
    taken.

    Both shorten only the window the selector scores.  Because the simulated
    path is one continuous replay that is never restarted, they remove no
    session from the out-of-sample record, so being conservative here is nearly
    free.  The longest-lookback derivation --- setting the embargo to 252
    sessions because a training and a test return can share a price observation
    --- was considered and rejected: sharing an input is not contamination when
    the sharing is what the live strategy does, and at the first boundary it
    would leave too few scoreable sessions for the selector to be measuring
    anything.
    """

    anchor: str
    boundaries: tuple[str, ...]
    end: str
    sealed_start: str
    purge_sessions: int = 23
    embargo_sessions: int = 21
    minimum_selection_sessions: int = 252
    lineage_id: str = "etf_regime_allocation_v1"

    def __post_init__(self) -> None:
        for name in ("anchor", "end", "sealed_start", "lineage_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.boundaries, tuple) or not self.boundaries:
            raise ValueError("boundaries must be a non-empty tuple of date strings")
        stamps = [pd.Timestamp(value) for value in self.boundaries]
        if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
            raise ValueError("boundaries must be sorted and unique")
        if stamps[0] <= pd.Timestamp(self.anchor):
            raise ValueError("the first boundary must fall after the anchor")
        if stamps[-1] >= pd.Timestamp(self.end):
            raise ValueError("the last boundary must fall before the evaluation end")
        sealed = pd.Timestamp(self.sealed_start)
        if not stamps[0] <= sealed <= stamps[-1]:
            raise ValueError(
                "the sealed block must start on or between the first and last "
                "boundary, or the fold table cannot label which claim each row "
                "supports"
            )
        for name in ("purge_sessions", "embargo_sessions"):
            _require_whole(getattr(self, name), name, minimum=0)
        _require_whole(
            self.minimum_selection_sessions, "minimum_selection_sessions", minimum=2
        )

    @property
    def boundary_stamps(self) -> tuple[pd.Timestamp, ...]:
        return tuple(pd.Timestamp(value) for value in self.boundaries)

    @property
    def stand_off_sessions(self) -> int:
        return int(self.purge_sessions) + int(self.embargo_sessions)


def annual_boundaries(
    sessions: pd.DatetimeIndex,
    *,
    first_year: int,
    last_year: int,
) -> tuple[str, ...]:
    """The first traded session on or after 1 January of each named year.

    A boundary that is not itself a session is harmless because the window test
    is ``session >= boundary``, but resolving to a real session makes the fold
    table's arithmetic checkable by eye against the panel calendar.
    """

    if not isinstance(sessions, pd.DatetimeIndex) or len(sessions) == 0:
        raise ValueError("sessions must be a non-empty DatetimeIndex")
    if last_year < first_year:
        raise ValueError("last_year cannot precede first_year")
    stamps: list[str] = []
    for year in range(int(first_year), int(last_year) + 1):
        candidates = sessions[sessions >= pd.Timestamp(year=year, month=1, day=1)]
        if len(candidates) == 0:
            continue
        stamps.append(candidates[0].date().isoformat())
    if not stamps:
        raise ValueError("no boundary falls inside the session calendar")
    return tuple(dict.fromkeys(stamps))


@dataclass
class WalkForwardResult:
    """The single spliced replay, its folds, and the paths that fed the selector."""

    label: str
    mode: str
    plan: WalkForwardPlan
    folds: pd.DataFrame
    stitched: AllocationLedger
    anchor_ledger: AllocationLedger
    candidate_ledgers: dict[str, AllocationLedger]
    spliced: DesiredWeights
    summary: pd.Series

    @property
    def spliced_weights(self) -> pd.DataFrame:
        return self.spliced.weights


def _standoff_selector(
    plan: WalkForwardPlan,
) -> Callable[[pd.DataFrame], tuple[str, float]]:
    """The house selector, handed a training frame short of the boundary.

    ``validation._select_highest_sharpe`` is reused unchanged so the ETF and
    futures walk-forwards select on the same statistic.  The wrapper trims the
    stand-off and refuses a stub rather than selecting on one: a selector
    scoring forty sessions is not a weak selector, it is a coin flip with a
    column name.
    """

    stand_off = plan.stand_off_sessions

    def select(training: pd.DataFrame) -> tuple[str, float]:
        trimmed = training.iloc[: len(training) - stand_off] if stand_off else training
        if len(trimmed) < plan.minimum_selection_sessions:
            raise ValueError(
                f"the selection window holds {len(trimmed)} sessions after a "
                f"{stand_off}-session stand-off, below the "
                f"{plan.minimum_selection_sessions} required; move the first "
                "boundary later rather than selecting on a stub"
            )
        return _select_highest_sharpe(trimmed)

    return select


def run_walk_forward(
    inputs: AllocationInputs,
    config: AllocationConfig,
    plan: WalkForwardPlan,
    *,
    candidates: Sequence[AllocationCandidate] = DECLARED_CANDIDATES,
    label: str = "etf_regime_allocation_walk_forward",
    select_fn: Callable[[pd.DataFrame], tuple[str, float]] | None = None,
) -> WalkForwardResult:
    """Select inside each training span, splice, and simulate the book once.

    The object being evaluated is not one strategy per fold.  It is a single
    strategy whose rule reads "each January, take the candidate with the highest
    purged and embargoed training-span Sharpe and hold it for the year", and
    every cost that rule incurs is its cost.  Because the ledger is simulated
    once over the spliced weight frame it sees the jump in wanted weights on the
    first rebalance after a switch and charges the full round trip
    automatically; adding a separate switching charge on top would double-count.

    With one candidate there is no selector and the mode is relabelled: the
    result is then a stability description of one frozen specification across
    expanding anchors, not validation.
    """

    members = tuple(candidates)
    if not members:
        raise ValueError("at least one candidate is required")
    names = [candidate.name for candidate in members]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    mode = (
        WALK_FORWARD_SELECTION_MODE
        if len(members) > 1
        else WALK_FORWARD_STABILITY_MODE
    )
    permitted_use = (
        SELECTOR_PERMITTED_USE if len(members) > 1 else STABILITY_PERMITTED_USE
    )

    wanted = {
        candidate.name: desired_weight_frame(inputs, candidate, config)
        for candidate in members
    }
    ledgers = {
        name: simulate_allocation(frame, inputs, config, label=name)
        for name, frame in wanted.items()
    }
    candidate_daily = pd.DataFrame(
        {name: ledgers[name].net_return for name in names}
    )

    sessions = inputs.panel.returns.index
    anchor = pd.Timestamp(plan.anchor)
    stamps = plan.boundary_stamps
    end = pd.Timestamp(plan.end)
    sealed_start = pd.Timestamp(plan.sealed_start)
    selector = select_fn or _standoff_selector(plan)

    spliced = wanted[names[0]].weights.copy()
    spliced_risk = wanted[names[0]].risk_weights.copy()
    spliced_scale = wanted[names[0]].volatility_scale.copy()
    spliced_eligibility = wanted[names[0]].eligibility.copy()
    membership = membership_mask(inputs, config)
    records: list[dict[str, object]] = []
    # Every decision before the first boundary comes from the first declared
    # candidate, so that candidate --- not "nothing" --- is the regime in force
    # when fold zero chooses.  Seeding with ``None`` records a fold-zero
    # departure from the warm-up frame as no switch at all, which undercounts
    # the discontinuities the spliced weight frame actually contains by one and
    # hides the largest of them: at the first boundary of the supplied run the
    # spliced wanted vector jumps by an L1 norm of 1.03 against 0.40 for the
    # anchor-only path.  ``validation.run_walk_forward`` seeds with names[0] for
    # exactly this reason and this mirrors it.
    previous: str = names[0]
    cumulative_sessions = 0
    seen_years: set[int] = set()

    for position, boundary in enumerate(stamps):
        final = position + 1 == len(stamps)
        stop = end if final else stamps[position + 1]
        window = (sessions >= boundary) & (
            (sessions <= stop) if final else (sessions < stop)
        )
        if not window.any():
            raise ValueError(
                f"fold {position} spans no sessions between {boundary.date()} "
                f"and {stop.date()}"
            )
        training = candidate_daily.loc[
            (candidate_daily.index >= anchor) & (candidate_daily.index < boundary)
        ]
        if mode == WALK_FORWARD_SELECTION_MODE:
            choice, score = selector(training)
            if choice not in wanted:
                raise ValueError(f"the selector returned an unknown candidate: {choice!r}")
        else:
            choice, score = names[0], float("nan")

        stand_off = plan.stand_off_sessions
        scored = training.iloc[: len(training) - stand_off] if stand_off else training
        # The weight frames carry the panel's own calendar, which is one session
        # longer than the return calendar the fold windows are built on, so the
        # splice mask is rebuilt on the weight index rather than reused.
        decisions = (spliced.index >= boundary) & (
            (spliced.index <= stop) if final else (spliced.index < stop)
        )
        spliced.loc[decisions] = wanted[choice].weights.loc[decisions]
        spliced_risk.loc[decisions] = wanted[choice].risk_weights.loc[decisions]
        spliced_scale.loc[decisions] = wanted[choice].volatility_scale.loc[decisions]
        spliced_eligibility.loc[decisions] = wanted[choice].eligibility.loc[decisions]

        fold_sessions = int(window.sum())
        cumulative_sessions += fold_sessions
        fold_index = sessions[window]
        seen_years.update(int(year) for year in fold_index.year.unique())
        records.append(
            {
                "lineage_id": plan.lineage_id,
                "mode": mode,
                "fold": position,
                "boundary": boundary.date().isoformat(),
                "test_start": fold_index[0].date().isoformat(),
                "test_end": fold_index[-1].date().isoformat(),
                "calendar_year": int(fold_index[0].year),
                "sessions": fold_sessions,
                "cumulative_sessions": cumulative_sessions,
                "cumulative_calendar_years": len(seen_years),
                "training_sessions": int(len(training)),
                "selection_window_start": (
                    scored.index[0].date().isoformat() if len(scored) else "none"
                ),
                "selection_window_end": (
                    scored.index[-1].date().isoformat() if len(scored) else "none"
                ),
                "selection_sessions": int(len(scored)),
                # Both of these are annualized, and both column names say so.
                # ``validation._select_highest_sharpe`` returns a *per-session*
                # Sharpe, and the standard error sqrt(annualization / n) is the
                # standard error of an *annualized* one, so recording the two
                # side by side untouched put a factor of sqrt(252) between the
                # only pair of numbers a reader has for judging whether the
                # selector measured signal or noise.  The score is annualized
                # here, at the point of recording, which is a relabelling and
                # not a change of selection: the transform is monotone, so the
                # candidate chosen is identical.
                "selection_standard_error_of_annualized_sharpe": (
                    float(math.sqrt(config.annualization / len(scored)))
                    if len(scored)
                    else float("nan")
                ),
                "purge_sessions": int(plan.purge_sessions),
                "embargo_sessions": int(plan.embargo_sessions),
                "universe_members": int(membership.loc[fold_index[0]].sum()),
                "selected_candidate": choice,
                "selection_score_annualized_sharpe": (
                    float(score) * math.sqrt(config.annualization)
                ),
                "selection_switched": bool(choice != previous),
                "oos_basis": (
                    SEALED_BASIS if fold_index[0] >= sealed_start else SELECTOR_BASIS
                ),
                "hac_convention": _HAC_CONVENTION,
                "selection_adjusted": False,
                "permitted_use": (
                    SEALED_PERMITTED_USE
                    if fold_index[0] >= sealed_start
                    else permitted_use
                ),
                "panel_limitation": "; ".join(ALLOCATION_LIMITATIONS),
                "_window": window,
            }
        )
        previous = choice

    # The ancillary series are spliced alongside the weights so the returned
    # object describes the path that was actually simulated.  Carrying the
    # anchor candidate's risk weights and volatility scale on a stitched result
    # would silently mislabel every fold after the first switch.
    stitched_desired = DesiredWeights(
        candidate=label,
        weights=spliced,
        risk_weights=spliced_risk,
        volatility_scale=spliced_scale,
        membership=membership,
        eligibility=spliced_eligibility,
        unscaled_book_return=wanted[names[0]].unscaled_book_return,
        first_live_session=wanted[names[0]].first_live_session,
    )
    stitched = simulate_allocation(stitched_desired, inputs, config, label=label)
    net = stitched.net_return

    for record in records:
        window = record.pop("_window")
        segment = net.loc[window]
        daily = stitched.daily.loc[window]
        # Anchoring each fold on the session before it makes the folds' elapsed
        # periods tile the stitched span, so the per-fold CAGRs compound to the
        # stitched CAGR instead of each dropping its own first day.
        statistics = path_statistics(
            segment,
            annualization=config.annualization,
            hac_lags=config.hac_lags,
            period_start=_previous_session(net.index, segment.index[0]),
        )
        years = max(len(segment) / config.annualization, 1e-9)
        record["fold_cagr"] = statistics["cagr"]
        record["fold_annualized_volatility"] = statistics["annualized_volatility"]
        record["fold_sharpe"] = statistics["sharpe"]
        record["fold_hac_sharpe"] = statistics["hac_sharpe"]
        record["fold_max_drawdown"] = statistics["max_drawdown"]
        record["fold_hit_rate"] = statistics["hit_rate"]
        record["fold_annual_one_way_turnover"] = float(
            daily["turnover_fraction"].sum() / years
        )
        record["fold_annual_cost_drag"] = float(
            daily["cost_return"].mean() * config.annualization
        )

    folds = pd.DataFrame(records, columns=list(FOLD_COLUMNS))
    anchor_ledger = ledgers[names[0]]

    window = (net.index >= stamps[0]) & (net.index <= end)
    stitched_statistics = path_statistics(
        net.loc[window],
        annualization=config.annualization,
        hac_lags=config.hac_lags,
        period_start=_previous_session(net.index, net.loc[window].index[0]),
    )
    stitched_frame = stitched.daily.loc[window]
    anchor_frame = anchor_ledger.daily.loc[window]
    years = max(int(window.sum()) / config.annualization, 1e-9)
    summary = pd.Series(
        {
            "label": label,
            "mode": mode,
            "lineage_id": plan.lineage_id,
            "anchor": anchor.date().isoformat(),
            "folds": int(len(folds)),
            "candidates": int(len(members)),
            "stitched_start": net.loc[window].index[0].date().isoformat(),
            "stitched_end": net.loc[window].index[-1].date().isoformat(),
            "stitched_sessions": int(window.sum()),
            "stitched_calendar_years": _calendar_years(net.loc[window].index),
            "stitched_cagr": stitched_statistics["cagr"],
            "stitched_annualized_volatility": stitched_statistics[
                "annualized_volatility"
            ],
            "stitched_sharpe": stitched_statistics["sharpe"],
            "stitched_hac_sharpe": stitched_statistics["hac_sharpe"],
            "stitched_sortino": stitched_statistics["sortino"],
            "stitched_max_drawdown": stitched_statistics["max_drawdown"],
            "stitched_calmar": stitched_statistics["calmar"],
            "selection_switches": int(folds["selection_switched"].sum()),
            "distinct_candidates_selected": int(folds["selected_candidate"].nunique()),
            "stitched_annual_one_way_turnover": float(
                stitched_frame["turnover_fraction"].sum() / years
            ),
            "anchor_annual_one_way_turnover": float(
                anchor_frame["turnover_fraction"].sum() / years
            ),
            "stitched_annual_cost_drag": float(
                stitched_frame["cost_return"].mean() * config.annualization
            ),
            "anchor_annual_cost_drag": float(
                anchor_frame["cost_return"].mean() * config.annualization
            ),
            "hac_convention": _HAC_CONVENTION,
            "selection_adjusted": False,
            "permitted_use": permitted_use,
            "panel_limitation": "; ".join(ALLOCATION_LIMITATIONS),
        }
    )
    return WalkForwardResult(
        label=label,
        mode=mode,
        plan=plan,
        folds=folds,
        stitched=stitched,
        anchor_ledger=anchor_ledger,
        candidate_ledgers=ledgers,
        spliced=stitched_desired,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# out-of-sample accounting
# ---------------------------------------------------------------------------


def out_of_sample_accounting(result: WalkForwardResult) -> pd.DataFrame:
    """The arithmetic a reviewer checks in under a minute, without rerunning.

    Three units on every row, primary first: complete calendar years, sessions,
    and elapsed days divided by 365.2425.  Calendar years lead because that is a
    counting argument nobody can argue with, whereas a day-count claim depends on
    a divisor convention and the same five-year window measures 4.99 years by one
    convention and 5.02 by another.  Publishing all three, and showing they agree
    to within the convention, is what makes the claim auditable rather than
    assertable.

    Three properties are asserted here rather than described, because "no
    double-counting" is either a proven property or a hope:

    * every pair of fold test windows is disjoint;
    * their union is exactly the stitched out-of-sample index; and
    * the sum of the per-fold session counts equals the number of distinct
      sessions in that index.

    Note that fold count and out-of-sample time are independent quantities.
    Halving the boundary spacing doubles the fold count and leaves all three
    units unchanged, so a write-up that reports a fold count next to a five-year
    requirement is inviting the reader to make an error.
    """

    folds = result.folds
    sessions = result.stitched.net_return.index
    stamps = result.plan.boundary_stamps
    end = pd.Timestamp(result.plan.end)
    sealed_start = pd.Timestamp(result.plan.sealed_start)

    windows: list[pd.DatetimeIndex] = []
    for position, boundary in enumerate(stamps):
        final = position + 1 == len(stamps)
        stop = end if final else stamps[position + 1]
        mask = (sessions >= boundary) & (
            (sessions <= stop) if final else (sessions < stop)
        )
        windows.append(sessions[mask])

    for left in range(len(windows)):
        for right in range(left + 1, len(windows)):
            overlap = windows[left].intersection(windows[right])
            if len(overlap):
                raise ValueError(
                    f"folds {left} and {right} share {len(overlap)} sessions; the "
                    "out-of-sample count would double-count them"
                )
    union: pd.DatetimeIndex = windows[0]
    for extra in windows[1:]:
        union = union.union(extra)
    stitched_index = sessions[(sessions >= stamps[0]) & (sessions <= end)]
    if not union.equals(stitched_index):
        raise ValueError(
            "the union of the fold test windows is not the stitched out-of-sample "
            "index; one of them contains a session the other does not"
        )
    if int(folds["sessions"].sum()) != len(stitched_index):
        raise ValueError(
            f"the fold session counts sum to {int(folds['sessions'].sum())} against "
            f"{len(stitched_index)} distinct stitched sessions"
        )

    def block(index: pd.DatetimeIndex, name: str, basis: str, use: str) -> dict[str, object]:
        if len(index) == 0:
            raise ValueError(f"{name} holds no sessions")
        elapsed = (index[-1] - index[0]).days
        covered = index.intersection(union)
        return {
            "Claim": name,
            "Basis": basis,
            "First session": index[0].date().isoformat(),
            "Last session": index[-1].date().isoformat(),
            "Complete calendar years": _complete_calendar_years(index, sessions),
            "Calendar years touched": _calendar_years(index),
            "Distinct sessions": int(len(index)),
            "Elapsed days": int(elapsed),
            "Elapsed years (365.2425)": float(elapsed / 365.2425),
            "Years at 252 sessions": float(len(index) / 252.0),
            "Folds": int(
                (
                    (pd.to_datetime(folds["test_start"]) >= index[0])
                    & (pd.to_datetime(folds["test_end"]) <= index[-1])
                ).sum()
            ),
            # Measured against the sessions of this block that any fold covers,
            # not against the block's own length: the training span is covered
            # by no fold at all, and subtracting its length would report a
            # nonsense negative rather than the zero it deserves.
            "Sessions double counted": int(
                sum(len(window.intersection(index)) for window in windows)
                - len(covered)
            ),
            "HAC convention": _HAC_CONVENTION,
            "Selection adjusted": False,
            "Permitted use": use,
            "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
        }

    development = stitched_index[stitched_index < sealed_start]
    sealed = stitched_index[stitched_index >= sealed_start]
    training = sessions[(sessions >= pd.Timestamp(result.plan.anchor)) & (sessions < stamps[0])]

    rows = [
        # Named for its endpoints rather than as "the training span".  The
        # degradation table in ``in_sample_out_of_sample_report`` measures its
        # in-sample column from the first session on which the book actually
        # holds risk, which is later than the anchor, so two artifacts written
        # side by side would otherwise share a phrase while describing windows
        # of different length --- and a reader joining them on that phrase would
        # take the wrong denominator for the sleeve's most important table.
        block(
            training,
            "In-sample span, anchor to first boundary",
            IN_SAMPLE_BASIS,
            "reused history the specification was written with; not evidence",
        ),
        block(
            development,
            "Rolling walk-forward, pre-sealed block",
            SELECTOR_BASIS,
            SELECTOR_PERMITTED_USE,
        ),
        block(
            sealed,
            "Sealed contiguous block",
            SEALED_BASIS,
            SEALED_PERMITTED_USE,
        ),
        block(
            stitched_index,
            "Combined out-of-sample record",
            SELECTOR_BASIS,
            SELECTOR_PERMITTED_USE,
        ),
    ]
    return pd.DataFrame(rows)


def fold_dispersion_report(folds: pd.DataFrame) -> pd.DataFrame:
    """Per-fold dispersion of every statistic the fold table carries.

    A stitched Sharpe is one number; twelve fold Sharpes with a standard
    deviation twice their mean is a different and more honest object.  Dispersion
    is reported for every statistic rather than only for the flattering ones.
    """

    if not isinstance(folds, pd.DataFrame) or folds.empty:
        raise ValueError("folds must be a non-empty DataFrame")
    columns = [
        column
        for column in folds.columns
        if column.startswith("fold_") and pd.api.types.is_numeric_dtype(folds[column])
    ]
    rows = []
    for column in columns:
        values = folds[column].astype(float).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Statistic": column,
                "Folds": int(len(values)),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Standard deviation": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "Minimum": float(values.min()),
                "Maximum": float(values.max()),
                "Folds above zero": int((values > 0).sum()),
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# in sample against out of sample, and the references
# ---------------------------------------------------------------------------


#: The in-sample column of the degradation table.  It is deliberately NOT
#: called "training span": it starts at the first session on which the book
#: actually holds risk, which is later than the anchor, so it is a shorter
#: window than the "In-sample span, anchor to first boundary" row of
#: ``out_of_sample_accounting``.  Two artifacts written into the same directory
#: sharing a phrase while describing windows of different length is how a
#: degradation figure gets quoted against the wrong denominator.
IN_SAMPLE_DEGRADATION_WINDOW = "In sample (first live session to first boundary)"

#: Rows that describe each window rather than score it.  They lead the frame so
#: the artifact is self-describing without a join against anything else.
_WINDOW_DESCRIPTORS: tuple[tuple[str, Callable[[pd.Series], str]], ...] = (
    ("First session", lambda s: s.index[0].date().isoformat() if len(s) else "none"),
    ("Last session", lambda s: s.index[-1].date().isoformat() if len(s) else "none"),
    ("Sessions", lambda s: str(int(len(s)))),
)

_DEGRADATION_STATISTICS = (
    ("CAGR", "cagr"),
    ("Annualized volatility", "annualized_volatility"),
    ("Sharpe (rf=0)", "sharpe"),
    ("HAC Sharpe", "hac_sharpe"),
    ("Sortino", "sortino"),
    ("Max drawdown", "max_drawdown"),
    ("Calmar", "calmar"),
    ("Hit rate", "hit_rate"),
)


def in_sample_out_of_sample_report(
    result: WalkForwardResult,
    config: AllocationConfig,
) -> pd.DataFrame:
    """The single most informative table this sleeve produces.

    A strategy's in-sample statistics are what the specification was written to
    produce; its out-of-sample statistics are what survived contact with a
    window the selector never saw.  The difference between them is the estimate
    of how much of the in-sample result was fitting.  An honest large
    degradation is a better deliverable than a flattering small one, so the
    change is reported for every statistic with no filtering and no commentary
    column.

    One property of the in-sample window is worth naming rather than leaving to
    be inferred.  No selection has happened before the first boundary, so the
    spliced frame there is the *first declared candidate's* frame unchanged.
    The in-sample column therefore describes the frozen anchor specification,
    not a selected one, and the degradation it anchors is degradation from a
    frozen rule's own training span rather than from a fitted optimum.  That
    makes the comparison conservative: a genuinely fitted in-sample number
    would be higher and the measured drop larger.

    The window also starts at the first session on which the book actually holds
    risk, not at the anchor.  Including the warm-up months, when the book is
    entirely in the cash sleeve by construction, would measure the short-duration
    Treasury fund and call the result in-sample performance.

    That choice makes this table's in-sample window *shorter* than the
    "In-sample span, anchor to first boundary" row of
    ``out_of_sample_accounting``, and the two are written into the same
    directory by the same script.  So each window states its own first session,
    last session and session count in the first three rows, and the in-sample
    column is named for its start rather than for the phrase it shares with the
    accounting table.  A degradation figure quoted against the wrong denominator
    is a worse error than no degradation figure at all.
    """

    net = result.stitched.net_return
    anchor = pd.Timestamp(result.plan.anchor)
    first_boundary = result.plan.boundary_stamps[0]
    sealed_start = pd.Timestamp(result.plan.sealed_start)
    end = pd.Timestamp(result.plan.end)

    live = result.stitched.daily["book_live"].to_numpy(dtype=bool)
    live_index = net.index[live]
    if len(live_index) == 0:
        raise ValueError("the book was never live; there is nothing to compare")
    training_start = max(anchor, live_index[0])

    in_sample_name = IN_SAMPLE_DEGRADATION_WINDOW
    windows = {
        in_sample_name: net.loc[
            (net.index >= training_start) & (net.index < first_boundary)
        ],
        "Out of sample (rolling, pre-sealed)": net.loc[
            (net.index >= first_boundary) & (net.index < sealed_start)
        ],
        "Out of sample (sealed block)": net.loc[
            (net.index >= sealed_start) & (net.index <= end)
        ],
        "Out of sample (combined)": net.loc[
            (net.index >= first_boundary) & (net.index <= end)
        ],
    }
    statistics = {
        name: path_statistics(
            series,
            annualization=config.annualization,
            hac_lags=config.hac_lags,
            period_start=_previous_session(net.index, series.index[0]),
        )
        for name, series in windows.items()
        if len(series) > config.hac_lags
    }
    base = statistics[in_sample_name]
    rows: list[dict[str, object]] = []
    # The window bounds lead the table.  Without them the frame publishes four
    # unnamed windows and a reader has to infer their extent from the source,
    # which is exactly what makes a degradation figure quotable against the
    # wrong denominator.
    for display, describe in _WINDOW_DESCRIPTORS:
        row = {"Statistic": display}
        for name, series in windows.items():
            row[name] = describe(series)
        row["Change, in sample to combined out of sample"] = ""
        row["Change, in sample to sealed block"] = ""
        row["Limitation"] = "; ".join(ALLOCATION_LIMITATIONS)
        rows.append(row)
    for display, key in _DEGRADATION_STATISTICS:
        row = {"Statistic": display}
        for name in windows:
            row[name] = statistics.get(name, {}).get(key, float("nan"))
        row["Change, in sample to combined out of sample"] = (
            statistics.get("Out of sample (combined)", {}).get(key, float("nan"))
            - base.get(key, float("nan"))
        )
        row["Change, in sample to sealed block"] = (
            statistics.get("Out of sample (sealed block)", {}).get(key, float("nan"))
            - base.get(key, float("nan"))
        )
        row["Limitation"] = "; ".join(ALLOCATION_LIMITATIONS)
        rows.append(row)
    return pd.DataFrame(rows)


def reference_ledgers(
    inputs: AllocationInputs,
    config: AllocationConfig,
) -> dict[str, AllocationLedger]:
    """Doing nothing, three ways, on the same universe and the same costs.

    One deliberate difference from the allocator: the references are simulated
    with the no-trade band switched off, and that is what makes them references
    rather than three more strategies.

    The band is a rebalancing device.  Applied to a book being built from flat
    it is also an *entry* device, and it stops the first purchase a band-width
    short on every leg.  A "60/40" reference given a single trade session then
    buys 58% SPY and 38% IEF, leaves 4% sitting in the short-duration Treasury
    sleeve, and --- because it never trades again --- holds that stub for the
    whole replay.  The published path would not be a 60/40 at any point in its
    life, and the error runs in the direction that flatters the sleeve being
    measured against it, worth roughly 0.20 percentage points a year here.  The
    equal-weight panel is distorted the same way, holding 10.3% cash against
    9.8% SPY where an equal weight is 9.1% of each.

    So the references keep every cost the allocator pays --- the same spread,
    commission, square-root impact and participation cap, which is what makes
    the comparison like for like on execution --- and drop only the band, which
    is what makes each of them the portfolio its name claims.
    """

    reference_config = replace(config, no_trade_band=0.0)
    monthly = fixed_weight_frame(
        inputs,
        reference_config,
        SIXTY_FORTY_WEIGHTS,
        label="sixty_forty_rebalanced_monthly",
    )
    drifting = fixed_weight_frame(
        inputs,
        reference_config,
        SIXTY_FORTY_WEIGHTS,
        label="sixty_forty_buy_and_hold",
    )
    equal = equal_weight_reference(inputs, reference_config)

    # "Buy and hold" means one purchase and then nothing, so the single trading
    # session is the first rebalance date on which the reference actually has
    # something to buy.  Taking the calendar's first rebalance date instead
    # would buy nothing at all --- the funds are not yet admissible under the
    # causal membership rule --- and the reference would silently be a cash
    # account wearing a 60/40 label.
    trading = rebalance_sessions(inputs.panel.sessions)
    fundable = drifting.risk_weights.sum(axis=1).reindex(trading).fillna(0.0)
    live = trading[(fundable > 0.0).to_numpy()]
    first_trade = live[:1] if len(live) else trading[:1]
    return {
        monthly.candidate: simulate_allocation(
            monthly, inputs, reference_config, label=monthly.candidate
        ),
        drifting.candidate: simulate_allocation(
            drifting,
            inputs,
            reference_config,
            label=drifting.candidate,
            trade_sessions=first_trade,
        ),
        equal.candidate: simulate_allocation(
            equal, inputs, reference_config, label=equal.candidate
        ),
    }


def comparison_report(
    ledgers: dict[str, AllocationLedger],
    *,
    window: str,
    basis: str,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    permitted_use: str = SELECTOR_PERMITTED_USE,
) -> pd.DataFrame:
    """Several paths over one window, in the declared order they were given.

    No verdict column, no ranking, no winner.  The rows are in the order the
    caller supplied them, which is the declaration order, so a reader cannot
    mistake the table's layout for a judgement about which path is better.
    """

    frames = [
        path_report(
            ledger,
            window=window,
            basis=basis,
            start=start,
            end=end,
            permitted_use=permitted_use,
            label=name,
        )
        for name, ledger in ledgers.items()
    ]
    return pd.concat(frames, ignore_index=True)


def straddling_episode_report(
    ledgers: dict[str, AllocationLedger],
    plan: WalkForwardPlan,
    *,
    episode: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> pd.DataFrame:
    """An episode that crosses the first boundary, labelled for what it is.

    The only full equity bear market this panel holds runs from mid-2007 into
    mid-2009, and the first walk-forward boundary falls in the middle of it.
    Neither available basis is true of the whole episode: a quarter of its
    sessions sit inside the first out-of-sample fold, so stamping it
    ``IN_SAMPLE_BASIS`` overstates how much of it the specification was written
    with, and stamping it ``SELECTOR_BASIS`` would be far worse.  Truncating the
    window at the boundary would make one label true and cut away the trough,
    which is the part of the episode worth reading.

    So the episode is published whole under ``MIXED_BASIS``, with the split
    counted from the session calendar rather than asserted in prose, and the two
    halves follow beneath it under their own honest bases.  A reader who wants
    the episode gets the episode; a reader who wants a clean basis gets two.

    Raises rather than mislabels if the window does not in fact straddle: a
    window wholly on one side of the boundary should use ``comparison_report``
    with the basis that is true of it.
    """

    if not ledgers:
        raise ValueError("at least one path is required")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first >= last:
        raise ValueError("the episode start must precede its end")
    boundary = plan.boundary_stamps[0]
    sessions = next(iter(ledgers.values())).daily.index
    covered = sessions[(sessions >= first) & (sessions <= last)]
    before = covered[covered < boundary]
    after = covered[covered >= boundary]
    if len(before) == 0 or len(after) == 0:
        raise ValueError(
            f"the window {first.date()}..{last.date()} does not straddle the first "
            f"boundary {boundary.date()}; use comparison_report with the basis "
            "that is true of it"
        )

    whole = comparison_report(
        ledgers,
        window=f"{episode} {first.date()}..{last.date()}, whole episode",
        basis=MIXED_BASIS,
        start=first,
        end=last,
        permitted_use=(
            f"{episode}, and it straddles the first boundary "
            f"{boundary.date()}: {len(before)} of its {len(covered)} sessions "
            f"lie in the training span and {len(after)} fall inside the first "
            "out-of-sample fold; a description of one path across a boundary "
            "and not evidence for either claim on its own"
        ),
    )
    training_part = comparison_report(
        ledgers,
        window=f"{episode}, in-sample part {first.date()}..{before[-1].date()}",
        basis=IN_SAMPLE_BASIS,
        start=first,
        end=before[-1],
        permitted_use=(
            "the part of the episode inside the training span; reused history "
            "the specification was written with, and not evidence"
        ),
    )
    fold_part = comparison_report(
        ledgers,
        window=f"{episode}, first-fold part {after[0].date()}..{last.date()}",
        basis=SELECTOR_BASIS,
        start=after[0],
        end=last,
        permitted_use=SELECTOR_PERMITTED_USE,
    )
    return pd.concat([whole, training_part, fold_part], ignore_index=True)


def risk_matched_comparison(
    ledgers: dict[str, AllocationLedger],
    *,
    reference_label: str,
    window: str,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    annualization: int = 252,
) -> pd.DataFrame:
    """Every path rescaled to one realized volatility, so returns are comparable.

    An unlevered long-only book of ten diversified sleeves with a trend gate
    runs at a lower volatility than a 60/40, and comparing their raw CAGRs
    rewards the one carrying more risk.  Each path is therefore uniformly
    rescaled to the reference path's realized volatility using
    ``attribution.uniform_rescale_frontier``, which is the repository's existing
    statement of what a constant de-lever buys and already carries the caveat
    that a uniform rescale slightly flatters a real one, because fixed costs do
    not shrink with the book.

    ``Drawdown per unit of volatility`` is the invariant a constant rescale
    leaves unchanged, so it is the column that distinguishes a genuine risk
    improvement from a smaller allocation.  A path that improves it has done
    something a de-lever cannot do; a path that improves only ``Max drawdown``
    has done something the committee can do with one number.
    """

    from delta1_strategy.research.attribution import uniform_rescale_frontier

    if reference_label not in ledgers:
        raise ValueError(f"{reference_label!r} is not among the supplied paths")

    def slice_returns(ledger: AllocationLedger) -> pd.Series:
        series = ledger.net_return
        if start is not None:
            series = series.loc[series.index >= pd.Timestamp(start)]
        if end is not None:
            series = series.loc[series.index <= pd.Timestamp(end)]
        return series.dropna().astype(float)

    reference = slice_returns(ledgers[reference_label])
    if reference.empty:
        raise ValueError(f"{reference_label}: the requested window holds no sessions")
    matched_volatility = float(reference.std(ddof=1) * math.sqrt(annualization))
    if matched_volatility <= 0.0:
        raise ValueError("the reference path has no volatility to match")

    rows = []
    for label, ledger in ledgers.items():
        series = slice_returns(ledger)
        volatility = float(series.std(ddof=1) * math.sqrt(annualization))
        if volatility <= 0.0:
            continue
        multiplier = matched_volatility / volatility
        frontier = uniform_rescale_frontier(
            series, multipliers=(multiplier,), annualization=annualization
        ).iloc[0]
        rows.append(
            {
                "Path": label,
                "Window": window,
                "Sessions": int(len(series)),
                "Realized volatility": volatility,
                "Volatility matched to": matched_volatility,
                "Exposure multiplier applied": float(multiplier),
                "CAGR at matched volatility": float(frontier["CAGR"]),
                "Max drawdown at matched volatility": float(frontier["Max drawdown"]),
                "Calmar at matched volatility": float(frontier["Calmar"]),
                "Sharpe (rf=0)": float(frontier["Sharpe (rf=0)"]),
                "Drawdown per unit of volatility": float(
                    frontier["Drawdown per unit of volatility"]
                ),
                "Matching reference": reference_label,
                "Selection adjusted": False,
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    return pd.DataFrame(rows)


def exposure_share_report(
    ledger: AllocationLedger,
    inputs: AllocationInputs,
    *,
    window: str = "full sample",
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Realized weight share and realized risk share, by asset class.

    Concentration by construction is the failure mode an equal budget hides and
    an inverse-volatility budget makes worse.  Six of the eleven sleeves are
    fixed income, so an equal budget is already duration-heavy before any state
    is applied; weighting inversely to volatility then multiplies the tilt,
    because a Treasury fund at 1% volatility attracts many times the weight of
    an equity fund at 16%.

    Weight share alone understates the problem, which is why risk share is
    reported beside it.  Risk share is the exact decomposition of realized
    portfolio variance, ``w_i cov(r_i, r_p) / var(r_p)`` aggregated by class, so
    the two columns together show a book that can be three-quarters bonds by
    weight and half equities by risk.  Neither number is adjusted after being
    seen; they are reported so the tilt is disclosed, not corrected.
    """

    daily = ledger.daily
    weights = ledger.weights
    returns = inputs.panel.returns.reindex(weights.index)
    # The book that earned session ``t`` is the post-trade book of ``t-1``, so
    # the shift is taken on the whole path before the window is cut.  Cutting
    # first would zero the first session's holdings while the ledger's own gross
    # return for that session still reflects them, and the variance shares would
    # no longer sum to one.
    held = weights.shift(1)
    for stamp, comparison in (
        (start, "ge"),
        (end, "le"),
    ):
        if stamp is None:
            continue
        cut = pd.Timestamp(stamp)
        keep = daily.index >= cut if comparison == "ge" else daily.index <= cut
        daily, weights, returns, held = (
            daily.loc[keep],
            weights.loc[keep],
            returns.loc[keep],
            held.loc[keep],
        )
    if daily.empty:
        raise ValueError(f"{ledger.label}: the requested window holds no sessions")

    portfolio = daily["gross_return"].astype(float)
    variance = float(portfolio.var(ddof=0))
    held = held.fillna(0.0)
    contributions = {}
    for ticker in weights.columns:
        sleeve = (held[ticker] * returns[ticker]).astype(float)
        covariance = float(np.cov(sleeve, portfolio, ddof=0)[0, 1])
        contributions[ticker] = covariance / variance if variance > 0 else float("nan")

    rows = []
    classes = sorted({etfs.ASSET_CLASS.get(t, "unclassified") for t in weights.columns})
    for asset_class in classes:
        members = [
            t for t in weights.columns if etfs.ASSET_CLASS.get(t, "unclassified") == asset_class
        ]
        rows.append(
            {
                "Asset class": asset_class,
                "Window": window,
                "Sleeves": ", ".join(members),
                "Mean weight share": float(weights[members].sum(axis=1).mean()),
                "Maximum weight share": float(weights[members].sum(axis=1).max()),
                "Realized risk share": float(
                    sum(contributions[t] for t in members)
                ),
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    for ticker in weights.columns:
        rows.append(
            {
                "Asset class": f"sleeve: {ticker}",
                "Window": window,
                "Sleeves": ticker,
                "Mean weight share": float(weights[ticker].mean()),
                "Maximum weight share": float(weights[ticker].max()),
                "Realized risk share": float(contributions[ticker]),
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    return pd.DataFrame(rows)


def volatility_budget_diagnostic(
    ledgers: dict[str, AllocationLedger],
    config: AllocationConfig,
    *,
    scales: dict[str, pd.Series] | None = None,
    window: str = "full sample",
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Whether the declared volatility budget binds, and in which direction.

    The budget is 7% annualized, chosen so the sleeve is comparable to the
    incumbent futures CTA at the same ex-ante risk.  It is enforced through a
    scalar capped at one, because the book is unlevered, and that cap has a
    consequence which must be reported rather than discovered: a *ceiling*
    cannot raise volatility.  If the unlevered diversified book's own volatility
    sits below the budget, the budget never binds, the scalar sits at its
    ceiling, and the sleeve runs below its stated risk.  Reporting the realized
    volatility beside the budget, and the share of sessions spent at the
    ceiling, is what turns that from a silent shortfall into a stated one.
    """

    rows = []
    for label, ledger in ledgers.items():
        daily = ledger.daily
        if start is not None:
            daily = daily.loc[daily.index >= pd.Timestamp(start)]
        if end is not None:
            daily = daily.loc[daily.index <= pd.Timestamp(end)]
        if daily.empty:
            continue
        returns = daily["net_return"].astype(float)
        volatility = float(returns.std(ddof=1) * math.sqrt(config.annualization))
        scale = None if scales is None else scales.get(label)
        if scale is not None:
            scale = scale.reindex(daily.index).astype(float)
        rows.append(
            {
                "Path": label,
                "Window": window,
                "Sessions": int(len(daily)),
                "Volatility budget (annualized)": float(
                    config.volatility_budget_annualized
                ),
                "Realized volatility (annualized)": volatility,
                "Realized over budget": volatility
                / float(config.volatility_budget_annualized),
                "Mean volatility scale": (
                    float(scale.mean()) if scale is not None else float("nan")
                ),
                "Share of sessions at the scale ceiling": (
                    float(
                        (scale >= config.max_volatility_scale - 1e-12).mean()
                    )
                    if scale is not None
                    else float("nan")
                ),
                "Share of sessions with no risk position": float(
                    (daily["risk_weight_sum"] <= 1e-9).mean()
                ),
                "Mean risk weight": float(daily["risk_weight_sum"].mean()),
                "Mean cash weight": float(daily["cash_weight"].mean()),
                "Share of sessions with a deferred order": float(
                    daily["order_deferred"].mean()
                ),
                "Minimum participation fill fraction": float(
                    daily["participation_fill_fraction"].min()
                ),
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    return pd.DataFrame(rows)


def paired_reference_inference(
    ledgers: dict[str, AllocationLedger],
    *,
    subject_label: str,
    reference_labels: Sequence[str],
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    annualization: int = 252,
    hac_lags: int = 21,
) -> pd.DataFrame:
    """Is the gap to a reference larger than the noise in measuring it?

    Reuses ``inference.paired_hac_report`` and ``inference.minimum_detectable
    _effect`` unchanged.  Pairing is the whole source of power: one shared
    market history cancels in the differential, so the interval describes the
    part of the variation the design choice itself caused rather than the part
    the decade caused.

    Every interval here is a sampling-uncertainty interval and nothing more.
    ``Selection adjusted`` is False on every row, exactly as it is throughout
    ``research.inference`` and ``research.diagnostics``, and no row licenses a
    promotion or a rejection.  A gap whose confidence bound spans zero is a gap
    this sample cannot resolve, which is a statement about the sample.
    """

    from delta1_strategy.research.inference import (
        minimum_detectable_effect,
        paired_hac_report,
    )

    if subject_label not in ledgers:
        raise ValueError(f"{subject_label!r} is not among the supplied paths")

    def slice_returns(label: str) -> pd.Series:
        series = ledgers[label].net_return
        if start is not None:
            series = series.loc[series.index >= pd.Timestamp(start)]
        if end is not None:
            series = series.loc[series.index <= pd.Timestamp(end)]
        return series.dropna().astype(float)

    subject = slice_returns(subject_label)
    rows = []
    for label in reference_labels:
        if label not in ledgers or label == subject_label:
            continue
        reference = slice_returns(label)
        shared = subject.index.intersection(reference.index)
        left = reference.loc[shared]
        right = subject.loc[shared]
        report = paired_hac_report(
            left,
            right,
            comparison=f"{subject_label} minus {label}",
            lags=hac_lags,
            annualization=annualization,
        )
        differential = (right - left).to_numpy(dtype=float)
        tracking = float(differential.std(ddof=0) * math.sqrt(annualization))
        volatility = float(left.std(ddof=0) * math.sqrt(annualization))
        years = len(shared) / annualization
        power = minimum_detectable_effect(
            max(tracking, 1e-12), max(volatility, 1e-12), max(years, 1e-9)
        )
        rows.append(
            {
                "Comparison": report["Comparison"],
                "Sessions": int(len(shared)),
                "Method": report["Method"],
                "Annualized mean differential": float(report["Point estimate"]),
                "Standard error (annualized)": float(
                    report["Standard error (annualized)"]
                ),
                "t statistic": float(report["t statistic"]),
                "Confidence lower (95% one-sided)": float(report["Confidence lower"]),
                "Confidence upper (95% one-sided)": float(report["Confidence upper"]),
                "Tracking error (annualized)": tracking,
                "Return correlation": float(np.corrcoef(left, right)[0, 1]),
                "Minimum detectable Sharpe effect": float(
                    power["Minimum detectable effect (95% one-sided)"]
                ),
                "Selection adjusted": False,
                "Permitted use": report["Permitted use"],
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cost sensitivity and custody
# ---------------------------------------------------------------------------

DEFAULT_COST_MULTIPLIERS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)


def cost_sensitivity_report(
    inputs: AllocationInputs,
    config: AllocationConfig,
    plan: WalkForwardPlan,
    *,
    multipliers: Sequence[float] = DEFAULT_COST_MULTIPLIERS,
    candidates: Sequence[AllocationCandidate] = DECLARED_CANDIDATES,
) -> pd.DataFrame:
    """Rerun the whole walk-forward at each cost assumption.

    Sweeping the cost multiplier is not the same kind of act as sweeping a
    strategy parameter.  A strategy parameter swept over five values is five
    trials under ``docs/research-methodology.md``; a cost multiplier swept over
    five values is one strategy priced under five assumptions, and the reason to
    do it is that the assumption is not a measurement.  The promotion gate that
    matters is the one at twice the declared level, following the
    ``friction.run_friction_stress_suite`` pattern.

    The selector re-runs at each level, so the selected candidate may change
    with costs.  That is realistic --- a live desk selecting on net returns would
    see the same thing --- and the selected sequence is reported so the reader
    can see whether it did.

    Two columns report the sealed block and they are different objects, so both
    are published side by side rather than one standing in for the other.
    ``Sealed-block Sharpe (rf=0)`` is the Sharpe of the sealed block's own
    return path, computed the same way and on the same window as the sealed
    comparison table.  ``Mean sealed fold Sharpe (rf=0)`` is the arithmetic mean
    of the annual fold Sharpes.  They are not close: a mean of per-year Sharpes
    weights a quiet year with a small numerator equally with a volatile one, and
    on the supplied run it reads 0.58 against a block Sharpe of 0.39.  This
    sweep is the table a reviewer uses to decide whether the sealed result
    survives realistic costs, so the block statistic has to be the one carrying
    the block's name.
    """

    values = [float(value) for value in multipliers]
    if not values:
        raise ValueError("at least one cost multiplier is required")
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("cost multipliers must be finite and non-negative")

    rows = []
    for multiplier in values:
        scenario = config.with_cost_multiplier(multiplier)
        result = run_walk_forward(
            inputs,
            scenario,
            plan,
            candidates=candidates,
            label=f"walk_forward_costs_x{multiplier:g}",
        )
        summary = result.summary
        sealed = result.folds.loc[result.folds["oos_basis"] == SEALED_BASIS]
        # The block statistic, read off the block's own return path over the
        # same window the sealed comparison table uses, not a mean of the annual
        # fold Sharpes.  A mean of per-year Sharpes is a different quantity and
        # it is biased upward whenever the annual means disperse.
        net = result.stitched.net_return
        sealed_window = net.loc[
            (net.index >= pd.Timestamp(plan.sealed_start))
            & (net.index <= pd.Timestamp(plan.end))
        ]
        block_sharpe = float("nan")
        block_sessions = int(len(sealed_window))
        if block_sessions > scenario.hac_lags:
            block_sharpe = float(
                path_statistics(
                    sealed_window,
                    annualization=scenario.annualization,
                    hac_lags=scenario.hac_lags,
                    period_start=_previous_session(net.index, sealed_window.index[0]),
                )["sharpe"]
            )
        rows.append(
            {
                "Cost multiplier": multiplier,
                "One-way fixed cost (bps)": float(
                    scenario.costs.fixed_cost_rate * 10_000.0
                ),
                "Impact coefficient (bps)": float(
                    scenario.costs.impact_coefficient_bps * multiplier
                ),
                "Out-of-sample CAGR": float(summary["stitched_cagr"]),
                "Out-of-sample volatility": float(
                    summary["stitched_annualized_volatility"]
                ),
                "Out-of-sample Sharpe (rf=0)": float(summary["stitched_sharpe"]),
                "Out-of-sample HAC Sharpe": float(summary["stitched_hac_sharpe"]),
                "Out-of-sample max drawdown": float(summary["stitched_max_drawdown"]),
                "Annual one-way turnover": float(
                    summary["stitched_annual_one_way_turnover"]
                ),
                "Annual cost drag": float(summary["stitched_annual_cost_drag"]),
                "Sealed-block sessions": block_sessions,
                "Sealed-block Sharpe (rf=0)": block_sharpe,
                "Mean sealed fold Sharpe (rf=0)": float(
                    sealed["fold_sharpe"].mean() if len(sealed) else float("nan")
                ),
                "Selection switches": int(summary["selection_switches"]),
                "Candidate sequence": " -> ".join(
                    result.folds["selected_candidate"].tolist()
                ),
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        )
    return pd.DataFrame(rows)


def truncate_inputs(
    inputs: AllocationInputs, last_session: pd.Timestamp | str
) -> AllocationInputs:
    """The same inputs with every frame cut at ``last_session``.

    An in-memory truncation.  It is a weaker custody proof than reloading from
    disk with the loader's own guard armed, because nothing here physically
    prevents a sealed byte from having been read; what it does still catch is
    any full-sample quantity in the state stack, because such a quantity would
    change when the later rows disappear.
    """

    ceiling = pd.Timestamp(last_session)
    panel = inputs.panel
    keep = panel.sessions <= ceiling
    if int(keep.sum()) < 2:
        raise ValueError("truncation leaves fewer than two sessions")
    fields = {
        name: getattr(panel, name).loc[keep]
        for name in (
            "adjusted_close",
            "unadjusted_close",
            "volume",
            "shares_traded",
            "turnover_usd",
            "dividend",
            "index_membership_sp_500",
            "index_membership_sp_midcap_400",
            "index_membership_sp_smallcap_600",
        )
    }
    prices = fields["adjusted_close"]
    truncated = etfs.EtfPanel(
        tickers=panel.tickers,
        returns=prices.div(prices.shift(1)).sub(1.0).iloc[1:],
        alignment=panel.alignment,
        **fields,
    )
    return AllocationInputs(
        panel=truncated,
        history_close=inputs.history_close.loc[inputs.history_close.index <= ceiling],
        history_turnover=inputs.history_turnover.loc[
            inputs.history_turnover.index <= ceiling
        ],
        permitted_last_session=ceiling,
    )


def holdout_custody_report(
    inputs: AllocationInputs,
    config: AllocationConfig,
    plan: WalkForwardPlan,
    *,
    candidates: Sequence[AllocationCandidate] = DECLARED_CANDIDATES,
    data_dir: Path | str | None = None,
    reload_from_disk: bool = True,
) -> pd.DataFrame:
    """Prove the sealed block could not have informed the development path.

    Custody is a claim about what a run read, and the cheapest available proof
    of it is arithmetic rather than ceremony.  A second set of inputs is loaded
    with ``last_session`` pinned to the session before the sealed block, so the
    loader physically refuses to return a sealed row, and the walk-forward is
    re-run on those inputs alone.  If any quantity anywhere in the state stack
    were fitted on the full sample --- a percentile taken over everything, a
    centred window, a normalisation --- the two development paths would differ.

    They are compared to zero tolerance.  A near-match is not a match: floating
    point is deterministic here because both runs execute the same operations in
    the same order on the same prefix of the same bytes.
    """

    sealed_start = pd.Timestamp(plan.sealed_start)
    sessions = inputs.panel.sessions
    before = sessions[sessions < sealed_start]
    if len(before) == 0:
        raise ValueError("the sealed block starts before the panel does")
    development_last = before[-1]

    if reload_from_disk:
        guarded = load_allocation_inputs(
            inputs.tickers,
            start=sessions[0],
            end=development_last,
            last_session=development_last,
            data_dir=data_dir,
        )
    else:
        guarded = truncate_inputs(inputs, development_last)
    development_boundaries = tuple(
        boundary
        for boundary in plan.boundaries
        if pd.Timestamp(boundary) < sealed_start
    )
    if not development_boundaries:
        raise ValueError("no boundary falls inside the development window")
    development_plan = replace(
        plan,
        boundaries=development_boundaries,
        end=development_last.date().isoformat(),
        sealed_start=development_boundaries[-1],
    )
    development = run_walk_forward(
        guarded,
        config,
        development_plan,
        candidates=candidates,
        label="development_only_replay",
    )
    full = run_walk_forward(
        inputs, config, plan, candidates=candidates, label="full_replay"
    )

    guarded_net = development.stitched.net_return
    full_net = full.stitched.net_return.reindex(guarded_net.index)
    difference = (guarded_net - full_net).abs()
    identical = bool((difference.to_numpy() == 0.0).all())

    return pd.DataFrame(
        [
            {
                "Check": "development replay never reads a sealed session",
                "Guard": (
                    "reloaded from disk with the loader's last-session guard armed"
                    if reload_from_disk
                    else "truncated in memory; the loader guard was not exercised"
                ),
                "Development last session": development_last.date().isoformat(),
                "Sealed block starts": sealed_start.date().isoformat(),
                "Guarded input last session": guarded.panel.last_session.date().isoformat(),
                "Compared sessions": int(len(guarded_net)),
                "Maximum absolute difference": float(difference.max()),
                "Byte identical on the overlap": identical,
                "Guarded candidate sequence": " -> ".join(
                    development.folds["selected_candidate"].tolist()
                ),
                "Full-run candidate sequence": " -> ".join(
                    full.folds.loc[
                        full.folds["oos_basis"] == SELECTOR_BASIS, "selected_candidate"
                    ].tolist()
                ),
                "Limitation": "; ".join(ALLOCATION_LIMITATIONS),
            }
        ]
    )


def sealed_block_state_coverage(
    inputs: AllocationInputs,
    plan: WalkForwardPlan,
    *,
    equity_ticker: str = "SPY",
) -> InferenceResult:
    """Whether the sealed block supports a full-bear conditional estimate.

    The Daniel-Moskowitz state is an external coverage diagnostic, not an input
    to the allocator's Faber and time-series-momentum gates. If the sealed block
    holds no session in that state, full-bear conditional performance is not
    estimable out of sample, and the honest output is a refusal rather than a
    reassuring number computed on zero observations.
    """

    bear = regimes.daniel_moskowitz_bear_state(
        inputs.history_close[equity_ticker].dropna()
    ).reindex(inputs.panel.sessions)
    sealed = bear.loc[bear.index >= pd.Timestamp(plan.sealed_start)]
    occupied = int(sealed.eq(regimes.BEAR).sum())
    if occupied == 0:
        return InferenceResult(
            statistic="Sealed-block performance conditional on an equity bear market",
            status=NOT_ESTIMABLE,
            value=None,
            reason=(
                f"the sealed block {plan.sealed_start}..{plan.end} holds "
                f"{len(sealed)} sessions and none of them is in the "
                "Daniel-Moskowitz bear state, so the conditional statistic has no "
                "observations; performance through a full equity bear is not "
                "tested by this block"
            ),
        )
    return InferenceResult(
        statistic="Sealed-block sessions in the equity bear state",
        status=ESTIMATED,
        value=float(occupied),
        reason=(
            f"{occupied} of {len(sealed)} sealed sessions carry the bear label"
        ),
    )


__all__ = [
    "ALLOCATION_LIMITATIONS",
    "COST_LIMITATION",
    "DECLARED_CANDIDATES",
    "DEFAULT_CASH_TICKER",
    "DEFAULT_COST_MULTIPLIERS",
    "EQUAL_WEIGHT",
    "FABER_TREND",
    "FOLD_COLUMNS",
    "FUNDING_LIMITATION",
    "IN_SAMPLE_BASIS",
    "IN_SAMPLE_DEGRADATION_WINDOW",
    "INVERSE_VOLATILITY",
    "LEDGER_COLUMNS",
    "MIXED_BASIS",
    "NO_GATE",
    "PATH_REPORT_COLUMNS",
    "SEALED_BASIS",
    "SEALED_BLOCK_LIMITATION",
    "SEALED_PERMITTED_USE",
    "SELECTOR_BASIS",
    "SELECTOR_PERMITTED_USE",
    "SIXTY_FORTY_WEIGHTS",
    "STABILITY_PERMITTED_USE",
    "TIME_SERIES_MOMENTUM",
    "AllocationCandidate",
    "AllocationConfig",
    "AllocationInputs",
    "AllocationLedger",
    "CostModel",
    "DesiredWeights",
    "WalkForwardPlan",
    "WalkForwardResult",
    "annual_boundaries",
    "candidate_table",
    "comparison_report",
    "cost_sensitivity_report",
    "desired_weight_frame",
    "equal_weight_reference",
    "exposure_share_report",
    "fixed_weight_frame",
    "fold_dispersion_report",
    "holdout_custody_report",
    "in_sample_out_of_sample_report",
    "lagged_capacity",
    "load_allocation_inputs",
    "membership_mask",
    "out_of_sample_accounting",
    "paired_reference_inference",
    "path_report",
    "path_statistics",
    "rebalance_sessions",
    "reference_ledgers",
    "risk_matched_comparison",
    "run_walk_forward",
    "sealed_block_state_coverage",
    "simulate_allocation",
    "straddling_episode_report",
    "truncate_inputs",
    "volatility_budget_diagnostic",
]
