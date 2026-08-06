"""Aligned ETF panel, the pre-declared allocation universe, and its defects.

This module is the data boundary for the ETF dynamic regime allocation sleeve.
It loads the vendor extract, assembles an aligned panel, and states in code the
rule by which the universe was chosen.  Nothing here computes a return, a
volatility, a Sharpe ratio or a drawdown, and that omission is the point: a
reader can confirm by inspection that the selection path never touched a
performance statistic.

What this module does NOT prove
-------------------------------
**It does not correct survivorship, and it cannot.**  Every one of the 745
files in this extract ends on 2018-12-31 *with positive volume on that exact
session*.  The distribution of "last session with positive volume" is 745 on
2018-12-31 and zero on every other date, so no fund that closed or delisted
between 1993 and 2018 is present.  Restricting the universe to eleven large,
broad, index-tracking asset-class funds reduces exposure to that defect but
does not remove it: the inception dates and the vendor's inclusion decision
remain.  Selecting these eleven is a *judgement* that they were never plausible
delisting candidates, not a *measurement* that they were not.  Every frame this
module returns carries ``SURVIVORSHIP_LIMITATION`` and every downstream artifact
must stamp it onto its rows, following the ``collateral_yield_leg_only``
convention in ``research/collateral.py``.

**It does not establish that the universe is a good allocation set.**  The
selection rule reads asset-class coverage, inception date and liquidity, and
nothing else.  ``UniverseMember.__post_init__`` refuses a selection reason that
mentions a performance statistic, so the pre-declaration is enforced rather
than asserted.

**It does not make the early history tradeable.**  Median daily turnover per
fund per year is reported so the caller can see which funds were thin when.
Inside the 2006-2018 window the loader flags seven fund-years below USD 25m per
day --- IEF 2006-2007, LQD 2006-2008, DBC 2006-2007 --- and the caller is
responsible for either excluding or sizing around them.  The capacity of this
sleeve in its early years is a statement about DBC and LQD, not about the panel
average.

**It does not reconcile against the vendor's ``Dividend`` column, because that
column is empty.**  It is identically 0.0 in every file on every session.
``Close`` already contains the distributions.  The failure mode to guard is not
double-counting but its opposite: a reader who sees ``Dividend == 0``, concludes
``Close`` is price-only, bolts on an external dividend series, and inflates
every return in the study by 1.9% (SPY) to 4.7% (a REIT fund) per year.
``dividend_reconciliation_report`` performs the reconciliation by the only route
the data permits --- inverting the back-adjustment factor --- and reports the
recovered distributions so the claim is checkable.

The selection rule, stated once
-------------------------------
For each pre-declared exposure, take the fund with the earliest first quoted
date whose median daily USD turnover over ``TURNOVER_MEASUREMENT_WINDOW`` is
large enough to carry an institutional allocation; where two candidates tie on
inception, take the more liquid on the same measure.  Coverage spans US
large-cap equity, US small-cap equity, developed ex-US equity, emerging equity,
US real estate, the three-point Treasury curve, investment-grade credit, gold
and broad commodities.  ``EXCLUDED_CANDIDATES`` records every fund that lost,
and why, so the rule can be audited without rerunning it.

Every turnover figure quoted in this module --- in ``UNIVERSE``, in
``EXCLUDED_CANDIDATES`` and in ``universe_table`` --- is measured on that one
window, 2006-02-03 to 2018-12-31, which is the traded panel's own calendar.  It
is deliberately not the fund's whole life: a life median mixes a fund's thin
first years into its screen figure and the two differ by up to 2.6x here (SPY
USD 7.79bn life against USD 20.26bn on the window).  ``universe_turnover_audit``
recomputes each declared figure from the files under exactly that definition so
the pre-declaration is checkable rather than asserted.

Two disclosures travel with that choice.  The window ends 2018-12-31, so the
liquidity screen was measured on data that includes the block the allocation
sleeve later seals; liquidity is not a performance statistic and the screen
never reads a return, but the measurement is a full-panel one and is not
out-of-sample with respect to anything.  And the screen is a *documentation*
figure: nothing in the traded pipeline reads it.  The causal membership rule in
``causal_membership_mask`` admits a fund from its own trailing turnover with a
one-session lag and never consults these constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_SUBDIR = "Round1AllData/Quant Researcher/Delta1"
PRICE_SUBDIR = "ETF Data"
CATALOGUE_FILENAME = "CATALOGUE_Delta1_ETF.csv"

#: The vendor writes exactly these twelve columns, in this order, in all 745
#: files.  A file that differs is a different extract and must not be mixed in.
PRICE_COLUMNS = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Turnover",
    "Unadjusted Close",
    "Dividend",
    "Constituent_S&P 500",
    "Constituent_S&P MidCap 400",
    "Constituent_S&P SmallCap 600",
)

#: The vendor's first-quoted field is D/M/YY.  Parsing it as M/D/YY raises on
#: 435 of 745 rows and, worse, silently succeeds on the other 310 with month and
#: day transposed, so the format is pinned rather than inferred.
CATALOGUE_DATE_FORMAT = "%d/%m/%y"

#: The window every declared turnover figure in this module is measured over:
#: the traded panel's own calendar, from the date on which all eleven members
#: have quoted to the last session in the extract.  Named as a constant so the
#: audit function and the artifact heading cannot drift apart from the numbers.
TURNOVER_MEASUREMENT_WINDOW: tuple[str, str] = ("2006-02-03", "2018-12-31")

SURVIVORSHIP_LIMITATION = (
    "survivors_only_universe; all_745_panel_files_end_2018-12-31_with_positive_volume; "
    "no_closed_or_delisted_etf_is_present; "
    "exposure_limited_by_allocation_across_a_fixed_set_of_large_broad_index_funds_"
    "rather_than_cross_sectional_selection"
)

EMPTY_VENDOR_COLUMNS_LIMITATION = (
    "vendor_dividend_column_identically_zero; "
    "vendor_index_membership_columns_identically_zero"
)

EARLY_LIQUIDITY_LIMITATION = (
    "early_window_capacity_set_by_dbc_and_lqd_2006_2008; "
    "panel_average_turnover_overstates_tradeable_size"
)

PANEL_LIMITATIONS: tuple[str, ...] = (
    SURVIVORSHIP_LIMITATION,
    EMPTY_VENDOR_COLUMNS_LIMITATION,
    EARLY_LIQUIDITY_LIMITATION,
)

# A selection reason that reaches for one of these words is a performance
# argument wearing a coverage argument's clothes.  The universe is pre-declared
# on coverage, inception and liquidity, so the guard is mechanical.
_PERFORMANCE_WORDS = (
    "sharpe",
    "cagr",
    "drawdown",
    "outperform",
    "performance",
    "profit",
    "alpha",
    "backtest",
    "return",
    "volatility",
    "correlation",
)


def _reject_performance_language(text: str, label: str) -> None:
    lowered = text.lower()
    for word in _PERFORMANCE_WORDS:
        if word in lowered:
            raise ValueError(
                f"{label} mentions {word!r}; the universe is pre-declared on "
                "coverage, inception and liquidity only"
            )


@dataclass(frozen=True)
class UniverseMember:
    """One pre-declared exposure and the fund selected to carry it.

    ``selection_reason`` is validated, not decorative.  It must name the
    coverage, inception or liquidity ground on which the fund was taken, and it
    may not mention any performance statistic.

    ``panel_window_median_daily_turnover_usd`` is named for what was measured
    rather than for what would sound tidier.  It is the median of the fund's
    daily USD ``Turnover`` over ``TURNOVER_MEASUREMENT_WINDOW`` --- the traded
    panel's own calendar --- and it is *not* the median over the fund's whole
    life.  The two differ by a lot for the older funds, because a fund's early
    years are its thinnest: SPY's life median is USD 7.79bn against USD 20.26bn
    over the panel window, IEF's USD 56.4m against USD 82.8m.  Quoting a
    panel-window figure under a "life" heading would leave a reader unable to
    reproduce the pre-declared rule from the data, so the field says which
    window it is.  ``universe_turnover_audit`` recomputes it from the files.
    """

    ticker: str
    exposure: str
    asset_class: str
    first_quoted_date: str
    panel_window_median_daily_turnover_usd: float
    selection_reason: str

    def __post_init__(self) -> None:
        for field in ("ticker", "exposure", "asset_class", "selection_reason"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.ticker != self.ticker.upper():
            raise ValueError(f"ticker must be upper case, got {self.ticker!r}")
        try:
            parsed = pd.Timestamp(self.first_quoted_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"first_quoted_date must be an ISO date, got {self.first_quoted_date!r}"
            ) from exc
        if pd.isna(parsed):
            raise ValueError("first_quoted_date must be an ISO date")
        turnover = self.panel_window_median_daily_turnover_usd
        if isinstance(turnover, bool) or not isinstance(
            turnover, (int, float, np.floating)
        ):
            raise ValueError(
                "panel_window_median_daily_turnover_usd must be a real number"
            )
        if not np.isfinite(float(turnover)) or float(turnover) <= 0.0:
            raise ValueError(
                "panel_window_median_daily_turnover_usd must be finite and positive"
            )
        if len(self.selection_reason) < 40:
            raise ValueError(
                f"{self.ticker}: selection_reason must state the coverage, inception "
                "or liquidity ground for the choice"
            )
        _reject_performance_language(
            self.selection_reason, f"{self.ticker} selection_reason"
        )

    @property
    def first_quoted(self) -> pd.Timestamp:
        return pd.Timestamp(self.first_quoted_date)


@dataclass(frozen=True)
class ExcludedCandidate:
    """A fund that could have carried an exposure and did not, with the reason.

    Recording the losers is what makes the selection rule auditable.  Without
    this list a reader can see only that eleven funds were chosen, not that the
    alternatives lost on inception or liquidity rather than on results.
    """

    ticker: str
    exposure: str
    reason: str

    def __post_init__(self) -> None:
        for field in ("ticker", "exposure", "reason"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        _reject_performance_language(self.reason, f"{self.ticker} exclusion reason")


#: The pre-declared universe.  Frozen at module level so a downstream fold
#: cannot swap a member after seeing how it did; that would void the
#: pre-declaration and turn the survivorship disclosure from a bound into an
#: understatement.
UNIVERSE: tuple[UniverseMember, ...] = (
    UniverseMember(
        ticker="SPY",
        exposure="us_large_cap_equity",
        asset_class="equity",
        first_quoted_date="1993-01-29",
        panel_window_median_daily_turnover_usd=20_256_600_000.0,
        selection_reason=(
            "Oldest fund in the entire 745-file extract (1993-01-29, 6528 sessions) "
            "and the most liquid by a factor of 5.8x over the next-largest fund. "
            "IVV (2000-05-19) and VTI (2001-05-31) are seven to eight years younger "
            "and 37x/126x thinner, so neither criterion is in tension."
        ),
    ),
    UniverseMember(
        ticker="IWM",
        exposure="us_small_cap_equity",
        asset_class="equity",
        first_quoted_date="2000-05-26",
        panel_window_median_daily_turnover_usd=3_818_700_000.0,
        selection_reason=(
            "Oldest liquid US small-cap fund. IJR shares the identical inception "
            "date of 2000-05-26 but trades USD 93.9m per day against IWM's USD "
            "3818.7m, 41x thinner; inception ties, so liquidity decides."
        ),
    ),
    UniverseMember(
        ticker="IYR",
        exposure="us_real_estate",
        asset_class="real_estate",
        first_quoted_date="2000-06-19",
        panel_window_median_daily_turnover_usd=574_000_000.0,
        selection_reason=(
            "Oldest and most liquid listed-real-estate fund in the extract: "
            "2000-06-19 against VNQ 2004-09-29 (USD 164.3m per day), ICF 2001-02-02 "
            "(USD 30.5m) and RWR 2001-04-27 (USD 16.9m). Wins on both criteria at "
            "once, so no trade-off is being made."
        ),
    ),
    UniverseMember(
        ticker="EFA",
        exposure="developed_ex_us_equity",
        asset_class="equity",
        first_quoted_date="2001-08-17",
        panel_window_median_daily_turnover_usd=1_021_900_000.0,
        selection_reason=(
            "Oldest broad developed-ex-US fund and the most liquid. VEA (2007), VEU "
            "(2007) and ACWX (2008) are six or more years younger and 10-20x "
            "thinner. Single-country funds from 1996 are not a broad exposure and "
            "would require constructing a basket, which is a portfolio decision "
            "rather than a fund selection."
        ),
    ),
    UniverseMember(
        ticker="EEM",
        exposure="emerging_equity",
        asset_class="equity",
        first_quoted_date="2003-04-11",
        panel_window_median_daily_turnover_usd=2_210_000_000.0,
        selection_reason=(
            "Oldest broad emerging-market fund and the more liquid of the two "
            "credible candidates: VWO is 23 months younger at USD 475.4m per day "
            "against EEM's USD 2210.0m."
        ),
    ),
    UniverseMember(
        ticker="SHY",
        exposure="treasury_1_3y_cash_proxy",
        asset_class="rates",
        first_quoted_date="2002-07-26",
        panel_window_median_daily_turnover_usd=75_900_000.0,
        selection_reason=(
            "Oldest short-duration Treasury fund by 4.5 years and the most liquid. "
            "The purer bill proxies arrive far later and thinner (SHV 2007-01-11, "
            "BIL 2007-05-30); taking one of them would push the fully-quoted date "
            "from 2006-02-03 to 2007-05-30 and cost 341 sessions for a 1.9-year "
            "versus 0.3-year duration difference."
        ),
    ),
    UniverseMember(
        ticker="IEF",
        exposure="treasury_7_10y",
        asset_class="rates",
        first_quoted_date="2002-07-26",
        panel_window_median_daily_turnover_usd=82_800_000.0,
        selection_reason=(
            "Oldest intermediate Treasury fund by 4.5 years; IEI launched "
            "2007-01-11 at USD 20.3m per day, later and 4x thinner. Required to "
            "span the Treasury curve between SHY and TLT. Thinnest fixed-income leg "
            "in the early window at USD 13.1m per day in 2006."
        ),
    ),
    UniverseMember(
        ticker="TLT",
        exposure="treasury_20y_plus",
        asset_class="rates",
        first_quoted_date="2002-07-26",
        panel_window_median_daily_turnover_usd=701_900_000.0,
        selection_reason=(
            "Oldest long-duration Treasury fund by 4.5 years and overwhelmingly the "
            "most liquid. TLH and EDV are both later and roughly 200x thinner; "
            "neither could carry an institutional allocation at any point in the "
            "sample."
        ),
    ),
    UniverseMember(
        ticker="LQD",
        exposure="investment_grade_credit",
        asset_class="credit",
        first_quoted_date="2002-07-26",
        panel_window_median_daily_turnover_usd=163_700_000.0,
        selection_reason=(
            "Oldest corporate credit fund by 4.5 years and the most liquid "
            "investment-grade vehicle; USIG, IGSB and IGIB all launched 2007-01-11. "
            "HYG is a distinct exposure rather than a peer, and admitting it would "
            "push the fully-quoted date to 2007-04-11 at a cost of 296 sessions."
        ),
    ),
    UniverseMember(
        ticker="GLD",
        exposure="gold",
        asset_class="commodity",
        first_quoted_date="2004-11-18",
        panel_window_median_daily_turnover_usd=982_900_000.0,
        selection_reason=(
            "Oldest gold vehicle in the extract by 71 days and 19.5x more liquid "
            "than its only peer, IAU. Also the cleanest instrument in the panel: "
            "zero distributions and zero splits over its whole life, so adjusted "
            "and unadjusted closes are identical and it serves as the analytic "
            "control for the adjustment reconciliation."
        ),
    ),
    UniverseMember(
        ticker="DBC",
        exposure="broad_commodities",
        asset_class="commodity",
        first_quoted_date="2006-02-03",
        panel_window_median_daily_turnover_usd=33_500_000.0,
        selection_reason=(
            "Oldest broad-commodity fund and 5.5x more liquid than its only peer, "
            "GSG, which is untradeable at institutional size across the whole "
            "sample. Binding constraint on the panel start date: it is the last "
            "member to exist and therefore alone sets the fully-quoted date at "
            "2006-02-03."
        ),
    ),
)

UNIVERSE_TICKERS: tuple[str, ...] = tuple(member.ticker for member in UNIVERSE)

EXPOSURE: dict[str, str] = {m.ticker: m.exposure for m in UNIVERSE}
ASSET_CLASS: dict[str, str] = {m.ticker: m.asset_class for m in UNIVERSE}

#: The funds that lost, and the criterion they lost on.  Every reason is a
#: coverage, inception or liquidity statement; none is a result.
EXCLUDED_CANDIDATES: tuple[ExcludedCandidate, ...] = (
    ExcludedCandidate("IVV", "us_large_cap_equity", "Later inception (2000-05-19) and 37x thinner than SPY."),
    ExcludedCandidate("VTI", "us_large_cap_equity", "Later inception (2001-05-31) and 126x thinner than SPY."),
    ExcludedCandidate("QQQ", "us_large_cap_equity", "A single-exchange growth index, not a broad US large-cap exposure."),
    ExcludedCandidate("IJR", "us_small_cap_equity", "Ties IWM on inception (2000-05-26) and is 41x thinner."),
    ExcludedCandidate("VNQ", "us_real_estate", "Later inception (2004-09-29) and 3.5x thinner than IYR."),
    ExcludedCandidate("ICF", "us_real_estate", "Later inception (2001-02-02) and 19x thinner than IYR."),
    ExcludedCandidate("RWR", "us_real_estate", "Later inception (2001-04-27) and 34x thinner than IYR."),
    ExcludedCandidate("VEA", "developed_ex_us_equity", "Later inception (2007-07-26) and 10x thinner than EFA."),
    ExcludedCandidate("VEU", "developed_ex_us_equity", "Later inception (2007-03-08) and 22x thinner than EFA."),
    ExcludedCandidate("VGK", "developed_ex_us_equity", "Europe only; not a broad developed-ex-US exposure."),
    ExcludedCandidate("IEV", "developed_ex_us_equity", "Europe only, and 53x thinner than EFA."),
    ExcludedCandidate("VWO", "emerging_equity", "Later inception (2005-03-10) and 4.6x thinner than EEM."),
    ExcludedCandidate("SHV", "treasury_1_3y_cash_proxy", "Later inception (2007-01-11) and 2.2x thinner than SHY; would cost 341 panel sessions."),
    ExcludedCandidate("BIL", "treasury_1_3y_cash_proxy", "Later inception (2007-05-30) and 3.9x thinner than SHY; would cost 341 panel sessions."),
    ExcludedCandidate("IEI", "treasury_7_10y", "Later inception (2007-01-11) and 4x thinner than IEF."),
    ExcludedCandidate("TLH", "treasury_20y_plus", "Later inception (2007-01-11) and roughly 230x thinner than TLT."),
    ExcludedCandidate("EDV", "treasury_20y_plus", "Later inception (2007-12-19) and roughly 185x thinner than TLT."),
    ExcludedCandidate("USIG", "investment_grade_credit", "Later inception (2007-01-11) and 36x thinner than LQD."),
    ExcludedCandidate("IGSB", "investment_grade_credit", "Later inception (2007-01-11) and 4x thinner than LQD."),
    ExcludedCandidate("IGIB", "investment_grade_credit", "Later inception (2007-01-11) and 7x thinner than LQD."),
    ExcludedCandidate("HYG", "high_yield_credit", "A distinct exposure rather than a peer of LQD; admitting it would move the fully-quoted date to 2007-04-11 and cost 296 sessions."),
    ExcludedCandidate("IAU", "gold", "Later inception (2005-01-28) and 19.5x thinner than GLD."),
    ExcludedCandidate("GSG", "broad_commodities", "Later inception (2006-07-21) and 5.5x thinner than DBC; median daily turnover of roughly USD 6m cannot carry an institutional allocation."),
    ExcludedCandidate("AGG", "us_aggregate_bond", "By construction Treasuries plus agency mortgages plus investment-grade credit, all of which SHY, IEF, TLT and LQD already carry; adding it introduces a near-linear dependency in the panel without adding an exposure. A construction argument from the fund's mandate."),
    ExcludedCandidate("TIP", "inflation_linked_treasury", "A genuinely distinct exposure that the pre-declared coverage list does not name; admitting it after the coverage list was written would be a universe change rather than an application of the rule."),
)


def default_data_dir() -> Path:
    """Root of the vendor extract, overridable by ``DELTA1_DATA_DIR``."""

    return Path(os.environ.get("DELTA1_DATA_DIR", DEFAULT_DATA_SUBDIR))


def price_dir(data_dir: Path | str | None = None) -> Path:
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    return root / PRICE_SUBDIR


def catalogue_path(data_dir: Path | str | None = None) -> Path:
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    return root / CATALOGUE_FILENAME


#: Heading for the declared liquidity figure.  Built from the constant so the
#: column can never claim a window the numbers were not measured over.
TURNOVER_COLUMN = (
    "Median daily turnover, traded panel window "
    f"{TURNOVER_MEASUREMENT_WINDOW[0]}..{TURNOVER_MEASUREMENT_WINDOW[1]} (USD)"
)


def universe_table() -> pd.DataFrame:
    """The pre-declared universe as a frame, for stamping into artifacts."""

    return pd.DataFrame(
        [
            {
                "Ticker": m.ticker,
                "Exposure": m.exposure,
                "Asset class": m.asset_class,
                "First quoted": m.first_quoted.date().isoformat(),
                TURNOVER_COLUMN: float(m.panel_window_median_daily_turnover_usd),
                "Selection ground": m.selection_reason,
                "Selection inputs": "asset-class coverage, inception date, liquidity",
                "Limitation": SURVIVORSHIP_LIMITATION,
            }
            for m in UNIVERSE
        ]
    )


def excluded_candidate_table() -> pd.DataFrame:
    """Funds considered for an exposure and not taken, with the criterion.

    This is the artifact on which the survivorship caveat is most load-bearing,
    which is why it carries the same stamp as ``universe_table`` rather than
    being treated as mere bookkeeping.  Every fund named here was considered
    from a survivors-only pool: no closed or delisted fund appears among the
    rejections because none is present in the extract to reject.
    """

    return pd.DataFrame(
        [
            {
                "Ticker": c.ticker,
                "Exposure": c.exposure,
                "Exclusion ground": c.reason,
                "Selection inputs": "asset-class coverage, inception date, liquidity",
                "Limitation": SURVIVORSHIP_LIMITATION,
            }
            for c in EXCLUDED_CANDIDATES
        ]
    )


def universe_turnover_audit(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Recompute every declared liquidity figure from the files.

    The declared constants are documentation, and documentation that cannot be
    reproduced from the data is an assertion.  This recomputes each member's
    median daily USD turnover over ``TURNOVER_MEASUREMENT_WINDOW`` and reports
    it beside the declared value and beside the fund's whole-life median, so a
    reader can see both that the declared figure is right and that it is not the
    life median it might otherwise be read as.
    """

    first, last = (pd.Timestamp(value) for value in TURNOVER_MEASUREMENT_WINDOW)
    rows = []
    for member in UNIVERSE:
        turnover = load_fund(member.ticker, data_dir)["turnover_usd"].astype(float)
        window = turnover.loc[(turnover.index >= first) & (turnover.index <= last)]
        if window.empty:
            raise ValueError(
                f"{member.ticker}: no session falls inside the declared turnover "
                "measurement window"
            )
        declared = float(member.panel_window_median_daily_turnover_usd)
        rows.append(
            {
                "Ticker": member.ticker,
                "Measurement window": (
                    f"{TURNOVER_MEASUREMENT_WINDOW[0]}..{TURNOVER_MEASUREMENT_WINDOW[1]}"
                ),
                "Declared median daily turnover (USD)": declared,
                "Recomputed median daily turnover (USD)": float(window.median()),
                "Recomputed over declared": float(window.median()) / declared,
                "Whole-life median daily turnover (USD)": float(turnover.median()),
                "Whole-life over declared": float(turnover.median()) / declared,
                "Limitation": SURVIVORSHIP_LIMITATION,
            }
        )
    return pd.DataFrame(rows)


def load_catalogue(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Vendor catalogue with ``first_quoted_date`` parsed as D/M/YY.

    The taxonomy fields carry no information in this extract: ``base_type`` is
    "Stock Market" for all 745 rows, ``subtype1`` is "Exchange Traded Product",
    ``subtype2`` is "Exchange Traded Fund (ETF)" and ``subtype3`` is missing
    everywhere.  Asset class is therefore a research judgement, written down in
    ``UNIVERSE`` rather than deferred to a vendor field that does not exist.
    """

    path = catalogue_path(data_dir)
    if not path.is_file():
        raise ValueError(f"no ETF catalogue at {path}")
    catalogue = pd.read_csv(path)
    if "symbol" not in catalogue.columns:
        raise ValueError(f"{path} has no 'symbol' column")
    if catalogue["symbol"].duplicated().any():
        raise ValueError(f"{path} contains duplicate symbols")
    parsed = pd.to_datetime(
        catalogue["first_quoted_date"], format=CATALOGUE_DATE_FORMAT, errors="raise"
    )
    return catalogue.assign(first_quoted_date=parsed)


def load_fund(ticker: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """One fund's session history with the derived columns the panel needs.

    ``adjusted_close``
        The vendor's ``Close``, back-adjusted for distributions and splits.  It
        is the only column from which a return may be taken.

    ``unadjusted_close``
        The raw traded print.  It retains splits --- EFA goes 157.00 to 52.54 on
        2005-06-09 --- so it must never be differenced.  It answers share-count
        and price-level questions only.

    ``turnover_usd``
        True USD notional.  This is the liquidity screen.

    ``volume``
        The vendor's ``Volume``, which is *back-adjusted* onto the same scale as
        ``Close`` and is not a raw share count.  It is carried unmodified for
        fidelity and must not be used for liquidity: because the adjustment
        factor is smallest early in life, a volume-based screen overstates
        early-life tradeability by the reciprocal of that factor.

    ``shares_traded``
        The recovered raw share count, ``Volume * Close / Unadjusted Close``.

    ``dividend`` and the three ``index_membership_*`` columns are carried
    unmodified and are identically zero throughout the extract.
    """

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    path = price_dir(data_dir) / f"{ticker}.csv"
    if not path.is_file():
        raise ValueError(f"no price file for {ticker!r} at {path}")

    raw = pd.read_csv(path, parse_dates=["Date"], date_format="%Y-%m-%d")
    missing = [column for column in PRICE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"{ticker}: price file is missing columns {missing}")
    frame = raw.set_index("Date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{ticker}: duplicate session dates")

    close = frame["Close"].astype(float)
    unadjusted = frame["Unadjusted Close"].astype(float)
    for name, series in (("Close", close), ("Unadjusted Close", unadjusted)):
        values = series.to_numpy()
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"{ticker}: {name} must be finite and positive")

    adjustment_factor = close / unadjusted
    turnover = frame["Turnover"].astype(float)
    if (turnover < 0).any():
        raise ValueError(f"{ticker}: Turnover cannot be negative")

    return pd.DataFrame(
        {
            "adjusted_close": close,
            "unadjusted_close": unadjusted,
            "volume": frame["Volume"].astype(float),
            "shares_traded": frame["Volume"].astype(float) * adjustment_factor,
            "turnover_usd": turnover,
            "dividend": frame["Dividend"].astype(float),
            "adjustment_factor": adjustment_factor,
            "index_membership_sp_500": frame["Constituent_S&P 500"].astype(float),
            "index_membership_sp_midcap_400": frame[
                "Constituent_S&P MidCap 400"
            ].astype(float),
            "index_membership_sp_smallcap_600": frame[
                "Constituent_S&P SmallCap 600"
            ].astype(float),
        },
        index=frame.index,
    )


_PANEL_FIELDS = (
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


@dataclass(frozen=True)
class EtfPanel:
    """Aligned wide frames for one set of funds over one window.

    Every frame shares one index and one column order.  The index is the
    *intersection* of the members' calendars, never the union: a union
    concatenation manufactures a missing cell for any fund that did not quote
    that day, and a downstream forward fill turns that cell into a fabricated
    zero return.  ``alignment`` reports how many sessions the intersection cost
    so the caller can confirm it is the figure they expect.
    """

    tickers: tuple[str, ...]
    adjusted_close: pd.DataFrame
    unadjusted_close: pd.DataFrame
    volume: pd.DataFrame
    shares_traded: pd.DataFrame
    turnover_usd: pd.DataFrame
    dividend: pd.DataFrame
    index_membership_sp_500: pd.DataFrame
    index_membership_sp_midcap_400: pd.DataFrame
    index_membership_sp_smallcap_600: pd.DataFrame
    returns: pd.DataFrame
    alignment: pd.DataFrame
    limitations: tuple[str, ...] = PANEL_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.tickers, tuple) or not self.tickers:
            raise ValueError("tickers must be a non-empty tuple")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError(f"tickers contains duplicates: {self.tickers}")
        reference = self.adjusted_close
        if not isinstance(reference, pd.DataFrame) or reference.empty:
            raise ValueError("adjusted_close must be a non-empty DataFrame")
        if not isinstance(reference.index, pd.DatetimeIndex):
            raise ValueError("panel index must be a DatetimeIndex")
        if not reference.index.is_monotonic_increasing or reference.index.has_duplicates:
            raise ValueError("panel index must be unique and increasing")
        for field in _PANEL_FIELDS:
            frame = getattr(self, field)
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(f"{field} must be a pandas DataFrame")
            if list(frame.columns) != list(self.tickers):
                raise ValueError(f"{field} columns do not match tickers")
            if not frame.index.equals(reference.index):
                raise ValueError(f"{field} index does not match the panel index")
            if frame.isna().to_numpy().any():
                raise ValueError(f"{field} carries a missing value after alignment")
        if list(self.returns.columns) != list(self.tickers):
            raise ValueError("returns columns do not match tickers")
        if len(self.returns) != len(reference) - 1:
            raise ValueError("returns must be one row shorter than the price panel")
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("limitations must be a non-empty tuple")
        if SURVIVORSHIP_LIMITATION not in self.limitations:
            raise ValueError("the survivorship limitation may not be dropped")

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.adjusted_close.index

    @property
    def first_session(self) -> pd.Timestamp:
        return self.adjusted_close.index[0]

    @property
    def last_session(self) -> pd.Timestamp:
        return self.adjusted_close.index[-1]


def load_panel(
    tickers: tuple[str, ...] | list[str] = UNIVERSE_TICKERS,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    last_session: pd.Timestamp | str | None = None,
    data_dir: Path | str | None = None,
) -> EtfPanel:
    """Assemble the aligned panel on the intersection calendar.

    ``last_session`` is the custody guard.  A development run passes the last
    development session and the loader refuses to return a row beyond it, so a
    sealed block cannot be read by accident.  The guard is checked twice: once
    against the requested window and once against the assembled frames, because
    the second check is the one that catches a bug in the first.
    """

    ordered = tuple(tickers)
    if not ordered:
        raise ValueError("at least one ticker is required")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"tickers contains duplicates: {ordered}")

    ceiling = pd.Timestamp(last_session) if last_session is not None else None
    window_end = pd.Timestamp(end) if end is not None else None
    if ceiling is not None:
        if window_end is None:
            window_end = ceiling
        elif window_end > ceiling:
            raise ValueError(
                f"requested end {window_end.date()} is beyond the permitted last "
                f"session {ceiling.date()}"
            )
    window_start = pd.Timestamp(start) if start is not None else None
    if window_start is not None and window_end is not None and window_start >= window_end:
        raise ValueError(
            f"start {window_start.date()} must precede end {window_end.date()}"
        )

    frames = {t: load_fund(t, data_dir) for t in ordered}
    for ticker, frame in frames.items():
        if window_start is not None and frame.index.min() > window_start:
            raise ValueError(
                f"{ticker} first quotes {frame.index.min().date()}, after the "
                f"requested start {window_start.date()}; the panel would be unbalanced"
            )

    intersection: pd.DatetimeIndex | None = None
    union: pd.DatetimeIndex | None = None
    for frame in frames.values():
        window = frame.index
        if window_start is not None:
            window = window[window >= window_start]
        if window_end is not None:
            window = window[window <= window_end]
        intersection = window if intersection is None else intersection.intersection(window)
        union = window if union is None else union.union(window)
    assert intersection is not None and union is not None
    if len(intersection) < 2:
        raise ValueError("the requested window leaves fewer than two shared sessions")

    aligned = {
        field: pd.DataFrame(
            {t: frames[t][field].reindex(intersection) for t in ordered},
            index=intersection,
        )
        for field in _PANEL_FIELDS
    }
    prices = aligned["adjusted_close"]
    returns = prices.div(prices.shift(1)).sub(1.0).iloc[1:]

    alignment = pd.DataFrame(
        [
            {
                "Union sessions": int(len(union)),
                "Intersection sessions": int(len(intersection)),
                "Sessions dropped by intersection": int(len(union) - len(intersection)),
                "First session": intersection[0].date().isoformat(),
                "Last session": intersection[-1].date().isoformat(),
                "Permitted last session": (
                    ceiling.date().isoformat() if ceiling is not None else "unbounded"
                ),
                "Limitation": SURVIVORSHIP_LIMITATION,
            }
        ]
    )

    panel = EtfPanel(
        tickers=ordered,
        returns=returns,
        alignment=alignment,
        **aligned,
    )
    if ceiling is not None and panel.last_session > ceiling:
        raise ValueError(
            f"assembled panel reaches {panel.last_session.date()}, beyond the "
            f"permitted last session {ceiling.date()}"
        )
    return panel


def tradeable_mask(
    turnover_usd: pd.DataFrame,
    *,
    window: int = 252,
    min_median_turnover_usd: float = 10_000_000.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Causal liquidity screen on trailing median USD turnover.

    Mirrors ``strategy.tradeable_mask``: a trailing median over a fixed window,
    compared against a pre-declared floor.  Two differences are deliberate.
    The screen reads USD notional rather than contract volume, because the
    vendor's ``Volume`` column is back-adjusted and overstates early-life
    activity by the reciprocal of the adjustment factor.  And the result is
    shifted by ``lag`` sessions, because universe membership is a decision and
    obeys the same causality contract as any other: a fund is admitted on
    session ``t`` only on evidence published strictly before ``t``.

    A missing cell means the fund did not quote, which is not tradeable, so it
    is treated as zero turnover rather than dropped.
    """

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be a whole number of at least two sessions")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive whole number of sessions")
    if (
        isinstance(min_median_turnover_usd, bool)
        or not isinstance(min_median_turnover_usd, (int, float, np.floating))
        or not np.isfinite(float(min_median_turnover_usd))
        or float(min_median_turnover_usd) < 0.0
    ):
        raise ValueError("min_median_turnover_usd must be finite and non-negative")

    trailing = turnover_usd.fillna(0.0).rolling(window, min_periods=window).median()
    admitted = trailing >= float(min_median_turnover_usd)
    return admitted.shift(lag, fill_value=False).astype(bool)


def causal_membership_mask(
    turnover_usd: pd.DataFrame,
    *,
    liquidity_window: int = 252,
    min_median_turnover_usd: float = 10_000_000.0,
    min_history_sessions: int = 315,
    lag: int = 1,
) -> pd.DataFrame:
    """Whether each fund may be held on each session, decided causally.

    Two conditions, both computed from data published strictly before the
    session they govern.  The fund must have at least ``min_history_sessions``
    quoted sessions of its own --- the repository's 315 default, which is the
    252-session longest signal lookback plus a 63-session second-moment warm-up
    --- and it must clear the liquidity screen.

    A fixed hand-picked entry date is not an alternative to this: it is a
    full-sample judgement about when a fund became tradeable, made by someone
    who has seen the whole panel.
    """

    if (
        isinstance(min_history_sessions, bool)
        or not isinstance(min_history_sessions, int)
        or min_history_sessions < 1
    ):
        raise ValueError("min_history_sessions must be a positive whole number")

    liquid = tradeable_mask(
        turnover_usd,
        window=liquidity_window,
        min_median_turnover_usd=min_median_turnover_usd,
        lag=lag,
    )
    quoted = turnover_usd.notna()
    age = quoted.cumsum().where(quoted)
    seasoned = (age >= min_history_sessions).fillna(False)
    seasoned = seasoned.shift(lag, fill_value=False).astype(bool)
    return (liquid & seasoned).astype(bool)


def causal_entry_dates(
    turnover_usd: pd.DataFrame,
    *,
    liquidity_window: int = 252,
    min_median_turnover_usd: float = 10_000_000.0,
    min_history_sessions: int = 315,
    lag: int = 1,
) -> pd.Series:
    """First session on which each fund may enter the book, or ``NaT``."""

    mask = causal_membership_mask(
        turnover_usd,
        liquidity_window=liquidity_window,
        min_median_turnover_usd=min_median_turnover_usd,
        min_history_sessions=min_history_sessions,
        lag=lag,
    )
    entries = {}
    for ticker in mask.columns:
        hits = mask.index[mask[ticker].to_numpy()]
        entries[ticker] = hits.min() if len(hits) else pd.NaT
    return pd.Series(entries, name="Causal entry date")


def lagged_median_turnover(
    turnover_usd: pd.DataFrame,
    *,
    window: int = 60,
    lag: int = 1,
) -> pd.DataFrame:
    """Trailing median USD turnover, lagged: the pre-session capacity evidence.

    This is the quantity an order may be sized from.  Sizing from the executing
    session's own turnover would let the book trade more precisely when
    liquidity turned out well, which is an optimistic look-ahead;
    ``docs/causality.md`` separates that from the market-response rule, under
    which realized depth may only *reduce* a fill.
    """

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be a whole number of at least two sessions")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("lag must be a positive whole number of sessions")
    trailing = turnover_usd.rolling(window, min_periods=window).median()
    return trailing.shift(lag)


def participation_capacity_usd(
    turnover_usd: pd.DataFrame,
    *,
    participation_limit: float = 0.05,
    window: int = 60,
    lag: int = 1,
) -> pd.DataFrame:
    """Daily traded notional each fund supports at a pre-declared participation."""

    if (
        isinstance(participation_limit, bool)
        or not isinstance(participation_limit, (int, float, np.floating))
        or not 0.0 < float(participation_limit) <= 1.0
    ):
        raise ValueError("participation_limit must be in (0, 1]")
    return lagged_median_turnover(turnover_usd, window=window, lag=lag) * float(
        participation_limit
    )


def median_annual_turnover(
    tickers: tuple[str, ...] | list[str] = UNIVERSE_TICKERS,
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Median daily USD turnover per fund per calendar year, over full lives."""

    columns = {}
    for ticker in tickers:
        fund = load_fund(ticker, data_dir)
        turnover = fund["turnover_usd"]
        columns[ticker] = turnover.groupby(turnover.index.year).median()
    return pd.DataFrame(columns).sort_index()


def thin_fund_years(
    tickers: tuple[str, ...] | list[str] = UNIVERSE_TICKERS,
    *,
    min_median_turnover_usd: float = 25_000_000.0,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Fund-years whose median daily turnover sits below a stated floor."""

    table = median_annual_turnover(tickers, data_dir=data_dir)
    if start is not None:
        table = table.loc[table.index >= pd.Timestamp(start).year]
    if end is not None:
        table = table.loc[table.index <= pd.Timestamp(end).year]
    rows: list[dict[str, object]] = []
    for ticker in table.columns:
        column = table[ticker].dropna()
        thin = column[column < float(min_median_turnover_usd)]
        for year, value in thin.items():
            rows.append(
                {
                    "Ticker": ticker,
                    "Year": int(year),
                    "Median daily turnover (USD)": float(value),
                    "Liquidity floor (USD)": float(min_median_turnover_usd),
                    "Limitation": EARLY_LIQUIDITY_LIMITATION,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Year",
                "Median daily turnover (USD)",
                "Liquidity floor (USD)",
                "Limitation",
            ]
        )
    return pd.DataFrame(rows).sort_values(["Ticker", "Year"]).reset_index(drop=True)


def fully_quoted_date(
    tickers: tuple[str, ...] | list[str] = UNIVERSE_TICKERS,
    *,
    data_dir: Path | str | None = None,
) -> pd.Timestamp:
    """The session on which every named fund has quoted at least once.

    Earlier history is warm-up for the funds that exist.  It is never an
    implicit zero weight for the funds that do not, which is why ``load_panel``
    raises on a start date before a member's inception rather than padding.
    """

    firsts = [load_fund(t, data_dir).index.min() for t in tickers]
    return max(firsts)


def availability_report(
    tickers: tuple[str, ...] | list[str] = UNIVERSE_TICKERS,
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Per fund: life, coverage gaps, and when the universe becomes complete.

    A coverage gap is a session on which some other member of the universe
    quoted and this one did not, measured only over the fund's own life so that
    a later inception is not miscounted as a hole.  In this extract exactly one
    such gap exists among the eleven members --- IYR misses five sessions in
    2000 --- and it lies outside every window the sleeve uses.  The distinction
    matters because there is no missing data anywhere in the extract: an absent
    value is always an absent fund.
    """

    funds = {t: load_fund(t, data_dir) for t in tickers}
    union: pd.DatetimeIndex | None = None
    for frame in funds.values():
        union = frame.index if union is None else union.union(frame.index)
    assert union is not None
    complete = max(frame.index.min() for frame in funds.values())

    rows: list[dict[str, object]] = []
    for ticker, frame in funds.items():
        life = union[(union >= frame.index.min()) & (union <= frame.index.max())]
        gaps = life.difference(frame.index)
        gap_positions = np.flatnonzero(life.isin(gaps))
        longest = 0
        run = 0
        previous = -2
        for position in gap_positions:
            run = run + 1 if position == previous + 1 else 1
            longest = max(longest, run)
            previous = int(position)
        turnover = frame["turnover_usd"]
        close = frame["adjusted_close"]
        rows.append(
            {
                "Ticker": ticker,
                "Exposure": EXPOSURE.get(ticker, "unclassified"),
                "Asset class": ASSET_CLASS.get(ticker, "unclassified"),
                "First session": frame.index.min().date().isoformat(),
                "Last session": frame.index.max().date().isoformat(),
                "Sessions": int(len(frame)),
                "Universe sessions over its life": int(len(life)),
                "Coverage gap sessions": int(len(gaps)),
                "Longest coverage gap (sessions)": int(longest),
                "First coverage gap": (
                    gaps.min().date().isoformat() if len(gaps) else "none"
                ),
                "Zero turnover sessions": int((turnover == 0.0).sum()),
                "Exactly flat closes": int((close.diff() == 0.0).sum()),
                "Nonzero vendor dividend rows": int((frame["dividend"] != 0.0).sum()),
                "Nonzero index membership rows": int(
                    (
                        frame[
                            [
                                "index_membership_sp_500",
                                "index_membership_sp_midcap_400",
                                "index_membership_sp_smallcap_600",
                            ]
                        ]
                        != 0.0
                    )
                    .to_numpy()
                    .sum()
                ),
                "Universe fully quoted from": complete.date().isoformat(),
                "Limitation": SURVIVORSHIP_LIMITATION,
            }
        )
    return pd.DataFrame(rows)


def adjustment_events(
    ticker: str,
    *,
    tolerance: float = 1e-6,
    split_threshold: float = 0.15,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Distributions and splits recovered by inverting the back-adjustment.

    The vendor's ``Dividend`` column is empty, so the requested reconciliation
    of adjusted against unadjusted price plus distributions cannot be performed
    as posed.  What can be used is ``A_t = Close_t / Unadjusted Close_t``, which
    is piecewise constant and steps only on ex-distribution dates, by
    ``1 / (1 - D_t / U_{t-1})``, and on splits, by the split ratio.  Inverting
    the step recovers ``D_t``.

    The split classification is a magnitude rule, not a vendor label: a step
    further than ``split_threshold`` from one is a share-count change, because
    no single distribution on a broad index fund is fifteen percent of price.
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(split_threshold) or split_threshold <= 0:
        raise ValueError("split_threshold must be finite and positive")

    fund = load_fund(ticker, data_dir)
    factor = fund["adjustment_factor"]
    step = factor / factor.shift(1)
    events = step[(step - 1.0).abs() > tolerance]
    previous_price = fund["unadjusted_close"].shift(1).reindex(events.index)
    kind = np.where((events - 1.0).abs() > split_threshold, "split", "distribution")
    frame = pd.DataFrame(
        {
            "Ticker": ticker,
            "Adjustment step": events.astype(float),
            "Prior traded price": previous_price.astype(float),
            "Implied distribution per share": previous_price * (1.0 - 1.0 / events),
            "Event kind": kind,
        },
        index=events.index,
    )
    frame.loc[frame["Event kind"] == "split", "Implied distribution per share"] = np.nan
    frame.index.name = "Date"
    return frame


def dividend_reconciliation_report(
    tickers: tuple[str, ...] | list[str] = UNIVERSE_TICKERS,
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Prove that ``Close`` is total return, by arithmetic rather than by label.

    The three quantities that carry the proof are the event *cadence*, the
    implied *yield*, and the *residual*.  A fund whose recovered events arrive
    four times a year at a plausible yield, and whose adjusted and unadjusted
    returns agree to print precision on every other session, has a
    total-return-adjusted close.  A fund with no events at all and a residual of
    exactly zero --- GLD, which has never distributed and never split --- is the
    analytic control: it shows the adjustment is distribution-driven rather than
    a generic rescale.

    ``Nonzero vendor dividend rows`` is zero for every fund in this extract.
    That is the reported fact, not a defect of this function.
    """

    rows: list[dict[str, object]] = []
    for ticker in tickers:
        fund = load_fund(ticker, data_dir)
        events = adjustment_events(ticker, data_dir=data_dir)
        splits = events[events["Event kind"] == "split"]
        distributions = events[events["Event kind"] == "distribution"]

        factor = fund["adjustment_factor"]
        years = (fund.index[-1] - fund.index[0]).days / 365.2425
        split_product = (
            float(splits["Adjustment step"].prod()) if len(splits) else 1.0
        )
        cumulative = float(factor.iloc[-1] / factor.iloc[0]) / split_product
        implied_yield = cumulative ** (1.0 / years) - 1.0 if years > 0 else np.nan

        adjusted_step = fund["adjusted_close"].div(fund["adjusted_close"].shift(1))
        traded_step = fund["unadjusted_close"].div(fund["unadjusted_close"].shift(1))
        quiet = ~fund.index.isin(events.index)
        residual = (adjusted_step - traded_step).abs()[quiet].dropna()

        rows.append(
            {
                "Ticker": ticker,
                "Sessions": int(len(fund)),
                "Years": float(years),
                "Adjustment events": int(len(events)),
                "Distribution events": int(len(distributions)),
                "Split events": int(len(splits)),
                "Distribution events per year": (
                    float(len(distributions) / years) if years > 0 else np.nan
                ),
                "Split ratio product": split_product,
                "Cumulative adjustment excluding splits": cumulative,
                "Implied annual distribution yield": float(implied_yield),
                "Median absolute quiet-session step difference": (
                    float(residual.median()) if len(residual) else 0.0
                ),
                "Maximum absolute quiet-session step difference": (
                    float(residual.max()) if len(residual) else 0.0
                ),
                "Nonzero vendor dividend rows": int((fund["dividend"] != 0.0).sum()),
                "Limitation": EMPTY_VENDOR_COLUMNS_LIMITATION,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ASSET_CLASS",
    "CATALOGUE_DATE_FORMAT",
    "EARLY_LIQUIDITY_LIMITATION",
    "EMPTY_VENDOR_COLUMNS_LIMITATION",
    "EXCLUDED_CANDIDATES",
    "EXPOSURE",
    "EtfPanel",
    "ExcludedCandidate",
    "PANEL_LIMITATIONS",
    "PRICE_COLUMNS",
    "SURVIVORSHIP_LIMITATION",
    "TURNOVER_COLUMN",
    "TURNOVER_MEASUREMENT_WINDOW",
    "UNIVERSE",
    "UNIVERSE_TICKERS",
    "UniverseMember",
    "adjustment_events",
    "availability_report",
    "catalogue_path",
    "causal_entry_dates",
    "causal_membership_mask",
    "default_data_dir",
    "dividend_reconciliation_report",
    "excluded_candidate_table",
    "fully_quoted_date",
    "lagged_median_turnover",
    "load_catalogue",
    "load_fund",
    "load_panel",
    "median_annual_turnover",
    "participation_capacity_usd",
    "price_dir",
    "thin_fund_years",
    "tradeable_mask",
    "universe_table",
    "universe_turnover_audit",
]
