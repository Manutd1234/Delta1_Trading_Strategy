from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.marketdata.etfs import (
    EXCLUDED_CANDIDATES,
    PRICE_COLUMNS,
    SURVIVORSHIP_LIMITATION,
    TURNOVER_COLUMN,
    TURNOVER_MEASUREMENT_WINDOW,
    UNIVERSE,
    UNIVERSE_TICKERS,
    EtfPanel,
    ExcludedCandidate,
    UniverseMember,
    adjustment_events,
    availability_report,
    causal_entry_dates,
    causal_membership_mask,
    dividend_reconciliation_report,
    excluded_candidate_table,
    fully_quoted_date,
    lagged_median_turnover,
    load_catalogue,
    load_fund,
    load_panel,
    median_annual_turnover,
    participation_capacity_usd,
    thin_fund_years,
    tradeable_mask,
    universe_table,
    universe_turnover_audit,
)


DATA_DIR = Path(
    os.environ.get("DELTA1_DATA_DIR", "Round1AllData/Quant Researcher/Delta1")
)
REAL_PRICE_DIR = DATA_DIR / "ETF Data"

BANNED_SUBSTRINGS = ("target", "pass", "fail", "verdict", "rank", "winner", "recommend")


# --------------------------------------------------------------------------
# a synthetic vendor extract, built so the adjustment arithmetic is known
# --------------------------------------------------------------------------


def synthetic_fund(
    sessions: pd.DatetimeIndex,
    *,
    start_price: float,
    drift: float = 0.0002,
    distributions: dict[str, float] | None = None,
    splits: dict[str, float] | None = None,
    raw_shares: float = 1_000_000.0,
) -> pd.DataFrame:
    """A vendor-shaped file whose back-adjustment factor is known exactly.

    The traded price path is deterministic.  Distributions and splits are
    applied to it, the adjustment factor is accumulated from the same steps the
    loader inverts, and ``Close`` is the product.  Recovering the schedule from
    ``Close / Unadjusted Close`` is therefore an exact arithmetic identity and
    the reconciliation test can assert equality rather than a tolerance band.
    """

    distributions = {} if distributions is None else distributions
    splits = {} if splits is None else splits
    count = len(sessions)
    wave = np.sin(np.arange(count) / 23.0) * 0.01
    traded = np.empty(count)
    traded[0] = start_price
    factor = np.ones(count)
    stamps = [pd.Timestamp(d) for d in sessions]
    for position in range(1, count):
        step = 1.0 + drift + wave[position] - wave[position - 1]
        previous = traded[position - 1]
        traded[position] = previous * step
        adjustment = 1.0
        key = stamps[position].date().isoformat()
        if key in distributions:
            amount = distributions[key]
            traded[position] -= amount
            adjustment *= 1.0 / (1.0 - amount / previous)
        if key in splits:
            ratio = splits[key]
            traded[position] /= ratio
            adjustment *= ratio
        factor[position] = factor[position - 1] * adjustment
    factor = factor / factor[-1]
    close = factor * traded
    volume = raw_shares / factor
    turnover = raw_shares * traded
    return pd.DataFrame(
        {
            "Date": [s.date().isoformat() for s in stamps],
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": volume,
            "Turnover": turnover,
            "Unadjusted Close": traded,
            "Dividend": np.zeros(count),
            "Constituent_S&P 500": np.zeros(count, dtype=int),
            "Constituent_S&P MidCap 400": np.zeros(count, dtype=int),
            "Constituent_S&P SmallCap 600": np.zeros(count, dtype=int),
        }
    )


SYNTHETIC_DISTRIBUTIONS = {
    "2001-03-15": 0.31,
    "2001-06-15": 0.29,
    "2001-09-14": 0.33,
    "2001-12-14": 0.42,
    "2002-03-15": 0.30,
}
SYNTHETIC_SPLIT = {"2002-06-14": 3.0}


class SyntheticExtract:
    """A temporary directory laid out exactly like the vendor extract."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="etf-panel-"))
        (self.root / "ETF Data").mkdir()
        self.calendar = pd.bdate_range("2000-01-03", "2003-12-31")
        self.specs = {
            "AAA": {
                "start": self.calendar[0],
                "price": 100.0,
                "distributions": SYNTHETIC_DISTRIBUTIONS,
                "splits": SYNTHETIC_SPLIT,
                "shares": 4_000_000.0,
            },
            "BBB": {
                "start": self.calendar[0],
                "price": 60.0,
                "distributions": {},
                "splits": {},
                "shares": 800_000.0,
            },
            "CCC": {
                "start": self.calendar[250],
                "price": 25.0,
                "distributions": {},
                "splits": {},
                "shares": 90_000.0,
            },
        }
        #: BBB skips two sessions, so the intersection alignment has something
        #: real to drop and the coverage-gap column has something to count.
        self.skipped = [self.calendar[400], self.calendar[401]]
        rows = []
        for ticker, spec in self.specs.items():
            sessions = self.calendar[self.calendar >= spec["start"]]
            if ticker == "BBB":
                sessions = sessions.difference(pd.DatetimeIndex(self.skipped))
            frame = synthetic_fund(
                sessions,
                start_price=float(spec["price"]),
                distributions=dict(spec["distributions"]),
                splits=dict(spec["splits"]),
                raw_shares=float(spec["shares"]),
            )
            frame.to_csv(self.root / "ETF Data" / f"{ticker}.csv", index=False)
            first = sessions[0]
            rows.append(
                {
                    "symbol": ticker,
                    "assetid_norgate": 100000 + len(rows),
                    "securityname_norgate": f"{ticker} Synthetic Fund",
                    "exchange_name": "NYSE",
                    "exchange_name_full": "New York Stock Exchange",
                    "base_type": "Stock Market",
                    "subtype1": "Exchange Traded Product",
                    "subtype2": "Exchange Traded Fund (ETF)",
                    "subtype3": "",
                    "financial_summary": "",
                    "business_summary": "synthetic",
                    # Day first, and above twelve, so a month-first parse would raise.
                    "first_quoted_date": f"{first.day}/{first.month}/{first.strftime('%y')}",
                }
            )
        pd.DataFrame(rows).to_csv(self.root / "CATALOGUE_Delta1_ETF.csv", index=False)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class UniverseDeclarationTests(unittest.TestCase):
    def test_the_universe_is_one_fund_per_exposure_and_frozen(self) -> None:
        self.assertEqual(len(UNIVERSE), 11)
        self.assertEqual(len(set(UNIVERSE_TICKERS)), len(UNIVERSE_TICKERS))
        exposures = [member.exposure for member in UNIVERSE]
        self.assertEqual(len(set(exposures)), len(exposures))
        classes = {member.asset_class for member in UNIVERSE}
        self.assertEqual(
            classes, {"equity", "rates", "credit", "commodity", "real_estate"}
        )
        with self.assertRaises(AttributeError):
            UNIVERSE[0].ticker = "XXX"  # type: ignore[misc]

    def test_no_fund_is_both_selected_and_excluded(self) -> None:
        selected = set(UNIVERSE_TICKERS)
        excluded = {candidate.ticker for candidate in EXCLUDED_CANDIDATES}
        self.assertEqual(selected & excluded, set())
        self.assertGreaterEqual(len(excluded), 20)

    def test_a_selection_reason_may_not_argue_from_results(self) -> None:
        # The pre-declaration is enforced rather than asserted: a reason that
        # reaches for a performance statistic is refused at construction.
        for phrase in (
            "the strongest return of the candidates",
            "the best Sharpe among peers over the sample",
            "the smallest drawdown of the group in stress",
        ):
            with self.assertRaises(ValueError):
                UniverseMember(
                    ticker="ZZZ",
                    exposure="test",
                    asset_class="equity",
                    first_quoted_date="2000-01-03",
                    panel_window_median_daily_turnover_usd=1e8,
                    selection_reason=phrase + " and therefore selected here",
                )
            with self.assertRaises(ValueError):
                ExcludedCandidate("ZZZ", "test", phrase)

    def test_member_validation_refuses_bad_fields(self) -> None:
        base = {
            "ticker": "ZZZ",
            "exposure": "test",
            "asset_class": "equity",
            "first_quoted_date": "2000-01-03",
            "panel_window_median_daily_turnover_usd": 1e8,
            "selection_reason": (
                "Oldest fund carrying this exposure and the most liquid by a wide "
                "margin over every peer in the extract."
            ),
        }
        UniverseMember(**base)
        for override in (
            {"ticker": ""},
            {"ticker": "zzz"},
            {"exposure": "  "},
            {"first_quoted_date": "not-a-date"},
            {"panel_window_median_daily_turnover_usd": True},
            {"panel_window_median_daily_turnover_usd": 0.0},
            {"panel_window_median_daily_turnover_usd": float("nan")},
            {"selection_reason": "short"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                UniverseMember(**{**base, **override})

    def test_the_universe_tables_carry_the_survivorship_disclosure(self) -> None:
        table = universe_table()
        self.assertEqual(len(table), len(UNIVERSE))
        self.assertTrue((table["Limitation"] == SURVIVORSHIP_LIMITATION).all())
        self.assertTrue(
            (
                table["Selection inputs"]
                == "asset-class coverage, inception date, liquidity"
            ).all()
        )
        self.assertEqual(len(excluded_candidate_table()), len(EXCLUDED_CANDIDATES))

    def test_the_excluded_table_carries_the_disclosure_too(self) -> None:
        # The rejection list is where the caveat is most load-bearing: every
        # fund named here was considered from a survivors-only pool, and no
        # closed fund appears among the rejections because none is present to
        # reject.  A table of twenty-five losers with no disclosure reads as a
        # complete competitive field, which it is not.
        table = excluded_candidate_table()
        self.assertIn("Limitation", table.columns)
        self.assertTrue((table["Limitation"] == SURVIVORSHIP_LIMITATION).all())

    def test_the_turnover_column_names_the_window_it_was_measured_over(self) -> None:
        # The declared figures are panel-window medians, not life medians, and
        # the two differ by up to 2.6x.  The heading has to say which, or the
        # pre-declared rule cannot be reproduced from the data.
        first, last = TURNOVER_MEASUREMENT_WINDOW
        self.assertIn(first, TURNOVER_COLUMN)
        self.assertIn(last, TURNOVER_COLUMN)
        self.assertNotIn("Life median", TURNOVER_COLUMN)
        table = universe_table()
        self.assertIn(TURNOVER_COLUMN, table.columns)
        self.assertEqual(
            [float(value) for value in table[TURNOVER_COLUMN]],
            [
                float(member.panel_window_median_daily_turnover_usd)
                for member in UNIVERSE
            ],
        )


class SyntheticLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extract = SyntheticExtract()
        cls.root = cls.extract.root

    @classmethod
    def tearDownClass(cls) -> None:
        cls.extract.cleanup()

    def test_the_catalogue_date_is_parsed_day_first(self) -> None:
        catalogue = load_catalogue(self.root)
        self.assertEqual(len(catalogue), 3)
        parsed = dict(zip(catalogue["symbol"], catalogue["first_quoted_date"]))
        for ticker in ("AAA", "BBB", "CCC"):
            fund = load_fund(ticker, self.root)
            self.assertEqual(parsed[ticker], fund.index.min())
        # A month-first parse of the same field raises rather than transposing,
        # because every synthetic inception day exceeds twelve.
        raw = pd.read_csv(self.root / "CATALOGUE_Delta1_ETF.csv")
        with self.assertRaises(ValueError):
            pd.to_datetime(raw["first_quoted_date"], format="%m/%d/%y", errors="raise")

    def test_the_loader_recovers_shares_and_the_turnover_identity(self) -> None:
        fund = load_fund("AAA", self.root)
        self.assertEqual(list(fund.columns)[:6], [
            "adjusted_close",
            "unadjusted_close",
            "volume",
            "shares_traded",
            "turnover_usd",
            "dividend",
        ])
        np.testing.assert_allclose(fund["shares_traded"], 4_000_000.0, rtol=1e-12)
        identity = fund["turnover_usd"] / (fund["volume"] * fund["adjusted_close"])
        np.testing.assert_allclose(identity, 1.0, rtol=1e-12)
        # The vendor's volume is on the adjusted scale, so the naive product is
        # wrong by the adjustment factor -- by construction, not by accident.
        naive = fund["turnover_usd"] / (fund["volume"] * fund["unadjusted_close"])
        self.assertLess(float(naive.iloc[0]), 0.9)

    def test_the_distribution_schedule_is_recovered_exactly(self) -> None:
        events = adjustment_events("AAA", data_dir=self.root)
        distributions = events[events["Event kind"] == "distribution"]
        splits = events[events["Event kind"] == "split"]
        self.assertEqual(len(distributions), len(SYNTHETIC_DISTRIBUTIONS))
        self.assertEqual(len(splits), 1)
        for stamp, amount in SYNTHETIC_DISTRIBUTIONS.items():
            recovered = float(
                distributions.loc[pd.Timestamp(stamp), "Implied distribution per share"]
            )
            self.assertAlmostEqual(recovered, amount, places=9)
        self.assertAlmostEqual(
            float(splits["Adjustment step"].iloc[0]), 3.0, places=12
        )

    def test_a_fund_with_no_distribution_has_an_identical_adjusted_close(self) -> None:
        # The analytic control.  Without an event the adjustment is not a
        # generic rescale, so the two price columns coincide bit for bit.
        fund = load_fund("BBB", self.root)
        pd.testing.assert_series_equal(
            fund["adjusted_close"], fund["unadjusted_close"], check_names=False
        )
        self.assertTrue(adjustment_events("BBB", data_dir=self.root).empty)

    def test_the_reconciliation_report_states_the_empty_dividend_column(self) -> None:
        report = dividend_reconciliation_report(("AAA", "BBB"), data_dir=self.root)
        row = report.set_index("Ticker").loc["AAA"]
        self.assertEqual(int(row["Distribution events"]), 5)
        self.assertEqual(int(row["Split events"]), 1)
        self.assertEqual(int(row["Nonzero vendor dividend rows"]), 0)
        self.assertAlmostEqual(float(row["Split ratio product"]), 3.0, places=12)
        self.assertLess(float(row["Maximum absolute quiet-session step difference"]), 1e-12)
        control = report.set_index("Ticker").loc["BBB"]
        self.assertEqual(int(control["Adjustment events"]), 0)
        self.assertEqual(
            float(control["Maximum absolute quiet-session step difference"]), 0.0
        )

    def test_the_panel_takes_the_intersection_and_never_fills_forward(self) -> None:
        panel = load_panel(("AAA", "BBB"), data_dir=self.root)
        alignment = panel.alignment.iloc[0]
        self.assertEqual(int(alignment["Sessions dropped by intersection"]), 2)
        self.assertEqual(
            int(alignment["Union sessions"]) - int(alignment["Intersection sessions"]), 2
        )
        for stamp in self.extract.skipped:
            self.assertNotIn(stamp, panel.sessions)
        self.assertFalse(panel.adjusted_close.isna().to_numpy().any())
        self.assertEqual(len(panel.returns), len(panel.adjusted_close) - 1)
        self.assertIn(SURVIVORSHIP_LIMITATION, panel.limitations)

    def test_the_panel_refuses_a_start_before_a_member_exists(self) -> None:
        with self.assertRaises(ValueError):
            load_panel(("AAA", "CCC"), start="2000-01-03", data_dir=self.root)
        # Starting after every inception is fine and pads nothing.
        panel = load_panel(("AAA", "CCC"), start="2001-01-03", data_dir=self.root)
        self.assertEqual(panel.first_session, pd.Timestamp("2001-01-03"))

    def test_the_custody_guard_refuses_a_row_beyond_the_permitted_session(self) -> None:
        with self.assertRaises(ValueError):
            load_panel(
                ("AAA", "BBB"),
                end="2003-12-31",
                last_session="2002-12-31",
                data_dir=self.root,
            )
        panel = load_panel(("AAA", "BBB"), last_session="2002-12-31", data_dir=self.root)
        self.assertLessEqual(panel.last_session, pd.Timestamp("2002-12-31"))

    def test_the_availability_report_counts_real_coverage_gaps(self) -> None:
        report = availability_report(("AAA", "BBB", "CCC"), data_dir=self.root).set_index(
            "Ticker"
        )
        self.assertEqual(int(report.loc["BBB", "Coverage gap sessions"]), 2)
        self.assertEqual(int(report.loc["BBB", "Longest coverage gap (sessions)"]), 2)
        self.assertEqual(int(report.loc["AAA", "Coverage gap sessions"]), 0)
        # A later inception is a later inception, never a hole.
        self.assertEqual(int(report.loc["CCC", "Coverage gap sessions"]), 0)
        self.assertTrue(
            (report["Universe fully quoted from"] == "2001-01-01").all()
            or (
                report["Universe fully quoted from"]
                == fully_quoted_date(("AAA", "BBB", "CCC"), data_dir=self.root)
                .date()
                .isoformat()
            ).all()
        )
        self.assertTrue((report["Nonzero vendor dividend rows"] == 0).all())
        self.assertTrue((report["Nonzero index membership rows"] == 0).all())

    def test_turnover_summaries_are_available_per_year(self) -> None:
        table = median_annual_turnover(("AAA", "CCC"), data_dir=self.root)
        self.assertEqual(list(table.columns), ["AAA", "CCC"])
        self.assertTrue(table.index.min() >= 2000)
        thin = thin_fund_years(
            ("AAA", "CCC"), min_median_turnover_usd=5_000_000.0, data_dir=self.root
        )
        self.assertTrue(set(thin["Ticker"]).issubset({"AAA", "CCC"}))

    def test_reports_carry_no_banned_column_name(self) -> None:
        frames = [
            universe_table(),
            excluded_candidate_table(),
            availability_report(("AAA", "BBB"), data_dir=self.root),
            dividend_reconciliation_report(("AAA", "BBB"), data_dir=self.root),
            adjustment_events("AAA", data_dir=self.root),
            thin_fund_years(("AAA",), data_dir=self.root),
            load_panel(("AAA", "BBB"), data_dir=self.root).alignment,
        ]
        for position, frame in enumerate(frames):
            for column in frame.columns:
                lowered = str(column).lower()
                for banned in BANNED_SUBSTRINGS:
                    with self.subTest(frame=position, column=column, banned=banned):
                        self.assertNotIn(banned, lowered)


class LiquidityScreenTests(unittest.TestCase):
    def turnover(self, periods: int = 900, seed: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        index = pd.bdate_range("2000-01-03", periods=periods)
        thick = pd.Series(rng.lognormal(19.0, 0.3, periods), index=index)
        ramp = pd.Series(
            np.exp(np.linspace(14.5, 19.5, periods)) * rng.lognormal(0, 0.2, periods),
            index=index,
        )
        late = pd.Series(rng.lognormal(18.5, 0.3, periods), index=index)
        late.iloc[:400] = np.nan
        return pd.DataFrame({"THICK": thick, "RAMP": ramp, "LATE": late})

    def test_the_mask_is_truncation_invariant(self) -> None:
        # The causality proof that matters: values computed on a longer history
        # must be identical on the overlap.  Any forward-looking term --- a
        # centred window, a backward fill, a full-sample normalisation --- would
        # make the earlier values move when later data is appended.
        frame = self.turnover()
        full = tradeable_mask(frame, window=63)
        truncated = tradeable_mask(frame.iloc[:600], window=63)
        overlap = truncated.index
        self.assertGreater(len(overlap), 500)
        pd.testing.assert_frame_equal(full.loc[overlap], truncated)

    def test_membership_and_entry_dates_are_truncation_invariant(self) -> None:
        frame = self.turnover()
        full = causal_membership_mask(
            frame, liquidity_window=63, min_history_sessions=120
        )
        truncated = causal_membership_mask(
            frame.iloc[:600], liquidity_window=63, min_history_sessions=120
        )
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)
        early = causal_entry_dates(
            frame.iloc[:600], liquidity_window=63, min_history_sessions=120
        )
        late = causal_entry_dates(frame, liquidity_window=63, min_history_sessions=120)
        for ticker in early.index:
            if pd.notna(early[ticker]):
                self.assertEqual(early[ticker], late[ticker])

    def test_lagged_median_turnover_is_truncation_invariant(self) -> None:
        frame = self.turnover()
        full = lagged_median_turnover(frame, window=60)
        truncated = lagged_median_turnover(frame.iloc[:600], window=60)
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)

    def test_the_executing_session_cannot_admit_a_fund(self) -> None:
        frame = self.turnover()
        base = tradeable_mask(frame, window=63)
        shocked = frame.copy()
        shocked.iloc[300] = shocked.iloc[300] * 1e6
        after = tradeable_mask(shocked, window=63)
        self.assertEqual(bool(base.iloc[300, 0]), bool(after.iloc[300, 0]))
        self.assertEqual(bool(base.iloc[300, 1]), bool(after.iloc[300, 1]))

    def test_the_mask_consumes_the_window_before_admitting_anything(self) -> None:
        frame = self.turnover()
        mask = tradeable_mask(frame, window=63, lag=1)
        self.assertFalse(mask.iloc[:63].to_numpy().any())
        self.assertTrue(bool(mask.iloc[63:, 0].any()))

    def test_a_fund_that_does_not_quote_is_never_tradeable(self) -> None:
        frame = self.turnover()
        mask = tradeable_mask(frame, window=63)
        self.assertFalse(bool(mask["LATE"].iloc[:400].any()))

    def test_capacity_scales_with_the_participation_limit(self) -> None:
        frame = self.turnover()
        one = participation_capacity_usd(frame, participation_limit=0.05, window=60)
        two = participation_capacity_usd(frame, participation_limit=0.10, window=60)
        ratio = (two / one).dropna()
        np.testing.assert_allclose(ratio.to_numpy(), 2.0, rtol=1e-12)

    def test_invalid_arguments_are_refused(self) -> None:
        frame = self.turnover(periods=100)
        for kwargs in (
            {"window": 1},
            {"window": True},
            {"lag": 0},
            {"lag": True},
            {"min_median_turnover_usd": -1.0},
            {"min_median_turnover_usd": True},
            {"min_median_turnover_usd": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tradeable_mask(frame, **kwargs)
        with self.assertRaises(ValueError):
            causal_membership_mask(frame, min_history_sessions=0)
        with self.assertRaises(ValueError):
            participation_capacity_usd(frame, participation_limit=0.0)


class PanelValidationTests(unittest.TestCase):
    def panel_kwargs(self) -> dict[str, object]:
        index = pd.bdate_range("2010-01-04", periods=20)
        columns = ["AAA", "BBB"]
        prices = pd.DataFrame(
            np.linspace(100.0, 120.0, 40).reshape(20, 2), index=index, columns=columns
        )
        ones = pd.DataFrame(1.0, index=index, columns=columns)
        zeros = pd.DataFrame(0.0, index=index, columns=columns)
        return {
            "tickers": ("AAA", "BBB"),
            "adjusted_close": prices,
            "unadjusted_close": prices,
            "volume": ones,
            "shares_traded": ones,
            "turnover_usd": ones,
            "dividend": zeros,
            "index_membership_sp_500": zeros,
            "index_membership_sp_midcap_400": zeros,
            "index_membership_sp_smallcap_600": zeros,
            "returns": prices.div(prices.shift(1)).sub(1.0).iloc[1:],
            "alignment": pd.DataFrame([{"Intersection sessions": 20}]),
        }

    def test_a_valid_panel_is_accepted(self) -> None:
        panel = EtfPanel(**self.panel_kwargs())
        self.assertEqual(panel.tickers, ("AAA", "BBB"))
        self.assertEqual(len(panel.sessions), 20)

    def test_the_panel_refuses_inconsistent_frames(self) -> None:
        base = self.panel_kwargs()
        with self.assertRaises(ValueError):
            EtfPanel(**{**base, "tickers": ("AAA", "AAA")})
        broken = base["turnover_usd"].copy()
        broken.iloc[3, 1] = np.nan
        with self.assertRaises(ValueError):
            EtfPanel(**{**base, "turnover_usd": broken})
        with self.assertRaises(ValueError):
            EtfPanel(**{**base, "volume": base["volume"].iloc[:10]})
        with self.assertRaises(ValueError):
            EtfPanel(**{**base, "limitations": ("something else",)})

    def test_the_survivorship_disclosure_cannot_be_dropped(self) -> None:
        base = self.panel_kwargs()
        with self.assertRaises(ValueError):
            EtfPanel(**{**base, "limitations": ()})


@unittest.skipUnless(
    REAL_PRICE_DIR.is_dir(), "Supplied DELTA1 ETF data directory is not available"
)
class SuppliedEtfDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = load_panel(
            UNIVERSE_TICKERS, start="2006-02-03", end="2018-12-31", data_dir=DATA_DIR
        )
        cls.reconciliation = dividend_reconciliation_report(
            UNIVERSE_TICKERS, data_dir=DATA_DIR
        ).set_index("Ticker")

    def test_the_extract_has_the_stated_shape(self) -> None:
        files = sorted(REAL_PRICE_DIR.glob("*.csv"))
        self.assertEqual(len(files), 745)
        catalogue = load_catalogue(DATA_DIR)
        self.assertEqual(len(catalogue), 745)
        self.assertEqual(
            set(catalogue["symbol"]), {path.stem for path in files}
        )
        sample = pd.read_csv(files[0], nrows=1)
        self.assertEqual(tuple(sample.columns), PRICE_COLUMNS)

    def test_spy_prices_match_the_published_pair(self) -> None:
        spy = load_fund("SPY", DATA_DIR)
        self.assertAlmostEqual(
            float(spy.loc["2018-12-31", "adjusted_close"]), 230.31606, places=5
        )
        self.assertAlmostEqual(
            float(spy.loc["2018-12-31", "unadjusted_close"]), 249.92, places=2
        )
        self.assertEqual(spy.index.min(), pd.Timestamp("1993-01-29"))
        self.assertEqual(len(spy), 6528)

    def test_the_vendor_dividend_and_membership_columns_are_empty(self) -> None:
        report = availability_report(UNIVERSE_TICKERS, data_dir=DATA_DIR)
        self.assertTrue((report["Nonzero vendor dividend rows"] == 0).all())
        self.assertTrue((report["Nonzero index membership rows"] == 0).all())

    def test_the_recovered_distributions_match_the_published_schedule(self) -> None:
        # The reconciliation the brief asks for, by the only route the data
        # permits: the Dividend column is empty, so the schedule is recovered
        # from the adjustment factor and checked against the published figures.
        spy = adjustment_events("SPY", data_dir=DATA_DIR)
        published = {
            "2018-12-21": 1.43539,
            "2018-09-21": 1.32277,
            "2018-06-15": 1.24564,
            "2018-03-16": 1.09675,
            "2017-12-15": 1.35130,
        }
        for stamp, amount in published.items():
            recovered = float(
                spy.loc[pd.Timestamp(stamp), "Implied distribution per share"]
            )
            self.assertAlmostEqual(recovered, amount, places=3)
        tlt = adjustment_events("TLT", data_dir=DATA_DIR)
        self.assertAlmostEqual(
            float(tlt.loc["2018-12-18", "Implied distribution per share"]),
            0.281894,
            places=6,
        )

    def test_gold_is_the_analytic_control(self) -> None:
        gld = load_fund("GLD", DATA_DIR)
        self.assertEqual(gld["adjustment_factor"].nunique(), 1)
        pd.testing.assert_series_equal(
            gld["adjusted_close"], gld["unadjusted_close"], check_names=False
        )
        self.assertTrue(adjustment_events("GLD", data_dir=DATA_DIR).empty)
        self.assertEqual(
            float(self.reconciliation.loc["GLD", "Implied annual distribution yield"]),
            0.0,
        )

    def test_the_ishares_split_day_is_recovered_at_exactly_three(self) -> None:
        efa = adjustment_events("EFA", data_dir=DATA_DIR)
        row = efa.loc[pd.Timestamp("2005-06-09")]
        self.assertEqual(row["Event kind"], "split")
        # Three to the print precision of the source file, which stores prices
        # as float32; the residual is 1.3e-7, not a partial split.
        self.assertAlmostEqual(float(row["Adjustment step"]), 3.0, places=6)
        self.assertAlmostEqual(float(row["Prior traded price"]), 157.00, places=2)

    def test_the_implied_yields_are_economically_plausible(self) -> None:
        yields = self.reconciliation["Implied annual distribution yield"]
        cadence = self.reconciliation["Distribution events per year"]
        self.assertAlmostEqual(float(yields["SPY"]), 0.0192, places=3)
        self.assertAlmostEqual(float(cadence["SPY"]), 4.09, places=1)
        self.assertAlmostEqual(float(cadence["TLT"]), 11.99, places=1)
        self.assertGreater(float(yields["IYR"]), float(yields["SPY"]))
        for ticker in UNIVERSE_TICKERS:
            with self.subTest(ticker=ticker):
                self.assertLessEqual(float(yields[ticker]), 0.06)
                self.assertGreaterEqual(float(yields[ticker]), 0.0)
                self.assertLess(
                    float(
                        self.reconciliation.loc[
                            ticker, "Maximum absolute quiet-session step difference"
                        ]
                    ),
                    1e-6,
                )

    def test_the_window_panel_is_complete_and_aligned(self) -> None:
        alignment = self.panel.alignment.iloc[0]
        self.assertEqual(int(alignment["Union sessions"]), 3249)
        self.assertEqual(int(alignment["Intersection sessions"]), 3249)
        self.assertEqual(int(alignment["Sessions dropped by intersection"]), 0)
        self.assertEqual(self.panel.returns.shape, (3248, 11))
        self.assertFalse(self.panel.returns.isna().to_numpy().any())
        self.assertEqual(fully_quoted_date(UNIVERSE_TICKERS, data_dir=DATA_DIR),
                         pd.Timestamp("2006-02-03"))

    def test_turnover_is_notional_on_the_adjusted_scale(self) -> None:
        for ticker in UNIVERSE_TICKERS:
            fund = load_fund(ticker, DATA_DIR)
            identity = fund["turnover_usd"] / (fund["volume"] * fund["adjusted_close"])
            with self.subTest(ticker=ticker):
                self.assertAlmostEqual(float(identity.median()), 1.0, places=3)

    def test_iyr_is_the_only_fund_with_a_coverage_gap(self) -> None:
        report = availability_report(UNIVERSE_TICKERS, data_dir=DATA_DIR).set_index(
            "Ticker"
        )
        self.assertEqual(int(report.loc["IYR", "Coverage gap sessions"]), 5)
        others = report.drop(index="IYR")["Coverage gap sessions"]
        self.assertTrue((others == 0).all())

    def test_the_liquidity_screen_admits_the_whole_universe_before_the_crisis(self) -> None:
        turnover = pd.DataFrame(
            {t: load_fund(t, DATA_DIR)["turnover_usd"] for t in UNIVERSE_TICKERS}
        )
        entries = causal_entry_dates(turnover, min_median_turnover_usd=10_000_000.0)
        self.assertFalse(entries.isna().any())
        self.assertEqual(entries.max(), pd.Timestamp("2008-04-30"))
        self.assertEqual(entries.idxmax(), "DBC")
        strict = causal_entry_dates(turnover, min_median_turnover_usd=25_000_000.0)
        # The stricter gate keeps credit and belly duration out of the book for
        # the whole 2008 crisis, which is the episode the sleeve exists for.
        self.assertGreater(strict["LQD"], pd.Timestamp("2008-12-31"))
        self.assertGreater(strict["IEF"], pd.Timestamp("2008-01-01"))

    def test_the_thin_fund_years_inside_the_window_are_the_stated_seven(self) -> None:
        thin = thin_fund_years(
            UNIVERSE_TICKERS,
            min_median_turnover_usd=25_000_000.0,
            start="2006-01-01",
            data_dir=DATA_DIR,
        )
        self.assertEqual(len(thin), 7)
        self.assertEqual(set(thin["Ticker"]), {"DBC", "IEF", "LQD"})
        self.assertEqual(thin["Year"].max(), 2008)

    def test_the_real_panel_is_truncation_invariant(self) -> None:
        early = load_panel(
            UNIVERSE_TICKERS, start="2006-02-03", end="2015-06-30", data_dir=DATA_DIR
        )
        overlap = early.sessions
        pd.testing.assert_frame_equal(
            self.panel.adjusted_close.loc[overlap], early.adjusted_close
        )
        pd.testing.assert_frame_equal(
            self.panel.turnover_usd.loc[overlap], early.turnover_usd
        )
        pd.testing.assert_frame_equal(
            self.panel.returns.loc[early.returns.index], early.returns
        )

    def test_the_declared_turnover_is_reproducible_under_its_stated_definition(
        self,
    ) -> None:
        # The published figure has to be recomputable from the files under the
        # definition the heading gives, or the pre-declared liquidity screen is
        # an assertion.  It is the panel-window median; it is emphatically NOT
        # the life median, and the audit reports both so the gap is visible
        # rather than latent.
        audit = universe_turnover_audit(DATA_DIR).set_index("Ticker")
        self.assertEqual(len(audit), len(UNIVERSE))
        for member in UNIVERSE:
            row = audit.loc[member.ticker]
            with self.subTest(ticker=member.ticker):
                # The declared constants are quoted to three or four significant
                # figures, so agreement is required to a tenth of a percent
                # rather than exactly.
                self.assertLess(
                    abs(float(row["Recomputed over declared"]) - 1.0), 1e-3
                )
        # SPY is the extreme case: 2.6x apart, so a "life median" heading on the
        # declared value would have been wrong by a factor, not a rounding.
        self.assertLess(float(audit.loc["SPY", "Whole-life over declared"]), 0.5)
        self.assertTrue((audit["Whole-life over declared"] <= 1.001).all())

    def test_the_universe_declaration_matches_the_extract(self) -> None:
        catalogue = load_catalogue(DATA_DIR).set_index("symbol")
        for member in UNIVERSE:
            with self.subTest(ticker=member.ticker):
                self.assertEqual(
                    catalogue.loc[member.ticker, "first_quoted_date"],
                    member.first_quoted,
                )
                self.assertEqual(
                    load_fund(member.ticker, DATA_DIR).index.min(), member.first_quoted
                )
        for candidate in EXCLUDED_CANDIDATES:
            with self.subTest(ticker=candidate.ticker):
                self.assertIn(candidate.ticker, catalogue.index)

    def test_every_supplied_file_ends_on_the_same_session(self) -> None:
        # The survivorship defect, restated as a test so it cannot be forgotten.
        last = []
        for path in sorted(REAL_PRICE_DIR.glob("*.csv")):
            tail = pd.read_csv(path, usecols=["Date", "Volume"]).iloc[-1]
            last.append((tail["Date"], float(tail["Volume"])))
        self.assertEqual({stamp for stamp, _ in last}, {"2018-12-31"})
        self.assertTrue(all(volume > 0 for _, volume in last))
        self.assertIn("survivors_only_universe", SURVIVORSHIP_LIMITATION)


if __name__ == "__main__":
    unittest.main()
