"""Tests for the ETF regime allocation sleeve.

Two families of test carry most of the weight here.

The first is truncation invariance, in the pattern of
``tests/test_attribution.py::TrailingCorrelationTests::test_the_series_is
_truncation_invariant``.  Every causal series in this module --- the desired
weights, the simulated path, and the whole walk-forward --- is recomputed on a
truncated history and required to reproduce the longer run's prefix.  That is
the property leakage breaks: a full-sample normalisation, a centred window or an
end-of-sample universe filter anywhere in the stack would make the earlier
values move when later data is appended.

The second is the splice identity.  A spliced replay that always selects the
same candidate must reproduce that candidate's single-shot backtest
elementwise.  Without that check the walk-forward harness could be silently
inserting or dropping a decision at every boundary and every downstream number
would be wrong in a way no summary statistic would reveal.
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.marketdata.etfs import (
    SURVIVORSHIP_LIMITATION,
    UNIVERSE_TICKERS,
    EtfPanel,
)
from delta1_strategy.research.allocation import (
    ALLOCATION_LIMITATIONS,
    DECLARED_CANDIDATES,
    EQUAL_WEIGHT,
    FABER_TREND,
    IN_SAMPLE_BASIS,
    IN_SAMPLE_DEGRADATION_WINDOW,
    INVERSE_VOLATILITY,
    MIXED_BASIS,
    NO_GATE,
    SEALED_BASIS,
    SEALED_PERMITTED_USE,
    SELECTOR_BASIS,
    SIXTY_FORTY_WEIGHTS,
    TIME_SERIES_MOMENTUM,
    AllocationCandidate,
    AllocationConfig,
    AllocationInputs,
    AllocationLedger,
    CostModel,
    WalkForwardPlan,
    _banded_hold,
    _cap_and_normalise,
    annual_boundaries,
    candidate_table,
    comparison_report,
    cost_sensitivity_report,
    desired_weight_frame,
    equal_weight_reference,
    exposure_share_report,
    fixed_weight_frame,
    fold_dispersion_report,
    holdout_custody_report,
    in_sample_out_of_sample_report,
    lagged_capacity,
    load_allocation_inputs,
    membership_mask,
    out_of_sample_accounting,
    paired_reference_inference,
    path_report,
    path_statistics,
    rebalance_sessions,
    reference_ledgers,
    risk_matched_comparison,
    run_walk_forward,
    sealed_block_state_coverage,
    simulate_allocation,
    straddling_episode_report,
    volatility_budget_diagnostic,
)
from delta1_strategy.research.inference import ESTIMATED, NOT_ESTIMABLE


DATA_DIR = Path(
    os.environ.get("DELTA1_DATA_DIR", "Round1AllData/Quant Researcher/Delta1")
)
REAL_PRICE_DIR = DATA_DIR / "ETF Data"

BANNED_SUBSTRINGS = ("target", "pass", "fail", "verdict", "rank", "winner", "recommend")

SESSIONS = 1900


def synthetic_inputs(
    periods: int = SESSIONS,
    seed: int = 11,
    *,
    turnover_multiple: float = 1.0,
    last_session: pd.Timestamp | str | None = None,
) -> AllocationInputs:
    """An eleven-sleeve panel with a common factor and a stress block.

    Built in memory rather than from files so the tests run without the
    licensed extract, and given the real tickers so the asset-class map and the
    cash sleeve resolve exactly as they do in production.  The stress block
    exists so the trend gate actually switches; a panel that only rises would
    make every gate test vacuous.
    """

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2004-01-02", periods=periods)
    tickers = list(UNIVERSE_TICKERS)
    common = rng.normal(0.0004, 0.007, periods)
    stress = slice(int(periods * 0.40), int(periods * 0.52))
    common[stress] = rng.normal(-0.0025, 0.024, common[stress].shape[0])

    closes = {}
    turnovers = {}
    for position, ticker in enumerate(tickers):
        beta = 0.15 + 0.18 * position
        noise = rng.normal(0.0, 0.004 + 0.0009 * position, periods)
        noise[stress] *= 2.0
        returns = beta * common + noise
        closes[ticker] = 100.0 * np.exp(np.cumsum(returns))
        level = 40_000_000.0 * (1.0 + position) * float(turnover_multiple)
        turnovers[ticker] = level * np.exp(rng.normal(0.0, 0.2, periods))

    close = pd.DataFrame(closes, index=index)
    turnover = pd.DataFrame(turnovers, index=index)
    if last_session is not None:
        keep = index <= pd.Timestamp(last_session)
        close, turnover, index = close.loc[keep], turnover.loc[keep], index[keep]

    ones = pd.DataFrame(1.0, index=index, columns=tickers)
    zeros = pd.DataFrame(0.0, index=index, columns=tickers)
    panel = EtfPanel(
        tickers=tuple(tickers),
        adjusted_close=close,
        unadjusted_close=close,
        volume=ones,
        shares_traded=ones,
        turnover_usd=turnover,
        dividend=zeros,
        index_membership_sp_500=zeros,
        index_membership_sp_midcap_400=zeros,
        index_membership_sp_smallcap_600=zeros,
        returns=close.div(close.shift(1)).sub(1.0).iloc[1:],
        alignment=pd.DataFrame([{"Intersection sessions": len(index)}]),
    )
    return AllocationInputs(
        panel=panel,
        history_close=close,
        history_turnover=turnover,
        permitted_last_session=(
            pd.Timestamp(last_session) if last_session is not None else None
        ),
    )


def synthetic_plan(
    inputs: AllocationInputs,
    *,
    first_year: int = 2009,
    last_year: int = 2010,
    sealed_start: str = "2010-01-01",
) -> WalkForwardPlan:
    sessions = inputs.panel.returns.index
    return WalkForwardPlan(
        anchor=sessions[0].date().isoformat(),
        boundaries=annual_boundaries(
            sessions, first_year=first_year, last_year=last_year
        ),
        end=sessions[-1].date().isoformat(),
        sealed_start=sealed_start,
    )


class CostModelTests(unittest.TestCase):
    def test_the_fixed_rate_is_spread_plus_commission_scaled_by_the_dial(self) -> None:
        model = CostModel(half_spread_bps=2.0, commission_bps=0.5)
        self.assertAlmostEqual(model.fixed_cost_rate, 2.5 / 10_000.0, places=15)
        doubled = CostModel(half_spread_bps=2.0, commission_bps=0.5, cost_multiplier=2.0)
        self.assertAlmostEqual(doubled.fixed_cost_rate, 5.0 / 10_000.0, places=15)
        free = CostModel(cost_multiplier=0.0)
        self.assertEqual(free.fixed_cost_rate, 0.0)

    def test_out_of_range_fields_are_refused(self) -> None:
        CostModel()
        for override in (
            {"half_spread_bps": -1.0},
            {"commission_bps": True},
            {"impact_coefficient_bps": float("nan")},
            {"participation_limit": 0.0},
            {"participation_limit": 1.5},
            {"turnover_window": 1},
            {"turnover_window": True},
            {"turnover_lag": 0},
            {"cost_multiplier": -0.5},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                CostModel(**override)

    def test_the_assumption_table_names_every_component(self) -> None:
        table = CostModel().assumption_table()
        self.assertEqual(len(table), 6)
        self.assertIn("cost_multiplier", set(table["Component"]))
        self.assertTrue((table["Limitation"] != "").all())


class AllocationConfigTests(unittest.TestCase):
    def test_out_of_range_fields_are_refused(self) -> None:
        AllocationConfig()
        for override in (
            {"volatility_budget_annualized": 0.0},
            {"volatility_budget_annualized": True},
            {"minimum_sleeve_volatility_annualized": -0.01},
            {"max_sleeve_weight": 0.0},
            {"max_sleeve_weight": 1.5},
            {"max_volatility_scale": 1.5},
            {"no_trade_band": 1.0},
            {"no_trade_band": -0.01},
            {"initial_capital_usd": 0.0},
            {"min_median_turnover_usd": -1.0},
            {"book_volatility_window": 1},
            {"faber_months": True},
            {"annualization": 0},
            {"cash_ticker": ""},
            {"costs": "not a cost model"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                AllocationConfig(**override)

    def test_leverage_is_refused_with_a_message_that_says_why(self) -> None:
        with self.assertRaises(ValueError) as caught:
            AllocationConfig(max_volatility_scale=1.4)
        self.assertIn("leverage decision", str(caught.exception))

    def test_the_cost_dial_produces_a_new_frozen_configuration(self) -> None:
        base = AllocationConfig()
        doubled = base.with_cost_multiplier(2.0)
        self.assertEqual(base.costs.cost_multiplier, 1.0)
        self.assertEqual(doubled.costs.cost_multiplier, 2.0)
        self.assertEqual(doubled.volatility_budget_annualized, base.volatility_budget_annualized)


class CandidateTests(unittest.TestCase):
    def test_the_declared_batch_holds_three_distinct_rules(self) -> None:
        self.assertEqual(len(DECLARED_CANDIDATES), 3)
        names = [candidate.name for candidate in DECLARED_CANDIDATES]
        self.assertEqual(len(set(names)), 3)
        shapes = {
            (candidate.gate, candidate.weighting) for candidate in DECLARED_CANDIDATES
        }
        self.assertEqual(len(shapes), 3)

    def test_a_malformed_candidate_is_refused(self) -> None:
        AllocationCandidate(
            name="ok",
            gate=NO_GATE,
            weighting=EQUAL_WEIGHT,
            claim="a claim long enough to state the economic content of the rule",
            citation="somebody (2000)",
        )
        for override in (
            {"name": ""},
            {"gate": "coin flip"},
            {"weighting": "minimum variance"},
            {"claim": "too short"},
            {"citation": "  "},
        ):
            base = {
                "name": "ok",
                "gate": NO_GATE,
                "weighting": EQUAL_WEIGHT,
                "claim": "a claim long enough to state the economic content of the rule",
                "citation": "somebody (2000)",
            }
            base.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                AllocationCandidate(**base)

    def test_the_candidate_table_carries_the_survivorship_limitation(self) -> None:
        table = candidate_table()
        self.assertTrue((table["Limitation"] == SURVIVORSHIP_LIMITATION).all())


class BudgetArithmeticTests(unittest.TestCase):
    def test_an_uncapped_budget_normalises_to_one(self) -> None:
        weights = _cap_and_normalise(np.array([1.0, 2.0, 1.0]), 1.0)
        np.testing.assert_allclose(weights, [0.25, 0.5, 0.25], rtol=0, atol=1e-15)

    def test_the_ceiling_is_respected_after_the_spill(self) -> None:
        # Cap-then-renormalise would push the freed weight back over the cap.
        # Water-filling must not.
        weights = _cap_and_normalise(np.array([10.0, 1.0, 1.0, 1.0]), 0.30)
        self.assertLessEqual(float(weights.max()), 0.30 + 1e-12)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_a_ceiling_below_one_over_n_leaves_the_shortfall_for_cash(self) -> None:
        weights = _cap_and_normalise(np.array([1.0, 1.0]), 0.30)
        np.testing.assert_allclose(weights, [0.30, 0.30], rtol=0, atol=1e-15)
        self.assertAlmostEqual(float(weights.sum()), 0.60, places=12)

    def test_an_empty_budget_stays_empty(self) -> None:
        weights = _cap_and_normalise(np.array([0.0, 0.0, np.nan]), 0.30)
        np.testing.assert_allclose(weights, [0.0, 0.0, 0.0], rtol=0, atol=0)


class NoTradeBandTests(unittest.TestCase):
    def test_a_gap_inside_the_band_does_not_trade(self) -> None:
        held = np.array([0.10, 0.20, 0.70])
        want = np.array([0.11, 0.20, 0.69])
        banded = _banded_hold(want, held, 0.02, cash_position=2)
        self.assertAlmostEqual(float(banded[0]), 0.10, places=15)
        self.assertAlmostEqual(float(banded[1]), 0.20, places=15)
        self.assertAlmostEqual(float(banded.sum()), 1.0, places=15)

    def test_a_gap_outside_the_band_trades_only_to_the_edge(self) -> None:
        held = np.array([0.10, 0.20, 0.70])
        want = np.array([0.30, 0.20, 0.50])
        banded = _banded_hold(want, held, 0.02, cash_position=2)
        # Partial adjustment: stop 0.02 short of the desired weight.
        self.assertAlmostEqual(float(banded[0]), 0.28, places=15)
        self.assertAlmostEqual(float(banded.sum()), 1.0, places=15)

    def test_the_band_is_symmetric_in_direction(self) -> None:
        held = np.array([0.30, 0.20, 0.50])
        want = np.array([0.10, 0.20, 0.70])
        banded = _banded_hold(want, held, 0.02, cash_position=2)
        self.assertAlmostEqual(float(banded[0]), 0.12, places=15)

    def test_a_zero_band_reproduces_the_desired_vector(self) -> None:
        held = np.array([0.10, 0.20, 0.70])
        want = np.array([0.30, 0.05, 0.65])
        banded = _banded_hold(want, held, 0.0, cash_position=2)
        np.testing.assert_allclose(banded, want, rtol=0, atol=1e-15)


class RebalanceCalendarTests(unittest.TestCase):
    def test_the_book_trades_on_the_session_after_each_month_end(self) -> None:
        index = pd.bdate_range("2020-01-01", "2020-04-30")
        trading = rebalance_sessions(index)
        self.assertEqual(
            [stamp.date().isoformat() for stamp in trading[:3]],
            ["2020-02-03", "2020-03-02", "2020-04-01"],
        )

    def test_the_final_incomplete_month_produces_no_unfillable_decision(self) -> None:
        index = pd.bdate_range("2020-01-01", "2020-02-12")
        trading = rebalance_sessions(index)
        self.assertEqual(list(trading), [pd.Timestamp("2020-02-03")])

    def test_an_empty_calendar_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            rebalance_sessions(pd.DatetimeIndex([]))


class DesiredWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()
        cls.desired = desired_weight_frame(
            cls.inputs, DECLARED_CANDIDATES[0], cls.config
        )

    def test_the_book_is_fully_funded_long_only_and_unlevered(self) -> None:
        weights = self.desired.weights
        np.testing.assert_allclose(
            weights.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-12
        )
        self.assertGreaterEqual(float(weights.to_numpy().min()), -1e-12)
        self.assertLessEqual(float(weights.to_numpy().max()), 1.0 + 1e-12)

    def test_a_sleeve_is_never_wanted_before_it_is_a_member(self) -> None:
        members = membership_mask(self.inputs, self.config)
        risk = self.desired.weights.drop(columns=[self.config.cash_ticker])
        outside = risk.where(~members[risk.columns], 0.0)
        self.assertAlmostEqual(float(outside.to_numpy().sum()), 0.0, places=12)

    def test_a_gated_off_sleeve_holds_no_weight_and_its_budget_goes_to_cash(self) -> None:
        weights = self.desired.weights
        off = ~self.desired.eligibility
        risk_columns = [c for c in weights.columns if c != self.config.cash_ticker]
        gated = weights[risk_columns].where(off[risk_columns], 0.0)
        self.assertAlmostEqual(float(gated.to_numpy().sum()), 0.0, places=12)
        # Faber routes an off sleeve to cash rather than to the sleeves that are
        # on, so the risk book shrinks when the gate closes rather than
        # concentrating.
        risk_sum = weights[risk_columns].sum(axis=1)
        eligible_count = self.desired.eligibility[risk_columns].sum(axis=1)
        live = eligible_count > 0
        self.assertGreater(
            float(np.corrcoef(risk_sum[live], eligible_count[live])[0, 1]), 0.5
        )

    def test_the_sleeve_ceiling_is_respected(self) -> None:
        risk = self.desired.risk_weights
        self.assertLessEqual(
            float(risk.to_numpy().max()), self.config.max_sleeve_weight + 1e-12
        )

    def test_inverse_volatility_weights_track_the_reciprocal_of_volatility(self) -> None:
        candidate = AllocationCandidate(
            name="ungated_inverse_volatility",
            gate=NO_GATE,
            weighting=INVERSE_VOLATILITY,
            claim="an ungated inverse-volatility book used only to check the arithmetic",
            citation="Kirby and Ostdiek (2012)",
        )
        config = AllocationConfig(max_sleeve_weight=1.0)
        desired = desired_weight_frame(self.inputs, candidate, config)
        sessions = self.inputs.panel.sessions
        stamp = sessions[-1]
        risk = desired.risk_weights.loc[stamp]
        risk = risk[risk > 0]
        window = config.sleeve_volatility_window
        position = sessions.get_loc(stamp)
        history = self.inputs.panel.returns.reindex(sessions).iloc[
            position - window : position
        ]
        sigma = history.std(ddof=0) * math.sqrt(config.annualization)
        expected = (1.0 / sigma[risk.index])
        expected = expected / expected.sum()
        np.testing.assert_allclose(
            risk.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-12
        )

    def test_the_volatility_scale_is_zero_before_it_is_estimable(self) -> None:
        scale = self.desired.volatility_scale
        self.assertEqual(float(scale.iloc[0]), 0.0)
        self.assertGreater(float(scale.iloc[-1]), 0.0)
        self.assertLessEqual(
            float(scale.max()), self.config.max_volatility_scale + 1e-12
        )

    def test_the_frame_is_truncation_invariant(self) -> None:
        # The causality proof.  Any full-sample normalisation anywhere in the
        # gate, the membership screen or the volatility scalar would move an
        # earlier weight when later data is appended.
        cut = self.inputs.panel.sessions[1400]
        truncated = synthetic_inputs(last_session=cut)
        short = desired_weight_frame(truncated, DECLARED_CANDIDATES[0], self.config)
        overlap = short.weights.index
        self.assertGreater(len(overlap), 1000)
        pd.testing.assert_frame_equal(
            self.desired.weights.loc[overlap], short.weights, check_exact=True
        )

    def test_a_price_shock_never_moves_an_earlier_weight(self) -> None:
        shocked = synthetic_inputs()
        position = 1200
        stamp = shocked.panel.sessions[position]
        bumped = shocked.history_close.copy()
        bumped.iloc[position:] *= 1.5
        shocked = AllocationInputs(
            panel=EtfPanel(
                tickers=shocked.panel.tickers,
                adjusted_close=bumped.loc[shocked.panel.sessions],
                unadjusted_close=bumped.loc[shocked.panel.sessions],
                volume=shocked.panel.volume,
                shares_traded=shocked.panel.shares_traded,
                turnover_usd=shocked.panel.turnover_usd,
                dividend=shocked.panel.dividend,
                index_membership_sp_500=shocked.panel.index_membership_sp_500,
                index_membership_sp_midcap_400=shocked.panel.index_membership_sp_midcap_400,
                index_membership_sp_smallcap_600=shocked.panel.index_membership_sp_smallcap_600,
                returns=bumped.loc[shocked.panel.sessions]
                .div(bumped.loc[shocked.panel.sessions].shift(1))
                .sub(1.0)
                .iloc[1:],
                alignment=shocked.panel.alignment,
            ),
            history_close=bumped,
            history_turnover=shocked.history_turnover,
            permitted_last_session=None,
        )
        after = desired_weight_frame(shocked, DECLARED_CANDIDATES[0], self.config)
        before = self.desired.weights.loc[:stamp]
        pd.testing.assert_frame_equal(
            before, after.weights.loc[:stamp], check_exact=True
        )

    def test_a_missing_cash_sleeve_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            desired_weight_frame(
                self.inputs, DECLARED_CANDIDATES[0], AllocationConfig(cash_ticker="ZZZ")
            )


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()
        cls.desired = desired_weight_frame(
            cls.inputs, DECLARED_CANDIDATES[0], cls.config
        )
        cls.ledger = simulate_allocation(cls.desired, cls.inputs, cls.config)

    def test_the_book_stays_fully_invested_long_only_on_every_session(self) -> None:
        weights = self.ledger.weights
        np.testing.assert_allclose(
            weights.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-10
        )
        self.assertGreaterEqual(float(weights.to_numpy().min()), -1e-12)

    def test_zero_costs_leave_the_net_return_equal_to_the_gross(self) -> None:
        free = self.config.with_cost_multiplier(0.0)
        ledger = simulate_allocation(self.desired, self.inputs, free)
        pd.testing.assert_series_equal(
            ledger.daily["gross_return"],
            ledger.daily["net_return"],
            check_names=False,
        )

    def test_the_gross_return_is_the_prior_book_dotted_with_the_session(self) -> None:
        weights = self.ledger.weights
        returns = self.inputs.panel.returns
        expected = (weights.shift(1) * returns).sum(axis=1).iloc[1:]
        np.testing.assert_allclose(
            self.ledger.daily["gross_return"].iloc[1:].to_numpy(),
            expected.to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_with_no_band_and_deep_liquidity_the_book_reaches_what_it_wanted(self) -> None:
        config = AllocationConfig(
            no_trade_band=0.0, costs=CostModel(participation_limit=1.0)
        )
        deep = synthetic_inputs(turnover_multiple=1000.0)
        desired = desired_weight_frame(deep, DECLARED_CANDIDATES[0], config)
        ledger = simulate_allocation(desired, deep, config)
        trading = rebalance_sessions(deep.panel.sessions)
        trading = trading[trading.isin(ledger.weights.index)][20:]
        pd.testing.assert_frame_equal(
            ledger.weights.loc[trading],
            desired.weights.loc[trading],
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
        )

    def test_a_thin_sleeve_defers_the_residual_rather_than_overfilling(self) -> None:
        # The liquidity floor is relaxed so the funds are still admissible; the
        # point of the test is the participation cap, not the membership screen.
        thin = synthetic_inputs(turnover_multiple=0.002)
        config = AllocationConfig(min_median_turnover_usd=1_000.0)
        desired = desired_weight_frame(thin, DECLARED_CANDIDATES[0], config)
        ledger = simulate_allocation(desired, thin, config)
        self.assertGreater(int(ledger.daily["order_deferred"].sum()), 50)
        self.assertLess(float(ledger.daily["participation_fill_fraction"].min()), 0.5)
        # Depth can only take away: a fill never exceeds what was requested.
        self.assertLessEqual(
            float(ledger.daily["participation_fill_fraction"].max()), 1.0 + 1e-12
        )

    def test_the_fill_is_monotone_in_liquidity(self) -> None:
        config = AllocationConfig(min_median_turnover_usd=1_000.0)
        thin = synthetic_inputs(turnover_multiple=0.01)
        thick = synthetic_inputs(turnover_multiple=1.0)
        thin_ledger = simulate_allocation(
            desired_weight_frame(thin, DECLARED_CANDIDATES[0], config), thin, config
        )
        thick_ledger = simulate_allocation(
            desired_weight_frame(thick, DECLARED_CANDIDATES[0], config), thick, config
        )
        self.assertLessEqual(
            float(thin_ledger.daily["participation_fill_fraction"].mean()),
            float(thick_ledger.daily["participation_fill_fraction"].mean()),
        )

    def test_costs_only_reduce_the_path(self) -> None:
        previous = None
        for multiplier in (0.0, 1.0, 4.0):
            config = self.config.with_cost_multiplier(multiplier)
            ledger = simulate_allocation(self.desired, self.inputs, config)
            drag = float(ledger.daily["cost_return"].sum())
            self.assertGreaterEqual(drag, 0.0)
            if previous is not None:
                self.assertGreaterEqual(drag, previous)
            previous = drag

    def test_the_path_is_truncation_invariant(self) -> None:
        cut = self.inputs.panel.sessions[1400]
        truncated = synthetic_inputs(last_session=cut)
        short = simulate_allocation(
            desired_weight_frame(truncated, DECLARED_CANDIDATES[0], self.config),
            truncated,
            self.config,
        )
        overlap = short.daily.index
        self.assertGreater(len(overlap), 1000)
        pd.testing.assert_frame_equal(
            self.ledger.daily.loc[overlap], short.daily, check_exact=True
        )

    def test_the_ledger_refuses_a_dropped_survivorship_limitation(self) -> None:
        with self.assertRaises(ValueError):
            AllocationLedger(
                label="x",
                daily=self.ledger.daily,
                weights=self.ledger.weights,
                config=self.config,
                limitations=("nothing to see here",),
            )

    def test_the_ledger_refuses_a_mismatched_weight_index(self) -> None:
        with self.assertRaises(ValueError):
            AllocationLedger(
                label="x",
                daily=self.ledger.daily,
                weights=self.ledger.weights.iloc[:-1],
                config=self.config,
            )


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()

    def test_the_sixty_forty_reference_holds_the_declared_weights(self) -> None:
        frame = fixed_weight_frame(
            self.inputs, self.config, SIXTY_FORTY_WEIGHTS, label="reference"
        )
        last = frame.weights.iloc[-1]
        self.assertAlmostEqual(float(last["SPY"]), 0.60, places=12)
        self.assertAlmostEqual(float(last["IEF"]), 0.40, places=12)
        self.assertAlmostEqual(float(last.sum()), 1.0, places=12)

    def test_a_reference_naming_a_fund_outside_the_panel_is_refused(self) -> None:
        for weights in ({"NOPE": 1.0}, {"SPY": -0.2, "IEF": 0.4}, {"SPY": 1.2}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fixed_weight_frame(
                    self.inputs, self.config, weights, label="reference"
                )

    def test_the_equal_weight_reference_spreads_across_admitted_sleeves(self) -> None:
        frame = equal_weight_reference(self.inputs, self.config)
        members = membership_mask(self.inputs, self.config).iloc[-1]
        risk = frame.risk_weights.iloc[-1]
        expected = 1.0 / float(members.sum())
        for ticker in risk.index:
            if ticker == self.config.cash_ticker:
                continue
            self.assertAlmostEqual(
                float(risk[ticker]), expected if members[ticker] else 0.0, places=12
            )

    def test_the_buy_and_hold_reference_places_one_order_and_then_drifts(self) -> None:
        ledgers = reference_ledgers(self.inputs, self.config)
        drifting = ledgers["sixty_forty_buy_and_hold"]
        rebalanced = ledgers["sixty_forty_rebalanced_monthly"]
        self.assertLess(
            float(drifting.daily["turnover_fraction"].sum()),
            float(rebalanced.daily["turnover_fraction"].sum()),
        )
        # Exactly one decision is taken.  It may still take several sessions to
        # fill, because buying sixty percent of a hundred-million-dollar book in
        # one print is not something the participation cap permits, and the
        # reference is charged the same capacity constraint as the allocator so
        # the comparison stays like for like.
        self.assertEqual(int(drifting.daily["rebalance_session"].sum()), 1)
        traded = drifting.daily.index[drifting.daily["turnover_fraction"] > 1e-12]
        positions = drifting.daily.index.get_indexer(traded)
        np.testing.assert_array_equal(
            np.diff(positions), np.ones(len(positions) - 1, dtype=int)
        )
        first = drifting.daily.index.get_indexer(
            drifting.daily.index[drifting.daily["rebalance_session"]]
        )[0]
        self.assertEqual(int(positions[0]), int(first))

    def test_the_sixty_forty_reference_is_actually_sixty_forty_when_it_lands(
        self,
    ) -> None:
        # The no-trade band is a rebalancing device.  Left switched on it is
        # also an ENTRY device, and it stops a from-flat purchase a band width
        # short on every leg: the reference labelled 60/40 bought 58/38, parked
        # the remaining 4% in the short-duration Treasury sleeve, and --- having
        # only one trade session --- held that stub for the whole replay.  The
        # references drop the band so each is the portfolio its name claims.
        ledgers = reference_ledgers(self.inputs, self.config)
        drifting = ledgers["sixty_forty_buy_and_hold"]
        traded = drifting.daily.index[drifting.daily["turnover_fraction"] > 1e-12]
        landed = drifting.weights.loc[traded[-1]]
        self.assertAlmostEqual(float(landed["SPY"]), 0.60, places=9)
        self.assertAlmostEqual(float(landed["IEF"]), 0.40, places=9)
        self.assertAlmostEqual(float(landed[self.config.cash_ticker]), 0.0, places=9)

        # The equal-weight panel is distorted the same way by a live band, so it
        # is checked on the session a rebalance actually completes on, before
        # the sleeves drift apart again.
        equal = ledgers["equal_weight_panel"]
        settled = equal.daily.index[
            equal.daily["rebalance_session"].to_numpy(dtype=bool)
            & (equal.daily["participation_fill_fraction"].to_numpy(dtype=float) >= 1.0)
            & ~equal.daily["order_deferred"].to_numpy(dtype=bool)
        ]
        self.assertGreater(len(settled), 0)
        held = equal.weights.loc[settled[-1]]
        members = membership_mask(self.inputs, self.config).loc[settled[-1]]
        share = 1.0 / float(members.sum())
        for ticker in held.index:
            with self.subTest(ticker=ticker):
                self.assertAlmostEqual(
                    float(held[ticker]), share if bool(members[ticker]) else 0.0,
                    places=9,
                )

    def test_the_references_keep_every_cost_the_allocator_pays(self) -> None:
        # Dropping the band must not quietly drop the cost model with it, or the
        # comparison stops being like for like on execution.
        ledgers = reference_ledgers(self.inputs, self.config)
        for label, ledger in ledgers.items():
            with self.subTest(path=label):
                self.assertEqual(
                    ledger.config.costs.fixed_cost_rate,
                    self.config.costs.fixed_cost_rate,
                )
                self.assertEqual(
                    ledger.config.costs.participation_limit,
                    self.config.costs.participation_limit,
                )
                self.assertEqual(float(ledger.config.no_trade_band), 0.0)
                self.assertGreater(float(ledger.daily["cost_return"].sum()), 0.0)


class PathStatisticTests(unittest.TestCase):
    def test_a_constant_positive_series_has_no_drawdown_and_no_calmar(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=300)
        series = pd.Series(0.001, index=index)
        statistics = path_statistics(series)
        self.assertAlmostEqual(statistics["max_drawdown"], 0.0, places=12)
        self.assertTrue(math.isnan(statistics["calmar"]))
        self.assertTrue(math.isnan(statistics["sortino"]))
        self.assertAlmostEqual(statistics["hit_rate"], 1.0, places=12)

    def test_the_cagr_is_annualized_on_elapsed_calendar_time(self) -> None:
        # Deliberately not the session-count convention.  ``validation
        # ._path_statistics`` divides accumulated log growth by elapsed calendar
        # years so a CAGR from this sleeve is the same quantity as a CAGR from
        # the futures sleeve, and this test pins that rather than the easier
        # ``1.001 ** 252`` a session-count convention would give.
        index = pd.bdate_range("2010-01-01", periods=252)
        series = pd.Series(0.001, index=index)
        statistics = path_statistics(series, annualization=252)
        years = (index[-1] - index[0]).total_seconds() / (365.2425 * 24 * 60 * 60)
        self.assertAlmostEqual(
            statistics["cagr"], math.expm1(252 * math.log1p(0.001) / years), places=10
        )

    def test_the_period_start_makes_consecutive_windows_tile(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=504)
        series = pd.Series(0.001, index=index)
        whole = path_statistics(series, annualization=252)
        # Anchoring the second half on the last session of the first half makes
        # the two elapsed periods add to the whole, which is exactly what a fold
        # table needs if its CAGRs are to compound to the stitched CAGR.
        first, second = series.iloc[:252], series.iloc[252:]
        left = (first.index[-1] - first.index[0]).days
        right = (second.index[-1] - first.index[-1]).days
        self.assertEqual(left + right, (index[-1] - index[0]).days)
        tiled = path_statistics(
            second, annualization=252, period_start=first.index[-1]
        )
        self.assertGreater(tiled["cagr"], 0.0)
        self.assertLess(abs(tiled["cagr"] - whole["cagr"]), 0.05)

    def test_sortino_penalises_only_the_downside(self) -> None:
        index = pd.bdate_range("2010-01-01", periods=4)
        series = pd.Series([0.02, -0.01, 0.02, -0.01], index=index)
        statistics = path_statistics(series, annualization=4, hac_lags=1)
        downside = math.sqrt((0.01**2 + 0.01**2) / 4)
        self.assertAlmostEqual(
            statistics["sortino"], 0.005 / downside * 2.0, places=12
        )

    def test_an_empty_series_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            path_statistics(pd.Series(dtype=float))


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()
        cls.plan = synthetic_plan(cls.inputs)
        cls.result = run_walk_forward(cls.inputs, cls.config, cls.plan)

    def test_a_selector_that_never_switches_reproduces_the_single_shot_run(self) -> None:
        # The splice identity.  With the choice pinned, the spliced replay must
        # be the anchor candidate's own backtest elementwise: any decision the
        # harness inserted or dropped at a boundary would show up here and
        # nowhere else.
        pinned = run_walk_forward(
            self.inputs,
            self.config,
            self.plan,
            select_fn=lambda training: (DECLARED_CANDIDATES[0].name, 0.0),
        )
        standalone = pinned.candidate_ledgers[DECLARED_CANDIDATES[0].name]
        pd.testing.assert_frame_equal(
            pinned.stitched.daily, standalone.daily, check_exact=True
        )
        pd.testing.assert_frame_equal(
            pinned.stitched.weights, standalone.weights, check_exact=True
        )

    def test_one_candidate_degenerates_to_a_stability_description(self) -> None:
        result = run_walk_forward(
            self.inputs,
            self.config,
            self.plan,
            candidates=DECLARED_CANDIDATES[:1],
        )
        self.assertEqual(result.mode, "frozen_specification_stability")
        self.assertIn("not validation", str(result.summary["permitted_use"]))
        self.assertTrue(result.folds["fold_sharpe"].notna().all())

    def test_the_walk_forward_is_truncation_invariant(self) -> None:
        # The strongest single test in the design.  A full-sample fit, a centred
        # window or an end-of-sample universe filter anywhere in the state stack
        # makes this fail.
        boundary = pd.Timestamp(self.plan.boundaries[-1])
        cut = self.inputs.panel.sessions[
            self.inputs.panel.sessions < boundary
        ][-1]
        truncated = synthetic_inputs(last_session=cut)
        short_plan = WalkForwardPlan(
            anchor=self.plan.anchor,
            boundaries=self.plan.boundaries[:-1],
            end=cut.date().isoformat(),
            sealed_start=self.plan.boundaries[0],
        )
        short = run_walk_forward(truncated, self.config, short_plan)
        overlap = short.stitched.net_return.index
        self.assertGreater(len(overlap), 200)
        pd.testing.assert_series_equal(
            self.result.stitched.net_return.loc[overlap],
            short.stitched.net_return,
            check_exact=True,
        )

    def test_the_selector_never_sees_a_session_at_or_after_its_boundary(self) -> None:
        seen: list[pd.Timestamp] = []

        def spy(training: pd.DataFrame) -> tuple[str, float]:
            seen.append(training.index[-1])
            return DECLARED_CANDIDATES[0].name, 0.0

        run_walk_forward(self.inputs, self.config, self.plan, select_fn=spy)
        for stamp, boundary in zip(seen, self.plan.boundary_stamps):
            self.assertLess(stamp, boundary)

    def test_the_stand_off_trims_the_scored_window_and_refuses_a_stub(self) -> None:
        folds = self.result.folds
        stand_off = self.plan.stand_off_sessions
        self.assertEqual(stand_off, 44)
        for _, row in folds.iterrows():
            self.assertEqual(
                int(row["training_sessions"]) - int(row["selection_sessions"]),
                stand_off,
            )
        starved = WalkForwardPlan(
            anchor=self.plan.anchor,
            boundaries=self.plan.boundaries,
            end=self.plan.end,
            sealed_start=self.plan.sealed_start,
            minimum_selection_sessions=100_000,
        )
        with self.assertRaises(ValueError) as caught:
            run_walk_forward(self.inputs, self.config, starved)
        self.assertIn("selecting on a stub", str(caught.exception))

    def test_the_fold_table_carries_the_declared_schema_and_no_verdict(self) -> None:
        folds = self.result.folds
        self.assertFalse(folds["selection_adjusted"].any())
        for column in folds.columns:
            lowered = column.lower()
            for banned in BANNED_SUBSTRINGS:
                with self.subTest(column=column, banned=banned):
                    self.assertNotIn(banned, lowered)

    def test_a_malformed_plan_is_refused(self) -> None:
        boundaries = self.plan.boundaries
        for override in (
            {"boundaries": ()},
            {"boundaries": (boundaries[-1], boundaries[0])},
            {"anchor": "2030-01-01"},
            {"end": "2000-01-01"},
            {"sealed_start": "1990-01-01"},
            {"purge_sessions": -1},
            {"minimum_selection_sessions": 1},
        ):
            base = {
                "anchor": self.plan.anchor,
                "boundaries": boundaries,
                "end": self.plan.end,
                "sealed_start": self.plan.sealed_start,
            }
            base.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                WalkForwardPlan(**base)

    def test_a_fold_zero_departure_from_the_warm_up_frame_counts_as_a_switch(
        self,
    ) -> None:
        # Everything before the first boundary is the FIRST declared candidate's
        # frame, so a fold-zero choice of anything else is a real discontinuity
        # in the spliced weight vector and the ledger charges the round trip for
        # it.  Recording it as no switch undercounts the switches by one and
        # hides the largest jump in the frame.
        names = [candidate.name for candidate in DECLARED_CANDIDATES]
        self.assertGreater(len(names), 1)
        forced = run_walk_forward(
            self.inputs,
            self.config,
            self.plan,
            select_fn=lambda training: (names[1], 0.0),
        )
        self.assertTrue(bool(forced.folds.loc[0, "selection_switched"]))
        self.assertEqual(int(forced.summary["selection_switches"]), 1)

        pinned = run_walk_forward(
            self.inputs,
            self.config,
            self.plan,
            select_fn=lambda training: (names[0], 0.0),
        )
        self.assertEqual(int(pinned.summary["selection_switches"]), 0)

        # And in general the count is the number of adjacent changes in the
        # sequence the warm-up frame starts.
        chosen = [names[0], *self.result.folds["selected_candidate"].tolist()]
        expected = sum(
            1 for left, right in zip(chosen, chosen[1:]) if left != right
        )
        self.assertEqual(int(self.result.summary["selection_switches"]), expected)

    def test_the_selection_score_and_its_standard_error_share_one_unit(self) -> None:
        # The score comes back per session from ``_select_highest_sharpe`` and
        # the standard error sqrt(annualization / n) belongs to an annualized
        # Sharpe.  Recorded untouched they sat a factor of sqrt(252) apart, so a
        # reader forming a selector t statistic from the two published columns
        # was out by sixteen.  Both are annualized now, like fold_sharpe.
        captured: list[tuple[pd.DataFrame, float]] = []

        def spy(training: pd.DataFrame) -> tuple[str, float]:
            trimmed = training.iloc[: len(training) - self.plan.stand_off_sessions]
            scores = trimmed.mean(axis=0) / trimmed.std(axis=0, ddof=0)
            captured.append((trimmed, float(scores.max())))
            return str(scores.idxmax()), float(scores.max())

        result = run_walk_forward(self.inputs, self.config, self.plan, select_fn=spy)
        folds = result.folds
        self.assertIn("selection_score_annualized_sharpe", folds.columns)
        self.assertIn("selection_standard_error_of_annualized_sharpe", folds.columns)
        self.assertNotIn("selection_score", folds.columns)
        for position, (trimmed, per_session) in enumerate(captured):
            with self.subTest(fold=position):
                self.assertAlmostEqual(
                    float(folds.loc[position, "selection_score_annualized_sharpe"]),
                    per_session * math.sqrt(self.config.annualization),
                    places=12,
                )
                self.assertAlmostEqual(
                    float(
                        folds.loc[
                            position, "selection_standard_error_of_annualized_sharpe"
                        ]
                    ),
                    math.sqrt(self.config.annualization / len(trimmed)),
                    places=12,
                )

    def test_the_sealed_permitted_use_does_not_claim_a_pre_fitting_seal(self) -> None:
        # The selector demonstrably fits inside the sealed block, and the
        # specification around it was written after the whole panel had been
        # read.  The string has to say that rather than assert a seal taken
        # before any fitting decision.
        self.assertNotIn("sealed before any fitting decision", SEALED_PERMITTED_USE)
        self.assertIn("selector still fits inside", SEALED_PERMITTED_USE)
        self.assertIn("after the whole panel had been read", SEALED_PERMITTED_USE)
        self.assertIn("not a prospective track record", SEALED_PERMITTED_USE)

    def test_the_switching_turnover_is_charged_by_the_single_simulation(self) -> None:
        # No separate switching adjustment exists, and none should: the ledger
        # sees the jump in wanted weights at the first rebalance after a switch.
        summary = self.result.summary
        self.assertIn("stitched_annual_one_way_turnover", summary.index)
        self.assertIn("anchor_annual_one_way_turnover", summary.index)
        self.assertGreater(float(summary["stitched_annual_one_way_turnover"]), 0.0)


class OutOfSampleAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()
        cls.plan = synthetic_plan(cls.inputs)
        cls.result = run_walk_forward(cls.inputs, cls.config, cls.plan)
        cls.accounting = out_of_sample_accounting(cls.result)

    def test_the_fold_sessions_sum_to_the_distinct_stitched_sessions(self) -> None:
        combined = self.accounting.loc[
            self.accounting["Claim"] == "Combined out-of-sample record"
        ].iloc[0]
        self.assertEqual(
            int(self.result.folds["sessions"].sum()),
            int(combined["Distinct sessions"]),
        )
        self.assertEqual(int(combined["Sessions double counted"]), 0)

    def test_the_three_units_agree_to_within_the_convention(self) -> None:
        for _, row in self.accounting.iterrows():
            elapsed = float(row["Elapsed years (365.2425)"])
            sessions = float(row["Years at 252 sessions"])
            with self.subTest(claim=row["Claim"]):
                self.assertLess(abs(elapsed - sessions), 0.35)
                self.assertGreaterEqual(
                    int(row["Complete calendar years"]), math.floor(elapsed)
                )

    def test_more_folds_do_not_add_out_of_sample_time(self) -> None:
        # Fold count and out-of-sample time are independent quantities.  This is
        # the arithmetic that keeps a write-up from implying otherwise.
        sessions = self.inputs.panel.returns.index
        semiannual: list[str] = []
        for boundary in self.plan.boundaries:
            stamp = pd.Timestamp(boundary)
            semiannual.append(boundary)
            july = sessions[sessions >= pd.Timestamp(year=stamp.year, month=7, day=1)]
            if len(july) and july[0] < pd.Timestamp(self.plan.end):
                semiannual.append(july[0].date().isoformat())
        dense = WalkForwardPlan(
            anchor=self.plan.anchor,
            boundaries=tuple(dict.fromkeys(semiannual)),
            end=self.plan.end,
            sealed_start=self.plan.sealed_start,
        )
        result = run_walk_forward(self.inputs, self.config, dense)
        accounting = out_of_sample_accounting(result)
        base = self.accounting.loc[
            self.accounting["Claim"] == "Combined out-of-sample record"
        ].iloc[0]
        denser = accounting.loc[
            accounting["Claim"] == "Combined out-of-sample record"
        ].iloc[0]
        self.assertGreater(len(result.folds), len(self.result.folds))
        self.assertEqual(
            int(base["Distinct sessions"]), int(denser["Distinct sessions"])
        )
        self.assertEqual(
            int(base["Complete calendar years"]),
            int(denser["Complete calendar years"]),
        )

    def test_a_mis_stated_session_count_is_caught_rather_than_counted(self) -> None:
        # A copy, not the shared fixture: a test that corrupts the object every
        # other test reads is a test that makes the suite order-dependent.
        broken = run_walk_forward(self.inputs, self.config, self.plan)
        folds = broken.folds.copy()
        folds.loc[0, "sessions"] = int(folds.loc[0, "sessions"]) + 5
        broken.folds = folds
        with self.assertRaises(ValueError):
            out_of_sample_accounting(broken)

    def test_the_dispersion_report_covers_every_fold_statistic(self) -> None:
        dispersion = fold_dispersion_report(self.result.folds)
        self.assertIn("fold_sharpe", set(dispersion["Statistic"]))
        self.assertIn("fold_max_drawdown", set(dispersion["Statistic"]))
        self.assertTrue((dispersion["Folds"] > 0).all())


class EvidenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()
        cls.plan = synthetic_plan(cls.inputs)
        cls.result = run_walk_forward(cls.inputs, cls.config, cls.plan)
        cls.references = reference_ledgers(cls.inputs, cls.config)
        cls.ledgers = {
            "etf_regime_allocation": cls.result.stitched,
            **cls.result.candidate_ledgers,
            **cls.references,
        }

    def test_the_degradation_table_reports_every_statistic_and_the_change(self) -> None:
        report = in_sample_out_of_sample_report(self.result, self.config)
        self.assertIn("Change, in sample to combined out of sample", report.columns)
        self.assertIn("Change, in sample to sealed block", report.columns)
        self.assertEqual(len(report), 11)
        self.assertIn("Sharpe (rf=0)", set(report["Statistic"]))

    def test_the_degradation_table_states_each_window_it_measures(self) -> None:
        # Without bounds on the artifact a reader has to infer the in-sample
        # window from the source, and the neighbouring accounting table names a
        # DIFFERENT window with a similar phrase.  Every window states its own
        # first session, last session and session count, and the in-sample
        # column is named for its start so the two cannot be joined by mistake.
        report = in_sample_out_of_sample_report(self.result, self.config)
        statistics = list(report["Statistic"])
        self.assertEqual(statistics[:3], ["First session", "Last session", "Sessions"])
        self.assertIn(IN_SAMPLE_DEGRADATION_WINDOW, report.columns)
        self.assertNotIn("training span", IN_SAMPLE_DEGRADATION_WINDOW)

        described = report.set_index("Statistic")[IN_SAMPLE_DEGRADATION_WINDOW]
        net = self.result.stitched.net_return
        live = self.result.stitched.daily["book_live"].to_numpy(dtype=bool)
        first_live = net.index[live][0]
        boundary = self.result.plan.boundary_stamps[0]
        window = net.loc[(net.index >= first_live) & (net.index < boundary)]
        self.assertEqual(described["First session"], window.index[0].date().isoformat())
        self.assertEqual(described["Last session"], window.index[-1].date().isoformat())
        self.assertEqual(int(described["Sessions"]), len(window))

        # And the accounting table's own in-sample row is a different, longer
        # window, which is precisely why it no longer shares the phrase.
        accounting = out_of_sample_accounting(self.result)
        row = accounting.loc[
            accounting["Claim"] == "In-sample span, anchor to first boundary"
        ]
        self.assertEqual(len(row), 1)
        self.assertGreaterEqual(
            int(row.iloc[0]["Distinct sessions"]), int(described["Sessions"])
        )

    def test_the_comparison_preserves_the_declared_order_and_adds_no_verdict(self) -> None:
        report = comparison_report(
            self.ledgers, window="synthetic", basis=SELECTOR_BASIS
        )
        self.assertEqual(list(report["Path"]), list(self.ledgers))
        self.assertEqual(len(report), len(self.ledgers))
        for column in report.columns:
            lowered = column.lower()
            for banned in BANNED_SUBSTRINGS:
                with self.subTest(column=column, banned=banned):
                    self.assertNotIn(banned, lowered)

    def test_the_risk_matched_table_leaves_the_matched_path_unchanged(self) -> None:
        matched = risk_matched_comparison(
            self.ledgers,
            reference_label="etf_regime_allocation",
            window="synthetic",
        )
        row = matched.loc[matched["Path"] == "etf_regime_allocation"].iloc[0]
        self.assertAlmostEqual(float(row["Exposure multiplier applied"]), 1.0, places=12)
        # The invariant a constant rescale cannot change.
        for _, other in matched.iterrows():
            self.assertAlmostEqual(
                float(other["Realized volatility"])
                * float(other["Exposure multiplier applied"]),
                float(row["Realized volatility"]),
                places=10,
            )

    def test_the_volatility_budget_diagnostic_reports_the_shortfall(self) -> None:
        scales = {
            candidate.name: desired_weight_frame(
                self.inputs, candidate, self.config
            ).volatility_scale
            for candidate in DECLARED_CANDIDATES
        }
        report = volatility_budget_diagnostic(self.ledgers, self.config, scales=scales)
        self.assertIn("Realized over budget", report.columns)
        self.assertTrue((report["Volatility budget (annualized)"] == 0.07).all())
        row = report.loc[report["Path"] == DECLARED_CANDIDATES[0].name].iloc[0]
        self.assertGreaterEqual(float(row["Share of sessions at the scale ceiling"]), 0.0)
        self.assertLessEqual(float(row["Share of sessions at the scale ceiling"]), 1.0)

    def test_the_exposure_shares_decompose_realized_variance(self) -> None:
        # From the second session, so the first row's absent prior book does not
        # leave a sliver of variance unattributed.
        report = exposure_share_report(
            self.result.stitched,
            self.inputs,
            start=self.inputs.panel.returns.index[1],
        )
        classes = report.loc[~report["Asset class"].str.startswith("sleeve")]
        self.assertAlmostEqual(
            float(classes["Realized risk share"].sum()), 1.0, places=9
        )
        self.assertAlmostEqual(
            float(classes["Mean weight share"].sum()), 1.0, places=6
        )

    def test_the_paired_inference_is_never_selection_adjusted(self) -> None:
        report = paired_reference_inference(
            self.ledgers,
            subject_label="etf_regime_allocation",
            reference_labels=["sixty_forty_rebalanced_monthly", "equal_weight_panel"],
        )
        self.assertEqual(len(report), 2)
        self.assertFalse(report["Selection adjusted"].any())
        self.assertTrue((report["Sessions"] > 0).all())
        self.assertTrue(
            (
                report["Confidence upper (95% one-sided)"]
                >= report["Confidence lower (95% one-sided)"]
            ).all()
        )

    def test_a_straddling_episode_is_not_labelled_in_sample(self) -> None:
        # The bear-market artifact was stamped ``in_sample_training_span`` for a
        # window a quarter of whose sessions sit inside the first out-of-sample
        # fold.  The whole episode now carries a basis that says it straddles,
        # with the split counted off the calendar, and the two halves follow
        # under bases that are true of them.
        sessions = self.result.stitched.daily.index
        boundary = self.result.plan.boundary_stamps[0]
        before_index = sessions[sessions < boundary]
        after_index = sessions[sessions >= boundary]
        start = before_index[-120]
        end = after_index[119]
        report = straddling_episode_report(
            self.ledgers,
            self.result.plan,
            episode="a test episode",
            start=start,
            end=end,
        )
        bases = list(dict.fromkeys(report["Basis"]))
        self.assertEqual(bases, [MIXED_BASIS, IN_SAMPLE_BASIS, SELECTOR_BASIS])
        self.assertEqual(len(report), 3 * len(self.ledgers))

        whole = report.loc[report["Basis"] == MIXED_BASIS]
        covered = sessions[(sessions >= start) & (sessions <= end)]
        self.assertTrue((whole["Sessions"] == len(covered)).all())
        for use in whole["Permitted use"]:
            self.assertIn(
                f"{int((covered < boundary).sum())} of its {len(covered)} sessions",
                use,
            )
            self.assertIn("straddles the first boundary", use)

        in_sample = report.loc[report["Basis"] == IN_SAMPLE_BASIS]
        self.assertTrue(
            (
                pd.to_datetime(in_sample["End"]) < boundary
            ).all()
        )
        out_of_sample = report.loc[report["Basis"] == SELECTOR_BASIS]
        self.assertTrue(
            (
                pd.to_datetime(out_of_sample["Start"]) >= boundary
            ).all()
        )
        self.assertEqual(
            int(in_sample.iloc[0]["Sessions"]) + int(out_of_sample.iloc[0]["Sessions"]),
            len(covered),
        )

    def test_a_window_that_does_not_straddle_is_refused_rather_than_relabelled(
        self,
    ) -> None:
        sessions = self.result.stitched.daily.index
        boundary = self.result.plan.boundary_stamps[0]
        wholly_before = sessions[sessions < boundary]
        with self.assertRaises(ValueError) as caught:
            straddling_episode_report(
                self.ledgers,
                self.result.plan,
                episode="a test episode",
                start=wholly_before[-200],
                end=wholly_before[-1],
            )
        self.assertIn("does not straddle", str(caught.exception))

    def test_the_path_report_refuses_an_empty_window(self) -> None:
        with self.assertRaises(ValueError):
            path_report(
                self.result.stitched,
                window="empty",
                basis=SELECTOR_BASIS,
                start="2099-01-01",
            )


class CostAndCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.config = AllocationConfig()
        cls.plan = synthetic_plan(cls.inputs)

    def test_the_cost_sweep_reports_a_monotone_drag(self) -> None:
        report = cost_sensitivity_report(
            self.inputs, self.config, self.plan, multipliers=(0.0, 1.0, 2.0)
        )
        self.assertEqual(len(report), 3)
        drags = report["Annual cost drag"].to_numpy(dtype=float)
        self.assertTrue(np.all(np.diff(drags) >= -1e-15))
        self.assertEqual(float(drags[0]), 0.0)

    def test_the_sealed_column_is_the_block_statistic_not_a_fold_average(
        self,
    ) -> None:
        # The column published under the block's name was the arithmetic mean of
        # the annual fold Sharpes, which is a different quantity and is biased
        # upward whenever the annual means disperse.  It has to be the Sharpe of
        # the block's own return path, computed on the same window the sealed
        # comparison table uses.
        #
        # The block deliberately spans more than one fold.  With a single sealed
        # fold the two quantities coincide, so a one-fold block would let the
        # defect back in unnoticed.
        plan = synthetic_plan(self.inputs, last_year=2011)
        self.assertGreater(len(plan.boundaries), 2)
        report = cost_sensitivity_report(
            self.inputs, self.config, plan, multipliers=(1.0,)
        )
        row = report.iloc[0]
        result = run_walk_forward(self.inputs, self.config, plan)
        self.assertGreater(
            int((result.folds["oos_basis"] == SEALED_BASIS).sum()), 1
        )
        net = result.stitched.net_return
        window = net.loc[
            (net.index >= pd.Timestamp(plan.sealed_start))
            & (net.index <= pd.Timestamp(plan.end))
        ]
        expected = path_statistics(
            window,
            annualization=self.config.annualization,
            hac_lags=self.config.hac_lags,
            period_start=net.index[net.index < window.index[0]][-1],
        )["sharpe"]
        self.assertEqual(int(row["Sealed-block sessions"]), len(window))
        self.assertAlmostEqual(
            float(row["Sealed-block Sharpe (rf=0)"]), float(expected), places=12
        )

        # The fold average is still published, under a name that says what it
        # is, so a reader comparing against the fold table can see the gap.
        folds = result.folds
        sealed = folds.loc[folds["oos_basis"] == SEALED_BASIS, "fold_sharpe"]
        self.assertAlmostEqual(
            float(row["Mean sealed fold Sharpe (rf=0)"]),
            float(sealed.mean()),
            places=12,
        )

    def test_the_cost_frames_carry_the_whole_limitation_stack(self) -> None:
        # A frame carrying only the cost caveat reads as though the others do
        # not apply to it, and these two publish out-of-sample and sealed-block
        # performance figures.
        sweep = cost_sensitivity_report(
            self.inputs, self.config, self.plan, multipliers=(0.0, 1.0)
        )
        assumptions = self.config.costs.assumption_table()
        for name, frame in (("sweep", sweep), ("assumptions", assumptions)):
            with self.subTest(frame=name):
                self.assertIn("Limitation", frame.columns)
                stamped = frame["Limitation"].astype(str)
                self.assertTrue(stamped.str.contains("survivors_only_universe").all())
                for limitation in ALLOCATION_LIMITATIONS:
                    self.assertTrue(stamped.str.contains(limitation, regex=False).all())

    def test_an_invalid_cost_sweep_is_refused(self) -> None:
        for multipliers in ((), (-1.0,), (float("nan"),)):
            with self.subTest(multipliers=multipliers), self.assertRaises(ValueError):
                cost_sensitivity_report(
                    self.inputs, self.config, self.plan, multipliers=multipliers
                )

    def test_the_custody_replay_is_byte_identical_on_the_overlap(self) -> None:
        # Custody is a claim about what a run read.  This is the arithmetic
        # proof: a run that physically cannot load a sealed row reproduces the
        # full run's development prefix exactly.
        report = holdout_custody_report(
            self.inputs, self.config, self.plan, reload_from_disk=False
        )
        self.assertIn("truncated in memory", report.iloc[0]["Guard"])
        self.assertTrue(bool(report.iloc[0]["Byte identical on the overlap"]))
        self.assertEqual(float(report.iloc[0]["Maximum absolute difference"]), 0.0)
        self.assertGreater(int(report.iloc[0]["Compared sessions"]), 100)


class InputValidationTests(unittest.TestCase):
    def test_history_must_cover_the_traded_calendar(self) -> None:
        inputs = synthetic_inputs()
        with self.assertRaises(ValueError):
            AllocationInputs(
                panel=inputs.panel,
                history_close=inputs.history_close.iloc[100:],
                history_turnover=inputs.history_turnover,
                permitted_last_session=None,
            )

    def test_the_custody_guard_is_checked_against_the_history_frames(self) -> None:
        inputs = synthetic_inputs()
        with self.assertRaises(ValueError):
            AllocationInputs(
                panel=inputs.panel,
                history_close=inputs.history_close,
                history_turnover=inputs.history_turnover,
                permitted_last_session=inputs.panel.sessions[100],
            )

    def test_mismatched_columns_are_refused(self) -> None:
        inputs = synthetic_inputs()
        with self.assertRaises(ValueError):
            AllocationInputs(
                panel=inputs.panel,
                history_close=inputs.history_close.iloc[:, :3],
                history_turnover=inputs.history_turnover,
                permitted_last_session=None,
            )

    def test_the_capacity_series_is_lagged(self) -> None:
        inputs = synthetic_inputs()
        config = AllocationConfig()
        capacity = lagged_capacity(inputs, config)
        window = config.costs.turnover_window
        self.assertTrue(capacity.iloc[: window].isna().to_numpy().all())
        self.assertTrue(np.isfinite(capacity.iloc[-1].to_numpy()).all())


@unittest.skipUnless(
    REAL_PRICE_DIR.is_dir(), "Supplied DELTA1 ETF data directory is not available"
)
class SuppliedEtfAllocationTests(unittest.TestCase):
    """Against the licensed extract, with the real figures pinned.

    These assertions are deliberately tight on the structural facts --- session
    counts, fold arithmetic, custody --- and loose on the performance figures,
    which are properties of one realized path and would be false precision if
    pinned to four decimals.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_allocation_inputs(data_dir=DATA_DIR)
        cls.config = AllocationConfig()
        sessions = cls.inputs.panel.returns.index
        cls.plan = WalkForwardPlan(
            anchor=sessions[0].date().isoformat(),
            boundaries=annual_boundaries(sessions, first_year=2009, last_year=2018),
            end="2018-12-31",
            sealed_start="2014-01-01",
        )
        cls.result = run_walk_forward(cls.inputs, cls.config, cls.plan)

    def test_the_panel_is_the_eleven_fund_universe_from_the_fully_quoted_date(self) -> None:
        self.assertEqual(self.inputs.panel.tickers, UNIVERSE_TICKERS)
        self.assertEqual(self.inputs.panel.first_session, pd.Timestamp("2006-02-03"))
        self.assertEqual(self.inputs.panel.last_session, pd.Timestamp("2018-12-31"))
        self.assertEqual(len(self.inputs.panel.sessions), 3249)
        self.assertEqual(len(self.inputs.history_close), 6528)

    def test_membership_reproduces_the_declared_causal_entry_dates(self) -> None:
        members = membership_mask(self.inputs, self.config)
        entries = {
            ticker: members.index[members[ticker].to_numpy()][0].date().isoformat()
            for ticker in members.columns
        }
        self.assertEqual(entries["GLD"], "2006-02-21")
        self.assertEqual(entries["DBC"], "2008-04-30")
        for ticker in ("SPY", "IWM", "IYR", "EFA", "EEM", "SHY", "IEF", "TLT", "LQD"):
            self.assertEqual(entries[ticker], "2006-02-03")

    def test_the_fold_arithmetic_reaches_ten_years_and_five_sealed_years(self) -> None:
        accounting = out_of_sample_accounting(self.result)
        combined = accounting.loc[
            accounting["Claim"] == "Combined out-of-sample record"
        ].iloc[0]
        sealed = accounting.loc[
            accounting["Claim"] == "Sealed contiguous block"
        ].iloc[0]
        self.assertEqual(int(combined["Complete calendar years"]), 10)
        self.assertEqual(int(combined["Distinct sessions"]), 2516)
        self.assertEqual(int(combined["Sessions double counted"]), 0)
        self.assertEqual(int(sealed["Complete calendar years"]), 5)
        self.assertEqual(int(sealed["Distinct sessions"]), 1258)
        self.assertEqual(sealed["First session"], "2014-01-02")
        self.assertEqual(sealed["Last session"], "2018-12-31")
        self.assertEqual(len(self.result.folds), 10)
        self.assertEqual(int(self.result.folds["sessions"].sum()), 2516)

    def test_the_book_runs_below_its_declared_volatility_budget(self) -> None:
        # A scalar capped at one is a ceiling, and a ceiling cannot raise
        # volatility.  The unlevered diversified book sits materially below the
        # 7% budget, which is a fact about the design and not a defect.
        summary = self.result.summary
        realized = float(summary["stitched_annualized_volatility"])
        self.assertLess(realized, self.config.volatility_budget_annualized)
        self.assertGreater(realized, 0.03)

    def test_the_sealed_block_never_exercises_the_bear_state(self) -> None:
        coverage = sealed_block_state_coverage(self.inputs, self.plan)
        self.assertEqual(coverage.status, NOT_ESTIMABLE)
        self.assertIsNone(coverage.value)
        self.assertIn("not tested by this block", coverage.reason)

    def test_the_full_sample_does_exercise_it(self) -> None:
        plan = WalkForwardPlan(
            anchor=self.plan.anchor,
            boundaries=self.plan.boundaries,
            end=self.plan.end,
            sealed_start="2009-01-02",
        )
        coverage = sealed_block_state_coverage(self.inputs, plan)
        self.assertEqual(coverage.status, ESTIMATED)
        self.assertGreater(float(coverage.value), 0.0)

    def test_the_custody_replay_is_byte_identical(self) -> None:
        report = holdout_custody_report(
            self.inputs, self.config, self.plan, data_dir=DATA_DIR
        )
        self.assertTrue(bool(report.iloc[0]["Byte identical on the overlap"]))
        self.assertEqual(float(report.iloc[0]["Maximum absolute difference"]), 0.0)
        self.assertEqual(report.iloc[0]["Guarded input last session"], "2013-12-31")

    def test_every_artifact_carries_the_survivorship_limitation(self) -> None:
        references = reference_ledgers(self.inputs, self.config)
        ledgers = {"etf_regime_allocation": self.result.stitched, **references}
        frames = [
            self.result.folds.rename(columns={"panel_limitation": "Limitation"}),
            out_of_sample_accounting(self.result),
            comparison_report(ledgers, window="w", basis=SELECTOR_BASIS),
            in_sample_out_of_sample_report(self.result, self.config),
            exposure_share_report(self.result.stitched, self.inputs),
            volatility_budget_diagnostic(ledgers, self.config),
        ]
        for position, frame in enumerate(frames):
            with self.subTest(frame=position):
                self.assertIn("Limitation", frame.columns)
                self.assertTrue(
                    frame["Limitation"]
                    .astype(str)
                    .str.contains("survivors_only_universe")
                    .all()
                )

    def test_the_sealed_folds_are_labelled_under_their_own_basis(self) -> None:
        folds = self.result.folds
        sealed = folds.loc[folds["oos_basis"] == SEALED_BASIS]
        selector = folds.loc[folds["oos_basis"] == SELECTOR_BASIS]
        self.assertEqual(len(sealed), 5)
        self.assertEqual(len(selector), 5)
        self.assertEqual(sealed["calendar_year"].tolist(), [2014, 2015, 2016, 2017, 2018])
        for row in folds["panel_limitation"]:
            self.assertIn(ALLOCATION_LIMITATIONS[1], row)

    def test_the_selector_demonstrably_fits_inside_the_sealed_block(self) -> None:
        # This is the fact ``SEALED_PERMITTED_USE`` now discloses instead of
        # denying.  Four of the five sealed folds score a window that closes
        # inside the sealed block, and the chosen candidate changes as a result,
        # so the block was not sealed before any fitting decision.  That is
        # correct walk-forward behaviour --- every selection reads only sessions
        # published before its own boundary --- but it is not what the earlier
        # wording claimed.
        sealed_start = pd.Timestamp(self.plan.sealed_start)
        sealed = self.result.folds.loc[
            self.result.folds["oos_basis"] == SEALED_BASIS
        ]
        ends = pd.to_datetime(sealed["selection_window_end"])
        self.assertGreater(int((ends >= sealed_start).sum()), 0)
        boundaries = pd.to_datetime(sealed["boundary"])
        # Causality still holds: every scoring window closes before its own
        # boundary, by at least the declared stand-off.
        self.assertTrue(bool((ends < boundaries).all()))
        self.assertGreater(int(sealed["selection_switched"].sum()), 0)
        for use in sealed["permitted_use"]:
            self.assertIn("selector still fits inside", use)
            self.assertNotIn("sealed before any fitting decision", use)

    def test_every_artifact_the_script_writes_carries_the_survivorship_stamp(
        self,
    ) -> None:
        # A sweep over the frames the evidence script publishes.  Two of them
        # --- the cost sweep and the cost-model assumptions --- carried the cost
        # caveat alone while publishing out-of-sample and sealed-block
        # performance figures, and the excluded-candidate table carried no
        # limitation column at all.
        from delta1_strategy.marketdata import etfs as etf_module

        references = reference_ledgers(self.inputs, self.config)
        ledgers = {"etf_regime_allocation": self.result.stitched, **references}
        frames = {
            "universe": etf_module.universe_table(),
            "excluded": etf_module.excluded_candidate_table(),
            "candidates": candidate_table(),
            "cost_assumptions": self.config.costs.assumption_table(),
            "folds": self.result.folds.rename(
                columns={"panel_limitation": "Limitation"}
            ),
            "accounting": out_of_sample_accounting(self.result),
            "dispersion": fold_dispersion_report(self.result.folds),
            "degradation": in_sample_out_of_sample_report(self.result, self.config),
            "comparison": comparison_report(
                ledgers, window="w", basis=SELECTOR_BASIS
            ),
            "risk_matched": risk_matched_comparison(
                ledgers, reference_label="etf_regime_allocation", window="w"
            ),
            "budget": volatility_budget_diagnostic(ledgers, self.config),
            "exposure": exposure_share_report(self.result.stitched, self.inputs),
            "paired": paired_reference_inference(
                ledgers,
                subject_label="etf_regime_allocation",
                reference_labels=list(references),
            ),
            "cost_sweep": cost_sensitivity_report(
                self.inputs, self.config, self.plan, multipliers=(1.0,)
            ),
            "custody": holdout_custody_report(
                self.inputs, self.config, self.plan, data_dir=DATA_DIR
            ),
            "turnover_audit": etf_module.universe_turnover_audit(DATA_DIR),
        }
        for name, frame in frames.items():
            with self.subTest(frame=name):
                self.assertIn("Limitation", frame.columns)
                self.assertGreater(len(frame), 0)
                self.assertTrue(
                    frame["Limitation"]
                    .astype(str)
                    .str.contains("survivors_only_universe")
                    .all()
                )

    def test_the_declared_candidates_all_produce_a_live_book(self) -> None:
        for candidate in DECLARED_CANDIDATES:
            with self.subTest(candidate=candidate.name):
                desired = desired_weight_frame(self.inputs, candidate, self.config)
                self.assertIsNotNone(desired.first_live_session)
                self.assertLess(
                    desired.first_live_session, pd.Timestamp("2007-01-01")
                )
                self.assertEqual(
                    candidate.gate in (FABER_TREND, TIME_SERIES_MOMENTUM), True
                )

    def test_the_supplied_panel_path_is_truncation_invariant(self) -> None:
        truncated = load_allocation_inputs(
            start="2006-02-03",
            end="2015-06-30",
            last_session="2015-06-30",
            data_dir=DATA_DIR,
        )
        full = simulate_allocation(
            desired_weight_frame(self.inputs, DECLARED_CANDIDATES[0], self.config),
            self.inputs,
            self.config,
        )
        short = simulate_allocation(
            desired_weight_frame(truncated, DECLARED_CANDIDATES[0], self.config),
            truncated,
            self.config,
        )
        overlap = short.daily.index
        self.assertGreater(len(overlap), 2000)
        pd.testing.assert_frame_equal(
            full.daily.loc[overlap], short.daily, check_exact=True
        )


if __name__ == "__main__":
    unittest.main()
