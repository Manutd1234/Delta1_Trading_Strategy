from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd

from production import (
    ProductionLimits,
    ReadinessEvidence,
    build_order_intents,
    evaluate_runtime_health,
    overall_readiness_status,
    production_readiness_report,
)


class TestProductionLimitsAndEvidence(unittest.TestCase):
    def test_defaults_are_conservative_and_evidence_is_absent(self) -> None:
        limits = ProductionLimits()
        self.assertEqual(limits.max_order_participation, 0.02)
        self.assertEqual(limits.max_gross_notional_multiple, 6.0)
        self.assertEqual(limits.max_margin_fraction, 0.35)
        self.assertEqual(limits.max_drawdown_fraction, 0.15)
        self.assertEqual(limits.max_data_age_seconds, 300.0)
        evidence = ReadinessEvidence()
        self.assertTrue(all(getattr(evidence, item.name) is False for item in fields(evidence)))

    def test_invalid_limits_fail_fast(self) -> None:
        invalid = (
            {"max_order_participation": 0},
            {"max_order_participation": 1.01},
            {"max_gross_notional_multiple": np.inf},
            {"max_margin_fraction": -0.1},
            {"max_drawdown_fraction": np.nan},
            {"max_data_age_seconds": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ProductionLimits(**values)


class TestOrderIntents(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamp = pd.Timestamp("2025-01-02 16:00", tz="UTC")

    def build(self, **overrides: object) -> pd.DataFrame:
        inputs: dict[str, object] = {
            "current_contracts": {"ES": 0},
            "target_contracts": {"ES": 10},
            "raw_prices": {"ES": 100.0},
            "point_values": {"ES": 10.0},
            "margin_per_contract": {"ES": 500.0},
            "adv": {"ES": 100.0},
            "nav": 100_000.0,
            "decision_timestamp": self.timestamp,
            "limits": ProductionLimits(),
        }
        inputs.update(overrides)
        return build_order_intents(**inputs)

    def test_orders_are_deterministic_sorted_and_participation_capped(self) -> None:
        first = self.build(
            current_contracts={"NQ": 0, "ES": 0},
            target_contracts={"NQ": -1, "ES": 10},
            raw_prices={"NQ": 50, "ES": 100},
            point_values={"NQ": 20, "ES": 10},
            margin_per_contract={"NQ": 400, "ES": 500},
            adv={"NQ": 100, "ES": 100},
        )
        second = self.build(
            current_contracts={"ES": 0, "NQ": 0},
            target_contracts={"ES": 10, "NQ": -1},
            raw_prices={"ES": 100, "NQ": 50},
            point_values={"ES": 10, "NQ": 20},
            margin_per_contract={"ES": 500, "NQ": 400},
            adv={"ES": 100, "NQ": 100},
        )
        self.assertEqual(first["symbol"].tolist(), ["ES", "NQ"])
        self.assertEqual(first["order_id"].tolist(), second["order_id"].tolist())
        es = first.set_index("symbol").loc["ES"]
        self.assertEqual(es["requested_quantity"], 10)
        self.assertEqual(es["capped_quantity"], 2)
        self.assertEqual(es["quantity"], 2)
        self.assertEqual(es["side"], "BUY")
        self.assertAlmostEqual(es["projected_participation"], 0.02)
        self.assertEqual(es["projected_notional_usd"], 2_000)
        self.assertEqual(es["projected_margin_usd"], 1_000)
        self.assertTrue(es["approved"])
        self.assertEqual(es["status"], "APPROVED_PARTIAL")

    def test_portfolio_breach_blocks_risk_increase(self) -> None:
        row = self.build(
            current_contracts={"ES": 5},
            target_contracts={"ES": 6},
            raw_prices={"ES": 200_000},
            point_values={"ES": 1},
            adv={"ES": 1_000},
        ).iloc[0]
        self.assertFalse(row["approved"])
        self.assertEqual(row["status"], "BLOCKED")
        self.assertIn("gross notional", row["reason"])

    def test_reduce_only_order_is_allowed_during_a_limit_breach(self) -> None:
        row = self.build(
            current_contracts={"ES": 10},
            target_contracts={"ES": 0},
            raw_prices={"ES": 100_000},
            point_values={"ES": 1},
            margin_per_contract={"ES": 10_000},
            adv={"ES": 100},
        ).iloc[0]
        self.assertTrue(row["approved"])
        self.assertEqual(row["side"], "SELL")
        self.assertEqual(row["status"], "APPROVED_REDUCE_ONLY")
        self.assertEqual(row["style"], "REDUCE_ONLY")

    def test_missing_or_invalid_inputs_never_approve_an_order(self) -> None:
        cases = (
            {"raw_prices": {}},
            {"adv": {"ES": 0}},
            {"nav": np.nan},
            {"decision_timestamp": None},
            {"target_contracts": {"ES": 1.5}},
        )
        for values in cases:
            with self.subTest(values=values):
                row = self.build(**values).iloc[0]
                self.assertFalse(row["approved"])
                self.assertEqual(row["status"], "BLOCKED")

    def test_no_change_is_not_an_approved_order(self) -> None:
        row = self.build(current_contracts={"ES": 3}, target_contracts={"ES": 3}).iloc[0]
        self.assertEqual(row["side"], "HOLD")
        self.assertFalse(row["approved"])
        self.assertEqual(row["status"], "NO_ACTION")

    def test_negative_futures_price_is_valued_by_absolute_exposure(self) -> None:
        row = self.build(
            target_contracts={"CL": 1},
            current_contracts={"CL": 0},
            raw_prices={"CL": -10},
            point_values={"CL": 1_000},
            margin_per_contract={"CL": 2_000},
            adv={"CL": 1_000},
        ).iloc[0]
        self.assertTrue(row["approved"])
        self.assertEqual(row["projected_notional_usd"], 10_000)


class TestRuntimeHealth(unittest.TestCase):
    def test_complete_fresh_snapshot_passes(self) -> None:
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        health = evaluate_runtime_health(
            data_timestamp=now - pd.Timedelta(seconds=30),
            current_time=now,
            nav=100,
            peak_nav=105,
            gross_notional=500,
            margin_requirement=20,
            broker_connected=True,
            broker_reconciled=True,
            monitoring_healthy=True,
            kill_switch_ready=True,
        )
        self.assertTrue(health.healthy)
        self.assertEqual(health.status, "PASS")
        self.assertEqual(health.reasons, ())
        self.assertEqual(health.data_age_seconds, 30)

    def test_defaults_stale_future_and_breaches_fail_closed(self) -> None:
        self.assertFalse(evaluate_runtime_health().healthy)
        now = pd.Timestamp("2025-01-02 16:00", tz="UTC")
        common = {
            "current_time": now,
            "nav": 100,
            "peak_nav": 100,
            "gross_notional": 100,
            "margin_requirement": 10,
            "broker_connected": True,
            "broker_reconciled": True,
            "monitoring_healthy": True,
            "kill_switch_ready": True,
        }
        for timestamp in (now - pd.Timedelta(seconds=301), now + pd.Timedelta(seconds=1)):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(evaluate_runtime_health(data_timestamp=timestamp, **common).healthy)
        breached = evaluate_runtime_health(
            data_timestamp=now,
            current_time=now,
            nav=100,
            peak_nav=125,
            gross_notional=601,
            margin_requirement=36,
            broker_connected=True,
            broker_reconciled=True,
            monitoring_healthy=True,
            kill_switch_ready=True,
        )
        self.assertFalse(breached.healthy)
        self.assertGreaterEqual(len(breached.reasons), 3)


def safe_result() -> SimpleNamespace:
    daily = pd.DataFrame(
        {
            "nav": [100.0, 102.0, 99.0, 103.0],
            "equity": [1.0, 1.02, 0.99, 1.03],
            "net_return": [0.0, 0.02, -0.0294118, 0.040404],
            "cost": [0.0, 0.001, 0.001, 0.001],
            "gross_notional_multiple": [0.0, 2.0, 3.0, 1.0],
            "static_margin_fraction": [0.0, 0.10, 0.20, 0.05],
            "max_order_participation": [0.0, 0.01, 0.02, 0.01],
            "pending_markets": [0, 0, 1, 0],
        },
        index=pd.date_range("2025-01-01", periods=4),
    )
    return SimpleNamespace(daily=daily)


class TestProductionReadiness(unittest.TestCase):
    def test_all_critical_gates_must_pass(self) -> None:
        all_evidence = ReadinessEvidence(
            **{item.name: True for item in fields(ReadinessEvidence)}
        )
        report = production_readiness_report(safe_result(), evidence=all_evidence)
        self.assertTrue(report["critical"].all())
        self.assertTrue(report["status"].eq("PASS").all())
        self.assertEqual(overall_readiness_status(report), "READY")

    def test_external_evidence_defaults_to_blocked(self) -> None:
        report = production_readiness_report(safe_result())
        external = report.loc[report["category"].eq("external evidence")]
        self.assertEqual(len(external), len(fields(ReadinessEvidence)))
        self.assertTrue(external["status"].eq("BLOCKED").all())
        self.assertEqual(overall_readiness_status(report), "BLOCKED")

    def test_missing_columns_nonfinite_values_and_limit_breaches_block(self) -> None:
        missing = production_readiness_report(SimpleNamespace(daily=pd.DataFrame({"nav": [100]})))
        self.assertEqual(overall_readiness_status(missing), "BLOCKED")

        result = safe_result()
        result.daily.loc[result.daily.index[-1], "gross_notional_multiple"] = 6.01
        result.daily.loc[result.daily.index[-1], "static_margin_fraction"] = np.nan
        result.daily.loc[result.daily.index[-1], "max_order_participation"] = 0.021
        report = production_readiness_report(result)
        blocked = set(report.loc[report["status"].eq("BLOCKED"), "gate"])
        self.assertIn("historical_gross_notional", blocked)
        self.assertIn("historical_margin", blocked)
        self.assertIn("historical_order_participation", blocked)

    def test_negative_risk_metrics_and_initial_equity_loss_block(self) -> None:
        result = safe_result()
        result.daily.loc[:, "gross_notional_multiple"] = -0.1
        result.daily.loc[:, "equity"] = [0.80, 0.82, 0.83, 0.84]
        report = production_readiness_report(result)
        blocked = set(report.loc[report["status"].eq("BLOCKED"), "gate"])
        self.assertIn("historical_gross_notional", blocked)
        self.assertIn("historical_drawdown", blocked)

        drawdown = report.set_index("gate").loc["historical_drawdown"]
        self.assertAlmostEqual(drawdown["observed"], 0.20)

    def test_overall_status_rejects_malformed_or_vacuous_reports(self) -> None:
        self.assertEqual(overall_readiness_status(None), "BLOCKED")
        self.assertEqual(overall_readiness_status(pd.DataFrame()), "BLOCKED")
        self.assertEqual(
            overall_readiness_status(pd.DataFrame({"critical": [False], "status": ["PASS"]})),
            "BLOCKED",
        )


if __name__ == "__main__":
    unittest.main()
