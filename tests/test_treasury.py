from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

import pandas as pd

from delta1_strategy.controls.treasury import (
    AccountSnapshot,
    CashBalance,
    CollateralLot,
    JournalEntry,
    LedgerPosting,
    MarginQuote,
    SettlementMark,
    TreasuryEvidenceBundle,
    TreasuryFreshnessLimits,
    TreasuryValidationError,
    ZERO_SHA256,
    record_sha256,
    validate_journal_chain,
    validate_treasury_evidence,
)


NOW = pd.Timestamp("2026-08-04 08:00:00", tz="UTC")
POSITION_HASH = "1" * 64
ORDERS_HASH = "2" * 64
SOURCE_HASH = "a" * 64


def posting(side: str, account: str, amount: float = 25.0) -> LedgerPosting:
    return LedgerPosting(
        ledger_account=account,
        currency="USD",
        base_currency="USD",
        side=side,
        amount_native=amount,
        fx_to_base=1.0,
        contract_id="ESU6",
    )


def journal_entry(
    *, sequence: int = 1, previous: str = ZERO_SHA256, entry_id: str = "entry-1"
) -> JournalEntry:
    return JournalEntry(
        sequence=sequence,
        entry_id=entry_id,
        idempotency_key=f"idempotency-{entry_id}",
        account_id="account-1",
        base_currency="USD",
        event_type="VARIATION_MARGIN",
        effective_at=NOW - pd.Timedelta(minutes=1),
        recorded_at=NOW - pd.Timedelta(seconds=20),
        value_date=NOW.date(),
        postings=(
            posting("DEBIT", "settled_cash"),
            posting("CREDIT", "variation_margin_pnl"),
        ),
        previous_entry_sha256=previous,
        source_id="broker-event-1",
        source_sha256=SOURCE_HASH,
    )


def valid_bundle() -> TreasuryEvidenceBundle:
    entry = journal_entry()
    quote = MarginQuote(
        account_id="account-1",
        base_currency="USD",
        generated_at=NOW - pd.Timedelta(seconds=30),
        effective_from=NOW - pd.Timedelta(days=1),
        expires_at=NOW + pd.Timedelta(days=1),
        position_snapshot_sha256=POSITION_HASH,
        open_orders_snapshot_sha256=ORDERS_HASH,
        initial_margin_base=300.0,
        maintenance_margin_base=250.0,
        house_addon_base=20.0,
        concentration_addon_base=10.0,
        open_order_reserve_base=5.0,
        margin_model="BROKER_PORTFOLIO",
        model_version="effective-2026-08-04",
        source_id="margin-quote-1",
        source_sha256=SOURCE_HASH,
    )
    cash = CashBalance(
        currency="USD",
        base_currency="USD",
        settled_amount=1_000.0,
        fx_to_base=1.0,
        fx_timestamp=NOW - pd.Timedelta(minutes=1),
        margin_eligible=True,
        margin_haircut_fraction=0.05,
        source_id="cash-line-1",
        source_sha256=SOURCE_HASH,
    )
    collateral = CollateralLot(
        collateral_id="collateral-1",
        account_id="account-1",
        asset_id="UST-1",
        asset_type="TREASURY",
        currency="USD",
        base_currency="USD",
        market_value_native=100.0,
        fx_to_base=1.0,
        haircut_fraction=0.10,
        eligible=True,
        valuation_timestamp=NOW - pd.Timedelta(minutes=1),
        effective_from=NOW - pd.Timedelta(days=1),
        effective_to=NOW + pd.Timedelta(days=1),
        concentration_group="US_GOVERNMENT",
        concentration_limit_base=80.0,
        custodian="test-custodian",
        source_id="collateral-file-1",
        source_sha256=SOURCE_HASH,
    )
    account = AccountSnapshot(
        account_id="account-1",
        base_currency="USD",
        as_of=NOW - pd.Timedelta(seconds=10),
        cash_balances=(cash,),
        collateral_lots=(collateral,),
        unsettled_vm_receivable_base=10.0,
        unsettled_vm_payable_base=5.0,
        pending_fees_base=2.0,
        other_assets_liabilities_base=-3.0,
        account_equity_base=1_100.0,
        initial_margin_requirement_base=quote.total_initial_margin_base,
        maintenance_margin_requirement_base=quote.total_maintenance_margin_base,
        broker_available_funds_base=688.0,
        broker_excess_liquidity_base=688.0,
        margin_call_base=0.0,
        position_snapshot_sha256=POSITION_HASH,
        open_orders_snapshot_sha256=ORDERS_HASH,
        margin_quote_sha256=record_sha256(quote),
        journal_head_sha256=record_sha256(entry),
        source_id="account-snapshot-1",
        source_sha256=SOURCE_HASH,
    )
    mark = SettlementMark(
        contract_id="ESU6",
        session_date=NOW.date(),
        settlement_timestamp=NOW - pd.Timedelta(hours=1),
        prior_settlement=6300.0,
        current_settlement=6310.0,
        settlement_currency="USD",
        base_currency="USD",
        point_value=50.0,
        fx_to_base=1.0,
        fx_timestamp=NOW - pd.Timedelta(hours=1),
        source_id="settlement-file-1",
        source_sha256=SOURCE_HASH,
    )
    return TreasuryEvidenceBundle(
        account_snapshot=account,
        margin_quote=quote,
        settlement_marks=(mark,),
        journal_entries=(entry,),
    )


def limits() -> TreasuryFreshnessLimits:
    return TreasuryFreshnessLimits(
        max_account_age_seconds=60.0,
        max_margin_quote_age_seconds=60.0,
        max_settlement_age_seconds=7_200.0,
        max_fx_age_seconds=7_200.0,
        max_collateral_age_seconds=3_600.0,
        monetary_tolerance_base=0.01,
    )


class TreasuryRecordTests(unittest.TestCase):
    def test_records_are_frozen_and_canonicalize_timestamps(self) -> None:
        mark = valid_bundle().settlement_marks[0]
        self.assertEqual(str(mark.settlement_timestamp.tz), "UTC")
        with self.assertRaises(FrozenInstanceError):
            mark.point_value = 10.0  # type: ignore[misc]

    def test_rejects_naive_timestamp_and_invalid_hash(self) -> None:
        bundle = valid_bundle()
        mark = bundle.settlement_marks[0]
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            replace(mark, settlement_timestamp=pd.Timestamp("2026-08-04"))
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            replace(mark, source_sha256="NOT-A-HASH")

    def test_collateral_haircut_and_concentration_cap_are_explicit(self) -> None:
        account = valid_bundle().account_snapshot
        self.assertAlmostEqual(account.collateral_market_value_base, 100.0)
        self.assertAlmostEqual(account.eligible_collateral_base, 80.0)
        self.assertAlmostEqual(account.eligible_margin_resources_base, 1_030.0)

    def test_unbalanced_journal_entry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not balanced"):
            replace(
                journal_entry(),
                postings=(
                    posting("DEBIT", "settled_cash", 25.0),
                    posting("CREDIT", "variation_margin_pnl", 24.0),
                ),
            )


class TreasuryValidationTests(unittest.TestCase):
    def validate(self, bundle: TreasuryEvidenceBundle, *, as_of=NOW) -> None:
        validate_treasury_evidence(
            bundle,
            as_of=as_of,
            limits=limits(),
            expected_position_snapshot_sha256=POSITION_HASH,
            expected_open_orders_snapshot_sha256=ORDERS_HASH,
            expected_contract_ids=("ESU6",),
        )

    def test_complete_content_addressed_bundle_passes(self) -> None:
        bundle = valid_bundle()
        self.assertIs(
            validate_treasury_evidence(
                bundle,
                as_of=NOW,
                limits=limits(),
                expected_position_snapshot_sha256=POSITION_HASH,
                expected_open_orders_snapshot_sha256=ORDERS_HASH,
                expected_contract_ids=("ESU6",),
            ),
            bundle,
        )

    def test_stale_snapshot_fails_closed(self) -> None:
        bundle = valid_bundle()
        stale_cash = replace(
            bundle.account_snapshot.cash_balances[0],
            fx_timestamp=NOW - pd.Timedelta(minutes=3),
        )
        stale_collateral = replace(
            bundle.account_snapshot.collateral_lots[0],
            valuation_timestamp=NOW - pd.Timedelta(minutes=3),
        )
        stale = replace(
            bundle,
            account_snapshot=replace(
                bundle.account_snapshot,
                as_of=NOW - pd.Timedelta(minutes=2),
                cash_balances=(stale_cash,),
                collateral_lots=(stale_collateral,),
            ),
        )
        with self.assertRaisesRegex(TreasuryValidationError, "account snapshot is stale"):
            self.validate(stale)

    def test_snapshot_hash_mismatch_fails_closed(self) -> None:
        bundle = valid_bundle()
        mismatched = replace(
            bundle,
            account_snapshot=replace(
                bundle.account_snapshot,
                position_snapshot_sha256="f" * 64,
            ),
        )
        with self.assertRaisesRegex(TreasuryValidationError, "hash does not match"):
            self.validate(mismatched)

    def test_equity_roll_forward_must_reconcile(self) -> None:
        bundle = valid_bundle()
        mismatched = replace(
            bundle,
            account_snapshot=replace(bundle.account_snapshot, account_equity_base=1_101.0),
        )
        with self.assertRaisesRegex(TreasuryValidationError, "equity roll-forward"):
            self.validate(mismatched)

    def test_expired_margin_quote_fails_closed(self) -> None:
        bundle = valid_bundle()
        with self.assertRaisesRegex(TreasuryValidationError, "margin quote has expired"):
            self.validate(bundle, as_of=NOW + pd.Timedelta(days=2))

    def test_missing_held_contract_settlement_fails_closed(self) -> None:
        bundle = replace(valid_bundle(), settlement_marks=())
        with self.assertRaisesRegex(TreasuryValidationError, "missing contracts: ESU6"):
            self.validate(bundle)

    def test_journal_chain_detects_wrong_predecessor(self) -> None:
        first = journal_entry()
        second = journal_entry(
            sequence=2,
            previous="f" * 64,
            entry_id="entry-2",
        )
        with self.assertRaisesRegex(TreasuryValidationError, "hash does not reconcile"):
            validate_journal_chain((first, second))


if __name__ == "__main__":
    unittest.main()
