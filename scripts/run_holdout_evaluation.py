#!/usr/bin/env python3
"""Score the frozen specification once on data the pipeline has never read.

The canonical run's source manifest ends on 2014-12-31 and contains none of the
files used here, so the 2015-2016 continuation is genuinely unseen. That makes
this the repository's first out-of-sample evidence — and also its narrowest.
Twelve of fifty-nine roots survive the data constraints, the basis sleeve
cannot be reconstructed without unadjusted prices, and five hundred sessions
cannot resolve a Sharpe difference of the size the promotion gates require.

So this is a subset consistency diagnostic, not a holdout for the flagship. It
does not make the incumbent promotable and the artifacts say so. What it
establishes is that the protocol runs: seal the bytes, write the criteria down,
look once, and record the look where a second one will be refused.

Order matters and is enforced. The dataset is hashed and the criteria are
registered before any price is read.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from delta1_strategy.research import holdout as holdout_module
from delta1_strategy.research.holdout import (
    SUBSET_LIMITATIONS,
    HoldoutConsumption,
    HoldoutLedger,
    HoldoutRegistration,
    evaluate_acceptance,
    load_extension_frames,
    seal_dataset,
    verify_continuity,
)
from delta1_strategy.research.strategy import (
    StrategyConfig,
    load_delivery_months,
    load_fx_rates,
    load_metadata,
    load_prices,
    load_volumes,
    run_backtest,
)

# All eleven tradeable government bonds plus gold: the roots for which the
# vendor supplies a post-2014 continuation.  Chosen by data availability alone,
# before any 2015-2016 return was computed.
HOLDOUT_ROOTS = (
    "CGB", "FGBL", "FGBM", "FGBS", "FGBX", "GC",
    "LLG", "SJB", "ZB", "ZF", "ZN", "ZT",
)

WINDOW_START = "2015-01-01"
WINDOW_END = "2016-12-30"

# Written before the dataset was read.  Deliberately weak: with five hundred
# sessions the standard error on a Sharpe estimate is far too wide to justify
# a demanding threshold, and pretending otherwise would be the whole failure
# this protocol exists to prevent.
ACCEPTANCE_CRITERIA = {
    "annualized_return": 0.0,
    "naive_sharpe": 0.0,
    "max_drawdown": -0.15,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--forex-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/holdout"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/run_manifest.json"))
    parser.add_argument(
        "--ledger", type=Path, default=Path("outputs/holdout/holdout_ledger.jsonl")
    )
    parser.add_argument(
        "--as-of",
        type=str,
        required=True,
        help="UTC timestamp for the ledger events, e.g. 2026-08-06T00:00:00Z",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    # ---- 1. Seal the bytes, before reading a single value from them. --------
    dataset = seal_dataset(
        args.extension_dir,
        HOLDOUT_ROOTS,
        dataset_id="fxfi-2015-2016-bond-and-gold-subset",
        description=(
            "Vendor continuation series for the twelve traded roots that extend "
            "past 2014-12-31; absent from the canonical source manifest"
        ),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )

    protocol_text = json.dumps(
        {
            "roots": list(HOLDOUT_ROOTS),
            "window": [WINDOW_START, WINDOW_END],
            "acceptance_criteria": ACCEPTANCE_CRITERIA,
            "limitations": list(SUBSET_LIMITATIONS),
            "sleeve": "trend only; basis-momentum unavailable without unadjusted prices",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    registration = HoldoutRegistration(
        dataset=dataset,
        model_fingerprint_sha256=manifest["implementation_fingerprint_sha256"],
        config_fingerprint_sha256=manifest["config_sha256"],
        protocol_sha256=hashlib.sha256(protocol_text.encode("utf-8")).hexdigest(),
        acceptance_criteria=ACCEPTANCE_CRITERIA,
        limitations=SUBSET_LIMITATIONS,
        custodian="local repository; not an independent custodian",
    )

    ledger = HoldoutLedger.load(args.ledger)
    already = {
        event.consumption.dataset_id
        for event in ledger.events
        if event.consumption is not None
    }
    if dataset.dataset_id in already:
        raise SystemExit(
            f"{dataset.dataset_id} has already been consumed; a second look "
            "would make it reused history. Refusing to run."
        )
    if not any(
        event.registration is not None
        and event.registration.dataset.dataset_id == dataset.dataset_id
        for event in ledger.events
    ):
        ledger = ledger.register(registration, occurred_at=args.as_of)
        ledger.save(args.ledger)
        print(f"sealed {dataset.dataset_id} ({len(dataset.file_sha256)} files)")

    # ---- 2. Build the joined history and prove the join is sound. -----------
    symbols = list(HOLDOUT_ROOTS)
    prices = load_prices(args.data_dir, symbols=symbols)
    volumes = load_volumes(args.data_dir, symbols=symbols)
    deliveries = load_delivery_months(args.data_dir, symbols=symbols)
    extension = load_extension_frames(args.extension_dir, symbols)

    continuity = verify_continuity(prices, extension["prices"])
    continuity.to_csv(output_dir / "holdout_continuity.csv", index=False)
    if not (continuity["status"] == "PASS").all():
        raise SystemExit(
            "extension does not continue the canonical series; appending would "
            "restate history. See holdout_continuity.csv"
        )
    print(f"continuity verified across {len(continuity)} roots, zero difference")

    calendar = pd.bdate_range(prices.index.min(), pd.Timestamp(WINDOW_END))

    def joined(base: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
        tail = tail.loc[tail.index > base.index.max()]
        return pd.concat([base, tail]).reindex(calendar).sort_index()

    all_prices = joined(prices, extension["prices"]).ffill(limit=10)
    all_volumes = joined(volumes, extension["volumes"])
    all_deliveries = joined(deliveries, extension["delivery_months"])
    metadata = load_metadata(args.data_dir, symbols=symbols)
    # The canonical currency futures also stop in 2014, so the foreign-
    # denominated bonds cannot be valued past the join without extending them.
    canonical_fx = load_fx_rates(args.data_dir, calendar)
    fx_rates, fx_splice = holdout_module.load_extension_fx_rates(
        args.forex_dir, canonical_fx, calendar
    )
    fx_splice.to_csv(output_dir / "holdout_fx_splice.csv", index=False)
    print(
        "fx extended from spot; futures basis at the join "
        f"{fx_splice['basis_fraction'].abs().max():.4%} maximum"
    )

    # ---- 3. Replay the frozen specification, scoring only the new window. ---
    # Trend only: the vendor ships no unadjusted post-2014 series, so the roll
    # gap the basis sleeve is built from cannot be reconstructed.  Passing the
    # back-adjusted frame as if it were unadjusted would silently zero the roll
    # gap and halve the forecast, so the sleeve is disabled explicitly instead.
    config = dataclasses.replace(
        StrategyConfig(data_dir=args.data_dir, output_dir=output_dir),
        basis_weight=0.0,
    )
    result = run_backtest(
        config,
        prices=all_prices,
        observed_opens=all_prices,
        observed_closes=all_prices,
        unadjusted=all_prices,
        metadata=metadata,
        volumes=all_volumes,
        delivery_months=all_deliveries,
        fx_rates=fx_rates,
    )

    window = result.daily.loc[WINDOW_START:WINDOW_END]
    returns = window["net_return"].dropna()
    years = max((returns.index[-1] - returns.index[0]).days / 365.2425, 1e-9)
    equity = (1.0 + returns).cumprod()
    observed = {
        "annualized_return": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "naive_sharpe": float(
            returns.mean() / returns.std(ddof=0) * np.sqrt(config.annualization)
        ),
        "max_drawdown": float((equity / equity.cummax().clip(lower=1.0) - 1.0).min()),
        "annualized_volatility": float(
            returns.std(ddof=0) * np.sqrt(config.annualization)
        ),
        "sessions": float(len(returns)),
    }

    acceptance = evaluate_acceptance(observed, ACCEPTANCE_CRITERIA)
    summary = pd.DataFrame(
        [
            {
                "dataset_id": dataset.dataset_id,
                "window_start": WINDOW_START,
                "window_end": WINDOW_END,
                "roots": len(HOLDOUT_ROOTS),
                "sleeve": "trend only",
                "validation_status": "out_of_sample_subset_consistency_diagnostic",
                "selection_adjusted": False,
                "promotes_the_incumbent": False,
                **{key: value for key, value in observed.items()},
                "limitations": "; ".join(SUBSET_LIMITATIONS),
            }
        ]
    )
    summary.to_csv(output_dir / "holdout_summary.csv", index=False)
    acceptance.to_csv(output_dir / "holdout_acceptance.csv", index=False)
    window.to_csv(output_dir / "holdout_daily.csv", index_label="date")

    # ---- 4. Record the look, so a second one is refused. --------------------
    artifact_digest = hashlib.sha256(
        (output_dir / "holdout_daily.csv").read_bytes()
    ).hexdigest()
    ledger = ledger.consume(
        HoldoutConsumption(
            dataset_id=dataset.dataset_id,
            registration_sha256=registration.registration_sha256,
            result_artifact_sha256=artifact_digest,
            observed=observed,
            completed_at=args.as_of,
        ),
        occurred_at=args.as_of,
    )
    ledger.save(args.ledger)

    print(f"\nout-of-sample window {WINDOW_START}..{WINDOW_END}, {len(returns)} sessions")
    for name in ("annualized_return", "naive_sharpe", "annualized_volatility", "max_drawdown"):
        print(f"  {name:24} {observed[name]:+.4f}")
    print()
    print(acceptance.to_string(index=False))
    print(f"\nledger head {ledger.head_sha256[:16]}... — this dataset is now spent")


if __name__ == "__main__":
    main()
