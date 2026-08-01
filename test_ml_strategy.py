from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_engineer_features import _numeric_notional, aggregate_fx_snapshot, snapshot_audit
from delta1_cta import ASSET_CLASSES, SYMBOL_TO_CLASS, BacktestConfig, load_metadata, load_prices
from ml_strategy import (
    FEATURE_COLUMNS,
    MACRO_FEATURES,
    MLConfig,
    build_feature_panel,
    load_external_macro,
    predictions_to_signal_matrix,
    walk_forward_predict,
)


DATA_DIR = Path(os.environ.get("DELTA1_DATA_DIR", "data/Delta1"))
EXTERNAL_MACRO = Path(os.environ.get("DELTA1_EXTERNAL_MACRO", "data/external/fred_macro.csv"))


class TestPointInTimeFeatures(unittest.TestCase):
    def test_macro_is_lagged_one_business_day(self) -> None:
        calendar = pd.bdate_range("2001-01-01", periods=5)
        source = pd.DataFrame(
            {
                "Date": calendar,
                "VIXCLS": [10, 20, 30, 40, 50],
                "T10Y2Y": [1, 2, 3, 4, 5],
                "BAA10Y": [2, 3, 4, 5, 6],
            }
        )
        path = Path("/private/tmp/delta1_macro_lag_test.csv")
        source.to_csv(path, index=False)
        features = load_external_macro(path, calendar)
        self.assertTrue(np.isnan(features.iloc[0]["yield_curve"]))
        self.assertEqual(features.iloc[1]["yield_curve"], 1)
        self.assertEqual(features.iloc[1]["baa_credit_spread"], 2)

    def test_future_price_mutation_cannot_change_past_features(self) -> None:
        rng = np.random.default_rng(11)
        index = pd.bdate_range("1998-01-01", periods=800)
        symbols = [symbol for group in ASSET_CLASSES.values() for symbol in group]
        prices = pd.DataFrame(
            100 + np.cumsum(rng.normal(size=(len(index), len(symbols))), axis=0),
            index=index,
            columns=symbols,
        )
        metadata = pd.DataFrame(
            {"asset_class": [SYMBOL_TO_CLASS[symbol] for symbol in symbols]},
            index=symbols,
        )
        macro = pd.DataFrame(0.0, index=index, columns=MACRO_FEATURES)
        original = build_feature_panel(prices, metadata, macro)
        cutoff = index[650]
        altered_prices = prices.copy()
        altered_prices.loc[cutoff:] *= 10
        altered = build_feature_panel(altered_prices, metadata, macro)
        columns = ["feature_date", "symbol", *FEATURE_COLUMNS]
        left = original.loc[original["feature_date"].lt(cutoff), columns].reset_index(drop=True)
        right = altered.loc[altered["feature_date"].lt(cutoff), columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


class TestWalkForwardControls(unittest.TestCase):
    def test_training_labels_precede_each_test_year(self) -> None:
        rng = np.random.default_rng(3)
        dates = pd.date_range("1997-01-31", "2005-12-31", freq="ME")
        rows = []
        for symbol, asset_class in (("ES", "Equity indices"), ("ZN", "Government bonds")):
            for date in dates:
                row = {column: float(rng.normal()) for column in FEATURE_COLUMNS if column != "asset_class"}
                row.update(
                    {
                        "asset_class": asset_class,
                        "symbol": symbol,
                        "feature_date": date,
                        "label_end": date + pd.offsets.MonthEnd(1),
                        "future_risk_return": float(rng.normal()),
                        "target": float(rng.integers(0, 2)),
                    }
                )
                rows.append(row)
        panel = pd.DataFrame(rows)
        result = walk_forward_predict(
            panel,
            MLConfig(prediction_start="2003-01-01", prediction_end="2005-12-31", validation_months=24),
        )
        for row in result.model_selection.itertuples():
            self.assertLess(pd.Timestamp(row.last_available_label), pd.Timestamp(row.year, 1, 1))

    def test_hybrid_and_ml_signals_are_bounded(self) -> None:
        dates = pd.bdate_range("2005-01-03", periods=30)
        month_end = dates[-1]
        predictions = pd.DataFrame(
            {
                "feature_date": [month_end, month_end],
                "symbol": ["ES", "ZN"],
                "probability_up": [0.99, 0.01],
                "trend_252": [-1.0, 1.0],
            }
        )
        prices = pd.DataFrame(100.0, index=dates, columns=["ES", "ZN"])
        hybrid, ml = predictions_to_signal_matrix(
            predictions, prices, probability_scale=0.15, ml_blend_weight=0.5
        )
        self.assertLessEqual(float(hybrid.abs().max().max()), 1.0)
        self.assertLessEqual(float(ml.abs().max().max()), 1.0)

    def test_invalid_time_windows_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            MLConfig(training_start="2005-01-01", prediction_start="2003-01-01").validate()


class TestDataEngineerSnapshot(unittest.TestCase):
    def test_capped_notional_parser(self) -> None:
        values, capped = _numeric_notional(pd.Series(["1,250", "250+", None]))
        self.assertEqual(values.iloc[0], 1250)
        self.assertEqual(values.iloc[1], 250)
        self.assertTrue(bool(capped.iloc[1]))

    def test_snapshot_is_explicitly_excluded_from_historical_training(self) -> None:
        frame = pd.DataFrame(
            {
                "Dissemination Identifier": [1],
                "event_time": pd.to_datetime(["2024-04-08T00:00:00Z"]),
                "is_new_trade": [True],
                "usd_notional": [1_000_000.0],
                "notional_is_capped": [False],
            }
        )
        audit = snapshot_audit(frame)
        self.assertFalse(audit.usable_in_2005_2014_backtest)
        self.assertIn("look-ahead", audit.exclusion_reason)


@unittest.skipUnless(DATA_DIR.exists() and EXTERNAL_MACRO.exists(), "Supplied datasets are unavailable")
class TestMLIntegration(unittest.TestCase):
    def test_real_panel_has_expected_point_in_time_coverage(self) -> None:
        prices = load_prices(DATA_DIR)
        metadata = load_metadata(DATA_DIR)
        macro = load_external_macro(EXTERNAL_MACRO, prices.index)
        panel = build_feature_panel(prices, metadata, macro)
        self.assertEqual(set(panel["symbol"]), set(prices.columns))
        evaluation = panel.loc[panel["feature_date"].between("2005-01-01", "2014-12-31")]
        self.assertGreater(len(evaluation), 2500)
        labelled = evaluation.loc[evaluation["label_end"].notna()]
        self.assertTrue(labelled["label_end"].gt(labelled["feature_date"]).all())


if __name__ == "__main__":
    unittest.main()
