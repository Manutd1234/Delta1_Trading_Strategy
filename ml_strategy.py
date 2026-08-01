"""Leakage-safe walk-forward machine-learning extension for DELTA1 futures.

The model is intentionally modest for the sample size: each year it selects a
regularized logistic regression or shallow gradient-boosted tree using only a
trailing validation window, refits on all labels available before that year,
and predicts the next month's direction for each contract. The ML forecast is
blended with the literature-specified 12-month trend prior before entering the
same risk, execution, and accounting pipeline as the baseline.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from delta1_cta import (
    ASSET_CLASSES,
    BacktestConfig,
    BacktestResult,
    load_metadata,
    load_prices,
    performance_metrics,
    run_backtest,
)


INSTRUMENT_FEATURES = (
    "trend_21",
    "trend_63",
    "trend_126",
    "trend_252",
    "vol_ratio_20_120",
    "breakout_252",
    "skew_63",
    "down_day_share_63",
    "cross_section_rank_252",
    "class_trend_252",
)
MACRO_FEATURES = (
    "vix_log",
    "vix_change_21",
    "yield_curve",
    "yield_curve_change_21",
    "baa_credit_spread",
    "baa_credit_spread_change_21",
)
FEATURE_COLUMNS = (*INSTRUMENT_FEATURES, *MACRO_FEATURES, "asset_class")
NUMERIC_FEATURES = (*INSTRUMENT_FEATURES, *MACRO_FEATURES)


@dataclass(frozen=True)
class MLConfig:
    training_start: str = "1997-01-01"
    prediction_start: str = "2003-01-01"
    prediction_end: str = "2014-12-31"
    validation_months: int = 24
    probability_scale: float = 0.15
    ml_blend_weight: float = 0.50
    random_state: int = 17
    bootstrap_samples: int = 2000
    bootstrap_block_months: int = 6

    def validate(self) -> None:
        if pd.Timestamp(self.training_start) >= pd.Timestamp(self.prediction_start):
            raise ValueError("training_start must precede prediction_start")
        if pd.Timestamp(self.prediction_start) > pd.Timestamp(self.prediction_end):
            raise ValueError("prediction_start must not follow prediction_end")
        if self.validation_months < 12:
            raise ValueError("validation_months must be at least 12")
        if self.probability_scale <= 0:
            raise ValueError("probability_scale must be positive")
        if not 0 <= self.ml_blend_weight <= 1:
            raise ValueError("ml_blend_weight must be in [0, 1]")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


@dataclass
class WalkForwardOutput:
    predictions: pd.DataFrame
    model_selection: pd.DataFrame
    feature_importance: pd.DataFrame


@dataclass
class MLPipelineOutput:
    hybrid: BacktestResult
    ml_only: BacktestResult
    baseline: BacktestResult
    walk_forward: WalkForwardOutput
    feature_panel: pd.DataFrame
    metrics: pd.DataFrame
    robustness: pd.DataFrame
    bootstrap: pd.DataFrame


def load_external_macro(path: Path, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Load FRED features, align to trading days, and lag one business day."""
    frame = pd.read_csv(path, parse_dates=["Date"]).drop_duplicates("Date", keep="last")
    required = {"VIXCLS", "T10Y2Y", "BAA10Y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing external series: {sorted(missing)}")
    macro = frame.set_index("Date").sort_index().reindex(calendar).ffill(limit=10).shift(1)
    out = pd.DataFrame(index=calendar)
    out["vix_log"] = np.log1p(macro["VIXCLS"])
    out["vix_change_21"] = out["vix_log"].diff(21)
    out["yield_curve"] = macro["T10Y2Y"]
    out["yield_curve_change_21"] = macro["T10Y2Y"].diff(21)
    out["baa_credit_spread"] = macro["BAA10Y"]
    out["baa_credit_spread_change_21"] = macro["BAA10Y"].diff(21)
    return out


def _month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    marker = pd.Series(index, index=index)
    return pd.DatetimeIndex(marker.groupby(index.to_period("M")).tail(1).values)


def build_feature_panel(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Create monthly point-in-time features and next-month labels."""
    changes = prices.diff()
    slow_vol = changes.ewm(span=120, min_periods=120, adjust=False).std()
    fast_vol = changes.ewm(span=20, min_periods=20, adjust=False).std()
    features: dict[str, pd.DataFrame] = {}
    for horizon in (21, 63, 126, 252):
        features[f"trend_{horizon}"] = (
            (prices - prices.shift(horizon)) / (slow_vol * math.sqrt(horizon))
        ).clip(-6, 6)
    features["vol_ratio_20_120"] = (fast_vol / slow_vol.replace(0, np.nan)).clip(0.2, 5)
    rolling_min = prices.rolling(252, min_periods=252).min()
    rolling_max = prices.rolling(252, min_periods=252).max()
    features["breakout_252"] = (
        2 * (prices - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan) - 1
    ).clip(-1, 1)
    features["skew_63"] = changes.rolling(63, min_periods=50).skew().clip(-5, 5)
    features["down_day_share_63"] = changes.lt(0).rolling(63, min_periods=50).mean()
    features["cross_section_rank_252"] = features["trend_252"].rank(axis=1, pct=True)

    class_trend = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for _, symbols in ASSET_CLASSES.items():
        class_mean = features["trend_252"].loc[:, symbols].mean(axis=1)
        class_trend.loc[:, symbols] = np.repeat(
            class_mean.to_numpy()[:, None], len(symbols), axis=1
        )
    features["class_trend_252"] = class_trend

    month_ends = _month_end_index(prices.index)
    monthly_prices = prices.loc[month_ends]
    future_change = monthly_prices.shift(-1) - monthly_prices
    future_risk_return = future_change / (slow_vol.loc[month_ends] * math.sqrt(21))
    next_dates = pd.Series(month_ends, index=month_ends).shift(-1)

    frames = []
    macro_monthly = macro.reindex(month_ends)
    for symbol in prices.columns:
        frame = pd.DataFrame(index=month_ends)
        for feature_name, values in features.items():
            frame[feature_name] = values.loc[month_ends, symbol]
        for macro_name in MACRO_FEATURES:
            frame[macro_name] = macro_monthly[macro_name]
        frame["asset_class"] = metadata.loc[symbol, "asset_class"]
        frame["symbol"] = symbol
        frame["feature_date"] = month_ends
        frame["label_end"] = next_dates.to_numpy()
        frame["future_risk_return"] = future_risk_return[symbol].to_numpy()
        frame["target"] = np.where(
            frame["future_risk_return"].notna(),
            (frame["future_risk_return"] > 0).astype(float),
            np.nan,
        )
        frames.append(frame.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["feature_date", "symbol"]).reset_index(drop=True)


def _preprocessor(feature_columns: tuple[str, ...] = FEATURE_COLUMNS) -> ColumnTransformer:
    numeric = [column for column in feature_columns if column != "asset_class"]
    categorical = ["asset_class"] if "asset_class" in feature_columns else []
    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "class",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0)


def candidate_models(
    random_state: int,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Pipeline]:
    """Small, declared candidate set for nested walk-forward selection."""
    constructors: dict[str, Callable[[], BaseEstimator]] = {
        "Ridge logistic C=0.05": lambda: LogisticRegression(
            C=0.05, max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Ridge logistic C=0.20": lambda: LogisticRegression(
            C=0.20, max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Shallow GBDT depth=1": lambda: HistGradientBoostingClassifier(
            max_depth=1,
            max_iter=100,
            learning_rate=0.05,
            l2_regularization=5.0,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Shallow GBDT depth=2": lambda: HistGradientBoostingClassifier(
            max_depth=2,
            max_iter=100,
            learning_rate=0.05,
            l2_regularization=5.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline([("preprocess", _preprocessor(feature_columns)), ("model", constructor())])
        for name, constructor in constructors.items()
    }


def walk_forward_predict(
    panel: pd.DataFrame,
    config: MLConfig,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> WalkForwardOutput:
    """Select, refit, and predict one calendar year at a time without leakage."""
    config.validate()
    start = pd.Timestamp(config.prediction_start)
    end = pd.Timestamp(config.prediction_end)
    years = sorted(panel.loc[panel["feature_date"].between(start, end), "feature_date"].dt.year.unique())
    prediction_frames = []
    selection_rows = []
    importance_rows = []

    for year in years:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=31)
        test = panel.loc[panel["feature_date"].between(test_start, test_end)].copy()
        available = panel.loc[
            panel["target"].notna()
            & panel["label_end"].notna()
            & panel["label_end"].lt(test_start)
            & panel["feature_date"].ge(pd.Timestamp(config.training_start))
        ].copy()
        dates = pd.DatetimeIndex(sorted(available["feature_date"].unique()))
        if len(dates) <= config.validation_months + 12:
            continue
        validation_dates = dates[-config.validation_months :]
        validation_start = validation_dates[0]
        fit = available.loc[available["feature_date"].lt(validation_start)]
        validation = available.loc[available["feature_date"].ge(validation_start)]

        candidates = candidate_models(config.random_state, feature_columns)
        scores = []
        for name, model in candidates.items():
            model.fit(fit.loc[:, feature_columns], fit["target"].astype(int))
            probability = model.predict_proba(validation.loc[:, feature_columns])[:, 1]
            scores.append(
                {
                    "model": name,
                    "validation_log_loss": log_loss(validation["target"], probability, labels=[0, 1]),
                    "validation_auc": roc_auc_score(validation["target"], probability),
                    "model_object": model,
                }
            )
        score_frame = pd.DataFrame(scores).sort_values(
            ["validation_log_loss", "validation_auc"], ascending=[True, False]
        )
        best_name = str(score_frame.iloc[0]["model"])
        best = candidate_models(config.random_state, feature_columns)[best_name]
        best.fit(available.loc[:, feature_columns], available["target"].astype(int))

        predicted = test[[
            "feature_date",
            "label_end",
            "symbol",
            "asset_class",
            "target",
            "future_risk_return",
            "trend_252",
        ]].copy()
        predicted["probability_up"] = best.predict_proba(test.loc[:, feature_columns])[:, 1]
        predicted["selected_model"] = best_name
        predicted["walk_forward_year"] = year
        prediction_frames.append(predicted)

        best_validation = score_frame.iloc[0]
        selection_rows.append(
            {
                "year": year,
                "selected_model": best_name,
                "train_rows": len(available),
                "train_start": available["feature_date"].min(),
                "last_available_label": available["label_end"].max(),
                "validation_start": validation_start,
                "validation_rows": len(validation),
                "validation_log_loss": best_validation["validation_log_loss"],
                "validation_auc": best_validation["validation_auc"],
            }
        )

        # Model-agnostic importance on the same trailing validation slice used
        # for selection. Permutations operate on the original feature columns.
        validation_model = best_validation["model_object"]
        importance = permutation_importance(
            validation_model,
            validation.loc[:, feature_columns],
            validation["target"].astype(int),
            scoring="neg_log_loss",
            n_repeats=3,
            random_state=config.random_state,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            feature_columns, importance.importances_mean, importance.importances_std, strict=True
        ):
            importance_rows.append(
                {"year": year, "feature": feature, "importance": mean, "importance_std": std}
            )

    if not prediction_frames:
        raise ValueError("No walk-forward years had sufficient training history")
    return WalkForwardOutput(
        predictions=pd.concat(prediction_frames, ignore_index=True),
        model_selection=pd.DataFrame(selection_rows),
        feature_importance=pd.DataFrame(importance_rows),
    )


def probability_to_signal(probability: pd.Series, scale: float) -> pd.Series:
    return ((probability - 0.5) / scale).clip(-1, 1)


def predictions_to_signal_matrix(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    probability_scale: float,
    ml_blend_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = predictions.copy()
    rows["ml_signal"] = probability_to_signal(rows["probability_up"], probability_scale)
    rows["trend_prior"] = np.sign(rows["trend_252"])
    rows["hybrid_signal"] = (
        ml_blend_weight * rows["ml_signal"]
        + (1 - ml_blend_weight) * rows["trend_prior"]
    ).clip(-1, 1)

    def matrix(column: str) -> pd.DataFrame:
        monthly = rows.pivot(index="feature_date", columns="symbol", values=column)
        output = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        common_dates = output.index.intersection(monthly.index)
        output.loc[common_dates, monthly.columns] = monthly.loc[common_dates]
        return output

    return matrix("hybrid_signal"), matrix("ml_signal")


def classification_metrics(predictions: pd.DataFrame, oos_start: str) -> pd.Series:
    evaluation = predictions.loc[
        predictions["feature_date"].ge(oos_start) & predictions["target"].notna()
    ]
    y = evaluation["target"].astype(int)
    probability = evaluation["probability_up"]
    label = probability.ge(0.5).astype(int)
    return pd.Series(
        {
            "Observations": len(evaluation),
            "Accuracy": accuracy_score(y, label),
            "Balanced accuracy": balanced_accuracy_score(y, label),
            "ROC AUC": roc_auc_score(y, probability),
            "Log loss": log_loss(y, probability, labels=[0, 1]),
            "Brier score": brier_score_loss(y, probability),
            "Unconditional up rate": y.mean(),
        },
        name="Walk-forward classifier",
    )


def _block_bootstrap_sharpe(
    returns: pd.Series,
    *,
    samples: int,
    block_months: int,
    random_state: int,
) -> dict[str, float]:
    monthly = ((1 + returns).resample("ME").prod() - 1).dropna().to_numpy()
    rng = np.random.default_rng(random_state)
    n = len(monthly)
    starts = np.arange(max(1, n - block_months + 1))
    sharpes = np.empty(samples)
    for sample in range(samples):
        draws = []
        while len(draws) < n:
            start = int(rng.choice(starts))
            draws.extend(monthly[start : start + block_months])
        draw = np.asarray(draws[:n])
        sharpes[sample] = draw.mean() / draw.std(ddof=1) * math.sqrt(12) if draw.std(ddof=1) else np.nan
    finite = sharpes[np.isfinite(sharpes)]
    return {
        "Sharpe 5%": float(np.quantile(finite, 0.05)),
        "Sharpe median": float(np.quantile(finite, 0.50)),
        "Sharpe 95%": float(np.quantile(finite, 0.95)),
        "P(Sharpe > 0)": float((finite > 0).mean()),
    }


def bootstrap_table(
    results: list[BacktestResult],
    base_config: BacktestConfig,
    ml_config: MLConfig,
) -> pd.DataFrame:
    rows = []
    for offset, result in enumerate(results):
        returns = result.daily.loc[base_config.oos_start : base_config.oos_end, "net_return"]
        rows.append(
            {
                "Strategy": result.name,
                **_block_bootstrap_sharpe(
                    returns,
                    samples=ml_config.bootstrap_samples,
                    block_months=ml_config.bootstrap_block_months,
                    random_state=ml_config.random_state + offset,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")


def robustness_table(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    base_config: BacktestConfig,
    ml_config: MLConfig,
) -> pd.DataFrame:
    rows = []
    scenarios = []
    for blend in (0.25, 0.50, 0.75, 1.00):
        scenarios.append((f"ML blend {blend:.0%}", blend, ml_config.probability_scale, 1.0))
    for scale in (0.10, 0.20, 0.30):
        scenarios.append((f"Probability scale {scale:.2f}", ml_config.ml_blend_weight, scale, 1.0))
    for cost in (2.0, 3.0):
        scenarios.append((f"{cost:.0f}x trading costs", ml_config.ml_blend_weight, ml_config.probability_scale, cost))

    for label, blend, scale, cost_multiple in scenarios:
        signal, _ = predictions_to_signal_matrix(
            predictions,
            prices,
            probability_scale=scale,
            ml_blend_weight=blend,
        )
        scenario_config = replace(
            base_config,
            half_spread_ticks=base_config.half_spread_ticks * cost_multiple,
            commission_per_contract=base_config.commission_per_contract * cost_multiple,
        )
        result = run_backtest(
            scenario_config,
            name=label,
            signal_override=signal,
            prices=prices,
            metadata=metadata,
        )
        metrics = performance_metrics(result, base_config.oos_start, base_config.oos_end)
        rows.append(
            {
                "Scenario": label,
                "CAGR": metrics["CAGR"],
                "Volatility": metrics["Annualized volatility"],
                "Sharpe": metrics["Sharpe (rf=0)"],
                "Max drawdown": metrics["Max drawdown"],
                "Annual cost drag": metrics["Annual cost drag"],
            }
        )
    return pd.DataFrame(rows).set_index("Scenario")


def _save_plot(results: list[BacktestResult], base_config: BacktestConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = pd.concat(
        {
            result.name: (
                1 + result.daily.loc[base_config.oos_start : base_config.oos_end, "net_return"]
            ).cumprod()
            for result in results
        },
        axis=1,
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    ax = curves.plot(figsize=(12, 5.5), color=["#7B2CBF", "#D97706", "#006D77"])
    for line, width in zip(ax.lines, [2.1, 1.6, 1.8], strict=True):
        line.set_linewidth(width)
    ax.set_yscale("log")
    ax.set_title("Walk-forward ML hybrid vs. ML-only and 12-month trend baseline")
    ax.set_ylabel("Growth of $1, net of estimated costs (log scale)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(base_config.output_dir / "ml_performance.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_outputs(
    output: MLPipelineOutput,
    base_config: BacktestConfig,
) -> None:
    directory = base_config.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    output.metrics.to_csv(directory / "ml_metrics.csv")
    output.robustness.to_csv(directory / "ml_robustness.csv")
    output.bootstrap.to_csv(directory / "ml_bootstrap.csv")
    classification_metrics(output.walk_forward.predictions, base_config.oos_start).to_csv(
        directory / "ml_classification_metrics.csv", header=True
    )
    _save_plot([output.hybrid, output.ml_only, output.baseline], base_config)


def run_ml_pipeline(
    base_config: BacktestConfig,
    ml_config: MLConfig,
    external_macro_path: Path,
) -> MLPipelineOutput:
    ml_config.validate()
    prices = load_prices(base_config.data_dir)
    metadata = load_metadata(base_config.data_dir)
    macro = load_external_macro(external_macro_path, prices.index)
    panel = build_feature_panel(prices, metadata, macro)
    walk_forward = walk_forward_predict(panel, ml_config)
    hybrid_signal, ml_signal = predictions_to_signal_matrix(
        walk_forward.predictions,
        prices,
        probability_scale=ml_config.probability_scale,
        ml_blend_weight=ml_config.ml_blend_weight,
    )
    hybrid = run_backtest(
        base_config,
        name="Walk-forward ML hybrid CTA",
        signal_override=hybrid_signal,
        prices=prices,
        metadata=metadata,
    )
    ml_only = run_backtest(
        base_config,
        name="Walk-forward ML-only CTA",
        signal_override=ml_signal,
        prices=prices,
        metadata=metadata,
    )
    baseline = run_backtest(base_config, prices=prices, metadata=metadata)
    metrics = pd.concat(
        [
            performance_metrics(result, base_config.oos_start, base_config.oos_end)
            for result in (hybrid, ml_only, baseline)
        ],
        axis=1,
    ).T
    robustness = robustness_table(
        walk_forward.predictions, prices, metadata, base_config, ml_config
    )
    bootstrap = bootstrap_table([hybrid, ml_only, baseline], base_config, ml_config)
    output = MLPipelineOutput(
        hybrid=hybrid,
        ml_only=ml_only,
        baseline=baseline,
        walk_forward=walk_forward,
        feature_panel=panel,
        metrics=metrics,
        robustness=robustness,
        bootstrap=bootstrap,
    )
    save_outputs(output, base_config)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--external-macro", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--oos-start", default="2005-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = BacktestConfig(args.data_dir, args.output_dir, oos_start=args.oos_start)
    output = run_ml_pipeline(base, MLConfig(), args.external_macro)
    print(output.metrics.to_string())


if __name__ == "__main__":
    main()
