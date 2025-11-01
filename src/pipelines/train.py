"""Training pipeline orchestration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import dump

from src.evaluation.metrics import regression_report
from src.evaluation.visualization import plot_actual_vs_predicted, save_figure_html, save_figure_png
from src.features.engineering import build_feature_matrix
from src.models.training import train_model


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_macro_frames(macro_cfg: Dict) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for entry in macro_cfg:
        macro_path = Path(entry["path"])
        if not macro_path.exists():
            continue
        date_column = entry.get("date_column", "Date")
        value_column = entry.get("value_column", "value")
        alias = entry.get("alias", macro_path.stem)
        macro_df = pd.read_csv(macro_path)
        macro_df = macro_df.rename(columns={date_column: "Date", value_column: alias})
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        macro_df[alias] = pd.to_numeric(macro_df[alias], errors="coerce")
        frames[alias] = macro_df[["Date", alias]]
    return frames


def load_benchmark_frames(benchmark_cfg: Dict) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for entry in benchmark_cfg:
        benchmark_path = Path(entry["path"])
        if not benchmark_path.exists():
            continue
        date_column = entry.get("date_column", "Date")
        value_column = entry.get("value_column", "Close")
        alias = entry.get("alias", benchmark_path.stem)
        benchmark_df = pd.read_csv(benchmark_path)
        benchmark_df = benchmark_df.rename(columns={date_column: "Date", value_column: alias})
        benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"])
        benchmark_df[alias] = pd.to_numeric(benchmark_df[alias], errors="coerce")
        frames[alias] = benchmark_df[["Date", alias]]
    return frames


def trim_to_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if df.empty or window_days is None:
        return df
    cutoff = df["Date"].max() - pd.Timedelta(days=int(window_days))
    return df[df["Date"] >= cutoff].reset_index(drop=True)


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    split_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("validation_ratio", 0.15))
    n_samples = len(features)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X = features.drop(columns=["Date"])
    dates = features["Date"].to_numpy()

    X_train = X.iloc[:train_end].to_numpy()
    y_train = target.iloc[:train_end].to_numpy()

    X_val = X.iloc[train_end:val_end].to_numpy()
    y_val = target.iloc[train_end:val_end].to_numpy()

    X_test = X.iloc[val_end:].to_numpy()
    y_test = target.iloc[val_end:].to_numpy()

    date_test = dates[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test, date_test


def run_training(config_path: str) -> Dict:
    config = load_yaml(Path(config_path))

    price_df = load_price_data(Path(config["data"]["price_path"]))
    macro_frames = load_macro_frames(config["data"].get("macro", []))
    benchmark_frames = load_benchmark_frames(config["data"].get("benchmarks", []))
    window_days = config["data"].get("window_days")
    if window_days:
        price_df = trim_to_window(price_df, window_days)
        macro_frames = {
            alias: trim_to_window(frame, window_days) for alias, frame in macro_frames.items()
        }
        benchmark_frames = {
            alias: trim_to_window(frame, window_days) for alias, frame in benchmark_frames.items()
        }

    features, target = build_feature_matrix(
        price_df=price_df,
        macro_frames=macro_frames,
        benchmark_frames=benchmark_frames,
        feature_cfg=config.get("feature_engineering"),
        target_cfg=config.get("target"),
    )

    if len(features) == 0 or len(target) == 0:
        raise ValueError(
            "Feature matrix is empty after preprocessing. "
            "Reduce `data.window_days`, review feature settings, or ensure input datasets contain data for the selected window."
        )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_dates,
    ) = split_dataset(features, target, config.get("split", {}))

    X_train_combined = X_train
    y_train_combined = y_train
    if len(X_val):
        X_train_combined = np.vstack([X_train, X_val]) if len(X_train) else X_val
        y_train_combined = np.concatenate([y_train, y_val]) if len(y_train) else y_val
    if len(X_train_combined) == 0 or len(y_train_combined) == 0:
        raise ValueError(
            "No training samples available. Check the split ratios and window size; "
            "increase `train_ratio` or expand `data.window_days`."
        )

    models_cfg = config.get("models")
    model_info = {}

    if models_cfg:
        candidates_info: List[Dict] = []
        best_candidate_cfg = None
        best_candidate_info = None
        best_val_score = float("inf")

        for candidate_cfg in models_cfg:
            candidate_name = candidate_cfg.get("name") or candidate_cfg.get("type", "model")
            estimator, info = train_model(candidate_cfg, X_train, y_train)
            val_predictions = estimator.predict(X_val) if len(X_val) else np.array([])
            val_metrics = (
                regression_report(y_val, val_predictions) if len(val_predictions) else {}
            )
            val_rmse = val_metrics.get("rmse", float("inf")) if len(val_predictions) else float("inf")

            info.update(
                {
                    "candidate": candidate_name,
                    "validation_metrics": val_metrics,
                }
            )
            candidates_info.append(info)

            if len(X_val):
                score = val_rmse
            else:
                best_score = info.get("best_score")
                score = -best_score if best_score is not None else float("inf")

            if score < best_val_score:
                best_val_score = score
                best_candidate_cfg = candidate_cfg
                best_candidate_info = info

        if best_candidate_cfg is None or best_candidate_info is None:
            raise ValueError("No valid model candidate produced a result.")

        combined_cfg = deepcopy(best_candidate_cfg)
        combined_params = combined_cfg.get("params", {}).copy()
        combined_params.update(best_candidate_info.get("best_params", {}))
        combined_cfg["params"] = combined_params
        combined_cfg.pop("search", None)

        final_model, final_info = train_model(combined_cfg, X_train_combined, y_train_combined)
        final_info.update(best_candidate_info)
        final_info["selected_on"] = "validation_rmse" if len(X_val) else "cv_best_score"
        final_info["candidates"] = candidates_info
        model = final_model
        model_info = final_info
    else:
        model, model_info = train_model(
            config.get("model", {}),
            X_train_combined,
            y_train_combined,
        )
        fallback_model_cfg = config.get("model", {}) or {}
        model_info.setdefault(
            "candidate", fallback_model_cfg.get("name") or fallback_model_cfg.get("type", "model")
        )

    val_predictions = model.predict(X_val) if len(X_val) else np.array([])
    test_predictions = model.predict(X_test) if len(X_test) else np.array([])

    metrics = {
        "validation": regression_report(y_val, val_predictions) if len(val_predictions) else {},
        "test": regression_report(y_test, test_predictions) if len(test_predictions) else {},
        "model": model_info,
    }

    outputs = config.get("outputs", {})
    metrics_path = Path(outputs.get("metrics_path", "reports/metrics/latest.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Daily predictions on the evaluation horizon
    predictions_path = Path(outputs.get("predictions_path", "reports/datasets/latest_predictions.csv"))
    predictions_weekly_path = Path(
        outputs.get("predictions_weekly_path", "reports/datasets/latest_predictions_weekly.csv")
    )
    predictions_full_path = Path(
        outputs.get("predictions_full_path", "reports/datasets/latest_predictions_full.csv")
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_weekly_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_full_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "date": test_dates,
            "actual": y_test,
            "predicted": test_predictions,
        }
    )
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])
    prediction_frame.to_csv(predictions_path, index=False)

    # Predictions across the entire window (train/val/test)
    full_features = features.drop(columns=["Date"])
    full_predictions = model.predict(full_features.to_numpy())
    full_frame = pd.DataFrame(
        {
            "date": features["Date"],
            "actual": target,
            "predicted": full_predictions,
        }
    ).sort_values("date")
    full_frame.to_csv(predictions_full_path, index=False)

    weekly_freq = outputs.get("weekly_freq", "W-MON")
    weekly_weeks = int(outputs.get("weekly_weeks", 13))
    weekly_series = (
        full_frame.set_index("date")
        .resample(weekly_freq, label="left", closed="left")
        .mean()
        .dropna()
    )
    if weekly_weeks > 0:
        weekly_series = weekly_series.tail(weekly_weeks)
    weekly_frame = weekly_series.reset_index()
    weekly_frame.to_csv(predictions_weekly_path, index=False)

    model_path = Path(outputs.get("model_path", "trained_models/latest_model.pkl"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)

    target_cfg = config.get("target", {})
    transform = target_cfg.get("transform", "return")
    y_axis_label = "Return" if transform in {"return", "log_return"} else "Price"

    fig = plot_actual_vs_predicted(
        test_dates,
        y_test,
        test_predictions,
        title="Actual vs Predicted",
        y_axis_label=y_axis_label,
    )
    plot_path = Path(outputs.get("plot_path", "reports/figures/latest_actual_vs_predicted.html"))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_html(fig, plot_path)
    save_figure_png(fig, plot_path.with_suffix(".png"))

    weekly_fig = plot_actual_vs_predicted(
        weekly_frame["date"],
        weekly_frame["actual"],
        weekly_frame["predicted"],
        title="Actual vs Predicted (Weekly)",
        y_axis_label=y_axis_label,
    )
    plot_weekly_path = Path(
        outputs.get(
            "plot_weekly_path",
            "reports/figures/latest_actual_vs_predicted_weekly.html",
        )
    )
    plot_weekly_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_html(weekly_fig, plot_weekly_path)
    save_figure_png(weekly_fig, plot_weekly_path.with_suffix(".png"))

    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "predictions_full_path": str(predictions_full_path),
        "predictions_weekly_path": str(predictions_weekly_path),
        "model_path": str(model_path),
        "weekly_plot_path": str(plot_weekly_path),
        "model_info": model_info,
    }


if __name__ == "__main__":  # pragma: no cover - CLI usage
    import argparse

    parser = argparse.ArgumentParser(description="Run training pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to pipeline configuration file.",
    )
    args = parser.parse_args()
    run_training(args.config)
