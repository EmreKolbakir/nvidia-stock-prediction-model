import json
from pathlib import Path

import pandas as pd

from src.pipelines.train import run_training


def build_price_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=120, freq="D"),
            "Open": 100 + pd.Series(range(120)) * 0.1,
            "High": 100 + pd.Series(range(120)) * 0.1 + 1,
            "Low": 100 + pd.Series(range(120)) * 0.1 - 1,
            "Close": 100 + pd.Series(range(120)) * 0.1,
            "Adj Close": 100 + pd.Series(range(120)) * 0.1,
            "Volume": 1_000_000 + pd.Series(range(120)) * 1000,
        }
    )
    frame.to_csv(path, index=False)
    return path


def build_macro_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=120, freq="D"),
            "value": 0.05,
        }
    )
    frame.to_csv(path, index=False)
    return path


def build_sentiment_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=120, freq="D"),
            "sentiment_score": pd.Series(range(120)).apply(lambda x: ((x % 10) - 5) / 5),
        }
    )
    frame.to_csv(path, index=False)
    return path


def build_benchmark_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=120, freq="D"),
            "Close": 95 + pd.Series(range(120)) * 0.08,
        }
    )
    frame.to_csv(path, index=False)
    return path


def write_pipeline_config(path: Path, price_file: Path, macro_file: Path, sentiment_file: Path, benchmark_file: Path, outputs_dir: Path) -> Path:
    config = f"""
data:
  price_path: {price_file}
  window_days: 90
  macro:
    - path: {macro_file}
      date_column: date
      value_column: value
      alias: fed_funds_rate
    - path: {sentiment_file}
      date_column: date
      value_column: sentiment_score
      alias: nvda_sentiment
  benchmarks:
    - path: {benchmark_file}
      date_column: Date
      value_column: Close
      alias: spy_close

target:
  column: Close
  horizon: 1
  transform: price

feature_engineering:
  lag_features:
    columns: [Close]
    lags: [1, 2]
  rolling_windows:
    columns: [Close]
    windows: [3]
    statistics: [mean]
  technical_indicators:
    rsi:
      column: Close
      window: 5
  momentum:
    columns: [Close]
    periods: [1]
  volatility:
    columns: [Close]
    windows: [3]
  bollinger:
    columns: [Close]
    window: 5
    std_multiplier: 2
  volume:
    pct_change_periods: [1]
    rolling_zscore_windows: [5]
  calendar:
    include: [day_of_week]
  benchmark_relations:
    - benchmark: spy_close
      price_column: Close
      feature: relative_return
  sentiment:
    columns: [nvda_sentiment]
    windows: [3]
    statistics: [mean]

split:
  train_ratio: 0.6
  validation_ratio: 0.2
  test_ratio: 0.2

models:
  - name: rf
    type: random_forest
    params:
      n_estimators: 50
      random_state: 42
    search:
      type: grid
      param_grid:
        n_estimators: [25, 50]
        max_depth: [3, 5]
      cv: 3
      n_jobs: 1
  - name: gb
    type: gradient_boosting
    params:
      n_estimators: 40
      learning_rate: 0.1
      max_depth: 3
      random_state: 42

outputs:
  metrics_path: {outputs_dir / "metrics.json"}
  predictions_path: {outputs_dir / "predictions.csv"}
  predictions_full_path: {outputs_dir / "predictions_full.csv"}
  predictions_weekly_path: {outputs_dir / "predictions_weekly.csv"}
  model_path: {outputs_dir / "model.pkl"}
  plot_path: {outputs_dir / "plot.html"}
  plot_weekly_path: {outputs_dir / "plot_weekly.html"}
  weekly_weeks: 4
  weekly_freq: W-MON
"""
    path.write_text(config, encoding="utf-8")
    return path


def test_run_training_creates_outputs(tmp_path):
    price_file = build_price_csv(tmp_path / "prices.csv")
    macro_file = build_macro_csv(tmp_path / "macro.csv")
    sentiment_file = build_sentiment_csv(tmp_path / "sentiment.csv")
    benchmark_file = build_benchmark_csv(tmp_path / "benchmark.csv")
    outputs_dir = tmp_path / "outputs"
    config_path = write_pipeline_config(
        tmp_path / "pipeline.yaml", price_file, macro_file, sentiment_file, benchmark_file, outputs_dir
    )

    result = run_training(str(config_path))
    assert "metrics" in result
    assert "model_info" in result
    assert result["model_info"]["candidate"] in {"rf", "gb"}
    assert "best_params" in result["model_info"]
    assert "predictions_weekly_path" in result
    assert "weekly_plot_path" in result

    metrics_file = outputs_dir / "metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "test" in metrics
    assert "model" in metrics
    assert "best_params" in metrics["model"]
    assert "candidates" in metrics["model"]
    assert len(metrics["model"]["candidates"]) == 2
    assert "mae" in metrics["test"]
    assert "r2" in metrics["test"]

    predictions_file = outputs_dir / "predictions.csv"
    assert predictions_file.exists()

    predictions_full_file = outputs_dir / "predictions_full.csv"
    assert predictions_full_file.exists()
    full_df = pd.read_csv(predictions_full_file)
    assert not full_df.empty

    predictions_weekly_file = outputs_dir / "predictions_weekly.csv"
    assert predictions_weekly_file.exists()
    weekly_df = pd.read_csv(predictions_weekly_file)
    assert not weekly_df.empty

    model_file = outputs_dir / "model.pkl"
    assert model_file.exists()

    plot_file = outputs_dir / "plot.html"
    assert plot_file.exists()

    plot_weekly_file = outputs_dir / "plot_weekly.html"
    assert plot_weekly_file.exists()
