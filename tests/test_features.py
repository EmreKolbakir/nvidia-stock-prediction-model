import numpy as np
import pandas as pd

from src.features.engineering import build_feature_matrix


def test_build_feature_matrix_generates_expected_columns():
    date_range = pd.date_range("2020-01-01", periods=60, freq="D")
    price_df = pd.DataFrame(
        {
            "Date": date_range,
            "Open": range(60),
            "High": range(1, 61),
            "Low": range(60),
            "Close": [100 + i * 0.5 for i in range(60)],
            "Adj Close": [100 + i * 0.5 for i in range(60)],
            "Volume": [1_000_000 + i * 1000 for i in range(60)],
        }
    )

    macro_df = pd.DataFrame({"Date": date_range, "fed_funds_rate": 0.05})
    sentiment_df = pd.DataFrame({"Date": date_range, "nvda_sentiment": np.linspace(-1, 1, 60)})
    benchmark_df = pd.DataFrame({"Date": date_range, "spy_close": 200 + pd.Series(range(60)) * 0.3})

    features, target = build_feature_matrix(
        price_df,
        macro_frames={"fed_funds_rate": macro_df, "nvda_sentiment": sentiment_df},
        benchmark_frames={"spy_close": benchmark_df},
        feature_cfg={
            "lag_features": {"columns": ["Close"], "lags": [1, 2]},
            "rolling_windows": {"columns": ["Close"], "windows": [3], "statistics": ["mean"]},
            "technical_indicators": {"rsi": {"column": "Close", "window": 5}},
            "momentum": {"columns": ["Close"], "periods": [1]},
            "volatility": {"columns": ["Close"], "windows": [3]},
            "bollinger": {"columns": ["Close"], "window": 3, "std_multiplier": 2},
            "volume": {"pct_change_periods": [1], "rolling_zscore_windows": [3]},
            "calendar": {"include": ["day_of_week"]},
            "benchmark_relations": [{"benchmark": "spy_close", "price_column": "Close", "feature": "relative_return"}],
            "sentiment": {"columns": ["nvda_sentiment"], "windows": [3], "statistics": ["mean"]},
        },
        target_cfg={"column": "Close", "horizon": 1, "transform": "price"},
    )

    assert "Close_lag_1" in features.columns
    assert "Close_roll_mean_3" in features.columns
    assert "Close_rsi_5" in features.columns
    assert "fed_funds_rate" in features.columns
    assert "spy_close" in features.columns
    assert "Close_return_1" in features.columns
    assert "Close_volatility_3" in features.columns
    assert "Close_bollinger_upper_3" in features.columns
    assert "Volume_pct_change_1" in features.columns
    assert "day_of_week" in features.columns
    assert "Close_vs_spy_close_return_spread" in features.columns
    assert "nvda_sentiment_sent_mean_3" in features.columns
    assert len(features) == len(target)
