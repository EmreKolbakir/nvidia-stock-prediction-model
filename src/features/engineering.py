"""Feature engineering utilities for NVDA stock prediction."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_lag_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    lags: Iterable[int],
) -> pd.DataFrame:
    """Append lagged versions of selected columns."""
    result = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        for lag in lags:
            result[f"{col}_lag_{lag}"] = frame[col].shift(lag)
    return result


def compute_rolling_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
    statistics: Iterable[str],
) -> pd.DataFrame:
    """Append rolling window statistics."""
    result = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        for window in windows:
            roll = series.rolling(window=window)
            for stat in statistics:
                key = f"{col}_roll_{stat}_{window}"
                if stat == "mean":
                    result[key] = roll.mean()
                elif stat == "std":
                    result[key] = roll.std()
                elif stat == "min":
                    result[key] = roll.min()
                elif stat == "max":
                    result[key] = roll.max()
    return result


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Moving Average Convergence Divergence indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def apply_technical_indicators(
    frame: pd.DataFrame,
    indicators: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Compute technical indicators from configuration."""
    result = frame.copy()
    for name, params in indicators.items():
        column = params.get("column", "Close")
        if column not in frame.columns:
            continue
        if name.lower() == "rsi":
            window = int(params.get("window", 14))
            result[f"{column}_rsi_{window}"] = compute_rsi(frame[column], window=window)
        elif name.lower() == "macd":
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            signal = int(params.get("signal", 9))
            macd, signal_line = compute_macd(frame[column], fast=fast, slow=slow, signal=signal)
            result[f"{column}_macd_{fast}_{slow}"] = macd
            result[f"{column}_macd_signal_{signal}"] = signal_line
    return result


def add_momentum_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    periods: Iterable[int],
) -> pd.DataFrame:
    """Add price momentum/return based features."""
    result = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        for period in periods:
            result[f"{col}_return_{period}"] = series.pct_change(periods=period, fill_method=None)
            result[f"{col}_momentum_{period}"] = series - series.shift(period)
    return result


def add_volatility_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    """Attach rolling volatility descriptors."""
    result = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        for window in windows:
            roll = series.rolling(window=window)
            result[f"{col}_volatility_{window}"] = roll.std()
            result[f"{col}_range_{window}"] = roll.max() - roll.min()
    return result


def add_bollinger_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    window: int,
    std_multiplier: float,
) -> pd.DataFrame:
    """Append Bollinger Band statistics."""
    result = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = ma + std_multiplier * std
        lower = ma - std_multiplier * std
        result[f"{col}_bollinger_upper_{window}"] = upper
        result[f"{col}_bollinger_lower_{window}"] = lower
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"{col}_bollinger_bandwidth_{window}"] = (upper - lower) / ma
    return result


def add_volume_features(
    frame: pd.DataFrame,
    pct_change_periods: Iterable[int],
    zscore_windows: Iterable[int],
) -> pd.DataFrame:
    """Create volume-derived features."""
    result = frame.copy()
    if "Volume" not in frame.columns:
        return result
    volume = frame["Volume"]
    for period in pct_change_periods:
        result[f"Volume_pct_change_{period}"] = volume.pct_change(periods=period, fill_method=None)
    for window in zscore_windows:
        mean = volume.rolling(window=window).mean()
        std = volume.rolling(window=window).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"Volume_zscore_{window}"] = (volume - mean) / std
    return result


def add_sentiment_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
    statistics: Iterable[str],
) -> pd.DataFrame:
    """Create rolling aggregates over sentiment scores."""
    result = frame.copy()
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        for window in windows:
            roll = series.rolling(window=window)
            for stat in statistics:
                key = f"{col}_sent_{stat}_{window}"
                if stat == "mean":
                    result[key] = roll.mean()
                elif stat == "std":
                    result[key] = roll.std()
                elif stat == "min":
                    result[key] = roll.min()
                elif stat == "max":
                    result[key] = roll.max()
            result[f"{col}_sent_change_{window}"] = series.diff(window)
        result[col] = result[col].bfill()
    return result


def add_calendar_features(
    frame: pd.DataFrame,
    include: Iterable[str],
) -> pd.DataFrame:
    """Add calendar/time based features."""
    result = frame.copy()
    if "Date" not in frame.columns:
        return result
    if "day_of_week" in include:
        dow = result["Date"].dt.weekday
        result["day_of_week"] = dow
        result["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        result["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    if "month" in include:
        month = result["Date"].dt.month
        result["month"] = month
        result["month_sin"] = np.sin(2 * np.pi * month / 12)
        result["month_cos"] = np.cos(2 * np.pi * month / 12)
    if "day_of_month" in include:
        result["day_of_month"] = result["Date"].dt.day
    return result


def add_benchmark_relations(
    frame: pd.DataFrame,
    relations: Iterable[Dict[str, object]],
) -> pd.DataFrame:
    """Create features comparing price series with benchmark indices."""
    result = frame.copy()
    for relation in relations:
        benchmark = relation.get("benchmark")
        price_col = relation.get("price_column", "Close")
        feature = str(relation.get("feature", "relative_return")).lower()
        if benchmark not in frame.columns or price_col not in frame.columns:
            continue
        price_series = frame[price_col]
        benchmark_series = frame[benchmark]
        price_return = price_series.pct_change(fill_method=None)
        benchmark_return = benchmark_series.pct_change(fill_method=None)
        base_name = f"{price_col}_vs_{benchmark}"
        if feature == "relative_return":
            result[f"{base_name}_return_spread"] = price_return - benchmark_return
            with np.errstate(divide="ignore", invalid="ignore"):
                result[f"{base_name}_ratio"] = price_series / benchmark_series
        elif feature == "spread":
            result[f"{base_name}_spread"] = price_series - benchmark_series
        elif feature == "correlation":
            window = int(relation.get("window", 20))
            result[f"{base_name}_rolling_corr_{window}"] = price_return.rolling(window).corr(
                benchmark_return
            )
    return result


def build_feature_matrix(
    price_df: pd.DataFrame,
    macro_frames: Optional[Dict[str, pd.DataFrame]] = None,
    benchmark_frames: Optional[Dict[str, pd.DataFrame]] = None,
    feature_cfg: Optional[Dict] = None,
    target_cfg: Optional[Dict] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create model-ready features and target from price, macro, and benchmark data."""
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["return_1d"] = df["Close"].pct_change(fill_method=None)

    feature_cfg = feature_cfg or {}

    lag_settings = feature_cfg.get("lag_features", {})
    lag_columns = lag_settings.get("columns", ["Close"])
    df = compute_lag_features(df, lag_columns, lag_settings.get("lags", [1, 2, 3]))

    rolling_cfg = feature_cfg.get("rolling_windows", {})
    roll_columns = rolling_cfg.get("columns", ["Close"])
    df = compute_rolling_features(
        df,
        roll_columns,
        rolling_cfg.get("windows", [5, 10, 21]),
        rolling_cfg.get("statistics", ["mean", "std"]),
    )

    indicators_cfg = feature_cfg.get("technical_indicators", {})
    df = apply_technical_indicators(df, indicators_cfg)

    macro_columns: List[str] = []
    if macro_frames:
        for alias, macro_df in macro_frames.items():
            macro = macro_df.copy()
            macro["Date"] = pd.to_datetime(macro["Date"])
            if alias not in macro.columns:
                value_cols = [col for col in macro.columns if col != "Date"]
                if value_cols:
                    macro = macro.rename(columns={value_cols[0]: alias})
            df = df.merge(macro[["Date", alias]], on="Date", how="left")
            macro_columns.append(alias)
    if macro_columns:
        df[macro_columns] = df[macro_columns].ffill().bfill()

    benchmark_columns: List[str] = []
    if benchmark_frames:
        for alias, benchmark_df in benchmark_frames.items():
            bench = benchmark_df.copy()
            bench["Date"] = pd.to_datetime(bench["Date"])
            if alias not in bench.columns:
                value_cols = [col for col in bench.columns if col != "Date"]
                if value_cols:
                    bench = bench.rename(columns={value_cols[0]: alias})
            df = df.merge(bench[["Date", alias]], on="Date", how="left")
            benchmark_columns.append(alias)
    if benchmark_columns:
        df[benchmark_columns] = df[benchmark_columns].ffill().bfill()

    momentum_cfg = feature_cfg.get("momentum", {})
    if momentum_cfg:
        df = add_momentum_features(
            df,
            momentum_cfg.get("columns", ["Close"]),
            momentum_cfg.get("periods", [1, 5]),
        )

    volatility_cfg = feature_cfg.get("volatility", {})
    if volatility_cfg:
        df = add_volatility_features(
            df,
            volatility_cfg.get("columns", ["Close"]),
            volatility_cfg.get("windows", [5, 10, 21]),
        )

    bollinger_cfg = feature_cfg.get("bollinger")
    if bollinger_cfg:
        df = add_bollinger_features(
            df,
            bollinger_cfg.get("columns", ["Close"]),
            int(bollinger_cfg.get("window", 20)),
            float(bollinger_cfg.get("std_multiplier", 2)),
        )

    volume_cfg = feature_cfg.get("volume", {})
    if volume_cfg:
        df = add_volume_features(
            df,
            volume_cfg.get("pct_change_periods", [1]),
            volume_cfg.get("rolling_zscore_windows", [10]),
        )

    calendar_cfg = feature_cfg.get("calendar", {})
    if calendar_cfg:
        df = add_calendar_features(
            df,
            calendar_cfg.get("include", ["day_of_week", "month"]),
        )

    benchmark_rel_cfg = feature_cfg.get("benchmark_relations", [])
    if benchmark_rel_cfg:
        df = add_benchmark_relations(df, benchmark_rel_cfg)

    sentiment_cfg = feature_cfg.get("sentiment", {})
    if sentiment_cfg:
        df = add_sentiment_features(
            df,
            sentiment_cfg.get("columns", []),
            sentiment_cfg.get("windows", [5]),
            sentiment_cfg.get("statistics", ["mean"]),
        )

    if len(df) == 0:
        raise ValueError(
            "Feature engineering produced an empty dataframe. Check window size, data availability, or feature configurations."
        )

    target_cfg = target_cfg or {}
    horizon = int(target_cfg.get("horizon", 1))
    column = target_cfg.get("column", "Close")
    transform = target_cfg.get("transform", "return")

    if transform == "return":
        future = df[column].shift(-horizon)
        df["target"] = future.divide(df[column]) - 1
    elif transform == "log_return":
        future = df[column].shift(-horizon)
        df["target"] = np.log(future / df[column])
    else:
        df["target"] = df[column].shift(-horizon)

    df = df.dropna().reset_index(drop=True)
    features = df.drop(columns=["target"])
    target = df["target"]
    return features, target
