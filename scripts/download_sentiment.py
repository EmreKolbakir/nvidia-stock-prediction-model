#!/usr/bin/env python
"""Download daily sentiment scores for NVDA using external APIs.

Currently supported provider:
    - Alpha Vantage (NEWS_SENTIMENT endpoint)

Usage:
    python scripts/download_sentiment.py \
        --provider alpha_vantage \
        --symbol NVDA \
        --output data/external/nvda_sentiment.csv \
        --apikey-env ALPHA_VANTAGE_API_KEY \
        --lookback-days 365

The script requires an API key stored in the specified environment variable.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NVDA sentiment data from external APIs.")
    parser.add_argument(
        "--provider",
        choices=["alpha_vantage"],
        default="alpha_vantage",
        help="Sentiment data provider to use.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="NVDA",
        help="Ticker symbol to retrieve sentiment for.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/nvda_sentiment.csv"),
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--apikey-env",
        type=str,
        default="ALPHA_VANTAGE_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Number of days to include from today when filtering the feed.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def aggregate_daily_sentiment(records: Iterable[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for item in records:
        published = item.get("time_published")
        score = item.get("overall_sentiment_score")
        if published is None or score is None:
            continue
        # Alpha Vantage returns timestamps in YYYYMMDDTHHMM format.
        date = datetime.strptime(published[:8], "%Y%m%d").date()
        rows.append({"date": date, "sentiment_score": float(score)})
    if not rows:
        raise ValueError("No sentiment records were parsed from the API response.")
    df = pd.DataFrame(rows)
    aggregated = (
        df.groupby("date", as_index=False)["sentiment_score"]
        .mean()
        .sort_values("date")
    )
    return aggregated


def fetch_alpha_vantage(symbol: str, api_key: str, lookback_days: int) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": api_key,
        "sort": "LATEST",
        "limit": 1000,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "feed" not in payload:
        raise ValueError(f"Unexpected response from Alpha Vantage: {payload}")

    lookback_cutoff = datetime.utcnow().date() - timedelta(days=lookback_days)
    filtered_feed = [
        item for item in payload["feed"]
        if item.get("time_published")
        and datetime.strptime(item["time_published"][:8], "%Y%m%d").date() >= lookback_cutoff
    ]
    if not filtered_feed:
        raise ValueError("Alpha Vantage returned no news items within the lookback window.")

    return aggregate_daily_sentiment(filtered_feed)


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.apikey_env)
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Please export {args.apikey_env}=<your_api_key> before running."
        )

    if args.provider == "alpha_vantage":
        sentiment_df = fetch_alpha_vantage(args.symbol, api_key, args.lookback_days)
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    ensure_directory(args.output)
    sentiment_df.to_csv(args.output, index=False)
    print(f"Saved sentiment scores to {args.output}")


if __name__ == "__main__":
    main()
