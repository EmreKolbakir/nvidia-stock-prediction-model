"""Data ingestion helpers for NVDA stock prediction project."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directory(directory: Path) -> None:
    """Create directory if it does not exist."""
    directory.mkdir(parents=True, exist_ok=True)


def _safe_import_yfinance() -> Any:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise ImportError(
            "yfinance is required for downloading price data. "
            "Install it via `pip install yfinance`."
        ) from exc
    return yf


def _safe_import_requests() -> Any:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise ImportError(
            "requests is required for fetching macro indicators. "
            "Install it via `pip install requests`."
        ) from exc
    return requests


def download_price_series(entry: Dict[str, Any], target_dir: Path) -> Path:
    """Download OHLCV data for a ticker using Yahoo Finance."""
    yf = _safe_import_yfinance()

    symbol = entry["symbol"]
    interval = entry.get("interval", "1d")
    start_date = entry.get("start_date")
    end_date = entry.get("end_date")
    output_template = entry.get("output_template", "{name}_prices.csv")

    LOGGER.info("Downloading price data for %s (%s)", entry.get("name", symbol), symbol)

    dataset = yf.download(
        tickers=symbol,
        interval=interval,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if dataset.empty:
        raise ValueError(f"No data returned for ticker {symbol}")

    dataset = dataset.reset_index()
    ensure_directory(target_dir)
    output_path = target_dir / output_template.format(**entry)
    dataset.to_csv(output_path, index=False)
    return output_path


def download_macro_series(
    entry: Dict[str, Any],
    target_dir: Path,
    date_format: str = "%Y-%m-%d",
) -> Optional[Path]:
    """Download macroeconomic series from the FRED API."""
    series_id = entry.get("series_id")
    if not series_id:
        LOGGER.warning("Skipping macro entry without series_id: %s", entry)
        return None

    api_key_env = entry.get("api_key_env", "FRED_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        LOGGER.warning(
            "Environment variable %s not set; skipping macro series %s",
            api_key_env,
            series_id,
        )
        return None

    requests = _safe_import_requests()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": entry.get("start_date"),
        "observation_end": entry.get("end_date"),
        "frequency": entry.get("frequency", "m"),
    }

    LOGGER.info("Fetching FRED series %s", series_id)
    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations", [])

    if not observations:
        LOGGER.warning("No observations returned for %s", series_id)
        return None

    ensure_directory(target_dir)
    output_template = entry.get("output_template", "{name}_macro.csv")
    output_path = target_dir / output_template.format(**entry)

    rows = [
        {
            "date": obs["date"],
            "value": float(obs["value"]) if obs["value"] not in {"", "."} else None,
        }
        for obs in observations
    ]

    output_path.write_text(
        "date,value\n"
        + "\n".join(
            f"{row['date']},{'' if row['value'] is None else row['value']}"
            for row in rows
        ),
        encoding="utf-8",
    )
    return output_path


def process_price_entries(entries: Iterable[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Download all price series defined in the configuration."""
    output_paths: List[Path] = []
    for entry in entries:
        try:
            output_paths.append(download_price_series(entry, output_dir))
        except Exception as exc:  # pragma: no cover - download guard
            LOGGER.error("Failed to download %s: %s", entry.get("name", entry.get("symbol")), exc)
            raise
    return output_paths


def process_macro_entries(
    entries: Iterable[Dict[str, Any]],
    output_dir: Path,
    date_format: str,
) -> List[Path]:
    """Download all macro series defined in the configuration."""
    output_paths: List[Path] = []
    for entry in entries:
        try:
            path = download_macro_series(entry, output_dir, date_format=date_format)
        except Exception as exc:  # pragma: no cover - download guard
            LOGGER.error("Failed to download macro series %s: %s", entry.get("name"), exc)
            raise
        if path:
            output_paths.append(path)
    return output_paths


def download_from_config(config_path: Path) -> Dict[str, List[Path]]:
    """Download datasets described in the YAML config."""
    config = load_yaml_config(config_path)

    data_dir = Path(config.get("data_dir", "data/raw"))
    external_dir = Path(config.get("external_dir", "data/external"))
    date_format = config.get("date_format", "%Y-%m-%d")

    results: Dict[str, List[Path]] = {"prices": [], "benchmarks": [], "macro": []}

    price_entries = config.get("price_series", [])
    if price_entries:
        results["prices"] = process_price_entries(price_entries, data_dir)

    benchmark_entries = config.get("benchmarks", [])
    if benchmark_entries:
        benchmark_dir = data_dir / "benchmarks"
        ensure_directory(benchmark_dir)
        results["benchmarks"] = process_price_entries(benchmark_entries, benchmark_dir)

    macro_entries = config.get("macro_indicators", [])
    if macro_entries:
        results["macro"] = process_macro_entries(macro_entries, Path(config.get("external_dir", "data/external")), date_format)

    return results


def _configure_logging(verbosity: int = 1) -> None:
    """Set up simple console logging."""
    level = logging.INFO if verbosity > 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(config: str, verbosity: int = 1) -> None:
    """Entry point for CLI usage."""
    _configure_logging(verbosity)
    load_dotenv()
    results = download_from_config(Path(config))
    message = {
        "prices": [str(path) for path in results["prices"]],
        "benchmarks": [str(path) for path in results["benchmarks"]],
        "macro": [str(path) for path in results["macro"]],
    }
    LOGGER.info("Download completed:\n%s", json.dumps(message, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI usage
    import argparse

    parser = argparse.ArgumentParser(description="Download NVDA datasets based on YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to the YAML configuration file describing data sources.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity.",
    )
    args = parser.parse_args()

    verbosity_level = 0 if args.quiet else 1
    main(config=args.config, verbosity=verbosity_level)
