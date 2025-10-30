#!/usr/bin/env python
"""Command-line wrapper for data ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import main as download_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets required for the NVIDIA stock prediction project.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run with reduced logging verbosity.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    verbosity = 0 if args.quiet else 1
    download_main(config=args.config, verbosity=verbosity)


if __name__ == "__main__":
    run()
