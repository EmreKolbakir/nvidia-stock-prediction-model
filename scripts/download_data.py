#!/usr/bin/env python
"""Command-line wrapper for data ingestion."""

from __future__ import annotations

import argparse

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
