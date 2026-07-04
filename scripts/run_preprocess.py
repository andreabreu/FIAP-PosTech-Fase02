#!/usr/bin/env python3
"""CLI: preprocess raw interactions for the DVC pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.preprocess import preprocess_interactions


def main() -> int:
    """Run preprocess stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/interactions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/interactions_clean.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/preprocess.json"),
    )
    args = parser.parse_args()
    summary = preprocess_interactions(args.input, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
