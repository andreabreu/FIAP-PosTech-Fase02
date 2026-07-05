#!/usr/bin/env python3
"""CLI: feature engineering stage for the DVC pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.features.engineering import build_features


def main() -> int:
    """Run feature engineering stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/interactions_clean.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/features"),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/feature_eng.json"),
    )
    args = parser.parse_args()
    summary = build_features(args.input, args.output_dir, args.params)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
