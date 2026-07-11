#!/usr/bin/env python3
"""CLI: train (+ evaluate) stage for DVC / local runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.pipeline import train_and_evaluate


def main() -> int:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/features/train.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/features/test.csv"),
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/recommender.pt"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/train.json"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    payload = train_and_evaluate(
        train_path=args.train,
        test_path=args.test,
        params_path=args.params,
        model_out=args.model_out,
        metrics_out=args.metrics,
        run_name=args.run_name,
        log_mlflow=not args.no_mlflow,
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
