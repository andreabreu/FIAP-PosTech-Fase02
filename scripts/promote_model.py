#!/usr/bin/env python3
"""Promote best MLflow run to Staging then Production."""

from __future__ import annotations

import argparse
import json

from src.config import get_settings
from src.training.mlflow_tracking import configure_mlflow
from src.training.registry import (
    promote_best_run_to_staging,
    promote_staging_to_production,
)


def main() -> int:
    """CLI for registry promotion workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument(
        "--to",
        choices=["staging", "production", "both"],
        default="both",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_mlflow(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    summary = {}
    if args.to in {"staging", "both"}:
        summary["staging"] = promote_best_run_to_staging(
            settings.mlflow_experiment_name,
            metric_key=args.metric,
        )
    if args.to in {"production", "both"}:
        summary["production"] = promote_staging_to_production()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
