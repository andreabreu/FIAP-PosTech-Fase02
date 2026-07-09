#!/usr/bin/env python3
"""CLI: train stage placeholder wired for DVC (real training lands in F4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from src.config import get_settings
from src.models import ModelFactory
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def main() -> int:
    """Fit a stub model on engineered features and persist a marker artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/features/train.csv"),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/recommender_stub.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/train.json"),
    )
    args = parser.parse_args()

    settings = get_settings()
    with args.params.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    train_params = params.get("train", {})
    seed = int(train_params.get("seed", params.get("seed", settings.random_seed)))
    set_seed(seed)

    train_df = pd.read_csv(args.train)
    model_name = str(train_params.get("model_name", settings.model_name))
    model = ModelFactory.create(
        model_name,
        embedding_dim=int(train_params.get("embedding_dim", settings.embedding_dim)),
    )
    rows = train_df.to_dict(orient="records")
    model.fit(rows)

    artifact = {
        "model_name": model_name,
        "n_train_rows": int(len(train_df)),
        "n_users": int(train_df["user_idx"].nunique()) if "user_idx" in train_df else 0,
        "n_items": int(train_df["item_idx"].nunique()) if "item_idx" in train_df else 0,
        "status": "stub_fitted",
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("train stage finished: %s", artifact)
    print(json.dumps(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
