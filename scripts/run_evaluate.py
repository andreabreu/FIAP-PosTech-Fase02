#!/usr/bin/env python3
"""CLI: evaluate stage placeholder for DVC (real metrics land in F4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def _popularity_hit_rate(train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> float:
    """Simple popularity@K hit-rate baseline on held-out interactions."""
    if test.empty or train.empty:
        return 0.0
    top_items = (
        train.groupby("item_idx").size().sort_values(ascending=False).head(k).index
    )
    top_set = set(top_items.tolist())
    hits = 0
    total = 0
    for _, group in test.groupby("user_idx"):
        total += 1
        if any(item in top_set for item in group["item_idx"].tolist()):
            hits += 1
    return hits / total if total else 0.0


def main() -> int:
    """Compute lightweight ranking metrics for the DVC evaluate stage."""
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
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/recommender_stub.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/evaluate.json"),
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    model_meta = json.loads(args.model.read_text(encoding="utf-8"))

    hit_rate = _popularity_hit_rate(train_df, test_df, k=args.k)
    coverage = (
        test_df["item_idx"].nunique() / max(train_df["item_idx"].nunique(), 1)
        if not train_df.empty
        else 0.0
    )
    metrics = {
        "model_name": model_meta.get("model_name", "unknown"),
        "n_test_rows": int(len(test_df)),
        "n_test_users": int(test_df["user_idx"].nunique()) if not test_df.empty else 0,
        f"popularity_hit_rate_at_{args.k}": round(float(hit_rate), 6),
        "test_item_coverage": round(float(coverage), 6),
        "status": "ok",
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("evaluate stage finished: %s", metrics)
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
