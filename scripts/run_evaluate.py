#!/usr/bin/env python3
"""CLI: evaluate ranking metrics for a saved model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from src.evaluation.metrics import evaluate_ranking
from src.models.baselines import PopularityRecommender, SVDRecommender
from src.models.mlp import MLPRecommender
from src.utils import get_logger

logger = get_logger(__name__)


def _load_scorer(model_path: Path, train_df: pd.DataFrame):
    if model_path.suffix == ".pt":
        model = MLPRecommender.load(model_path)
        return model.score_user_items, model.name
    meta = json.loads(model_path.read_text(encoding="utf-8"))
    name = meta.get("model_name", "popularity")
    if name == "svd":
        model = SVDRecommender().fit(train_df)
        return model.score_user_items, name
    model = PopularityRecommender().fit(train_df)
    return model.score_user_items, name


def main() -> int:
    """Compute Precision/Recall/Hit/NDCG at K."""
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
        default=Path("models/recommender.pt"),
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
    score_fn, model_name = _load_scorer(args.model, train_df)
    ranking = evaluate_ranking(train_df, test_df, score_fn, k=args.k)
    payload = {"model_name": model_name, "status": "ok", **ranking}
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("evaluate finished: %s", payload)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
