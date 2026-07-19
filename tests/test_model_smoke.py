"""Smoke tests for metrics, MLP training and factory wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.evaluation.metrics import (
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models import ModelFactory
from src.models.baselines import PopularityRecommender
from src.training.loop import train_mlp


def test_ranking_metric_helpers() -> None:
    recommended = [1, 2, 3, 4]
    relevant = {2, 9}
    assert precision_at_k(recommended, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(recommended, relevant, 3) == pytest.approx(0.5)
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0
    assert ndcg_at_k(recommended, relevant, 3) > 0.0


def test_factory_registers_real_models() -> None:
    assert "mlp" in ModelFactory.available()
    assert "popularity" in ModelFactory.available()
    assert "svd" in ModelFactory.available()
    model = ModelFactory.create("popularity")
    assert model is not None


def test_mlp_smoke_train_and_rank(tmp_path: Path) -> None:
    rows = []
    for user in range(20):
        for item in range(5):
            rows.append(
                {
                    "user_idx": user,
                    "item_idx": (user + item) % 30,
                    "score": 1.0,
                }
            )
    train_df = pd.DataFrame(rows)
    test_df = train_df.sample(frac=0.2, random_state=0)
    model, result = train_mlp(
        train_df,
        embedding_dim=8,
        hidden_dim=16,
        batch_size=32,
        max_epochs=2,
        patience=2,
        seed=0,
    )
    assert model.is_fitted
    assert result.best_epoch >= 1
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = type(model).load(path)
    metrics = evaluate_ranking(train_df, test_df, loaded.score_user_items, k=5)
    assert "ndcg_at_5" in metrics
    assert metrics["n_eval_users"] > 0


def test_popularity_baseline_scores() -> None:
    frame = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 1, 1],
            "score": [1, 1, 1, 1],
        }
    )
    model = PopularityRecommender().fit(frame)
    scores = model.score_user_items(0, frame["item_idx"].unique())
    assert scores[list(frame["item_idx"].unique()).index(1)] > scores[
        list(frame["item_idx"].unique()).index(0)
    ]
