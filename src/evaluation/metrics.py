"""Ranking metrics for top-K recommendation evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Precision@K for one user."""
    if k <= 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Recall@K for one user."""
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Hit-Rate@K (1 if any relevant item in top-K)."""
    if not relevant or k <= 0:
        return 0.0
    top = set(recommended[:k])
    return 1.0 if top & relevant else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain@K."""
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


ScoreFn = Callable[[int, np.ndarray], np.ndarray]


def evaluate_ranking(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_fn: ScoreFn,
    k: int = 10,
    n_items: int | None = None,
) -> dict[str, float]:
    """Evaluate ranking metrics averaged over test users."""
    if n_items is None:
        n_items = int(max(train_df["item_idx"].max(), test_df["item_idx"].max())) + 1
    all_items = np.arange(n_items, dtype=np.int64)
    train_seen = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        if not train_df.empty
        else {}
    )
    test_rel = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    metrics = {"precision": [], "recall": [], "hit_rate": [], "ndcg": []}
    for user, relevant in test_rel.items():
        seen = train_seen.get(int(user), set())
        candidates = np.array([i for i in all_items if i not in seen], dtype=np.int64)
        if len(candidates) == 0:
            continue
        scores = score_fn(int(user), candidates)
        order = np.argsort(-scores)
        recommended = candidates[order].tolist()
        metrics["precision"].append(precision_at_k(recommended, relevant, k))
        metrics["recall"].append(recall_at_k(recommended, relevant, k))
        metrics["hit_rate"].append(hit_rate_at_k(recommended, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recommended, relevant, k))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        f"precision_at_{k}": round(_mean(metrics["precision"]), 6),
        f"recall_at_{k}": round(_mean(metrics["recall"]), 6),
        f"hit_rate_at_{k}": round(_mean(metrics["hit_rate"]), 6),
        f"ndcg_at_{k}": round(_mean(metrics["ndcg"]), 6),
        "n_eval_users": float(len(metrics["precision"])),
    }
