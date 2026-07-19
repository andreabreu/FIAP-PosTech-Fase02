"""Inference helpers for the serving API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.models.mlp import MLPRecommender

DEFAULT_MODEL = Path("artifacts/serving/recommender.pt")
DEFAULT_MAPS = Path("artifacts/serving/id_maps.json")


@lru_cache(maxsize=1)
def load_bundle(
    model_path: str = str(DEFAULT_MODEL),
    maps_path: str = str(DEFAULT_MAPS),
) -> tuple[MLPRecommender, dict[str, int], dict[int, str]]:
    """Carrega modelo + mapas de id (cache por processo)."""
    maps = json.loads(Path(maps_path).read_text(encoding="utf-8"))
    user_map: dict[str, int] = {str(k): int(v) for k, v in maps["user_map"].items()}
    item_map: dict[str, int] = {str(k): int(v) for k, v in maps["item_map"].items()}
    reverse_items = {idx: item_id for item_id, idx in item_map.items()}
    model = MLPRecommender.load(Path(model_path))
    return model, user_map, reverse_items


def recommend_for_user(user_id: str, k: int = 10) -> list[dict[str, float | str]]:
    """Top-k itens para um user_id externo (ex.: U0001)."""
    if k < 1 or k > 100:
        raise ValueError("k must be between 1 and 100")
    model, user_map, reverse_items = load_bundle()
    if user_id not in user_map:
        raise KeyError(user_id)
    user_idx = user_map[user_id]
    candidates = np.arange(model.n_items, dtype=np.int64)
    scores = model.score_user_items(user_idx, candidates)
    top = np.argsort(-scores)[:k]
    return [
        {"item_id": reverse_items[int(i)], "score": float(scores[int(i)])}
        for i in top
        if int(i) in reverse_items
    ]
