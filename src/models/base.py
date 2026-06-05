"""Concrete model stubs living outside the factory module."""

from __future__ import annotations

from typing import Any

from src.domain.interfaces import RecommenderModel


class PlaceholderRecommender(RecommenderModel):
    """Temporary recommender until PyTorch models land."""

    def __init__(self, name: str, **hparams: Any) -> None:
        self.name = name
        self.hparams = hparams
        self.is_fitted = False

    def fit(self, interactions: Any) -> PlaceholderRecommender:
        """Mark model as fitted.

        Args:
            interactions: Training interactions (unused).

        Returns:
            PlaceholderRecommender: Fitted instance.
        """
        _ = interactions
        self.is_fitted = True
        return self

    def predict(self, user_ids: Any, item_ids: Any) -> list[float]:
        """Return zero scores as placeholder.

        Args:
            user_ids: User ids.
            item_ids: Item ids.

        Returns:
            list[float]: Placeholder scores.
        """
        size = max(len(user_ids), len(item_ids), 0)
        return [0.0] * size
