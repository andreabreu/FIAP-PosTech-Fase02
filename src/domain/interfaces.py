"""Core abstractions used across the recommender pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataLoader(Protocol):
    """Loads raw interaction data from a configured source."""

    def load(self) -> Any:
        """Load raw data from the underlying store."""


class RecommenderModel(ABC):
    """Contract for trainable recommender models."""

    @abstractmethod
    def fit(self, interactions: Any) -> RecommenderModel:
        """Fit the model using user-item interactions.

        Args:
            interactions: Dataset of user-item interactions.

        Returns:
            RecommenderModel: Fitted model instance.
        """

    @abstractmethod
    def predict(self, user_ids: Any, item_ids: Any) -> Any:
        """Score user-item pairs.

        Args:
            user_ids: User identifiers.
            item_ids: Item identifiers.

        Returns:
            Any: Predicted scores.
        """
