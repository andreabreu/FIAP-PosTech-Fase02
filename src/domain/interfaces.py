"""Interfaces usadas pelos modelos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RecommenderModel(ABC):
    """Contrato mínimo: fit + predict."""

    @abstractmethod
    def fit(self, interactions: Any) -> RecommenderModel:
        """Treina com interações user–item."""

    @abstractmethod
    def predict(self, user_ids: Any, item_ids: Any) -> Any:
        """Retorna scores para pares user–item."""
