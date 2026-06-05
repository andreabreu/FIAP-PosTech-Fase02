"""Factory pattern for creating recommender model instances."""

from __future__ import annotations

from typing import Any

from src.domain.interfaces import RecommenderModel
from src.models.base import PlaceholderRecommender


class ModelFactory:
    """Create recommender models by registered name."""

    _registry: dict[str, type[RecommenderModel]] = {
        "mlp": PlaceholderRecommender,
        "embedding": PlaceholderRecommender,
    }

    @classmethod
    def available(cls) -> list[str]:
        """List registered model names.

        Returns:
            list[str]: Available model keys.
        """
        return sorted(cls._registry)

    @classmethod
    def create(cls, name: str, **hparams: Any) -> RecommenderModel:
        """Instantiate a model from the registry.

        Args:
            name: Registered model name.
            **hparams: Model hyperparameters.

        Returns:
            RecommenderModel: New model instance.

        Raises:
            KeyError: If the model name is unknown.
        """
        if name not in cls._registry:
            known = ", ".join(cls.available())
            raise KeyError(f"unknown model '{name}'. known: {known}")
        model_cls = cls._registry[name]
        return model_cls(name=name, **hparams)  # type: ignore[call-arg]
