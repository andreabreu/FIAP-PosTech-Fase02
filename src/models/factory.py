"""Factory pattern for creating recommender model instances."""

from __future__ import annotations

from typing import Any

from src.domain.interfaces import RecommenderModel


class _PlaceholderModel(RecommenderModel):
    """Minimal model used until PyTorch trainers are implemented."""

    def __init__(self, name: str, **hparams: Any) -> None:
        self.name = name
        self.hparams = hparams
        self.is_fitted = False

    def fit(self, interactions: Any) -> _PlaceholderModel:
        """Mark the placeholder as fitted.

        Args:
            interactions: Unused interactions payload.

        Returns:
            _PlaceholderModel: Fitted placeholder.
        """
        _ = interactions
        self.is_fitted = True
        return self

    def predict(self, user_ids: Any, item_ids: Any) -> list[float]:
        """Return zeros as placeholder scores.

        Args:
            user_ids: User identifiers.
            item_ids: Item identifiers.

        Returns:
            list[float]: Placeholder scores.
        """
        size = max(len(user_ids), len(item_ids), 0)
        return [0.0] * size


class ModelFactory:
    """Create recommender models by registered name."""

    _registry: dict[str, type[RecommenderModel]] = {
        "mlp": _PlaceholderModel,
        "embedding": _PlaceholderModel,
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
