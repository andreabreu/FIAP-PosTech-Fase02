"""Factory simples para criar os recomendadores pelo nome."""

from __future__ import annotations

from typing import Any

from src.domain.interfaces import RecommenderModel
from src.models.baselines import PopularityRecommender, SVDRecommender
from src.models.mlp import MLPRecommender


class ModelFactory:
    """Cria o modelo a partir de `params.yaml` (`train.model_name`)."""

    _registry: dict[str, type[RecommenderModel]] = {
        "mlp": MLPRecommender,
        "embedding": MLPRecommender,
        "popularity": PopularityRecommender,
        "svd": SVDRecommender,
    }

    @classmethod
    def available(cls) -> list[str]:
        """Nomes registrados."""
        return sorted(cls._registry)

    @classmethod
    def create(cls, name: str, **hparams: Any) -> RecommenderModel:
        """Instancia o modelo. Lança KeyError se o nome não existir."""
        if name not in cls._registry:
            known = ", ".join(cls.available())
            raise KeyError(f"modelo desconhecido '{name}'. opções: {known}")
        model_cls = cls._registry[name]
        return model_cls(name=name, **hparams)  # type: ignore[call-arg]
