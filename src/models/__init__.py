"""Modelos do challenge (MLP + baselines)."""

from src.models.baselines import PopularityRecommender, SVDRecommender
from src.models.factory import ModelFactory
from src.models.mlp import MLPRecommender

__all__ = [
    "ModelFactory",
    "MLPRecommender",
    "PopularityRecommender",
    "SVDRecommender",
]
