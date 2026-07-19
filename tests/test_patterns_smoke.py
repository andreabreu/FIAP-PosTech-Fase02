"""Smoke tests dos padrões Factory (e Strategy legado)."""

from __future__ import annotations

import pytest
from src.features.preprocessors import PassthroughPreprocessor
from src.models import ModelFactory


def test_passthrough_strategy_fit_transform() -> None:
    data = [{"user_id": "u1", "item_id": "i1"}]
    preprocessor = PassthroughPreprocessor()
    assert preprocessor.fit_transform(data) == data


def test_model_factory_creates_known_models() -> None:
    model = ModelFactory.create("mlp", embedding_dim=8)
    assert model.name == "mlp"
    model.fit([])
    assert model.predict(["u1"], ["i1"]) == [0.0]


def test_model_factory_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        ModelFactory.create("unknown-model")


def test_available_models_sorted() -> None:
    available = ModelFactory.available()
    assert available == sorted(available)
    assert set(available) == {"embedding", "mlp", "popularity", "svd"}
