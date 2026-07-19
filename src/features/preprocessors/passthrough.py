"""Preprocessor que só repassa os dados (útil como default)."""

from __future__ import annotations

from typing import Any

from src.features.preprocessors.base import PreprocessorStrategy


class PassthroughPreprocessor(PreprocessorStrategy):
    """Não altera o dataset."""

    def fit(self, data: Any) -> PassthroughPreprocessor:
        return self

    def transform(self, data: Any) -> Any:
        return data
