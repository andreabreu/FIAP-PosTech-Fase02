"""Encadeia strategies de preprocessor."""

from __future__ import annotations

from typing import Any

from src.features.preprocessors.base import PreprocessorStrategy
from src.features.preprocessors.passthrough import PassthroughPreprocessor


class FeaturePipeline:
    """Roda as strategies em sequência (default = passthrough)."""

    def __init__(self, strategies: list[PreprocessorStrategy] | None = None) -> None:
        self.strategies = strategies or [PassthroughPreprocessor()]

    def run(self, data: Any) -> Any:
        result = data
        for strategy in self.strategies:
            result = strategy.fit_transform(result)
        return result
