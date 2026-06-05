"""Feature pipeline orchestration (no model dependencies)."""

from __future__ import annotations

from typing import Any

from src.features.preprocessors.base import PreprocessorStrategy
from src.features.preprocessors.passthrough import PassthroughPreprocessor


class FeaturePipeline:
    """Runs one or more preprocessor strategies in sequence."""

    def __init__(self, strategies: list[PreprocessorStrategy] | None = None) -> None:
        self.strategies = strategies or [PassthroughPreprocessor()]

    def run(self, data: Any) -> Any:
        """Apply all strategies sequentially.

        Args:
            data: Input dataset.

        Returns:
            Any: Transformed dataset.
        """
        result = data
        for strategy in self.strategies:
            result = strategy.fit_transform(result)
        return result
