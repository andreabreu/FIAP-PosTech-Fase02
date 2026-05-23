"""No-op preprocessor used as the default strategy."""

from __future__ import annotations

from typing import Any

from src.features.preprocessors.base import PreprocessorStrategy


class PassthroughPreprocessor(PreprocessorStrategy):
    """Returns the input dataset unchanged."""

    def fit(self, data: Any) -> PassthroughPreprocessor:
        """Fit is a no-op for passthrough.

        Args:
            data: Unused input data.

        Returns:
            PassthroughPreprocessor: Self.
        """
        return self

    def transform(self, data: Any) -> Any:
        """Return data without modification.

        Args:
            data: Input dataset.

        Returns:
            Any: The same dataset instance.
        """
        return data
