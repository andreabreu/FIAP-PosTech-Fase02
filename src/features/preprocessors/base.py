"""Strategy interface for feature preprocessors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PreprocessorStrategy(ABC):
    """Strategy contract for transforming interaction datasets."""

    @abstractmethod
    def fit(self, data: Any) -> PreprocessorStrategy:
        """Learn transformation parameters from data.

        Args:
            data: Raw or intermediate interaction data.

        Returns:
            PreprocessorStrategy: Fitted strategy.
        """

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Apply the learned transformation.

        Args:
            data: Dataset to transform.

        Returns:
            Any: Transformed dataset.
        """

    def fit_transform(self, data: Any) -> Any:
        """Fit on data and transform in one step.

        Args:
            data: Dataset to fit and transform.

        Returns:
            Any: Transformed dataset.
        """
        return self.fit(data).transform(data)
