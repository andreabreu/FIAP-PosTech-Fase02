"""Strategy base dos preprocessors (fit/transform)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PreprocessorStrategy(ABC):
    """Interface fit/transform — dá para trocar o preprocessor sem mudar o pipeline."""

    @abstractmethod
    def fit(self, data: Any) -> PreprocessorStrategy:
        """Ajusta a strategy aos dados."""

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Aplica a transformação."""

    def fit_transform(self, data: Any) -> Any:
        """Atalho fit + transform."""
        return self.fit(data).transform(data)
