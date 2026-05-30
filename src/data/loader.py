"""High-level data loader facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.readers import InteractionReader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CsvInteractionLoader:
    """Load interactions from a CSV path using a dedicated reader."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader = InteractionReader(self.path)

    def load(self) -> list[dict[str, Any]]:
        """Load interaction rows from disk.

        Returns:
            list[dict[str, Any]]: Raw interaction dictionaries.
        """
        rows = self._reader.read()
        logger.info("loaded %s rows from %s", len(rows), self.path)
        return rows
