"""Low-level readers for interaction sources."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class InteractionReader:
    """Read interaction files without applying business transforms."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        """Read CSV rows if the file exists, otherwise return empty.

        Returns:
            list[dict[str, Any]]: Parsed rows.
        """
        if not self.path.exists():
            return []
        with self.path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
