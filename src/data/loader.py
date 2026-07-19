"""Leitura simples do CSV de interações."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_interactions(path: str | Path) -> list[dict[str, Any]]:
    """Lê o CSV; se o arquivo não existir, devolve lista vazia."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    logger.info("loaded %s rows from %s", len(rows), file_path)
    return rows


class CsvInteractionLoader:
    """Wrapper usado nos testes / CLI curtos."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        return load_interactions(self.path)
