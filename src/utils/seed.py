"""Helpers de seed."""

from __future__ import annotations

import os
import random
from typing import Any


def set_seed(seed: int = 42) -> int:
    """Fixa seed do random e PYTHONHASHSEED."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def seeded_sample(items: list[Any], k: int, seed: int = 42) -> list[Any]:
    """Sample com RNG dedicado (não mexe no random global)."""
    rng = random.Random(seed)
    return rng.sample(items, k=min(k, len(items)))
