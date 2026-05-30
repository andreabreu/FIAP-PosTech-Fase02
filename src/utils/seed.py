"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from typing import Any


def set_seed(seed: int = 42) -> int:
    """Fix Python and environment seeds for reproducibility.

    Args:
        seed: Random seed value.

    Returns:
        int: The seed that was applied.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def seeded_sample(items: list[Any], k: int, seed: int = 42) -> list[Any]:
    """Sample items with a dedicated RNG instance.

    Args:
        items: Population to sample from.
        k: Number of items to draw.
        seed: RNG seed.

    Returns:
        list[Any]: Sampled items.
    """
    rng = random.Random(seed)
    return rng.sample(items, k=min(k, len(items)))
