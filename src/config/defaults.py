"""Default configuration values (stubs)."""

from __future__ import annotations

from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "fiap-postech-fase02",
    "random_seed": 42,
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
    },
    "model": {
        "name": "mlp",
        "embedding_dim": 32,
    },
}


def get_default_config() -> dict[str, Any]:
    """Return a shallow copy of the default configuration.

    Returns:
        dict[str, Any]: Default project configuration.
    """
    return dict(DEFAULT_CONFIG)
