#!/usr/bin/env python3
"""Validate runtime environment for FIAP Tech Challenge Fase 02."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REQUIRED_PACKAGES = (
    "pydantic",
    "pydantic_settings",
    "numpy",
    "pandas",
    "yaml",
    "dotenv",
    "torch",
    "sklearn",
    "mlflow",
)

REQUIRED_DIRS = (
    "src",
    "tests",
    "data/raw",
    "data/processed",
    "models",
    "configs",
    "scripts",
)

REQUIRED_FILES = (
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    "configs/default.yaml",
)


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def validate_packages() -> list[str]:
    """Return missing importable package names."""
    return [name for name in REQUIRED_PACKAGES if not _package_available(name)]


def validate_paths(root: Path) -> list[str]:
    """Return missing required paths relative to root."""
    missing: list[str] = []
    for relative in REQUIRED_DIRS:
        if not (root / relative).is_dir():
            missing.append(f"dir:{relative}")
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            missing.append(f"file:{relative}")
    return missing


def validate_settings() -> list[str]:
    """Return settings validation errors."""
    try:
        from src.config import get_settings

        settings = get_settings()
        if settings.embedding_dim < 1:
            return ["settings: embedding_dim must be >= 1"]
        if not settings.mlflow_tracking_uri:
            return ["settings: mlflow_tracking_uri is empty"]
    except Exception as exc:  # noqa: BLE001 - surface bootstrap errors
        return [f"settings: {exc}"]
    return []


def main() -> int:
    """Run environment checks and print a concise report."""
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    errors.extend(validate_paths(root))
    errors.extend(validate_packages())
    errors.extend(validate_settings())

    if errors:
        print("Environment validation FAILED:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("Environment validation OK")
    print(f"  root={root}")
    print("  packages=ok settings=ok paths=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
