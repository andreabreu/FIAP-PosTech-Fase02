"""Tests for scripts/validate_env.py."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_env import (
    REQUIRED_FILES,
    validate_packages,
    validate_paths,
    validate_settings,
)


def test_required_paths_exist_in_repo() -> None:
    root = Path(__file__).resolve().parents[1]
    assert validate_paths(root) == []


def test_required_files_constant_covers_lock_and_env_example() -> None:
    assert "uv.lock" in REQUIRED_FILES
    assert ".env.example" in REQUIRED_FILES


def test_settings_validate_with_defaults() -> None:
    assert validate_settings() == []


def test_validate_packages_reports_only_missing(monkeypatch) -> None:
    def fake_find(name: str):  # noqa: ANN001
        return object() if name != "torch" else None

    monkeypatch.setattr(
        "scripts.validate_env.importlib.util.find_spec",
        fake_find,
    )
    missing = validate_packages()
    assert "torch" in missing
