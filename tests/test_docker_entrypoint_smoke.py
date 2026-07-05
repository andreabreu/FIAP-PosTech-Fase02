"""Smoke checks for Docker train entrypoint wiring."""

from __future__ import annotations

from pathlib import Path

import yaml
from src.training.entrypoint import run_training

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_train_entrypoint() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "src.training.entrypoint" in dockerfile
    assert "USER appuser" in dockerfile
    assert "AS builder" in dockerfile
    assert "AS runtime" in dockerfile


def test_compose_train_service_points_to_mlflow() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    train = compose["services"]["train"]
    assert train["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert train["command"] == ["python", "-m", "src.training.entrypoint"]
    assert "./models:/app/models" in train["volumes"]
    assert "./metrics:/app/metrics" in train["volumes"]


def test_train_entrypoint_runs_locally() -> None:
    summary = run_training()
    assert summary["status"] == "ok"
    assert summary["model"] == "mlp"
    assert summary["scores"] == [0.0]
