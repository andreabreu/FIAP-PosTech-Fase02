"""MLflow helpers for experiment tracking."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import get_logger

logger = get_logger(__name__)


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Configure tracking URI and experiment."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    name = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "recommender-fase02",
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    logger.info("mlflow tracking_uri=%s experiment=%s", uri, name)
    return name


@contextmanager
def start_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager around ``mlflow.start_run``."""
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params_metrics(
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log flat params and metrics to the active run."""
    clean_params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(clean_params)
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_artifact_path(path: Path) -> None:
    """Log a local file or directory as an artifact."""
    if path.is_dir():
        mlflow.log_artifacts(str(path))
    elif path.exists():
        mlflow.log_artifact(str(path))


def get_client() -> MlflowClient:
    """Build an MLflow client using the active tracking URI."""
    return MlflowClient()
