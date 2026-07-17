"""MLflow Model Registry helpers (Staging → Production)."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import get_logger

logger = get_logger(__name__)

REGISTERED_MODEL_NAME = "fiap-fase02-recommender"


def register_run_model(
    run_id: str,
    artifact_path: str = "model",
    model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    """Register a logged model from a run into the Model Registry.

    Args:
        run_id: Source MLflow run id.
        artifact_path: Artifact subpath used when logging the model.
        model_name: Registered model name.

    Returns:
        str: Created model version string.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)
    logger.info("registered %s version=%s", model_name, result.version)
    return str(result.version)


def transition_stage(
    version: str,
    stage: str,
    model_name: str = REGISTERED_MODEL_NAME,
    archive_existing: bool = True,
) -> None:
    """Transition a model version to ``Staging`` or ``Production``.

    Args:
        version: Model version.
        stage: Target stage name.
        model_name: Registered model name.
        archive_existing: Archive other versions in the target stage.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info("model %s v%s -> %s", model_name, version, stage)


def promote_best_run_to_staging(
    experiment_name: str,
    metric_key: str = "ndcg_at_10",
    model_name: str = REGISTERED_MODEL_NAME,
) -> dict[str, Any]:
    """Pick the best finished run by metric and promote to Staging.

    Returns:
        dict[str, Any]: Summary with run_id, metric, version.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"experiment not found: {experiment_name}")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.model = 'mlp'",
        order_by=[f"metrics.{metric_key} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("no finished runs to promote")
    best = runs[0]
    run_id = best.info.run_id
    score = float(best.data.metrics.get(metric_key, 0.0))
    version = register_run_model(run_id, model_name=model_name)
    transition_stage(version, "Staging", model_name=model_name)
    return {
        "run_id": run_id,
        "metric_key": metric_key,
        "metric_value": score,
        "version": version,
        "stage": "Staging",
    }


def promote_staging_to_production(
    model_name: str = REGISTERED_MODEL_NAME,
) -> dict[str, Any]:
    """Promote the current Staging version to Production."""
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        raise RuntimeError(f"no Staging version for {model_name}")
    version = versions[0].version
    transition_stage(version, "Production", model_name=model_name)
    return {"version": version, "stage": "Production", "model_name": model_name}
