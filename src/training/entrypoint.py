"""Container training entrypoint (placeholder until F4 model training)."""

from __future__ import annotations

import os
from typing import Any

from src.config import get_settings
from src.models import ModelFactory
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def run_training() -> dict[str, Any]:
    """Execute a minimal training stub wired for Docker/MLflow.

    Returns:
        dict[str, Any]: Summary payload for observability.
    """
    settings = get_settings()
    set_seed(settings.random_seed)
    settings.ensure_directories()

    model = ModelFactory.create(
        settings.model_name,
        embedding_dim=settings.embedding_dim,
    )
    model.fit([])
    scores = model.predict(["smoke-user"], ["smoke-item"])

    summary = {
        "project": settings.project_name,
        "model": settings.model_name,
        "mlflow_tracking_uri": os.getenv(
            "MLFLOW_TRACKING_URI",
            settings.mlflow_tracking_uri,
        ),
        "scores": scores,
        "status": "ok",
    }
    logger.info("training stub finished: %s", summary)
    return summary


def main() -> int:
    """CLI entrypoint used by Docker CMD."""
    run_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
