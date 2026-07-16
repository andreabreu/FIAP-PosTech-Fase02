"""Container / CLI training entrypoint wired to the real pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import get_settings
from src.training.pipeline import train_and_evaluate
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def run_training() -> dict[str, Any]:
    """Execute train+eval using env settings and default feature paths."""
    settings = get_settings()
    set_seed(settings.random_seed)
    settings.ensure_directories()

    train_path = Path(settings.processed_data_dir) / "features" / "train.csv"
    test_path = Path(settings.processed_data_dir) / "features" / "test.csv"
    params_path = Path("params.yaml")
    model_out = Path(settings.models_dir) / "recommender.pt"
    metrics_out = Path(settings.metrics_dir) / "train.json"

    if not train_path.exists() or not test_path.exists():
        logger.warning("feature tables missing; run `dvc repro` first")
        return {"status": "skipped", "reason": "missing_features"}

    return train_and_evaluate(
        train_path=train_path,
        test_path=test_path,
        params_path=params_path,
        model_out=model_out,
        metrics_out=metrics_out,
        run_name="docker-train",
        log_mlflow=True,
    )


def main() -> int:
    """CLI entrypoint used by Docker CMD."""
    summary = run_training()
    logger.info("entrypoint finished: %s", summary)
    return 0 if summary.get("status") in {"fitted", "ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
