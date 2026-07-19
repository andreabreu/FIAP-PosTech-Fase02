"""Treino + avaliação usados pelo DVC / CLI / Docker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from src.config import get_settings
from src.evaluation.metrics import evaluate_ranking
from src.models.factory import ModelFactory
from src.models.mlp import MLPRecommender
from src.training.loop import train_mlp
from src.training.mlflow_tracking import (
    configure_mlflow,
    log_artifact_path,
    log_params_metrics,
    start_run,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def load_params(path: Path) -> dict[str, Any]:
    """Lê params.yaml."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    params_path: Path,
    model_out: Path,
    metrics_out: Path,
    run_name: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Treina o modelo do params.yaml, avalia ranking e opcionalmente loga no MLflow."""
    settings = get_settings()
    params = load_params(params_path)
    train_cfg = params.get("train", {})
    eval_cfg = params.get("evaluate", {})
    seed = int(train_cfg.get("seed", params.get("seed", settings.random_seed)))
    set_seed(seed)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model_name = str(train_cfg.get("model_name", settings.model_name))
    k = int(eval_cfg.get("k", 10))
    embedding_dim = int(train_cfg.get("embedding_dim", settings.embedding_dim))

    train_meta: dict[str, Any]
    score_fn: Any

    # MLP usa loop próprio (early stopping); baselines passam pela Factory
    if model_name in {"mlp", "embedding"}:
        model, result = train_mlp(
            train_df,
            embedding_dim=embedding_dim,
            hidden_dim=int(train_cfg.get("hidden_dim", 64)),
            batch_size=int(train_cfg.get("batch_size", settings.batch_size)),
            max_epochs=int(train_cfg.get("max_epochs", settings.max_epochs)),
            patience=int(
                train_cfg.get(
                    "early_stopping_patience",
                    settings.early_stopping_patience,
                )
            ),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            neg_per_pos=int(train_cfg.get("neg_per_pos", 1)),
            seed=seed,
        )
        if model_out.suffix != ".pt":
            model_out = model_out.with_suffix(".pt")
        model.save(model_out)
        score_fn = model.score_user_items
        train_meta = {
            "model_name": model_name,
            "best_epoch": result.best_epoch,
            "best_val_loss": round(result.best_val_loss, 6),
            "n_train_rows": int(len(train_df)),
            "n_users": model.n_users,
            "n_items": model.n_items,
            "model_path": str(model_out),
            "status": "fitted",
        }
    elif model_name in {"popularity", "svd"}:
        factory_kwargs: dict[str, Any] = {}
        if model_name == "svd":
            factory_kwargs["n_components"] = int(
                train_cfg.get("embedding_dim", embedding_dim)
            )
        model = ModelFactory.create(model_name, **factory_kwargs)
        model.fit(train_df)
        score_fn = model.score_user_items  # type: ignore[attr-defined]
        meta = {
            "model_name": model_name,
            "n_train_rows": int(len(train_df)),
            "n_items": getattr(model, "n_items", 0),
            "status": "fitted",
        }
        if hasattr(model, "n_users"):
            meta["n_users"] = model.n_users
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        train_meta = {**meta, "model_path": str(model_out)}
    else:
        raise KeyError(f"modelo não suportado: {model_name}")
    ranking = evaluate_ranking(train_df, test_df, score_fn, k=k)
    payload = {**train_meta, **ranking, "k": k}

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("train+eval finished: %s", payload)

    if log_mlflow:
        configure_mlflow(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)
        with start_run(run_name or f"{model_name}-train", tags={"model": model_name}):
            log_params_metrics(
                {
                    "model_name": model_name,
                    "embedding_dim": embedding_dim,
                    "seed": seed,
                    "k": k,
                    **{f"train.{key}": value for key, value in train_cfg.items()},
                },
                {key: float(value) for key, value in ranking.items()},
            )
            if model_name in {"mlp", "embedding"} and "best_val_loss" in train_meta:
                mlflow.log_metric("best_val_loss", float(train_meta["best_val_loss"]))
                mlflow.log_metric("best_epoch", float(train_meta["best_epoch"]))
                net = MLPRecommender.load(Path(train_meta["model_path"])).net
                mlflow.pytorch.log_model(
                    net,
                    name="model",
                    serialization_format="pickle",
                )
            log_artifact_path(metrics_out)
            payload["mlflow_run_id"] = mlflow.active_run().info.run_id  # type: ignore[union-attr]

    return payload
