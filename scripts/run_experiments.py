#!/usr/bin/env python3
"""Run baseline and MLP experiment sweeps logged to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from src.training.pipeline import train_and_evaluate
from src.utils import get_logger

logger = get_logger(__name__)


def _write_params(base: dict, model_name: str, overrides: dict, path: Path) -> None:
    payload = json.loads(json.dumps(base))  # deep copy via JSON
    payload.setdefault("train", {})
    payload["train"]["model_name"] = model_name
    payload["train"].update(overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    """Execute configured experiment matrix and write a summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/features/train.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/features/test.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("metrics/experiments_summary.json"),
    )
    parser.add_argument("--suite", choices=["baselines", "mlp", "all"], default="all")
    args = parser.parse_args()

    base = yaml.safe_load(args.params.read_text(encoding="utf-8")) or {}
    tmp_dir = Path("metrics/tmp_params")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, dict]] = []
    if args.suite in {"baselines", "all"}:
        jobs.append(("popularity", {}))
        jobs.append(("svd", {"embedding_dim": 32}))
    if args.suite in {"mlp", "all"}:
        jobs.append(("mlp", {"embedding_dim": 16, "max_epochs": 8, "hidden_dim": 32}))
        jobs.append(("mlp", {"embedding_dim": 32, "max_epochs": 12, "hidden_dim": 64}))

    results = []
    for idx, (model_name, overrides) in enumerate(jobs):
        param_path = tmp_dir / f"params_{idx}_{model_name}.yaml"
        _write_params(base, model_name, overrides, param_path)
        model_out = Path("models") / f"exp_{idx}_{model_name}"
        metrics_out = Path("metrics") / f"exp_{idx}_{model_name}.json"
        run_name = f"exp-{idx}-{model_name}"
        logger.info("running %s overrides=%s", run_name, overrides)
        payload = train_and_evaluate(
            train_path=args.train,
            test_path=args.test,
            params_path=param_path,
            model_out=model_out,
            metrics_out=metrics_out,
            run_name=run_name,
            log_mlflow=True,
        )
        results.append(payload)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
