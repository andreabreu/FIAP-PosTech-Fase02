# Setup — ambiente limpo (uv)

## Pré-requisitos

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recomendado) **ou** Poetry

## Instalação com uv (recomendado)

```bash
cd FIAP-PosTech-Fase02
uv sync --all-groups
cp .env.example .env
uv run python scripts/validate_env.py
```

Isso usa `pyproject.toml` + `uv.lock` para reinstalar as mesmas versões.

## Instalação com Poetry (alternativa)

```bash
poetry install --with dev
cp .env.example .env
poetry run python scripts/validate_env.py
```

> Se preferir Poetry, gere `poetry.lock` localmente; o repositório oficializa o lock via **uv**.

## Validação rápida

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/validate_env.py
```

## Variáveis de ambiente

Copie `.env.example` → `.env` e ajuste URIs locais.  
**Nunca** faça commit do `.env` real.

## Docker

Ver [DOCKER.md](DOCKER.md) para `docker compose up --build` (treino + MLflow).

## DVC

```bash
uv run dvc pull
uv run dvc repro
```

Detalhes em [DVC.md](DVC.md).

## Treino / MLflow

```bash
# Treino MLP + métricas (sem MLflow)
uv run python scripts/run_train.py --no-mlflow

# Experimentos + Registry
uv run python scripts/run_experiments.py --suite all
uv run python scripts/promote_model.py --to both
```

Tracking local default: `sqlite:///mlflow.db` (ver `.env.example`).  
No Docker Compose o serviço `train` usa `http://mlflow:5000`.
