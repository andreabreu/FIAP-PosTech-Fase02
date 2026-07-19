# Setup — ambiente limpo (uv)

Guia completo (setup + treino + evaluate + Docker + DVC): **[GUIDE.md](GUIDE.md)**.

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

## Instalação com Poetry (alternativa)

```bash
poetry install --with dev
cp .env.example .env
poetry run python scripts/validate_env.py
```

> O repositório oficializa o lock via **uv** (`uv.lock`).

## Validação rápida

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/validate_env.py
```

## Variáveis de ambiente

Copie `.env.example` → `.env`. **Nunca** faça commit do `.env` real.  
Tracking local default: `sqlite:///mlflow.db`.

## Próximos passos

```bash
uv run dvc pull && uv run dvc repro
uv run python scripts/run_experiments.py --suite all
uv run python scripts/promote_model.py --to both
```

Ver também: [DOCKER.md](DOCKER.md) · [DVC.md](DVC.md) · [INSTALL_CHECKLIST.md](INSTALL_CHECKLIST.md).
