# Docker — build e Compose

## Serviços

| Serviço | Porta | Função |
|---------|-------|--------|
| `mlflow` | `5000` | Tracking server MLflow |
| `train` | — | Job de treino (`src.training.entrypoint`) ligado ao MLflow |

## Caminho recomendado (treino + MLflow)

```bash
# 1) sobe MLflow + build/run do train
docker compose up --build

# UI
open http://127.0.0.1:5000
```

Fluxo esperado:
1. `mlflow` fica healthy em `:5000`
2. `train` inicia com `MLFLOW_TRACKING_URI=http://mlflow:5000`
3. entrypoint executa `python -m src.training.entrypoint`
4. artefatos locais montados em `./models` e `./metrics`

## Build da imagem

```bash
docker build -t fiap-postech-fase02:local .
```

Multi-stage:
- `builder` em `/build` — `uv sync --frozen`
- `runtime` em `/app` — usuário non-root `appuser` (uid 10001)

## Comandos úteis

```bash
# só tracking
docker compose up mlflow

# treino one-shot (reusa a stack)
docker compose run --rm train

# validar compose
docker compose config
```

## Smoke local (sem container)

```bash
uv run python -m src.training.entrypoint
uv run pytest tests/test_docker_entrypoint_smoke.py -q
```
