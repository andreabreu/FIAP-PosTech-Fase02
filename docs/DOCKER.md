# Docker — build e Compose

## Serviços

| Serviço | Porta | Função |
|---------|-------|--------|
| `mlflow` | `5000` | Tracking server MLflow |
| `train` | — | Job de treino (entrypoint stub) que aponta para o MLflow |

## Build da imagem

```bash
docker build -t fiap-postech-fase02:local .
```

A imagem é **multi-stage**:
1. `builder` — resolve deps com `uv sync --frozen`
2. `runtime` — copia o `.venv`, roda como usuário non-root `appuser`

## Subir stack

```bash
docker compose up --build
```

- UI MLflow: http://127.0.0.1:5000
- O serviço `train` sobe após o healthcheck do MLflow e executa `python -m src.training.entrypoint`

## Só MLflow (sem treino)

```bash
docker compose up mlflow
```

## Treino one-shot

```bash
docker compose run --rm train
```
