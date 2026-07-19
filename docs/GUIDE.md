# Guia final — Setup · Treino · Evaluate · Docker · DVC

Documento único para reproduzir o Tech Challenge Fase 02 de ponta a ponta.

## 1. Pré-requisitos

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recomendado)
- Docker + Compose (opcional, para MLflow em container)
- Git + acesso SSH ao repositório

## 2. Setup do ambiente

```bash
git clone git@github.com:andreabreu/FIAP-PosTech-Fase02.git
cd FIAP-PosTech-Fase02
uv sync --all-groups
cp .env.example .env
uv run python scripts/validate_env.py
```

Validação rápida:

```bash
uv run pytest -q
uv run ruff check src tests scripts
```

Variáveis importantes em `.env` / `.env.example`:

| Variável | Default local | Notas |
|----------|---------------|--------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Local sem Docker |
| `MLFLOW_EXPERIMENT_NAME` | `recommender-fase02` | |
| `MODEL_NAME` | `mlp` | `mlp` · `popularity` · `svd` |
| `RANDOM_SEED` | `42` | |

**Nunca** commitar o `.env` real.

Detalhes extras: [SETUP.md](SETUP.md) · [INSTALL_CHECKLIST.md](INSTALL_CHECKLIST.md)

## 3. Dados e pipeline DVC

Stages em `dvc.yaml`:

1. `preprocess` → `data/processed/interactions_clean.csv`
2. `feature_eng` → `data/processed/features/{train,test}.csv`
3. `train` → `models/recommender.pt` + `metrics/train.json`
4. `evaluate` → `metrics/evaluate.json` (Precision / Recall / Hit / NDCG @K)

```bash
uv run dvc pull          # baixa raw versionado
uv run dvc repro         # executa stages desatualizados
uv run dvc metrics show
uv run dvc dag
```

Stages individuais:

```bash
uv run dvc repro preprocess
uv run dvc repro feature_eng
uv run dvc repro train
uv run dvc repro evaluate
```

Hiperparâmetros: `params.yaml`.  
Remote local: `.dvc-remote/` (ver [DVC.md](DVC.md)).

## 4. Treino

### Via DVC (recomendado)

```bash
uv run dvc repro train
```

Usa MLP default (`params.yaml`), gera `models/recommender.pt`.

### Via CLI

```bash
# MLP + métricas, sem MLflow
uv run python scripts/run_train.py \
  --train data/processed/features/train.csv \
  --test data/processed/features/test.csv \
  --params params.yaml \
  --model-out models/recommender.pt \
  --metrics metrics/train.json \
  --no-mlflow

# Com MLflow (sqlite local)
uv run python scripts/run_train.py --run-name local-mlp
```

### Experimentos (baselines + MLP)

```bash
uv run python scripts/run_experiments.py --suite all
# ou: --suite baselines | --suite mlp
```

Resultados agregados: `metrics/experiments_summary.json`.

## 5. Evaluate

```bash
uv run dvc repro evaluate
# ou
uv run python scripts/run_evaluate.py \
  --model models/recommender.pt \
  --metrics metrics/evaluate.json \
  --k 10
```

Métricas obrigatórias (@K):

1. Precision@K  
2. Recall@K  
3. Hit-Rate@K  
4. NDCG@K  

## 6. MLflow Registry

```bash
# após experimentos com log_mlflow=True
uv run python scripts/promote_model.py --to both
# --to staging | production | both
```

- Modelo registrado: `fiap-fase02-recommender`
- Critério de promoção: melhor `ndcg_at_10` entre runs tagueados `model=mlp`
- Fluxo: **Staging → Production**
- Registro local: `metrics/registry_promotion.json`
- Model Card: [MODEL_CARD.md](../MODEL_CARD.md)

## 7. Docker

| Serviço | Porta host | Função |
|---------|------------|--------|
| `mlflow` | **5001**→5000 | UI / tracking (macOS evita conflito com :5000) |
| `train` | — | `python -m src.training.entrypoint` |

```bash
docker compose up --build
open http://127.0.0.1:5001

# só MLflow
docker compose up mlflow

# treino one-shot
docker compose run --rm train
```

No Compose, `MLFLOW_TRACKING_URI=http://mlflow:5000`.  
Volumes: `./models`, `./metrics`.  
Detalhes: [DOCKER.md](DOCKER.md).

## 8. Fluxo feliz (resumo)

```bash
uv sync --all-groups && cp .env.example .env
uv run python scripts/validate_env.py
uv run dvc pull && uv run dvc repro
uv run python scripts/run_experiments.py --suite all
uv run python scripts/promote_model.py --to both
uv run pytest -q
```

## 9. Entrega

- Repositório GitHub reproduzível (este guia + README)
- Vídeo STAR ≤ 5 min (arquivo local em `docs/STAR_Fase02_apresentacao.mp4` — subir YouTube/Drive e colar link no README)
- Deploy nuvem (bônus): https://study-pos-fiap-fase02.aedigital.solutions/health
