# FIAP-PosTech-Fase02

FIAP Pós Tech — Machine Learning Engineering and Analytics  
**Tech Challenge Fase 02** — sistema de recomendação para e-commerce.

## Visão geral

Empresa de e-commerce precisa recomendar produtos com base no comportamento de navegação / interações user–item.  
Stack: **PyTorch MLP**, baselines **Scikit-Learn** (popularity + SVD), **DVC**, **MLflow Registry**, **Docker**.

## Relação com a Fase 01

- Fase 01 (churn): [FIAP-PosTech](https://github.com/andreabreu/FIAP-PosTech)
- Fase 02 (recomendação): este repositório

## Equipe

- Andre Abreu — `andre.abreu@live.com`
- Elizandra Monteiro — `monteiroelizandra2017@gmail.com`

## Status

Pipeline completo: DVC → treino MLP/baselines → ≥4 métricas → MLflow tracking → Registry **Staging → Production**.  
Model Card: [MODEL_CARD.md](MODEL_CARD.md).

## Setup

```bash
uv sync --all-groups
cp .env.example .env
uv run python scripts/validate_env.py
uv run dvc pull && uv run dvc repro
```

Docs: [SETUP](docs/SETUP.md) · [DVC](docs/DVC.md) · [DOCKER](docs/DOCKER.md) · [INSTALL](docs/INSTALL_CHECKLIST.md)

## Treino e experimentos

```bash
# Pipeline DVC (MLP default)
uv run dvc repro

# Sweep baselines + MLP (loga no MLflow sqlite:///mlflow.db)
uv run python scripts/run_experiments.py --suite all

# Promover melhor MLP (NDCG@10) → Staging → Production
uv run python scripts/promote_model.py --to both
```

## Resultados (K=10, dataset sintético)

| Modelo | Precision@10 | Recall@10 | Hit@10 | NDCG@10 |
|--------|-------------:|----------:|-------:|--------:|
| popularity | 0.0026 | 0.0082 | 0.0263 | 0.0054 |
| svd | 0.0025 | 0.0074 | 0.0238 | 0.0054 |
| **mlp (promoted)** | **0.0026** | **0.0086** | **0.0263** | **0.0062** |
| mlp (emb=32) | 0.0018 | 0.0060 | 0.0175 | 0.0034 |

> Números baixos são esperados em interações sintéticas aleatórias; o objetivo da fase é a engenharia MLOps reproduzível. Detalhes em `metrics/experiments_summary.json`.

Registered model: `fiap-fase02-recommender` (Production).
