# Tech Challenge Fase 02 — Recomendação

FIAP Pós Tech · Machine Learning Engineering and Analytics

Sistema de recomendação de produtos (e-commerce) com interações user–item.

**Dupla:** Andre Abreu (`andre.abreu@live.com`) e Elizandra Monteiro (`monteiroelizandra2017@gmail.com`)

Fase 01 (churn): [FIAP-PosTech](https://github.com/andreabreu/FIAP-PosTech)

## O que tem no projeto

- Modelo MLP (PyTorch) + baselines popularity e SVD (sklearn)
- Pipeline DVC (preprocess → features → train → evaluate)
- MLflow (tracking + Model Registry Staging → Production)
- Docker multi-stage + Compose
- Model Card: [MODEL_CARD.md](MODEL_CARD.md)

## Como rodar

```bash
uv sync --all-groups
cp .env.example .env
uv run python scripts/validate_env.py
uv run dvc pull
uv run dvc repro
```

Mais detalhes: [docs/GUIDE.md](docs/GUIDE.md), [docs/SETUP.md](docs/SETUP.md), [docs/DVC.md](docs/DVC.md), [docs/DOCKER.md](docs/DOCKER.md).

## API (bônus — deploy)

- Health: https://study-pos-fiap-fase02.aedigital.solutions/health
- Recommend: https://study-pos-fiap-fase02.aedigital.solutions/recommend?user_id=U0001&k=10

```bash
uv sync --extra api
uv run uvicorn src.api.main:app --port 8000
```

## Resultados (K=10, dataset sintético)

| Modelo | Precision@10 | Recall@10 | Hit@10 | NDCG@10 |
|--------|-------------:|----------:|-------:|--------:|
| popularity | 0.0026 | 0.0082 | 0.0263 | 0.0054 |
| svd | 0.0025 | 0.0074 | 0.0238 | 0.0054 |
| **mlp (Production)** | **0.0026** | **0.0086** | **0.0263** | **0.0062** |
| mlp (emb=32) | 0.0018 | 0.0060 | 0.0175 | 0.0034 |

Números baixos no sintético são esperados.  
Modelo no Registry: `fiap-fase02-recommender` (Production).
