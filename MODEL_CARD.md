# Model Card — FIAP Pós Tech Fase 02 Recommender

## Model details

| Field | Value |
|-------|--------|
| Name | `fiap-fase02-recommender` |
| Task | Implicit feedback product recommendation |
| Framework | PyTorch MLP (user/item embeddings + MLP head) |
| Baselines | Popularity · TruncatedSVD (sklearn) |
| Owners | Andre Abreu · Elizandra Monteiro |
| Version | 1.0.0-rc |

## Intended use

Suggest top-K catalog items for an e-commerce user based on historical
user–item interactions (view / click / cart / purchase).

**Out of scope:** real-time session ranking, content-based cold-start for
brand-new SKUs without any interactions, fairness auditing at segment level.

## Training data

- Synthetic interactions CSV (`data/raw/interactions.csv`) versioned with DVC
- ≥ 10,000 rows · ~800 users · ~1,200 items
- Implicit scores derived from event type (+ optional rating)

## Metrics (held-out users, K=10)

Latest sweep (`metrics/experiments_summary.json`):

| Modelo | P@10 | R@10 | Hit@10 | NDCG@10 |
|--------|-----:|-----:|-------:|--------:|
| popularity | 0.0026 | 0.0082 | 0.0263 | 0.0054 |
| svd | 0.0025 | 0.0074 | 0.0238 | 0.0054 |
| mlp (best → Production) | 0.0026 | 0.0086 | 0.0263 | 0.0062 |

Primary ranking metrics:

1. Precision@10  
2. Recall@10  
3. Hit-Rate@10  
4. NDCG@10

## Limits and biases

- Synthetic data does **not** reflect real purchase seasonality or catalog churn.
- Popularity baseline over-recommends head items (long-tail under-exposure).
- MLP trained with random negatives may underperform on sparse users.
- No demographic attributes → cannot measure disparate impact across groups.
- Cold-start users/items without history receive weak / random scores.

## Ethical considerations

Recommendations can amplify popularity bias and reduce catalog diversity.
Production use should add business rules (stock, category caps, exploration).

## How to reproduce

```bash
uv sync --all-groups
uv run dvc pull && uv run dvc repro
uv run python scripts/run_experiments.py --suite all
uv run python scripts/promote_model.py --to both
```
