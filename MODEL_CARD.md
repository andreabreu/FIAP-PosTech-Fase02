# Model Card — Fase 02 (recomendação)

## Modelo

| Campo | Valor |
|-------|--------|
| Nome | `fiap-fase02-recommender` |
| Tarefa | Recomendação (feedback implícito) |
| Framework | PyTorch MLP (embeddings user/item) |
| Baselines | Popularity e TruncatedSVD (sklearn) |
| Autores | Andre Abreu e Elizandra Monteiro |

## Uso

Sugerir top-K itens do catálogo a partir do histórico de interações (view, click, cart, purchase).

## Dados de treino

- CSV sintético em `data/raw/interactions.csv` (DVC)
- ~10k linhas, ~800 users, ~1200 items

## Métricas (K=10)

| Modelo | P@10 | R@10 | Hit@10 | NDCG@10 |
|--------|-----:|-----:|-------:|--------:|
| popularity | 0.0026 | 0.0082 | 0.0263 | 0.0054 |
| svd | 0.0025 | 0.0074 | 0.0238 | 0.0054 |
| mlp (Production) | 0.0026 | 0.0086 | 0.0263 | 0.0062 |

## Limitações

- Dados sintéticos (não representam sazonalidade real)
- Popularity favorece itens head / long-tail fica fraco
- Cold-start (user/item sem histórico) tem score fraco
- Sem atributos demográficos

## Como reproduzir

```bash
uv sync --all-groups
uv run dvc pull && uv run dvc repro
```
