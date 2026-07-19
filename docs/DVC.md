# DVC — dados e pipeline

## Visão geral

O projeto versiona dados com **DVC** e orquestra o pipeline em `dvc.yaml`:

1. `preprocess` — limpa e deduplica interações
2. `feature_eng` — encoding user/item + split train/test
3. `train` — MLP PyTorch (`models/recommender.pt`)
4. `evaluate` — Precision / Recall / Hit / NDCG @K

Dataset raw (≥10k interações sintéticas): `data/raw/interactions.csv`  
Pointer Git: `data/raw/interactions.csv.dvc`

Guia unificado: [GUIDE.md](GUIDE.md).

## Pré-requisitos

```bash
uv sync --all-groups
```

## Remote local

Remote default: diretório `.dvc-remote/` (gitignored).

```bash
uv run dvc remote list
uv run dvc push
uv run dvc pull
```

Para remote compartilhado (S3/GCS/SSH):

```bash
uv run dvc remote add -d myremote s3://bucket/path
uv run dvc push
```

## Regenerar sample (opcional)

```bash
uv run python scripts/generate_sample_data.py --n-rows 12000
uv run dvc add data/raw/interactions.csv
```

## Rodar o pipeline

```bash
uv run dvc pull
uv run dvc repro
uv run dvc metrics show
uv run dvc dag
```

```bash
uv run dvc repro preprocess
uv run dvc repro feature_eng
uv run dvc repro train
uv run dvc repro evaluate
```

## Params

Hiperparâmetros em `params.yaml`. Alterar params invalida stages dependentes.

## Artefatos

| Path | Origem |
|------|--------|
| `data/raw/interactions.csv` | DVC tracked |
| `data/processed/interactions_clean.csv` | preprocess |
| `data/processed/features/` | feature_eng |
| `models/recommender.pt` | train |
| `metrics/*.json` | métricas (`cache: false`) |
