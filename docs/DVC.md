# DVC — dados e pipeline

## Visão geral

O projeto versiona dados com **DVC** e orquestra o pipeline em `dvc.yaml`:

1. `preprocess` — limpa e deduplica interações
2. `feature_eng` — encoding user/item + split train/test
3. `train` — treino (stub até a F4; artifact em `models/`)
4. `evaluate` — métricas leves em `metrics/evaluate.json`

Dataset raw (>=10k interações sintéticas): `data/raw/interactions.csv`  
Pointer Git: `data/raw/interactions.csv.dvc`

## Pré-requisitos

```bash
uv sync --all-groups
```

## Remote local

Remote default: diretório `.dvc-remote/` (gitignored).

```bash
# já configurado no .dvc/config como remote "localremote"
uv run dvc remote list
uv run dvc push
uv run dvc pull
```

Para um remote compartilhado (S3/GCS/SSH), altere `.dvc/config` ou use:

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
uv run dvc pull          # baixa dados versionados
uv run dvc repro         # executa stages desatualizados
uv run dvc metrics show  # métricas cache:false
uv run dvc dag           # visualiza dependências
```

Stages individuais:

```bash
uv run dvc repro preprocess
uv run dvc repro feature_eng
uv run dvc repro train
uv run dvc repro evaluate
```

## Params

Hiperparâmetros de features/treino em `params.yaml`.  
Alterar params invalida stages dependentes no próximo `dvc repro`.

## Artefatos

| Path | Origem |
|------|--------|
| `data/raw/interactions.csv` | DVC tracked |
| `data/processed/interactions_clean.csv` | stage preprocess |
| `data/processed/features/` | stage feature_eng |
| `models/recommender_stub.json` | stage train |
| `metrics/*.json` | métricas (commitadas, `cache: false`) |
