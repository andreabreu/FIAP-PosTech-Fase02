# Checklist — instalação limpa ponta a ponta

Execute em um diretório temporário / máquina limpa:

1. `git clone git@github.com:andreabreu/FIAP-PosTech-Fase02.git && cd FIAP-PosTech-Fase02`
2. `uv sync --all-groups`
3. `cp .env.example .env`
4. `uv run python scripts/validate_env.py` → deve imprimir `Environment validation OK`
5. `uv run pytest -q` → testes verdes
6. `uv run ruff check src tests scripts` → sem erros
7. `uv run dvc pull` → baixa `data/raw/interactions.csv`
8. `uv run dvc repro` → pipeline ≥3 stages OK

## Extras opcionais

```bash
# API scaffolding (FastAPI) — futuro endpoint de recomendação
uv sync --extra api

# Training tooling explícito (já coberto pelas deps principais + tqdm)
uv sync --extra training

# Docker
docker compose up --build
```

Se algum passo falhar, registre o log e abra issue/PR na dupla.
