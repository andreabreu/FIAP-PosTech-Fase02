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
9. (Opcional) `uv run python scripts/run_experiments.py --suite all`
10. (Opcional) `uv run python scripts/promote_model.py --to both`

Guia unificado: [GUIDE.md](GUIDE.md).

## Extras opcionais

```bash
# API scaffolding (FastAPI)
uv sync --extra api

# Docker (MLflow UI em http://127.0.0.1:5001)
docker compose up --build
```

Se algum passo falhar, registre o log e abra issue/PR na dupla.
