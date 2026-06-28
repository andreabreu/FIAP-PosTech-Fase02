# FIAP-PosTech-Fase02

FIAP Pós Tech — Machine Learning Engineering and Analytics  
**Tech Challenge Fase 02** — sistema de recomendação para e-commerce.

## Visão geral

Empresa de e-commerce precisa recomendar produtos com base no comportamento de navegação / interações user–item.  
Stack alvo: **PyTorch** (MLP ou embeddings), **Scikit-Learn** (baselines), **DVC**, **MLflow**, **Docker**, clean code profissional.

## Relação com a Fase 01

- Fase 01 (churn): [FIAP-PosTech](https://github.com/andreabreu/FIAP-PosTech)
- Fase 02 (recomendação): este repositório

## Equipe

- Andre Abreu — `andre.abreu@live.com`
- Elizandra Monteiro — `monteiroelizandra2017@gmail.com`

## Status

Repositório em fase inicial (kickoff). Estrutura, pipeline e modelo serão evoluídos em commits semânticos.

## Setup

Ambiente reproduzível com **uv** (lock commitado):

```bash
uv sync --all-groups
cp .env.example .env
uv run python scripts/validate_env.py
```

Detalhes em [docs/SETUP.md](docs/SETUP.md).

Checklist completo: [docs/INSTALL_CHECKLIST.md](docs/INSTALL_CHECKLIST.md).

Docker/Compose: [docs/DOCKER.md](docs/DOCKER.md).
