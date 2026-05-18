# Contribuição — fluxo da dupla

## Autores

| Pessoa | Git name | E-mail |
|--------|----------|--------|
| Andre Abreu | `Andre Abreu` | `andre.abreu@live.com` |
| Elizandra Monteiro | `Elizandra Monteiro` | `monteiroelizandra2017@gmail.com` |

## Papéis

- **Elizandra:** estrutura, dependências, Docker, DVC/dados, README de setup
- **André:** modelo PyTorch, baselines, MLflow/Registry, Model Card, métricas

## Commits

- Conventional Commits em inglês (`feat:`, `fix:`, `docs:`, `chore:`, …)
- Commits pequenos e atômicos
- Preferência por sessões em finais de semana e período noturno
- Author e committer devem refletir quem implementou o pedaço

## Branches

Usar branches curtas por feature (`feat/structure`, `feat/dvc`, `feat/mlp`) e merge na `main` após revisão do par quando possível.

## O que não versionar

Secrets (`.env`), datasets grandes (usar DVC), caches de experimento e artefatos pesados de modelo.
