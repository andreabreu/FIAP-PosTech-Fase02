# `src/` — responsabilidades dos módulos

| Pacote | Responsabilidade |
|--------|------------------|
| `config` | Defaults e settings do projeto |
| `domain` | Interfaces (DataLoader, RecommenderModel) |
| `schemas` | Contratos Pydantic de interações |
| `data` | Leitura/carga de datasets (reader + loader) |
| `features` | Pré-processamento (Strategy) e feature eng. |
| `models` | Factory e implementações de recomendadores |
| `training` | Loops de treino (próximas etapas) |
| `evaluation` | Métricas e relatórios (próximas etapas) |
| `utils` | Seed, logging e helpers transversais |

## Padrões aplicados

- **Strategy:** `features/preprocessors` (`PreprocessorStrategy`)
- **Factory:** `models/factory.py` (`ModelFactory`)

## Convenções

- Type hints em APIs públicas
- Docstrings estilo Google
- Funções curtas e módulos com responsabilidade única (SOLID)
