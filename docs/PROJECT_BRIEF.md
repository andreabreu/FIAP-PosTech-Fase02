# Project brief — Tech Challenge Fase 02

## Problema de negócio

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos
baseado no comportamento de navegação e interações user–item.

## Objetivos técnicos

- Treinar rede neural (MLP ou embedding-based) com **PyTorch**
- Comparar com baselines **Scikit-Learn** usando ≥ 4 métricas
- Versionar dados e pipeline com **DVC** (≥ 3 stages)
- Rastrear experimentos e registrar modelo com **MLflow** (Staging → Production)
- Empacotar com **Docker** multi-stage + Compose (treino + MLflow)
- Código com clean code, type hints e pelo menos um design pattern

## Datasets candidatos

- Instacart Market Basket
- RetailRocket
- MovieLens (recomendação)
- Qualquer corpus com **≥ 10.000** interações user–item

## Entregas

- Repositório GitHub reproduzível
- ~~Vídeo STAR ≤ 5 minutos~~ — **fora de escopo** (decisão da dupla)
- (Opcional) Deploy em nuvem

## Fora de escopo

- Vídeo STAR (roteiro, gravação, upload e link no README)
- Implementação adicional pós code freeze, salvo hotfix crítico
