# Model Card — MLP Temporal Recommender v1

> Referência: Mitchell et al., "Model Cards for Model Reporting" (FAccT 2019).
> Formato adaptado para **ranking de recomendações** com feedback implícito.

---

## 1. Detalhes do Modelo

| Campo | Valor |
|---|---|
| **Nome** | `mlp-temporal-recommender` (MLflow Model Registry) |
| **Versão** | v1 — run de referência `mlp_temporal_v1_catfix` |
| **Data de treinamento** | Julho/2026 |
| **Tipo** | Ranker supervisionado (re-ranking de candidatos, feedback implícito) |
| **Framework** | PyTorch 2.x + scikit-learn (padronização de features) |
| **Arquitetura** | 4 embeddings (`user_id`, `product_id`, `aisle_id`, `department_id`, dim=32) + 17 features numéricas padronizadas → concat (145) → `Linear 128 → ReLU → Dropout 0,25` → `Linear 64 → ReLU → Dropout 0,25` → `Linear 1` (logit de relevância) |
| **Loss** | `BCEWithLogitsLoss` (positivo = produto presente no pedido-alvo da janela) |
| **Otimizador** | Adam (lr=1e-3, weight_decay=1e-5) + `ReduceLROnPlateau` (patience=5, factor=0,5) |
| **Regularização** | Dropout 0,25 + early stopping (patience=10, máx. 30 épocas) — melhor época: **2** |
| **Batch size** | 8.192 |
| **Categorias não vistas** | Índice UNK reservado (`padding_idx=0`, embedding zero) — permite pontuar usuários/produtos fora do vocabulário de treino |
| **Seleção de modelo** | NDCG@10 no split de validação |
| **Reprodutibilidade** | `RANDOM_SEED=42` (Python, NumPy, PyTorch); pipeline DVC `preprocess → train → evaluate` |
| **Autores** | Grupo 16 — POSTECH Tech Challenge Fase 02 |
| **Licença** | MIT |
| **Experimento MLflow** | `mlp-market-recommender-system-temporal-v1` |

**Contexto no sistema.** O modelo é o segundo estágio de um sistema de dois
estágios: um **gerador de candidatos** seleciona até 200 produtos por janela de
usuário a partir de 4 fontes (recompra, similaridade Jaccard entre usuários,
afinidade de categoria e popularidade global); o MLP **reordena** essa lista
pontuando cada par (janela, produto). Para usuários sem histórico, a inferência
usa **fallback de popularidade** (top-100 produtos globais).

---

## 2. Uso Pretendido

### Casos de uso primário

- **Recomendação do próximo carrinho:** dado o histórico de pedidos de um
  usuário de e-commerce alimentar, prever os produtos do próximo pedido — com
  ênfase em **recompras** — e exibir um top-K personalizado.
- **Ordenação de vitrines "compre de novo":** ranking de produtos já
  conhecidos do usuário por probabilidade de recompra.

### Usuários pretendidos

| Usuário | Como usa |
|---|---|
| Equipe de produto/e-commerce | Consome o top-K por usuário para vitrines personalizadas |
| Equipe de dados/ML | Opera o pipeline (`dvc repro`), monitora métricas e retreina |
| Avaliadores do Tech Challenge | Reproduzem o pipeline e auditam runs no MLflow |

### Fora de escopo

- **Descoberta de produtos novos** — o modelo é forte em recompra; a fonte de
  candidatos por categoria/popularidade dá alguma exploração, mas o sinal
  dominante é o histórico do próprio usuário.
- **Previsão de demanda ou estoque** — o score ordena produtos para *um*
  usuário; não estima volume agregado de vendas.
- **Precificação ou promoções dirigidas** — o dataset não tem preço; qualquer
  uso comercial sensível exigiria novas features e nova avaliação.
- **Outros domínios de varejo** — treinado apenas em supermercado (Instacart,
  EUA, 2017); catálogos e hábitos de outros segmentos exigem retreino.
- **Decisões automatizadas sobre pessoas** — o score expressa afinidade
  usuário-produto, nada além disso.

---

## 3. Fatores

Sinais que mais influenciam o ranking, pela evidência dos baselines
(notebook 05) e do desenho de features (notebook 04):

### Recompra e recência (maior peso)

| Sinal | Observação |
|---|---|
| `user_product_purchase_count` / `user_product_reorder_count` | Frequência do par usuário-produto no histórico — o baseline puro de recompra quase empata com o melhor baseline (NDCG@10 0,4677), mostrando a força deste sinal |
| `user_product_orders_since_last_purchase` / `..._days_since_last_purchase` | Recência: produtos comprados há pouco tendem a voltar ao carrinho |
| `user_product_was_bought_before` | Flag binária de recompra — separa o regime "reordenar" do regime "descobrir" |

### Comportamento do usuário

| Sinal | Observação |
|---|---|
| `user_reorder_rate` | Usuários habituais concentram o ganho do modelo |
| `user_avg_basket_size` / `user_avg_days_between_orders` | Calibram quantos itens e com que cadência o usuário compra |
| `user_has_single_prior_order` | Históricos curtos têm features de recompra pouco informativas — regime mais difícil |

### Identidade e categoria (embeddings)

| Sinal | Observação |
|---|---|
| Embeddings de `user_id` e `product_id` | Capturam afinidades latentes não explicáveis pelas features manuais |
| Embeddings de `aisle_id` e `department_id` + `user_aisle/department_purchase_count` | Generalizam para produtos pouco vistos via categoria |

---

## 4. Métricas de Performance

### Definições

Avaliação por **ranking dentro de cada janela** (`user_window_id`), nos cortes
k = 5, 10 e 20:

- **NDCG@k** — qualidade da ordenação (posiciona os relevantes no topo). Métrica
  primária de seleção de modelo, em k=10.
- **Hit rate@k** — fração de janelas com ≥ 1 produto relevante no top-k.
- **Recall local@k** — fração dos positivos *presentes nos candidatos* que
  chegou ao top-k (isola a qualidade do ranker).
- **Precisão@k** — fração do top-k que é relevante.

> O **recall global** (sobre o carrinho completo, incluindo produtos que o
> gerador de candidatos não cobriu) mede o sistema de dois estágios inteiro:
> ~0,34 em k=10 para os melhores rankers. Ele é reportado como contexto — o
> teto é dado pela cobertura do gerador de candidatos (notebook 03), não pelo
> MLP.

### MLP — teste (holdout final, nunca usado em treino/seleção)

| k | NDCG | Hit rate | Recall local | Precisão |
|---|---|---|---|---|
| 5 | 0,4860 | 0,8377 | 0,3326 | 0,3985 |
| **10** | **0,4948** | **0,9058** | **0,4793** | **0,3130** |
| 20 | 0,5409 | 0,9484 | 0,6373 | 0,2252 |

### MLP vs. baselines — validação, k=10

| Modelo | NDCG@10 | Hit rate@10 | Recall local@10 | Precisão@10 |
|---|---|---|---|---|
| Afinidade de categoria | 0,2174 | 0,6481 | 0,2120 | 0,1264 |
| Heurístico temporal | 0,4557 | 0,8563 | 0,4413 | 0,2736 |
| Recompra do usuário | 0,4677 | 0,8645 | 0,4564 | 0,2809 |
| Ordem do gerador de candidatos (melhor baseline) | 0,4678 | 0,8646 | 0,4567 | 0,2810 |
| **MLP temporal (este modelo)** | **0,4999** | **0,9086** | **0,4876** | **0,3119** |

Ganho absoluto de **+0,0321 em NDCG@10** sobre o melhor baseline, consistente
também em hit rate, recall local e precisão — o modelo não só ordena melhor,
como coloca mais itens relevantes no top-10. Números completos (3 splits ×
3 cortes de k): `reports/metrics.json` e notebooks 05–06.

### Leitura de negócio

Em 10 sugestões, **~91% dos pedidos têm ao menos um acerto** e ~3 sugestões são
compradas (precisão 0,31 com carrinho médio de ~10 itens). A precisão cai com k
maior por construção — o carrinho é finito.

---

## 5. Dados de Avaliação

| Campo | Valor |
|---|---|
| **Validação** | 115.909 janelas (3ª janela `prior` de cada usuário elegível) — usada para early stopping e seleção de modelo |
| **Teste** | 115.909 janelas — o pedido `eval_set=train` original do Instacart, preservado como **holdout final** (nunca visto em treino ou seleção) |
| **Candidatos por janela** | Até 200 produtos (4 fontes) |
| **Contaminação** | Nenhuma: features usam apenas eventos com `order_number ≤ history_end_order_number` da janela; scaler e mapas categóricos ajustados só no treino |

---

## 6. Dados de Treinamento

| Campo | Valor |
|---|---|
| **Fonte** | Instacart Online Grocery Shopping Dataset 2017 (Kaggle) — ~3,4M pedidos, ~206 mil usuários, ~50 mil produtos |
| **Unidade de modelagem** | Par `user_window_id`-`product_id`: cada usuário elegível gera 3 janelas temporais no conjunto `prior` (2 de treino + 1 de validação); o alvo de cada janela é o pedido imediatamente seguinte |
| **Tamanho** | 231.818 janelas de treino; ~92,7M pares candidato-janela no dataset completo; 3.454.841 positivos (~3,7%) |
| **Rótulo** | Implícito: 1 se o produto candidato aparece no pedido-alvo da janela |
| **Armazenamento** | `data/features/temporal_modeling_dataset_v1` — 47 partições parquet (~605 MB), versionado por DVC em GCS |
| **Determinismo** | Geração de candidatos com desempate estável (`mergesort`) — bug de reprodutibilidade cross-platform corrigido e documentado em `docs/revisao_integracao_notebooks.md` |

---

## 7. Análises Quantitativas

### Generalização validação → teste

A diferença entre validação e teste é pequena e uniforme (NDCG@10: 0,4999 →
0,4948; hit rate@10: 0,9086 → 0,9058), indicando que a seleção por validação
não sobreajustou e que o corte temporal das janelas evitou vazamento.

### Overfitting controlado

A partir da época 3 a loss de treino continua caindo enquanto o NDCG@10 de
validação piora — o early stopping preservou o checkpoint da época 2. Curvas
completas em `reports/training_history.csv`.

### Dependência do gerador de candidatos

O recall local@10 de 0,4876 mostra que o ranker ainda deixa ~51% dos positivos
cobertos fora do top-10; o recall global (~0,34@10) mostra que o teto do
sistema é limitado pela cobertura do gerador (analisada no notebook 03).
Melhorar o gerador provavelmente rende mais que melhorar o ranker neste ponto.

### Usuário desconhecido (cold start)

Dois mecanismos cobrem o regime sem histórico: o índice UNK nos embeddings
(usuário fora do vocabulário ainda é pontuado pelas features e demais
embeddings — simulável com `scripts/predict.py --simulate-unknown-user`) e o
fallback de popularidade (top-100 global) quando não há candidatos. A
performance nesse regime **não foi medida formalmente** — ver Seção 9.

---

## 8. Considerações Éticas

### Proteção de dados

- O dataset é **público e anonimizado**: usuários são IDs numéricos, sem
  qualquer atributo demográfico (idade, gênero, localização, renda).
- O pipeline não ingere nem loga PII; artefatos e logs contêm apenas IDs e
  métricas agregadas.

### Vieses identificados

| Risco | Detalhes | Mitigação |
|---|---|---|
| **Reforço de hábito (feedback loop)** | O sinal dominante é recompra: o modelo tende a recomendar o que o usuário já compra, reduzindo descoberta e podendo cristalizar padrões de consumo | Fontes de candidatos por categoria/popularidade dão exploração mínima; em produção, considerar cota de diversidade no top-K |
| **Viés de popularidade** | Produtos populares aparecem mais como candidatos e no fallback — efeito "rico fica mais rico" para o catálogo | Monitorar cobertura de catálogo das recomendações |
| **Auditoria de fairness demográfica impossível** | Sem atributos de pessoa no dataset, não há como medir disparidade entre grupos | Documentado como limitação; reavaliar se o modelo for aplicado a dados com demografia |
| **Contexto do dataset** | Clientes Instacart (EUA, 2017) — população com acesso a delivery premium; hábitos não representam outros mercados | Retreinar com dados locais antes de qualquer transferência |

### Uso responsável

- O score deve ser usado para **conveniência** (lembrar/reordenar), não para
  induzir compra de categorias sensíveis (ex.: álcool) — se o catálogo real
  tiver essas categorias, aplicar regras de negócio por fora do modelo.
- Recomendações não devem ser apresentadas como aval nutricional ou de saúde.

---

## 9. Ressalvas e Recomendações

| Ressalva | Impacto | Recomendação |
|---|---|---|
| Avaliação exclusivamente offline | Métricas de ranking não garantem lift de negócio (CTR, conversão) | Validar com teste A/B antes de decisões de produto |
| Dataset estático de 2017, sem datas absolutas | Sazonalidade (festas, verão) não modelada; `days_since_prior_order` é capado em 30 | Tratar resultados como prova de conceito metodológica |
| Cold start não medido formalmente | Performance para usuários novos é desconhecida (só há mecanismo de fallback) | Medir NDCG/hit rate no regime UNK antes de expor a usuários novos |
| Teto do gerador de candidatos | Recall global ~0,34@10 independe do ranker | Priorizar cobertura do gerador na v2 |
| Sem preço, promoção ou estoque | Sinais decisivos de compra real ausentes | Enriquecer features antes de uso comercial |
| Score não é probabilidade calibrada | O logit serve para **ordenar**, não para estimar P(compra) | Aplicar calibração se algum consumidor precisar de probabilidade |
| Uma única run de arquitetura final | Sem intervalo de confiança entre seeds | Rodar ≥ 3 seeds no ciclo de retreino para estimar variância |

### Gatilhos de retreinamento

- NDCG@10 offline do modelo cair abaixo do melhor baseline heurístico
  (0,4678) em nova amostra de janelas;
- Mudança relevante de catálogo (novos departamentos/aisles dominantes) —
  embeddings de categoria ficam obsoletos;
- Queda na taxa média de reorder da base (mudança de comportamento) — o sinal
  central do modelo perde força.

---

## Reprodução

```bash
uv run dvc pull     # dados versionados (GCS)
uv run dvc repro    # preprocess → train → evaluate
cat reports/metrics.json
```

Histórico completo de decisões: notebooks `01`–`07` e
`docs/revisao_integracao_notebooks.md`.
