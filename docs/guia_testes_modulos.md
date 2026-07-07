# Guia de testes — módulos do `src/`

> Como testar cada módulo do projeto, o que cada um faz e o que os testes
> provam. Atualizado em 2026-07-04 (34 testes na suíte).

## ⚠️ Antes de tudo: `--no-sync`

O ambiente local tem a build CUDA do torch instalada por fora do lock
(RTX 5080). Um `uv run` comum **re-sincroniza com o lock e reverte o torch
para CPU** silenciosamente. Por isso, todos os comandos usam:

```bash
uv run --no-sync <comando>
```

## Comandos rápidos

```bash
# Suíte completa (34 testes, ~3 s)
uv run --no-sync pytest

# Um módulo específico, com saída detalhada
uv run --no-sync pytest tests/test_settings.py -v

# Um teste específico (para depurar)
uv run --no-sync pytest tests/test_models.py::test_factory_constroi_mlp_temporal -v

# Qualidade de código (lint + formatação)
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts

# Cobertura (qual % das linhas de src/ os testes exercitam)
uv run --no-sync pytest --cov=src --cov-report=term-missing
```

## O fluxo completo (onde cada módulo entra)

```
data/features/temporal_modeling_dataset_v1   (47 partições parquet, DVC)
        │
        ▼
[src/features]  ajusta mapas categóricos + scaler  ──►  JSONs (stage preprocess)
        │
        ▼
[src/data]      TemporalParquetDataset: lê partição a partição,
                aplica o pré-processamento e emite batches de tensores
        │
        ▼
[src/models]    Factory constrói o MLPRecommender (embeddings + MLP)
        │
        ▼
[src/training]  loop de épocas + EarlyStopping + checkpoint
        │
        ▼
[src/evaluation] métricas de ranking (NDCG@k etc.) por janela de usuário
        │
        ▼
reports/metrics.json  (stage evaluate → dvc metrics show)

[src/config] alimenta tudo com configuração do .env (URIs, seed, caminhos)
```

## Módulo a módulo

### 1. `src/config` — configuração centralizada

**O que faz:** carrega configurações (URI do MLflow, credenciais, seed,
caminhos) do `.env`/ambiente via Pydantic Settings. É o que permite trocar
de provedor (local → GCP) sem tocar em código: só o `.env` muda.

**Teste:** `uv run --no-sync pytest tests/test_settings.py -v`

| Teste | O que prova |
|---|---|
| `test_defaults_sem_env` | Sem `.env`, os padrões seguros valem |
| `test_sobrescrita_por_variavel_de_ambiente` | Variável de ambiente tem precedência (12-factor app) |
| `test_segredo_nao_vaza_em_repr` | O token do MLflow não aparece em logs (`SecretStr`) |
| `test_get_settings_e_singleton` | O `.env` é lido uma única vez por processo |

### 2. `src/features` — pré-processamento (anti-leakage)

**O que faz:** duas transformações, ambas "aprendidas" **só no split de
treino**: (a) mapas categóricos — `product_id` bruto → índice contíguo 1..N,
com 0 reservado para categorias nunca vistas (UNK); (b) z-score — média/desvio
das 17 features numéricas, calculados em streaming pelas partições (o dataset
não cabe em memória). Também salva/carrega esses artefatos em JSON.

**Por que "fit só no treino" importa:** se a média incluísse linhas de
validação, o modelo receberia informação do futuro — *data leakage*, o erro
metodológico nº 1 em ML.

**Teste:** `uv run --no-sync pytest tests/test_data_features.py tests/test_inference_persistence.py -v`

| Teste | O que prova |
|---|---|
| `test_mapas_usam_apenas_o_treino` | Produto que só existe na validação fica FORA do mapa |
| `test_valor_nao_visto_vira_unk` | Categoria desconhecida → índice UNK (0), nunca erro |
| `test_scaler_calculado_so_no_treino` | Média/desvio ignoram validação; std 0 vira 1 (sem divisão por zero) |
| `test_zscore_aplicado_em_float32` | Padronização correta e no dtype que o modelo espera |
| `test_mapas_sobrevivem_ao_roundtrip_json` | Salvar/carregar preserva chaves `int` (JSON as tornaria string → tudo viraria UNK silenciosamente) |

### 3. `src/data` — dataset iterável (out-of-core)

**O que faz:** o `TemporalParquetDataset` percorre as 47 partições **uma por
vez** (92,7M de linhas nunca ficam inteiras na RAM), aplica o pré-processamento
e fatia em batches de tensores prontos para o modelo. No treino, embaralha
partições e linhas de forma determinística (seed).

**Teste:** `uv run --no-sync pytest tests/test_data_features.py -v` (segunda metade)

| Teste | O que prova |
|---|---|
| `test_dataset_emite_batches_do_split_com_shapes_corretos` | Só linhas do split pedido; shapes/dtypes certos |
| `test_dataset_respeita_batch_size` | Nenhum batch estoura o tamanho configurado |
| `test_dataset_metadata_acompanha_o_batch` | Na avaliação, os IDs acompanham os tensores (para casar score com janela) |
| `test_embaralhamento_e_deterministico` | Mesmo seed → mesma ordem (reprodutibilidade do treino) |

### 4. `src/models` — arquitetura + Factory

**O que faz:** o `MLPRecommender` (embeddings de user/product/aisle/department
concatenados às 17 features → MLP 128→64→1 → logit de relevância) e a
**Factory** `build_model(nome, config)` — o design pattern do projeto: quem
treina não conhece a classe concreta, pede pelo nome.

**Teste:** `uv run --no-sync pytest tests/test_models.py -v`

| Teste | O que prova |
|---|---|
| `test_forward_retorna_um_logit_por_exemplo` | Contrato de saída: um score por candidato |
| `test_indice_unk_tem_embedding_nulo` | UNK tem vetor zero (`padding_idx`) — não injeta ruído |
| `test_construcao_e_deterministica_com_seed` | Mesma seed → mesmos pesos iniciais |
| `test_factory_constroi_mlp_temporal` / `_falha_alto` / `_estende` / `_duplicado` | O contrato da Factory: constrói, falha listando opções, aceita extensão, rejeita sobrescrita |

**Validação extra (fidelidade):** o checkpoint real treinado no notebook
carrega com `load_state_dict(strict=True)` — 5.310.657 parâmetros idênticos.

### 5. `src/training` — loop, early stopping e inferência

**O que faz:** `train_one_epoch` (uma passada de gradiente pelos batches),
`EarlyStopping` (para o treino quando a validação estagna por N épocas —
melhora *estrita* zera o contador), `seed_everything`/`get_device`
(reprodutibilidade e CPU/GPU), `save_checkpoint` (formato compatível com os
notebooks) e `predict_scores` (pontua um split para avaliação).

**Teste:** `uv run --no-sync pytest tests/test_training.py tests/test_inference_persistence.py -v`

| Teste | O que prova |
|---|---|
| `test_early_stopping_segue_semantica_do_notebook` | Empate NÃO é melhora; para após `patience` épocas estagnadas |
| `test_train_one_epoch_reduz_a_loss` | O loop aprende de verdade (loss cai em dados separáveis) |
| `test_seed_everything_torna_o_treino_deterministico` | Mesma seed → mesma loss (reprodutibilidade auditável) |
| `test_checkpoint_roundtrip` | O que se salva recarrega inteiro (`strict=True`) |
| `test_predict_scores_junta_metadata_e_score` | Score alinhado com os IDs da janela |
| `test_predict_scores_e_deterministico_em_eval` | `model.eval()` desliga o dropout na inferência |

### 6. `src/evaluation` — métricas de ranking

**O que faz:** dado um DataFrame (janela, produto, target, score), ordena cada
janela pelo score (desempate determinístico por `product_id` — a lição do bug
do notebook 03 virou regra) e calcula **NDCG@k** (qualidade da ordenação),
**hit rate@k** (acertou ≥1?), **recall local@k** (fração dos positivos
recuperada) e **precision@k**. O `RankingMetricsAccumulator` permite avaliar
partição a partição e consolidar no final.

**Teste:** `uv run --no-sync pytest tests/test_ranking_metrics.py -v`

| Teste | O que prova |
|---|---|
| `test_metricas_com_valores_conhecidos` | A matemática confere com cálculo manual (janelas pequenas) |
| `test_janela_sem_positivo_e_excluida` | Janelas inavaliáveis não distorcem o denominador |
| `test_desempate_deterministico_por_product_id` | Empates de score resolvem sempre igual, em qualquer máquina |
| `test_acumulador_particionado_equivale_a_avaliacao_unica` | Avaliar em chunks = avaliar tudo de uma vez |
| `test_colunas_ausentes_levantam_erro` | Entrada errada falha alto, não silenciosamente |
| `test_sem_janelas_elegiveis_retorna_nan` | Sem dados → NaN (não um zero que parece resultado) |

## O teste de integração definitivo: o pipeline

Os testes acima são **unitários** (rápidos, dados sintéticos). A prova de que
os módulos funcionam **juntos, com os dados reais**, é o pipeline DVC:

```bash
uv run --no-sync dvc repro      # preprocess → train → evaluate (~40 min c/ GPU)
uv run --no-sync dvc metrics show
```

Critério de aceite: `reports/metrics.json` com NDCG@10 validação = **0,499856**
e melhor época = 2 — os mesmos números da run `mlp_temporal_v1_catfix` dos
notebooks (já validado em 2026-07-04: reprodução exata).

## Como ler a saída do pytest

- `34 passed` verde = tudo certo.
- `FAILED tests/test_x.py::nome_do_teste` = leia o bloco do erro: o pytest
  mostra a linha do `assert` que falhou e os valores de cada lado.
- Docstrings dos testes dizem **a intenção** — se um teste falhar, a docstring
  responde "que garantia do sistema acabou de quebrar?".
