# Revisão e integração dos notebooks — branch `feat/integration`

> Registro das alterações feitas nos notebooks originais da branch
> `feature/helio-notebooks` durante a integração (2026-07-03), com o motivo de
> cada mudança. Serve como memória técnica do projeto e insumo para o vídeo
> STAR e o Model Card.

## Contexto

O time decidiu adotar o pipeline temporal desta branch como base do projeto
(desenho de janelas temporais por usuário, geração de candidatos em 4 fontes e
MLP ranker). Antes de construir em cima, fizemos uma revisão completa:
análise de código, verificação de convenções, e **re-execução integral dos 6
notebooks em outra máquina** (Windows/x86 + RTX 5080, vs. o macOS/ARM
original) para validar a reprodutibilidade de ponta a ponta.

Essa re-execução cross-platform foi o que revelou o achado mais importante da
revisão (seção "Bug de reprodutibilidade", abaixo).

## Resumo das alterações

| Notebook | Alteração | Tipo |
|---|---|---|
| todos (6) | Correções automáticas do ruff + formatação | qualidade |
| 03 | `zip(..., strict=True)` no mapa de vizinhos Jaccard | robustez |
| 03 | **Desempate determinístico no `build_category_pool`** (`kind="mergesort"`) | **bug fix** |
| 03 | `OVERWRITE_TEMPORAL_CANDIDATES = True` | operacional |
| 04 | Remoção de 2 células mortas (`build_order_timeline`, `summarize_window`) | limpeza |
| 05 | Persistência de `baseline_results.parquet` para consumo downstream | arquitetura |
| 05 | Remoção do parâmetro morto `baseline_name` em `evaluate_by_segment` | limpeza |
| 05 | `RUN_MLFLOW = False` como default seguro | operacional |
| 06 | Baseline lido do artefato do nb05 (fim dos valores hardcoded) | **arquitetura** |
| 06 | Célula final ("Decisão final e limitações") completada | documentação |
| 06 | Run de MLflow renomeada (`mlp_temporal_v1_catfix`) | rastreabilidade |

---

## 1. Qualidade de código (todos os notebooks)

**O que:** `ruff check --fix` + `ruff format` nos 6 notebooks — 25 apontamentos
corrigidos: linhas em branco com espaços (W293), imports não usados (F401:
`numpy` no nb04, `pyarrow.compute`/`pyarrow.parquet` nos nb05/06), f-string sem
placeholder (F541), casts `int()` redundantes (RUF046).

**Por quê:** o projeto exige "ruff sem erros" como critério de clean code.
Imports não usados são o caso mais didático: eles mentem para o leitor sobre as
dependências reais do código. Nenhuma dessas correções altera comportamento —
por isso puderam ser aplicadas automaticamente sem risco.

## 2. `zip(strict=True)` no nb03

**O que:** no dicionário de vizinhos Jaccard, o `zip()` que pareia as colunas
`neighbor_user_id` e `jaccard_score` ganhou `strict=True`.

**Por quê:** as duas colunas vêm do mesmo DataFrame, então **hoje** têm sempre o
mesmo tamanho. O `strict=True` protege o **futuro**: se uma refatoração quebrar
esse pareamento, o código levanta `ValueError` na hora, em vez de truncar
silenciosamente o lado maior — o pior tipo de bug, o que não faz barulho.
É o princípio *fail fast*: erros devem aparecer o mais perto possível da causa.

## 3. 🐛 Bug de reprodutibilidade no `build_category_pool` (nb03) — o achado principal

### Como foi descoberto

Após regenerar o dataset completo nesta máquina, o `dvc status` acusou o
`temporal_modeling_dataset_v1` como **modificado** — o md5 não batia com o
versionado, gerado no macOS. Um pipeline 100% determinístico deveria produzir
bytes idênticos. A investigação comparou as 47 partições regeneradas contra as
originais do cache DVC, linha a linha:

- **96.518 candidatos trocados** de ~92,7 milhões (0,10%), em trocas simétricas
  (mesmas janelas, mesma quantidade, produtos diferentes);
- todos da fonte **`category`** (efeitos residuais na `global` por cascata);
- `recompra` e `similarity`: **zero divergência**;
- efeito líquido: 3 positivos a menos (3.454.841 vs 3.454.844) e métricas
  movendo na 6ª casa decimal (ex.: NDCG@10 do melhor baseline 0,467765 vs
  0,467768).

### A causa

```python
top_aisles = (
    history_df.groupby("aisle_id")
    .size()
    .sort_values(ascending=False)   # ← empates sem critério de desempate
    .head(CATEGORY_TOP_AISLES)
)
```

Quando dois aisles têm a **mesma contagem** no histórico do usuário (muito
comum em históricos curtos), quem decide a ordem entre eles é o algoritmo de
ordenação. O default do pandas (`quicksort`) **não é estável**: a ordem
relativa de elementos empatados não é garantida e varia entre versões de
numpy/plataformas. No corte do `head(N)`, aisles empatados na fronteira
entravam no pool numa máquina e ficavam de fora na outra — arrastando pools
de produtos inteiros.

O detalhe irônico: o restante do notebook faz desempate explícito em todos os
rankings (por `product_id`, `aisle_id` etc.). Escapou só este ponto — e nenhum
teste na máquina do autor detectaria, porque **dentro de uma mesma máquina o
resultado é estável**. Só a reprodução em outra plataforma expôs o problema.

### A correção

```python
.sort_values(ascending=False, kind="mergesort")
```

O `groupby` já entrega o índice ordenado por `aisle_id`; como o *mergesort* é
**estável**, empates de contagem preservam essa ordem — ou seja, o desempate
passa a ser "menor `aisle_id` primeiro", determinístico em qualquer máquina.

### As lições (para o vídeo STAR)

1. **Versionamento de dados não é burocracia**: foi o hash md5 do DVC que
   denunciou o problema. Sem DVC, essa divergência passaria invisível.
2. **Reprodutibilidade se testa em outra máquina**, não na do autor.
3. **Ordenação com empates precisa de critério de desempate explícito** — em
   qualquer pipeline que corta top-N.

### Consequência: novo dataset canônico

Com a correção, o pipeline foi re-executado do nb03 ao nb06 (2026-07-03,
~70 min de cadeia) e o dataset recebeu novo md5 no DVC:

- **md5 anterior** (gerado no macOS, ordem de empates não determinística):
  `8a10ede5efe3d492a1a07aed9977dd69.dir`
- **md5 novo (canônico)**: `0f3e051a218992ccdf587871fa284517.dir`
  (47 partições, 604.676.028 bytes)

A versão anterior permanece recuperável pelo histórico git do arquivo `.dvc`.
Como previsto, as métricas mudaram apenas na margem e as conclusões não mudam:

| Métrica (validação, k=10) | Antes (dataset original) | Depois (dataset corrigido) |
|---|---|---|
| Melhor baseline (NDCG) | 0,467768 | 0,467773 |
| MLP — NDCG | 0,499809 | **0,499856** |
| MLP — ganho vs baseline | +0,032041 | **+0,032083** |
| MLP — NDCG teste | 0,494838 | 0,494804 |
| Melhor época | 3 | 2 |

## 4. Células mortas removidas (nb04)

**O que:** `build_order_timeline` e `summarize_window` — versões por-janela
substituídas pela implementação vetorizada, mas nunca removidas.

**Por quê:** código morto tem custo real: o leitor gasta energia entendendo
funções que não participam do resultado, e uma manutenção futura pode "corrigir"
a função errada. Clean code trata remoção como melhoria, não como perda — o git
preserva a história se algum dia for preciso resgatar.

## 5. Persistência dos resultados dos baselines (nb05) + fim do hardcode (nb06)

**O que:** o nb05 agora salva `data/processed/baseline_results.parquet` com
todas as métricas (baseline × split × k). O nb06, que **copiava à mão** as
métricas do melhor baseline num dicionário hardcoded, passou a ler esse
arquivo e selecionar o melhor baseline programaticamente.

**Por quê:** valores copiados entre notebooks criam uma dependência invisível:
se o nb05 mudar (como mudou agora, com o dataset corrigido), o nb06 continuaria
comparando o MLP contra números velhos — e a tabela final **mentiria**. Com o
artefato em disco há uma única fonte de verdade e o acoplamento fica explícito
(o nb06 falha alto se o nb05 não tiver rodado). É o mesmo princípio de DRY
aplicado a dados: *don't repeat numbers*.

**Bônus:** `RUN_MLFLOW` do nb05 ficou `False` por default — re-executar o
notebook não cria runs duplicadas no MLflow do time por acidente; o logging
passa a ser uma ação consciente.

## 6. Célula final do nb06 completada

**O que:** o markdown "Decisão final e limitações" terminava no meio de uma
frase. Foi completado com a conclusão (por que o MLP vale a pena) e as
limitações explícitas: ganho incremental (+0,032 NDCG@10), teto de recall
(~70%) imposto pela geração de candidatos, ausência de avaliação de cold
start, e o experimento com/sem `candidate_rank` como trabalho futuro.

**Por quê:** limitações documentadas são requisito do Model Card e material de
avaliação honesta do projeto. Um documento truncado no repositório também
passa impressão de trabalho inacabado.

## 7. Runs de MLflow da integração (rastreabilidade)

| Run | O que registra |
|---|---|
| `mlp_temporal_v1_base` (original) | Treino original (macOS/mps), NDCG@10 val 0,499809 |
| `mlp_temporal_v1_rev` | **Reprodução** do treino em Windows/cuda sobre o dataset original: NDCG@10 val 0,499800 (Δ 9×10⁻⁶) — valida seeds, DVC e pipeline |
| `mlp_temporal_v1_catfix` | Treino sobre o **dataset corrigido** (novo canônico): NDCG@10 val 0,499856 / test 0,494804 — run `c80d4dd3` no experimento 5 |

**Por quê:** cada run documenta um estado (código + dados + hardware) — juntas
contam a história: original → reproduzido → corrigido. Isso também compõe o
requisito de ≥ 3 runs rastreadas no MLflow.

## Pendências conhecidas (próximos passos)

- **Decisão (2026-07-04):** o time optou por **não** adicionar um baseline
  treinado com scikit-learn (regressão logística); os 4 baselines heurísticos
  do notebook 05 permanecem como régua de comparação. Justificar no Model
  Card/vídeo (heurísticas de domínio como comparação principal).
- **MLflow Model Registry**: registrar o MLP e promover a Production — será
  feito quando o servidor de tracking definitivo existir (MLflow no
  docker-compose da Etapa 3; migração para GCP planejada na sequência).
- Refatoração dos notebooks para `src/` com type hints, docstrings Google
  style e design patterns (Factory/Strategy) — **próxima fase do projeto**.
- Padronizar nomes de experimentos no MLflow e acentuação dos markdowns.
