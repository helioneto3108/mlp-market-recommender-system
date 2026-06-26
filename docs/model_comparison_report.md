# Relatório de Modelos e Comparação Final
## Sistema de Recomendação de Produtos — Instacart

> **Para quem é este documento:** o time técnico que precisa entender **quais
> modelos** treinamos, **como** os comparamos de forma justa e **qual venceu**; e
> o time de negócio que quer a conclusão de **qual abordagem recomenda melhor** —
> sem ler código.
>
> Gerado a partir de [`notebooks/06_models_comparison.ipynb`](../notebooks/06_models_comparison.ipynb).

---

## 🎯 Resumo executivo

- Comparamos **quatro modelos** sob o **mesmo protocolo** e nos **mesmos 26.241
  usuários held-out** (nunca vistos no treino) — uma régua justa.
- A **MLP com features** (rede neural) é a **vencedora** em todas as métricas,
  superando o forte baseline de histórico em **+10,5%** de NDCG@10.
- O **NCF de IDs puros** fica ≈ popularidade: confirma, com dados, que
  *collaborative filtering* sozinho **não captura frequência/recência**.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| Popularidade | 0,169 | 0,214 | 0,237 | 0,133 |
| Histórico (frequência) | 0,267 | 0,320 | 0,388 | 0,263 |
| NCF (apenas IDs) | 0,175 | 0,222 | 0,244 | 0,139 |
| **MLP com features** | **0,298** | **0,353** | **0,428** | **0,300** |

---

## 1. O protocolo de avaliação

> 👔 **Visão de negócio:** para cada cliente, cada modelo monta um *ranking* dos
> produtos que ele costuma comprar; comparamos os 10 primeiros com o que ele
> realmente levou no próximo pedido. Todos os modelos são julgados pela mesma
> régua e nos mesmos clientes.

> 🔬 **Visão técnica:** protocolo de **re-ranking de candidatos** — cada modelo
> atribui um *score* aos candidatos (produtos do histórico do usuário) e
> ranqueamos. Por ser mais focado que varrer o catálogo inteiro, os números aqui
> são **mais altos** que os do `notebook 04` (full-catalog). O que importa é a
> **comparação relativa**: todos sob a mesma régua, nos mesmos usuários held-out.

A função de avaliação aceita **qualquer** vetor de scores — princípio
**aberto/fechado** (SOLID): novos modelos entram sem alterar a avaliação.

---

## 2. Os quatro modelos

| Modelo | O que usa | Papel |
|---|---|---|
| **Popularidade** | popularidade global do produto | piso não-personalizado |
| **Histórico (freq)** | nº de compras do usuário | baseline forte (recompra) |
| **NCF (IDs)** | embeddings de usuário e produto | *collaborative filtering* puro |
| **MLP (features)** | embeddings de produto/aisle/dept + features | modelo proposto |

Os dois neurais (PyTorch) foram treinados na **GPU**, com **early stopping** na
MLP (parou na época 5, pico na 2 — evitando overfitting).

---

## 3. Resultado

![Comparação de modelos](img/model/01_model_comparison.png)

**Leitura:**

- **A MLP com features vence em tudo** (NDCG@10 0,428). O diferencial são as
  features que os baselines não têm — sobretudo a **recência** — e os embeddings
  de categoria.
- **O NCF de IDs puros (0,244) fica ≈ popularidade (0,237).** Apesar de a sua
  *loss* de treino cair bem (0,23 → 0,15), o ranking quase não melhora: o modelo
  só de identidades **não distingue** os itens por frequência/recência, então
  aproxima a popularidade. É a confirmação empírica da motivação do
  `notebook 05`.
- **O histórico (0,388) continua um baseline difícil** — muito acima da
  popularidade. Superá-lo de forma consistente é o que valida a rede neural.

---

## 4. Por que a MLP vence (e o NCF não)

> 🔬 Ambos são redes neurais com embeddings. A diferença é a **informação de
> entrada**: o NCF recebe só dois IDs e não pode saber *quantas vezes* nem *há
> quanto tempo* o usuário comprou o item (no treino, todos os itens comprados têm
> rótulo 1). A MLP recebe essas features explicitamente e aprende a combiná-las
> com a similaridade de produtos (embeddings de aisle/department). Não é "rede
> neural vs heurística" — é **representação rica vs representação pobre**.

---

## 5. Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `models/feature_mlp.pt` | pesos da MLP vencedora |
| `models/ncf.pt` | pesos do NCF (modelo de comparação) |
| `models/feature_mlp_scaler.json` | média/desvio das features (para servir o modelo) |
| `data/processed/model_comparison.parquet` | tabela de métricas (4 modelos × K) |
| `data/processed/feature_mlp_history.parquet` | curva de treino (early stopping) |

---

## 6. Próximos passos

- **MLflow + Registry:** rastrear estes runs (≥3) e promover a **MLP** a
  *Production*.
- **Model Card:** documentar performance, limitações (não recomenda itens novos
  fora do histórico) e vieses (popularidade).
- **Refatorar para `src/`:** mover o código validado dos notebooks para módulos
  (Factory de modelos, Strategy de preprocessors), com testes.

---

<sub>Números e gráfico derivados de [`notebooks/06_models_comparison.ipynb`](../notebooks/06_models_comparison.ipynb)
e dos artefatos em `data/processed/` e `models/`.</sub>
