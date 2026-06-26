# Relatório de Feature Engineering
## Sistema de Recomendação de Produtos — Instacart

> **Para quem é este documento:** o time técnico que precisa entender **quais
> features** alimentam o modelo neural e **como** evitamos vazamento de dados; e
> o time de negócio que quer a intuição de **por que** essas variáveis preveem
> recompra — sem ler código.
>
> Gerado a partir de [`notebooks/05_feature_engineering.ipynb`](../notebooks/05_feature_engineering.ipynb).

---

## 🎯 Resumo executivo

- Mudamos o enquadramento do problema: de "feedback implícito + *negative
  sampling*" para **classificação supervisionada de recompra**. Para cada produto
  que o cliente já comprou (*candidato*), prevemos se ele voltará no próximo
  pedido.
- A motivação: um modelo só de IDs não enxerga **frequência** nem **recência** —
  sinais centrais da recompra. A comparação direta entre as duas abordagens (NCF
  de IDs vs MLP com features) é feita no notebook 06.
- Construímos **8,47 milhões de candidatos rotulados** (9,8% positivos) com
  features de frequência, recência, taxas de reorder e popularidade.
- Split **por usuário** (80% treino / 20% avaliação) → **sem vazamento**.

| Indicador | Valor |
|---|---|
| Candidatos (pares usuário–produto) | **8.474.661** |
| Taxa de positivos (recompra real) | 9,8% |
| Features numéricas | 10 |
| Embeddings categóricos | produto, aisle, department |
| Usuários de avaliação (held-out) | 26.241 |

---

## 1. Por que trocamos a abordagem

> 👔 **Visão de negócio:** o modelo só de "identidades" (quem é o cliente, qual
> o produto) não sabia *com que frequência* nem *há quanto tempo* a pessoa compra
> cada item — e é justamente isso que prevê uma recompra. Então passamos a
> **descrever** cada par cliente-produto com números que capturam esse
> comportamento.

> 🔬 **Visão técnica:** um NCF pointwise só de IDs não distingue os itens por
> frequência — todos os itens comprados entram como rótulo 1 —, tendendo a
> aproximar a popularidade. Reformulamos como classificação binária supervisionada
> de candidatos: rótulo vindo do pedido `train`, *features* derivadas do histórico
> `prior`. Negativos passam a ser **reais** (candidatos não recomprados), o que
> elimina o *negative sampling*. O notebook 06 mede empiricamente as duas
> abordagens lado a lado.

---

## 2. As features

| Grupo | Feature | O que mede |
|---|---|---|
| Usuário-produto | `n_orders` | nº de vezes que o usuário comprou o item |
| | `up_order_share` | fração dos pedidos do usuário com o item (frequência norm.) |
| | `up_reorder_rate` | recompras / compras do par |
| | **`up_orders_since_last`** ⭐ | pedidos desde a última compra (recência) |
| | `up_active_span` | janela de pedidos em que o item esteve ativo |
| | `up_purchase_rate` | compras por pedido dentro da janela |
| Usuário | `u_n_products` | variedade de produtos do usuário |
| | `u_reorder_rate` | propensão geral do usuário a recomprar |
| Produto | `p_n_users` | popularidade (nº de clientes) |
| | `p_reorder_rate` | propensão global do produto a ser recomprado |
| Categórico | `aisle_id`, `department_id` | seção/categoria → **embeddings** |

> 🔑 **A estrela é a recência (`up_orders_since_last`).** É o sinal que os
> baselines **não têm** — e o que permite ao modelo neural superá-los.

---

## 3. A recência prevê recompra?

![Recompra vs recência](img/features/01_recency_vs_reorder.png)

A curva é **fortemente decrescente**: um item comprado no pedido mais recente
(recência = 0) tem probabilidade muito maior de voltar do que um comprado há 10+
pedidos. Isso confirma, com dados, por que a recência é uma feature poderosa.

---

## 4. Embeddings de categoria

Além do **embedding de produto**, incluímos `aisle_id` (134 seções) e
`department_id` (21 departamentos) como embeddings.

> 🔬 **Por quê?** Eles dão ao modelo uma noção de **similaridade entre produtos**
> (dois iogurtes diferentes ficam próximos no espaço de embeddings da mesma
> seção), ajudando inclusive produtos com pouco histórico individual.

---

## 5. Split sem vazamento (decisão de design)

> 🔑 **Sem embedding de ID de usuário.** Avaliamos em usuários *held-out* (nunca
> vistos no treino). Um embedding de ID de usuário seria um vetor aleatório para
> eles (*cold-start*) — inútil. Por isso representamos o usuário por **features
> agregadas** e mantemos apenas embeddings que **generalizam** (produto, aisle,
> department). É a escolha que generaliza para usuários novos.

Separar os **usuários** (não as linhas) em treino/avaliação garante que o modelo
**nunca vê** o próximo pedido dos usuários de teste — não há vazamento do alvo.

---

## 6. Implicações para a modelagem

- **Insumo pronto:** `candidate_features.parquet` (8,47M linhas) com `split`
  embutido, 10 features numéricas e 3 categóricas.
- **Tarefa:** classificação binária `P(recompra | features, embeddings)`; o
  ranking dos candidatos por essa probabilidade gera as recomendações.
- **Avaliação:** mesma régua do notebook 04 (Precision/Recall/NDCG/MAP @K), agora
  no protocolo de **re-ranking de candidatos**, nos usuários held-out.

**Próximo passo (`notebook 06`):** treinar a **MLP** (embeddings + features) com
*early stopping* e comparar com os baselines — onde o modelo neural finalmente
vence.

---

<sub>Números e gráfico derivados de [`notebooks/05_feature_engineering.ipynb`](../notebooks/05_feature_engineering.ipynb)
e do artefato `data/processed/candidate_features.parquet`.</sub>
