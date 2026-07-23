"""Ranking de popularidade — fallback de cold start.

Usuários sem histórico de compras não entram no funil personalizado
(candidatos → MLP): não têm features nem candidatos. A estratégia de
degradação graciosa é servir a *vitrine de popularidade*: os produtos
comprados pelo maior número de usuários distintos, contados apenas no
split de treino (mesma disciplina anti-leakage do restante do pipeline).
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

PopularityRanking = list[dict[str, int]]


def compute_popularity_ranking(
    part_paths: Iterable[Path], top_n: int
) -> PopularityRanking:
    """Calcula o top-N de produtos por nº de usuários distintos compradores.

    Considera apenas linhas do split de treino cujo produto já foi comprado
    pelo usuário (``user_product_purchase_count > 0``); cada par
    (usuário, produto) conta uma única vez, mesmo aparecendo em várias
    janelas. Empates são desfeitos por ``product_id`` (determinismo).

    Args:
        part_paths: Partições parquet do dataset temporal.
        top_n: Tamanho do ranking.

    Returns:
        Lista ordenada de ``{"product_id": ..., "n_users": ...}``.
    """
    pairs = []
    columns = ["split", "user_id", "product_id", "user_product_purchase_count"]
    for part_path in part_paths:
        part = pd.read_parquet(part_path, columns=columns)
        bought = part[
            (part["split"] == "train") & (part["user_product_purchase_count"] > 0)
        ]
        pairs.append(bought[["user_id", "product_id"]].drop_duplicates())
    unique_pairs = pd.concat(pairs, ignore_index=True).drop_duplicates()
    ranking = (
        unique_pairs.groupby("product_id")
        .size()
        .reset_index(name="n_users")
        .sort_values(
            ["n_users", "product_id"], ascending=[False, True], kind="mergesort"
        )
        .head(top_n)
    )

    return [
        {"product_id": int(row.product_id), "n_users": int(row.n_users)}
        for row in ranking.itertuples()
    ]
