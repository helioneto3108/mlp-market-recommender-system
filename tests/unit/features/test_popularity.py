"""Testes do ranking de popularidade (`src.features.popularity`)."""

import pandas as pd

from src.features import compute_popularity_ranking


def build_partition(tmp_path, name: str, rows: list[tuple]) -> object:
    """Grava uma partição sintética (split, user, product, purchase_count)."""
    frame = pd.DataFrame(
        rows, columns=["split", "user_id", "product_id", "user_product_purchase_count"]
    )
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


def test_ranking_conta_usuarios_distintos_uma_vez(tmp_path) -> None:
    """O mesmo usuário em várias janelas conta uma única vez por produto."""
    path = build_partition(
        tmp_path,
        "part-0000.parquet",
        [
            ("train", 1, 50, 3),  # usuário 1 comprou o produto 50...
            ("train", 1, 50, 5),  # ...aparece em 2 janelas: conta 1 vez
            ("train", 2, 50, 1),
            ("train", 1, 60, 2),
        ],
    )

    ranking = compute_popularity_ranking([path], top_n=10)

    assert ranking[0] == {"product_id": 50, "n_users": 2}
    assert ranking[1] == {"product_id": 60, "n_users": 1}


def test_ranking_ignora_validacao_e_nao_comprados(tmp_path) -> None:
    """Linhas de outros splits e candidatos nunca comprados ficam de fora."""
    path = build_partition(
        tmp_path,
        "part-0000.parquet",
        [
            ("train", 1, 50, 2),
            ("validation", 2, 50, 9),  # split errado: fora
            ("train", 3, 70, 0),  # candidato nunca comprado: fora
        ],
    )

    ranking = compute_popularity_ranking([path], top_n=10)

    assert ranking == [{"product_id": 50, "n_users": 1}]


def test_empate_desempata_por_product_id(tmp_path) -> None:
    """Produtos com a mesma contagem saem em ordem determinística de ID."""
    path = build_partition(
        tmp_path,
        "part-0000.parquet",
        [("train", 1, 90, 1), ("train", 1, 20, 1), ("train", 1, 55, 1)],
    )

    ranking = compute_popularity_ranking([path], top_n=2)

    assert [item["product_id"] for item in ranking] == [20, 55]  # empate → menor ID
