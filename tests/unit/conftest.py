"""Fixtures compartilhadas pelos testes unitários."""

import pandas as pd
import pytest


@pytest.fixture
def part_paths(tmp_path):
    """Cria duas partições sintéticas do dataset temporal.

    O produto 99 aparece apenas na validação para permitir a verificação
    de que os artefatos de preprocessing são ajustados somente no treino.
    """
    part_a = pd.DataFrame(
        {
            "split": ["train", "train", "validation"],
            "product_id": [10, 20, 99],
            "f1": [1.0, 3.0, 100.0],
            "f2": [5.0, 5.0, 100.0],
            "target": [1, 0, 1],
            "user_window_id": ["u1_t", "u1_t", "u1_v"],
        }
    )
    part_b = pd.DataFrame(
        {
            "split": ["train", "validation"],
            "product_id": [30, 10],
            "f1": [5.0, 2.0],
            "f2": [5.0, 7.0],
            "target": [0, 1],
            "user_window_id": ["u2_t", "u2_v"],
        }
    )

    paths = []

    for name, frame in [
        ("part-0000.parquet", part_a),
        ("part-0001.parquet", part_b),
    ]:
        path = tmp_path / name
        frame.to_parquet(path, index=False)
        paths.append(path)

    return paths
