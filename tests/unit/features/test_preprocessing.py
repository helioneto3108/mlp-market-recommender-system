"""Testes do preprocessing de features (`src.features.preprocessing`)."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_category_maps,
    compute_numeric_scaler_stats,
    load_category_maps,
    map_categorical_columns,
    scale_numeric_columns,
)
from src.models import UNK_INDEX
from src.utils import save_json


def test_mapas_usam_apenas_o_treino(part_paths) -> None:
    """Valores exclusivos de validação (99) ficam fora do mapa (anti-leakage)."""
    maps = build_category_maps(part_paths, ["product_id"])

    assert sorted(maps["product_id"]) == [10, 20, 30]
    assert sorted(maps["product_id"].values()) == [1, 2, 3]  # 0 é reservado ao UNK


def test_valor_nao_visto_vira_unk(part_paths) -> None:
    """Categoria fora do treino é mapeada para o índice UNK do modelo."""
    maps = build_category_maps(part_paths, ["product_id"])
    frame = pd.DataFrame({"product_id": [10, 99]})

    mapped = map_categorical_columns(frame, maps, ["product_id"])

    assert mapped["product_id"].tolist() == [maps["product_id"][10], UNK_INDEX]


def test_scaler_calculado_so_no_treino(part_paths) -> None:
    """Média/desvio consideram apenas linhas de treino; std 0 vira 1."""
    stats = compute_numeric_scaler_stats(part_paths, ["f1", "f2"])

    train_f1 = np.array([1.0, 3.0, 5.0])  # linhas de treino das 2 partições
    assert stats["mean"]["f1"] == pytest.approx(train_f1.mean())
    assert stats["std"]["f1"] == pytest.approx(train_f1.std())
    assert stats["std"]["f2"] == 1.0  # f2 é constante no treino → evita divisão por 0


def test_zscore_aplicado_em_float32(part_paths) -> None:
    """A padronização usa as estatísticas do treino e devolve float32."""
    stats = compute_numeric_scaler_stats(part_paths, ["f1", "f2"])
    frame = pd.DataFrame({"f1": [3.0], "f2": [6.0]})

    scaled = scale_numeric_columns(frame, ["f1", "f2"], stats)

    assert scaled["f1"].iloc[0] == pytest.approx(0.0)  # 3.0 é a média do treino
    assert scaled["f2"].iloc[0] == pytest.approx(1.0)  # (6-5)/1
    assert scaled.dtypes.unique().tolist() == [np.dtype("float32")]


def test_mapas_sobrevivem_ao_roundtrip_json(tmp_path) -> None:
    """A persistência dos mapas preserva as chaves inteiras."""
    frame = pd.DataFrame(
        {
            "split": ["train", "train"],
            "product_id": [10, 20],
        }
    )
    parquet_path = tmp_path / "part-0000.parquet"
    frame.to_parquet(parquet_path, index=False)

    original = build_category_maps(
        [parquet_path],
        ["product_id"],
    )

    output_path = tmp_path / "maps.json"
    save_json(original, output_path)

    reloaded = load_category_maps(output_path)

    assert reloaded == original
    assert all(isinstance(key, int) for key in reloaded["product_id"])
