"""Testes do pré-processamento (`src.features`) e do dataset (`src.data`)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import TemporalParquetDataset
from src.features import (
    build_category_maps,
    compute_numeric_scaler_stats,
    map_categorical_columns,
    scale_numeric_columns,
)
from src.models import UNK_INDEX

MODEL_COLUMNS = ["split", "product_id", "f1", "f2", "target", "user_window_id"]


@pytest.fixture
def part_paths(tmp_path):
    """Duas partições sintéticas; o product_id 99 só aparece na validação."""
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
    for name, frame in [("part-0000.parquet", part_a), ("part-0001.parquet", part_b)]:
        path = tmp_path / name
        frame.to_parquet(path, index=False)
        paths.append(path)
    return paths


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


@pytest.fixture
def dataset_kwargs(part_paths):
    """Argumentos comuns do TemporalParquetDataset para os testes."""
    return {
        "part_paths": part_paths,
        "model_columns": MODEL_COLUMNS,
        "embedding_columns": ["product_id"],
        "numeric_columns": ["f1", "f2"],
        "category_maps": build_category_maps(part_paths, ["product_id"]),
        "scaler_stats": compute_numeric_scaler_stats(part_paths, ["f1", "f2"]),
        "batch_size": 2,
        "metadata_columns": ["user_window_id", "product_id", "target"],
    }


def test_dataset_emite_batches_do_split_com_shapes_corretos(dataset_kwargs) -> None:
    """Só linhas do split pedido; tensores com dtypes e shapes esperados."""
    batches = list(TemporalParquetDataset(split="train", **dataset_kwargs))

    total_rows = sum(len(batch["target"]) for batch in batches)
    assert total_rows == 3  # linhas de treino nas 2 partições
    first = batches[0]
    assert first["categorical"].dtype == torch.long
    assert first["categorical"].shape[1] == 1
    assert first["numeric"].dtype == torch.float32
    assert first["numeric"].shape[1] == 2


def test_dataset_respeita_batch_size(dataset_kwargs) -> None:
    """Nenhum batch excede o tamanho configurado."""
    batches = list(TemporalParquetDataset(split="train", **dataset_kwargs))

    assert all(len(batch["target"]) <= 2 for batch in batches)


def test_dataset_metadata_acompanha_o_batch(dataset_kwargs) -> None:
    """Com include_metadata, o DataFrame acompanha as linhas do batch."""
    batches = list(
        TemporalParquetDataset(
            split="validation", include_metadata=True, **dataset_kwargs
        )
    )

    metadata = pd.concat([batch["metadata"] for batch in batches])
    assert sorted(metadata["user_window_id"]) == ["u1_v", "u2_v"]
    assert list(metadata.columns) == ["user_window_id", "product_id", "target"]


def test_embaralhamento_e_deterministico(dataset_kwargs) -> None:
    """Duas iterações com a mesma seed produzem os mesmos batches."""
    dataset = TemporalParquetDataset(
        split="train", shuffle_partitions=True, seed=7, **dataset_kwargs
    )

    first_pass = [batch["categorical"] for batch in dataset]
    second_pass = [batch["categorical"] for batch in dataset]

    for tensor_a, tensor_b in zip(first_pass, second_pass, strict=True):
        assert torch.equal(tensor_a, tensor_b)
