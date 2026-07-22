"""Testes do dataset temporal (`src.data.dataset`)."""

import pandas as pd
import pytest
import torch

from src.data import TemporalParquetDataset
from src.features import build_category_maps, compute_numeric_scaler_stats

MODEL_COLUMNS = [
    "split",
    "product_id",
    "f1",
    "f2",
    "target",
    "user_window_id",
]


@pytest.fixture
def dataset_kwargs(part_paths):
    """Cria os argumentos comuns do dataset temporal."""
    return {
        "part_paths": part_paths,
        "model_columns": MODEL_COLUMNS,
        "embedding_columns": ["product_id"],
        "numeric_columns": ["f1", "f2"],
        "category_maps": build_category_maps(
            part_paths,
            ["product_id"],
        ),
        "scaler_stats": compute_numeric_scaler_stats(
            part_paths,
            ["f1", "f2"],
        ),
        "batch_size": 2,
        "metadata_columns": [
            "user_window_id",
            "product_id",
            "target",
        ],
    }


def test_dataset_emite_batches_do_split_com_shapes_corretos(
    dataset_kwargs,
) -> None:
    """Emite somente as linhas do split com shapes e tipos corretos."""
    batches = list(
        TemporalParquetDataset(
            split="train",
            **dataset_kwargs,
        )
    )

    total_rows = sum(len(batch["target"]) for batch in batches)

    assert total_rows == 3

    first = batches[0]

    assert first["categorical"].dtype == torch.long
    assert first["categorical"].shape[1] == 1
    assert first["numeric"].dtype == torch.float32
    assert first["numeric"].shape[1] == 2


def test_dataset_respeita_batch_size(dataset_kwargs) -> None:
    """Nenhum batch excede o tamanho configurado."""
    batches = list(
        TemporalParquetDataset(
            split="train",
            **dataset_kwargs,
        )
    )

    assert all(len(batch["target"]) <= 2 for batch in batches)


def test_dataset_metadata_acompanha_o_batch(dataset_kwargs) -> None:
    """A metadata acompanha corretamente as linhas do batch."""
    batches = list(
        TemporalParquetDataset(
            split="validation",
            include_metadata=True,
            **dataset_kwargs,
        )
    )

    metadata = pd.concat(
        [batch["metadata"] for batch in batches]
    )

    assert sorted(metadata["user_window_id"]) == [
        "u1_v",
        "u2_v",
    ]
    assert list(metadata.columns) == [
        "user_window_id",
        "product_id",
        "target",
    ]


def test_embaralhamento_e_deterministico(dataset_kwargs) -> None:
    """A mesma seed produz os mesmos batches."""
    dataset = TemporalParquetDataset(
        split="train",
        shuffle_partitions=True,
        seed=7,
        **dataset_kwargs,
    )

    first_pass = [
        batch["categorical"] for batch in dataset
    ]
    second_pass = [
        batch["categorical"] for batch in dataset
    ]

    for tensor_a, tensor_b in zip(
        first_pass,
        second_pass,
        strict=True,
    ):
        assert torch.equal(tensor_a, tensor_b)