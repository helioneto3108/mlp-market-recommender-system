"""Testes da inferência (`predict_scores`) e persistência dos mapas."""

import pandas as pd
import torch

from src.features import build_category_maps, load_category_maps
from src.models import MLPRecommender
from src.training import predict_scores
from src.utils import save_json

TINY_CONFIG = {
    "embedding_cardinalities": {"product_id": 8},
    "embedding_columns": ["product_id"],
    "embedding_dim": 4,
    "numeric_input_dim": 2,
    "hidden_dims": [8],
    "dropout": 0.0,
}


def build_batch(products: list[int]) -> dict:
    """Batch sintético com metadata, no formato do TemporalParquetDataset."""
    size = len(products)
    return {
        "categorical": torch.tensor([[p] for p in products], dtype=torch.long),
        "numeric": torch.zeros(size, 2),
        "metadata": pd.DataFrame(
            {
                "user_window_id": [f"w{i}" for i in range(size)],
                "product_id": products,
                "target": [0] * size,
            }
        ),
    }


def test_predict_scores_junta_metadata_e_score() -> None:
    """A saída preserva as colunas de metadata e adiciona `score` alinhado."""
    torch.manual_seed(0)
    model = MLPRecommender(**TINY_CONFIG)
    batches = [build_batch([1, 2]), build_batch([3])]

    predictions = predict_scores(model, batches, torch.device("cpu"))

    assert len(predictions) == 3
    assert list(predictions.columns) == [
        "user_window_id",
        "product_id",
        "target",
        "score",
    ]
    assert predictions["score"].dtype.kind == "f"


def test_predict_scores_e_deterministico_em_eval() -> None:
    """Duas passadas do mesmo modelo produzem os mesmos scores (dropout off)."""
    torch.manual_seed(0)
    model = MLPRecommender(**{**TINY_CONFIG, "dropout": 0.5})
    batches = [build_batch([1, 2, 3])]

    first = predict_scores(model, batches, torch.device("cpu"))
    second = predict_scores(model, batches, torch.device("cpu"))

    assert first["score"].tolist() == second["score"].tolist()


def test_mapas_sobrevivem_ao_roundtrip_json(tmp_path) -> None:
    """save/load preserva as chaves inteiras (JSON as converteria p/ string)."""
    frame = pd.DataFrame({"split": ["train", "train"], "product_id": [10, 20]})
    parquet = tmp_path / "part-0000.parquet"
    frame.to_parquet(parquet, index=False)
    original = build_category_maps([parquet], ["product_id"])

    path = tmp_path / "maps.json"
    save_json(original, path)
    reloaded = load_category_maps(path)

    assert reloaded == original
    assert all(isinstance(key, int) for key in reloaded["product_id"])
