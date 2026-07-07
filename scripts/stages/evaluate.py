"""Stage `evaluate`: avalia o checkpoint treinado em validação e teste.

Gera `reports/metrics.json` no formato de métricas do DVC (`dvc metrics
show`), com NDCG, hit rate, recall local e precision por split e por k.
"""

import json
from pathlib import Path

import torch
import yaml

from src.data import TemporalParquetDataset
from src.evaluation import evaluate_ranking_scores
from src.features import load_category_maps
from src.models import build_model
from src.training import get_device, predict_scores, seed_everything

EVALUATION_METADATA = ("user_window_id", "product_id", "target")


def load_params() -> tuple[dict, dict, dict]:
    """Carrega as seções `preprocess`, `train` e `evaluate` do params.yaml."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    return params["preprocess"], params["train"], params["evaluate"]


def build_split_dataset(
    split: str, preprocess_params: dict, train_params: dict
) -> TemporalParquetDataset:
    """Monta o dataset de avaliação de um split, com metadata."""
    preprocessing_dir = Path(preprocess_params["output_dir"])
    embedding_columns = preprocess_params["embedding_columns"]
    numeric_columns = preprocess_params["numeric_feature_columns"]
    return TemporalParquetDataset(
        part_paths=sorted(
            Path(preprocess_params["dataset_dir"]).glob("part-*.parquet")
        ),
        split=split,
        model_columns=[
            "split",
            "user_window_id",
            *embedding_columns,
            *numeric_columns,
            preprocess_params["target_column"],
        ],
        embedding_columns=embedding_columns,
        numeric_columns=numeric_columns,
        category_maps=load_category_maps(preprocessing_dir / "category_maps.json"),
        scaler_stats=json.loads(
            (preprocessing_dir / "scaler_stats.json").read_text(encoding="utf-8")
        ),
        batch_size=train_params["batch_size"],
        target_column=preprocess_params["target_column"],
        include_metadata=True,
        metadata_columns=EVALUATION_METADATA,
    )


def metrics_as_dict(metrics_frame) -> dict:
    """Converte o DataFrame de métricas em dict achatado (`ndcg_at_10`, ...)."""
    flat = {}
    for _, row in metrics_frame.iterrows():
        k = int(row["k"])
        for metric in ("ndcg", "hit_rate", "recall_local", "precision"):
            flat[f"{metric}_at_{k}"] = round(float(row[metric]), 6)
    return flat


def main() -> None:
    """Executa o stage de avaliação."""
    preprocess_params, train_params, evaluate_params = load_params()
    seed_everything(train_params["random_seed"])
    device = get_device()

    checkpoint = torch.load(
        train_params["checkpoint_path"], map_location=device, weights_only=False
    )
    model = build_model(
        train_params["model_name"],
        {
            "embedding_cardinalities": checkpoint["embedding_cardinalities"],
            "embedding_columns": checkpoint["feature_config"]["embedding_columns"],
            "embedding_dim": checkpoint["experiment_config"]["embedding_dim"],
            "numeric_input_dim": len(
                checkpoint["feature_config"]["numeric_feature_columns"]
            ),
            "hidden_dims": checkpoint["experiment_config"]["hidden_dims"],
            "dropout": checkpoint["experiment_config"]["dropout"],
        },
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)

    results = {"best_epoch": int(checkpoint["epoch"])}
    for split in ("validation", "test"):
        dataset = build_split_dataset(split, preprocess_params, train_params)
        predictions = predict_scores(model, dataset, device)
        metrics_frame = evaluate_ranking_scores(
            predictions, k_values=evaluate_params["k_values"]
        )
        results[split] = metrics_as_dict(metrics_frame)
        print(f"{split}: {results[split]}")

    metrics_path = Path(evaluate_params["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"métricas salvas em {metrics_path}")


if __name__ == "__main__":
    main()
