"""Stage `train`: treina o MLP temporal com early stopping.

Reproduz o protocolo da run validada `mlp_temporal_v1_catfix`: AdamW +
ReduceLROnPlateau monitorando NDCG@k na validação, early stopping com
melhora estrita, checkpoint da melhor época no formato dos notebooks.
"""

import json
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from torch import nn

from src.config import get_settings
from src.data import TemporalParquetDataset
from src.evaluation import evaluate_ranking_scores
from src.features import load_category_maps
from src.models import build_model
from src.training import (
    EarlyStopping,
    get_device,
    predict_scores,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)

EVALUATION_METADATA = ("user_window_id", "product_id", "target")


def load_params() -> tuple[dict, dict]:
    """Carrega as seções `preprocess` e `train` do params.yaml."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    return params["preprocess"], params["train"]


def build_datasets(
    preprocess_params: dict, train_params: dict, category_maps: dict, scaler_stats: dict
) -> tuple[TemporalParquetDataset, TemporalParquetDataset]:
    """Monta os datasets de treino (com shuffle) e validação (com metadata)."""
    part_paths = sorted(Path(preprocess_params["dataset_dir"]).glob("part-*.parquet"))
    embedding_columns = preprocess_params["embedding_columns"]
    numeric_columns = preprocess_params["numeric_feature_columns"]
    model_columns = [
        "split",
        "user_window_id",
        *embedding_columns,
        *numeric_columns,
        preprocess_params["target_column"],
    ]
    common = {
        "part_paths": part_paths,
        "model_columns": model_columns,
        "embedding_columns": embedding_columns,
        "numeric_columns": numeric_columns,
        "category_maps": category_maps,
        "scaler_stats": scaler_stats,
        "batch_size": train_params["batch_size"],
        "target_column": preprocess_params["target_column"],
        "seed": train_params["random_seed"],
    }
    train_dataset = TemporalParquetDataset(
        split="train", shuffle_partitions=True, **common
    )
    validation_dataset = TemporalParquetDataset(
        split="validation",
        include_metadata=True,
        metadata_columns=EVALUATION_METADATA,
        **common,
    )
    return train_dataset, validation_dataset


def validation_metric(
    model: torch.nn.Module,
    dataset: TemporalParquetDataset,
    device: torch.device,
    train_params: dict,
) -> float:
    """NDCG@primary_k do modelo na validação."""
    predictions = predict_scores(model, dataset, device)
    metrics = evaluate_ranking_scores(predictions, k_values=[train_params["primary_k"]])
    return float(metrics.iloc[0][train_params["primary_metric"]])


def mlflow_enabled() -> bool:
    """Indica se o tracking MLflow deve ser executado neste ambiente."""
    settings = get_settings()
    return bool(settings.mlflow_tracking_uri)


def log_training_run(
    preprocess_params: dict,
    train_params: dict,
    history_path: Path,
    best_epoch: int,
    best_metric: float,
) -> None:
    """Registra parâmetros, métricas e histórico do stage de treino no MLflow."""
    settings = get_settings()
    if not mlflow_enabled():
        print("MLFLOW_TRACKING_URI não configurado; logging MLflow pulado.")
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="train_mlp_temporal"):
        mlflow.log_params(
            {f"train_{key}": value for key, value in train_params.items()}
        )
        mlflow.log_param("preprocess_dataset_dir", preprocess_params["dataset_dir"])
        mlflow.log_param("preprocess_target_column", preprocess_params["target_column"])
        mlflow.log_param(
            "preprocess_numeric_feature_count",
            len(preprocess_params["numeric_feature_columns"]),
        )
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_validation_metric", best_metric)
        mlflow.log_artifact(str(history_path), "training")


def main() -> None:
    """Executa o stage de treino."""
    preprocess_params, train_params = load_params()
    seed_everything(train_params["random_seed"])
    device = get_device()
    print(f"device: {device}")

    preprocessing_dir = Path(preprocess_params["output_dir"])
    category_maps = load_category_maps(preprocessing_dir / "category_maps.json")
    scaler_stats = json.loads(
        (preprocessing_dir / "scaler_stats.json").read_text(encoding="utf-8")
    )
    train_stats = json.loads(
        (preprocessing_dir / "train_stats.json").read_text(encoding="utf-8")
    )

    train_dataset, validation_dataset = build_datasets(
        preprocess_params, train_params, category_maps, scaler_stats
    )
    embedding_cardinalities = {
        column: len(mapping) + 1 for column, mapping in category_maps.items()
    }
    model_config = {
        "embedding_cardinalities": embedding_cardinalities,
        "embedding_columns": preprocess_params["embedding_columns"],
        "embedding_dim": train_params["embedding_dim"],
        "numeric_input_dim": len(preprocess_params["numeric_feature_columns"]),
        "hidden_dims": train_params["hidden_dims"],
        "dropout": train_params["dropout"],
    }
    model = build_model(train_params["model_name"], model_config).to(device)

    pos_weight = torch.tensor(train_stats["pos_weight"], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=train_params["lr_scheduler_patience"],
        factor=train_params["lr_scheduler_factor"],
    )

    checkpoint_path = Path(train_params["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=train_params["patience"])
    history = []

    for epoch in range(1, train_params["max_epochs"] + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_dataset, criterion, optimizer, device)
        metric = validation_metric(model, validation_dataset, device, train_params)

        if early_stopping.update(metric, epoch):
            save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                best_metric=early_stopping.best_value,
                extra={
                    "experiment_config": {**train_params, "device": str(device)},
                    "feature_config": {
                        "embedding_columns": preprocess_params["embedding_columns"],
                        "numeric_feature_columns": preprocess_params[
                            "numeric_feature_columns"
                        ],
                    },
                    "embedding_cardinalities": embedding_cardinalities,
                },
            )
        scheduler.step(metric)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_metric": metric,
                "best_epoch": early_stopping.best_epoch,
                "best_metric": early_stopping.best_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epochs_without_improvement": early_stopping.epochs_without_improvement,
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        print(
            f"epoch {epoch}/{train_params['max_epochs']} | loss={train_loss:.5f} | "
            f"val={metric:.6f} | best={early_stopping.best_value:.6f} "
            f"(epoch {early_stopping.best_epoch}) | "
            f"sem_melhora={early_stopping.epochs_without_improvement}/{train_params['patience']}"
        )
        if early_stopping.should_stop:
            print("early stopping")
            break

    history_path = Path(train_params["history_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(
        f"melhor época: {early_stopping.best_epoch} | métrica: {early_stopping.best_value:.6f}"
    )

    log_training_run(
        preprocess_params,
        train_params,
        history_path,
        int(early_stopping.best_epoch),
        float(early_stopping.best_value),
    )


if __name__ == "__main__":
    main()
