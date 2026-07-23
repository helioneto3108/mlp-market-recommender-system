"""Registra o MLP no MLflow Model Registry e promove staging → production.

Cumpre o requisito "Registrar modelo no MLflow Model Registry → Staging →
Production". Nas versões atuais do MLflow, o ciclo por *stages* foi
substituído por **aliases** — usamos os aliases ``staging`` e ``production``,
que têm a mesma semântica.

O destino é definido por ``MLFLOW_TRACKING_URI`` no ``.env`` (Settings):
funciona igual contra um SQLite local (``sqlite:///mlflow.db``), o servidor
do docker-compose ou o MLflow no Cloud Run.

Uso:
    uv run python -m scripts.register_model
"""

from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from mlflow import MlflowClient

from src.config import get_settings
from src.models import build_model
from src.utils import load_json

REGISTERED_MODEL_NAME = "mlp-temporal-recommender"


def load_train_params() -> dict[str, Any]:
    """Carrega a seção `train` do params.yaml."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    return params["train"]


def load_model_from_checkpoint(
    train_params: dict[str, Any],
) -> tuple[torch.nn.Module, dict]:
    """Reconstrói o modelo treinado a partir do checkpoint do pipeline."""
    checkpoint = torch.load(
        train_params["checkpoint_path"], map_location="cpu", weights_only=False
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
    model.eval()
    return model, checkpoint


def log_registration_run(model: torch.nn.Module, checkpoint: dict) -> str:
    """Loga a run de registro (params, métricas, modelo) e retorna o model_uri."""
    metrics_path = Path("reports/metrics.json")
    metrics = load_json(metrics_path)
    with mlflow.start_run(run_name="register_mlp_temporal"):
        mlflow.log_params(dict(checkpoint["experiment_config"]))
        mlflow.log_metric("best_epoch", checkpoint["epoch"])
        for split in ("validation", "test"):
            for metric_name, value in metrics[split].items():
                mlflow.log_metric(f"{split}_{metric_name}", value)
        mlflow.log_artifact(str(metrics_path), "evaluation")
        # pip_requirements explícito: venvs do uv não têm pip, o que quebra a
        # inferência automática do MLflow — e declarar é mais reprodutível.
        torch_version = torch.__version__.split("+")[0]
        model_info = mlflow.pytorch.log_model(
            model, name="model", pip_requirements=[f"torch=={torch_version}"]
        )
    return model_info.model_uri


def promote_through_aliases(model_uri: str) -> int:
    """Registra a versão e aplica os aliases staging → production."""
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = MlflowClient()
    # O fluxo pedido no enunciado (Staging → Production), na API atual (aliases):
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "staging", registered.version
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "production", registered.version
    )
    return int(registered.version)


def main() -> None:
    """Executa o registro e a promoção do modelo."""
    settings = get_settings()
    assert settings.mlflow_tracking_uri, "MLFLOW_TRACKING_URI não configurado no .env"
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_params = load_train_params()
    model, checkpoint = load_model_from_checkpoint(train_params)
    print(
        f"checkpoint: época {checkpoint['epoch']} | métrica {checkpoint['best_metric']:.6f}"
    )

    model_uri = log_registration_run(model, checkpoint)
    version = promote_through_aliases(model_uri)

    print(f"modelo '{REGISTERED_MODEL_NAME}' versão {version} registrado")
    print(f"aliases aplicados: @staging, @production → v{version}")
    print(f"tracking: {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
