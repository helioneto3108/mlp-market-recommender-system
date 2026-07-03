"""Registra os experimentos (4 modelos) no MLflow do DagsHub.

Loga params/métricas/artefatos de cada modelo já treinado (a partir de
`data/processed/model_comparison.parquet` e dos artefatos em `models/`),
registra a MLP no Model Registry e a promove a "Production" (via alias).

Pré-requisitos: preencher `.env` (copie de `.env.example`) com as credenciais
do DagsHub. Rodar: `uv run python scripts/track_experiments.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from mlflow import MlflowClient


def find_project_root(marker: str = 'pyproject.toml') -> Path:
    """Sobe na arvore de diretorios ate encontrar o marcador do projeto."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f'Marcador {marker} nao encontrado.')


def load_dotenv(path: Path) -> None:
    """Carrega variaveis de um arquivo .env para o ambiente (sem dependencias)."""
    if not path.exists():
        raise FileNotFoundError('Arquivo .env nao encontrado — copie de .env.example.')
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        # remove espacos, aspas e eventuais colchetes de placeholder (< >)
        value = value.strip().strip('"').strip("'").lstrip('<').rstrip('>').strip()
        os.environ[key.strip()] = value


class FeatureMLP(nn.Module):
    """Embeddings de produto/aisle/department + features numericas -> MLP."""

    def __init__(self, n_products: int, n_aisles: int, n_depts: int, n_feats: int,
                 layers: tuple = (128, 64), dropout: float = 0.2):
        super().__init__()
        self.product_emb = nn.Embedding(n_products, 32)
        self.aisle_emb = nn.Embedding(n_aisles, 8)
        self.dept_emb = nn.Embedding(n_depts, 4)
        dims = [32 + 8 + 4 + n_feats, *layers]
        blocks = []
        for a, b in zip(dims, dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        blocks.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, product, aisle, dept, feats):
        x = torch.cat([self.product_emb(product), self.aisle_emb(aisle),
                       self.dept_emb(dept), feats], 1)
        return self.net(x).squeeze(1)


def metric_dict(row: pd.Series) -> dict[str, float]:
    """Converte as colunas de metrica de uma linha em chaves validas do MLflow."""
    rename = {'Precision@K': 'precision', 'Recall@K': 'recall',
              'NDCG@K': 'ndcg', 'MAP@K': 'map'}
    k = int(row['k'])
    return {f'{short}_at_{k}': float(row[col]) for col, short in rename.items()}


# Parametros declarados de cada modelo (para rastreabilidade).
MODEL_PARAMS = {
    'Popularidade': {'method': 'global_popularity', 'personalized': False},
    'Histórico (freq)': {'method': 'user_frequency', 'personalized': True},
    'NCF (IDs)': {'model_type': 'NCF', 'emb_dim': 32, 'layers': '128-64-32',
                  'neg_sampling_ratio': 4, 'epochs': 10, 'lr': 1e-3},
    'MLP (features)': {'model_type': 'FeatureMLP', 'product_emb': 32, 'aisle_emb': 8,
                       'dept_emb': 4, 'n_features': 10, 'layers': '128-64',
                       'dropout': 0.2, 'lr': 1e-3, 'batch_size': 16384,
                       'early_stopping': True},
}
COMMON_PARAMS = {'eval_users': 26241, 'protocol': 'candidate_reranking', 'primary_metric': 'ndcg_at_10'}
REGISTERED_NAME = 'instacart_feature_mlp'


def log_baseline_or_ncf(name: str, comparison: pd.DataFrame, artifact: Path) -> None:
    """Loga um run de modelo sem peso registrado (baselines e NCF)."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**COMMON_PARAMS, **MODEL_PARAMS[name]})
        for _, row in comparison[comparison['model'] == name].iterrows():
            mlflow.log_metrics(metric_dict(row))
        if artifact.exists():
            mlflow.log_artifact(str(artifact), artifact_path='figures')
        print(f'  ✓ run logado: {name}')


def log_feature_mlp(comparison: pd.DataFrame, models_dir: Path, artifact: Path) -> None:
    """Loga a MLP com o modelo PyTorch e a registra no Model Registry."""
    state = torch.load(models_dir / 'feature_mlp.pt', map_location='cpu')
    model = FeatureMLP(state['product_emb.weight'].shape[0], state['aisle_emb.weight'].shape[0],
                       state['dept_emb.weight'].shape[0], MODEL_PARAMS['MLP (features)']['n_features'])
    model.load_state_dict(state)
    model.eval()
    with mlflow.start_run(run_name='MLP (features)') as run:
        mlflow.log_params({**COMMON_PARAMS, **MODEL_PARAMS['MLP (features)']})
        for _, row in comparison[comparison['model'] == 'MLP (features)'].iterrows():
            mlflow.log_metrics(metric_dict(row))
        for extra in [artifact, models_dir / 'feature_mlp_scaler.json']:
            if extra.exists():
                mlflow.log_artifact(str(extra))
        mlflow.pytorch.log_model(model, name='model', registered_model_name=REGISTERED_NAME)
        print(f'  ✓ run + modelo registrado: MLP (features) | run_id={run.info.run_id}')
    return REGISTERED_NAME


def set_model_stage(name: str, stage: str = 'Staging') -> None:
    """Marca a versao mais recente do modelo com o estagio dado (alias do MLflow 3.x)."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    try:
        client.set_registered_model_alias(name, stage.lower(), latest.version)
        client.set_model_version_tag(name, latest.version, 'stage', stage)
        print(f'  ✓ {name} v{latest.version} marcado como {stage} (alias={stage.lower()})')
    except Exception as exc:  # noqa: BLE001 — DagsHub pode não suportar alias
        print(f'  ! alias falhou ({exc}); marcando via tag stage={stage}')
        client.set_model_version_tag(name, latest.version, 'stage', stage)


def main() -> None:
    """Orquestra o registro de todos os experimentos no MLflow do DagsHub."""
    root = find_project_root()
    load_dotenv(root / '.env')
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', 'instacart-recommender'))
    print('Tracking URI:', mlflow.get_tracking_uri())

    comparison = pd.read_parquet(root / 'data' / 'processed' / 'model_comparison.parquet')
    figure = root / 'docs' / 'img' / 'model' / '01_model_comparison.png'
    models_dir = root / 'models'

    for name in ['Popularidade', 'Histórico (freq)', 'NCF (IDs)']:
        log_baseline_or_ncf(name, comparison, figure)
    registered = log_feature_mlp(comparison, models_dir, figure)
    set_model_stage(registered, 'Staging')
    print('\nPronto! Veja em: ', os.environ['MLFLOW_TRACKING_URI'].replace('.mlflow', '/experiments'))


if __name__ == '__main__':
    main()
