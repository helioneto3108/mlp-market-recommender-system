"""Inferência por linha de comando: top-K recomendações para um usuário.

Usa os mesmos artefatos do pipeline (checkpoint + mapas + scaler) — é a
camada de *serving* em miniatura. Exemplo:

    uv run --no-sync python -m scripts.predict --user-id 1000 --top-k 10
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import torch
import yaml

from src.features import (
    load_category_maps,
    load_popularity_ranking,
    map_categorical_columns,
    scale_numeric_columns,
)
from src.models import build_model


def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Top-K recomendações para um usuário.")
    parser.add_argument(
        "--user-id", type=int, required=True, help="ID do usuário (Instacart)"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test"]
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Tamanho da lista (padrão: 10)"
    )
    parser.add_argument(
        "--simulate-unknown-user",
        action="store_true",
        help=(
            "Simula um usuário desconhecido: mantém os candidatos/features do "
            "usuário informado, mas codifica a identidade como UNK (embedding "
            "zero). Útil para medir a contribuição do embedding de usuário."
        ),
    )
    return parser.parse_args()


def load_pipeline_artifacts() -> tuple[dict, dict, dict, torch.nn.Module]:
    """Carrega params, mapas, scaler e o modelo do checkpoint do pipeline."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    preprocessing_dir = Path(params["preprocess"]["output_dir"])
    category_maps = load_category_maps(preprocessing_dir / "category_maps.json")
    scaler_stats = json.loads(
        (preprocessing_dir / "scaler_stats.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        params["train"]["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = build_model(
        params["train"]["model_name"],
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
    return params, category_maps, scaler_stats, model


def load_user_candidates(params: dict, user_id: int, split: str) -> pd.DataFrame:
    """Busca os candidatos do usuário no split (predicate pushdown do parquet)."""
    preprocess = params["preprocess"]
    columns = [
        "user_window_id",
        "target",
        *preprocess["embedding_columns"],
        *preprocess["numeric_feature_columns"],
    ]
    dataset = ds.dataset(preprocess["dataset_dir"], format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=(ds.field("user_id") == user_id) & (ds.field("split") == split),
    )
    return table.to_pandas()


@torch.no_grad()
def score_candidates(
    frame: pd.DataFrame, params: dict, category_maps: dict, scaler_stats: dict, model
) -> pd.DataFrame:
    """Codifica, pontua e ordena os candidatos (desempate por product_id)."""
    preprocess = params["preprocess"]
    categorical = torch.tensor(
        map_categorical_columns(
            frame, category_maps, preprocess["embedding_columns"]
        ).to_numpy(),
        dtype=torch.long,
    )
    numeric = torch.tensor(
        scale_numeric_columns(
            frame, preprocess["numeric_feature_columns"], scaler_stats
        ).to_numpy(),
        dtype=torch.float32,
    )
    scored = frame.assign(score=model(categorical, numeric).numpy())
    return scored.sort_values(
        ["score", "product_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def attach_product_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Junta os nomes dos produtos (se o raw estiver disponível)."""
    products_path = Path("data/raw/products.csv")
    if not products_path.exists():
        return frame.assign(product_name="(products.csv indisponível)")
    products = pd.read_csv(products_path)[["product_id", "product_name"]]
    return frame.merge(products, on="product_id", how="left")


def popularity_fallback(params: dict, user_id: int, top_k: int) -> None:
    """Vitrine de popularidade para usuários sem histórico (cold start)."""
    ranking_path = (
        Path(params["preprocess"]["popularity_output_dir"]) / "top_products.json"
    )
    ranking = pd.DataFrame(load_popularity_ranking(ranking_path)).head(top_k)
    ranking = attach_product_names(ranking)
    print(f"usuário {user_id} sem histórico no dataset [fallback: popularidade]")
    print(f"\nTOP-{top_k} MAIS POPULARES (nº de usuários distintos que compraram):")
    print(ranking[["product_name", "n_users"]].to_string(index=True))


def main() -> None:
    """Executa a inferência e imprime o top-K do usuário."""
    args = parse_args()
    params, category_maps, scaler_stats, model = load_pipeline_artifacts()
    candidates = load_user_candidates(params, args.user_id, args.split)
    if candidates.empty:
        # Cold start: sem histórico não há candidatos nem features — degrada
        # graciosamente para a vitrine de popularidade (não personalizada).
        popularity_fallback(params, args.user_id, args.top_k)
        return

    if args.simulate_unknown_user:
        # ID inexistente nos mapas do treino → codificado como UNK (índice 0,
        # embedding zero). Candidatos e features de histórico são mantidos.
        candidates = candidates.assign(user_id=-1)

    ranked = attach_product_names(
        score_candidates(candidates, params, category_maps, scaler_stats, model)
    )
    top = ranked.head(args.top_k)

    window = ranked["user_window_id"].iloc[0]
    mode = " [identidade UNK simulada]" if args.simulate_unknown_user else ""
    print(f"usuário {args.user_id}{mode} | janela {window} | {len(ranked)} candidatos")
    print(f"\nTOP-{args.top_k} RECOMENDAÇÕES:")
    display = top[["product_name", "score", "target"]].rename(
        columns={"target": "comprou_de_fato"}
    )
    print(display.to_string(index=True, float_format=lambda value: f"{value:.3f}"))
    hits = int(top["target"].sum())
    total = int(ranked["target"].sum())
    print(
        f"\n{hits}/{args.top_k} do top eram compras reais | {total} compras entre os candidatos"
    )


if __name__ == "__main__":
    main()
