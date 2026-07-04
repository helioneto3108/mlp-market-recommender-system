"""Stage `preprocess`: ajusta mapas categóricos, scaler e estatísticas do treino.

Todos os artefatos são ajustados exclusivamente no split de treino
(anti-leakage) e salvos em JSON para consumo dos stages `train` e `evaluate`.
"""

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.features import (
    build_category_maps,
    compute_numeric_scaler_stats,
    save_category_maps,
)


def compute_train_class_stats(part_paths: list[Path], target_column: str) -> dict:
    """Conta linhas/positivos do treino e deriva o pos_weight da loss."""
    n_rows = 0
    n_positives = 0
    for part_path in part_paths:
        part_df = pd.read_parquet(part_path, columns=["split", target_column])
        train_df = part_df[part_df["split"] == "train"]
        n_rows += len(train_df)
        n_positives += int(train_df[target_column].sum())
    pos_weight = (n_rows - n_positives) / n_positives
    return {"n_rows": n_rows, "n_positives": n_positives, "pos_weight": pos_weight}


def main() -> None:
    """Executa o stage de pré-processamento."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))[
        "preprocess"
    ]
    part_paths = sorted(Path(params["dataset_dir"]).glob("part-*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"Nenhuma partição em {params['dataset_dir']}")
    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    category_maps = build_category_maps(part_paths, params["embedding_columns"])
    save_category_maps(category_maps, output_dir / "category_maps.json")

    scaler_stats = compute_numeric_scaler_stats(
        part_paths, params["numeric_feature_columns"]
    )
    (output_dir / "scaler_stats.json").write_text(
        json.dumps(scaler_stats, indent=2), encoding="utf-8"
    )

    train_stats = compute_train_class_stats(part_paths, params["target_column"])
    (output_dir / "train_stats.json").write_text(
        json.dumps(train_stats, indent=2), encoding="utf-8"
    )

    cardinalities = {
        column: len(mapping) + 1 for column, mapping in category_maps.items()
    }
    print(f"cardinalidades: {cardinalities}")
    print(
        f"treino: {train_stats['n_rows']:,} linhas | pos_weight={train_stats['pos_weight']:.2f}"
    )
    print(f"preprocess concluído em {time.time() - start:.1f}s → {output_dir}")


if __name__ == "__main__":
    main()
