"""Pré-processamento de features (mapas categóricos, normalização, popularidade)."""

from src.features.popularity import (
    compute_popularity_ranking,
    load_popularity_ranking,
    save_popularity_ranking,
)
from src.features.preprocessing import (
    build_category_maps,
    compute_numeric_scaler_stats,
    load_category_maps,
    map_categorical_columns,
    save_category_maps,
    scale_numeric_columns,
)

__all__ = [
    "build_category_maps",
    "compute_numeric_scaler_stats",
    "compute_popularity_ranking",
    "load_category_maps",
    "load_popularity_ranking",
    "map_categorical_columns",
    "save_category_maps",
    "save_popularity_ranking",
    "scale_numeric_columns",
]
