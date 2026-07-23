"""Pré-processamento de features (mapas categóricos, normalização, popularidade)."""

from src.features.popularity import (
    compute_popularity_ranking,
)
from src.features.preprocessing import (
    build_category_maps,
    compute_numeric_scaler_stats,
    load_category_maps,
    map_categorical_columns,
    scale_numeric_columns,
)

__all__ = [
    "build_category_maps",
    "compute_numeric_scaler_stats",
    "compute_popularity_ranking",
    "load_category_maps",
    "map_categorical_columns",
    "scale_numeric_columns",
]
