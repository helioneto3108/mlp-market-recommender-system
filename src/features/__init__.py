"""Pré-processamento de features (mapas categóricos e normalização)."""

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
    "load_category_maps",
    "map_categorical_columns",
    "save_category_maps",
    "scale_numeric_columns",
]
