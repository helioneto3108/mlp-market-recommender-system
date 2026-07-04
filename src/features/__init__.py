"""Pré-processamento de features (mapas categóricos e normalização)."""

from src.features.preprocessing import (
    build_category_maps,
    compute_numeric_scaler_stats,
    map_categorical_columns,
    scale_numeric_columns,
)

__all__ = [
    "build_category_maps",
    "compute_numeric_scaler_stats",
    "map_categorical_columns",
    "scale_numeric_columns",
]
