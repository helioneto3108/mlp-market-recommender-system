"""Pré-processamento das features do dataset temporal (do notebook 06).

Duas responsabilidades, ambas com *fit* exclusivamente no split de treino
(anti-leakage):

- **Mapas categóricos**: valores brutos → índices contíguos a partir de 1;
  o índice 0 é reservado para categorias não vistas (``UNK_INDEX``), casando
  com o ``padding_idx`` dos embeddings do modelo.
- **Normalização z-score**: média/desvio calculados em streaming pelas
  partições do treino, sem carregar o dataset inteiro em memória.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# O índice UNK é contrato entre a codificação e o padding_idx do modelo —
# importado de src.models para existir uma única fonte de verdade.
from src.models.mlp import UNK_INDEX
from src.utils.serialization import load_json

CategoryMaps = dict[str, dict[int, int]]
ScalerStats = dict[str, dict[str, float]]


def build_category_maps(
    part_paths: Iterable[Path], categorical_columns: Sequence[str]
) -> CategoryMaps:
    """Constrói os mapas valor→índice a partir do split de treino.

    Os índices começam em 1 (0 é o ``UNK_INDEX``) e seguem a ordem crescente
    dos valores — determinístico em qualquer máquina.

    Args:
        part_paths: Partições parquet do dataset temporal.
        categorical_columns: Colunas categóricas a mapear.

    Returns:
        Dicionário coluna → {valor original → índice contíguo}.
    """
    category_values: dict[str, set[int]] = {
        column: set() for column in categorical_columns
    }

    for part_path in part_paths:
        part_df = pd.read_parquet(part_path, columns=["split", *categorical_columns])
        train_df = part_df[part_df["split"] == "train"]

        for column in categorical_columns:
            category_values[column].update(
                train_df[column].dropna().astype(int).unique().tolist()
            )

    return {
        column: {value: index for index, value in enumerate(sorted(values), start=1)}
        for column, values in category_values.items()
    }


def compute_numeric_scaler_stats(
    part_paths: Iterable[Path], numeric_columns: Sequence[str]
) -> ScalerStats:
    """Calcula média e desvio das features numéricas no treino (streaming).

    Desvios nulos são substituídos por 1 para evitar divisão por zero em
    colunas constantes.

    Args:
        part_paths: Partições parquet do dataset temporal.
        numeric_columns: Colunas numéricas a padronizar.

    Returns:
        ``{"mean": {coluna: média}, "std": {coluna: desvio}}``.
    """

    sums = pd.Series(0.0, index=numeric_columns)
    squared_sums = pd.Series(0.0, index=numeric_columns)
    count = 0

    for part_path in part_paths:
        part_df = pd.read_parquet(part_path, columns=["split", *numeric_columns])
        train_df = part_df[part_df["split"] == "train"][numeric_columns]

        sums += train_df.sum()
        squared_sums += (train_df**2).sum()
        count += len(train_df)

    means = sums / count
    variances = (squared_sums / count) - (means**2)
    stds = np.sqrt(variances.clip(lower=0)).replace(0, 1)

    return {"mean": means.to_dict(), "std": stds.to_dict()}


def map_categorical_columns(
    df: pd.DataFrame, category_maps: CategoryMaps, categorical_columns: Sequence[str]
) -> pd.DataFrame:
    """Aplica os mapas categóricos; valores não vistos viram ``UNK_INDEX``.

    Args:
        df: DataFrame com as colunas categóricas brutas.
        category_maps: Mapas construídos por :func:`build_category_maps`.
        categorical_columns: Colunas a converter (ordem preservada).

    Returns:
        DataFrame só com as colunas mapeadas, em ``int64``.
    """
    mapped_df = pd.DataFrame(index=df.index)

    for column in categorical_columns:
        mapped_df[column] = (
            df[column].map(category_maps[column]).fillna(UNK_INDEX).astype("int64")
        )

    return mapped_df


def scale_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    scaler_stats: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Padroniza as features numéricas com as estatísticas do treino.

    Args:
        df: DataFrame com as colunas numéricas brutas.
        numeric_columns: Colunas a padronizar.
        scaler_stats: Estatísticas de :func:`compute_numeric_scaler_stats`.

    Returns:
        DataFrame padronizado (z-score) em ``float32``.
    """
    numeric_df = df[numeric_columns].astype("float32")
    means = pd.Series(scaler_stats["mean"])
    stds = pd.Series(scaler_stats["std"])

    return ((numeric_df - means) / stds).astype("float32")


def load_category_maps(path: Path) -> CategoryMaps:
    """Carrega mapas categóricos de JSON, restaurando as chaves inteiras.

    JSON serializa chaves de dicionário como string; sem esta reconversão,
    ``Series.map`` não encontraria nenhum valor e **todas** as categorias
    virariam UNK silenciosamente.

    Args:
        path: Arquivo JSON contendo os mapas categóricos.

    Returns:
        Mapas com chaves ``int``, idênticos aos originais.
    """
    raw = load_json(path)

    return {
        column: {int(value): index for value, index in mapping.items()}
        for column, mapping in raw.items()
    }
