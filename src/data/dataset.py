"""Dataset iterável sobre o dataset temporal particionado (do notebook 06).

O dataset de modelagem tem 92,7M de linhas em 47 partições parquet — não cabe
confortavelmente em memória junto com o treino. O ``TemporalParquetDataset``
é um ``IterableDataset`` que carrega **uma partição por vez**, aplica o
pré-processamento (mapas categóricos + z-score) e emite batches prontos para
o modelo (*out-of-core*).
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from src.features.preprocessing import (
    CategoryMaps,
    ScalerStats,
    map_categorical_columns,
    scale_numeric_columns,
)

Batch = dict[str, torch.Tensor | pd.DataFrame]

DEFAULT_METADATA_COLUMNS = (
    "split",
    "user_window_id",
    "target_order_id",
    "product_id",
    "target",
)


class TemporalParquetDataset(IterableDataset):
    """Itera batches de um split do dataset temporal, partição a partição.

    Args:
        part_paths: Partições parquet do dataset de modelagem.
        split: Split a emitir (``train``, ``validation`` ou ``test``).
        model_columns: Colunas a ler do parquet (inclui ``split`` e alvo).
        embedding_columns: Colunas categóricas, na ordem esperada pelo modelo.
        numeric_columns: Colunas numéricas, na ordem esperada pelo modelo.
        category_maps: Mapas categóricos ajustados no treino.
        scaler_stats: Estatísticas de normalização ajustadas no treino.
        batch_size: Tamanho dos batches emitidos.
        target_column: Nome da coluna alvo (0/1).
        shuffle_partitions: Se ``True``, embaralha a ordem das partições e as
            linhas dentro de cada partição (uso: treino). Determinístico dada
            a ``seed``.
        seed: Semente do embaralhamento.
        include_metadata: Se ``True``, cada batch inclui um DataFrame
            ``metadata`` com as colunas de avaliação (uso: predição).
        metadata_columns: Colunas incluídas no ``metadata``.
    """

    def __init__(
        self,
        part_paths: Sequence[Path],
        split: str,
        model_columns: Sequence[str],
        embedding_columns: Sequence[str],
        numeric_columns: Sequence[str],
        category_maps: CategoryMaps,
        scaler_stats: ScalerStats,
        batch_size: int,
        target_column: str = "target",
        shuffle_partitions: bool = False,
        seed: int = 42,
        include_metadata: bool = False,
        metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    ) -> None:
        self.part_paths = list(part_paths)
        self.split = split
        self.model_columns = list(model_columns)
        self.embedding_columns = list(embedding_columns)
        self.numeric_columns = list(numeric_columns)
        self.category_maps = category_maps
        self.scaler_stats = scaler_stats
        self.batch_size = batch_size
        self.target_column = target_column
        self.shuffle_partitions = shuffle_partitions
        self.seed = seed
        self.include_metadata = include_metadata
        self.metadata_columns = list(metadata_columns)

    def __iter__(self) -> Iterator[Batch]:
        """Emite batches do split configurado, uma partição por vez."""
        for part_path in self._partition_order():
            part_df = self._load_split_partition(part_path)
            if part_df is None:
                continue
            yield from self._partition_batches(part_df)

    def _partition_order(self) -> list[Path]:
        """Ordem das partições (embaralhada de forma determinística no treino)."""
        part_paths = self.part_paths.copy()
        if self.shuffle_partitions:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(part_paths)
        return part_paths

    def _load_split_partition(self, part_path: Path) -> pd.DataFrame | None:
        """Carrega uma partição filtrada pelo split; ``None`` se vazia."""
        part_df = pd.read_parquet(part_path, columns=self.model_columns)
        part_df = part_df[part_df["split"] == self.split].copy()
        if part_df.empty:
            return None
        if self.shuffle_partitions:
            part_df = part_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return part_df

    def _partition_batches(self, part_df: pd.DataFrame) -> Iterator[Batch]:
        """Pré-processa a partição e fatia em batches de tensores."""
        categorical_df = map_categorical_columns(part_df, self.category_maps, self.embedding_columns)
        numeric_df = scale_numeric_columns(part_df, self.numeric_columns, self.scaler_stats)
        target = part_df[self.target_column].astype("float32").to_numpy()
        for start in range(0, len(part_df), self.batch_size):
            end = start + self.batch_size
            batch: Batch = {
                "categorical": torch.tensor(categorical_df.iloc[start:end].to_numpy(), dtype=torch.long),
                "numeric": torch.tensor(numeric_df.iloc[start:end].to_numpy(), dtype=torch.float32),
                "target": torch.tensor(target[start:end], dtype=torch.float32),
            }
            if self.include_metadata:
                batch["metadata"] = part_df.iloc[start:end][self.metadata_columns].copy()
            yield batch
