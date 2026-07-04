"""Métricas de ranking por janela de usuário.

Implementa o protocolo de avaliação dos notebooks 05/06: dentro de cada
``user_window_id``, os candidatos são ordenados por score (com desempate
determinístico por ``product_id``) e as métricas — NDCG, hit rate, recall
local e precision — são calculadas em k ∈ ``k_values``, considerando apenas
janelas com pelo menos um positivo coberto pelos candidatos.

A implementação é vetorizada (``np.add.reduceat`` soma por janela sem loop
Python) e acumulável: :class:`RankingMetricsAccumulator` permite avaliar
datasets particionados sem carregá-los inteiros em memória (*out-of-core*),
que é o cenário real do projeto (92,7M de linhas em 47 partições).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("user_window_id", "product_id", "target", "score")

_METRIC_KEYS = ("ndcg", "hit_rate", "recall_local", "precision", "windows")


def _validate_columns(frame: pd.DataFrame) -> None:
    """Garante que o DataFrame tem as colunas exigidas pelo protocolo."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para avaliação de ranking: {missing}")


def _sort_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordena por janela e score decrescente, com desempate por produto.

    O ``mergesort`` (estável) + desempate explícito por ``product_id`` garante
    ranking determinístico entre plataformas — mesma lição do bug corrigido no
    notebook 03.
    """
    return frame.sort_values(
        ["user_window_id", "score", "product_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def _window_starts(window_ids: np.ndarray) -> np.ndarray:
    """Índices onde cada janela começa no array ordenado."""
    is_start = np.r_[True, window_ids[1:] != window_ids[:-1]]
    return np.flatnonzero(is_start)


def _ranks_within_windows(n_rows: int, starts: np.ndarray) -> np.ndarray:
    """Posição (0-based) de cada linha dentro da sua janela."""
    sizes = np.diff(np.r_[starts, n_rows])
    return np.arange(n_rows) - np.repeat(starts, sizes)


def _ideal_dcg_table(max_k: int) -> np.ndarray:
    """Tabela cumulativa de DCG ideal: ``tabela[m]`` = IDCG com m positivos."""
    gains = 1.0 / np.log2(np.arange(max_k) + 2.0)
    return np.concatenate([[0.0], np.cumsum(gains)])


def _metric_sums_for_k(
    relevance: np.ndarray,
    ranks: np.ndarray,
    starts: np.ndarray,
    window_positives: np.ndarray,
    eligible: np.ndarray,
    idcg_table: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Somas das métricas em k sobre as janelas elegíveis (≥1 positivo)."""
    in_top_k = ranks < k
    hits = np.add.reduceat(relevance * in_top_k, starts)
    dcg = np.add.reduceat(relevance * in_top_k / np.log2(ranks + 2.0), starts)
    idcg = idcg_table[np.minimum(window_positives, k).astype(int)]
    return {
        "ndcg": float((dcg[eligible] / idcg[eligible]).sum()),
        "hit_rate": float((hits[eligible] > 0).sum()),
        "recall_local": float((hits[eligible] / window_positives[eligible]).sum()),
        "precision": float((hits[eligible] / k).sum()),
        "windows": float(eligible.sum()),
    }


@dataclass
class RankingMetricsAccumulator:
    """Acumula métricas de ranking ao longo de partições de um dataset.

    Uso típico (avaliação *out-of-core*)::

        accumulator = RankingMetricsAccumulator(k_values=[5, 10, 20])
        for particao in particoes:
            accumulator.update(frame_pontuado)
        metricas = accumulator.to_frame()

    Attributes:
        k_values: Valores de corte (k) a avaliar.
    """

    k_values: Sequence[int]
    _totals: dict[int, dict[str, float]] = field(init=False, repr=False)
    _idcg_table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.k_values:
            raise ValueError("k_values não pode ser vazio.")
        self._idcg_table = _ideal_dcg_table(max(self.k_values))
        self._totals = {k: dict.fromkeys(_METRIC_KEYS, 0.0) for k in self.k_values}

    def update(self, frame: pd.DataFrame) -> None:
        """Acumula as métricas de uma partição já pontuada.

        Args:
            frame: DataFrame com as colunas ``user_window_id``, ``product_id``,
                ``target`` (0/1) e ``score``. Cada janela deve estar inteira na
                partição (janelas não podem cruzar partições).
        """
        _validate_columns(frame)
        if frame.empty:
            return
        ordered = _sort_for_ranking(frame)
        starts = _window_starts(ordered["user_window_id"].to_numpy())
        ranks = _ranks_within_windows(len(ordered), starts)
        relevance = ordered["target"].to_numpy(dtype=np.float64)
        window_positives = np.add.reduceat(relevance, starts)
        eligible = window_positives > 0
        for k in self.k_values:
            sums = _metric_sums_for_k(
                relevance,
                ranks,
                starts,
                window_positives,
                eligible,
                self._idcg_table,
                k,
            )
            for key, value in sums.items():
                self._totals[k][key] += value

    def to_frame(self) -> pd.DataFrame:
        """Consolida as médias por k em um DataFrame de métricas.

        Returns:
            DataFrame com colunas ``k``, ``windows_evaluated``, ``ndcg``,
            ``hit_rate``, ``recall_local`` e ``precision``. Se nenhuma janela
            elegível foi vista, as métricas ficam ``NaN``.
        """
        rows = []
        for k in self.k_values:
            totals = self._totals[k]
            windows = totals["windows"]
            denominator = windows if windows > 0 else float("nan")
            rows.append(
                {
                    "k": k,
                    "windows_evaluated": int(windows),
                    "ndcg": totals["ndcg"] / denominator,
                    "hit_rate": totals["hit_rate"] / denominator,
                    "recall_local": totals["recall_local"] / denominator,
                    "precision": totals["precision"] / denominator,
                }
            )
        return pd.DataFrame(rows)


def evaluate_ranking_scores(
    frame: pd.DataFrame, k_values: Sequence[int]
) -> pd.DataFrame:
    """Avalia um DataFrame único de candidatos pontuados.

    Atalho para o caso em que o dataset cabe em memória; para avaliação
    particionada use :class:`RankingMetricsAccumulator` diretamente.

    Args:
        frame: Candidatos pontuados (ver :meth:`RankingMetricsAccumulator.update`).
        k_values: Valores de corte (k) a avaliar.

    Returns:
        DataFrame de métricas por k (ver :meth:`RankingMetricsAccumulator.to_frame`).
    """
    accumulator = RankingMetricsAccumulator(k_values=k_values)
    accumulator.update(frame)
    return accumulator.to_frame()
