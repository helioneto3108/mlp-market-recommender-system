"""Testes das métricas de ranking (`src.evaluation`)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import RankingMetricsAccumulator, evaluate_ranking_scores

IDCG_2 = 1.0 + 1.0 / np.log2(3.0)


def build_window(
    window_id: str, products: list[int], scores: list[float], targets: list[int]
) -> pd.DataFrame:
    """Monta o DataFrame de uma janela de avaliação."""
    return pd.DataFrame(
        {
            "user_window_id": window_id,
            "product_id": products,
            "target": targets,
            "score": scores,
        }
    )


@pytest.fixture
def two_windows() -> pd.DataFrame:
    """Duas janelas com métricas conhecidas (calculadas à mão).

    - w1: positivos nas posições 1 e 3 do ranking (p1 e p3).
    - w2: empate de score; o positivo (p9) perde o desempate para p8.
    """
    w1 = build_window("w1", [1, 2, 3, 4], [4.0, 3.0, 2.0, 1.0], [1, 0, 1, 0])
    w2 = build_window("w2", [8, 9], [1.0, 1.0], [0, 1])
    return pd.concat([w1, w2], ignore_index=True)


def test_metricas_com_valores_conhecidos(two_windows) -> None:
    """NDCG/hit/recall/precision @2 conferem com o cálculo manual."""
    metrics = evaluate_ranking_scores(two_windows, k_values=[2]).iloc[0]

    ndcg_w1 = 1.0 / IDCG_2  # positivo no rank 0; idcg com 2 positivos
    ndcg_w2 = 1.0 / np.log2(3.0)  # positivo no rank 1; idcg com 1 positivo
    assert metrics["windows_evaluated"] == 2
    assert metrics["ndcg"] == pytest.approx((ndcg_w1 + ndcg_w2) / 2)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["recall_local"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["precision"] == pytest.approx(0.5)


def test_janela_sem_positivo_e_excluida(two_windows) -> None:
    """Janelas sem positivo coberto não entram no denominador."""
    w3 = build_window("w3", [5, 6], [2.0, 1.0], [0, 0])
    frame = pd.concat([two_windows, w3], ignore_index=True)

    metrics = evaluate_ranking_scores(frame, k_values=[2]).iloc[0]
    reference = evaluate_ranking_scores(two_windows, k_values=[2]).iloc[0]

    assert metrics["windows_evaluated"] == 2
    assert metrics["ndcg"] == pytest.approx(reference["ndcg"])


def test_desempate_deterministico_por_product_id() -> None:
    """Com scores empatados, o menor product_id fica à frente no ranking."""
    positivo_perde = build_window("w", [8, 9], [1.0, 1.0], [0, 1])
    positivo_ganha = build_window("w", [8, 9], [1.0, 1.0], [1, 0])

    assert (
        evaluate_ranking_scores(positivo_perde, k_values=[1]).iloc[0]["hit_rate"] == 0.0
    )
    assert (
        evaluate_ranking_scores(positivo_ganha, k_values=[1]).iloc[0]["hit_rate"] == 1.0
    )


def test_acumulador_particionado_equivale_a_avaliacao_unica(two_windows) -> None:
    """Avaliar em chunks (out-of-core) dá o mesmo resultado da avaliação única."""
    accumulator = RankingMetricsAccumulator(k_values=[2])
    for _, chunk in two_windows.groupby("user_window_id"):
        accumulator.update(chunk)

    pd.testing.assert_frame_equal(
        accumulator.to_frame(),
        evaluate_ranking_scores(two_windows, k_values=[2]),
    )


def test_colunas_ausentes_levantam_erro() -> None:
    """DataFrame sem as colunas do protocolo falha alto, não silenciosamente."""
    frame = pd.DataFrame({"user_window_id": ["w"], "score": [1.0]})

    with pytest.raises(ValueError, match="Colunas ausentes"):
        evaluate_ranking_scores(frame, k_values=[2])


def test_sem_janelas_elegiveis_retorna_nan() -> None:
    """Sem nenhuma janela com positivo, as métricas ficam NaN (não zero)."""
    frame = build_window("w", [1, 2], [2.0, 1.0], [0, 0])

    metrics = evaluate_ranking_scores(frame, k_values=[2]).iloc[0]

    assert metrics["windows_evaluated"] == 0
    assert np.isnan(metrics["ndcg"])
