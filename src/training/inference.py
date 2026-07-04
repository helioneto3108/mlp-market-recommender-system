"""Inferência: pontuação de candidatos para avaliação de ranking."""

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import torch
from torch import nn


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    dataset: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    """Pontua todos os batches de um dataset com metadata.

    O score é o logit do modelo — monotônico na probabilidade, suficiente
    para ranking (a sigmoid é desnecessária e custaria uma passada extra).

    Args:
        model: Modelo treinado (será posto em modo ``eval``).
        dataset: Iterável de batches com ``categorical``, ``numeric`` e
            ``metadata`` (usar ``include_metadata=True`` no dataset).
        device: Device de execução.

    Returns:
        DataFrame com as colunas de metadata + coluna ``score``, na ordem
        de iteração do dataset.
    """
    model.eval()
    frames = []
    for batch in dataset:
        logits = model(
            categorical_inputs=batch["categorical"].to(device),
            numeric_inputs=batch["numeric"].to(device),
        )
        metadata = batch["metadata"].copy()
        metadata["score"] = logits.float().cpu().numpy()
        frames.append(metadata)
    return pd.concat(frames, ignore_index=True)
