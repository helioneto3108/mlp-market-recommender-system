"""Loop de treino e utilitários de reprodutibilidade (do notebook 06)."""

import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade (Python, NumPy, PyTorch).

    Também ativa os algoritmos determinísticos do PyTorch (``warn_only=True``:
    operações sem versão determinística emitem aviso em vez de falhar).

    Args:
        seed: Semente global de reprodutibilidade.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Detecta o melhor device disponível (cuda → mps → cpu).

    Returns:
        O ``torch.device`` a usar em treino e inferência.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Executa uma época de treino e retorna a loss média por exemplo.

    Args:
        model: Modelo a treinar (será posto em modo ``train``).
        loader: Iterável de batches com chaves ``categorical``, ``numeric``
            e ``target``.
        criterion: Função de perda (ex.: ``BCEWithLogitsLoss``).
        optimizer: Otimizador dos parâmetros do modelo.
        device: Device de execução dos tensores.

    Returns:
        Loss média ponderada pelo tamanho de cada batch.
    """
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        categorical_inputs = batch["categorical"].to(device)
        numeric_inputs = batch["numeric"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        logits = model(
            categorical_inputs=categorical_inputs, numeric_inputs=numeric_inputs
        )
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(targets)
        total_examples += len(targets)
    return total_loss / total_examples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    best_metric: float,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Salva o checkpoint do modelo com metadados de treino.

    Mantém o formato dos checkpoints dos notebooks (``model_state_dict``,
    ``epoch``, ``best_metric``); campos adicionais — ``experiment_config``,
    ``feature_config``, cardinalidades — entram via ``extra``.

    Args:
        path: Destino do arquivo ``.pt``.
        model: Modelo cujo ``state_dict`` será salvo.
        epoch: Época correspondente ao checkpoint.
        best_metric: Melhor métrica de validação até aqui.
        extra: Metadados adicionais a incluir no checkpoint.
    """
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
