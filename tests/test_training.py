"""Testes do treino (`src.training`): early stopping, loop e checkpoint."""

import pytest
import torch
from torch import nn

from src.models import MLPRecommender
from src.training import (
    EarlyStopping,
    get_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)

TINY_CONFIG = {
    "embedding_cardinalities": {"product_id": 8},
    "embedding_columns": ["product_id"],
    "embedding_dim": 4,
    "numeric_input_dim": 2,
    "hidden_dims": [8],
    "dropout": 0.0,
}


def build_batches(
    n_batches: int = 4, batch_size: int = 16
) -> list[dict[str, torch.Tensor]]:
    """Batches sintéticos separáveis: target depende do sinal da 1ª feature."""
    generator = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(n_batches):
        numeric = torch.randn(batch_size, 2, generator=generator)
        batches.append(
            {
                "categorical": torch.randint(
                    0, 8, (batch_size, 1), generator=generator
                ),
                "numeric": numeric,
                "target": (numeric[:, 0] > 0).float(),
            }
        )
    return batches


def test_early_stopping_segue_semantica_do_notebook() -> None:
    """Melhora estrita zera o contador; empate conta como estagnação."""
    early_stopping = EarlyStopping(patience=2)

    assert early_stopping.update(0.50, epoch=1) is True
    assert early_stopping.update(0.60, epoch=2) is True
    assert early_stopping.update(0.60, epoch=3) is False  # empate não é melhora
    assert not early_stopping.should_stop
    assert early_stopping.update(0.55, epoch=4) is False

    assert early_stopping.should_stop
    assert early_stopping.best_epoch == 2
    assert early_stopping.best_value == pytest.approx(0.60)


def test_early_stopping_patience_invalida() -> None:
    """Paciência menor que 1 é erro de configuração."""
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)


def test_train_one_epoch_reduz_a_loss() -> None:
    """Algumas épocas em dados separáveis devem reduzir a loss média."""
    seed_everything(42)
    model = MLPRecommender(**TINY_CONFIG)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batches = build_batches()
    device = torch.device("cpu")

    first_loss = train_one_epoch(model, batches, criterion, optimizer, device)
    for _ in range(9):
        last_loss = train_one_epoch(model, batches, criterion, optimizer, device)

    assert last_loss < first_loss


def test_seed_everything_torna_o_treino_deterministico() -> None:
    """Mesma seed → mesma loss após uma época (reprodutibilidade)."""
    losses = []
    for _ in range(2):
        seed_everything(123)
        model = MLPRecommender(**TINY_CONFIG)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        loss = train_one_epoch(
            model,
            build_batches(),
            nn.BCEWithLogitsLoss(),
            optimizer,
            torch.device("cpu"),
        )
        losses.append(loss)

    assert losses[0] == pytest.approx(losses[1])


def test_get_device_retorna_torch_device() -> None:
    """A detecção devolve um `torch.device` válido."""
    assert isinstance(get_device(), torch.device)


def test_checkpoint_roundtrip(tmp_path) -> None:
    """O checkpoint salvo recarrega no modelo com strict=True."""
    seed_everything(7)
    model = MLPRecommender(**TINY_CONFIG)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path, model, epoch=3, best_metric=0.5, extra={"experiment_config": TINY_CONFIG}
    )
    checkpoint = torch.load(path, weights_only=False)

    reloaded = MLPRecommender(**TINY_CONFIG)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["epoch"] == 3
    assert checkpoint["experiment_config"]["hidden_dims"] == [8]
