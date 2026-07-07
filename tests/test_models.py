"""Testes dos modelos e da Factory (`src.models`)."""

import pytest
import torch
from torch import nn

from src.models import UNK_INDEX, MLPRecommender, build_model, register_model

MODEL_CONFIG = {
    "embedding_cardinalities": {"user_id": 10, "product_id": 20, "aisle_id": 5},
    "embedding_columns": ["user_id", "product_id", "aisle_id"],
    "embedding_dim": 4,
    "numeric_input_dim": 6,
    "hidden_dims": [16, 8],
    "dropout": 0.1,
}


def build_test_inputs(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Gera um lote sintético compatível com MODEL_CONFIG."""
    generator = torch.Generator().manual_seed(42)
    categorical = torch.stack(
        [
            torch.randint(0, 10, (batch_size,), generator=generator),
            torch.randint(0, 20, (batch_size,), generator=generator),
            torch.randint(0, 5, (batch_size,), generator=generator),
        ],
        dim=1,
    )
    numeric = torch.randn(batch_size, 6, generator=generator)
    return categorical, numeric


def test_forward_retorna_um_logit_por_exemplo() -> None:
    """A saída tem shape (batch,) — um logit de relevância por candidato."""
    model = MLPRecommender(**MODEL_CONFIG)
    categorical, numeric = build_test_inputs(batch_size=3)

    logits = model(categorical, numeric)

    assert logits.shape == (3,)
    assert logits.dtype == torch.float32


def test_indice_unk_tem_embedding_nulo() -> None:
    """O padding_idx garante vetor zero para categorias não vistas no treino."""
    model = MLPRecommender(**MODEL_CONFIG)

    for embedding in model.embeddings:
        assert torch.all(embedding.weight[UNK_INDEX] == 0)


def test_construcao_e_deterministica_com_seed() -> None:
    """Mesma seed → mesmos pesos iniciais (reprodutibilidade)."""
    torch.manual_seed(42)
    model_a = MLPRecommender(**MODEL_CONFIG)
    torch.manual_seed(42)
    model_b = MLPRecommender(**MODEL_CONFIG)

    for param_a, param_b in zip(
        model_a.parameters(), model_b.parameters(), strict=True
    ):
        assert torch.equal(param_a, param_b)


def test_factory_constroi_mlp_temporal() -> None:
    """A Factory entrega o MLPRecommender configurado a partir do dict."""
    model = build_model("mlp_temporal", MODEL_CONFIG)

    assert isinstance(model, MLPRecommender)
    assert len(model.embeddings) == len(MODEL_CONFIG["embedding_columns"])


def test_factory_falha_alto_para_modelo_desconhecido() -> None:
    """Nome não registrado levanta ValueError listando os disponíveis."""
    with pytest.raises(ValueError, match="mlp_temporal"):
        build_model("modelo_inexistente", MODEL_CONFIG)


def test_registro_de_novo_modelo_estende_a_factory() -> None:
    """Novos modelos entram sem alterar consumidores (aberto/fechado)."""
    register_model("dummy_linear", lambda config: nn.Linear(2, 1))

    model = build_model("dummy_linear", {})

    assert isinstance(model, nn.Linear)


def test_registro_duplicado_e_rejeitado() -> None:
    """Sobrescrever uma arquitetura registrada é erro, não efeito silencioso."""
    with pytest.raises(ValueError, match="já registrado"):
        register_model("mlp_temporal", lambda config: nn.Linear(2, 1))
