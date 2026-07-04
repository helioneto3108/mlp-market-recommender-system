"""Factory de modelos de recomendação.

Materializa o padrão **Factory** (requisito do projeto): quem consome um
modelo — loop de treino, stage do DVC, notebook — pede pelo nome e entrega a
configuração; a construção concreta fica centralizada aqui. Novos modelos
entram via :func:`register_model`, sem alterar os consumidores (princípio
aberto/fechado, o "O" do SOLID).
"""

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from src.models.mlp import MLPRecommender

ModelBuilder = Callable[[Mapping[str, Any]], nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    """Registra um construtor de modelo sob um nome.

    Args:
        name: Identificador do modelo (ex.: ``"mlp_temporal"``).
        builder: Função que recebe a configuração e retorna o ``nn.Module``.

    Raises:
        ValueError: Se o nome já estiver registrado (evita sobrescrita
            silenciosa de arquiteturas).
    """
    if name in _REGISTRY:
        raise ValueError(f"Modelo '{name}' já registrado.")
    _REGISTRY[name] = builder


def build_model(name: str, config: Mapping[str, Any]) -> nn.Module:
    """Constrói um modelo registrado a partir da configuração.

    Args:
        name: Identificador do modelo no registro.
        config: Hiperparâmetros exigidos pelo construtor correspondente.

    Returns:
        Instância pronta do modelo (``nn.Module``).

    Raises:
        ValueError: Se o nome não estiver registrado — falha alto, listando
            os modelos disponíveis.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(f"Modelo desconhecido: '{name}'. Disponíveis: {available}")
    return _REGISTRY[name](config)


def _build_mlp_recommender(config: Mapping[str, Any]) -> nn.Module:
    """Constrói o :class:`MLPRecommender` a partir da configuração do treino."""
    return MLPRecommender(
        embedding_cardinalities=config["embedding_cardinalities"],
        embedding_columns=config["embedding_columns"],
        embedding_dim=config["embedding_dim"],
        numeric_input_dim=config["numeric_input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )


register_model("mlp_temporal", _build_mlp_recommender)
