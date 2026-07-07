"""Arquitetura do MLP ranker com embeddings (extraída do notebook 06).

O modelo pontua pares (janela de usuário, produto candidato): embeddings das
colunas categóricas são concatenados às features numéricas padronizadas e
passam por um MLP que produz um logit de relevância — a base do re-ranking
dos candidatos.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import nn

UNK_INDEX = 0
"""Índice reservado para categorias não vistas no treino (``padding_idx``)."""


class MLPRecommender(nn.Module):
    """MLP ranker com embeddings para recomendação temporal.

    Réplica fiel da arquitetura validada no notebook 06 (run
    ``mlp_temporal_v1_catfix``): um embedding por coluna categórica
    (``padding_idx=0`` para o índice UNK), concatenação com as features
    numéricas e camadas densas ``Linear → ReLU → Dropout``.

    Args:
        embedding_cardinalities: Nº de categorias (incluindo UNK) por coluna.
        embedding_columns: Ordem das colunas categóricas na entrada.
        embedding_dim: Dimensão de cada embedding.
        numeric_input_dim: Nº de features numéricas.
        hidden_dims: Larguras das camadas ocultas (ex.: ``[128, 64]``).
        dropout: Probabilidade de dropout após cada camada oculta.
    """

    def __init__(
        self,
        embedding_cardinalities: Mapping[str, int],
        embedding_columns: Sequence[str],
        embedding_dim: int,
        numeric_input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_columns = list(embedding_columns)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=embedding_cardinalities[column],
                    embedding_dim=embedding_dim,
                    padding_idx=UNK_INDEX,
                )
                for column in self.embedding_columns
            ]
        )
        input_dim = len(self.embedding_columns) * embedding_dim + numeric_input_dim
        self.mlp = _build_dense_layers(input_dim, hidden_dims, dropout)

    def forward(
        self, categorical_inputs: torch.Tensor, numeric_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Calcula o logit de relevância de cada par (janela, candidato).

        Args:
            categorical_inputs: Índices categóricos, shape ``(batch, n_colunas)``
                na ordem de ``embedding_columns``.
            numeric_inputs: Features numéricas padronizadas, shape
                ``(batch, numeric_input_dim)``.

        Returns:
            Logits com shape ``(batch,)`` — aplicar sigmoid dá a probabilidade.
        """
        embedded_inputs = [
            embedding(categorical_inputs[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([*embedded_inputs, numeric_inputs], dim=1)
        return self.mlp(features).squeeze(1)


def _build_dense_layers(
    input_dim: int, hidden_dims: Sequence[int], dropout: float
) -> nn.Sequential:
    """Monta a pilha densa ``Linear → ReLU → Dropout`` + camada de saída."""
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, 1))
    return nn.Sequential(*layers)
