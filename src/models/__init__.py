"""Modelos de recomendação e sua Factory."""

from src.models.factory import build_model, register_model
from src.models.mlp import UNK_INDEX, MLPRecommender

__all__ = ["UNK_INDEX", "MLPRecommender", "build_model", "register_model"]
