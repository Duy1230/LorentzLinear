"""Minimal hyperbolic graph Transformer with swappable attention.

Designed for node classification on small graphs (Disease, Cora, etc.).
Uses a single attention layer for simplicity -- this is a sanity check,
not a state-of-the-art architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from core.hyperbolic_ops import (
    sample_hyperboloid, project_to_hyperboloid, minkowski_norm_sq,
)
from core.lorentz_linear import LorentzLinearAttention
from core.baselines.quadratic_lorentz import QuadraticLorentzAttention
from core.baselines.hypformer_linear import HypformerLinearAttention


class EuclideanToHyperboloid(nn.Module):
    """Map Euclidean features to the hyperboloid via learned linear + projection."""

    def __init__(self, d_in: int, d_hyp: int, K: float = -1.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_hyp + 1)
        self.K = K

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear(x)
        return project_to_hyperboloid(h, self.K)


class HypGraphTransformer(nn.Module):
    """Single-layer hyperbolic graph Transformer for node classification.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_hyp : int
        Space-like dimension on the hyperboloid (embedding is d_hyp+1).
    n_classes : int
        Number of output classes.
    attn_type : str
        One of 'quadratic', 'lorentzlinear', 'hypformer'.
    R, M, beta, K : attention hyperparameters.
    """

    def __init__(
        self,
        d_in: int,
        d_hyp: int,
        n_classes: int,
        attn_type: str = "lorentzlinear",
        R: int = 2,
        M: int = 64,
        beta: float = 1.0,
        K: float = -1.0,
        use_orf: bool = False,
    ):
        super().__init__()
        self.embed = EuclideanToHyperboloid(d_in, d_hyp, K=K)

        if attn_type == "quadratic":
            self.attn = QuadraticLorentzAttention(beta=beta, K=K)
        elif attn_type == "lorentzlinear":
            self.attn = LorentzLinearAttention(
                d=d_hyp, R=R, M=M, beta=beta, K=K, use_orf=use_orf,
            )
        elif attn_type == "hypformer":
            self.attn = HypformerLinearAttention(d=d_hyp, M=M, beta=beta, K=K)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.classifier = nn.Linear(d_hyp + 1, n_classes)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: node features, (N, d_in).
            mask: unused for full-graph attention.

        Returns:
            Logits, (N, n_classes).
        """
        h = self.embed(x)           # (N, d_hyp+1)
        h = self.attn(h, h, h)      # (N, d_hyp+1)
        return self.classifier(h)   # (N, n_classes)
