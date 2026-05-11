"""Hyperbolic graph Transformer with swappable attention.

Designed for node classification on small graphs (Disease, Cora, etc.).
Supports multi-layer stacking with tangent-space residual connections.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from core.hyperbolic_ops import (
    project_to_hyperboloid, minkowski_inner,
)
from core.lorentz_linear import LorentzLinearAttention
from core.baselines.quadratic_lorentz import QuadraticLorentzAttention
from core.baselines.hypformer_linear import HypformerLinearAttention


def _make_attn(attn_type, d_hyp, R, M, beta, K, use_orf, seed=42):
    if attn_type == "quadratic":
        return QuadraticLorentzAttention(beta=beta, K=K)
    elif attn_type == "lorentzlinear":
        return LorentzLinearAttention(
            d=d_hyp, R=R, M=M, beta=beta, K=K, use_orf=use_orf, seed=seed,
        )
    elif attn_type == "hypformer":
        return HypformerLinearAttention(d=d_hyp, M=M, beta=beta, K=K)
    raise ValueError(f"Unknown attn_type: {attn_type}")


class EuclideanToHyperboloid(nn.Module):
    """Map Euclidean features to the hyperboloid via learned linear + projection."""

    def __init__(self, d_in: int, d_hyp: int, K: float = -1.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_hyp + 1)
        self.K = K

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear(x)
        return project_to_hyperboloid(h, self.K)


class HypAttnLayer(nn.Module):
    """Single attention layer with tangent-space residual and layer norm."""

    def __init__(self, d_hyp: int, attn_type: str, K: float = -1.0,
                 R: int = 2, M: int = 64, beta: float = 1.0,
                 use_orf: bool = False, dropout: float = 0.1, seed: int = 42):
        super().__init__()
        self.attn = _make_attn(attn_type, d_hyp, R, M, beta, K, use_orf, seed)
        self.norm = nn.LayerNorm(d_hyp + 1)
        self.ffn = nn.Sequential(
            nn.Linear(d_hyp + 1, d_hyp + 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hyp + 1, d_hyp + 1),
            nn.Dropout(dropout),
        )
        self.K = K
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        h_attn = self.attn(h, h, h)
        h = self.norm(h + self.dropout(h_attn))
        h = h + self.ffn(h)
        return project_to_hyperboloid(h, self.K)


class HypGraphTransformer(nn.Module):
    """Multi-layer hyperbolic graph Transformer for node classification.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_hyp : int
        Space-like dimension on the hyperboloid (embedding is d_hyp+1).
    n_classes : int
        Number of output classes.
    n_layers : int
        Number of attention layers.
    attn_type : str
        One of 'quadratic', 'lorentzlinear', 'hypformer'.
    R, M, beta, K : attention hyperparameters.
    dropout : float
        Dropout rate for FFN and attention residual.
    """

    def __init__(
        self,
        d_in: int,
        d_hyp: int,
        n_classes: int,
        attn_type: str = "lorentzlinear",
        n_layers: int = 2,
        R: int = 2,
        M: int = 64,
        beta: float = 1.0,
        K: float = -1.0,
        use_orf: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = EuclideanToHyperboloid(d_in, d_hyp, K=K)

        self.layers = nn.ModuleList([
            HypAttnLayer(
                d_hyp, attn_type, K=K, R=R, M=M, beta=beta,
                use_orf=use_orf, dropout=dropout, seed=42 + i,
            )
            for i in range(n_layers)
        ])

        self.classifier = nn.Linear(d_hyp + 1, n_classes)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(h)
