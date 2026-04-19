"""Full O(N^2) Lorentzian attention — the exact ground truth.

Computes  alpha_ij = exp(beta * <q_i, k_j>_L) / sum_l exp(beta * <q_i, k_l>_L)
then aggregates values and projects back onto the hyperboloid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..hyperbolic_ops import minkowski_inner, project_to_hyperboloid


class QuadraticLorentzAttention(nn.Module):
    def __init__(self, beta: float = 1.0, K: float = -1.0):
        super().__init__()
        self.beta = beta
        self.K = K

    def forward(self, Q: Tensor, K_in: Tensor, V: Tensor, causal: bool = False) -> Tensor:
        """Exact quadratic Lorentzian attention.

        Args:
            Q, K_in: (..., N, d+1)  on the hyperboloid.
            V:       (..., N, d_v).
            causal:  apply lower-triangular mask if True.

        Returns:
            Output projected onto the hyperboloid, (..., N, d_v).
        """
        # Pairwise Minkowski inner products — (..., N_q, N_k)
        scores = torch.einsum("...id,...jd->...ij", Q[..., 1:], K_in[..., 1:]) \
               - torch.einsum("...i,...j->...ij", Q[..., 0], K_in[..., 0])
        scores = scores * self.beta

        if causal:
            N = scores.shape[-1]
            mask = torch.triu(torch.ones(N, N, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (..., N_q, N_k)
        y = torch.einsum("...ij,...jd->...id", weights, V)
        return project_to_hyperboloid(y, self.K)

    def attention_weights(self, Q: Tensor, K_in: Tensor) -> Tensor:
        """Return the raw softmax attention weights (for diagnostics)."""
        scores = torch.einsum("...id,...jd->...ij", Q[..., 1:], K_in[..., 1:]) \
               - torch.einsum("...i,...j->...ij", Q[..., 0], K_in[..., 0])
        return torch.softmax(scores * self.beta, dim=-1)
