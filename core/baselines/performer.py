"""Euclidean Performer baseline — no hyperbolic geometry at all.

Uses positive random features to approximate exp(q^T k) in flat space.
Included as an ablation showing what is lost without hyperbolic structure.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PerformerAttention(nn.Module):
    def __init__(self, d_full: int, M: int = 64, seed: int = 42):
        """
        Args:
            d_full: full input dimension (d+1 for hyperboloid embeddings).
            M: number of random features.
        """
        super().__init__()
        self.M = M
        gen = torch.Generator().manual_seed(seed)
        omega = torch.randn(M, d_full, generator=gen)
        self.register_buffer("omega", omega)

    def _phi(self, x: Tensor) -> Tensor:
        proj = x @ self.omega.T  # (..., M)
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        log_phi = -x_norm_sq / 2.0 + proj
        max_log = log_phi.max(dim=-1, keepdim=True).values
        return torch.exp(log_phi - max_log) / math.sqrt(self.M)

    def forward(self, Q: Tensor, K_in: Tensor, V: Tensor, causal: bool = False) -> Tensor:
        phi_q = self._phi(Q)
        phi_k = self._phi(K_in)

        if causal:
            return self._causal(phi_q, phi_k, V)

        S = torch.einsum("...ni,...nj->...ij", phi_k, V)
        z = phi_k.sum(dim=-2)

        num = torch.einsum("...ni,...ij->...nj", phi_q, S)
        den = torch.einsum("...ni,...i->...n", phi_q, z).clamp(min=1e-6)
        return num / den.unsqueeze(-1)

    def _causal(self, phi_q: Tensor, phi_k: Tensor, V: Tensor) -> Tensor:
        is_batched = phi_q.dim() == 3
        if not is_batched:
            phi_q = phi_q.unsqueeze(0)
            phi_k = phi_k.unsqueeze(0)
            V = V.unsqueeze(0)

        B, N, M = phi_q.shape
        d_v = V.shape[-1]
        S = phi_q.new_zeros(B, M, d_v)
        z = phi_q.new_zeros(B, M)
        outputs = []
        for t in range(N):
            S = S + torch.einsum("bi,bj->bij", phi_k[:, t], V[:, t])
            z = z + phi_k[:, t]
            num = torch.einsum("bi,bij->bj", phi_q[:, t], S)
            den = (phi_q[:, t] * z).sum(-1).clamp(min=1e-6)
            outputs.append(num / den.unsqueeze(-1))

        y = torch.stack(outputs, dim=1)
        return y if is_batched else y.squeeze(0)
