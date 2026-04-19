"""Hypformer-style space-only linear attention baseline.

Discards the time-like component x0, applies Performer-style positive random
features to the space-like components xs only, then reconstructs x0 via
projection.  This breaks Lorentz boost equivariance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..space_features import SpaceFeatureMap
from ..hyperbolic_ops import project_to_hyperboloid


class HypformerLinearAttention(nn.Module):
    def __init__(self, d: int, M: int = 64, beta: float = 1.0, K: float = -1.0, seed: int = 42):
        super().__init__()
        self.d = d
        self.M = M
        self.K = K
        self.space = SpaceFeatureMap(d, M, beta, seed=seed)

    def forward(self, Q: Tensor, K_in: Tensor, V: Tensor, causal: bool = False) -> Tensor:
        """Space-only linear attention (Hypformer approximation).

        Q, K_in: (..., N, d+1) on the hyperboloid.
        V:       (..., N, d_v).
        """
        qs = Q[..., 1:]     # (..., N, d)
        ks = K_in[..., 1:]

        phi_q = self.space(qs)  # (..., N, M)
        phi_k = self.space(ks)

        if causal:
            return self._causal(phi_q, phi_k, V)

        S = torch.einsum("...ni,...nj->...ij", phi_k, V)  # (..., M, d_v)
        z = phi_k.sum(dim=-2)                               # (..., M)

        num = torch.einsum("...ni,...ij->...nj", phi_q, S)
        den = torch.einsum("...ni,...i->...n", phi_q, z)
        den = den.clamp(min=1e-6)

        y = num / den.unsqueeze(-1)
        return project_to_hyperboloid(y, self.K)

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
        result = project_to_hyperboloid(y, self.K)
        return result if is_batched else result.squeeze(0)
