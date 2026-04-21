"""Performer-style positive random features for the space-like kernel factor.

Approximates  exp(beta * qs^T ks)  via
  E[phi_E(qs)^T phi_E(ks)] = exp(beta * qs^T ks)
where
  phi_E(xs) = (1/sqrt(M)) * exp(-||xs||^2/2) * [exp(w1^T xs), ..., exp(wM^T xs)]
with  w_i ~ N(0, beta*I).

Overflow is prevented by subtracting a *shared* scalar h (the global maximum
of all log-features in the batch) so that exp() never overflows.  Because h is
the same for every sample, it cancels in both numerator and denominator of
normalised attention, and can be tracked as ``log_scale`` for absolute kernel
estimation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SpaceFeatureMap(nn.Module):
    """Performer-style positive random features for exp(beta * qs^T ks)."""

    def __init__(
        self,
        d_space: int,
        M: int,
        beta: float = 1.0,
        seed: int = 42,
        use_orf: bool = False,
    ):
        super().__init__()
        self.d_space = d_space
        self.M = M
        self.beta = beta
        self.use_orf = use_orf
        self._log_scale: Tensor | None = None

        gen = torch.Generator().manual_seed(seed)
        if use_orf:
            omega = self._build_orf_omega(M, d_space, beta, gen)
        else:
            omega = torch.randn(M, d_space, generator=gen) * math.sqrt(beta)
        self.register_buffer("omega", omega)  # (M, d_space), fixed

    @staticmethod
    def _build_orf_omega(
        M: int, d: int, beta: float, gen: torch.Generator,
    ) -> Tensor:
        """Construct ORF projection matrix via QR of Gaussian blocks.

        Each d x d Gaussian block is QR-decomposed to yield a uniformly random
        orthogonal matrix Q, then each row is rescaled by the chi-distributed
        norm of the original Gaussian row (preserving the expected row-norm
        distribution).  Multiple blocks are stacked when M > d.
        """
        n_blocks = math.ceil(M / d)
        blocks: list[Tensor] = []
        for _ in range(n_blocks):
            G = torch.randn(d, d, generator=gen)
            norms = G.norm(dim=1, keepdim=True)  # chi-distributed
            Q, _ = torch.linalg.qr(G)
            blocks.append(Q * norms)
        omega = torch.cat(blocks, dim=0)[:M] * math.sqrt(beta)
        return omega

    def forward(self, xs: Tensor) -> Tensor:
        """Compute phi_E(xs).

        Args:
            xs: space-like components, shape (..., d_space).

        Returns:
            Feature vector, shape (..., M).  Always non-negative.
        """
        proj = xs @ self.omega.T  # (..., M)
        xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)  # (..., 1)

        # log(phi_j) = -||xs||^2/2 + omega_j^T xs
        log_phi = -self.beta * xs_norm_sq / 2.0 + proj

        # Subtract the global max across all samples AND features in this
        # call.  Because it is a single scalar, it factors out of every
        # dot-product and cancels in normalised attention.
        h = log_phi.detach().max()
        self._log_scale = h
        log_phi = log_phi - h

        phi = torch.exp(log_phi) / math.sqrt(self.M)
        return phi

    @property
    def log_scale(self) -> Tensor:
        """Shared log-scale factor subtracted for overflow safety.

        For normalised attention the factor cancels automatically.
        For absolute kernel estimation multiply back:
            ``kernel *= exp(log_scale_q + log_scale_k)``
        """
        if self._log_scale is None:
            raise RuntimeError("forward() has not been called yet")
        return self._log_scale
