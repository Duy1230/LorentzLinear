"""LorentzLinear attention: O(N) linear attention on the Lorentz hyperboloid.

Combines Kreĭn time-like features with Performer space-like features via
Kronecker product, then runs two-channel linear attention with running sums.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .krein_features import KreinSplit
from .space_features import SpaceFeatureMap
from .hyperbolic_ops import split_time_space, project_to_hyperboloid


class LorentzLinearAttention(nn.Module):
    """Two-channel (Kreĭn) linear attention for the Lorentz model.

    Parameters
    ----------
    d : int
        Space-like dimension (the full embedding is d+1).
    R : int
        Taylor rank for the time-like approximation.
    M : int
        Number of random features for the space-like approximation.
    beta : float
        Temperature / kernel sharpness.
    K : float
        Hyperbolic curvature (must be negative).
    eps_reg : float
        Regularisation constant for the normaliser clamp.
    seed : int
        RNG seed for the fixed random projection matrix omega.
    """

    def __init__(
        self,
        d: int,
        R: int = 2,
        M: int = 64,
        beta: float = 1.0,
        K: float = -1.0,
        eps_reg: float = 1e-4,
        seed: int = 42,
    ):
        super().__init__()
        self.d = d
        self.R = R
        self.M = M
        self.beta = beta
        self.K = K
        self.eps_reg = eps_reg

        self.krein = KreinSplit(R, beta)
        self.space = SpaceFeatureMap(d, M, beta, seed=seed)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def feat(self, x: Tensor, channel: str) -> Tensor:
        """Compute Φ₊ or Φ₋ features for input x.

        Args:
            x: points on the hyperboloid, shape (..., d+1).
            channel: '+' or '-'.

        Returns:
            Feature vector, shape (..., time_dim * M).
        """
        x0, xs = split_time_space(x)  # (..., 1), (..., d)

        if channel == "+":
            psi = self.krein.psi_plus(x0)   # (..., T)
        else:
            psi = self.krein.psi_minus(x0)  # (..., T)

        phi = self.space(xs)  # (..., M)

        # Kronecker product → flatten last two dims
        combined = torch.einsum("...i,...j->...ij", psi, phi)  # (..., T, M)
        return combined.flatten(-2)  # (..., T*M)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, causal: bool = False) -> Tensor:
        """Compute LorentzLinear attention.

        Args:
            Q: queries on the hyperboloid, (N, d+1) or (B, N, d+1).
            K: keys, same shape as Q.
            V: values, same leading dims, last dim = d_v.
            causal: if True, use the incremental (autoregressive) variant.

        Returns:
            Output projected back onto the hyperboloid, same shape as Q.
        """
        if causal:
            return self._forward_causal(Q, K, V)
        return self._forward_full(Q, K, V)

    def _forward_full(self, Q: Tensor, K_in: Tensor, V: Tensor) -> Tensor:
        Q_p = self.feat(Q, "+")   # (..., N, F)
        Q_m = self.feat(Q, "-")
        K_p = self.feat(K_in, "+")
        K_m = self.feat(K_in, "-")

        # Accumulate KV statistics — einsum handles arbitrary batch dims
        S_p = torch.einsum("...ni,...nj->...ij", K_p, V)  # (..., F, d_v)
        S_m = torch.einsum("...ni,...nj->...ij", K_m, V)
        z_p = K_p.sum(dim=-2)  # (..., F)
        z_m = K_m.sum(dim=-2)

        N_hat = torch.einsum("...ni,...ij->...nj", Q_p, S_p) \
              - torch.einsum("...ni,...ij->...nj", Q_m, S_m)  # (..., N, d_v)
        S_hat = torch.einsum("...ni,...i->...n", Q_p, z_p) \
              - torch.einsum("...ni,...i->...n", Q_m, z_m)     # (..., N)

        inv = self._stable_inv(S_hat)  # (..., N)
        y = N_hat * inv.unsqueeze(-1)

        return project_to_hyperboloid(y, self.K)

    def _forward_causal(self, Q: Tensor, K_in: Tensor, V: Tensor) -> Tensor:
        """Autoregressive variant with running-sum recurrence."""
        is_batched = Q.dim() == 3
        if not is_batched:
            Q = Q.unsqueeze(0)
            K_in = K_in.unsqueeze(0)
            V = V.unsqueeze(0)

        B, N, _ = Q.shape
        d_v = V.shape[-1]

        Q_p = self.feat(Q, "+")  # (B, N, F)
        Q_m = self.feat(Q, "-")
        K_p = self.feat(K_in, "+")
        K_m = self.feat(K_in, "-")
        F = Q_p.shape[-1]

        S_p = Q.new_zeros(B, F, d_v)
        S_m = Q.new_zeros(B, F, d_v)
        z_p = Q.new_zeros(B, F)
        z_m = Q.new_zeros(B, F)

        outputs = []
        for t in range(N):
            kp_t = K_p[:, t, :]   # (B, F)
            km_t = K_m[:, t, :]
            v_t  = V[:, t, :]     # (B, d_v)

            S_p = S_p + torch.einsum("bi,bj->bij", kp_t, v_t)
            S_m = S_m + torch.einsum("bi,bj->bij", km_t, v_t)
            z_p = z_p + kp_t
            z_m = z_m + km_t

            qp_t = Q_p[:, t, :]  # (B, F)
            qm_t = Q_m[:, t, :]

            n_hat = torch.einsum("bi,bij->bj", qp_t, S_p) \
                  - torch.einsum("bi,bij->bj", qm_t, S_m)
            s_hat = (qp_t * z_p).sum(-1) - (qm_t * z_m).sum(-1)  # (B,)

            inv = self._stable_inv(s_hat)
            y_t = n_hat * inv.unsqueeze(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, N, d_v)
        result = project_to_hyperboloid(y, self.K)
        return result if is_batched else result.squeeze(0)

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------

    def _stable_inv(self, s: Tensor) -> Tensor:
        """Stable reciprocal: avoids division by zero near the Kreĭn cancellation region."""
        eps = self.eps_reg
        return torch.where(
            s.abs() > eps,
            1.0 / s,
            s / (s * s + eps * eps),
        )
