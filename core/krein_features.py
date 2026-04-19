"""Kreĭn decomposition for the time-like kernel factor.

Decomposes exp(-beta * q0 * k0) into two channels ψ₊, ψ₋ such that
  ψ₊(q0)^T ψ₊(k0)  -  ψ₋(q0)^T ψ₋(k0)  ≈  exp(-beta * q0 * k0)
using the identity  -q0*k0 = -(q0^2+k0^2)/2 + (q0-k0)^2/2
followed by a Taylor expansion of the correction term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass
class _Term:
    """One scalar term in the Kreĭn feature vector."""
    weight: float          # sqrt(|coefficient|)
    exponent_q: int        # power of q0 (or combined via sum/diff)
    exponent_k: int        # power of k0
    kind: str              # "central", "sum", or "diff"


class KreinSplit:
    """Precomputes the +/- channel polynomial terms from the binomial expansion.

    After construction, call ``psi_plus(x0)`` / ``psi_minus(x0)`` to obtain
    feature vectors of shape (..., feature_dim).
    """

    def __init__(self, R: int, beta: float):
        self.R = R
        self.beta = beta
        self._plus_terms: list[_Term] = []
        self._minus_terms: list[_Term] = []
        self._build(R, beta)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, R: int, beta: float) -> None:
        for r in range(R):
            c_r = (beta / 2.0) ** r / math.factorial(r)

            # Central binomial term: C(2r, r) * (-1)^r * (q0*k0)^r
            w_rr = c_r * math.comb(2 * r, r) * ((-1) ** r)
            term = _Term(weight=math.sqrt(abs(w_rr)), exponent_q=r, exponent_k=r, kind="central")
            if w_rr >= 0:
                self._plus_terms.append(term)
            else:
                self._minus_terms.append(term)

            # Paired off-center terms  (j < r)
            for j in range(r):
                w = c_r * math.comb(2 * r, j) * ((-1) ** j)
                a = 2 * r - j
                b = j
                sw = math.sqrt(abs(w))
                if w >= 0:
                    self._plus_terms.append(_Term(sw, a, b, "sum"))
                    self._minus_terms.append(_Term(sw, a, b, "diff"))
                else:
                    self._plus_terms.append(_Term(sw, a, b, "diff"))
                    self._minus_terms.append(_Term(sw, a, b, "sum"))

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Dimension of ψ₊ (same as ψ₋ by construction)."""
        return max(len(self._plus_terms), len(self._minus_terms))

    def _compute_features(self, x0: Tensor, terms: list[_Term]) -> Tensor:
        """Build the feature vector for one channel.

        x0: (..., 1)  —  the time-like component.
        Returns (..., len(terms)).
        """
        x0_sq = x0.squeeze(-1)  # (...,)
        decay = torch.exp(-self.beta * x0_sq * x0_sq / 2.0)  # separable decay

        feats: list[Tensor] = []
        for t in terms:
            if t.kind == "central":
                f = t.weight * x0_sq.pow(t.exponent_q)
            elif t.kind == "sum":
                f = t.weight * (x0_sq.pow(t.exponent_q) + x0_sq.pow(t.exponent_k)) / math.sqrt(2)
            else:  # diff
                f = t.weight * (x0_sq.pow(t.exponent_q) - x0_sq.pow(t.exponent_k)) / math.sqrt(2)
            feats.append(f)

        if not feats:
            return torch.zeros(*x0_sq.shape, 0, dtype=x0.dtype, device=x0.device)

        return decay.unsqueeze(-1) * torch.stack(feats, dim=-1)

    def psi_plus(self, x0: Tensor) -> Tensor:
        """ψ₊(x0), shape (..., feature_dim_plus)."""
        return self._compute_features(x0, self._plus_terms)

    def psi_minus(self, x0: Tensor) -> Tensor:
        """ψ₋(x0), shape (..., feature_dim_minus)."""
        return self._compute_features(x0, self._minus_terms)
