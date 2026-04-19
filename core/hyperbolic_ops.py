"""Lorentz model operations on the hyperboloid L^n_K.

All tensors have shape (..., d+1) where index 0 is the time-like coordinate.
Curvature K < 0 throughout; the hyperboloid satisfies <x,x>_L = 1/K.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Minkowski inner product and derived quantities
# ---------------------------------------------------------------------------

def minkowski_inner(x: Tensor, y: Tensor) -> Tensor:
    """<x, y>_L = -x0*y0 + xs^T ys.  Returns (...,) scalar per batch element."""
    time = -x[..., 0] * y[..., 0]
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return time + space


def minkowski_norm_sq(x: Tensor) -> Tensor:
    """<x, x>_L.  Negative for time-like vectors on the hyperboloid."""
    return minkowski_inner(x, x)


def split_time_space(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return (x0, xs) where x0 has shape (..., 1)."""
    return x[..., 0:1], x[..., 1:]


# ---------------------------------------------------------------------------
# Hyperboloid membership
# ---------------------------------------------------------------------------

def check_on_hyperboloid(x: Tensor, K: float, atol: float = 1e-4) -> bool:
    """Check that every point satisfies <x,x>_L ≈ 1/K."""
    norm_sq = minkowski_norm_sq(x)
    return bool(torch.allclose(norm_sq, torch.tensor(1.0 / K, dtype=x.dtype, device=x.device), atol=atol))


def project_to_hyperboloid(x: Tensor, K: float) -> Tensor:
    """Project x onto the hyperboloid via uniform rescaling (LorentzLinear Section 4.5).

    Rescales the full (d+1)-vector so that <x,x>_L = 1/K, preserving its
    direction in Minkowski space.  This is equivalent to the Einstein midpoint
    projection and is Lorentz-boost equivariant.

        o = x / sqrt(|K| * |<x, x>_L|)

    For near-degenerate vectors (Minkowski norm close to zero), falls back to
    recomputing x0 from xs to guarantee a valid hyperboloid point.
    """
    norm_sq = minkowski_norm_sq(x)  # (...)
    abs_norm_sq = torch.abs(norm_sq)

    # Uniform rescaling (equivariant path)
    abs_K = abs(K)
    scale = 1.0 / torch.sqrt(abs_K * abs_norm_sq + 1e-8)
    projected = x * scale.unsqueeze(-1)

    # Ensure upper sheet
    sign = projected[..., 0].sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    projected = projected * sign.unsqueeze(-1)

    # Fall back for near-degenerate vectors: recompute x0 from xs
    degenerate = abs_norm_sq < 1e-4
    if degenerate.any():
        xs = x[..., 1:]
        xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)
        x0_safe = torch.sqrt(xs_norm_sq - 1.0 / K)
        fallback = torch.cat([x0_safe, xs], dim=-1)
        projected = torch.where(degenerate.unsqueeze(-1), fallback, projected)

    return projected


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_hyperboloid(
    N: int, d: int, K: float = -1.0,
    *, scale: float = 1.0, dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu", generator: torch.Generator | None = None,
) -> Tensor:
    """Sample N points on L^d_K.

    d is the *space* dimension; output has shape (N, d+1).
    Space components are sampled from N(0, scale^2 I), then x0 = sqrt(||xs||^2 - 1/K).
    """
    xs = torch.randn(N, d, dtype=dtype, device=device, generator=generator) * scale
    xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)  # (N, 1)
    x0 = torch.sqrt(xs_norm_sq - 1.0 / K)             # 1/K is negative, so -1/K > 0
    return torch.cat([x0, xs], dim=-1)


# ---------------------------------------------------------------------------
# Lorentz boosts
# ---------------------------------------------------------------------------

def lorentz_boost(x: Tensor, v_direction: Tensor, v_magnitude: float) -> Tensor:
    """Apply a Lorentz boost to x.

    Args:
        x: points on the hyperboloid, shape (..., d+1).
        v_direction: unit vector in the spatial subspace, shape (d,).
                     Only the spatial dimensions are used.
        v_magnitude: rapidity (phi).  The "velocity" is tanh(phi).

    Returns:
        Boosted points, same shape as x.
    """
    phi = v_magnitude
    cosh_phi = torch.cosh(torch.tensor(phi, dtype=x.dtype, device=x.device))
    sinh_phi = torch.sinh(torch.tensor(phi, dtype=x.dtype, device=x.device))

    v = v_direction.to(dtype=x.dtype, device=x.device)
    v = v / (v.norm() + 1e-12)  # ensure unit

    x0 = x[..., 0]                          # (...,)
    xs = x[..., 1:]                          # (..., d)
    xs_par = (xs * v).sum(dim=-1)            # component parallel to v

    new_x0 = cosh_phi * x0 + sinh_phi * xs_par
    new_xs_par = sinh_phi * x0 + cosh_phi * xs_par
    new_xs = xs + (new_xs_par - xs_par).unsqueeze(-1) * v  # replace parallel component

    return torch.cat([new_x0.unsqueeze(-1), new_xs], dim=-1)


# ---------------------------------------------------------------------------
# Exponential and logarithmic maps
# ---------------------------------------------------------------------------

def exp_map(x: Tensor, v: Tensor, K: float) -> Tensor:
    """Exponential map on L^n_K: map tangent vector v at x to a point on the manifold.

    v must be in the tangent space at x, i.e. <x, v>_L = 0.
    """
    c = -K  # c > 0
    sqrt_c = c ** 0.5
    v_norm = torch.sqrt(torch.clamp(minkowski_inner(v, v), min=1e-12))
    coeff = torch.cosh(sqrt_c * v_norm)
    sinc = torch.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm + 1e-12)
    return coeff.unsqueeze(-1) * x + sinc.unsqueeze(-1) * v


def log_map(x: Tensor, y: Tensor, K: float) -> Tensor:
    """Logarithmic map on L^n_K: tangent vector at x pointing toward y."""
    c = -K
    sqrt_c = c ** 0.5
    inner = minkowski_inner(x, y)  # (...)
    inner = torch.clamp(inner, max=-1.0 / (-K) - 1e-7)  # keep in valid range
    alpha = torch.acosh(torch.clamp(-c * inner, min=1.0))
    diff = y - inner.unsqueeze(-1) / (1.0 / K) * x  # unnormalised tangent direction
    diff_norm = torch.sqrt(torch.clamp(minkowski_inner(diff, diff), min=1e-12))
    return alpha.unsqueeze(-1) / (sqrt_c * diff_norm.unsqueeze(-1) + 1e-12) * diff
