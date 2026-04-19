"""Experiment 1.4 — Equivariance Under Lorentz Boosts.

Apply Lorentz boosts of increasing rapidity and measure attention-output
discrepancy: ||Attn(Λq, Λk, Λv) − Λ·Attn(q, k, v)||.
Compare LorentzLinear, Hypformer, and quadratic Lorentzian.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from core.hyperbolic_ops import sample_hyperboloid, lorentz_boost
from core.lorentz_linear import LorentzLinearAttention
from core.baselines.quadratic_lorentz import QuadraticLorentzAttention
from core.baselines.hypformer_linear import HypformerLinearAttention

DTYPE = torch.float64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier1")


def _discrepancy(attn_fn, pts, V, v_dir, phi):
    """Measure ||Attn(Λx, Λx, ΛV) - Λ·Attn(x, x, V)||."""
    out_orig = attn_fn(pts, pts, V)
    boosted_pts = lorentz_boost(pts, v_dir, phi)
    boosted_V = lorentz_boost(V, v_dir, phi)
    out_boosted = attn_fn(boosted_pts, boosted_pts, boosted_V)
    out_orig_boosted = lorentz_boost(out_orig, v_dir, phi)
    return torch.linalg.norm(out_boosted - out_orig_boosted, dim=-1).mean().item()


def run(cfg: dict) -> dict:
    N = cfg.get("N", 100)
    d = cfg.get("d", 16)
    K = cfg.get("K", -1.0)
    R = cfg.get("R", 2)
    M = cfg.get("M", 64)
    beta = cfg.get("beta", 1.0)
    rapidities = cfg.get("rapidities", [0.1, 0.3, 0.5, 0.7, 0.9])
    seed = cfg.get("seed", 0)

    gen = torch.Generator().manual_seed(seed)
    pts = sample_hyperboloid(N, d, K=K, scale=0.5, dtype=DTYPE, generator=gen)
    V = pts.clone()

    v_dir = torch.zeros(d, dtype=DTYPE)
    v_dir[0] = 1.0  # boost in first spatial direction

    quad = QuadraticLorentzAttention(beta=beta, K=K)
    ll = LorentzLinearAttention(d=d, R=R, M=M, beta=beta, K=K).to(dtype=DTYPE)
    hyp = HypformerLinearAttention(d=d, M=M, beta=beta, K=K).to(dtype=DTYPE)

    results: dict = {"rapidities": rapidities, "quadratic": [], "lorentz_linear": [], "hypformer": []}

    for phi in rapidities:
        dq = _discrepancy(quad.forward, pts, V, v_dir, phi)
        dl = _discrepancy(ll.forward, pts, V, v_dir, phi)
        dh = _discrepancy(hyp.forward, pts, V, v_dir, phi)
        results["quadratic"].append(dq)
        results["lorentz_linear"].append(dl)
        results["hypformer"].append(dh)
        print(f"  rapidity={phi:.1f}  quad={dq:.4e}  LL={dl:.4e}  Hypformer={dh:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rapidities, results["quadratic"], "o-", label="Quadratic (exact)", color="#1b9e77")
    ax.plot(rapidities, results["lorentz_linear"], "s-", label="LorentzLinear", color="#d95f02")
    ax.plot(rapidities, results["hypformer"], "^-", label="Hypformer (space-only)", color="#7570b3")
    ax.set_xlabel("Boost rapidity φ")
    ax.set_ylabel("Mean output discrepancy")
    ax.set_title(f"Equivariance Under Lorentz Boosts  (d={d}, K={K}, R={R}, M={M})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_equivariance.png"), dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    run({})
