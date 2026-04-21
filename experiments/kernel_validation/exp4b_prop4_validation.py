"""Experiment 4b -- Proposition 4 Validation.

Plots three curves vs rapidity on the same axis:
  (a) Raw discrepancy  ||Attn(Lx) - L*Attn(x)||
  (b) epsilon(Lx)      kernel approximation error on boosted points
  (c) Ratio            discrepancy / epsilon   (Prop 4 predicts <= 2)

Also includes Hypformer for comparison.
"""

from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from core.hyperbolic_ops import (
    sample_hyperboloid, lorentz_boost, minkowski_inner, split_time_space,
)
from core.lorentz_linear import LorentzLinearAttention
from core.baselines.quadratic_lorentz import QuadraticLorentzAttention
from core.baselines.hypformer_linear import HypformerLinearAttention
from core.krein_features import KreinSplit

DTYPE = torch.float64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier1")


def _attn_discrepancy(attn_fn, pts, V, v_dir, phi):
    """||Attn(Lx, Lx, LV) - L*Attn(x, x, V)||, per-token mean."""
    out_orig = attn_fn(pts, pts, V)
    boosted_pts = lorentz_boost(pts, v_dir, phi)
    boosted_V = lorentz_boost(V, v_dir, phi)
    out_boosted = attn_fn(boosted_pts, boosted_pts, boosted_V)
    out_orig_boosted = lorentz_boost(out_orig, v_dir, phi)
    return torch.linalg.norm(out_boosted - out_orig_boosted, dim=-1).mean().item()


def _kernel_discrepancy(pts, R, M, beta, v_dir, phi, n_seeds=10):
    """Mean |K_approx(Lx_i, Lx_j) - K_approx(x_i, x_j)| over all pairs.

    This is the kernel-level equivariance error that Prop 4 bounds by 2*epsilon.
    """
    boosted_pts = lorentz_boost(pts, v_dir, phi)
    x0, xs = split_time_space(pts)
    bx0, bxs = split_time_space(boosted_pts)
    krein = KreinSplit(R, beta)

    psi_p = krein.psi_plus(x0)
    psi_m = krein.psi_minus(x0)
    bpsi_p = krein.psi_plus(bx0)
    bpsi_m = krein.psi_minus(bx0)

    errors = []
    for seed in range(n_seeds):
        gen = torch.Generator().manual_seed(seed)
        d_space = xs.shape[-1]
        omega = torch.randn(M, d_space, dtype=xs.dtype, generator=gen) * math.sqrt(beta)

        def _phi(xs_in):
            proj = xs_in @ omega.T
            ns = (xs_in * xs_in).sum(dim=-1, keepdim=True)
            return torch.exp(-ns / 2.0 + proj) / math.sqrt(M)

        phi_s = _phi(xs)
        bphi_s = _phi(bxs)

        fp = torch.einsum("...i,...j->...ij", psi_p, phi_s).flatten(-2)
        fm = torch.einsum("...i,...j->...ij", psi_m, phi_s).flatten(-2)
        K_orig = fp @ fp.T - fm @ fm.T

        bfp = torch.einsum("...i,...j->...ij", bpsi_p, bphi_s).flatten(-2)
        bfm = torch.einsum("...i,...j->...ij", bpsi_m, bphi_s).flatten(-2)
        K_boost = bfp @ bfp.T - bfm @ bfm.T

        errors.append((K_boost - K_orig).abs().mean().item())

    return float(np.median(errors))


def _kernel_epsilon(pts, R, M, beta, K, n_seeds=10):
    """Mean absolute per-entry kernel error on given points."""
    x0, xs = split_time_space(pts)
    krein = KreinSplit(R, beta)
    psi_p = krein.psi_plus(x0)
    psi_m = krein.psi_minus(x0)

    inner = minkowski_inner(pts.unsqueeze(1), pts.unsqueeze(0))
    K_exact = torch.exp(beta * inner)

    errors = []
    for seed in range(n_seeds):
        gen = torch.Generator().manual_seed(seed)
        d_space = xs.shape[-1]
        omega = torch.randn(M, d_space, dtype=xs.dtype, generator=gen) * math.sqrt(beta)
        proj = xs @ omega.T
        xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)
        phi = torch.exp(-xs_norm_sq / 2.0 + proj) / math.sqrt(M)

        feat_p = torch.einsum("...i,...j->...ij", psi_p, phi).flatten(-2)
        feat_m = torch.einsum("...i,...j->...ij", psi_m, phi).flatten(-2)
        K_hat = feat_p @ feat_p.T - feat_m @ feat_m.T

        mask = K_exact > 1e-6
        errors.append((K_exact - K_hat)[mask].abs().mean().item())

    return float(np.median(errors))


def run(cfg: dict) -> dict:
    N = cfg.get("N", 100)
    d = cfg.get("d", 16)
    K = cfg.get("K", -1.0)
    R = cfg.get("R", 2)
    M = cfg.get("M", 64)
    beta = cfg.get("beta", 1.0)
    rapidities = cfg.get("rapidities", [0.1, 0.3, 0.5, 0.7, 0.9])
    seed = cfg.get("seed", 0)
    n_seeds = cfg.get("n_seeds", 10)

    gen = torch.Generator().manual_seed(seed)
    pts = sample_hyperboloid(N, d, K=K, scale=0.5, dtype=DTYPE, generator=gen)
    V = pts.clone()

    v_dir = torch.zeros(d, dtype=DTYPE)
    v_dir[0] = 1.0

    ll = LorentzLinearAttention(d=d, R=R, M=M, beta=beta, K=K).to(dtype=DTYPE)
    hyp = HypformerLinearAttention(d=d, M=M, beta=beta, K=K).to(dtype=DTYPE)

    attn_disc_ll, attn_disc_hyp = [], []
    kernel_disc_ll = []
    epsilons = []
    ratios = []

    print(f"  {'Rapidity':>8}  {'AttnDisc':>10}  {'KernDisc':>10}  "
          f"{'eps(Lx)':>10}  {'Ratio':>10}  {'Hyp':>10}")
    for phi in rapidities:
        d_ll = _attn_discrepancy(ll.forward, pts, V, v_dir, phi)
        d_hyp = _attn_discrepancy(hyp.forward, pts, V, v_dir, phi)
        kd = _kernel_discrepancy(pts, R, M, beta, v_dir, phi, n_seeds=n_seeds)
        boosted_pts = lorentz_boost(pts, v_dir, phi)
        eps = _kernel_epsilon(boosted_pts, R, M, beta, K, n_seeds=n_seeds)

        ratio = kd / (eps + 1e-15)

        attn_disc_ll.append(d_ll)
        attn_disc_hyp.append(d_hyp)
        kernel_disc_ll.append(kd)
        epsilons.append(eps)
        ratios.append(ratio)

        print(f"  {phi:8.1f}  {d_ll:10.4e}  {kd:10.4e}  "
              f"{eps:10.4e}  {ratio:10.4f}  {d_hyp:10.4e}")

    results = {
        "rapidities": rapidities,
        "attn_discrepancy_ll": attn_disc_ll,
        "attn_discrepancy_hyp": attn_disc_hyp,
        "kernel_discrepancy_ll": kernel_disc_ll,
        "epsilon": epsilons,
        "ratio": ratios,
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(rapidities, attn_disc_ll, "s-", label="LorentzLinear (attn)", color="#d95f02")
    ax.plot(rapidities, attn_disc_hyp, "^-",
            label="Hypformer (lower by coincidence)", color="#7570b3", alpha=0.7)
    ax.plot(rapidities, kernel_disc_ll, "o--",
            label="LorentzLinear (kernel)", color="#1b9e77")
    ax.set_xlabel("Boost rapidity")
    ax.set_ylabel("Discrepancy")
    ax.set_title("(a) Equivariance discrepancy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rapidities, epsilons, "o-",
            label=r"$\varepsilon(\Lambda x)$", color="#1b9e77")
    ax.plot(rapidities, kernel_disc_ll, "s--",
            label="Kernel discrepancy", color="#d95f02", alpha=0.6)
    ax.set_xlabel("Boost rapidity")
    ax.set_ylabel("Error magnitude")
    ax.set_title(r"(b) $\varepsilon(\Lambda x)$ vs kernel discrepancy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(rapidities, ratios, "D-", color="#e7298a", linewidth=2)
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.7, label="Prop 4 bound (2.0)")
    ax.set_xlabel("Boost rapidity")
    ax.set_ylabel("Kernel discrepancy / epsilon")
    ax.set_title("(c) Ratio — Prop 4 validation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Proposition 4 Validation  (d={d}, K={K}, R={R}, M={M})", fontsize=13)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "exp4b_prop4_validation.png"), dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    run({})
