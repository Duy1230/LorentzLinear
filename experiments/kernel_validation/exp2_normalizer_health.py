"""Experiment 1.2 -- Normalizer Health.

Compute approximate normaliser S_hat_i and exact normaliser S_i for all
queries.  Plot the distribution of their ratio and track how often
S_hat_i < 0 or |S_hat_i| < eps_reg.

Because the SpaceFeatureMap uses a shared-max subtraction for overflow safety,
the raw dot-product normalisers are scaled by exp(-2h).  For the ratio test
we compensate by multiplying back exp(2h).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from core.hyperbolic_ops import sample_hyperboloid, minkowski_inner, split_time_space
from core.krein_features import KreinSplit
from core.space_features import SpaceFeatureMap

DTYPE = torch.float64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier1")


def run(cfg: dict) -> dict:
    N = cfg.get("N", 500)
    d = cfg.get("d", 16)
    K = cfg.get("K", -1.0)
    R_values = cfg.get("R_values", [1, 2, 3, 4])
    M_values = cfg.get("M_values", [32, 64, 128, 256])
    beta = cfg.get("beta", 1.0)
    eps_reg = cfg.get("eps_reg", 1e-4)
    seed = cfg.get("seed", 0)

    gen = torch.Generator().manual_seed(seed)
    pts = sample_hyperboloid(N, d, K=K, scale=0.5, dtype=DTYPE, generator=gen)

    inner = minkowski_inner(pts.unsqueeze(1), pts.unsqueeze(0))
    S_exact = torch.exp(beta * inner).sum(dim=-1)  # (N,)

    x0, xs = split_time_space(pts)

    results: dict = {}
    fig, axes = plt.subplots(len(R_values), len(M_values),
                             figsize=(4 * len(M_values), 3 * len(R_values)),
                             squeeze=False)

    for ri, R in enumerate(R_values):
        krein = KreinSplit(R, beta)
        psi_p = krein.psi_plus(x0)
        psi_m = krein.psi_minus(x0)

        for mi, M in enumerate(M_values):
            sfm = SpaceFeatureMap(d, M, beta, seed=42).to(dtype=DTYPE)
            phi = sfm(xs)
            h = sfm.log_scale

            feat_p = torch.einsum("...i,...j->...ij", psi_p, phi).flatten(-2)
            feat_m = torch.einsum("...i,...j->...ij", psi_m, phi).flatten(-2)

            z_p = feat_p.sum(dim=0)
            z_m = feat_m.sum(dim=0)
            S_hat_raw = (feat_p @ z_p) - (feat_m @ z_m)
            S_hat = S_hat_raw * torch.exp(2.0 * h)

            ratio = (S_hat / S_exact).detach().numpy()
            neg_frac = (S_hat < 0).float().mean().item()
            small_frac = (S_hat.abs() < eps_reg).float().mean().item()

            key = f"R={R}_M={M}"
            results[key] = {
                "ratio_mean": float(np.mean(ratio)),
                "ratio_std": float(np.std(ratio)),
                "neg_frac": neg_frac,
                "small_frac": small_frac,
            }

            ax = axes[ri][mi]
            ax.hist(ratio, bins=40, edgecolor="black", alpha=0.7)
            ax.axvline(1.0, color="red", ls="--", lw=1)
            ax.set_title(f"R={R}, M={M}\nneg={neg_frac:.1%}  small={small_frac:.1%}", fontsize=9)
            if ri == len(R_values) - 1:
                ax.set_xlabel("S_hat / S")
            if mi == 0:
                ax.set_ylabel("Count")

            print(f"  [{key}] ratio mean={np.mean(ratio):.4f}  std={np.std(ratio):.4f}  "
                  f"neg={neg_frac:.1%}  |S_hat|<eps={small_frac:.1%}")

    fig.suptitle(f"Normaliser Health  (d={d}, K={K}, N={N})", fontsize=12)
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_normalizer_health.png"), dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    run({})
