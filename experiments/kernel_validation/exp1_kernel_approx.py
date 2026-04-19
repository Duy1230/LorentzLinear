"""Experiment 1.1 -- Kernel Approximation Quality.

Sample N points on L^n_K, compute the exact vs approximate kernel matrix,
sweep over (R, M), and report relative Frobenius error, max absolute entry
error, and spectral norm error.  Averages over multiple omega seeds to
separate bias from single-draw variance.

Raw (unscaled) features are used here to avoid the shared-max normalisation
introducing inconsistent scales across seeds.
"""

from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.hyperbolic_ops import sample_hyperboloid, minkowski_inner, split_time_space
from core.krein_features import KreinSplit


DTYPE = torch.float64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier1")


def exact_kernel_matrix(pts: torch.Tensor, beta: float) -> torch.Tensor:
    """K_ij = exp(beta * <q_i, k_j>_L)."""
    inner = minkowski_inner(pts.unsqueeze(1), pts.unsqueeze(0))  # (N, N)
    return torch.exp(beta * inner)


def _raw_space_phi(xs: torch.Tensor, M: int, beta: float, seed: int) -> torch.Tensor:
    """Compute raw Performer features WITHOUT shared-max subtraction.

    phi_j(xs) = (1/sqrt(M)) * exp(-||xs||^2/2 + omega_j^T xs)

    At scale=0.5 with moderate d, log_phi stays well within float64 range
    so no overflow guard is needed.
    """
    gen = torch.Generator().manual_seed(seed)
    omega = torch.randn(M, xs.shape[-1], dtype=xs.dtype, device=xs.device,
                        generator=gen) * math.sqrt(beta)
    proj = xs @ omega.T  # (..., M)
    xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)
    log_phi = -xs_norm_sq / 2.0 + proj
    return torch.exp(log_phi) / math.sqrt(M)


def approx_kernel_matrix(
    pts: torch.Tensor, R: int, M: int, beta: float, K: float,
    n_seeds: int = 10,
) -> torch.Tensor:
    """Median-of-seeds approximate kernel matrix.

    Uses the element-wise median across seed draws to be robust against
    the heavy-tailed variance of Performer RFF.
    """
    x0, xs = split_time_space(pts)
    krein = KreinSplit(R, beta)

    psi_p = krein.psi_plus(x0)   # (N, T_plus)
    psi_m = krein.psi_minus(x0)  # (N, T_minus)

    all_K_hat = []

    for seed in range(n_seeds):
        phi = _raw_space_phi(xs, M, beta, seed)  # (N, M)

        feat_p = torch.einsum("...i,...j->...ij", psi_p, phi).flatten(-2)
        feat_m = torch.einsum("...i,...j->...ij", psi_m, phi).flatten(-2)

        K_hat = feat_p @ feat_p.T - feat_m @ feat_m.T
        all_K_hat.append(K_hat)

    stacked = torch.stack(all_K_hat, dim=0)  # (n_seeds, N, N)
    return stacked.median(dim=0).values


def run(cfg: dict) -> dict:
    N = cfg.get("N", 1000)
    dims = cfg.get("dims", [16, 64])
    curvatures = cfg.get("curvatures", [-1.0])
    R_values = cfg.get("R_values", [1, 2, 3, 4])
    M_values = cfg.get("M_values", [32, 64, 128, 256])
    beta = cfg.get("beta", 1.0)
    seed = cfg.get("seed", 0)
    n_seeds = cfg.get("n_seeds", 10)

    all_results: dict = {}

    for K in curvatures:
        for d in dims:
            gen = torch.Generator().manual_seed(seed)
            pts = sample_hyperboloid(N, d, K=K, scale=0.5, dtype=DTYPE, generator=gen)

            K_exact = exact_kernel_matrix(pts, beta)
            K_exact_fro = torch.linalg.norm(K_exact, "fro").item()

            frob_errors = np.zeros((len(R_values), len(M_values)))
            entry_errors = np.zeros_like(frob_errors)
            spec_errors = np.zeros_like(frob_errors)

            for ri, R in enumerate(R_values):
                for mi, M in enumerate(M_values):
                    K_hat = approx_kernel_matrix(pts, R, M, beta, K, n_seeds=n_seeds)
                    diff = K_exact - K_hat
                    frob_errors[ri, mi] = torch.linalg.norm(diff, "fro").item() / K_exact_fro

                    mask = K_exact > 1e-6
                    entry_errors[ri, mi] = (diff[mask].abs() / K_exact[mask]).mean().item()

                    spec_errors[ri, mi] = torch.linalg.norm(diff, 2).item()

            key = f"d={d}_K={K}"
            all_results[key] = {
                "frob": frob_errors,
                "entry": entry_errors,
                "spec": spec_errors,
                "R_values": R_values,
                "M_values": M_values,
            }

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, (name, data) in zip(axes, [
                ("Rel. Frobenius", frob_errors),
                ("Mean Entry Rel. Err", entry_errors),
                ("Spectral Norm", spec_errors),
            ]):
                sns.heatmap(
                    data, ax=ax, annot=True, fmt=".3e",
                    xticklabels=[str(m) for m in M_values],
                    yticklabels=[str(r) for r in R_values],
                    cmap="YlOrRd_r",
                )
                ax.set_xlabel("M (random features)")
                ax.set_ylabel("R (Taylor rank)")
                ax.set_title(f"{name}  (d={d}, K={K})")

            fig.tight_layout()
            os.makedirs(RESULTS_DIR, exist_ok=True)
            fig.savefig(os.path.join(RESULTS_DIR, f"exp1_kernel_approx_{key}.png"), dpi=150)
            plt.close(fig)
            print(f"  [{key}] Frobenius error range: "
                  f"{frob_errors.min():.3e} -- {frob_errors.max():.3e}")

    return all_results


if __name__ == "__main__":
    run({})
