"""Experiment 1.3 — Attention Output Error.

Compares exact quadratic Lorentzian attention output o_i with the
LorentzLinear approximate output ô_i.  Reports per-query relative error
as a box plot across (R, M) configurations.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from core.hyperbolic_ops import sample_hyperboloid
from core.lorentz_linear import LorentzLinearAttention
from core.baselines.quadratic_lorentz import QuadraticLorentzAttention

DTYPE = torch.float64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier1")


def run(cfg: dict) -> dict:
    N = cfg.get("N", 200)
    d = cfg.get("d", 16)
    K = cfg.get("K", -1.0)
    R_values = cfg.get("R_values", [1, 2, 3, 4])
    M_values = cfg.get("M_values", [32, 64, 128, 256])
    beta = cfg.get("beta", 1.0)
    seed = cfg.get("seed", 0)
    run_orf = cfg.get("run_orf", False)

    gen = torch.Generator().manual_seed(seed)
    pts = sample_hyperboloid(N, d, K=K, scale=0.5, dtype=DTYPE, generator=gen)
    V = pts.clone()

    quad = QuadraticLorentzAttention(beta=beta, K=K)
    o_exact = quad(pts, pts, V).detach()
    o_exact_norm = torch.linalg.norm(o_exact, dim=-1)  # (N,)

    results: dict = {}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for use_orf in ([False, True] if run_orf else [False]):
        tag = "orf" if use_orf else "iid"
        all_errors: list = []
        labels: list = []

        for R in R_values:
            for M in M_values:
                model = LorentzLinearAttention(
                    d=d, R=R, M=M, beta=beta, K=K, use_orf=use_orf,
                ).to(dtype=DTYPE)
                o_hat = model(pts, pts, V).detach()
                per_query = torch.linalg.norm(o_exact - o_hat, dim=-1) / (o_exact_norm + 1e-12)
                key = f"R={R}\nM={M}"
                results[f"{key}_{tag}"] = {
                    "mean": per_query.mean().item(),
                    "median": per_query.median().item(),
                    "max": per_query.max().item(),
                }
                all_errors.append(per_query.numpy())
                labels.append(key)
                print(f"  [{key.replace(chr(10), ', ')}, {tag}] mean={per_query.mean():.4e}  "
                      f"median={per_query.median():.4e}  max={per_query.max():.4e}")

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
        bp = ax.boxplot(all_errors, labels=labels, patch_artist=True)
        color = "#7570b3" if use_orf else "#7fc97f"
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
        ax.set_ylabel("Per-query relative error  ||o − ô|| / ||o||")
        ax.set_title(f"Attention Output Error  (d={d}, K={K}, N={N}, {tag.upper()})")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        fig.savefig(os.path.join(RESULTS_DIR, f"exp3_attention_error_{tag}.png"), dpi=150)
        plt.close(fig)

    return results


if __name__ == "__main__":
    run({})
