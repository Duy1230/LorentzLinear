"""Experiment 5 -- Near-Degenerate Output Investigation.

Analyses the "1 in 20" near-degenerate outputs from LorentzLinear:
  (a) Distribution of |<y,y>_L| before projection
  (b) Correlation with attention entropy and token depth (x0)
  (c) Quadratic baseline comparison
  (d) R=1 (PD-only) elimination test
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from core.hyperbolic_ops import sample_hyperboloid, minkowski_norm_sq
from core.lorentz_linear import LorentzLinearAttention
from core.baselines.quadratic_lorentz import QuadraticLorentzAttention

DTYPE = torch.float64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tier1")


def run(cfg: dict) -> dict:
    N = cfg.get("N", 1000)
    d = cfg.get("d", 16)
    K = cfg.get("K", -1.0)
    R = cfg.get("R", 2)
    M = cfg.get("M", 64)
    beta = cfg.get("beta", 1.0)
    seed = cfg.get("seed", 0)

    gen = torch.Generator().manual_seed(seed)
    pts = sample_hyperboloid(N, d, K=K, scale=0.5, dtype=DTYPE, generator=gen)
    V = pts.clone()

    # --- (a) Distribution of |<y,y>_L| ---
    ll = LorentzLinearAttention(d=d, R=R, M=M, beta=beta, K=K).to(dtype=DTYPE)
    y_ll = ll._forward_full_raw(pts, pts, V).detach()
    mink_sq_ll = minkowski_norm_sq(y_ll)  # (N,)
    abs_mink_ll = mink_sq_ll.abs()

    degen_threshold = 1e-4
    n_degen = (abs_mink_ll < degen_threshold).sum().item()
    pct_degen = 100.0 * n_degen / N

    print(f"  [a] Distribution of |<y,y>_L| for LorentzLinear (R={R}, M={M}):")
    print(f"      min={abs_mink_ll.min():.4e}  median={abs_mink_ll.median():.4e}  "
          f"max={abs_mink_ll.max():.4e}")
    print(f"      Degenerate (< {degen_threshold}): {n_degen}/{N} ({pct_degen:.1f}%)")
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(abs_mink_ll.numpy(), pctiles)
    for p, v in zip(pctiles, vals):
        print(f"      p{p:02d} = {v:.4e}")

    # --- (b) Correlation with attention entropy and depth ---
    quad = QuadraticLorentzAttention(beta=beta, K=K)
    weights = quad.attention_weights(pts, pts).detach()  # (N, N)
    log_w = torch.log(weights + 1e-30)
    entropy = -(weights * log_w).sum(dim=-1)  # (N,)
    x0 = pts[:, 0]  # (N,)

    corr_entropy = np.corrcoef(abs_mink_ll.numpy(), entropy.numpy())[0, 1]
    corr_x0 = np.corrcoef(abs_mink_ll.numpy(), x0.numpy())[0, 1]
    print(f"\n  [b] Correlations:")
    print(f"      |<y,y>_L| vs attention entropy: r = {corr_entropy:.4f}")
    print(f"      |<y,y>_L| vs x0 (depth):        r = {corr_x0:.4f}")

    # --- (c) Quadratic baseline comparison ---
    y_quad = quad.forward_raw(pts, pts, V).detach()
    mink_sq_quad = minkowski_norm_sq(y_quad)
    abs_mink_quad = mink_sq_quad.abs()

    n_degen_quad = (abs_mink_quad < degen_threshold).sum().item()
    pct_degen_quad = 100.0 * n_degen_quad / N

    print(f"\n  [c] Quadratic baseline |<y,y>_L|:")
    print(f"      min={abs_mink_quad.min():.4e}  median={abs_mink_quad.median():.4e}  "
          f"max={abs_mink_quad.max():.4e}")
    print(f"      Degenerate (< {degen_threshold}): {n_degen_quad}/{N} ({pct_degen_quad:.1f}%)")

    # --- (d) R=1 (PD-only) elimination test ---
    ll_r1 = LorentzLinearAttention(d=d, R=1, M=M, beta=beta, K=K).to(dtype=DTYPE)
    y_r1 = ll_r1._forward_full_raw(pts, pts, V).detach()
    mink_sq_r1 = minkowski_norm_sq(y_r1)
    abs_mink_r1 = mink_sq_r1.abs()

    n_degen_r1 = (abs_mink_r1 < degen_threshold).sum().item()
    pct_degen_r1 = 100.0 * n_degen_r1 / N

    print(f"\n  [d] R=1 (PD-only) |<y,y>_L|:")
    print(f"      min={abs_mink_r1.min():.4e}  median={abs_mink_r1.median():.4e}  "
          f"max={abs_mink_r1.max():.4e}")
    print(f"      Degenerate (< {degen_threshold}): {n_degen_r1}/{N} ({pct_degen_r1:.1f}%)")

    if n_degen_r1 == 0 and n_degen > 0:
        print("      -> Degenerate outputs ELIMINATED by R=1: confirms Krein cancellation.")
    elif n_degen_r1 > 0:
        print("      -> Degenerate outputs persist at R=1: not purely Krein cancellation.")

    # --- Plots ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Histogram
    ax = axes[0, 0]
    log_vals = np.log10(abs_mink_ll.numpy() + 1e-30)
    ax.hist(log_vals, bins=50, color="#d95f02", alpha=0.8, edgecolor="white")
    ax.axvline(np.log10(degen_threshold), color="red", linestyle="--",
               label=f"Degen threshold ({degen_threshold})")
    ax.set_xlabel("log10( |<y,y>_L| )")
    ax.set_ylabel("Count")
    ax.set_title(f"(a) Minkowski norm distribution  (R={R}, {pct_degen:.1f}% degenerate)")
    ax.legend(fontsize=8)

    # (b) Scatter: entropy & depth
    ax = axes[0, 1]
    sc = ax.scatter(entropy.numpy(), abs_mink_ll.numpy(),
                    c=x0.numpy(), cmap="viridis", s=8, alpha=0.7)
    ax.set_xlabel("Attention entropy")
    ax.set_ylabel("|<y,y>_L|")
    ax.set_title(f"(b) vs entropy (r={corr_entropy:.3f}), colored by x0")
    ax.set_yscale("log")
    plt.colorbar(sc, ax=ax, label="x0 (depth)")

    # (c) Comparison: LorentzLinear vs Quadratic
    ax = axes[1, 0]
    ax.hist(np.log10(abs_mink_ll.numpy() + 1e-30), bins=50, alpha=0.7,
            label=f"LorentzLinear R={R} ({pct_degen:.1f}% degen)", color="#d95f02")
    ax.hist(np.log10(abs_mink_quad.numpy() + 1e-30), bins=50, alpha=0.7,
            label=f"Quadratic ({pct_degen_quad:.1f}% degen)", color="#1b9e77")
    ax.set_xlabel("log10( |<y,y>_L| )")
    ax.set_ylabel("Count")
    ax.set_title("(c) LorentzLinear vs Quadratic")
    ax.legend(fontsize=8)

    # (d) R=1 vs R=2
    ax = axes[1, 1]
    ax.hist(np.log10(abs_mink_ll.numpy() + 1e-30), bins=50, alpha=0.7,
            label=f"R={R} ({pct_degen:.1f}% degen)", color="#d95f02")
    ax.hist(np.log10(abs_mink_r1.numpy() + 1e-30), bins=50, alpha=0.7,
            label=f"R=1 PD-only ({pct_degen_r1:.1f}% degen)", color="#7570b3")
    ax.set_xlabel("log10( |<y,y>_L| )")
    ax.set_ylabel("Count")
    ax.set_title("(d) R=1 (PD-only) vs R=2")
    ax.legend(fontsize=8)

    fig.suptitle(f"Near-Degenerate Output Analysis  (N={N}, d={d}, K={K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_degenerate_analysis.png"), dpi=150)
    plt.close(fig)

    return {
        "pct_degenerate_ll": pct_degen,
        "pct_degenerate_quad": pct_degen_quad,
        "pct_degenerate_r1": pct_degen_r1,
        "corr_entropy": corr_entropy,
        "corr_x0": corr_x0,
        "percentiles_ll": dict(zip(pctiles, vals.tolist())),
    }


if __name__ == "__main__":
    run({})
