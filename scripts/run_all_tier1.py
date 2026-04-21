"""Run all Tier 1 kernel-validation experiments.

Usage:
    python scripts/run_all_tier1.py [--config configs/tier1.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.kernel_validation import (
    exp1_kernel_approx,
    exp2_normalizer_health,
    exp3_attention_error,
    exp4_equivariance,
    exp4b_prop4_validation,
    exp5_degenerate_analysis,
)


def main():
    parser = argparse.ArgumentParser(description="Run Tier 1 experiments")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "tier1.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    shared = cfg.get("shared", {})

    experiments = [
        ("Exp 1.1: Kernel Approximation Quality", exp1_kernel_approx, "exp1_kernel_approx"),
        ("Exp 1.2: Normaliser Health",            exp2_normalizer_health, "exp2_normalizer_health"),
        ("Exp 1.3: Attention Output Error",       exp3_attention_error, "exp3_attention_error"),
        ("Exp 1.4: Equivariance Under Boosts",    exp4_equivariance, "exp4_equivariance"),
        ("Exp 4b: Proposition 4 Validation",      exp4b_prop4_validation, "exp4b_prop4_validation"),
        ("Exp 5:  Degenerate Output Analysis",    exp5_degenerate_analysis, "exp5_degenerate_analysis"),
    ]

    for title, module, key in experiments:
        exp_cfg = {**shared, **cfg.get(key, {})}
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        t0 = time.time()
        module.run(exp_cfg)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"  All Tier 1 experiments complete.")
    print(f"  Figures saved to: {os.path.join(ROOT, 'results', 'tier1')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
