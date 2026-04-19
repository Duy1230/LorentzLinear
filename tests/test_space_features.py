import math
import torch
import pytest

from core.space_features import SpaceFeatureMap


DTYPE = torch.float64


class TestSpaceFeatureMapBasic:
    def test_output_shape(self):
        sfm = SpaceFeatureMap(d_space=8, M=32, beta=1.0)
        xs = torch.randn(10, 8, dtype=DTYPE)
        phi = sfm.to(dtype=DTYPE)(xs)
        assert phi.shape == (10, 32)

    def test_output_nonnegative(self):
        sfm = SpaceFeatureMap(d_space=8, M=64, beta=1.0).to(dtype=DTYPE)
        xs = torch.randn(50, 8, dtype=DTYPE)
        phi = sfm(xs)
        assert (phi >= 0).all()

    def test_batch_dims(self):
        sfm = SpaceFeatureMap(d_space=4, M=16, beta=1.0).to(dtype=DTYPE)
        xs = torch.randn(3, 5, 4, dtype=DTYPE)
        phi = sfm(xs)
        assert phi.shape == (3, 5, 16)


class TestSpaceKernelApprox:
    def test_monte_carlo_kernel_estimate(self):
        """Average over many omega draws: E[phi(q)^T phi(k)] ~ exp(beta * q^T k).

        Each forward() call uses its own shared-max for overflow safety.
        To recover the unbiased kernel, we compensate with exp(h_q + h_k).
        """
        d, beta = 4, 1.0
        N_pairs = 20
        n_draws = 200

        qs = torch.randn(N_pairs, d, dtype=DTYPE) * 0.5
        ks = torch.randn(N_pairs, d, dtype=DTYPE) * 0.5
        exact = torch.exp(beta * (qs * ks).sum(dim=-1))  # (N_pairs,)

        estimates = []
        for seed in range(n_draws):
            sfm = SpaceFeatureMap(d_space=d, M=256, beta=beta, seed=seed).to(dtype=DTYPE)
            phi_q = sfm(qs)
            h_q = sfm.log_scale
            phi_k = sfm(ks)
            h_k = sfm.log_scale
            raw = (phi_q * phi_k).sum(dim=-1)
            est = raw * torch.exp(h_q + h_k)
            estimates.append(est)

        avg = torch.stack(estimates).mean(dim=0)
        rel_err = ((avg - exact) / (exact + 1e-12)).abs().mean().item()
        assert rel_err < 0.15, f"Relative error {rel_err:.4f} too large"


class TestOverflowGuard:
    def test_large_norm_no_overflow(self):
        """phi_E should remain finite even for ||xs|| = 20."""
        sfm = SpaceFeatureMap(d_space=4, M=64, beta=1.0).to(dtype=DTYPE)
        xs = torch.randn(10, 4, dtype=DTYPE) * 20.0
        phi = sfm(xs)
        assert torch.isfinite(phi).all(), "Overflow detected in space features"

    def test_zero_input(self):
        sfm = SpaceFeatureMap(d_space=4, M=64, beta=1.0).to(dtype=DTYPE)
        xs = torch.zeros(5, 4, dtype=DTYPE)
        phi = sfm(xs)
        assert torch.isfinite(phi).all()
        assert (phi >= 0).all()
