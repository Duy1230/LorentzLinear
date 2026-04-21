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


class TestOrthogonalFeatures:
    """Tests for the ORF (Orthogonal Random Features) variant."""

    def test_omega_rows_orthogonal(self):
        d = 8
        sfm = SpaceFeatureMap(d_space=d, M=d, beta=1.0, use_orf=True).to(dtype=DTYPE)
        omega_normed = sfm.omega / sfm.omega.norm(dim=1, keepdim=True)
        gram = omega_normed @ omega_normed.T
        eye = torch.eye(d, dtype=DTYPE)
        assert torch.allclose(gram, eye, atol=1e-6), "ORF rows are not orthogonal"

    def test_output_shape_matches_iid(self):
        sfm = SpaceFeatureMap(d_space=8, M=32, beta=1.0, use_orf=True).to(dtype=DTYPE)
        xs = torch.randn(10, 8, dtype=DTYPE)
        phi = sfm(xs)
        assert phi.shape == (10, 32)

    def test_output_nonnegative(self):
        sfm = SpaceFeatureMap(d_space=8, M=64, beta=1.0, use_orf=True).to(dtype=DTYPE)
        xs = torch.randn(50, 8, dtype=DTYPE)
        phi = sfm(xs)
        assert (phi >= 0).all()

    def test_m_greater_than_d(self):
        """When M > d_space, multiple orthogonal blocks are stacked."""
        sfm = SpaceFeatureMap(d_space=4, M=20, beta=1.0, use_orf=True).to(dtype=DTYPE)
        assert sfm.omega.shape == (20, 4)
        xs = torch.randn(10, 4, dtype=DTYPE)
        phi = sfm(xs)
        assert phi.shape == (10, 20)
        assert (phi >= 0).all()

    def test_monte_carlo_kernel_convergence(self):
        """ORF kernel estimate should converge (averaged over seeds)."""
        d, beta = 8, 1.0
        N_pairs = 20
        n_draws = 300

        qs = torch.randn(N_pairs, d, dtype=DTYPE) * 0.3
        ks = torch.randn(N_pairs, d, dtype=DTYPE) * 0.3
        exact = torch.exp(beta * (qs * ks).sum(dim=-1))

        estimates = []
        for seed in range(n_draws):
            sfm = SpaceFeatureMap(d_space=d, M=256, beta=beta, seed=seed, use_orf=True).to(dtype=DTYPE)
            phi_q = sfm(qs)
            h_q = sfm.log_scale
            phi_k = sfm(ks)
            h_k = sfm.log_scale
            raw = (phi_q * phi_k).sum(dim=-1)
            est = raw * torch.exp(h_q + h_k)
            estimates.append(est)

        avg = torch.stack(estimates).median(dim=0).values
        rel_err = ((avg - exact) / (exact + 1e-12)).abs().mean().item()
        assert rel_err < 0.25, f"ORF relative error {rel_err:.4f} too large"

    def test_lower_variance_than_iid(self):
        """Single-draw ORF should produce lower attention output variance than iid.

        Measured via normalised attention output (numerator/denominator) which is
        the quantity that actually matters for downstream use.  The ORF advantage
        is clearest when M <= d (single orthogonal block).
        """
        d, beta, M = 16, 1.0, 16
        N = 30
        n_draws = 80

        torch.manual_seed(123)
        from core.hyperbolic_ops import sample_hyperboloid
        pts = sample_hyperboloid(N, d, K=-1.0, scale=0.3, dtype=DTYPE)
        V = pts.clone()

        def _output_variance(use_orf):
            from core.lorentz_linear import LorentzLinearAttention
            outs = []
            for seed in range(n_draws):
                model = LorentzLinearAttention(d=d, R=2, M=M, beta=beta, K=-1.0, seed=seed, use_orf=use_orf).to(dtype=DTYPE)
                o = model(pts, pts, V).detach()
                outs.append(o)
            return torch.stack(outs).var(dim=0).mean().item()

        var_iid = _output_variance(False)
        var_orf = _output_variance(True)
        assert var_orf < var_iid, f"ORF variance ({var_orf:.4e}) not lower than iid ({var_iid:.4e})"


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
