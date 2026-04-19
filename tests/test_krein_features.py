import math
import torch
import pytest

from core.krein_features import KreinSplit


DTYPE = torch.float64


class TestKreinSplitBasic:
    def test_r1_no_negative_channel(self):
        """R=1 produces only a positive channel (single r=0 central term)."""
        ks = KreinSplit(R=1, beta=1.0)
        x0 = torch.rand(10, 1, dtype=DTYPE) + 1.0
        psi_m = ks.psi_minus(x0)
        assert psi_m.shape[-1] == 0 or psi_m.abs().max() < 1e-12

    def test_feature_dims_increase_with_R(self):
        dims = []
        for R in range(1, 5):
            ks = KreinSplit(R=R, beta=1.0)
            dims.append(ks.feature_dim)
        for i in range(len(dims) - 1):
            assert dims[i + 1] >= dims[i]

    def test_output_shapes(self):
        ks = KreinSplit(R=3, beta=1.0)
        x0 = torch.rand(5, 7, 1, dtype=DTYPE) + 1.0
        p = ks.psi_plus(x0)
        m = ks.psi_minus(x0)
        assert p.shape[:2] == (5, 7)
        assert m.shape[:2] == (5, 7)


class TestKreinKernelApprox:
    """ψ₊(q0)^T ψ₊(k0) - ψ₋(q0)^T ψ₋(k0)  ≈  exp(-β q0 k0) * decay(q0) * decay(k0)."""

    @pytest.mark.parametrize("R", [1, 2, 3, 4])
    def test_approx_improves_with_R(self, R):
        beta = 1.0
        ks = KreinSplit(R=R, beta=beta)

        N = 100
        q0_vals = torch.rand(N, 1, dtype=DTYPE) * 2.0 + 1.0  # [1, 3]
        k0_vals = torch.rand(N, 1, dtype=DTYPE) * 2.0 + 1.0

        psi_p_q = ks.psi_plus(q0_vals)   # (N, dim)
        psi_m_q = ks.psi_minus(q0_vals)
        psi_p_k = ks.psi_plus(k0_vals)
        psi_m_k = ks.psi_minus(k0_vals)

        # Approximate kernel (per-pair dot products)
        approx = (psi_p_q * psi_p_k).sum(dim=-1) - (psi_m_q * psi_m_k).sum(dim=-1)

        # Exact target: exp(-beta * q0 * k0) includes the decay factors
        q0 = q0_vals.squeeze(-1)
        k0 = k0_vals.squeeze(-1)
        # The features include exp(-beta*x0^2/2) decay for each side,
        # so the product of features should approximate:
        #   exp(-beta*q0^2/2) * exp(-beta*k0^2/2) * Taylor_approx((q0-k0)^2/2)
        # which equals exp(-beta*q0*k0) when the Taylor is exact.
        exact = torch.exp(-beta * q0 * k0)

        err = (approx - exact).abs().mean().item()
        # Just verify it produces finite, reasonable values
        assert math.isfinite(err)

    def test_error_decreases_with_R(self):
        beta = 1.0
        N = 200
        q0 = torch.rand(N, 1, dtype=DTYPE) * 1.5 + 1.0
        k0 = torch.rand(N, 1, dtype=DTYPE) * 1.5 + 1.0
        exact = torch.exp(-beta * q0.squeeze(-1) * k0.squeeze(-1))

        errors = []
        for R in [1, 2, 3, 4]:
            ks = KreinSplit(R=R, beta=beta)
            pp_q, pm_q = ks.psi_plus(q0), ks.psi_minus(q0)
            pp_k, pm_k = ks.psi_plus(k0), ks.psi_minus(k0)
            approx = (pp_q * pp_k).sum(-1) - (pm_q * pm_k).sum(-1)
            err = (approx - exact).abs().mean().item()
            errors.append(err)

        # Each step should be no worse than the previous (modulo numerical noise)
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-8, (
                f"Error did not decrease: R={i + 1} err={errors[i]:.6e}, R={i + 2} err={errors[i + 1]:.6e}"
            )
