import math
import torch
import pytest

from core.hyperbolic_ops import (
    minkowski_inner,
    minkowski_norm_sq,
    split_time_space,
    check_on_hyperboloid,
    project_to_hyperboloid,
    sample_hyperboloid,
    lorentz_boost,
    exp_map,
    log_map,
)


DTYPE = torch.float64
K = -1.0


@pytest.fixture
def points():
    """100 points on L^4_{-1}."""
    return sample_hyperboloid(100, 4, K=K, dtype=DTYPE)


# ---------------------------------------------------------------
# Minkowski inner product
# ---------------------------------------------------------------

class TestMinkowskiInner:
    def test_symmetric(self, points):
        x, y = points[:50], points[50:]
        assert torch.allclose(minkowski_inner(x, y), minkowski_inner(y, x), atol=1e-10)

    def test_bilinear(self, points):
        x, y, z = points[:30], points[30:60], points[60:90]
        a = 2.3
        lhs = minkowski_inner(a * x + y, z)
        rhs = a * minkowski_inner(x, z) + minkowski_inner(y, z)
        assert torch.allclose(lhs, rhs, atol=1e-8)

    def test_batch_shape(self):
        x = torch.randn(3, 5, 4, dtype=DTYPE)
        y = torch.randn(3, 5, 4, dtype=DTYPE)
        assert minkowski_inner(x, y).shape == (3, 5)


# ---------------------------------------------------------------
# Norm and splitting
# ---------------------------------------------------------------

class TestNormAndSplit:
    def test_norm_via_inner(self, points):
        assert torch.allclose(minkowski_norm_sq(points), minkowski_inner(points, points), atol=1e-10)

    def test_split_shapes(self, points):
        x0, xs = split_time_space(points)
        assert x0.shape == (100, 1)
        assert xs.shape == (100, 4)

    def test_split_reconstruct(self, points):
        x0, xs = split_time_space(points)
        recon = torch.cat([x0, xs], dim=-1)
        assert torch.allclose(recon, points, atol=1e-12)


# ---------------------------------------------------------------
# Hyperboloid membership
# ---------------------------------------------------------------

class TestHyperboloid:
    def test_sample_on_hyperboloid(self, points):
        assert check_on_hyperboloid(points, K, atol=1e-6)

    def test_sample_positive_time(self, points):
        assert (points[..., 0] > 0).all()

    def test_project_puts_on_hyperboloid(self):
        # Build vectors with time-like Minkowski signature (<x,x>_L < 0)
        # by sampling space components and computing a compatible time component
        # with some perturbation, so the projection has real work to do.
        xs = torch.randn(20, 4, dtype=DTYPE)
        xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(xs_norm_sq + 1.0) * 1.3  # larger than needed → not on hyperboloid
        x = torch.cat([x0, xs], dim=-1)
        proj = project_to_hyperboloid(x, K)
        assert check_on_hyperboloid(proj, K, atol=1e-4)

    def test_project_preserves_direction(self):
        """Uniform rescaling should preserve the Minkowski direction of x."""
        xs = torch.randn(10, 4, dtype=DTYPE) * 0.5
        xs_norm_sq = (xs * xs).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(xs_norm_sq + 2.0)  # off-manifold time-like vector
        x = torch.cat([x0, xs], dim=-1)
        proj = project_to_hyperboloid(x, K)
        # The ratio proj/x should be constant across all (d+1) components
        ratio = proj / x
        ratio_std = ratio.std(dim=-1)
        assert (ratio_std < 1e-6).all(), "Projection did not uniformly rescale"

    def test_sample_different_curvatures(self):
        for k in [-1.0, -0.5, -0.1]:
            pts = sample_hyperboloid(50, 8, K=k, dtype=DTYPE)
            assert check_on_hyperboloid(pts, k, atol=1e-6)


# ---------------------------------------------------------------
# Lorentz boosts
# ---------------------------------------------------------------

class TestLorentzBoost:
    def test_preserves_inner_product(self, points):
        x, y = points[:50], points[50:]
        v_dir = torch.zeros(4, dtype=DTYPE)
        v_dir[0] = 1.0  # boost in first spatial direction
        for phi in [0.3, 0.7, 1.5]:
            bx = lorentz_boost(x, v_dir, phi)
            by = lorentz_boost(y, v_dir, phi)
            orig = minkowski_inner(x, y)
            boosted = minkowski_inner(bx, by)
            assert torch.allclose(orig, boosted, atol=1e-7), f"Failed at rapidity {phi}"

    def test_keeps_on_hyperboloid(self, points):
        v_dir = torch.zeros(4, dtype=DTYPE)
        v_dir[1] = 1.0
        boosted = lorentz_boost(points, v_dir, 0.5)
        assert check_on_hyperboloid(boosted, K, atol=1e-5)

    def test_zero_boost_identity(self, points):
        v_dir = torch.zeros(4, dtype=DTYPE)
        v_dir[0] = 1.0
        boosted = lorentz_boost(points, v_dir, 0.0)
        assert torch.allclose(boosted, points, atol=1e-10)


# ---------------------------------------------------------------
# Exp / log maps (round-trip)
# ---------------------------------------------------------------

class TestExpLogMaps:
    def test_exp_stays_on_hyperboloid(self):
        x = sample_hyperboloid(10, 4, K=K, dtype=DTYPE)
        v = torch.randn(10, 5, dtype=DTYPE) * 0.1
        # project v into tangent space at x:  v -= <x,v>_L / <x,x>_L * x
        inner_xv = minkowski_inner(x, v)
        inner_xx = minkowski_norm_sq(x)
        v = v - (inner_xv / inner_xx).unsqueeze(-1) * x
        y = exp_map(x, v, K)
        assert check_on_hyperboloid(y, K, atol=1e-3)
