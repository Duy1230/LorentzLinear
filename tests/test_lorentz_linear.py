import torch
import pytest

from core.lorentz_linear import LorentzLinearAttention
from core.hyperbolic_ops import sample_hyperboloid, check_on_hyperboloid

DTYPE = torch.float64
K = -1.0
D_SPACE = 8


@pytest.fixture
def model():
    return LorentzLinearAttention(d=D_SPACE, R=2, M=32, beta=1.0, K=K).to(dtype=DTYPE)


@pytest.fixture
def data():
    N = 20
    pts = sample_hyperboloid(N, D_SPACE, K=K, dtype=DTYPE)
    V = pts.clone()  # use same dim for values
    return pts, pts, V


class TestLorentzLinearShape:
    def test_output_shape(self, model, data):
        Q, Ki, V = data
        out = model(Q, Ki, V)
        assert out.shape == Q.shape

    def test_batched_output_shape(self, model):
        B, N = 3, 15
        Q = sample_hyperboloid(B * N, D_SPACE, K=K, dtype=DTYPE).reshape(B, N, -1)
        V = Q.clone()
        out = model(Q, Q, V)
        assert out.shape == (B, N, D_SPACE + 1)


class TestLorentzLinearHyperboloid:
    def test_output_on_hyperboloid(self, model, data):
        Q, Ki, V = data
        out = model(Q, Ki, V)
        assert check_on_hyperboloid(out, K, atol=1e-3)

    def test_causal_output_on_hyperboloid(self, model, data):
        Q, Ki, V = data
        out = model(Q, Ki, V, causal=True)
        assert check_on_hyperboloid(out, K, atol=1e-3)


class TestLorentzLinearCausal:
    def test_causal_matches_full_last_token(self, model, data):
        """For the last token, causal and full should give the same result
        (since all tokens are visible)."""
        Q, Ki, V = data
        out_full = model(Q, Ki, V, causal=False)
        out_causal = model(Q, Ki, V, causal=True)
        assert torch.allclose(out_full[-1], out_causal[-1], atol=1e-5)


class TestLorentzLinearR1:
    def test_r1_no_negative_features(self):
        m = LorentzLinearAttention(d=D_SPACE, R=1, M=32, beta=1.0, K=K).to(dtype=DTYPE)
        x = sample_hyperboloid(10, D_SPACE, K=K, dtype=DTYPE)
        feat_m = m.feat(x, "-")
        # R=1 ⇒ Kreĭn minus channel has 0-dimensional features
        assert feat_m.shape[-1] == 0 or feat_m.abs().max() < 1e-10


class TestLorentzLinearGradient:
    def test_no_nan_gradients(self):
        m = LorentzLinearAttention(d=D_SPACE, R=2, M=32, beta=1.0, K=K).double()
        Q = sample_hyperboloid(10, D_SPACE, K=K, dtype=DTYPE).requires_grad_(True)
        V = Q.detach().clone().requires_grad_(True)
        out = m(Q, Q, V)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(Q.grad).all(), "NaN in Q gradients"
        assert torch.isfinite(V.grad).all(), "NaN in V gradients"
