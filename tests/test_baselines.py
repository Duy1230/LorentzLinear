import torch
import pytest

from core.baselines.quadratic_lorentz import QuadraticLorentzAttention
from core.baselines.hypformer_linear import HypformerLinearAttention
from core.baselines.performer import PerformerAttention
from core.hyperbolic_ops import sample_hyperboloid, check_on_hyperboloid

DTYPE = torch.float64
K = -1.0
D_SPACE = 8
N = 20


@pytest.fixture
def hyp_data():
    pts = sample_hyperboloid(N, D_SPACE, K=K, dtype=DTYPE)
    return pts, pts, pts.clone()


class TestQuadraticLorentz:
    def test_output_on_hyperboloid(self, hyp_data):
        Q, Ki, V = hyp_data
        m = QuadraticLorentzAttention(beta=1.0, K=K)
        out = m(Q, Ki, V)
        assert check_on_hyperboloid(out, K, atol=1e-3)

    def test_weights_sum_to_one(self, hyp_data):
        Q, Ki, _ = hyp_data
        m = QuadraticLorentzAttention(beta=1.0, K=K)
        w = m.attention_weights(Q, Ki)
        sums = w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_output_shape(self, hyp_data):
        Q, Ki, V = hyp_data
        m = QuadraticLorentzAttention(beta=1.0, K=K)
        out = m(Q, Ki, V)
        assert out.shape == Q.shape

    def test_causal_output_shape(self, hyp_data):
        Q, Ki, V = hyp_data
        m = QuadraticLorentzAttention(beta=1.0, K=K)
        out = m(Q, Ki, V, causal=True)
        assert out.shape == Q.shape


class TestHypformerLinear:
    def test_output_on_hyperboloid(self, hyp_data):
        Q, Ki, V = hyp_data
        m = HypformerLinearAttention(d=D_SPACE, M=64, beta=1.0, K=K).to(dtype=DTYPE)
        out = m(Q, Ki, V)
        assert check_on_hyperboloid(out, K, atol=1e-3)

    def test_output_shape(self, hyp_data):
        Q, Ki, V = hyp_data
        m = HypformerLinearAttention(d=D_SPACE, M=64, K=K).to(dtype=DTYPE)
        out = m(Q, Ki, V)
        assert out.shape == Q.shape


class TestPerformer:
    def test_output_shape(self, hyp_data):
        Q, Ki, V = hyp_data
        m = PerformerAttention(d_full=D_SPACE + 1, M=64).to(dtype=DTYPE)
        out = m(Q, Ki, V)
        assert out.shape == Q.shape

    def test_causal_output_shape(self, hyp_data):
        Q, Ki, V = hyp_data
        m = PerformerAttention(d_full=D_SPACE + 1, M=64).to(dtype=DTYPE)
        out = m(Q, Ki, V, causal=True)
        assert out.shape == Q.shape

    def test_finite_output(self, hyp_data):
        Q, Ki, V = hyp_data
        m = PerformerAttention(d_full=D_SPACE + 1, M=64).to(dtype=DTYPE)
        out = m(Q, Ki, V)
        assert torch.isfinite(out).all()
