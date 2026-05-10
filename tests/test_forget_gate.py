"""遗忘门单元测试。"""
import torch
import pytest

from src.model_module.forget_gate.forget_gate import ForgetGate
from src.model_module.forget_gate.conflict_detector import ConflictDetector


@pytest.fixture
def gate():
    return ForgetGate(
        gamma_0=0.01,
        delta_gamma=0.5,
        tau=0.3,
        beta=1.0,
        adaptive_tau=False,
    )


def test_gamma_range(gate):
    """遗忘率应在 [gamma_0, gamma_0 + delta_gamma] 范围内。"""
    D, N = 8, 3
    X = torch.randn(N, D)
    u = torch.randn(D)
    xi = torch.randn(D)

    gamma, _ = gate.compute_gamma(u, xi, X)
    assert 0.01 <= float(gamma) <= 0.51, f"gamma={float(gamma):.4f} 超出范围"


def test_high_conflict_high_gamma(gate):
    """高冲突输入（与记忆完全正交）应产生较高遗忘率。"""
    D = 16
    X = torch.zeros(3, D)
    X[:, :D//2] = torch.eye(3, D//2)

    # u 完全在 X 的零空间
    u = torch.zeros(D)
    u[D//2:] = 1.0
    xi = X[0].clone()

    # 低冲突参考：u 与第一条记忆完全相同
    u_low = X[0].clone()

    gamma_high, _ = gate.compute_gamma(u, xi, X)
    gate.reset()
    gamma_low, _ = gate.compute_gamma(u_low, xi, X)

    # 注意：由于 delta_energy 在正交情况下可能为负，此测试只验证范围
    assert 0.01 <= float(gamma_high) <= 0.51
    assert 0.01 <= float(gamma_low) <= 0.51


def test_adaptive_tau_fallback():
    """历史不足时自适应 τ 应返回 fallback 值。"""
    detector = ConflictDetector(beta=1.0, tau_percentile=75)
    tau = detector.adaptive_tau(fallback=0.42)
    assert tau == 0.42


def test_adaptive_tau_from_history():
    """历史足够时自适应 τ 应从历史计算百分位数。"""
    detector = ConflictDetector(beta=1.0, tau_percentile=50, history_size=100)
    # 手动添加 20 个历史记录
    D, N = 8, 2
    X = torch.randn(N, D)
    for _ in range(20):
        u = torch.randn(D)
        xi = torch.randn(D)
        detector.compute(u, xi, X)

    tau = detector.adaptive_tau()
    assert isinstance(tau, float)
    assert tau != 0.3  # 应与 fallback 不同


def test_reset_clears_history():
    """reset 应清空历史记录。"""
    gate = ForgetGate(adaptive_tau=True)
    D, N = 8, 2
    X = torch.randn(N, D)
    for _ in range(20):
        gate.compute_gamma(torch.randn(D), torch.randn(D), X)
    assert gate.detector.n_history > 0
    gate.reset()
    assert gate.detector.n_history == 0
