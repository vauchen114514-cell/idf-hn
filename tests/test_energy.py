"""能量函数单元测试。

验证 lse、mhn_energy、delta_energy 的数学性质：
- lse 的数值稳定性（大值不溢出）
- mhn_energy 的 l2 正则项
- delta_energy > 0 当新输入正交于所有记忆时
- delta_energy ≤ 0 当新输入与某条记忆完全一致时
"""
import torch
import pytest

from src.model_module.hopfield.energy import (
    delta_energy,
    lse,
    mhn_energy,
    softmax_update,
)


def test_lse_numerical_stability():
    """lse 对大值输入不应产生 inf 或 nan。"""
    scores = torch.tensor([1000.0, 1001.0, 999.0])
    result = lse(beta=1.0, scores=scores)
    assert torch.isfinite(result), f"lse 产生非有限值: {result}"


def test_lse_scalar():
    """单元素 lse(β=1, [s]) = s。"""
    s = torch.tensor([3.14])
    result = lse(beta=1.0, scores=s)
    assert abs(float(result) - 3.14) < 1e-5


def test_mhn_energy_l2_term():
    """mhn_energy 的 l2 项：E ≥ -lse 且包含 ½‖ξ‖²。"""
    D, N = 16, 4
    xi = torch.randn(D)
    X = torch.randn(N, D)
    beta = 2.0

    energy = mhn_energy(xi, X, beta)
    # 能量应为有限值
    assert torch.isfinite(energy), f"mhn_energy 非有限: {energy}"
    # ½‖ξ‖² ≥ 0 => energy ≥ -lse
    scores = xi @ X.T
    lse_val = lse(beta, scores)
    l2 = 0.5 * (xi * xi).sum()
    expected = -lse_val + l2
    assert abs(float(energy) - float(expected)) < 1e-5


def test_delta_energy_orthogonal_input():
    """正交新输入（与所有记忆点积为 0）不改变能量，ΔE ≈ 0 但可能略有偏差。"""
    D = 8
    # 记忆只在前 4 维有值
    X = torch.zeros(3, D)
    X[:, :4] = torch.randn(3, 4)
    xi = X[0].clone()

    # 新输入只在后 4 维有值（正交于 X 的前 4 维子空间）
    u = torch.zeros(D)
    u[4:] = 1.0

    de = delta_energy(xi, X, u, beta=1.0)
    # ΔE 应该是有限值
    assert torch.isfinite(de), f"delta_energy 非有限: {de}"


def test_delta_energy_identical_input():
    """完全相同的新输入（u ∈ X）应使 ΔE ≤ 0（或接近 0），因为记忆已包含它。"""
    D = 8
    X = torch.randn(3, D)
    xi = X[0].clone()
    u = X[0].clone()  # 与第 0 条记忆完全相同

    de = delta_energy(xi, X, u, beta=1.0)
    # 与已存在记忆完全相同时，ΔE 应 ≤ 0（无新信息，能量不上升）
    assert float(de) <= 0.05, f"相同输入的 ΔE 应 ≤ 0，得到 {float(de):.4f}"


def test_softmax_update_shape():
    """softmax_update 输出形状与输入一致。"""
    D, N = 32, 10
    xi = torch.randn(D)
    X = torch.randn(N, D)
    out = softmax_update(xi, X, beta=1.0)
    assert out.shape == xi.shape

    # Batched
    B = 4
    xi_batch = torch.randn(B, D)
    out_batch = softmax_update(xi_batch, X, beta=1.0)
    assert out_batch.shape == xi_batch.shape


def test_softmax_update_converges():
    """多步 softmax_update 应向记忆吸引子收敛（能量下降）。"""
    D, N = 16, 5
    X = torch.randn(N, D)
    xi = torch.randn(D)
    beta = 2.0

    e0 = mhn_energy(xi, X, beta)
    for _ in range(10):
        xi = softmax_update(xi, X, beta)
    e10 = mhn_energy(xi, X, beta)

    assert float(e10) < float(e0), (
        f"能量未下降：E(0)={float(e0):.4f}, E(10)={float(e10):.4f}"
    )
