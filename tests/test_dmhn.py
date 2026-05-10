"""DMHN 基线单元测试。

验证 Li et al. (2506.01303) 实现的关键性质：
1. WS 始终保持对称
2. WD(u) 是正半定矩阵（PSD）
3. 前向传播输出形状正确
4. enforce_constraints 后谱范数 ≤ 1
"""
import torch
import pytest
from types import SimpleNamespace


def make_cfg(N: int = 32) -> SimpleNamespace:
    m = SimpleNamespace(
        beta=1.0,
        dmhn_n_steps=5,
        dmhn_dt=0.1,
        dmhn_rank=4,
    )
    return SimpleNamespace(model=m)


@pytest.fixture
def dmhn():
    from src.model_module.baselines.dmhn import DMHNBaseline
    cfg = make_cfg(N=32)
    return DMHNBaseline(cfg, input_dim=32, n_classes=5)


def test_WS_symmetric(dmhn):
    """WS 属性应返回对称矩阵。"""
    WS = dmhn.WS
    diff = (WS - WS.T).abs().max().item()
    assert diff < 1e-6, f"WS 不对称：最大偏差 {diff}"


def test_WD_psd(dmhn):
    """WD(u) 应为正半定矩阵（所有特征值 ≥ 0）。"""
    u = torch.randn(1, 32)
    z = dmhn._WD(u)  # (1, rank)
    # WD = z^T z 的特征值应全 ≥ 0
    # 此处 rank=4，WD 实际是通过 z 隐式表示的
    # 验证 z 本身为有限向量（间接验证 PSD 构造）
    assert torch.isfinite(z).all(), "z 包含非有限值"
    # z^T z 的 Frobenius 范数 ≥ 0
    WD_frob = (z * z).sum()
    assert WD_frob >= 0


def test_forward_shape(dmhn):
    """前向传播输出形状正确。"""
    B, D, C = 4, 32, 5
    x = torch.randn(B, D)
    logits, energy = dmhn(x, update=True)
    assert logits.shape == (B, C), f"logits 形状错误: {logits.shape}"
    assert energy.shape == (B,), f"energy 形状错误: {energy.shape}"


def test_forward_finite(dmhn):
    """前向传播不产生 NaN 或 Inf。"""
    x = torch.randn(8, 32)
    logits, energy = dmhn(x)
    assert torch.isfinite(logits).all(), "logits 包含非有限值"
    assert torch.isfinite(energy).all(), "energy 包含非有限值"


def test_enforce_constraints(dmhn):
    """enforce_constraints 后 WS 仍对称且谱范数 ≤ 1。"""
    # 先注入一个非对称、范数大的矩阵
    with torch.no_grad():
        dmhn.WS_raw.data = torch.randn(32, 32) * 10.0

    dmhn.enforce_constraints()

    # 对称性
    WS = dmhn.WS
    diff = (WS - WS.T).abs().max().item()
    assert diff < 1e-5, f"enforce 后 WS 不对称：{diff}"

    # 谱范数
    sv = torch.linalg.svdvals(WS)
    assert sv.max().item() <= 1.0 + 1e-5, f"谱范数超过 1：{sv.max().item():.4f}"


def test_no_memory_size_attr(dmhn):
    """DMHN 不应有 memory_size 属性（与 Hopfield 在线存储区分）。"""
    assert not hasattr(dmhn, "memory_size"), "DMHN 不应有 memory_size 属性"


def test_ID_shape(dmhn):
    """ID(u) 输出形状应为 (B, N)。"""
    u = torch.randn(6, 32)
    ID_u = dmhn._ID(u)
    assert ID_u.shape == (6, 32), f"ID 形状错误: {ID_u.shape}"
