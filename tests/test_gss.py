"""GSSBaseline 单元测试。"""
import torch
import pytest
from types import SimpleNamespace

from src.model_module.baselines.gss import GSSBaseline


# -----------------------------------------------------------------------
# Fixture
# -----------------------------------------------------------------------

D = 16
N_CLASSES = 4
BUF_SIZE = 10
CAND_SIZE = 3


def make_cfg(memory_size=BUF_SIZE, candidate_size=CAND_SIZE):
    return SimpleNamespace(model=SimpleNamespace(
        memory_size=memory_size,
        candidate_size=candidate_size,
    ))


@pytest.fixture
def gss():
    return GSSBaseline(make_cfg(), input_dim=D, n_classes=N_CLASSES)


# -----------------------------------------------------------------------
# 基础接口测试
# -----------------------------------------------------------------------

def test_forward_shape(gss):
    """forward 返回 (logits, energy)，形状正确。"""
    x = torch.randn(8, D)
    y = torch.zeros(8, dtype=torch.long)
    logits, energy = gss(x, update=False)
    assert logits.shape == (8, N_CLASSES)
    assert energy.shape == (8,)


def test_forward_update_false_no_buffer_change(gss):
    """update=False 时不写入缓冲区。"""
    x = torch.randn(4, D)
    gss(x, update=False, labels=torch.zeros(4, dtype=torch.long))
    assert gss._n_valid == 0


def test_fill_phase(gss):
    """前 buffer_size 条样本直接写入，不做梯度计算。"""
    for i in range(BUF_SIZE):
        x = torch.randn(1, D)
        y = torch.tensor([i % N_CLASSES])
        gss(x, update=True, labels=y)
    assert gss._n_valid == BUF_SIZE


def test_buffer_does_not_exceed_size(gss):
    """超出 buffer_size 后有效条目数不增加。"""
    labels = torch.zeros(BUF_SIZE + 5, dtype=torch.long)
    for i in range(BUF_SIZE + 5):
        x = torch.randn(1, D)
        gss(x, update=True, labels=labels[i:i+1])
    assert gss._n_valid == BUF_SIZE


def test_sample_replay_none_when_empty(gss):
    """缓冲区为空时 sample_replay 返回 None。"""
    assert gss.sample_replay(5) is None


def test_sample_replay_shape(gss):
    """sample_replay 返回正确形状的 (feat, label)。"""
    for i in range(5):
        gss(torch.randn(1, D), update=True, labels=torch.tensor([i % N_CLASSES]))
    result = gss.sample_replay(3)
    assert result is not None
    feat, label = result
    assert feat.shape == (3, D)
    assert label.shape == (3,)


def test_sample_replay_smaller_than_requested(gss):
    """请求数量超过缓冲区时，返回全部有效样本。"""
    for i in range(3):
        gss(torch.randn(1, D), update=True, labels=torch.tensor([i % N_CLASSES]))
    feat, label = gss.sample_replay(100)
    assert feat.shape[0] == 3


def test_reset_for_task_keeps_buffer(gss):
    """reset_for_task 不清空缓冲区（跨任务保留历史样本）。"""
    for i in range(5):
        gss(torch.randn(1, D), update=True, labels=torch.tensor([i % N_CLASSES]))
    n_before = gss._n_valid
    gss.reset_for_task(1)
    assert gss._n_valid == n_before


# -----------------------------------------------------------------------
# 梯度向量测试
# -----------------------------------------------------------------------

def test_compute_grad_vector_shape(gss):
    """_compute_grad_vector 返回正确维度的展平梯度。"""
    feat = torch.randn(D)
    g = gss._compute_grad_vector(feat, 0)
    expected_dim = N_CLASSES * (D + 1)  # weight (C×D) + bias (C,)
    assert g.shape == (expected_dim,), f"期望 {expected_dim}，实际 {g.shape}"


def test_compute_grad_vector_detached(gss):
    """_compute_grad_vector 返回值不含梯度（已 detach）。"""
    feat = torch.randn(D)
    g = gss._compute_grad_vector(feat, 0)
    assert not g.requires_grad


def test_grad_does_not_contaminate_outer_graph(gss):
    """_gss_update 不影响外部训练循环的梯度计算。"""
    x = torch.randn(4, D)
    y = torch.zeros(4, dtype=torch.long)

    # 外部训练步：forward → loss → backward
    logits, _ = gss(x, update=True, labels=y)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    # 分类头应有梯度
    assert gss.classifier.weight.grad is not None
    assert not torch.all(gss.classifier.weight.grad == 0)


# -----------------------------------------------------------------------
# GSS 驱逐逻辑测试
# -----------------------------------------------------------------------

def test_gss_replaces_redundant_sample():
    """满载后，重复同一模式应触发替换（相似度 > 0）。"""
    gss = GSSBaseline(make_cfg(memory_size=3, candidate_size=3), input_dim=D, n_classes=N_CLASSES)
    fixed = torch.ones(D)

    # 填满缓冲区（全部相同模式）
    for _ in range(3):
        gss(fixed.unsqueeze(0), update=True, labels=torch.tensor([0]))
    assert gss._n_valid == 3

    # 写入第 4 条（与已有样本完全相同 → cos_sim = 1.0 > 0）
    # 应发生替换，缓冲区大小不变
    gss(fixed.unsqueeze(0), update=True, labels=torch.tensor([0]))
    assert gss._n_valid == 3


def test_registered_as_gss():
    """GSSBaseline 应以 'gss' 注册到 MODEL_REGISTRY。"""
    from src.model_module import MODEL_REGISTRY
    assert "gss" in MODEL_REGISTRY
