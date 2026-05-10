"""ModernHopfieldLayer 单元测试。"""
import torch
import pytest

from src.model_module.hopfield.mhn_layer import ModernHopfieldLayer


@pytest.fixture
def mhn():
    return ModernHopfieldLayer(input_dim=16, beta=1.0, max_memories=5)


def test_store_increments_memory(mhn):
    """store 应增加 memory_size。"""
    assert mhn.memory_size == 0
    mhn.store(torch.randn(16))
    assert mhn.memory_size == 1
    mhn.store(torch.randn(16))
    assert mhn.memory_size == 2


def test_store_fifo_eviction(mhn):
    """超出 max_memories 时应 FIFO 淘汰旧记忆。"""
    for _ in range(7):  # max=5
        mhn.store(torch.randn(16))
    assert mhn.memory_size == 5


def test_retrieve_returns_correct_shape(mhn):
    """retrieve 输出形状与 xi 相同。"""
    mhn.store(torch.randn(16))
    xi = torch.randn(16)
    out = mhn.retrieve(xi)
    assert out.shape == xi.shape


def test_retrieve_batch(mhn):
    """批量 retrieve 支持 (B, D) 输入。"""
    for _ in range(3):
        mhn.store(torch.randn(16))
    xi_batch = torch.randn(4, 16)
    out = mhn.retrieve(xi_batch)
    assert out.shape == xi_batch.shape


def test_forget_decays_memory(mhn):
    """forget 应减小记忆矩阵的范数。"""
    for _ in range(3):
        mhn.store(torch.ones(16))
    norm_before = mhn.memory_matrix.norm().item()
    mhn.forget(gamma=0.5)
    norm_after = mhn.memory_matrix.norm().item()
    assert norm_after < norm_before


def test_conflict_empty_memory(mhn):
    """空记忆时 conflict 应返回 0。"""
    u = torch.randn(16)
    xi = torch.randn(16)
    de = mhn.conflict(u, xi)
    assert float(de) == 0.0


def test_reset_memory(mhn):
    """reset_memory 应清空记忆矩阵。"""
    mhn.store(torch.randn(16))
    mhn.reset_memory()
    assert mhn.memory_size == 0


def test_store_wrong_dim(mhn):
    """store 应拒绝错误维度的输入。"""
    with pytest.raises(ValueError):
        mhn.store(torch.randn(32))


def test_norm_min_eviction_evicts_lowest_norm():
    """norm_min 策略：缓冲区满时应驱逐范数最小的槽。"""
    D = 8
    mhn = ModernHopfieldLayer(input_dim=D, beta=1.0, max_memories=3, eviction_policy="norm_min")

    # 写入 3 条：范数分别为大/小/大
    big1 = torch.ones(D) * 10.0
    small = torch.ones(D) * 0.001  # 范数最小，应被驱逐
    big2 = torch.ones(D) * 8.0
    mhn.store(big1)
    mhn.store(small)
    mhn.store(big2)
    assert mhn.memory_size == 3

    # 写入第 4 条，应覆盖 small 的位置
    new_pattern = torch.ones(D) * 5.0
    mhn.store(new_pattern)

    assert mhn.memory_size == 3  # 总数不变
    # small（范数 ≈ 0.008）应被替换；big1/big2/new_pattern 的范数远大于 small
    norms = mhn.memory_matrix.norm(dim=1).tolist()
    # 验证没有任何槽的范数接近 small（≈ 0.008）
    assert all(n > 0.1 for n in norms), f"small 应被驱逐，但仍存在: {norms}"


def test_invalid_eviction_policy():
    """不合法的驱逐策略应抛出 ValueError。"""
    with pytest.raises(ValueError, match="eviction_policy"):
        ModernHopfieldLayer(input_dim=8, beta=1.0, eviction_policy="random")


# ------------------------------------------------------------------
# 双缓冲区测试
# ------------------------------------------------------------------

def test_forget_does_not_modify_replay_buf():
    """forget() 只衰减能量缓冲区，回放缓冲区中的原始特征不变。"""
    D = 8
    mhn = ModernHopfieldLayer(input_dim=D, beta=1.0, max_memories=5, eviction_policy="norm_min")
    u = torch.ones(D)
    mhn.store(u, label=0)

    replay_norm_before = mhn._replay_buf[0].norm().item()
    mhn.forget(gamma=0.5)
    replay_norm_after = mhn._replay_buf[0].norm().item()

    # 能量缓冲区已衰减
    assert mhn._mem[0].norm().item() < u.norm().item() - 0.1
    # 回放缓冲区完全不变
    assert abs(replay_norm_before - replay_norm_after) < 1e-6, (
        f"forget() 不应修改回放缓冲区: before={replay_norm_before}, after={replay_norm_after}"
    )


def test_sample_replay_returns_original_features():
    """sample_replay() 应返回原始特征（不受 forget() 影响）。"""
    D = 8
    mhn = ModernHopfieldLayer(input_dim=D, beta=1.0, max_memories=5, eviction_policy="norm_min")
    u = torch.ones(D) * 2.0
    mhn.store(u, label=1)

    # 多次 forget，能量缓冲区趋近零
    for _ in range(50):
        mhn.forget(gamma=0.5)

    result = mhn.sample_replay(1)
    assert result is not None
    feat, label = result
    # 回放特征范数应接近原始（2.0 * sqrt(8) ≈ 5.66），而非接近 0
    assert feat.norm().item() > 1.0, f"回放特征不应被 forget 衰减: norm={feat.norm().item():.4f}"
    assert int(label[0].item()) == 1


def test_single_buffer_replay_returns_decayed_features():
    """dual_buffer=False 时，回放特征来自被 forget() 衰减后的能量缓冲区。"""
    D = 8
    mhn = ModernHopfieldLayer(
        input_dim=D,
        beta=1.0,
        max_memories=5,
        eviction_policy="norm_min",
        dual_buffer=False,
    )
    u = torch.ones(D) * 2.0
    mhn.store(u, label=1)

    for _ in range(10):
        mhn.forget(gamma=0.5)

    result = mhn.sample_replay(1)
    assert result is not None
    feat, label = result
    assert feat.norm().item() < 0.1, (
        f"单缓冲消融应暴露 replay feature collapse，当前 norm={feat.norm().item():.4f}"
    )
    assert int(label[0].item()) == 1


def test_reset_memory_clears_both_buffers():
    """reset_memory() 应同时清空能量缓冲区和回放缓冲区。"""
    D = 8
    mhn = ModernHopfieldLayer(input_dim=D, beta=1.0, max_memories=5)
    for _ in range(3):
        mhn.store(torch.randn(D), label=0)

    mhn.reset_memory()

    assert mhn._n_stored == 0
    assert mhn._replay_n_stored == 0
    assert mhn._replay_n_total_seen == 0
    assert mhn.sample_replay(1) is None
