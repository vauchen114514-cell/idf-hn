"""ER 基线单元测试。

验证 Experience Replay (Reservoir Sampling) 基线的关键性质：
1. Reservoir 填满前直接写入，_n_valid 正确递增
2. Reservoir 填满后不超出 buffer_size
3. sample_replay 返回形状正确
4. 缓冲区为空时 sample_replay 返回 None
5. 前向传播输出形状正确
6. reset_for_task 不清空缓冲区
7. 缓冲区样本分布均匀性（统计检验）
"""
import torch
import pytest
from types import SimpleNamespace


def make_cfg(memory_size: int = 100) -> SimpleNamespace:
    m = SimpleNamespace(memory_size=memory_size)
    return SimpleNamespace(model=m)


@pytest.fixture
def er():
    from src.model_module.baselines.er import ERBaseline
    cfg = make_cfg(memory_size=100)
    return ERBaseline(cfg, input_dim=16, n_classes=5)


def test_fill_phase(er):
    """前 buffer_size 条样本直接写入，_n_valid 正确递增。"""
    for i in range(50):
        er._reservoir_update(torch.randn(16), label=i % 5)
    assert er._n_valid == 50
    assert er._n_seen == 50
    # 标签不全为 -1
    assert (er._label_buf[:50] >= 0).all()


def test_buffer_size_not_exceeded(er):
    """写入超过 buffer_size 条样本后，_n_valid 不超过 buffer_size。"""
    for i in range(300):
        er._reservoir_update(torch.randn(16), label=i % 5)
    assert er._n_valid == 100
    assert er._n_seen == 300


def test_sample_replay_shape(er):
    """sample_replay 返回形状正确。"""
    for i in range(50):
        er._reservoir_update(torch.randn(16), label=i % 5)
    result = er.sample_replay(10)
    assert result is not None
    feat, label = result
    assert feat.shape == (10, 16)
    assert label.shape == (10,)


def test_sample_replay_empty():
    """缓冲区为空时 sample_replay 返回 None。"""
    from src.model_module.baselines.er import ERBaseline
    cfg = make_cfg(100)
    model = ERBaseline(cfg, input_dim=16, n_classes=5)
    assert model.sample_replay(10) is None


def test_sample_replay_smaller_than_requested(er):
    """请求数量大于有效样本时，返回实际有效数量。"""
    er._reservoir_update(torch.randn(16), label=0)
    result = er.sample_replay(50)
    assert result is not None
    feat, _ = result
    assert feat.shape[0] == 1


def test_forward_shape(er):
    """前向传播输出形状正确。"""
    x = torch.randn(8, 16)
    y = torch.randint(0, 5, (8,))
    logits, energy = er(x, update=True, labels=y)
    assert logits.shape == (8, 5)
    assert energy.shape == (8,)
    assert (energy == 0).all(), "ER 能量应全为 0"


def test_forward_writes_to_buffer(er):
    """update=True 时，forward 将样本写入缓冲区。"""
    x = torch.randn(20, 16)
    y = torch.randint(0, 5, (20,))
    er(x, update=True, labels=y)
    assert er._n_seen == 20
    assert er._n_valid == 20


def test_forward_no_update(er):
    """update=False 时，缓冲区不被修改。"""
    x = torch.randn(10, 16)
    er(x, update=False)
    assert er._n_seen == 0


def test_reset_for_task_keeps_buffer(er):
    """reset_for_task 不清空缓冲区。"""
    for i in range(50):
        er._reservoir_update(torch.randn(16), label=i % 5)
    er.reset_for_task(1)
    assert er._n_valid == 50, "reset_for_task 不应清空缓冲区"


def test_reservoir_uniformity():
    """统计验证 Reservoir 采样的均匀性。

    写入 10000 条样本（buffer_size=100），检查每个位置被最终保留的
    样本 index 分布是否均匀（近似）。
    """
    from src.model_module.baselines.er import ERBaseline
    import random

    cfg = make_cfg(memory_size=100)
    er = ERBaseline(cfg, input_dim=1, n_classes=2)

    n_total = 10000
    # 用 label 记录写入顺序（0 ~ 9999）
    for i in range(n_total):
        er._reservoir_update(torch.tensor([float(i)]), label=0)

    # 缓冲区中的特征值对应写入顺序 index
    indices = er._feat_buf[:er._n_valid, 0].tolist()

    # 分成 10 桶，每桶 1000 条，理想情况每桶应各占 ~10%
    buckets = [0] * 10
    for idx in indices:
        buckets[int(idx) // 1000] += 1

    # 放宽至 ±70%（Reservoir 方差，100 个槽的统计波动较大）
    for b_count in buckets:
        assert 3 <= b_count <= 17, f"分布严重不均匀：各桶={buckets}"
