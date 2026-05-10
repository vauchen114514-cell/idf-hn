"""持续学习指标单元测试。"""
import pytest

from src.trainer_module.metrics import ContinualMetrics


def test_average_accuracy():
    """AA 应为最终任务后所有任务准确率的均值。"""
    m = ContinualMetrics(n_tasks=3)
    # 填充 R 矩阵对角线下方（task_i evaluated after task_j）
    m.update(0, 0, 0.9)
    m.update(0, 1, 0.85)
    m.update(0, 2, 0.80)  # 任务0在最终时刻的准确率
    m.update(1, 1, 0.88)
    m.update(1, 2, 0.84)  # 任务1在最终时刻的准确率
    m.update(2, 2, 0.82)  # 任务2在最终时刻的准确率

    aa = m.average_accuracy()
    expected = (0.80 + 0.84 + 0.82) / 3
    assert abs(aa - expected) < 1e-6


def test_backward_transfer_negative():
    """发生遗忘时 BWT 应为负值。"""
    m = ContinualMetrics(n_tasks=3)
    # 对角线：任务完成时的准确率
    m.update(0, 0, 0.9)
    m.update(1, 1, 0.88)
    m.update(2, 2, 0.85)
    # 最终准确率（下降了）
    m.update(0, 2, 0.70)  # 任务0退化
    m.update(1, 2, 0.75)  # 任务1退化

    bwt = m.backward_transfer()
    assert bwt < 0, f"发生遗忘时 BWT 应为负，得到 {bwt:.4f}"


def test_backward_transfer_zero():
    """无遗忘时 BWT 应为 0。"""
    m = ContinualMetrics(n_tasks=3)
    m.update(0, 0, 0.9)
    m.update(0, 2, 0.9)  # 未退化
    m.update(1, 1, 0.88)
    m.update(1, 2, 0.88)  # 未退化
    m.update(2, 2, 0.85)

    bwt = m.backward_transfer()
    assert abs(bwt) < 1e-6, f"无遗忘时 BWT 应为 0，得到 {bwt:.4f}"


def test_single_task_bwt():
    """单任务时 BWT 应返回 0。"""
    m = ContinualMetrics(n_tasks=1)
    m.update(0, 0, 0.9)
    assert m.backward_transfer() == 0.0


def test_summary_keys():
    """summary() 应包含 AA、BWT、FWT 三个键。"""
    m = ContinualMetrics(n_tasks=2)
    m.update(0, 0, 0.9)
    m.update(0, 1, 0.85)
    m.update(1, 1, 0.88)
    s = m.summary()
    assert set(s.keys()) == {"AA", "BWT", "FWT"}
