"""Sparse Memory 基线：Hopfield 记忆 + 全局常数衰减（无 ForgetGate）。

设计目的：
    与 IDF-HN 的对照基线，隔离"输入依赖遗忘"的贡献。
    - IDF-HN: γ = f(conflict(u, M))，输入依赖
    - SparseMemoryBaseline: γ = γ₀（常数），与输入无关

    "Sparse" 指常数衰减使低频记忆范数趋零，记忆矩阵随时间变稀疏。

差异对比：
    vs ER           : 有 Hopfield 记忆结构（非纯随机回放）
    vs IDF-HN       : γ 固定，无冲突检测，不依赖输入
    vs time_decay   : 相同的 γ 固定思想，但 SparseMemory 始终写（write_threshold=None）
                      且用 reservoir 淘汰（不依赖范数）
"""
import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.hopfield.mhn_layer import ModernHopfieldLayer

logger = logging.getLogger(__name__)


@register_model("sparse_memory")
class SparseMemoryBaseline(nn.Module):
    """Sparse External Memory 基线（常数全局衰减 Hopfield）。

    结构与 IDF-HN 对齐：
        - 同样使用 ModernHopfieldLayer（Reservoir 淘汰）
        - 同样提供 sample_replay()
        - 分类走原始输入 u（与 IDF-HN 一致）
    区别：
        - 无 ForgetGate（无 conflict 检测）
        - 每步固定衰减：_mem.mul_(1 - gamma)
        - 始终写入（无 write_threshold 选择性）

    Args:
        cfg: 配置对象，需包含 model.memory_size, model.beta, model.gamma。
        input_dim: 输入特征维度 D。
        n_classes: 分类头类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model

        self.hopfield = ModernHopfieldLayer(
            input_dim=input_dim,
            beta=m.beta,
            max_memories=m.memory_size,
            eviction_policy="reservoir",   # 无范数驱逐（不依赖 ForgetGate 的衰减结果）
        )

        # 常数遗忘率：每步对所有记忆施加均匀衰减
        self.gamma: float = getattr(m, "gamma", 0.05)
        self.classifier = nn.Linear(input_dim, n_classes)

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        update: bool = True,
        labels: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """前向传播。

        Args:
            x: (B, D) 输入批次。
            update: 训练时为 True，执行衰减 + 写入。
            labels: (B,) 标签，update=True 时必须提供。

        Returns:
            logits: (B, C) 分类 logits。
            energy: (B,) 全零占位（与 ER 保持一致）。
        """
        logits = self.classifier(x)
        dummy_energy = torch.zeros(x.shape[0], device=x.device)

        if update and labels is not None:
            for i in range(x.shape[0]):
                u = x[i].detach()
                label = int(labels[i].item())
                # 常数全局衰减（在写入前施加，避免新写入向量被立即衰减）
                if self.hopfield.memory_size > 0:
                    self.hopfield.forget(self.gamma)
                self.hopfield.store(u, label=label)

        return logits, dummy_energy

    def sample_replay(self, n: int, energy_priority: bool = False) -> tuple[Tensor, Tensor] | None:
        """从回放缓冲区均匀随机采样（原始特征，不受 ForgetGate 衰减）。"""
        return self.hopfield.sample_replay(n)

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时不清空记忆（跨任务累积）。"""
        logger.info(
            f"SparseMemoryBaseline: 切换至 Task {task_id}"
            f"（记忆保留，energy_buf={self.hopfield.memory_size}）"
        )
