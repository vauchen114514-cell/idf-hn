"""Experience Replay（ER）基线：纯随机 Reservoir Sampling。

不含任何 Hopfield 动力学或 ForgetGate，作为 IDF-HN+Replay 的对照基线。
隔离问题：BWT 改善来自 Replay 本身 vs ForgetGate 选择性驱逐？

Reservoir Sampling 算法（Vitter 1985）：
- 前 buffer_size 条样本直接写入缓冲区
- 第 n 条（n > buffer_size）：以概率 buffer_size/n 随机替换一个已有槽位
- 保证缓冲区是全部已见样本的无偏均匀子集
"""
import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model

logger = logging.getLogger(__name__)


@register_model("er")
class ERBaseline(nn.Module):
    """Experience Replay 基线（Reservoir Sampling + 线性分类器）。

    与 IDF-HN 结构对齐：同样使用输入特征 u 直接分类，
    同样提供 sample_replay() 接口，供 ContinualTrainer 调用。
    区别：缓冲区通过均匀随机采样维护，无任何 Hopfield 或遗忘门逻辑。

    Args:
        cfg: 配置对象，需包含 model.memory_size。
        input_dim: 输入特征维度 D。
        n_classes: 分类头类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model
        self.buffer_size: int = getattr(m, "memory_size", 1000)
        self.input_dim = input_dim

        self.classifier = nn.Linear(input_dim, n_classes)

        # Reservoir 缓冲区（非参数，不参与梯度）
        self.register_buffer("_feat_buf", torch.zeros(self.buffer_size, input_dim))
        self.register_buffer("_label_buf", torch.full((self.buffer_size,), -1, dtype=torch.long))
        self._n_seen: int = 0          # 已见样本总数（含当前）
        self._n_valid: int = 0         # 缓冲区中有效槽位数

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
            update: 训练时为 True，将样本写入 Reservoir。
            labels: (B,) 标签，update=True 时必须提供。

        Returns:
            logits: (B, C) 分类 logits。
            energy: (B,) 全零占位（ER 无 Hopfield 能量）。
        """
        logits = self.classifier(x)
        dummy_energy = torch.zeros(x.shape[0], device=x.device)

        if update and labels is not None:
            for i in range(x.shape[0]):
                self._reservoir_update(x[i].detach(), int(labels[i].item()))

        return logits, dummy_energy

    def sample_replay(self, n: int, energy_priority: bool = False) -> tuple[Tensor, Tensor] | None:
        """从 Reservoir 缓冲区均匀随机采样 n 条 (特征, 标签)。

        Returns:
            (feat, label) 元组；缓冲区为空时返回 None。
        """
        if self._n_valid == 0:
            return None

        n_sample = min(n, self._n_valid)
        idx = torch.randperm(self._n_valid, device=self._feat_buf.device)[:n_sample]
        return self._feat_buf[idx].clone(), self._label_buf[idx].clone()

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时不清空缓冲区（ER 的关键特性：跨任务保留历史样本）。"""
        logger.info(f"ERBaseline: 切换至 Task {task_id}（缓冲区保留，当前有效={self._n_valid}）")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reservoir_update(self, feat: Tensor, label: int) -> None:
        """Reservoir Sampling 在线更新（Vitter 1985 Algorithm R）。

        Args:
            feat: (D,) 单条特征向量。
            label: 整数标签。
        """
        self._n_seen += 1

        if self._n_seen <= self.buffer_size:
            # 缓冲区未满：直接写入
            slot = self._n_seen - 1
            self._feat_buf[slot] = feat
            self._label_buf[slot] = label
            self._n_valid = self._n_seen
        else:
            # 以概率 buffer_size / n_seen 替换随机槽位
            j = torch.randint(0, self._n_seen, (1,)).item()
            if j < self.buffer_size:
                self._feat_buf[j] = feat
                self._label_buf[j] = label
