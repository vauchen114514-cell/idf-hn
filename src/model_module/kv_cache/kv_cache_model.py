"""KV-Cache 持续学习模型：点积注意力检索的显式记忆分类器。

与 IDF-HN 的对比关系：
  - IDF-HN:   Hopfield 能量最小化 + 输入依赖遗忘门 → ξ 作为分类输入
  - KV-Cache: 点积注意力 + Reservoir Sampling     → 无参数遗忘机制

分类原理（Parzen-window 分类器）：
    logits[b, c] = β · Σ_{j: label_j == c} (x[b] · key_j)

即对每个类别，将该类所有已存记忆与查询向量的点积求和，作为该类的未归一化得分。
这等价于在 softmax(β · keys @ x) 上做标签加权，是 MHN 的软最近邻变体。

记忆管理：Reservoir Sampling（Vitter Algorithm R）。
  相比 FIFO 更均匀，相比 IDF-HN 完全无结构选择性——形成对照实验。

注册名称：kv_cache
"""
import logging
import random

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model

logger = logging.getLogger(__name__)


@register_model("kv_cache")
class KVCacheModel(nn.Module):
    """KV-Cache 持续学习模型（特征输入版）。

    Args:
        cfg: Hydra 配置，需包含 model.memory_size 和 model.beta。
        input_dim: 输入特征维度 D。
        n_classes: 全局类别总数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()

        m = cfg.model
        self.n_classes = n_classes
        self.beta: float = getattr(m, "beta", 1.0)
        self.max_memories: int = m.memory_size

        # 键值存储（register_buffer 随 .to(device) 迁移，但不参与梯度）
        self.register_buffer("_keys", torch.zeros(self.max_memories, input_dim))
        self.register_buffer("_vals", torch.zeros(self.max_memories, dtype=torch.long))

        # Python 状态（不被 state_dict 保存，仅用于训练期间）
        self._n_stored: int = 0
        self._n_seen: int = 0

        # 后备线性分类器：记忆未预热时（_n_stored == 0）使用
        self.fallback_classifier = nn.Linear(input_dim, n_classes)

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        update: bool = True,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """前向传播：更新 KV 存储并返回分类 logits。

        Args:
            x: (B, D) 特征向量。
            update: 是否将当前批次写入 KV 存储。
            labels: (B,) 整数标签，update=True 时必须提供。

        Returns:
            logits: (B, n_classes) 分类得分（Parzen-window）。
            energy: (B,) 全零占位（KV-Cache 无能量函数）。
        """
        B = x.shape[0]
        device = x.device

        if update and labels is not None:
            for i in range(B):
                self._reservoir_add(x[i].detach(), int(labels[i].item()))

        if self._n_stored == 0:
            energy = torch.zeros(B, device=device)
            return self.fallback_classifier(x), energy

        keys = self._keys[:self._n_stored]       # (M, D)
        vals = self._vals[:self._n_stored]        # (M,)
        scores = (x @ keys.T) * self.beta         # (B, M)

        # Parzen-window：对每个类聚合相似度得分
        logits = torch.zeros(B, self.n_classes, device=device)
        logits.scatter_add_(1, vals.unsqueeze(0).expand(B, -1), scores)

        energy = torch.zeros(B, device=device)
        return logits, energy

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> tuple[Tensor, Tensor] | None:
        """随机采样 n 条 (key, label) 用于回放训练。"""
        if self._n_stored == 0:
            return None
        k = min(n, self._n_stored)
        idx = torch.randperm(self._n_stored, device=self._keys.device)[:k]
        return self._keys[idx].clone(), self._vals[idx].clone()

    def reset_for_task(self, task_id: int) -> None:
        """KV-Cache 跨任务保留所有记忆，无需重置。"""
        logger.info(f"KVCacheModel 切换至 Task {task_id}（记忆保留，n_stored={self._n_stored}）")

    # ------------------------------------------------------------------
    # 诊断属性
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return self._n_stored

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reservoir_add(self, feature: Tensor, label: int) -> None:
        """Vitter Algorithm R：以 1/n 概率替换已有记忆。"""
        self._n_seen += 1
        if self._n_stored < self.max_memories:
            slot = self._n_stored
            self._n_stored += 1
        else:
            j = random.randint(0, self._n_seen - 1)
            if j >= self.max_memories:
                return
            slot = j
        self._keys[slot] = feature
        self._vals[slot] = label
