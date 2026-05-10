"""KV-Cache + 可微调 DistilBERT encoder 的组合模型。

架构：
    DistilBertEncoder（fine-tunable）→ (B, 768) 特征
        → KVCacheModel（Parzen-window 分类 + Reservoir Sampling 记忆）

与 IDFHopfieldNetworkWithEncoder 的对比：
  - IDF-HN：能量最小化 + 选择性遗忘门（gamma 依赖输入冲突 ΔE）
  - KV-Cache：Parzen-window + 纯随机 Reservoir Sampling（无结构选择性）

两者使用相同的双路径 forward 设计和 text replay buffer，
确保唯一变量是记忆机制本身，而非 encoder 或 replay 策略。

注册名称：kv_cache_distilbert
"""
import logging
import random

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.encoders.distilbert_encoder import DistilBertEncoder
from src.model_module.kv_cache.kv_cache_model import KVCacheModel

logger = logging.getLogger(__name__)


@register_model("kv_cache_distilbert")
class KVCacheModelWithEncoder(nn.Module):
    """KV-Cache + 可微调 DistilBERT encoder，用于 NLP 持续学习。

    Args:
        cfg: Hydra 配置，需包含 model.encoder.* 和 model.* 字段。
        input_dim: 数据集报告的输入维度（应为 768）。
        n_classes: 全局类别总数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()

        enc_cfg = cfg.model.encoder
        self.encoder = DistilBertEncoder(
            model_name=enc_cfg.model_name,
            dropout=getattr(enc_cfg, "dropout", 0.1),
        )

        # KV-Cache 核心：直接实例化，encoder.hidden_size 固定为 768
        self.kv_core = KVCacheModel(cfg, input_dim=self.encoder.hidden_size, n_classes=n_classes)

        # Text replay buffer（与 IDFHopfieldNetworkWithEncoder 完全一致）
        self._text_buf: list[tuple[Tensor, Tensor, int]] = []  # (input_ids_1d, attn_mask_1d, label)
        self._buf_max: int = getattr(cfg.model, "memory_size", 1000)
        self._buf_seen: int = 0

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        update: bool = True,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """双路径前向传播。

        Args:
            x: tuple (input_ids, attn_mask) 或 FloatTensor (B, 768)。
            update: 是否执行 KV 存储写入（训练时为 True）。
            labels: (B,) 标签。

        Returns:
            logits: (B, n_classes)。
            energy: (B,) 全零占位。
        """
        if isinstance(x, (tuple, list)):
            input_ids, attn_mask = x
            h = self.encoder(input_ids, attn_mask)  # (B, 768)
            if self.training and update and labels is not None:
                self._reservoir_add(
                    input_ids.detach().cpu(),
                    attn_mask.detach().cpu(),
                    labels.detach().cpu(),
                )
        else:
            h = x  # 已编码特征（来自 sample_replay）

        return self.kv_core(h, update=update, labels=labels)

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> tuple[Tensor, Tensor] | None:
        """从 text buffer 采样，用当前 encoder 重编码（梯度流过 encoder）。

        Returns:
            (h, labels)：编码特征 (k, 768) 和标签 (k,)，None 若 buffer 为空。
        """
        if not self._text_buf:
            return None

        k = min(n, len(self._text_buf))
        indices = random.sample(range(len(self._text_buf)), k)

        device = next(self.parameters()).device
        input_ids = torch.stack([self._text_buf[i][0] for i in indices]).to(device)
        attn_mask = torch.stack([self._text_buf[i][1] for i in indices]).to(device)
        label_vals = torch.tensor(
            [self._text_buf[i][2] for i in indices], dtype=torch.long, device=device
        )

        h = self.encoder(input_ids, attn_mask)  # 梯度流过 encoder
        return h, label_vals

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时 KV 核无需重置（跨任务保留所有记忆）。"""
        self.kv_core.reset_for_task(task_id)

    # ------------------------------------------------------------------
    # 诊断属性（透传 kv_core）
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return self.kv_core.memory_size

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reservoir_add(
        self,
        input_ids: Tensor,   # (B, L) CPU LongTensor
        attn_mask: Tensor,   # (B, L) CPU LongTensor
        labels: Tensor,      # (B,) CPU LongTensor
    ) -> None:
        """Vitter Algorithm R Reservoir Sampling（与 IDFHopfieldNetworkWithEncoder 相同逻辑）。"""
        B = input_ids.size(0)
        for i in range(B):
            self._buf_seen += 1
            item = (input_ids[i].clone(), attn_mask[i].clone(), int(labels[i].item()))
            if len(self._text_buf) < self._buf_max:
                self._text_buf.append(item)
            else:
                j = random.randint(0, self._buf_seen - 1)
                if j < self._buf_max:
                    self._text_buf[j] = item
