"""IDF-HN + 可微调 DistilBERT encoder 的组合模型。

架构：
    DistilBertEncoder（fine-tunable）→ (B, 768) 特征
        → IDFHopfieldNetwork（能量正则 + 选择性遗忘）
        → 分类 logits

前向传播支持两种输入类型（双路径）：
  1. 文本输入：(input_ids, attn_mask) tuple → encoder → IDF-HN
  2. 特征输入：FloatTensor (B, 768) → 直接进 IDF-HN（用于 replay）

Replay 设计：
  - 独立 text replay buffer（Reservoir Sampling）存储原始 tokenized 文本
  - sample_replay() 从 text buffer 取样，用当前 encoder 重编码（梯度流过 encoder）
  - 确保 replay 时 encoder 参数得到有效更新，避免表示漂移

注册名称：idf_hn_distilbert
"""
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.encoders.distilbert_encoder import DistilBertEncoder
from src.model_module.idf_hn.idf_hopfield import IDFHopfieldNetwork

logger = logging.getLogger(__name__)


@register_model("idf_hn_distilbert")
class IDFHopfieldNetworkWithEncoder(nn.Module):
    """IDF-HN + 可微调 DistilBERT encoder，用于 NLP 持续学习。

    Args:
        cfg: Hydra/OmegaConf 配置，需包含 model.encoder.* 和 model.* 字段。
        input_dim: 数据集报告的输入维度（应为 768，与 DistilBERT 输出一致）。
        n_classes: 全局类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()

        enc_cfg = cfg.model.encoder
        self.encoder = DistilBertEncoder(
            model_name=enc_cfg.model_name,
            dropout=getattr(enc_cfg, "dropout", 0.1),
        )

        # IDF-HN 核心：直接实例化，复用全部逻辑（Hopfield + ForgetGate + Prototype + Dreaming）
        # encoder 输出固定为 768 维（hidden_size），与 input_dim 一致
        self.idf_core = IDFHopfieldNetwork(cfg, input_dim=self.encoder.hidden_size, n_classes=n_classes)

        # Text replay buffer：Reservoir Sampling 存储原始 tokenized 文本
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
        labels: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """前向传播，支持文本 tuple 或预编码 tensor 两种输入。

        Args:
            x: tuple (input_ids, attn_mask) 或 FloatTensor (B, 768)。
            update: 是否执行 IDF-HN 在线更新（训练时为 True）。
            labels: (B,) 标签，用于 IDF-HN 更新记录。

        Returns:
            logits: (B, n_classes) 分类 logits。
            energy: (B,) 每样本 Hopfield 能量。
        """
        if isinstance(x, (tuple, list)):
            input_ids, attn_mask = x
            h = self.encoder(input_ids, attn_mask)  # (B, 768)
            # 添加到 text replay buffer（Reservoir Sampling）
            if self.training and update and labels is not None:
                self._reservoir_add(
                    input_ids.detach().cpu(),
                    attn_mask.detach().cpu(),
                    labels.detach().cpu(),   # 一次性 GPU→CPU，避免逐元素同步
                )
        else:
            # 已编码特征（来自 sample_replay）
            h = x

        return self.idf_core(h, update=update, labels=labels)

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> Optional[tuple[Tensor, Tensor]]:
        """从 text buffer 采样，用当前 encoder 重编码（梯度流过 encoder）。

        Args:
            n: 采样数量。
            energy_priority: 忽略（text buffer 使用均匀采样）。

        Returns:
            (h, labels)：编码特征 (n, 768) 和标签 (n,)，None 若 buffer 为空。
        """
        if not self._text_buf:
            return None

        k = min(n, len(self._text_buf))
        indices = random.sample(range(len(self._text_buf)), k)

        ids_list = [self._text_buf[i][0] for i in indices]
        mask_list = [self._text_buf[i][1] for i in indices]
        label_list = [self._text_buf[i][2] for i in indices]

        device = next(self.parameters()).device
        input_ids = torch.stack(ids_list).to(device)
        attn_mask = torch.stack(mask_list).to(device)
        labels = torch.tensor(label_list, dtype=torch.long, device=device)

        # 用当前 encoder 重编码，梯度可以流回 encoder 参数
        h = self.encoder(input_ids, attn_mask)  # (k, 768)
        return h, labels

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时重置 IDF-HN 内部状态（不清空 text buffer）。"""
        self.idf_core.reset_for_task(task_id)

    # ------------------------------------------------------------------
    # 诊断属性（透传 idf_core）
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return self.idf_core.memory_size

    @property
    def last_stats(self):
        return self.idf_core.last_stats

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reservoir_add(
        self,
        input_ids: Tensor,   # (B, L) CPU LongTensor
        attn_mask: Tensor,   # (B, L) CPU LongTensor
        labels: Tensor,      # (B,) CPU LongTensor
    ) -> None:
        """Vitter Algorithm R Reservoir Sampling。"""
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
