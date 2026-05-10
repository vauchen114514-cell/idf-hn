"""DistilBERT + 标准 Experience Replay 基线。

对照 idf_hn_distilbert：两者都使用 DistilBERT encoder + 文本 replay 缓冲区，
唯一区别是 replay 管理策略：
  - distilbert_er:      纯随机 Reservoir Sampling（均匀采样）
  - idf_hn_distilbert:  IDF ForgetGate 选择性写入 + 能量优先 replay 采样

这使得 BWT 差异可以完全归因于记忆管理策略本身。

注册名称：distilbert_er
"""
import logging
import random

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.encoders.distilbert_encoder import DistilBertEncoder

logger = logging.getLogger(__name__)


@register_model("distilbert_er")
class DistilBertER(nn.Module):
    """DistilBERT + 标准 ER（随机 Reservoir Sampling 文本 replay 缓冲区）。

    与 idf_hn_distilbert 结构完全对齐：
      - 相同 encoder、相同线性分类头
      - 相同 text replay buffer 大小（memory_size）
      - 相同 dual-path forward（支持 tuple 和 tensor 输入）
    区别仅在于 replay 策略：均匀随机 vs IDF 选择性。

    Args:
        cfg: Hydra 配置，需包含 model.encoder.*、model.memory_size 字段。
        input_dim: 数据集报告的输入维度（被 encoder 768 维覆盖）。
        n_classes: 全局类别总数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()

        enc_cfg = cfg.model.encoder
        self.encoder = DistilBertEncoder(
            model_name=enc_cfg.model_name,
            dropout=getattr(enc_cfg, "dropout", 0.1),
        )
        self.classifier = nn.Linear(self.encoder.hidden_size, n_classes)

        # 文本回放缓冲区（Reservoir Sampling）
        self._text_buf: list[tuple[Tensor, Tensor, int]] = []
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
            update: 是否将当前批次写入文本缓冲区（训练时为 True）。
            labels: (B,) 样本标签。

        Returns:
            logits: (B, n_classes)。
            energy: (B,) 全零占位。
        """
        if isinstance(x, (tuple, list)):
            input_ids, attn_mask = x
            h = self.encoder(input_ids, attn_mask)
            if self.training and update and labels is not None:
                self._reservoir_add(
                    input_ids.detach().cpu(),
                    attn_mask.detach().cpu(),
                    labels.detach().cpu(),
                )
        else:
            h = x

        logits = self.classifier(h)
        energy = torch.zeros(h.shape[0], device=h.device)
        return logits, energy

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> tuple[Tensor, Tensor] | None:
        """从文本缓冲区均匀随机采样，用当前 encoder 重编码（梯度流过 encoder）。

        Returns:
            (h, labels) 或 None（缓冲区为空时）。
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
        """任务切换（跨任务保留缓冲区，与 ER 的标准行为一致）。"""
        logger.info(
            f"DistilBertER: 切换至 Task {task_id}（缓冲区保留，当前={len(self._text_buf)}）"
        )

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
