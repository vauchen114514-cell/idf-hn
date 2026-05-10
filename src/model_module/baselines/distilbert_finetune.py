"""纯 DistilBERT 顺序微调基线（无任何 Replay 机制）。

作为持续学习的下界对照：展示灾难性遗忘的严重程度。
架构：DistilBertEncoder → Linear(768, n_classes)
无记忆缓冲区，无跨任务保护，每个任务直接覆盖前序参数。

注册名称：distilbert_finetune
"""
import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.encoders.distilbert_encoder import DistilBertEncoder

logger = logging.getLogger(__name__)


@register_model("distilbert_finetune")
class DistilBertFinetune(nn.Module):
    """纯 DistilBERT 顺序微调，无 replay，用于展示灾难性遗忘下界。

    Args:
        cfg: Hydra 配置，需包含 model.encoder.* 字段。
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

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        update: bool = True,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """前向传播。

        Args:
            x: tuple (input_ids, attn_mask) 或 FloatTensor (B, 768)。
            update: 无实际作用（finetune 无缓冲区），保持接口一致。
            labels: 无实际作用，保持接口一致。

        Returns:
            logits: (B, n_classes)。
            energy: (B,) 全零占位。
        """
        if isinstance(x, (tuple, list)):
            input_ids, attn_mask = x
            h = self.encoder(input_ids, attn_mask)
        else:
            h = x

        logits = self.classifier(h)
        energy = torch.zeros(h.shape[0], device=h.device)
        return logits, energy

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> None:
        """无 replay 缓冲区，始终返回 None。"""
        return None

    def reset_for_task(self, task_id: int) -> None:
        """无状态重置（无缓冲区）。"""
        logger.info(f"DistilBertFinetune: 切换至 Task {task_id}（无 replay）")
