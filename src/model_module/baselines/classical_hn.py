"""经典 Hopfield 网络基线（无遗忘机制）。

直接用 MHN 存储所有输入，不执行任何遗忘，作为最差情况基线。
预期会严重的灾难性遗忘（高 BWT 负值）。
"""
import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.hopfield.mhn_layer import ModernHopfieldLayer

logger = logging.getLogger(__name__)


@register_model("classical_hn")
@register_model("mhn")   # mhn.yaml 使用此别名
class ClassicalHopfieldBaseline(nn.Module):
    """标准 MHN 基线（仅存储，无遗忘）。

    Args:
        cfg: 配置对象。
        input_dim: 输入维度 D。
        n_classes: 分类头类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model
        self.hopfield = ModernHopfieldLayer(
            input_dim=input_dim,
            beta=m.beta,
            max_memories=m.memory_size,
        )
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, x: Tensor, update: bool = True, **kwargs) -> tuple[Tensor, Tensor]:
        logits_list = []
        energies = []

        for i in range(x.shape[0]):
            u = x[i]
            if update:
                self.hopfield.store(u)
            xi = self.hopfield.retrieve(u, n_steps=1)
            logits_list.append(self.classifier(xi))
            energies.append(self.hopfield.energy(xi))

        return torch.stack(logits_list), torch.stack(energies)

    def reset_for_task(self, task_id: int) -> None:
        logger.info(f"ClassicalHN: 切换至 Task {task_id}（不清空记忆）")
