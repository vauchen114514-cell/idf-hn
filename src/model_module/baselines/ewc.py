"""EWC（Elastic Weight Consolidation）基线。

Kirkpatrick et al. 2017 的经典持续学习方法，通过 Fisher 信息矩阵
对重要参数添加弹性惩罚，防止灾难性遗忘。

此实现为 Online EWC（每个新任务更新 Fisher 信息）。
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model_module import register_model

logger = logging.getLogger(__name__)


@register_model("ewc")
class EWCModel(nn.Module):
    """EWC 持续学习基线。

    结构：两层 MLP（input_dim → hidden_dim → n_classes）。
    每完成一个任务后，通过 consolidate() 更新 Fisher 信息。

    Args:
        cfg: 配置对象，需包含 model.hidden_dim 和 model.ewc_lambda。
        input_dim: 输入维度 D。
        n_classes: 分类类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model
        hidden_dim: int = getattr(m, "hidden_dim", 256)
        self.ewc_lambda: float = getattr(m, "ewc_lambda", 400.0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

        # 保存旧任务参数和 Fisher 信息
        self._fisher: dict[str, Tensor] = {}
        self._old_params: dict[str, Tensor] = {}

    def forward(self, x: Tensor, update: bool = True, **kwargs) -> tuple[Tensor, Tensor]:
        logits = self.net(x)
        # 返回全零能量（EWC 不使用 Hopfield 能量）
        dummy_energy = torch.zeros(x.shape[0], device=x.device)
        return logits, dummy_energy

    def ewc_loss(self) -> Tensor:
        """计算 EWC 正则项：Σ_i F_i (θ_i - θ*_i)² / 2。"""
        if not self._fisher:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.net.named_parameters():
            if name in self._fisher:
                fisher = self._fisher[name].to(param.device)
                old = self._old_params[name].to(param.device)
                loss = loss + (fisher * (param - old) ** 2).sum()
        return self.ewc_lambda * loss / 2.0

    def consolidate(
        self,
        data_loader,
        criterion: Optional[nn.Module] = None,
        n_samples: int = 200,
    ) -> None:
        """任务结束后估计 Fisher 信息矩阵并保存当前参数。

        Args:
            data_loader: 当前任务验证集 DataLoader。
            criterion: 损失函数（默认 CrossEntropy）。
            n_samples: 用于估计 Fisher 的样本数。
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        device = next(self.parameters()).device
        fisher: dict[str, Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.net.named_parameters()
        }

        self.eval()
        n_processed = 0
        for x, y in data_loader:
            if n_processed >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = self.forward(x, update=False)
            loss = criterion(logits, y)
            self.zero_grad()
            loss.backward()

            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            n_processed += x.shape[0]

        n_processed = max(n_processed, 1)
        for name in fisher:
            fisher[name] /= n_processed

        self._fisher = fisher
        self._old_params = {
            name: param.detach().clone()
            for name, param in self.net.named_parameters()
        }
        logger.info(f"EWC consolidate 完成，样本数={n_processed}")

    def reset_for_task(self, task_id: int) -> None:
        logger.info(f"EWC: 切换至 Task {task_id}")
