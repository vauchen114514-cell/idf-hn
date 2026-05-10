"""冲突检测器：基于 ΔE 能量差计算输入与记忆的冲突程度。

实现 IDF-HN 论文 Decision 1：conflict(u, M) = ΔE（能量差度量）。
冲突度高表示新输入与当前记忆正交或矛盾，应触发较大遗忘率。
"""
import logging
from collections import deque

import torch
from torch import Tensor

from src.model_module.hopfield.energy import delta_energy

logger = logging.getLogger(__name__)


class ConflictDetector:
    """基于 ΔE 的冲突检测器。

    维护历史冲突分数的滑动窗口，用于自适应 τ 计算（百分位数阈值）。

    Args:
        beta: 逆温度参数（与 Hopfield 层共享）。
        history_size: 用于百分位数估计的历史窗口大小。
        tau_percentile: 自适应 τ 的百分位数（默认 75）。
    """

    def __init__(
        self,
        beta: float,
        history_size: int = 500,
        tau_percentile: float = 75.0,
    ) -> None:
        self.beta = beta
        self.tau_percentile = tau_percentile
        self._history: deque[float] = deque(maxlen=history_size)

    def compute(self, u: Tensor, xi: Tensor, X: Tensor) -> Tensor:
        """计算新输入 u 相对于当前记忆 X 和状态 xi 的冲突度 ΔE。

        ΔE = E(ξ; X∪{u}) - E(ξ; X)
           = -lse(β, [Xξ, u·ξ]) + lse(β, Xξ)

        当 X 为空时返回 0（无法比较冲突）。

        Args:
            u: (D,) 新输入向量。
            xi: (D,) 当前查询状态。
            X: (N, D) 当前记忆矩阵；N 可为 0。

        Returns:
            标量 ΔE（Tensor）。
        """
        if X.shape[0] == 0:
            return torch.tensor(0.0, device=u.device)

        conflict = delta_energy(xi, X, u, self.beta)
        self._history.append(float(conflict.item()))
        return conflict

    def adaptive_tau(self, fallback: float = 0.3) -> float:
        """根据历史冲突分布计算自适应阈值 τ。

        返回历史冲突分数的第 tau_percentile 百分位数；
        历史不足 10 条时返回 fallback。

        Args:
            fallback: 历史不足时的默认阈值。

        Returns:
            自适应 τ 值（float）。
        """
        if len(self._history) < 10:
            return fallback

        history_tensor = torch.tensor(list(self._history))
        tau = float(torch.quantile(history_tensor, self.tau_percentile / 100.0).item())
        logger.debug(f"adaptive_tau={tau:.4f} (p{self.tau_percentile:.0f})")
        return tau

    def reset_history(self) -> None:
        """清空历史冲突记录（任务切换时调用）。"""
        self._history.clear()

    @property
    def n_history(self) -> int:
        """当前历史记录条数。"""
        return len(self._history)
