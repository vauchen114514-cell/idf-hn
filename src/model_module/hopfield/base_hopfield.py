"""Hopfield 网络基类。

定义所有 Hopfield 变体的统一接口：存储、检索、遗忘与能量查询。
"""
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseHopfieldNetwork(ABC, nn.Module):
    """Hopfield 网络抽象基类。

    所有 Hopfield 变体（MHN、IDF-HN 等）均继承此类，实现统一的
    存储/检索/遗忘接口，方便 Trainer 层面的统一调用。
    """

    def __init__(self, input_dim: int, beta: float) -> None:
        """
        Args:
            input_dim: 输入向量维度 D。
            beta: 逆温度参数（控制检索锐度）。
        """
        super().__init__()
        self.input_dim = input_dim
        self.beta = beta

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def store(self, u: Tensor) -> None:
        """将新模式 u 写入记忆。

        Args:
            u: (D,) 输入向量。
        """

    @abstractmethod
    def retrieve(self, xi: Tensor, n_steps: int = 1) -> Tensor:
        """从记忆中检索与 xi 最近似的模式。

        Args:
            xi: (D,) 或 (B, D) 查询向量。
            n_steps: 更新迭代次数（默认 1 步）。

        Returns:
            更新后的状态，形状与 xi 相同。
        """

    @abstractmethod
    def forget(self, gamma: float) -> None:
        """按遗忘率 gamma 对记忆施加衰减。

        Args:
            gamma: 遗忘率 ∈ [0, 1]，0 表示不遗忘，1 表示全部清零。
        """

    @abstractmethod
    def energy(self, xi: Tensor) -> Tensor:
        """计算当前状态 xi 的能量值。

        Args:
            xi: (D,) 或 (B, D) 状态向量。

        Returns:
            标量或 (B,) 能量值。
        """

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def memory_size(self) -> int:
        """当前存储的记忆条数。"""

    @property
    @abstractmethod
    def memory_matrix(self) -> Tensor:
        """(N, D) 形状的记忆矩阵，N 为当前记忆条数。"""

    # ------------------------------------------------------------------
    # 通用辅助方法
    # ------------------------------------------------------------------

    def forward(self, xi: Tensor, n_steps: int = 1) -> Tensor:
        """前向传播等同于 retrieve，方便作为 nn.Module 使用。"""
        return self.retrieve(xi, n_steps=n_steps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"beta={self.beta}, "
            f"memory_size={self.memory_size})"
        )
