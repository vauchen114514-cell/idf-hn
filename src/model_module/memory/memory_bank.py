"""记忆库基类：定义统一的存储/检索/密度查询接口。

IDF-HN 使用记忆库来维护 Prototype 集合，用于计算有效激活密度
Ω_eff（正则项）。记忆库实现基于 Mini-Batch K-Means 聚类。
"""
import logging
from abc import ABC, abstractmethod

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseMemoryBank(ABC):
    """记忆库抽象基类。

    子类需实现 add、density、reset 方法。
    密度查询返回输入 u 附近的 Prototype 激活密度，
    用于 Ω_eff 正则项：Ω_eff = Σ_k ρ(u, c_k) · |M_k|

    Args:
        input_dim: 输入维度 D。
        n_prototypes: Prototype 数量 K。
        warmup_steps: 预热步数（此期间使用精确距离，跳过 ANN）。
    """

    def __init__(
        self,
        input_dim: int,
        n_prototypes: int = 50,
        warmup_steps: int = 200,
    ) -> None:
        self.input_dim = input_dim
        self.n_prototypes = n_prototypes
        self.warmup_steps = warmup_steps
        self._step = 0

    @property
    def is_warmed_up(self) -> bool:
        """Prototype 是否已完成预热。"""
        return self._step > self.warmup_steps

    @abstractmethod
    def add(self, u: Tensor) -> None:
        """将新样本 u 加入记忆库，增量更新 Prototype。

        Args:
            u: (D,) 输入向量。
        """

    @abstractmethod
    def density(self, u: Tensor) -> Tensor:
        """估计 u 附近的语义密度（Prototype 激活密度）。

        Args:
            u: (D,) 查询向量。

        Returns:
            标量 Tensor，密度值 ≥ 0。
        """

    @abstractmethod
    def reset(self) -> None:
        """清空记忆库（任务切换时使用）。"""

    @abstractmethod
    def prototype_centers(self) -> Tensor:
        """返回当前 Prototype 中心，形状 (K, D)。"""
