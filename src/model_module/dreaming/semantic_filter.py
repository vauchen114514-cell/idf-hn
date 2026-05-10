"""语义过滤器：在 Dreaming 过程中过滤掉与当前任务相关的记忆。

Dreaming 机制通过随机"取消学习"部分记忆来防止过拟合历史，
但必须保留与当前任务语义相关的记忆，避免灾难性遗忘加剧。

过滤策略：
    - 计算随机候选向量 r 与 Prototype 中心的余弦相似度
    - 相似度超过阈值 sim_threshold 的向量被判定为"当前任务相关"
    - 仅对低相似度（非相关）向量施加遗忘
"""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class SemanticFilter:
    """语义过滤器，用于 Dreaming 阶段筛选可遗忘的记忆。

    Args:
        sim_threshold: 余弦相似度阈值；高于此值的向量不参与 Dreaming。
    """

    def __init__(self, sim_threshold: float = 0.5) -> None:
        self.sim_threshold = sim_threshold

    def is_forgettable(
        self,
        candidate: Tensor,
        prototype_centers: Tensor,
    ) -> bool:
        """判断候选向量是否可以被遗忘（与现有 Prototype 语义距离远）。

        Args:
            candidate: (D,) 候选遗忘向量。
            prototype_centers: (K, D) 当前 Prototype 中心。

        Returns:
            True 表示可遗忘（与所有 Prototype 相似度均低于阈值）。
        """
        if prototype_centers.shape[0] == 0:
            return True  # 无 Prototype 时允许遗忘

        cand_norm = F.normalize(candidate.unsqueeze(0), dim=-1)   # (1, D)
        proto_norm = F.normalize(prototype_centers, dim=-1)       # (K, D)
        sims = (cand_norm @ proto_norm.T).squeeze(0)               # (K,)
        max_sim = float(sims.max().item())

        forgettable = max_sim < self.sim_threshold
        logger.debug(f"is_forgettable={forgettable} max_sim={max_sim:.4f}")
        return forgettable

    def filter_candidates(
        self,
        candidates: Tensor,
        prototype_centers: Tensor,
    ) -> Tensor:
        """从候选向量中筛选出可遗忘的子集。

        Args:
            candidates: (M, D) 候选向量矩阵。
            prototype_centers: (K, D) Prototype 中心。

        Returns:
            (M', D) 可遗忘向量；M' ≤ M。
        """
        if prototype_centers.shape[0] == 0:
            return candidates

        cand_norm = F.normalize(candidates, dim=-1)            # (M, D)
        proto_norm = F.normalize(prototype_centers, dim=-1)    # (K, D)
        sims = cand_norm @ proto_norm.T                        # (M, K)
        max_sims = sims.max(dim=-1).values                     # (M,)

        mask = max_sims < self.sim_threshold
        forgettable = candidates[mask]
        logger.debug(
            f"filter_candidates: {candidates.shape[0]} → {forgettable.shape[0]} "
            f"(sim_thresh={self.sim_threshold})"
        )
        return forgettable
