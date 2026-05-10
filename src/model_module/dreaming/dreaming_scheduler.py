"""Dreaming 调度器：周期性随机遗忘非相关记忆以防过拟合。

Dreaming 机制（受生物学启发）：
    每隔 freq 步，从记忆矩阵中随机采样 n_dream_samples 个向量，
    通过语义过滤器筛选非当前任务相关的向量，对其施加加速遗忘。

效果：等价于软"突触修剪"，避免记忆矩阵被早期任务永久占据。
"""
import logging

import torch
from torch import Tensor

from src.model_module.dreaming.semantic_filter import SemanticFilter

logger = logging.getLogger(__name__)


class DreamingScheduler:
    """Dreaming 调度器。

    Args:
        freq: 每隔多少步触发一次 Dreaming。
        n_dream_samples: 每次 Dreaming 采样的候选记忆数量。
        dream_gamma: Dreaming 时对目标记忆施加的额外遗忘率。
        sim_threshold: 传递给 SemanticFilter 的余弦相似度阈值。
        enabled: 是否启用 Dreaming。
    """

    def __init__(
        self,
        freq: int = 100,
        n_dream_samples: int = 50,
        dream_gamma: float = 0.3,
        sim_threshold: float = 0.5,
        enabled: bool = True,
        rng_seed: int = 1337,
    ) -> None:
        self.freq = freq
        self.n_dream_samples = n_dream_samples
        self.dream_gamma = dream_gamma
        self.enabled = enabled
        self._filter = SemanticFilter(sim_threshold=sim_threshold)
        self._step = 0
        # 独立随机生成器：避免 randperm 污染全局 RNG（DataLoader 随机状态）
        self._rng = torch.Generator()
        self._rng.manual_seed(rng_seed)

    def step(
        self,
        memory_matrix: Tensor,
        prototype_centers: Tensor,
    ) -> Tensor:
        """每步调用；在触发时机返回应施加额外遗忘的记忆行索引。

        Args:
            memory_matrix: (N, D) 当前记忆矩阵。
            prototype_centers: (K, D) Prototype 中心（用于语义过滤）。

        Returns:
            dream_indices: 应施加额外遗忘的行索引 Tensor；
                           未触发时返回空 Tensor。
        """
        self._step += 1
        dev = memory_matrix.device
        if not self.enabled or self._step % self.freq != 0:
            return torch.tensor([], dtype=torch.long, device=dev)
        return self._do_dream(memory_matrix, prototype_centers)

    def maybe_fire(
        self,
        n_steps: int,
        memory_matrix: Tensor,
        prototype_centers: Tensor,
    ) -> Tensor:
        """批量推进步数，越过 freq 边界时触发一次 Dreaming。

        Args:
            n_steps: 本批次样本数（推进的步数）。
            memory_matrix: (N, D) 当前记忆矩阵。
            prototype_centers: (K, D) Prototype 中心。

        Returns:
            dream_indices: 应施加额外遗忘的行索引；未触发时返回空 Tensor。
        """
        old_step = self._step
        self._step += n_steps
        dev = memory_matrix.device
        if not self.enabled or self.freq <= 0:
            return torch.tensor([], dtype=torch.long, device=dev)
        # 检测是否越过至少一个 freq 整数倍边界
        if self._step // self.freq <= old_step // self.freq:
            return torch.tensor([], dtype=torch.long, device=dev)
        return self._do_dream(memory_matrix, prototype_centers)

    def _do_dream(self, memory_matrix: Tensor, prototype_centers: Tensor) -> Tensor:
        """执行一次 Dreaming，返回应被遗忘的记忆行索引。"""
        n = memory_matrix.shape[0]
        dev = memory_matrix.device
        if n == 0:
            return torch.tensor([], dtype=torch.long, device=dev)

        # 随机采样候选记忆行
        n_sample = min(self.n_dream_samples, n)
        candidate_indices = torch.randperm(n, generator=self._rng)[:n_sample]
        candidates = memory_matrix[candidate_indices]

        # 语义过滤：只遗忘与 Prototype 语义距离远的记忆
        forgettable = self._filter.filter_candidates(candidates, prototype_centers)
        if forgettable.shape[0] == 0:
            logger.debug("Dreaming 触发但无可遗忘记忆（所有候选均为当前任务相关）")
            return torch.tensor([], dtype=torch.long, device=dev)

        # 找可遗忘向量在原始记忆矩阵中的行索引
        forget_indices = []
        for vec in forgettable:
            diffs = ((candidates - vec.unsqueeze(0)) ** 2).sum(dim=-1)
            local_idx = int(diffs.argmin().item())
            forget_indices.append(int(candidate_indices[local_idx].item()))

        dream_indices = torch.tensor(
            list(set(forget_indices)), dtype=torch.long, device=dev
        )
        logger.info(
            f"Dreaming 触发 (step={self._step})："
            f" 候选 {n_sample} → 可遗忘 {len(dream_indices)} 条记忆"
        )
        return dream_indices

    def reset(self) -> None:
        """重置步数计数器。"""
        self._step = 0

    @property
    def step_count(self) -> int:
        return self._step
