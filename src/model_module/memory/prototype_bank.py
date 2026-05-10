"""基于 Mini-Batch K-Means 的 Prototype 记忆库。

在线增量更新 Prototype 中心，无需存储所有历史样本。
密度估计使用 RBF 核：ρ(u, c_k) = exp(-‖u - c_k‖² / (2σ²))

IDF-HN 正则项：
    Ω_eff(M, u) = Σ_k ρ(u, c_k) · n_k / N_total

其中 n_k 是第 k 个 Prototype 所代表的样本数，N_total 为总样本数。
"""
import logging

import torch
from torch import Tensor

from src.model_module.memory.memory_bank import BaseMemoryBank

logger = logging.getLogger(__name__)


class PrototypeBank(BaseMemoryBank):
    """Mini-Batch K-Means Prototype 记忆库。

    预热阶段（step < warmup_steps）：收集样本，不更新 Prototype。
    预热完成后：使用 Mini-Batch K-Means 增量更新 Prototype 中心。

    Args:
        input_dim: 输入维度 D。
        n_prototypes: Prototype 数量 K。
        warmup_steps: 预热步数；预热完成时用缓存样本初始化 Prototype。
        sigma: RBF 核带宽（密度估计）。
        lr: Prototype 中心的在线更新步长（EMA 系数）。
        buffer_size: 预热缓冲区大小（超出时 FIFO 淘汰）。
    """

    def __init__(
        self,
        input_dim: int,
        n_prototypes: int = 50,
        warmup_steps: int = 200,
        sigma: float = 1.0,
        lr: float = 0.01,
        buffer_size: int = 1000,
    ) -> None:
        super().__init__(input_dim, n_prototypes, warmup_steps)
        self.sigma = sigma
        self.lr = lr

        # 预热缓冲区（收集样本用于初始化 K-Means）
        self._buffer: list[Tensor] = []
        self._buffer_size = buffer_size

        # Prototype 中心 (K, D) 和各 Prototype 的样本计数 (K,)
        self._centers: Tensor | None = None
        self._counts: Tensor | None = None  # (K,)

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------

    def add(self, u: Tensor) -> None:
        """增量添加新样本，更新 Prototype。

        Args:
            u: (D,) 输入向量。
        """
        u = u.detach().float()
        self._step += 1

        if not self.is_warmed_up:
            # 预热阶段：缓存样本
            self._buffer.append(u)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

            # 达到预热完成时初始化 Prototype
            if self._step == self.warmup_steps:
                self._init_prototypes()
        else:
            # 在线更新：找最近 Prototype，EMA 更新中心和计数
            self._online_update(u)

    def density(self, u: Tensor) -> Tensor:
        """RBF 加权 Prototype 激活密度。

        ρ(u) = Σ_k w_k · exp(-‖u - c_k‖² / (2σ²))
        其中 w_k = n_k / N_total 为 Prototype 权重。

        Args:
            u: (D,) 查询向量。

        Returns:
            标量密度值。
        """
        if self._centers is None:
            return torch.tensor(0.0)

        u = u.detach().float().to(self._centers.device)
        diff = u.unsqueeze(0) - self._centers          # (K, D)
        dist_sq = (diff * diff).sum(dim=-1)            # (K,)
        rbf = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))  # (K,)

        # 权重 = 各 Prototype 占比（_counts 移至与 rbf 同设备）
        counts = self._counts.to(self._centers.device).float()
        weights = counts / (counts.sum() + 1e-8)
        return (weights * rbf).sum()

    def add_batch(self, x: Tensor) -> None:
        """批量增量添加样本。

        Args:
            x: (B, D) 输入批次。
        """
        for u in x:
            self.add(u)

    def density_batch(self, x: Tensor) -> Tensor:
        """批量 RBF 加权密度估计。

        Args:
            x: (B, D) 查询批次。

        Returns:
            (B,) 每个样本的密度值。
        """
        if self._centers is None:
            return torch.zeros(x.shape[0], device=x.device)

        centers = self._centers.to(x.device)              # (K, D)
        counts = self._counts.to(x.device).float()        # (K,)
        weights = counts / (counts.sum() + 1e-8)          # (K,)

        x_f = x.detach().float()
        # cdist 避免 (B, K, D) 中间 tensor；对大批次尤关键
        dist_sq = torch.cdist(x_f, centers).pow(2)       # (B, K)
        rbf = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))  # (B, K)
        return (rbf * weights.unsqueeze(0)).sum(dim=-1)   # (B,)

    def reset(self) -> None:
        """清空 Prototype 记忆（任务切换时调用）。"""
        self._buffer.clear()
        self._centers = None
        self._counts = None
        self._step = 0
        logger.info("PrototypeBank 已重置")

    def prototype_centers(self) -> Tensor:
        """返回当前 Prototype 中心，(K, D)；未初始化时返回同设备的空 Tensor。"""
        if self._centers is None:
            device = self._buffer[-1].device if self._buffer else torch.device("cpu")
            return torch.zeros(0, self.input_dim, device=device)
        return self._centers

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _init_prototypes(self) -> None:
        """用预热缓冲区样本初始化 K-Means Prototype。

        使用 K-Means++ 初始化策略（通过 torch 实现）。
        """
        if not self._buffer:
            return

        data = torch.stack(self._buffer)   # (N_buf, D)
        k = min(self.n_prototypes, data.shape[0])

        # K-Means++ 初始化
        centers = self._kmeans_pp_init(data, k)

        # 运行少量迭代（10 步）
        for _ in range(10):
            dists = torch.cdist(data, centers)            # (N, K)
            assignments = dists.argmin(dim=1)             # (N,)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(k, dtype=torch.long, device=data.device)
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centers[j] = data[mask].mean(dim=0)
                    counts[j] = mask.sum()
                else:
                    new_centers[j] = centers[j]
            centers = new_centers
            self._counts = counts

        self._centers = centers
        logger.info(f"PrototypeBank 初始化完成：{k} 个 Prototype")

    def _kmeans_pp_init(self, data: Tensor, k: int) -> Tensor:
        """K-Means++ 初始化策略。"""
        n = data.shape[0]
        idx = torch.randint(n, (1,)).item()
        centers = [data[idx]]

        for _ in range(k - 1):
            stack = torch.stack(centers)               # (c, D)
            dists = torch.cdist(data, stack).min(dim=1).values  # (N,)
            probs = dists ** 2
            probs = probs / probs.sum()
            chosen = torch.multinomial(probs, 1).item()
            centers.append(data[chosen])

        return torch.stack(centers)                    # (k, D)

    def _online_update(self, u: Tensor) -> None:
        """找最近 Prototype，EMA 更新中心和计数。"""
        device = self._centers.device
        u = u.to(device)

        dists = torch.cdist(u.unsqueeze(0), self._centers).squeeze(0)  # (K,)
        nearest = int(dists.argmin().item())

        # EMA 更新中心：c_k ← (1 - lr) * c_k + lr * u
        self._centers[nearest] = (1.0 - self.lr) * self._centers[nearest] + self.lr * u
        self._counts[nearest] += 1
