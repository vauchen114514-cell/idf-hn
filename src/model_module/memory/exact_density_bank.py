"""精确 O(N) 密度库：直接在所有存储向量上计算 RBF 密度。

效率消融（ablation）用途：
    - exact: 本类（O(N) per query，无 Prototype 压缩）
    - prototype: PrototypeBank（O(K) per query，K-Means 压缩）
    - prototype_faiss: FAISSPrototypeBank（O(log K) for large K）

在 N≤1000 的实验规模下，三者精度应一致，
差异主要体现在每步密度计算的 FLOPs。
"""
import logging

import torch
from torch import Tensor

from src.model_module.memory.memory_bank import BaseMemoryBank

logger = logging.getLogger(__name__)


class ExactDensityBank(BaseMemoryBank):
    """精确密度库：在所有已存储向量上做 RBF 核密度估计（O(N)）。

    与 PrototypeBank 不同，本类不做 K-Means 压缩，
    直接对 _buffer 中的全量向量计算密度。
    buffer 满时使用 Reservoir Sampling 淘汰旧向量。

    Args:
        input_dim: 输入维度 D。
        sigma: RBF 核带宽。
        max_size: 最大保留向量数（对应 memory_size）。
    """

    def __init__(
        self,
        input_dim: int,
        sigma: float = 1.0,
        max_size: int = 1000,
        # 兼容 BaseMemoryBank 签名（n_prototypes/warmup_steps 不实际使用）
        n_prototypes: int = 0,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__(input_dim, n_prototypes=max_size, warmup_steps=warmup_steps)
        self.sigma = sigma
        self.max_size = max_size
        self._buf: list[Tensor] = []
        self._n_seen: int = 0

    @property
    def is_warmed_up(self) -> bool:
        """只要有至少一个向量就视为准备好（无预热概念）。"""
        return len(self._buf) > 0

    def add(self, u: Tensor) -> None:
        """Reservoir Sampling 加入新向量。"""
        u = u.detach().float()
        self._step += 1
        self._n_seen += 1

        if len(self._buf) < self.max_size:
            self._buf.append(u)
        else:
            j = int(torch.randint(0, self._n_seen, (1,)).item())
            if j < self.max_size:
                self._buf[j] = u

    def density(self, u: Tensor) -> Tensor:
        """RBF 核密度：ρ(u) = mean_j exp(-‖u - b_j‖² / (2σ²))。"""
        if not self._buf:
            return torch.tensor(0.0)

        B = torch.stack(self._buf)          # (N, D)
        u_f = u.detach().float().to(B.device)
        diff = u_f.unsqueeze(0) - B         # (N, D)
        dist_sq = (diff * diff).sum(dim=-1)  # (N,)
        rbf = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return rbf.mean()

    def density_batch(self, x: Tensor) -> Tensor:
        """批量 RBF 密度估计。

        Args:
            x: (B, D) 查询批次。

        Returns:
            (B,) 每个样本的密度值。
        """
        if not self._buf:
            return torch.zeros(x.shape[0], device=x.device)

        B = torch.stack(self._buf).to(x.device)   # (N, D)
        x_f = x.detach().float()
        dist_sq = torch.cdist(x_f, B).pow(2)      # (B, N)
        rbf = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return rbf.mean(dim=-1)                    # (B,)

    def reset(self) -> None:
        """清空 buffer。"""
        self._buf.clear()
        self._n_seen = 0
        self._step = 0
        logger.info("ExactDensityBank 已重置")

    def prototype_centers(self) -> Tensor:
        """返回所有存储向量作为"原型中心"（供 Dreaming 语义过滤）。"""
        if not self._buf:
            dev = torch.device("cpu")
            return torch.zeros(0, self.input_dim, device=dev)
        return torch.stack(self._buf)
