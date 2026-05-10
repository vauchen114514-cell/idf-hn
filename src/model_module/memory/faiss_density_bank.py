"""基于 FAISS ANN 的密度估计记忆库（效率消融用）。

与 PrototypeBank（O(K) Mini-Batch K-Means）和 ExactDensityBank（O(N) 精确 RBF）的对比：
  - PrototypeBank  : O(K) 常数，K=50；N 无关；小规模最优
  - ExactDensityBank: O(N) 线性；N=1K 时约 0.05ms，N=10K 约 0.5ms
  - FaissDensityBank: O(log N)（FAISS IVF）；N>10K 时加速显著

本模块实现 BaseMemoryBank 接口，密度估计使用：
  ρ(u) = mean(exp(-dist²_k / 2σ²))，其中 dist_k 为 u 到 k 个近邻的 L2 距离
"""
import logging

import torch
from torch import Tensor

from src.model_module.memory.memory_bank import BaseMemoryBank

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss 未安装，FaissDensityBank 不可用，请运行：uv add faiss-cpu")


class FaissDensityBank(BaseMemoryBank):
    """FAISS IVF 加速近邻密度估计记忆库。

    维护一个 FAISS 索引（IndexFlatL2 for exact，IndexIVFFlat for approx）。
    每隔 rebuild_interval 步重建索引以包含最新样本。
    密度查询使用 k 近邻 RBF：ρ(u) = mean_k exp(-d_k² / 2σ²)。

    Args:
        input_dim: 向量维度 D。
        n_prototypes: 近邻数量 k（等效于 PrototypeBank 的 K）。
        warmup_steps: 预热步数（<warmup_steps 时不建立索引）。
        sigma: RBF 核带宽。
        rebuild_interval: 每隔多少步重建 FAISS 索引（避免每步重建的开销）。
        use_ivf: True=使用 IndexIVFFlat（近似，大规模快）；
                 False=使用 IndexFlatL2（精确）。
        n_lists: IVF 聚类数（仅 use_ivf=True 时有效）。
    """

    def __init__(
        self,
        input_dim: int,
        n_prototypes: int = 50,
        warmup_steps: int = 200,
        sigma: float = 1.0,
        rebuild_interval: int = 200,
        use_ivf: bool = False,
        n_lists: int = 100,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError("FaissDensityBank 需要 faiss：uv add faiss-cpu")
        super().__init__(input_dim, n_prototypes, warmup_steps)
        self.sigma = sigma
        self.rebuild_interval = rebuild_interval
        self.use_ivf = use_ivf
        self.n_lists = n_lists

        self._vectors: list[np.ndarray] = []
        self._index: "faiss.Index | None" = None  # type: ignore

    # ------------------------------------------------------------------
    # BaseMemoryBank 接口
    # ------------------------------------------------------------------

    def add(self, u: Tensor) -> None:
        """增量添加新样本，定期重建 FAISS 索引。"""
        self._step += 1
        vec = u.detach().cpu().float().numpy()
        self._vectors.append(vec)

        # 达到预热完成 或 每隔 rebuild_interval 步 重建索引
        if self._step == self.warmup_steps or (
            self._step > self.warmup_steps and self._step % self.rebuild_interval == 0
        ):
            self._rebuild_index()

    def density(self, u: Tensor) -> Tensor:
        """RBF 加权 k 近邻密度估计。"""
        if self._index is None or len(self._vectors) == 0:
            return torch.tensor(0.0)

        k = min(self.n_prototypes, len(self._vectors))
        query = u.detach().cpu().float().numpy().reshape(1, -1)
        distances, _ = self._index.search(query, k)   # (1, k) L2²
        dist_sq = torch.from_numpy(distances[0]).float()
        rbf = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return rbf.mean()

    def density_batch(self, x: Tensor) -> Tensor:
        """批量密度估计（B 个查询并行搜索）。"""
        if self._index is None or len(self._vectors) == 0:
            return torch.zeros(x.shape[0])

        k = min(self.n_prototypes, len(self._vectors))
        queries = x.detach().cpu().float().numpy()
        distances, _ = self._index.search(queries, k)  # (B, k) L2²
        dist_sq = torch.from_numpy(distances).float()
        rbf = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return rbf.mean(dim=-1)

    def add_batch(self, x: Tensor) -> None:
        """批量添加样本。"""
        for u in x:
            self.add(u)

    def reset(self) -> None:
        """清空索引和向量缓存。"""
        self._vectors.clear()
        self._index = None
        self._step = 0
        logger.info("FaissDensityBank 已重置")

    def prototype_centers(self) -> Tensor:
        """返回最近 K 个向量作为近似 Prototype 中心（兼容接口）。"""
        if not self._vectors:
            return torch.zeros(0, self.input_dim)
        recent = self._vectors[-self.n_prototypes:]
        return torch.from_numpy(np.stack(recent)).float()

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """重建 FAISS 索引（包含全部已添加向量）。"""
        if not self._vectors:
            return
        import numpy as np  # noqa: F811
        data = np.stack(self._vectors).astype(np.float32)
        n = data.shape[0]

        if self.use_ivf and n >= self.n_lists * 10:
            # IndexIVFFlat：近似，大规模（N > 10K）时加速显著
            quantizer = faiss.IndexFlatL2(self.input_dim)
            n_lists = min(self.n_lists, n // 10)
            index = faiss.IndexIVFFlat(quantizer, self.input_dim, n_lists, faiss.METRIC_L2)
            index.train(data)
            index.add(data)
            index.nprobe = max(1, n_lists // 10)
        else:
            # IndexFlatL2：精确，适合 N ≤ 50K
            index = faiss.IndexFlatL2(self.input_dim)
            index.add(data)

        self._index = index
        logger.debug(f"FaissDensityBank 索引重建：{n} 个向量，use_ivf={self.use_ivf}")
