"""FAISS 加速的近邻索引（可选加速）。

当记忆库规模较大（N > 10000）时，使用 FAISS 替代精确线性扫描，
将密度查询从 O(N·D) 降至 O(log N · D)。

小规模记忆（N ≤ 10000）直接用 PrototypeBank 的精确计算。
此模块作为 PrototypeBank 的可选加速层。
"""
import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# FAISS 为可选依赖；单卡 8GB 场景中 N_prototype=50 无需 ANN
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.info("faiss 未安装，FAISSIndex 不可用（小规模记忆无需 ANN）")


class FAISSIndex:
    """FAISS 近似最近邻索引，用于加速大规模 Prototype 检索。

    仅在 n_prototypes > 1000 且 FAISS 可用时建议启用。
    当前默认配置（n_prototypes=50）直接使用精确计算即可。

    Args:
        input_dim: 向量维度 D。
        n_neighbors: 检索最近邻数量。
    """

    def __init__(self, input_dim: int, n_neighbors: int = 5) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISSIndex 需要 faiss 包：uv add faiss-cpu"
            )
        self.input_dim = input_dim
        self.n_neighbors = n_neighbors
        self._index: Optional["faiss.Index"] = None  # type: ignore
        self._vectors: list[Tensor] = []

    def build(self, vectors: Tensor) -> None:
        """用给定向量构建 FAISS 索引。

        Args:
            vectors: (N, D) 向量矩阵。
        """
        import faiss  # type: ignore

        np_vectors = vectors.float().cpu().numpy()
        index = faiss.IndexFlatL2(self.input_dim)
        index.add(np_vectors)
        self._index = index
        self._vectors = [v for v in vectors]
        logger.info(f"FAISS 索引构建完成：{len(self._vectors)} 个向量")

    def search(self, query: Tensor, k: Optional[int] = None) -> tuple[Tensor, Tensor]:
        """查询最近邻。

        Args:
            query: (D,) 查询向量。
            k: 返回最近邻数量；默认使用初始化时的 n_neighbors。

        Returns:
            distances: (k,) L2 距离。
            indices: (k,) 索引。
        """
        if self._index is None:
            raise RuntimeError("FAISSIndex 未初始化，请先调用 build()")

        k = k or self.n_neighbors
        np_query = query.float().cpu().unsqueeze(0).numpy()
        distances, indices = self._index.search(np_query, k)
        return (
            torch.from_numpy(distances[0]),
            torch.from_numpy(indices[0]),
        )

    def reset(self) -> None:
        """清空索引。"""
        self._index = None
        self._vectors.clear()
