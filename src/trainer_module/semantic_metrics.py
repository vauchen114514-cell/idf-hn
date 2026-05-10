"""WikiFacts 专用语义指标：语义相似度与压缩比。

这两个指标仅适用于具有 Hopfield 检索层的模型：
    ClassicalHopfieldBaseline、ClippingHNBaseline、SparseMemoryBaseline、IDFHopfieldNetwork

对于无 Hopfield 层的模型（EWC、GSS、ER、DMHN），
compute_semantic_metrics() 返回 None。

语义相似度（SemanticSimilarity）：
    对每个测试样本 x，计算检索向量 xi = hopfield.retrieve(x) 与 x 的余弦相似度。
    mean_sim ∈ [-1, 1]，越高说明 Hopfield 记忆越好地保留了语义信息。

压缩比（CompressionRatio）：
    N_stored / N_total_seen，即记忆库存储的样本数与累计见过的训练样本总数之比。
    反映模型的记忆压缩效率：越高说明 buffer 利用率越高（越接近 1 表示几乎不压缩，
    接近 0 表示高度压缩）。
"""
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class SemanticMetrics:
    """语义指标结果容器。"""
    semantic_similarity: float   # 平均余弦相似度（[-1, 1]）
    compression_ratio: float     # 记忆压缩比（(0, 1]）
    n_stored: int                # 当前记忆库存储数量
    n_total_seen: int            # 累计训练样本总数


def _has_hopfield(model: torch.nn.Module) -> bool:
    """检查模型是否具有 Hopfield 检索层。"""
    return hasattr(model, "hopfield") and hasattr(model.hopfield, "retrieve")


def compute_semantic_similarity(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    n_steps: int = 1,
    max_samples: int = 500,
) -> float | None:
    """计算 Hopfield 检索向量与原始输入的平均余弦相似度。

    仅适用于有 Hopfield 层的模型。

    Args:
        model: 评估目标模型。
        val_loader: 验证集 DataLoader。
        device: 计算设备。
        n_steps: Hopfield 检索迭代步数。
        max_samples: 最大采样数（避免全集太慢）。

    Returns:
        平均余弦相似度（float），或 None（模型无 Hopfield 层）。
    """
    if not _has_hopfield(model):
        return None
    if model.hopfield.memory_size == 0:
        logger.warning("compute_semantic_similarity: Hopfield 记忆为空，返回 None")
        return None

    model.eval()
    sims: list[float] = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            for i in range(x.shape[0]):
                u = x[i]
                xi = model.hopfield.retrieve(u, n_steps=n_steps)
                sim = F.cosine_similarity(xi.unsqueeze(0), u.unsqueeze(0)).item()
                sims.append(sim)
                if len(sims) >= max_samples:
                    break
            if len(sims) >= max_samples:
                break

    return float(sum(sims) / len(sims)) if sims else None


def compute_compression_ratio(
    model: torch.nn.Module,
) -> float | None:
    """计算记忆压缩比：N_stored / N_total_seen。

    N_stored   = 当前能量缓冲区已存储的记忆数
    N_total_seen = 能量缓冲区累计见过的样本总数

    Args:
        model: 评估目标模型。

    Returns:
        压缩比 ∈ (0, 1]，或 None（模型无 Hopfield 层）。
    """
    if not _has_hopfield(model):
        return None

    hopfield = model.hopfield
    n_stored = hopfield.memory_size
    n_total_seen = getattr(hopfield, "_n_total_seen", 0)

    if n_total_seen == 0:
        return None

    return float(n_stored) / float(n_total_seen)


def compute_semantic_metrics(
    model: torch.nn.Module,
    val_loaders: list[DataLoader],
    device: torch.device,
    n_steps: int = 1,
    max_samples_per_task: int = 300,
    n_total_training: int = 0,
) -> SemanticMetrics | None:
    """计算全部任务的平均语义指标。

    Args:
        model: 评估目标模型。
        val_loaders: 各任务验证集 DataLoader 列表。
        device: 计算设备。
        n_steps: Hopfield 检索迭代步数。
        max_samples_per_task: 每个任务最多采样数。
        n_total_training: 训练总样本数（trainer 传入，用于正确计算压缩比）。
            若为 0，则退化为模型内部计数器 _n_total_seen。

    Returns:
        SemanticMetrics，或 None（模型无 Hopfield 层）。
    """
    if not _has_hopfield(model):
        logger.info("模型无 Hopfield 层，跳过语义指标计算")
        return None

    all_sims: list[float] = []
    for loader in val_loaders:
        sim = compute_semantic_similarity(
            model, loader, device,
            n_steps=n_steps,
            max_samples=max_samples_per_task,
        )
        if sim is not None:
            all_sims.append(sim)

    if not all_sims:
        return None

    mean_sim = float(sum(all_sims) / len(all_sims))

    hopfield = model.hopfield
    n_stored = hopfield.memory_size
    # 优先使用 trainer 传入的训练总样本数（更准确，不受 write_threshold 影响）
    n_total = n_total_training if n_total_training > 0 else getattr(hopfield, "_n_total_seen", 0)
    comp_ratio = float(n_stored) / float(n_total) if n_total > 0 else 0.0

    return SemanticMetrics(
        semantic_similarity=mean_sim,
        compression_ratio=comp_ratio,
        n_stored=n_stored,
        n_total_seen=n_total,
    )
