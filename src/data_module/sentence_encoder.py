"""句子嵌入特征提取工具（sentence-transformers）。

支持 all-mpnet-base-v2（768 维）等模型，批量编码文本并 L2 归一化。
GPU 可用时自动使用 GPU 加速；结果为 float32 Tensor。
"""
import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 256


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    normalize: bool = True,
    show_progress: bool = True,
) -> Tensor:
    """将文本列表编码为句子嵌入矩阵。

    Args:
        texts: 待编码的字符串列表。
        model_name: sentence-transformers 模型名称。
        batch_size: 编码批次大小（GPU 可适当调大）。
        device: 指定设备（None = 自动选择 cuda/cpu）。
        normalize: 是否 L2 归一化（与 IDF-HN 能量框架对齐）。
        show_progress: 是否显示进度条。

    Returns:
        (N, D) float32 Tensor，D 由模型决定（all-mpnet-base-v2 = 768）。
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"加载 {model_name}（device={device}）...")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"编码 {len(texts)} 条文本（batch_size={batch_size}）...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    feat = torch.tensor(embeddings, dtype=torch.float32)
    logger.info(f"编码完成：{feat.shape}，norm range [{feat.norm(dim=1).min():.3f}, "
                f"{feat.norm(dim=1).max():.3f}]")
    return feat
