"""Hopfield 网络能量函数（数值稳定版）。

实现 Modern HN 的 log-sum-exp 能量函数，以及 IDF-HN 扩展的
带正则项能量函数。所有计算均使用 float32，通过 log-sum-exp trick
避免数值溢出。
"""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def lse(beta: float, scores: Tensor) -> Tensor:
    """数值稳定的 log-sum-exp 计算。

    log Σ_i exp(β·s_i) = max(β·s) + log Σ_i exp(β·s_i - max(β·s))

    Args:
        beta: 逆温度参数（控制检索锐度）。
        scores: (..., N) 形状的相似度分数。

    Returns:
        (...,) 形状的 log-sum-exp 值。
    """
    scaled = beta * scores
    # PyTorch 的 logsumexp 内置数值稳定实现
    return torch.logsumexp(scaled, dim=-1) / beta


def mhn_energy(xi: Tensor, X: Tensor, beta: float) -> Tensor:
    """Modern Hopfield Network 标准能量函数（Ramsauer et al. 2021）。

    E(ξ; X) = -lse(β, Xξ) + ½||ξ||² + C

    Args:
        xi: (D,) 或 (B, D) 查询向量（当前状态）。
        X: (N, D) 记忆矩阵（N 条记忆，每条 D 维）。
        beta: 逆温度参数。

    Returns:
        标量或 (B,) 形状的能量值。
    """
    # scores: (N,) 或 (B, N)
    scores = xi @ X.T if xi.dim() == 1 else xi @ X.T
    lse_val = lse(beta, scores)
    l2_term = 0.5 * (xi * xi).sum(dim=-1)
    return -lse_val + l2_term


def delta_energy(xi: Tensor, X: Tensor, u: Tensor, beta: float) -> Tensor:
    """计算新输入 u 加入记忆后的能量差 ΔE（冲突度量）。

    ΔE = E(ξ; X ∪ {u}) - E(ξ; X)
       = -lse(β, [Xξ, u·ξ]) + lse(β, Xξ)

    Args:
        xi: (D,) 查询向量（当前状态）。
        X: (N, D) 当前记忆矩阵。
        u: (D,) 新输入向量。
        beta: 逆温度参数。

    Returns:
        标量能量差（正值表示高冲突）。
    """
    # 与现有记忆的相似度分数
    scores_old = xi @ X.T                      # (N,)
    # 加入新输入后的分数
    new_score = (xi @ u).unsqueeze(0)          # (1,)
    scores_new = torch.cat([scores_old, new_score], dim=0)  # (N+1,)

    e_old = -lse(beta, scores_old) + 0.5 * (xi * xi).sum()
    e_new = -lse(beta, scores_new) + 0.5 * (xi * xi).sum()

    return e_new - e_old  # ΔE > 0 表示能量上升（高冲突）


def softmax_update(xi: Tensor, X: Tensor, beta: float) -> Tensor:
    """MHN 的 softmax 更新规则（单步检索）。

    ξ^{t+1} = Xᵀ · softmax(β · Xξ)

    Args:
        xi: (D,) 或 (B, D) 当前状态。
        X: (N, D) 记忆矩阵。
        beta: 逆温度参数。

    Returns:
        更新后的状态，形状与 xi 相同。
    """
    scores = xi @ X.T if xi.dim() == 1 else xi @ X.T  # (..., N)
    attention = F.softmax(beta * scores, dim=-1)        # (..., N)
    return attention @ X                                 # (..., D)
