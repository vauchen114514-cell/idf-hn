"""DMHN（Dynamic Manifold Hopfield Network）基线。

Li et al., "Dynamic Manifold Hopfield Networks for Context-Dependent
Associative Memory", arXiv:2506.01303, 2025.

核心思想：将 cue（输入信号）同时作为初始状态和上下文调制信号，
使权重矩阵和偏置变为 cue 依赖的动态量，从而重塑吸引子流形几何。

连续时间动力学（Euler 离散化，T=10 步）：
    ẋ = -diag(τ) x + [WS + WD(u)] Φ(x) + [IS + ID(u)]

cue 依赖分量（低秩参数化）：
    WD(u) = z^T z,   z = u W_wcue  ∈ R^r  →  WD ∈ R^{D×D} (rank-r PSD)
    ID(u) = u W_icue ∈ R^D

WS 约束为对称矩阵（保证能量单调递减）。
Φ = tanh。

注意：DMHN 是梯度下降训练的持续学习基线，参数不重置，
展示标准神经网络在持续学习下的遗忘行为（无显式遗忘保护）。
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model_module import register_model

logger = logging.getLogger(__name__)


@register_model("dmhn")
class DMHNBaseline(nn.Module):
    """Dynamic Manifold Hopfield Network 基线（分类适配版）。

    输入 u 同时作为：
        - 初始状态 x(0) = u
        - 上下文信号，调制 WD(u) 和 ID(u)

    运行 n_steps 步 Euler 积分后，用最终状态做分类。

    Args:
        cfg: 配置对象，需包含 model.* 字段。
        input_dim: 输入特征维度 D（= 网络神经元数 N）。
        n_classes: 分类头输出类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model
        N = input_dim
        self.N = N
        self.n_steps: int = getattr(m, "dmhn_n_steps", 10)
        self.dt: float = getattr(m, "dmhn_dt", 0.1)

        # --- 静态分量 ---
        # WS：对称权重矩阵（存储下三角，前向时对称化）
        self.WS_raw = nn.Parameter(torch.zeros(N, N))
        self.IS = nn.Parameter(torch.zeros(N))

        # --- 动态分量（cue 依赖）---
        # WD(u) = (u W_wcue)^T (u W_wcue)，低秩 r << N
        # 论文原文：rank-1；实际取 r = max(1, N//8) 保留更多容量
        rank: int = getattr(m, "dmhn_rank", max(1, N // 8))
        self.rank = rank
        # W_wcue：将 cue u (N,) 投影到 (rank,)
        self.W_wcue = nn.Parameter(
            torch.randn(N, rank) * (N ** -0.5)
        )
        # W_icue：将 cue u (N,) 映射到偏置调制 (N,)
        self.W_icue = nn.Parameter(
            torch.zeros(N, N)
        )

        # --- 泄漏向量 ---
        # τ_i ≥ 0（用 softplus 保证非负）
        self._tau_raw = nn.Parameter(torch.zeros(N))

        # --- 分类头 ---
        self.classifier = nn.Linear(N, n_classes)

        self._symmetrize_WS()

    # ------------------------------------------------------------------
    # 核心动力学
    # ------------------------------------------------------------------

    @property
    def WS(self) -> Tensor:
        """对称化的静态权重矩阵。"""
        return (self.WS_raw + self.WS_raw.T) / 2.0

    @property
    def tau(self) -> Tensor:
        """非负泄漏向量 τ（softplus 保证 τ_i > 0）。"""
        return F.softplus(self._tau_raw)

    def _WD(self, u: Tensor) -> Tensor:
        """计算 cue 依赖权重 WD(u)。

        WD(u) = z z^T，z = u W_wcue ∈ R^{r}
        → WD ∈ R^{r×r}（rank-1 PSD），通过左右投影施加于 Φ(x)。

        实现策略（避免显式构造 N×N 矩阵）：
            [WS + WD(u)] Φ(x) = WS Φ(x) + z (z^T (W_wcue^T Φ(x)))

        此方法返回 z ∈ R^{B, rank}，供 forward 使用。
        """
        # u: (B, N) → z: (B, rank)
        return u @ self.W_wcue  # (B, rank)

    def _ID(self, u: Tensor) -> Tensor:
        """cue 依赖偏置 ID(u) = u W_icue ∈ R^{B, N}。"""
        return u @ self.W_icue  # (B, N)

    def _dynamics_step(self, x: Tensor, z: Tensor, ID_u: Tensor) -> Tensor:
        """单步 Euler 积分。

        Δx = -diag(τ) x + [WS Φ(x) + z(z^T W_wcue^T Φ(x))] + IS + ID(u)

        Args:
            x: (B, N) 当前状态。
            z: (B, rank) cue 投影向量。
            ID_u: (B, N) cue 依赖偏置。

        Returns:
            x_next: (B, N) 更新后状态。
        """
        phi_x = torch.tanh(x)                          # (B, N)
        WS = self.WS                                    # (N, N)
        tau = self.tau                                  # (N,)

        # 静态权重贡献：WS Φ(x)
        static = phi_x @ WS.T                          # (B, N)

        # 动态权重贡献：WD(u) Φ(x) = z (z^T W_wcue^T Φ(x))
        # W_wcue^T Φ(x) → (rank, N)^T (B, N)^T = (B, rank)
        proj = phi_x @ self.W_wcue                     # (B, rank)
        # z * proj 按元素，再投影回 N 维
        dyn_coeff = (z * proj).sum(dim=-1, keepdim=True)    # (B, 1)，rank-1 近似
        dynamic = dyn_coeff * (z @ self.W_wcue.T)          # (B, N)

        # 完整 RHS
        rhs = -tau * x + static + dynamic + self.IS + ID_u  # (B, N)
        return x + self.dt * rhs

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, update: bool = True, **kwargs) -> tuple[Tensor, Tensor]:
        """前向传播：运行 n_steps 步动力学后分类。

        Args:
            x: (B, D) 输入特征（= cue u）。
            update: 保持接口一致（DMHN 无在线记忆写入）。

        Returns:
            logits: (B, C)。
            energy: (B,) 最终状态的近似能量（监控用）。
        """
        u = x                                           # cue = input
        state = x.clone()                              # x(0) = u

        # 预计算 cue 依赖分量（每个 episode 固定）
        z = self._WD(u)                                # (B, rank)
        ID_u = self._ID(u)                             # (B, N)

        # Euler 积分 n_steps 步
        for _ in range(self.n_steps):
            state = self._dynamics_step(state, z, ID_u)

        logits = self.classifier(state)

        # 近似能量（对称能量下界）
        phi = torch.tanh(state)
        WS = self.WS
        energy = (
            -0.5 * (phi * (phi @ WS.T)).sum(dim=-1)
            - (phi * (self.IS + ID_u)).sum(dim=-1)
        )
        return logits, energy

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时不重置参数（作为无遗忘保护基线）。"""
        logger.info(f"DMHN: 切换至 Task {task_id}（参数不重置，展示自然遗忘）")

    # ------------------------------------------------------------------
    # 训练辅助
    # ------------------------------------------------------------------

    def _symmetrize_WS(self) -> None:
        """强制 WS_raw 满足对称约束（每次梯度更新后调用）。"""
        with torch.no_grad():
            self.WS_raw.data = (self.WS_raw.data + self.WS_raw.data.T) / 2.0

    def enforce_constraints(self) -> None:
        """梯度更新后调用：强制对称性并约束谱范数。

        在 ContinualTrainer 的 optimizer.step() 后调用此方法。
        """
        self._symmetrize_WS()
        # 约束 WS 谱范数 ≤ 1（防止动力学发散）
        with torch.no_grad():
            sv = torch.linalg.svdvals(self.WS_raw)
            max_sv = sv.max()
            if max_sv > 1.0:
                self.WS_raw.data /= max_sv
