"""IDF-HN 遗忘门：根据冲突度动态计算遗忘率 γ。

遗忘率公式（Sigmoid 调制 + EMA 平滑）：
    γ_instant(u, t) = γ₀ + Δγ · σ(conflict(u(t), M) − τ)
    γ_ema(t)        = α · γ_ema(t-1) + (1-α) · γ_instant(t)

其中：
    γ₀    = 基础遗忘率（低冲突时的最小遗忘）
    Δγ    = 最大额外遗忘幅度
    σ     = Sigmoid 函数
    τ     = 冲突阈值（支持固定值或自适应百分位数）
    α     = EMA 平滑系数（0.9 时有效窗口 ≈ 10 步，消除单次冲突的瞬时峰值）
"""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model_module.forget_gate.conflict_detector import ConflictDetector

logger = logging.getLogger(__name__)


class ForgetGate:
    """输入依赖遗忘门。

    Args:
        gamma_0: 基础遗忘率（无冲突时的衰减下界）。
        delta_gamma: Sigmoid 调制的最大额外遗忘幅度。
        tau: 冲突阈值；若 adaptive_tau=True，此值仅用于预热阶段回退。
        beta: 逆温度（传递给 ConflictDetector）。
        adaptive_tau: 是否自动估计 τ（基于历史百分位数）。
        tau_percentile: 自适应 τ 的百分位数（默认 75）。
        history_size: 历史冲突窗口大小。
    """

    def __init__(
        self,
        gamma_0: float = 0.01,
        delta_gamma: float = 0.5,
        tau: float = 0.3,
        beta: float = 1.0,
        adaptive_tau: bool = True,
        tau_percentile: float = 75.0,
        history_size: int = 500,
        ema_alpha: float = 0.9,
    ) -> None:
        self.gamma_0 = gamma_0
        self.delta_gamma = delta_gamma
        self.tau_fixed = tau
        self.adaptive_tau = adaptive_tau
        self.ema_alpha = ema_alpha

        self.detector = ConflictDetector(
            beta=beta,
            history_size=history_size,
            tau_percentile=tau_percentile,
        )
        self._step = 0
        self._gamma_ema: float = gamma_0  # EMA 状态，初始化为基础遗忘率

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def compute_gamma(
        self,
        u: Tensor,
        xi: Tensor,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """计算遗忘率 γ（EMA 平滑后）和原始冲突度。

        Args:
            u: (D,) 当前输入向量。
            xi: (D,) 当前检索状态（query）。
            X: (N, D) 当前记忆矩阵。

        Returns:
            gamma: 标量 Tensor，EMA 平滑后的遗忘率 ∈ [γ₀, γ₀+Δγ]。
            conflict: 标量 Tensor，原始 ΔE 值。
        """
        conflict = self.detector.compute(u, xi, X)

        # 确定阈值 τ
        if self.adaptive_tau:
            tau = self.detector.adaptive_tau(fallback=self.tau_fixed)
        else:
            tau = self.tau_fixed

        # 瞬时遗忘率：γ_instant = γ₀ + Δγ · σ(conflict - τ)
        gate = torch.sigmoid(conflict - tau)
        gamma_instant = float((self.gamma_0 + self.delta_gamma * gate).item())

        # EMA 平滑：γ_ema(t) = α·γ_ema(t-1) + (1-α)·γ_instant(t)
        self._gamma_ema = self.ema_alpha * self._gamma_ema + (1.0 - self.ema_alpha) * gamma_instant
        gamma = torch.tensor(self._gamma_ema, dtype=torch.float32, device=u.device)

        self._step += 1
        logger.debug(
            f"step={self._step} conflict={float(conflict):.4f} "
            f"tau={tau:.4f} gamma_instant={gamma_instant:.4f} gamma_ema={self._gamma_ema:.4f}"
        )
        return gamma, conflict

    def compute_gamma_batch(self, delta_e: Tensor) -> Tensor:
        """批量计算遗忘率 γ（使用批次开始时的 τ 快照）。

        Args:
            delta_e: (B,) 批次内每个样本的 ΔE 值（≤ 0）。

        Returns:
            gamma_batch: (B,) 每个样本的遗忘率 ∈ [γ₀, γ₀+Δγ]。
        """
        # 先用当前历史计算 τ（批次级快照，避免批次内 τ 漂移）
        if self.adaptive_tau:
            tau = self.detector.adaptive_tau(fallback=self.tau_fixed)
        else:
            tau = self.tau_fixed

        # 批量追加历史（更新分布估计）
        self.detector._history.extend(delta_e.tolist())

        gate = torch.sigmoid(delta_e - tau)
        gamma_batch = self.gamma_0 + self.delta_gamma * gate

        self._step += delta_e.shape[0]
        return gamma_batch

    def reset(self) -> None:
        """重置历史记录和 EMA 状态（任务切换时调用）。"""
        self.detector.reset_history()
        self._step = 0
        self._gamma_ema = self.gamma_0  # EMA 重置为基础遗忘率，避免跨任务动量泄漏
        logger.info("ForgetGate 已重置")

    # ------------------------------------------------------------------
    # 诊断属性
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def current_tau(self) -> float:
        """返回当前有效 τ（自适应或固定）。"""
        if self.adaptive_tau:
            return self.detector.adaptive_tau(fallback=self.tau_fixed)
        return self.tau_fixed
