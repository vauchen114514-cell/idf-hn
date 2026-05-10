"""IDF-HN 更新规则：含遗忘门的在线增量更新。

IDF-HN 更新流程（每步处理一个样本 u）：
    1. 以 u 为查询，执行 softmax 检索得到 ξ = retrieve(u)
    2. 计算冲突度 ΔE = delta_energy(ξ, M, u)
    3. 通过遗忘门得到 γ（EMA 平滑）
    4. Selective Write 决策：
       - conflict > write_threshold（新颖模式，跨任务）→ forget + store
       - conflict ≤ write_threshold（冗余模式，同任务）→ 跳过，保留记忆范数
    5. 触发 Dreaming（每 freq 步）
    6. 更新 Prototype 记忆库

forget_mode 消融维度：
    "input_dependent" : IDF-HN 完整模式（默认）
    "time_decay"      : 通过配置令 delta_gamma=0，gamma 固定为 gamma_0
    "static_density"  : 每步按各槽位 Prototype 密度逐槽衰减（不依赖输入冲突）
    "none"            : 通过配置令 gamma_0=0，无遗忘
"""
import logging
from dataclasses import dataclass
from typing import Optional

from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class UpdateStats:
    """单步更新的诊断统计信息。"""
    conflict: float
    gamma: float
    tau: float
    memory_size: int
    dream_count: int
    wrote: bool  # 本步是否实际写入记忆


def idf_update_step(
    u: Tensor,
    hopfield,                          # ModernHopfieldLayer
    forget_gate,                       # ForgetGate
    prototype_bank,                    # PrototypeBank / ExactDensityBank
    dreaming_scheduler,                # DreamingScheduler
    write_threshold: Optional[float] = None,  # None=始终写；负数=新颖度阈值
    label: int = -1,                   # 样本标签（用于回放存储）
    forget_mode: str = "input_dependent",     # 遗忘机制消融选项
) -> UpdateStats:
    """执行 IDF-HN 的单步在线更新。

    Args:
        u: (D,) 当前输入向量。
        hopfield: ModernHopfieldLayer 实例。
        forget_gate: ForgetGate 实例。
        prototype_bank: PrototypeBank / ExactDensityBank 实例。
        dreaming_scheduler: DreamingScheduler 实例。
        write_threshold: 选择性写入阈值。None=始终写入，推荐 -0.1。
        label: 样本标签。
        forget_mode: 遗忘机制模式（消融实验用）。

    Returns:
        UpdateStats 包含本步的诊断信息。
    """
    # Step 1：以 u 为查询执行检索，得到当前吸引子状态 ξ
    if hopfield.memory_size > 0:
        xi = hopfield.retrieve(u, n_steps=1)
    else:
        xi = u

    conflict_val = 0.0
    gamma_val = 0.0
    tau_val = 0.0
    should_write = True

    if forget_mode == "static_density":
        # 静态密度消融：按各记忆槽位的 Prototype 密度逐槽衰减
        # γ[j] = γ₀ + Δγ · (density(M[j]) / max_density)，不依赖当前输入冲突
        if hopfield.memory_size > 0 and prototype_bank.is_warmed_up:
            mem = hopfield.memory_matrix.detach()             # (N, D)
            densities = prototype_bank.density_batch(mem)     # (N,)
            d_max = float(densities.max().item()) + 1e-8
            norm_d = (densities / d_max).clamp(0.0, 1.0)
            gammas = forget_gate.gamma_0 + forget_gate.delta_gamma * norm_d
            hopfield.forget_per_slot(gammas)
        hopfield.store(u, label=label)

    elif forget_mode in ("input_dependent", "time_decay", "none"):
        # 统一路径：ForgetGate 计算 gamma
        # time_decay 和 none 分别通过配置令 delta_gamma=0 / gamma_0=0 实现
        gamma, conflict = forget_gate.compute_gamma(
            u=u,
            xi=xi,
            X=hopfield.memory_matrix,
        )
        conflict_val = float(conflict.item())
        gamma_val = float(gamma.item())
        tau_val = forget_gate.current_tau
        should_write = (write_threshold is None) or (conflict_val > write_threshold)

        if should_write:
            hopfield.forget(gamma_val)
            hopfield.store(u, label=label)
        # 冗余模式：跳过 forget+store，保护记忆范数

    else:
        raise ValueError(f"未知 forget_mode: {forget_mode!r}，"
                         f"可选: input_dependent / time_decay / static_density / none")

    # Step 5：Dreaming（周期性随机遗忘非相关记忆）
    dream_indices = dreaming_scheduler.step(
        memory_matrix=hopfield.memory_matrix,
        prototype_centers=prototype_bank.prototype_centers(),
    )
    dream_count = len(dream_indices)
    if dream_count > 0:
        hopfield.memory_matrix[dream_indices] *= (1.0 - dreaming_scheduler.dream_gamma)

    # Step 6：更新 Prototype 记忆库（始终执行）
    prototype_bank.add(u)

    stats = UpdateStats(
        conflict=conflict_val,
        gamma=gamma_val,
        tau=tau_val,
        memory_size=hopfield.memory_size,
        dream_count=dream_count,
        wrote=should_write,
    )
    logger.debug(
        f"update_step[{forget_mode}]: conflict={stats.conflict:.4f} gamma={stats.gamma:.4f} "
        f"tau={stats.tau:.4f} mem={stats.memory_size} wrote={stats.wrote}"
    )
    return stats
