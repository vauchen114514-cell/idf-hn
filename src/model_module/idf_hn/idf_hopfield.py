"""IDF-HN 主模型：Input-Dependent Selective Forgetting Hopfield Network。

将 ModernHopfieldLayer、ForgetGate、PrototypeBank 和 DreamingScheduler
组合为完整的 IDF-HN 模型，并在顶层加入线性分类头。

能量函数：
    E_IDF(ξ; M) = -lse(β, M^T ξ) + ½‖ξ‖² + λ·Ω_eff(M, u)

前向传播：
    1. 用 u 检索 Hopfield 记忆，得到 ξ（吸引子特征）
    2. 在线执行 IDF 更新步（存储+遗忘）
    3. 用 ξ 做线性分类
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.dreaming.dreaming_scheduler import DreamingScheduler
from src.model_module.forget_gate.forget_gate import ForgetGate
from src.model_module.hopfield.mhn_layer import ModernHopfieldLayer
from src.model_module.idf_hn.update_rule import UpdateStats, idf_update_step
from src.model_module.memory.exact_density_bank import ExactDensityBank
from src.model_module.memory.prototype_bank import PrototypeBank

logger = logging.getLogger(__name__)


@register_model("idf_hn")
class IDFHopfieldNetwork(nn.Module):
    """IDF-HN：输入依赖选择性遗忘 Hopfield 网络。

    Args:
        cfg: Hydra/OmegaConf 配置节点，需包含 model.* 字段。
        input_dim: 输入特征维度 D。
        n_classes: 分类头输出类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()

        m = cfg.model  # 简写

        # --- Hopfield 核心层 ---
        self.hopfield = ModernHopfieldLayer(
            input_dim=input_dim,
            beta=m.beta,
            max_memories=m.memory_size,
            eviction_policy=getattr(m, "eviction_policy", "fifo"),
            dual_buffer=getattr(m, "dual_buffer", True),
        )

        # --- 遗忘门 ---
        fg = m.forget_gate
        self.forget_gate = ForgetGate(
            gamma_0=fg.gamma_0,
            delta_gamma=fg.delta_gamma,
            tau=fg.tau,
            beta=m.beta,
            adaptive_tau=fg.adaptive_tau,
            tau_percentile=fg.tau_percentile,
            ema_alpha=getattr(fg, "ema_alpha", 0.9),
        )

        # --- Prototype / Exact 记忆库（效率消融可切换）---
        pb = m.memory_bank
        bank_type = getattr(pb, "type", "prototype")
        if bank_type == "exact":
            self.prototype_bank = ExactDensityBank(
                input_dim=input_dim,
                sigma=getattr(pb, "sigma", 1.0),
                max_size=m.memory_size,
            )
        else:  # prototype（默认）
            self.prototype_bank = PrototypeBank(
                input_dim=input_dim,
                n_prototypes=pb.n_prototypes,
                warmup_steps=pb.warmup_steps,
                sigma=getattr(pb, "sigma", 1.0),
            )

        # --- Dreaming 调度器 ---
        dr = m.dreaming
        self.dreaming_scheduler = DreamingScheduler(
            freq=dr.freq,
            n_dream_samples=dr.n_dream_samples,
            dream_gamma=getattr(dr, "dream_gamma", 0.3),
            sim_threshold=getattr(dr, "semantic_threshold", 0.5),
            enabled=dr.enabled,
        )

        self.lambda_omega = m.lambda_omega
        # write_threshold: None=始终写入；负数=仅写新颖模式（conflict > threshold）
        self.write_threshold = getattr(m, "write_threshold", None)
        # forget_mode: 遗忘机制消融选项
        self.forget_mode: str = getattr(m, "forget_mode", "input_dependent")

        # --- 分类头 ---
        self.classifier = nn.Linear(input_dim, n_classes)

        self._last_stats: UpdateStats | None = None
        self.diagnostics_enabled: bool = getattr(m, "diagnostics_enabled", False)
        self._diagnostics_rows: list[dict[str, float | int]] = []
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> tuple[Tensor, Tensor] | None:
        """从记忆中采样 n 条 (特征, 标签) 用于回放训练。"""
        return self.hopfield.sample_replay(n, energy_priority=energy_priority)

    def forward(self, x: Tensor, update: bool = True, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """前向传播。

        Args:
            x: (B, D) 输入批次。
            update: 是否在每个样本上执行 IDF 在线更新（训练时为 True）。

        Returns:
            logits: (B, C) 分类 logits。
            energy: (B,) 每个样本的 Hopfield 能量值。
        """
        logits_list = []
        energies = []

        for i in range(x.shape[0]):
            u = x[i]  # (D,)

            if update:
                label = int(labels[i].item()) if labels is not None else -1
                self._last_stats = idf_update_step(
                    u=u,
                    hopfield=self.hopfield,
                    forget_gate=self.forget_gate,
                    prototype_bank=self.prototype_bank,
                    dreaming_scheduler=self.dreaming_scheduler,
                    write_threshold=self.write_threshold,
                    label=label,
                    forget_mode=self.forget_mode,
                )
                if self.diagnostics_enabled and self._last_stats is not None:
                    mem = self.hopfield.memory_matrix
                    mean_norm = float(mem.norm(dim=1).mean().item()) if mem.numel() else 0.0
                    min_norm = float(mem.norm(dim=1).min().item()) if mem.numel() else 0.0
                    replay_norm = 0.0
                    if self.hopfield._replay_n_stored > 0:
                        replay_norm = float(
                            self.hopfield._replay_buf[: self.hopfield._replay_n_stored]
                            .norm(dim=1)
                            .mean()
                            .item()
                        )
                    self._diagnostics_rows.append(
                        {
                            "step": self._global_step,
                            "label": label,
                            "conflict": self._last_stats.conflict,
                            "gamma": self._last_stats.gamma,
                            "tau": self._last_stats.tau,
                            "memory_size": self._last_stats.memory_size,
                            "wrote": int(self._last_stats.wrote),
                            "energy_mean_norm": mean_norm,
                            "energy_min_norm": min_norm,
                            "replay_mean_norm": replay_norm,
                        }
                    )
                self._global_step += 1

            # 用更新后的记忆检索 ξ（仅用于能量正则，不参与分类）
            # 分类走原始输入 u：避免记忆范数塌陷（20 任务 × 5000 次 forget → xi ≈ 0）
            xi = self.hopfield.retrieve(u, n_steps=1)
            logits_list.append(self.classifier(u))

            # 能量 + Ω_eff 正则
            e = self.hopfield.energy(xi)
            omega = self.prototype_bank.density(u) * self.lambda_omega
            energies.append(e + omega)

        logits = torch.stack(logits_list, dim=0)   # (B, C)
        energy = torch.stack(energies, dim=0)       # (B,)
        return logits, energy

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时重置遗忘门和 Prototype 历史（不清空记忆矩阵）。

        IDF-HN 设计为跨任务不清空记忆，只重置自适应 τ 历史。

        Args:
            task_id: 新任务编号（用于日志）。
        """
        self.forget_gate.reset()
        self.dreaming_scheduler.reset()
        logger.info(f"IDFHopfieldNetwork 已切换至 Task {task_id}")

    def save_diagnostics(self, path: str | Path) -> None:
        """将在线更新诊断信息保存为 CSV。"""
        if not self._diagnostics_rows:
            return
        import csv

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._diagnostics_rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._diagnostics_rows)
        logger.info(f"IDF-HN diagnostics saved to {out_path}")

    # ------------------------------------------------------------------
    # 诊断属性
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return self.hopfield.memory_size

    @property
    def last_stats(self) -> UpdateStats | None:
        return self._last_stats
