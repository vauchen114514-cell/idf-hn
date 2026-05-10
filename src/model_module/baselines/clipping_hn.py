"""Classical HN + Clipping 基线（Marinari et al., 2026）。

实现 Marinari 2026 Dreaming 论文中的两个核心机制：
  1. Memory norm clipping：每步写入后将记忆范数裁剪至 max_norm，
     防止少数记忆支配能量景观（"Clipping"）
  2. Periodic dreaming：每 dream_freq 步对随机子集施加衰减，
     模拟 Marinari 的随机 unlearning（"Dreaming"）

与现有基线的差异：
  vs classical_hn   : 新增 norm clipping + periodic dreaming
  vs sparse_memory  : 无常数 gamma，无 Replay 缓冲区
  vs IDF-HN         : 无输入依赖遗忘门，无冲突检测

Dreaming 使用独立 torch.Generator，避免污染全局随机状态
（教训来自 findings.md 第三节：全局 RNG 污染导致 BWT 噪声结果）。
"""
import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.hopfield.mhn_layer import ModernHopfieldLayer

logger = logging.getLogger(__name__)


@register_model("clipping_hn")
class ClippingHNBaseline(nn.Module):
    """Classical HN + Clipping（Marinari 2026 dreaming 机制）。

    Args:
        cfg: 配置对象，需包含 model.memory_size、model.beta。
             可选：model.max_norm（默认 1.0）、model.dream_freq（默认 500）、
                   model.dream_gamma（默认 0.5）。
        input_dim: 输入特征维度 D。
        n_classes: 分类头类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model

        self.hopfield = ModernHopfieldLayer(
            input_dim=input_dim,
            beta=m.beta,
            max_memories=m.memory_size,
            eviction_policy="fifo",   # 无 ForgetGate 指导，使用 FIFO
        )
        self.classifier = nn.Linear(input_dim, n_classes)

        self.max_norm: float = float(getattr(m, "max_norm", 1.0))
        self.dream_freq: int = int(getattr(m, "dream_freq", 500))
        self.dream_gamma: float = float(getattr(m, "dream_gamma", 0.5))
        self._step: int = 0

        # 独立 RNG：避免污染全局随机状态（参考 findings.md Section 3）
        self._rng = torch.Generator()
        self._rng.manual_seed(42)

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        update: bool = True,
        labels: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """前向传播。

        分类走原始输入 u（与 IDF-HN 对齐，避免记忆塌陷导致 AA=0.2）。

        Args:
            x: (B, D) 输入批次。
            update: 训练时为 True，执行写入 + clipping + dreaming。
            labels: (B,) 标签，update=True 时必须提供。

        Returns:
            logits: (B, C) 分类 logits。
            energy: (B,) 标量能量占位（全零，与 ER/SparseMemory 保持一致）。
        """
        logits = self.classifier(x)
        dummy_energy = torch.zeros(x.shape[0], device=x.device)

        if update:
            for i in range(x.shape[0]):
                u = x[i].detach()
                label = int(labels[i].item()) if labels is not None else -1

                self.hopfield.store(u, label=label)
                self._clip_norms()
                self._step += 1

                if self._step % self.dream_freq == 0:
                    self._dream()

        return logits, dummy_energy

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时不清空记忆（跨任务累积）。"""
        logger.info(
            f"ClippingHN: 切换至 Task {task_id}"
            f"（记忆保留，n_stored={self.hopfield.memory_size}）"
        )

    # ------------------------------------------------------------------
    # 核心机制
    # ------------------------------------------------------------------

    def _clip_norms(self) -> None:
        """裁剪能量缓冲区中所有记忆的范数至 max_norm（原地操作）。

        仅裁剪已写入的 n_stored 行，保持缓冲区其余部分为零。
        scale = min(1, max_norm / norm) → 超过 max_norm 的向量被缩短，
        未超过的保持不变。
        """
        n = self.hopfield.memory_size
        if n == 0:
            return
        mem = self.hopfield._mem[:n]
        norms = mem.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (self.max_norm / norms).clamp(max=1.0)
        self.hopfield._mem[:n].mul_(scale)

    def _dream(self) -> None:
        """Marinari 随机 dreaming：对约 10% 的记忆施加衰减。

        使用独立 Generator 确保 dreaming 不影响主训练循环的随机状态。
        """
        n = self.hopfield.memory_size
        if n == 0:
            return
        n_dream = max(1, n // 10)
        device = self.hopfield._mem.device
        perm = torch.randperm(n, generator=self._rng)  # CPU generator，在 CPU 上生成
        indices = perm[:n_dream].to(device)
        self.hopfield._mem[indices].mul_(1.0 - self.dream_gamma)
        logger.debug(
            f"dreaming: step={self._step}, n_dream={n_dream}, gamma={self.dream_gamma}"
        )
