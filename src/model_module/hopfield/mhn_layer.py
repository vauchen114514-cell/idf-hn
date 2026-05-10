"""Modern Hopfield Network（Ramsauer et al. 2021）实现。

作为 IDF-HN 的基线和父类，提供标准的 softmax 检索更新规则与
log-sum-exp 能量函数。记忆以预分配的循环缓冲区存储，支持 GPU。

双缓冲区设计：
  _mem（能量缓冲区）：受 ForgetGate 衰减，驱逐策略为 norm_min/fifo/reservoir，
                     用于 Hopfield 检索和能量计算。
  _replay_buf（回放缓冲区）：原始特征永不修改，Reservoir Sampling 管理，
                            用于 sample_replay() 回放旧任务样本。

分离原因：ForgetGate 的 mul_(1-gamma) 会使旧任务特征趋近零，
若回放缓冲区与能量缓冲区共享，早期任务样本将因范数归零而回放失效。
"""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model_module.hopfield.base_hopfield import BaseHopfieldNetwork
from src.model_module.hopfield.energy import (
    delta_energy,
    mhn_energy,
    softmax_update,
)

logger = logging.getLogger(__name__)


class ModernHopfieldLayer(BaseHopfieldNetwork):
    """Ramsauer 2021 MHN：有限容量记忆 + softmax 检索。

    记忆矩阵 X ∈ ℝ^{N×D} 以预分配的循环缓冲区维护，避免重复 torch.cat
    导致的内存碎片。遗忘通过原地 mul_() 实现，仅作用于能量缓冲区。

    Args:
        input_dim: 输入维度 D。
        beta: 逆温度参数。
        max_memories: 缓冲区最大行数（能量缓冲区与回放缓冲区相同大小）。
        eviction_policy: 能量缓冲区的驱逐策略（fifo / norm_min / reservoir）。
        dual_buffer: 若为 True，回放缓冲区保存原始特征且不受 ForgetGate 衰减；
            若为 False，sample_replay() 直接从能量缓冲区采样，用于单缓冲消融。
    """

    def __init__(
        self,
        input_dim: int,
        beta: float,
        max_memories: int = 1000,
        eviction_policy: str = "fifo",
        dual_buffer: bool = True,
    ) -> None:
        super().__init__(input_dim, beta)
        self.max_memories = max_memories
        if eviction_policy not in ("fifo", "norm_min", "reservoir"):
            raise ValueError(
                f"eviction_policy 必须为 'fifo' / 'norm_min' / 'reservoir'，收到 '{eviction_policy}'"
            )
        self.eviction_policy = eviction_policy
        self.dual_buffer = dual_buffer

        # --- 能量缓冲区：受 ForgetGate 衰减，用于 Hopfield 检索/能量计算 ---
        self.register_buffer("_mem", torch.zeros(max_memories, input_dim))
        self.register_buffer("_label_mem", torch.full((max_memories,), -1, dtype=torch.long))
        self._n_stored: int = 0       # 已写入行数（≤ max_memories）
        self._write_ptr: int = 0      # FIFO 循环写指针
        self._n_total_seen: int = 0   # 累计见过的样本总数（energy buffer reservoir 用）

        # --- 回放缓冲区：原始特征永不修改，Reservoir Sampling 管理 ---
        # dual_buffer=False 时仍注册这些 buffer 以保持 state_dict 形状稳定，
        # 但 store()/sample_replay() 不使用它们。
        self.register_buffer("_replay_buf", torch.zeros(max_memories, input_dim))
        self.register_buffer("_replay_label_buf", torch.full((max_memories,), -1, dtype=torch.long))
        self._replay_n_stored: int = 0
        self._replay_n_total_seen: int = 0

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------

    def store(self, u: Tensor, label: int = -1) -> None:
        """将 u 写入能量缓冲区（按 eviction_policy）和回放缓冲区（Reservoir）。

        能量缓冲区策略：
        - fifo：循环覆盖最旧记忆
        - norm_min：驱逐范数最小（被 ForgetGate 衰减最多）的记忆
        - reservoir：Vitter 1985 Algorithm R，均匀随机替换

        回放缓冲区：始终使用 Reservoir Sampling，保证全任务均匀覆盖。
        ForgetGate 的 forget() 不修改回放缓冲区，原始特征永久保留。

        Args:
            u: (D,) 输入向量。
            label: 样本类别标签（-1 表示未知）。
        """
        if u.dim() != 1 or u.shape[0] != self.input_dim:
            raise ValueError(
                f"store() 期望 ({self.input_dim},) 向量，收到 {u.shape}"
            )
        u = u.detach().to(self._mem.device)

        # -- 能量缓冲区写入 --
        self._n_total_seen += 1
        if self._n_stored < self.max_memories:
            self._mem[self._n_stored] = u
            self._label_mem[self._n_stored] = label
            self._n_stored += 1
            self._write_ptr = self._n_stored % self.max_memories
        elif self.eviction_policy == "reservoir":
            j = int(torch.randint(0, self._n_total_seen, (1,)).item())
            if j < self.max_memories:
                self._mem[j] = u
                self._label_mem[j] = label
        elif self.eviction_policy == "norm_min":
            slot = int(self._mem.norm(dim=1).argmin().item())
            self._mem[slot] = u
            self._label_mem[slot] = label
        else:  # fifo
            self._mem[self._write_ptr] = u
            self._label_mem[self._write_ptr] = label
            self._write_ptr = (self._write_ptr + 1) % self.max_memories

        # -- 回放缓冲区写入（双缓冲模式下启用）--
        if self.dual_buffer:
            self._replay_n_total_seen += 1
            if self._replay_n_stored < self.max_memories:
                self._replay_buf[self._replay_n_stored] = u
                self._replay_label_buf[self._replay_n_stored] = label
                self._replay_n_stored += 1
            else:
                j = int(torch.randint(0, self._replay_n_total_seen, (1,)).item())
                if j < self.max_memories:
                    self._replay_buf[j] = u
                    self._replay_label_buf[j] = label

        logger.debug(f"store: energy_buf={self._n_stored}, replay_buf={self._replay_n_stored}")

    def retrieve(self, xi: Tensor, n_steps: int = 1) -> Tensor:
        """执行 n 步 softmax 更新，返回更新后状态。"""
        if self._n_stored == 0:
            logger.warning("retrieve() 调用时记忆为空，返回原始输入")
            return xi

        state = xi.to(self._mem.device)
        active = self._mem[:self._n_stored]
        for _ in range(n_steps):
            state = softmax_update(state, active, self.beta)
        return state

    def forget(self, gamma: float) -> None:
        """原地指数衰减遗忘：仅作用于能量缓冲区 _mem，不修改回放缓冲区。

        Args:
            gamma: 遗忘率 ∈ [0, 1]。
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma 必须在 [0,1]，收到 {gamma}")
        self._mem.mul_(1.0 - gamma)
        logger.debug(f"forget: gamma={gamma:.4f}, energy_buf_norm={self._mem.norm():.4f}")

    def forget_per_slot(self, gammas: Tensor) -> None:
        """每个记忆槽位单独衰减（静态密度消融用）。

        Args:
            gammas: (N_stored,) 每个槽位的遗忘率，自动裁剪到 [0,1]。
        """
        n = self._n_stored
        if n == 0:
            return
        decay = (1.0 - gammas[:n].clamp(0.0, 1.0).to(self._mem.device)).unsqueeze(-1)
        self._mem[:n].mul_(decay)
        logger.debug(f"forget_per_slot: mean_gamma={gammas[:n].mean():.4f}")

    def energy(self, xi: Tensor) -> Tensor:
        """E(ξ; X) = -lse(β, Xξ) + ½‖ξ‖²。"""
        if self._n_stored == 0:
            return 0.5 * (xi * xi).sum(dim=-1)
        return mhn_energy(xi.to(self._mem.device), self._mem[:self._n_stored], self.beta)

    def conflict(self, u: Tensor, xi: Tensor) -> Tensor:
        """计算新模式 u 对当前状态 xi 的冲突度（ΔE）。"""
        if self._n_stored == 0:
            return torch.tensor(0.0, device=u.device)
        return delta_energy(
            xi.to(self._mem.device),
            self._mem[:self._n_stored],
            u.to(self._mem.device),
            self.beta,
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return self._n_stored

    @property
    def memory_matrix(self) -> Tensor:
        """返回当前活跃能量记忆的视图（不拷贝）。"""
        return self._mem[:self._n_stored]

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> tuple[Tensor, Tensor] | None:
        """从回放缓冲区采样 n 条 (特征, 标签)。

        两种采样策略：
        - energy_priority=False（默认）：均匀随机采样（Reservoir Sampling）
        - energy_priority=True：能量反比优先级采样——对每个候选 x 计算其与
          当前能量缓冲区 _mem 的最大注意力分数 max_j(β·x·mem[j])，低匹配分
          （被 ForgetGate 衰减导致记忆检索失效的样本）获得更高回放优先级。
          链路：ForgetGate 衰减 _mem → 衰减槽检索分低 → 对应旧样本优先回放

        Args:
            n: 采样数量。
            energy_priority: 是否启用能量反比优先级采样。

        Returns:
            (features, labels) 元组，或 None（回放缓冲区为空时）。
        """
        source_buf = self._replay_buf if self.dual_buffer else self._mem
        label_buf = self._replay_label_buf if self.dual_buffer else self._label_mem
        n_stored = self._replay_n_stored if self.dual_buffer else self._n_stored

        if n_stored == 0:
            return None
        valid_idx = (
            label_buf[:n_stored] >= 0
        ).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            return None
        n = min(n, len(valid_idx))

        if energy_priority and self._n_stored > 0:
            candidates = source_buf[valid_idx]                 # (M, D)
            mem = self._mem[:self._n_stored]                   # (K, D)
            # max attention score: β · max_j (x · mem[j])
            max_scores = (candidates @ mem.T).max(dim=1).values * self.beta  # (M,)
            # 反比权重：低检索分（已被遗忘）→ 高优先级
            weights = F.softmax(-max_scores, dim=0)            # (M,)
            sampled = torch.multinomial(weights, n, replacement=False)
            idx = valid_idx[sampled]
        else:
            perm = torch.randperm(len(valid_idx), device=source_buf.device)[:n]
            idx = valid_idx[perm]

        return source_buf[idx].clone().detach(), label_buf[idx].clone()

    def reset_memory(self) -> None:
        """清空能量缓冲区和回放缓冲区。"""
        self._mem.zero_()
        self._label_mem.fill_(-1)
        self._n_stored = 0
        self._write_ptr = 0
        self._n_total_seen = 0

        self._replay_buf.zero_()
        self._replay_label_buf.fill_(-1)
        self._replay_n_stored = 0
        self._replay_n_total_seen = 0
        logger.info(
            f"能量缓冲区和回放缓冲区已清空"
            f"（policy={self.eviction_policy}, dual_buffer={self.dual_buffer}）"
        )
