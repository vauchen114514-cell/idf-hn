"""IDF-HN KV Cache 层：将 IDF 选择性遗忘门嵌入 Transformer 的 KV 缓存中。

架构：
    历史 K_mem (N, D) / V_mem (N, D) 由 DistilBERT 编码的过往输入填充。
    当前输入 h (B, D) 作为查询，通过 softmax 交叉注意力检索历史 V_mem。
    ForgetGate 根据 h 与 K_mem 的冲突度决定 gamma，
    同步衰减 K_mem 和 V_mem 后写入新样本，使跨任务时旧任务记忆自动退场。

与标准 LLM KV-Cache 的对比：
    标准 KV-Cache: 静态追加，不做衰减/驱逐
    IDF-KV-Cache:  高冲突输入（新任务特征）触发较强衰减，
                   使 K_mem 自动"让位"给新任务的代表性模式。
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model_module.forget_gate.forget_gate import ForgetGate

logger = logging.getLogger(__name__)


class IDFHopfieldKVLayer(nn.Module):
    """IDF 选择性遗忘 KV 缓存层。

    维护一对预分配缓冲区 (K_mem, V_mem) ∈ ℝ^{N×D}，由 ForgetGate 动态管理。
    跨任务时高冲突输入触发较强衰减；低冲突（同任务）输入保留旧记忆范数。
    K_mem 与 V_mem 始终对齐同槽位，衰减与写入操作完全同步。

    Args:
        input_dim: 特征维度 D（DistilBERT 输出为 768）。
        beta: 注意力逆温度参数。
        memory_size: KV 缓冲区最大容量 N。
        forget_gate: 已构建的 ForgetGate 实例。
        eviction_policy: 缓冲区写满后的驱逐策略（norm_min / fifo）。
        write_threshold: 选择性写入阈值；None=始终写入。
        forget_mode: 遗忘机制（input_dependent / none）。
        dropout: W_O 输出层的 Dropout 比例。
    """

    def __init__(
        self,
        input_dim: int,
        beta: float,
        memory_size: int,
        forget_gate: ForgetGate,
        eviction_policy: str = "norm_min",
        write_threshold: float | None = -0.1,
        forget_mode: str = "input_dependent",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.beta = beta
        self.memory_size = memory_size
        self.forget_gate = forget_gate
        self.eviction_policy = eviction_policy
        self.write_threshold = write_threshold
        self.forget_mode = forget_mode

        # ---- KV 缓冲区（K = V = encoder 表示；decay 作用于 K 空间，V 同步衰减）----
        self.register_buffer("_k_mem", torch.zeros(memory_size, input_dim))
        self.register_buffer("_v_mem", torch.zeros(memory_size, input_dim))
        self.register_buffer("_label_buf", torch.full((memory_size,), -1, dtype=torch.long))
        self._n_stored: int = 0
        self._write_ptr: int = 0   # FIFO 循环指针
        self._n_total_seen: int = 0

        # ---- 注意力投影：W_Q 投影查询，W_O 投影交叉注意力输出 ----
        self.W_Q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_O = nn.Linear(input_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def forward(
        self,
        h: Tensor,
        update: bool = True,
        labels: Tensor | None = None,
    ) -> Tensor:
        """IDF-KV 前向：先检索历史记忆，再（可选）更新缓冲区。

        Args:
            h: (B, D) 当前批次编码特征。
            update: 是否执行 IDF 在线更新（训练时为 True）。
            labels: (B,) 样本标签（update=True 时使用）。

        Returns:
            h_aug: (B, D) 交叉注意力残差增强后的特征。
        """
        h_aug = self._cross_attend(h)

        if update:
            h_det = h.detach()
            for i in range(h_det.shape[0]):
                label = int(labels[i].item()) if labels is not None else -1
                self._idf_update(h_det[i], label)

        return h_aug

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时重置 ForgetGate 历史（不清空 KV 缓冲区）。"""
        self.forget_gate.reset()
        logger.info(f"IDFHopfieldKVLayer 已切换至 Task {task_id}")

    # ------------------------------------------------------------------
    # 诊断属性
    # ------------------------------------------------------------------

    @property
    def n_stored(self) -> int:
        return self._n_stored

    # ------------------------------------------------------------------
    # 内部方法：注意力与 IDF 更新
    # ------------------------------------------------------------------

    def _cross_attend(self, h: Tensor) -> Tensor:
        """h 对 (K_mem, V_mem) 的 softmax 交叉注意力，返回残差增强特征。

        scores = beta * W_Q(h) @ K_mem.T / sqrt(D)  (B, N)
        h_mem  = softmax(scores) @ V_mem             (B, D)
        output = h + dropout(W_O(h_mem))
        """
        if self._n_stored == 0:
            return h

        Q = self.W_Q(h)                               # (B, D)
        # .clone() 创建独立副本：_idf_update 的 in-place 操作（mul_ / slot 写入）
        # 会修改 _k_mem/_v_mem 底层存储；若只用 .detach()，autograd 保存的视图
        # 版本号会被篡改，导致 backward 抛出 "modified by an inplace operation"。
        K = self._k_mem[:self._n_stored].detach().clone()   # (N, D)
        V = self._v_mem[:self._n_stored].detach().clone()   # (N, D)

        scale = self.input_dim ** 0.5
        scores = self.beta * (Q @ K.T) / scale        # (B, N)
        weights = F.softmax(scores, dim=-1)            # (B, N)
        attended = weights @ V                         # (B, D)

        return h + self.dropout(self.W_O(attended))   # 残差连接

    def _idf_update(self, u: Tensor, label: int) -> None:
        """单样本 IDF-KV 更新：冲突检测 → 同步遗忘 K+V → 写入新槽位。"""
        if self.forget_mode == "input_dependent":
            if self._n_stored == 0:
                # 空记忆时跳过 ForgetGate（无冲突语义，直接写入）
                self._store_kv(u, label)
                return

            xi = self._retrieve_k(u)
            K = self._k_mem[:self._n_stored]
            gamma, conflict = self.forget_gate.compute_gamma(u, xi, K)
            conflict_val = float(conflict.item())
            gamma_val = float(gamma.item())
            should_write = (self.write_threshold is None) or (conflict_val > self.write_threshold)

            if should_write:
                self._forget_kv(gamma_val)
                self._store_kv(u, label)

        else:  # forget_mode == "none"：始终写入，无衰减
            self._store_kv(u, label)

    def _retrieve_k(self, u: Tensor) -> Tensor:
        """1步 softmax 检索（仅用 K_mem），用于 ForgetGate 冲突计算。

        xi = softmax(beta * K @ u / sqrt(D)) @ K
        """
        K = self._k_mem[:self._n_stored]              # (N, D)
        scores = self.beta * (K @ u) / (self.input_dim ** 0.5)  # (N,)
        weights = F.softmax(scores, dim=0)             # (N,)
        return weights @ K                             # (D,)

    def _forget_kv(self, gamma: float) -> None:
        """同步衰减 K_mem 和 V_mem：K_mem *= (1-gamma)，V_mem *= (1-gamma)。"""
        scale = 1.0 - gamma
        self._k_mem[:self._n_stored].mul_(scale)
        self._v_mem[:self._n_stored].mul_(scale)

    def _store_kv(self, u: Tensor, label: int) -> None:
        """按 eviction_policy 向 K_mem 和 V_mem 写入同槽位（K=V=u）。"""
        self._n_total_seen += 1

        if self._n_stored < self.memory_size:
            slot = self._n_stored
            self._n_stored += 1
            self._write_ptr = self._n_stored % self.memory_size
        elif self.eviction_policy == "norm_min":
            # 驱逐能量缓冲区中范数最小（被遗忘最多）的槽位
            slot = int(self._k_mem.norm(dim=1).argmin().item())
        else:  # fifo
            slot = self._write_ptr
            self._write_ptr = (self._write_ptr + 1) % self.memory_size

        self._k_mem[slot] = u
        self._v_mem[slot] = u   # K = V（同为 encoder 表示，初始相同）
        self._label_buf[slot] = label
