"""GSS-Greedy（Gradient-based Sample Selection）基线。

Aljundi et al., "Gradient based sample selection for online continual learning"，
NeurIPS 2019。

核心思想：维护梯度多样的回放缓冲区。对每个新样本 x：
1. 计算其对分类头的梯度向量 g(x)
2. 从缓冲区随机抽取 S 个候选样本，分别计算梯度
3. 找出与 g(x) 余弦相似度最高的候选（最冗余）
4. 若该相似度 > 0，用 x 替换该候选（保持梯度多样性）

梯度计算使用 torch.autograd.grad()（不修改 .grad 属性），与外部训练图完全隔离。
因 ResNet-18 特征已冻结，梯度仅作用于线性分类头参数。
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model_module import register_model

logger = logging.getLogger(__name__)


@register_model("gss")
class GSSBaseline(nn.Module):
    """GSS-Greedy 基线（梯度多样性驱动的缓冲区管理 + 线性分类器）。

    与 ERBaseline 结构对齐：same forward/sample_replay/reset_for_task 接口，
    区别在于缓冲区写入策略：随机 Reservoir → 梯度多样性选择。

    Args:
        cfg: Hydra 配置对象，需包含 cfg.model.memory_size（可选 candidate_size）。
        input_dim: 输入特征维度 D。
        n_classes: 分类头类别数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()
        m = cfg.model
        self.buffer_size: int = getattr(m, "memory_size", 1000)
        self.candidate_size: int = getattr(m, "candidate_size", 10)
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.classifier = nn.Linear(input_dim, n_classes)

        # 缓冲区（非参数，不参与梯度反向传播）
        self.register_buffer("_feat_buf", torch.zeros(self.buffer_size, input_dim))
        self.register_buffer("_label_buf", torch.full((self.buffer_size,), -1, dtype=torch.long))
        self._n_valid: int = 0   # 当前缓冲区中有效条目数（≤ buffer_size）

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

        Args:
            x: (B, D) 输入批次。
            update: 训练时为 True，触发 per-sample GSS 缓冲区更新。
            labels: (B,) 标签，update=True 时必须提供。

        Returns:
            logits: (B, C) 分类 logits。
            energy: (B,) 全零占位（GSS 无 Hopfield 能量）。
        """
        logits = self.classifier(x)
        dummy_energy = torch.zeros(x.shape[0], device=x.device)

        if update and labels is not None:
            for i in range(x.shape[0]):
                self._gss_update(x[i].detach(), int(labels[i].item()))

        return logits, dummy_energy

    def sample_replay(self, n: int, energy_priority: bool = False) -> tuple[Tensor, Tensor] | None:
        """从缓冲区均匀随机采样 n 条 (特征, 标签)。

        Returns:
            (feat, label) 元组；缓冲区为空时返回 None。
        """
        if self._n_valid == 0:
            return None
        n_sample = min(n, self._n_valid)
        idx = torch.randperm(self._n_valid, device=self._feat_buf.device)[:n_sample]
        return self._feat_buf[idx].clone(), self._label_buf[idx].clone()

    def reset_for_task(self, task_id: int) -> None:
        """任务切换时不清空缓冲区（跨任务梯度多样性需要历史样本）。"""
        logger.info(f"GSSBaseline: 切换至 Task {task_id}（缓冲区保留，当前有效={self._n_valid}）")

    # ------------------------------------------------------------------
    # GSS 核心逻辑
    # ------------------------------------------------------------------

    def _compute_grad_vector(self, feat: Tensor, label: int) -> Tensor:
        """计算单样本对分类头的梯度向量（与外部训练图隔离）。

        使用 torch.autograd.grad() 而非 loss.backward()，不修改 .grad 属性，
        不干扰外部训练循环的计算图。torch.enable_grad() 保证即使外层有
        no_grad() 上下文也能正确计算。

        Args:
            feat: (D,) 特征向量（已 detach）。
            label: 整数标签。

        Returns:
            (P,) 展平的梯度向量，P = n_classes × (input_dim + 1)。
        """
        label_t = torch.tensor([label], device=feat.device)
        params = list(self.classifier.parameters())

        with torch.enable_grad():
            logits = self.classifier(feat.detach().unsqueeze(0))
            loss = F.cross_entropy(logits, label_t)
            grads = torch.autograd.grad(loss, params, retain_graph=False)

        return torch.cat([g.detach().flatten() for g in grads])

    def _gss_update(self, feat: Tensor, label: int) -> None:
        """GSS-Greedy 缓冲区在线更新。

        填充阶段（缓冲区未满）：直接写入，与 ER 相同。
        满载阶段：
          1. 计算新样本梯度 g_new
          2. 随机采样 candidate_size 个已有样本，计算各自梯度
          3. 找余弦相似度最高的候选 i*（最冗余）
          4. 若 cos_sim(g_new, g_{i*}) > 0，用新样本替换 i*

        Args:
            feat: (D,) 已 detach 的特征向量。
            label: 整数标签。
        """
        if self._n_valid < self.buffer_size:
            # 填充阶段：直接写入
            self._feat_buf[self._n_valid] = feat
            self._label_buf[self._n_valid] = label
            self._n_valid += 1
            return

        # 满载阶段：GSS 驱逐策略
        g_new = self._compute_grad_vector(feat, label)

        # 随机采样候选集
        n_cand = min(self.candidate_size, self._n_valid)
        cand_indices = torch.randperm(self._n_valid, device=self._feat_buf.device)[:n_cand]

        max_sim = -float("inf")
        replace_slot = -1

        for idx in cand_indices:
            slot = int(idx.item())
            buf_label = int(self._label_buf[slot].item())
            if buf_label < 0:
                continue
            g_buf = self._compute_grad_vector(self._feat_buf[slot], buf_label)
            sim = float(
                F.cosine_similarity(g_new.unsqueeze(0), g_buf.unsqueeze(0)).item()
            )
            if sim > max_sim:
                max_sim = sim
                replace_slot = slot

        # 仅在找到冗余样本（相似度 > 0）时替换
        if max_sim > 0.0 and replace_slot >= 0:
            self._feat_buf[replace_slot] = feat
            self._label_buf[replace_slot] = label
            logger.debug(
                f"GSS: 替换 slot {replace_slot}（cos_sim={max_sim:.4f}）"
            )
        else:
            logger.debug(
                f"GSS: 跳过（max_sim={max_sim:.4f} ≤ 0，缓冲区已最大多样化）"
            )
