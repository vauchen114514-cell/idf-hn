"""IDF-HN Transformer 分类器：DistilBERT + IDF KV-Cache 交叉注意力 + 文本回放缓冲区。

架构：
    DistilBertEncoder（fine-tunable）→ h (B, 768)
        → IDFHopfieldKVLayer（IDF 选择性遗忘 KV 交叉注意力）→ h_aug (B, 768)
        → Linear(768, n_classes) → logits

与 idf_hn_distilbert 的区别：
    idf_hn_distilbert:   IDF-HN 管理平铺记忆向量，直接用线性层分类 encoder 输出
    idf_hn_transformer:  IDF-HN 管理 (K_mem, V_mem) 对，通过交叉注意力增强当前 h，
                         使分类器同时利用当前输入和历史记忆

注册名称：idf_hn_transformer
"""
import logging
import random

import torch
import torch.nn as nn
from torch import Tensor

from src.model_module import register_model
from src.model_module.encoders.distilbert_encoder import DistilBertEncoder
from src.model_module.forget_gate.forget_gate import ForgetGate
from src.model_module.idf_hn.idf_kv_layer import IDFHopfieldKVLayer

logger = logging.getLogger(__name__)


@register_model("idf_hn_transformer")
class IDFHopfieldTransformerClassifier(nn.Module):
    """IDF-HN Transformer 分类器。

    IDF-KV 核心：IDFHopfieldKVLayer 以 h 为查询，对 K_mem/V_mem 做 softmax
    交叉注意力，得到历史语义增强向量 h_mem，输出 h_aug = h + dropout(W_O(h_mem))。
    ForgetGate 基于 h 与 K_mem 的冲突度控制衰减强度，保证跨任务的选择性记忆管理。
    文本回放缓冲区（Reservoir Sampling）用于 encoder 微调时防止灾难性遗忘。

    Args:
        cfg: Hydra/OmegaConf 配置节点，需包含 model.encoder.*、model.forget_gate.* 等字段。
        input_dim: 数据集报告的输入维度（被 encoder 768 维覆盖）。
        n_classes: 全局类别总数。
    """

    def __init__(self, cfg, input_dim: int, n_classes: int) -> None:
        super().__init__()

        m = cfg.model
        enc_cfg = m.encoder

        # --- DistilBERT Encoder ---
        self.encoder = DistilBertEncoder(
            model_name=enc_cfg.model_name,
            dropout=getattr(enc_cfg, "dropout", 0.1),
        )
        D = self.encoder.hidden_size  # 768

        # --- ForgetGate（IDF-KV 内部使用）---
        fg = m.forget_gate
        forget_gate = ForgetGate(
            gamma_0=fg.gamma_0,
            delta_gamma=fg.delta_gamma,
            tau=fg.tau,
            beta=m.beta,
            adaptive_tau=fg.adaptive_tau,
            tau_percentile=fg.tau_percentile,
            ema_alpha=getattr(fg, "ema_alpha", 0.9),
        )

        # --- IDF KV 交叉注意力层 ---
        self.kv_layer = IDFHopfieldKVLayer(
            input_dim=D,
            beta=m.beta,
            memory_size=m.memory_size,
            forget_gate=forget_gate,
            eviction_policy=getattr(m, "eviction_policy", "norm_min"),
            write_threshold=getattr(m, "write_threshold", -0.1),
            forget_mode=getattr(m, "forget_mode", "input_dependent"),
            dropout=getattr(enc_cfg, "dropout", 0.1),
        )

        # --- 分类头（作用于增强特征 h_aug）---
        self.classifier = nn.Linear(D, n_classes)

        # --- 文本回放缓冲区（Reservoir Sampling，存储原始 tokenized 文本）---
        self._text_buf: list[tuple[Tensor, Tensor, int]] = []
        self._buf_max: int = m.memory_size
        self._buf_seen: int = 0

    # ------------------------------------------------------------------
    # nn.Module 接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        update: bool = True,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """双路径前向传播。

        Args:
            x: tuple (input_ids, attn_mask) 或 FloatTensor (B, 768)。
            update: 是否执行 KV 在线更新（训练时为 True）。
            labels: (B,) 样本标签。

        Returns:
            logits: (B, n_classes)。
            energy: (B,) 全零占位，与 IDFHopfieldNetwork 接口保持一致。
        """
        if isinstance(x, (tuple, list)):
            input_ids, attn_mask = x
            h = self.encoder(input_ids, attn_mask)   # (B, 768)
            if self.training and update and labels is not None:
                self._reservoir_add(
                    input_ids.detach().cpu(),
                    attn_mask.detach().cpu(),
                    labels.detach().cpu(),
                )
        else:
            h = x  # 已编码特征（来自 sample_replay）

        h_aug = self.kv_layer(h, update=update, labels=labels)   # (B, D)
        logits = self.classifier(h_aug)                           # (B, C)
        energy = torch.zeros(h.shape[0], device=h.device)
        return logits, energy

    def sample_replay(
        self, n: int, energy_priority: bool = False
    ) -> tuple[Tensor, Tensor] | None:
        """从文本缓冲区采样，用当前 encoder 重编码（梯度流过 encoder）。

        Args:
            n: 期望采样数量。
            energy_priority: 忽略（文本缓冲区使用均匀采样）。

        Returns:
            (h, labels)：编码特征 (k, 768) 和标签 (k,)；None 若缓冲区为空。
        """
        if not self._text_buf:
            return None

        k = min(n, len(self._text_buf))
        indices = random.sample(range(len(self._text_buf)), k)

        device = next(self.parameters()).device
        input_ids = torch.stack([self._text_buf[i][0] for i in indices]).to(device)
        attn_mask = torch.stack([self._text_buf[i][1] for i in indices]).to(device)
        label_vals = torch.tensor(
            [self._text_buf[i][2] for i in indices], dtype=torch.long, device=device
        )

        h = self.encoder(input_ids, attn_mask)  # 梯度流过 encoder
        return h, label_vals

    def reset_for_task(self, task_id: int) -> None:
        """任务切换：重置 ForgetGate 历史（保留 KV 缓冲区跨任务记忆）。"""
        self.kv_layer.reset_for_task(task_id)

    # ------------------------------------------------------------------
    # 诊断属性
    # ------------------------------------------------------------------

    @property
    def memory_size(self) -> int:
        return self.kv_layer.n_stored

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reservoir_add(
        self,
        input_ids: Tensor,   # (B, L) CPU LongTensor
        attn_mask: Tensor,   # (B, L) CPU LongTensor
        labels: Tensor,      # (B,) CPU LongTensor
    ) -> None:
        """Vitter Algorithm R Reservoir Sampling。"""
        B = input_ids.size(0)
        for i in range(B):
            self._buf_seen += 1
            item = (input_ids[i].clone(), attn_mask[i].clone(), int(labels[i].item()))
            if len(self._text_buf) < self._buf_max:
                self._text_buf.append(item)
            else:
                j = random.randint(0, self._buf_seen - 1)
                if j < self._buf_max:
                    self._text_buf[j] = item
