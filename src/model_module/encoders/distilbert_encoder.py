"""DistilBERT 编码器：输出 mean-pooled (B, 768) 特征的可微调包装器。"""
import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class DistilBertEncoder(nn.Module):
    """可微调的 DistilBERT 编码器，使用 mean pooling 输出句子表示。

    Args:
        model_name: HuggingFace 模型名称，默认 distilbert-base-uncased。
        dropout: 输出 dropout 概率。
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        logger.info(f"加载预训练模型：{model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size: int = self.bert.config.hidden_size  # 768

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """编码 tokenized 文本为固定维度句子表示。

        Args:
            input_ids: (B, L) LongTensor，token ID。
            attention_mask: (B, L) LongTensor，填充掩码（1=有效，0=padding）。

        Returns:
            (B, hidden_size) FloatTensor，mean-pooled 句子嵌入。
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state          # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        h = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, H)
        return self.dropout(h)
