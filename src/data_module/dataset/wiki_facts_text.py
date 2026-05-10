"""WikiFacts（DBpedia14）文本数据集（DistilBERT 预 tokenize 版本）。

复用 wiki_facts.py 的任务分组，但不预计算特征——
改为预 tokenize 后缓存 (input_ids, attention_mask, labels) 张量，
供可微调的 DistilBERT encoder 使用。

DataLoader 返回格式：((input_ids, attn_mask), label)
  其中 input_ids/attn_mask 均为 (B, max_len) LongTensor。
"""
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data_module import register_dataset
from src.data_module.dataset.base_dataset import BaseContinualDataset, TaskData
from src.data_module.dataset.wiki_facts import TASK_GROUPS, TASK_NAMES, _load_dbpedia_texts

logger = logging.getLogger(__name__)

REPORTED_INPUT_DIM = 768
N_CLASSES = 14
N_TASKS = 5
DEFAULT_MAX_LEN = 128


class _TokenizedTextDataset(Dataset):
    """内部：存储预 tokenize 结果，返回 ((input_ids, attn_mask), label)。"""

    def __init__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (self.input_ids[idx], self.attention_mask[idx]), self.labels[idx]


def _tokenize_texts(
    texts: list[str],
    model_name: str,
    max_length: int,
) -> tuple[Tensor, Tensor]:
    """Tokenize 文本列表，返回 (input_ids, attention_mask) LongTensor。"""
    from transformers import AutoTokenizer

    logger.info(f"加载 tokenizer：{model_name}（max_length={max_length}，n={len(texts)}）")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


@register_dataset("wiki_facts_text")
class WikiFactsText(BaseContinualDataset):
    """WikiFacts（DBpedia14，原始文本，预 tokenize 后缓存）。

    供可微调 DistilBERT encoder 使用；DataLoader 返回
    ((input_ids, attn_mask), label) 格式。
    """

    N_CLASSES = N_CLASSES

    def _setup(self) -> None:
        data_dir: str = getattr(self.cfg, "data_dir", "data/wiki_facts")
        model_name: str = getattr(self.cfg, "tokenizer_model", "distilbert-base-uncased")
        max_len: int = getattr(self.cfg, "max_length", DEFAULT_MAX_LEN)

        cache_subdir = f"tokenized_{model_name.replace('/', '_')}_{max_len}"
        cache_path = Path(data_dir) / cache_subdir
        cache_path.mkdir(parents=True, exist_ok=True)

        train_cache = cache_path / "train.pt"
        val_cache = cache_path / "val.pt"

        if train_cache.exists() and val_cache.exists():
            logger.info("加载 WikiFacts tokenized 缓存...")
            train_ids, train_masks, train_labels = torch.load(train_cache, weights_only=False)
            val_ids, val_masks, val_labels = torch.load(val_cache, weights_only=False)
        else:
            logger.info("首次运行，tokenize WikiFacts 文本（DBpedia14 约 5-10 分钟）...")
            train_texts, train_raw_labels, test_texts, test_raw_labels = _load_dbpedia_texts(data_dir)
            train_ids, train_masks = _tokenize_texts(train_texts, model_name, max_len)
            val_ids, val_masks = _tokenize_texts(test_texts, model_name, max_len)
            train_labels = torch.tensor(train_raw_labels, dtype=torch.long)
            val_labels = torch.tensor(test_raw_labels, dtype=torch.long)
            torch.save((train_ids, train_masks, train_labels), train_cache)
            torch.save((val_ids, val_masks, val_labels), val_cache)
            logger.info("WikiFacts tokenized 缓存已保存")

        self.INPUT_DIM = REPORTED_INPUT_DIM
        logger.info(
            f"构建 WikiFactsText：{N_TASKS} 任务，"
            f"max_len={max_len}，tokenizer={model_name}"
        )

        for task_id in range(N_TASKS):
            class_ids = TASK_GROUPS[task_id]
            class_set = set(class_ids)

            train_mask = torch.tensor([l.item() in class_set for l in train_labels])
            val_mask = torch.tensor([l.item() in class_set for l in val_labels])

            train_ds = _TokenizedTextDataset(
                train_ids[train_mask], train_masks[train_mask], train_labels[train_mask]
            )
            val_ds = _TokenizedTextDataset(
                val_ids[val_mask], val_masks[val_mask], val_labels[val_mask]
            )

            self._tasks.append(TaskData(
                task_id=task_id,
                n_classes=len(class_ids),
                class_ids=class_ids,
                train_loader=self._make_loader(train_ds, shuffle=True),
                val_loader=self._make_loader(val_ds, shuffle=False),
            ))
            logger.info(
                f"  Task {task_id} [{TASK_NAMES[task_id]}]: 类别 {class_ids}，"
                f"训练 {train_mask.sum()} 样本，验证 {val_mask.sum()} 样本"
            )

        logger.info("WikiFactsText 构建完成")

    def get_input_dim(self) -> int:
        return self.INPUT_DIM

    def get_n_classes_total(self) -> int:
        return self.N_CLASSES
