"""WikiFacts 持续学习数据集（基于 DBpedia14）。

使用 HuggingFace datasets 库加载 DBpedia14，按知识领域分为 5 个持续学习任务：
    Task 0: Organizations & Arts  (Company, Educational Institution, Artist)
    Task 1: People & Transport    (Athlete, Office Holder, Mean of Transportation)
    Task 2: Places                (Building, Natural Place, Village)
    Task 3: Natural World         (Animal, Plant)
    Task 4: Media & Works         (Album, Film, Written Work)

特征提取（可选）：
  - feature_type=tfidf_svd（默认，512 维）
  - feature_type=sentence_emb：all-mpnet-base-v2(768) → L2 归一化

评估指标（除 AA/BWT/FWT 外）：
    - 语义相似度（SemanticSimilarity）：检索向量与原始输入的 cosine 相似度
    - 压缩比（CompressionRatio）：记忆库有效容量 / 累计训练样本总数
"""
import logging
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.data_module import register_dataset
from src.data_module.dataset.base_dataset import BaseContinualDataset, TaskData

logger = logging.getLogger(__name__)

FEATURE_DIM_TFIDF = 512
FEATURE_DIM_SENTEMB = 768
N_CLASSES = 14
N_TASKS = 5

# DBpedia14 类别索引（HuggingFace 标签顺序）
# 0: Company
# 1: Educational Institution
# 2: Artist
# 3: Athlete
# 4: Office Holder
# 5: Mean of Transportation
# 6: Building
# 7: Natural Place
# 8: Village
# 9: Animal
# 10: Plant
# 11: Album
# 12: Film
# 13: Written Work

TASK_GROUPS: list[list[int]] = [
    [0, 1, 2],       # Task 0: Organizations & Arts
    [3, 4, 5],       # Task 1: People & Transport
    [6, 7, 8],       # Task 2: Places
    [9, 10],         # Task 3: Natural World
    [11, 12, 13],    # Task 4: Media & Works
]

TASK_NAMES = [
    "Organizations & Arts",
    "People & Transport",
    "Places",
    "Natural World",
    "Media & Works",
]


def _load_dbpedia_texts(
    data_dir: str,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """加载 DBpedia14 原始文本和标签。"""
    from datasets import load_dataset

    logger.info("加载 DBpedia14（HuggingFace datasets）...")
    ds = load_dataset("dbpedia_14", cache_dir=data_dir)

    def _texts(split):
        return [f"{row['title']} {row['content']}" for row in split]

    return (
        _texts(ds["train"]), [row["label"] for row in ds["train"]],
        _texts(ds["test"]),  [row["label"] for row in ds["test"]],
    )


def _extract_tfidf_features(
    data_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TF-IDF + TruncatedSVD 特征（512 维）。"""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    train_texts, train_raw_labels, test_texts, test_raw_labels = _load_dbpedia_texts(data_dir)

    logger.info(f"TF-IDF 特征提取（max_features=5000）... train={len(train_texts)}")
    vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words="english")
    train_sparse = vectorizer.fit_transform(train_texts)
    test_sparse = vectorizer.transform(test_texts)

    logger.info(f"TruncatedSVD 降维至 {FEATURE_DIM_TFIDF} 维...")
    svd = TruncatedSVD(n_components=FEATURE_DIM_TFIDF, random_state=42)
    train_dense = svd.fit_transform(train_sparse)
    test_dense = svd.transform(test_sparse)

    normalizer = Normalizer(norm="l2")
    train_dense = normalizer.fit_transform(train_dense)
    test_dense = normalizer.transform(test_dense)

    train_feat = torch.tensor(train_dense, dtype=torch.float32)
    val_feat = torch.tensor(test_dense, dtype=torch.float32)
    train_labels = torch.tensor(train_raw_labels, dtype=torch.long)
    val_labels = torch.tensor(test_raw_labels, dtype=torch.long)

    logger.info(f"TF-IDF+SVD 完成：train={train_feat.shape}, val={val_feat.shape}")
    return train_feat, train_labels, val_feat, val_labels


def _extract_sentemb_features(
    data_dir: str,
    model_name: str = "all-mpnet-base-v2",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sentence Transformer 特征（768 维）。DBpedia14 较大，GPU 约 5-10 分钟。"""
    from src.data_module.sentence_encoder import encode_texts

    train_texts, train_raw_labels, test_texts, test_raw_labels = _load_dbpedia_texts(data_dir)

    logger.info(f"句子嵌入编码 train ({len(train_texts)} 条)...")
    train_feat = encode_texts(train_texts, model_name=model_name)
    logger.info(f"句子嵌入编码 test ({len(test_texts)} 条)...")
    val_feat = encode_texts(test_texts, model_name=model_name)

    train_labels = torch.tensor(train_raw_labels, dtype=torch.long)
    val_labels = torch.tensor(test_raw_labels, dtype=torch.long)

    logger.info(f"句子嵌入完成：train={train_feat.shape}, val={val_feat.shape}")
    return train_feat, train_labels, val_feat, val_labels


@register_dataset("wiki_facts")
class WikiFactsDataset(BaseContinualDataset):
    """WikiFacts（DBpedia14）：按知识领域分割为 5 个持续学习任务。

    任务设计遵循语义领域漂移原则：
    Organizations → People → Places → Nature → Media

    支持两种特征类型（feature_type）：
    - tfidf_svd（默认，512 维）
    - sentence_emb（all-mpnet-base-v2，768 维，GPU 约 5-10 分钟首次提取）
    """

    N_CLASSES = N_CLASSES

    def _setup(self) -> None:
        data_dir: str = getattr(self.cfg, "data_dir", "data/wiki_facts")
        feature_type: str = getattr(self.cfg, "feature_type", "tfidf_svd")
        self._feature_type = feature_type

        if feature_type == "sentence_emb":
            cache_subdir = "sentence_emb_features"
            self.INPUT_DIM = FEATURE_DIM_SENTEMB
        else:
            cache_subdir = "tfidf_svd_features"
            self.INPUT_DIM = FEATURE_DIM_TFIDF

        cache_path = Path(data_dir) / cache_subdir
        cache_path.mkdir(parents=True, exist_ok=True)
        train_cache = cache_path / "train.pt"
        val_cache = cache_path / "val.pt"

        if train_cache.exists() and val_cache.exists():
            logger.info(f"加载 WikiFacts {feature_type} 特征缓存...")
            train_feat, train_labels = torch.load(train_cache, weights_only=False)
            val_feat, val_labels = torch.load(val_cache, weights_only=False)
        elif feature_type == "sentence_emb":
            logger.info("首次运行，提取 WikiFacts 句子嵌入特征（GPU 约 5-10 分钟）...")
            train_feat, train_labels, val_feat, val_labels = _extract_sentemb_features(data_dir)
            torch.save((train_feat, train_labels), train_cache)
            torch.save((val_feat, val_labels), val_cache)
            logger.info("WikiFacts 句子嵌入特征已缓存")
        else:
            logger.info("首次运行，提取 WikiFacts TF-IDF+SVD 特征（约 2-3 分钟）...")
            train_feat, train_labels, val_feat, val_labels = _extract_tfidf_features(data_dir)
            torch.save((train_feat, train_labels), train_cache)
            torch.save((val_feat, val_labels), val_cache)
            logger.info("WikiFacts TF-IDF+SVD 特征已缓存")

        logger.info(
            f"构建 WikiFacts：{self.n_tasks} 任务，feature_dim={self.INPUT_DIM}，"
            f"类别分组：{[TASK_NAMES[i] for i in range(self.n_tasks)]}"
        )

        for task_id in range(self.n_tasks):
            class_ids = TASK_GROUPS[task_id]
            class_set = set(class_ids)

            train_mask = torch.tensor([l.item() in class_set for l in train_labels])
            val_mask = torch.tensor([l.item() in class_set for l in val_labels])

            train_ds = TensorDataset(train_feat[train_mask], train_labels[train_mask])
            val_ds = TensorDataset(val_feat[val_mask], val_labels[val_mask])

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

        logger.info("WikiFacts 构建完成")

    def get_input_dim(self) -> int:
        return self.INPUT_DIM

    def get_n_classes_total(self) -> int:
        return self.N_CLASSES

    @property
    def feature_type(self) -> str:
        return getattr(self, "_feature_type", "tfidf_svd")
