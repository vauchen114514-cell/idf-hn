"""Split-20Newsgroups 持续学习数据集（WikiFacts 语义领域代理）。

使用 scikit-learn 的 20 Newsgroups 数据集，按主题分为 5 个持续学习任务：
    Task 0: talk/religion  (alt.atheism, soc.religion.christian,
                            talk.religion.misc, talk.politics.misc)
    Task 1: comp          (comp.graphics, comp.os.ms-windows.misc,
                            comp.sys.ibm.pc.hardware, comp.sys.mac.hardware)
    Task 2: comp+misc     (comp.windows.x, misc.forsale,
                            rec.autos, rec.motorcycles)
    Task 3: rec+sci       (rec.sport.baseball, rec.sport.hockey,
                            sci.crypt, sci.electronics)
    Task 4: sci+talk      (sci.med, sci.space,
                            talk.politics.guns, talk.politics.mideast)

特征提取（可选）：
  - feature_type=tfidf_svd（默认）：TF-IDF → TruncatedSVD(512) → L2 归一化
  - feature_type=sentence_emb：all-mpnet-base-v2(768) → L2 归一化
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
N_CLASSES = 20
N_TASKS = 5
CLASSES_PER_TASK = 4

# 20 Newsgroups 的全部类别名（sklearn 按字典序排列，索引 0-19）
# 0: alt.atheism
# 1: comp.graphics
# 2: comp.os.ms-windows.misc
# 3: comp.sys.ibm.pc.hardware
# 4: comp.sys.mac.hardware
# 5: comp.windows.x
# 6: misc.forsale
# 7: rec.autos
# 8: rec.motorcycles
# 9: rec.sport.baseball
# 10: rec.sport.hockey
# 11: sci.crypt
# 12: sci.electronics
# 13: sci.med
# 14: sci.space
# 15: soc.religion.christian
# 16: talk.politics.guns
# 17: talk.politics.mideast
# 18: talk.politics.misc
# 19: talk.religion.misc

TASK_GROUPS: list[list[int]] = [
    [0, 15, 18, 19],   # Task 0: talk+religion
    [1, 2, 3, 4],      # Task 1: comp（核心）
    [5, 6, 7, 8],      # Task 2: comp.windows + misc + rec
    [9, 10, 11, 12],   # Task 3: rec sport + sci（密码/电子）
    [13, 14, 16, 17],  # Task 4: sci（医学/太空）+ talk.politics
]


def _load_raw_texts(
    data_dir: str,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """加载 20Newsgroups 原始文本和标签。"""
    from sklearn.datasets import fetch_20newsgroups

    logger.info("从 sklearn 加载 20 Newsgroups...")
    train_data = fetch_20newsgroups(
        data_home=data_dir, subset="train", remove=("headers", "footers", "quotes")
    )
    test_data = fetch_20newsgroups(
        data_home=data_dir, subset="test", remove=("headers", "footers", "quotes")
    )
    return (
        train_data.data, train_data.target.tolist(),
        test_data.data, test_data.target.tolist(),
    )


def _extract_tfidf_features(
    data_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TF-IDF + TruncatedSVD 特征（512 维）。"""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    train_texts, train_raw_labels, test_texts, test_raw_labels = _load_raw_texts(data_dir)

    logger.info("TF-IDF 特征提取（max_features=5000）...")
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
    """Sentence Transformer 特征（768 维，all-mpnet-base-v2）。"""
    from src.data_module.sentence_encoder import encode_texts

    train_texts, train_raw_labels, test_texts, test_raw_labels = _load_raw_texts(data_dir)

    logger.info(f"句子嵌入编码（{model_name}，train={len(train_texts)}，test={len(test_texts)}）...")
    train_feat = encode_texts(train_texts, model_name=model_name)
    val_feat = encode_texts(test_texts, model_name=model_name)

    train_labels = torch.tensor(train_raw_labels, dtype=torch.long)
    val_labels = torch.tensor(test_raw_labels, dtype=torch.long)

    logger.info(f"句子嵌入完成：train={train_feat.shape}, val={val_feat.shape}")
    return train_feat, train_labels, val_feat, val_labels


@register_dataset("split_newsgroups")
class SplitNewsgroups(BaseContinualDataset):
    """Split-20Newsgroups：按主题分割为 5 个持续学习任务。

    支持两种特征类型（feature_type）：
    - tfidf_svd（默认，512 维）
    - sentence_emb（all-mpnet-base-v2，768 维）
    """

    N_CLASSES = N_CLASSES

    def _setup(self) -> None:
        data_dir: str = getattr(self.cfg, "data_dir", "data/newsgroups")
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
            logger.info(f"加载 20Newsgroups {feature_type} 特征缓存...")
            train_feat, train_labels = torch.load(train_cache, weights_only=False)
            val_feat, val_labels = torch.load(val_cache, weights_only=False)
        elif feature_type == "sentence_emb":
            logger.info("首次运行，提取句子嵌入特征（约 10-20 分钟，首次下载模型）...")
            train_feat, train_labels, val_feat, val_labels = _extract_sentemb_features(data_dir)
            torch.save((train_feat, train_labels), train_cache)
            torch.save((val_feat, val_labels), val_cache)
            logger.info("句子嵌入特征已缓存")
        else:
            logger.info("首次运行，提取 TF-IDF+SVD 特征（约 30 秒）...")
            train_feat, train_labels, val_feat, val_labels = _extract_tfidf_features(data_dir)
            torch.save((train_feat, train_labels), train_cache)
            torch.save((val_feat, val_labels), val_cache)
            logger.info("TF-IDF+SVD 特征已缓存")

        logger.info(f"构建 Split-20Newsgroups：{self.n_tasks} 任务，每任务 {CLASSES_PER_TASK} 类，feature_dim={self.INPUT_DIM}")

        for task_id in range(self.n_tasks):
            class_ids = TASK_GROUPS[task_id]
            class_set = set(class_ids)

            train_mask = torch.tensor([l.item() in class_set for l in train_labels])
            val_mask = torch.tensor([l.item() in class_set for l in val_labels])

            train_ds = TensorDataset(train_feat[train_mask], train_labels[train_mask])
            val_ds = TensorDataset(val_feat[val_mask], val_labels[val_mask])

            self._tasks.append(TaskData(
                task_id=task_id,
                n_classes=CLASSES_PER_TASK,
                class_ids=class_ids,
                train_loader=self._make_loader(train_ds, shuffle=True),
                val_loader=self._make_loader(val_ds, shuffle=False),
            ))
            logger.info(
                f"  Task {task_id}: 类别 {class_ids}，"
                f"训练 {train_mask.sum()} 样本，验证 {val_mask.sum()} 样本"
            )

        logger.info("Split-20Newsgroups 构建完成")

    def get_input_dim(self) -> int:
        return self.INPUT_DIM

    def get_n_classes_total(self) -> int:
        return self.N_CLASSES

    @property
    def feature_type(self) -> str:
        return getattr(self, "_feature_type", "tfidf_svd")
