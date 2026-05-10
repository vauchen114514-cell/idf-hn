"""Split-CIFAR-100 持续学习数据集。

将 CIFAR-100 按类别分割为 n_tasks 个任务（默认 20 任务，每任务 5 类）。
输入特征使用预提取的 ResNet 特征（512 维），避免 8GB 显存瓶颈。
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, models, transforms

from src.data_module import register_dataset
from src.data_module.dataset.base_dataset import BaseContinualDataset, TaskData

logger = logging.getLogger(__name__)

FEATURE_DIM = 512  # ResNet-18 的 avgpool 输出维度


def _extract_features(dataset, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """使用预训练 ResNet-18 提取特征（冻结权重）。"""
    encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()  # 去掉分类头，输出 512 维特征
    encoder = encoder.to(device).eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4)
    features, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = encoder(x)
            features.append(feat.cpu())
            labels.append(y)

    return torch.cat(features), torch.cat(labels)


@register_dataset("split_cifar100")
class SplitCIFAR100(BaseContinualDataset):
    """Split-CIFAR-100：使用 ResNet-18 预提取特征，分割为多任务。"""

    N_CLASSES = 100

    def _setup(self) -> None:
        classes_per_task: int = getattr(self.cfg, "classes_per_task", 5)
        data_dir: str = getattr(self.cfg, "data_dir", "data/cifar100")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 特征缓存路径
        cache_path = Path(data_dir) / "resnet18_features"
        cache_path.mkdir(parents=True, exist_ok=True)
        train_cache = cache_path / "train_features.pt"
        val_cache = cache_path / "val_features.pt"

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 优先加载缓存
        if train_cache.exists() and val_cache.exists():
            logger.info("加载 ResNet-18 特征缓存...")
            train_feat, train_labels = torch.load(train_cache, weights_only=False)
            val_feat, val_labels = torch.load(val_cache, weights_only=False)
        else:
            logger.info("首次运行，提取 ResNet-18 特征（约 2 分钟）...")
            train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
            val_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
            train_feat, train_labels = _extract_features(train_full, device)
            val_feat, val_labels = _extract_features(val_full, device)
            torch.save((train_feat, train_labels), train_cache)
            torch.save((val_feat, val_labels), val_cache)
            logger.info("特征提取完成，已缓存")

        self._input_dim = train_feat.shape[1]
        logger.info(f"构建 Split-CIFAR-100：{self.n_tasks} 任务，每任务 {classes_per_task} 类")

        for task_id in range(self.n_tasks):
            start = task_id * classes_per_task
            class_ids = list(range(start, start + classes_per_task))
            class_set = set(class_ids)

            train_mask = torch.tensor([l.item() in class_set for l in train_labels])
            val_mask = torch.tensor([l.item() in class_set for l in val_labels])

            train_ds = TensorDataset(train_feat[train_mask], train_labels[train_mask])
            val_ds = TensorDataset(val_feat[val_mask], val_labels[val_mask])

            self._tasks.append(TaskData(
                task_id=task_id,
                n_classes=classes_per_task,
                class_ids=class_ids,
                train_loader=self._make_loader(train_ds, shuffle=True),
                val_loader=self._make_loader(val_ds, shuffle=False),
            ))

        logger.info("Split-CIFAR-100 构建完成")

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_n_classes_total(self) -> int:
        return self.N_CLASSES
