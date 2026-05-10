"""Split-MNIST 持续学习数据集。

将 MNIST 按类别分割为 n_tasks 个任务，每个任务包含 classes_per_task 个类别。
默认：5 任务，每任务 2 类（0-1, 2-3, 4-5, 6-7, 8-9）。
"""
import logging
from typing import Optional

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.data_module import register_dataset
from src.data_module.dataset.base_dataset import BaseContinualDataset, TaskData

logger = logging.getLogger(__name__)


class _FlattenView:
    """将 (C, H, W) tensor 展平为一维向量，替代 Lambda 以支持 Windows multiprocessing。"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


class _ClassSubset(torch.utils.data.Dataset):
    """从完整数据集中提取指定类别的子集。"""

    def __init__(self, dataset, class_ids: list[int]) -> None:
        self.dataset = dataset
        self.class_ids = set(class_ids)
        # 筛选出属于目标类别的样本索引
        self.indices = [
            i for i, label in enumerate(dataset.targets.tolist())
            if label in self.class_ids
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


@register_dataset("split_mnist")
class SplitMNIST(BaseContinualDataset):
    """Split-MNIST：将 MNIST 分割为多个二分类持续学习任务。"""

    INPUT_DIM = 784   # 28×28 展平
    N_CLASSES = 10

    def _setup(self) -> None:
        classes_per_task: int = getattr(self.cfg, "classes_per_task", 2)
        data_dir: str = getattr(self.cfg, "data_dir", "data/mnist")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            _FlattenView(),  # 展平为 784 维向量
        ])

        train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        val_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        logger.info(f"构建 Split-MNIST：{self.n_tasks} 个任务，每任务 {classes_per_task} 类")

        for task_id in range(self.n_tasks):
            start = task_id * classes_per_task
            class_ids = list(range(start, start + classes_per_task))

            train_subset = _ClassSubset(train_full, class_ids)
            val_subset = _ClassSubset(val_full, class_ids)

            self._tasks.append(TaskData(
                task_id=task_id,
                n_classes=classes_per_task,
                class_ids=class_ids,
                train_loader=self._make_loader(train_subset, shuffle=True),
                val_loader=self._make_loader(val_subset, shuffle=False),
            ))
            logger.info(
                f"  Task {task_id}: 类别 {class_ids}，"
                f"训练 {len(train_subset)} 样本，验证 {len(val_subset)} 样本"
            )

    def get_input_dim(self) -> int:
        return self.INPUT_DIM

    def get_n_classes_total(self) -> int:
        return self.N_CLASSES
