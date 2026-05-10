"""Permuted-MNIST 非平稳持续学习数据集。

每个任务对 MNIST 像素施加不同的随机排列（permutation），模拟分布漂移。
"""
import logging

import torch
from torchvision import datasets, transforms

from src.data_module import register_dataset
from src.data_module.dataset.base_dataset import BaseContinualDataset, TaskData

logger = logging.getLogger(__name__)


class _FlattenView:
    """将 (C, H, W) tensor 展平为一维向量，替代 Lambda 以支持 Windows multiprocessing。"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


class _PermutedDataset(torch.utils.data.Dataset):
    """对图像像素施加固定随机排列的数据集包装器。"""

    def __init__(self, dataset, permutation: torch.Tensor) -> None:
        self.dataset = dataset
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return x[self.permutation], y


@register_dataset("permuted_mnist")
class PermutedMNIST(BaseContinualDataset):
    """Permuted-MNIST：每个任务使用不同的像素排列。"""

    INPUT_DIM = 784
    N_CLASSES = 10

    def _setup(self) -> None:
        data_dir: str = getattr(self.cfg, "data_dir", "data/mnist")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            _FlattenView(),  # 展平为 784 维向量
        ])

        train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        val_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        logger.info(f"构建 Permuted-MNIST：{self.n_tasks} 个任务")

        # Task 0 使用原始排列（identity permutation）
        permutations = [torch.arange(self.INPUT_DIM)]
        for _ in range(self.n_tasks - 1):
            permutations.append(torch.randperm(self.INPUT_DIM))

        for task_id, perm in enumerate(permutations):
            train_ds = _PermutedDataset(train_full, perm)
            val_ds = _PermutedDataset(val_full, perm)

            self._tasks.append(TaskData(
                task_id=task_id,
                n_classes=self.N_CLASSES,
                class_ids=list(range(self.N_CLASSES)),
                train_loader=self._make_loader(train_ds, shuffle=True),
                val_loader=self._make_loader(val_ds, shuffle=False),
            ))

        logger.info(f"  Permuted-MNIST 构建完成，共 {self.n_tasks} 个任务")

    def get_input_dim(self) -> int:
        return self.INPUT_DIM

    def get_n_classes_total(self) -> int:
        return self.N_CLASSES
