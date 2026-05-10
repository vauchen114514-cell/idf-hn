"""持续学习数据集的抽象基类。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TaskData:
    """单个持续学习任务的数据容器。"""
    task_id: int
    n_classes: int
    class_ids: list[int]
    train_loader: DataLoader
    val_loader: DataLoader


class BaseContinualDataset(ABC):
    """持续学习数据集基类。

    所有持续学习数据集必须继承此类，实现任务序列的构建逻辑。
    每个任务包含一个训练集和验证集的 DataLoader。
    """

    def __init__(self, cfg) -> None:
        """初始化数据集。

        Args:
            cfg: Hydra/OmegaConf 配置对象，包含 n_tasks、batch_size 等字段。
        """
        self.cfg = cfg
        self.n_tasks: int = cfg.n_tasks
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers
        self._tasks: list[TaskData] = []
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """构建所有任务的数据集，填充 self._tasks。"""
        ...

    @abstractmethod
    def get_input_dim(self) -> int:
        """返回单个样本的展平后输入维度。"""
        ...

    @abstractmethod
    def get_n_classes_total(self) -> int:
        """返回所有任务的总类别数。"""
        ...

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[TaskData]:
        return iter(self._tasks)

    def __getitem__(self, task_id: int) -> TaskData:
        if task_id >= self.n_tasks:
            raise IndexError(f"task_id {task_id} 超出范围（共 {self.n_tasks} 个任务）")
        return self._tasks[task_id]

    def _make_loader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """创建标准 DataLoader。"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
