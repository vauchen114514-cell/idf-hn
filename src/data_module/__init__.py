"""数据模块：支持持续学习的数据集工厂。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from src.data_module.dataset.base_dataset import BaseContinualDataset

# Dataset 注册表（运行时 BaseContinualDataset 仅作为字符串注解，无循环依赖）
DATASET_REGISTRY: dict[str, Type[BaseContinualDataset]] = {}


def register_dataset(name: str):
    """注册数据集类的装饰器。"""
    def decorator(cls: Type[BaseContinualDataset]) -> Type[BaseContinualDataset]:
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def DatasetFactory(name: str) -> Type[BaseContinualDataset]:
    """根据名称返回数据集类。"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {name}。可用: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


__all__ = ["register_dataset", "DatasetFactory", "DATASET_REGISTRY"]
