"""模型模块：Hopfield 网络族的工厂与注册表。"""
from typing import Type

import torch.nn as nn

# Model 注册表
MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """注册模型类的装饰器。"""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def ModelFactory(name: str) -> Type[nn.Module]:
    """根据名称返回模型类。"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}。可用: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


__all__ = ["register_model", "ModelFactory", "MODEL_REGISTRY"]
