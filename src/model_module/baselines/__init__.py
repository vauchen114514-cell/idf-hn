"""基线模型子模块（自动注册到 MODEL_REGISTRY）。"""
from src.model_module.baselines.classical_hn import ClassicalHopfieldBaseline
from src.model_module.baselines.clipping_hn import ClippingHNBaseline
from src.model_module.baselines.distilbert_er import DistilBertER
from src.model_module.baselines.distilbert_finetune import DistilBertFinetune
from src.model_module.baselines.dmhn import DMHNBaseline
from src.model_module.baselines.er import ERBaseline
from src.model_module.baselines.ewc import EWCModel
from src.model_module.baselines.gss import GSSBaseline
from src.model_module.baselines.sparse_memory import SparseMemoryBaseline

__all__ = [
    "ClassicalHopfieldBaseline",
    "ClippingHNBaseline",
    "DistilBertER",
    "DistilBertFinetune",
    "EWCModel",
    "DMHNBaseline",
    "ERBaseline",
    "GSSBaseline",
    "SparseMemoryBaseline",
]
