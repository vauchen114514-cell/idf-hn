"""数据集子模块：自动注册所有数据集。"""
from src.data_module.dataset.base_dataset import BaseContinualDataset, TaskData
from src.data_module.dataset.split_mnist import SplitMNIST
from src.data_module.dataset.permuted_mnist import PermutedMNIST
from src.data_module.dataset.split_cifar100 import SplitCIFAR100
from src.data_module.dataset.split_newsgroups import SplitNewsgroups
from src.data_module.dataset.wiki_facts import WikiFactsDataset
from src.data_module.dataset.split_newsgroups_text import SplitNewsgroupsText
from src.data_module.dataset.wiki_facts_text import WikiFactsText

__all__ = [
    "BaseContinualDataset",
    "TaskData",
    "SplitMNIST",
    "PermutedMNIST",
    "SplitCIFAR100",
    "SplitNewsgroups",
    "WikiFactsDataset",
    "SplitNewsgroupsText",
    "WikiFactsText",
]
