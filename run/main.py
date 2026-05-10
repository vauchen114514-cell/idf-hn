"""IDF-HN 实验主入口。

使用 Hydra 管理配置，支持命令行覆盖：
    uv run python run/main.py model=idf_hn dataset=split_mnist
    uv run python run/main.py model=ewc dataset=split_cifar100 trainer.n_epochs=5
"""
import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# 导入注册表，触发所有 @register_dataset / @register_model 装饰器
import src.data_module.dataset  # noqa: F401
import src.model_module.baselines  # noqa: F401
import src.model_module.idf_hn  # noqa: F401
import src.model_module.kv_cache  # noqa: F401
from src.data_module import DatasetFactory
from src.model_module import ModelFactory
from src.trainer_module import ContinualTrainer
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """实验主函数。

    Args:
        cfg: Hydra 解析后的配置对象。
    """
    # 基础设置
    setup_logging(log_level=getattr(cfg, "log_level", "INFO"))
    set_seed(cfg.seed)

    logger.info(f"实验配置：\n{OmegaConf.to_yaml(cfg)}")

    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备：{device}")
    if device == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # 数据集（传入 cfg.dataset，BaseContinualDataset 直接读取 n_tasks/batch_size 等字段）
    DatasetClass = DatasetFactory(cfg.dataset.name)
    dataset = DatasetClass(cfg.dataset)
    input_dim = dataset.get_input_dim()
    n_classes = dataset.get_n_classes_total()
    logger.info(f"数据集：{cfg.dataset.name}, input_dim={input_dim}, n_classes={n_classes}")

    # 模型
    ModelClass = ModelFactory(cfg.model.name)
    model = ModelClass(cfg=cfg, input_dim=input_dim, n_classes=n_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型：{cfg.model.name}, 可训练参数={n_params:,}")

    # 训练
    trainer = ContinualTrainer(model=model, cfg=cfg, device=device)
    metrics = trainer.fit(dataset)

    # 输出最终结果
    summary = metrics.summary()
    logger.info("=" * 50)
    logger.info(f"最终结果 ({cfg.model.name} on {cfg.dataset.name}):")
    logger.info(f"  AA  = {summary['AA']:.4f}")
    logger.info(f"  BWT = {summary['BWT']:.4f}")
    logger.info(f"  FWT = {summary['FWT']:.4f}")
    logger.info("=" * 50)

    # 保存结果到 metrics.json（供 collect_results.py 读取）
    result_dict: dict = {
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "aa": summary["AA"],
        "bwt": summary["BWT"],
        "fwt": summary["FWT"],
        "replay_strategy": getattr(cfg.trainer, "replay_strategy", "random"),
        "feature_type": getattr(cfg.dataset, "feature_type", "tfidf_svd"),
        "n_epochs": cfg.trainer.n_epochs,
        "trainer": getattr(cfg.trainer, "name", "continual"),
        "forget_mode": getattr(cfg.model, "forget_mode", None),
        "eviction_policy": getattr(cfg.model, "eviction_policy", None),
    }
    # 追加语义指标（wiki_facts / split_newsgroups）
    if trainer.semantic_metrics is not None:
        sm = trainer.semantic_metrics
        result_dict["semantic_similarity"] = sm.semantic_similarity
        result_dict["compression_ratio"] = sm.compression_ratio
        result_dict["n_stored"] = sm.n_stored
        result_dict["n_total_seen"] = sm.n_total_seen
        logger.info(
            f"  SemanticSim  = {sm.semantic_similarity:.4f}"
        )
        logger.info(
            f"  CompressRatio= {sm.compression_ratio:.4f}"
        )

    # 保存到 Hydra 输出目录（不依赖 chdir）
    out_path = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"结果已保存至：{out_path.resolve()}")


if __name__ == "__main__":
    main()
