"""持续学习训练器：协调多任务序贯训练与评估。

训练流程（每个任务 t）：
    1. 获取任务 t 的 DataLoader
    2. 前向传播 + 分类损失 + 能量正则
    3. 反向传播更新分类头参数
    4. 完成任务后，回测所有 t' ≤ t 的任务（填充 R 矩阵）
    5. 记录 AA / BWT / FWT 到 WandB（可选）

注意：Hopfield 记忆矩阵通过在线 idf_update_step 更新，
不参与梯度反向传播（detach）。
"""
import logging
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.trainer_module.metrics import ContinualMetrics
from src.trainer_module.semantic_metrics import SemanticMetrics, compute_semantic_metrics

logger = logging.getLogger(__name__)


class ContinualTrainer:
    """持续学习训练器。

    Args:
        model: 模型实例（需实现 forward(x, update) 和 reset_for_task(task_id)）。
        cfg: Hydra 配置对象。
        device: 训练设备（"cuda" 或 "cpu"）。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        t = cfg.trainer
        self.n_epochs: int = t.n_epochs
        self.lr: float = t.lr
        self.weight_decay: float = getattr(t, "weight_decay", 1e-4)
        self.energy_lambda: float = getattr(t, "energy_lambda", 0.001)
        self.grad_clip: float = getattr(t, "grad_clip", 1.0)
        self.replay_size: int = getattr(t, "replay_size", 0)
        self.replay_lambda: float = getattr(t, "replay_lambda", 1.0)
        self.energy_priority: bool = (
            getattr(t, "replay_strategy", "random") == "energy_priority"
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.metrics: Optional[ContinualMetrics] = None
        self.semantic_metrics: Optional[SemanticMetrics] = None

        # WandB（可选）
        self._wandb = None
        if getattr(cfg.trainer, "use_wandb", False):
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb 未安装，跳过日志记录")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def fit(self, dataset) -> ContinualMetrics:
        """在整个持续学习数据集上按序训练所有任务。

        Args:
            dataset: BaseContinualDataset 实例，支持 __iter__ 和 __len__。

        Returns:
            填充完成的 ContinualMetrics 对象。
        """
        n_tasks = len(dataset)
        task_loaders = list(dataset)  # list of TaskData

        # 随机基线 b_i = 1 / n_classes_per_task（task-oracle 下的随机猜测准确率）
        n_classes_per_task = len(task_loaders[0].class_ids) if task_loaders else 1
        random_baseline = 1.0 / n_classes_per_task
        self.metrics = ContinualMetrics(n_tasks, random_baseline=random_baseline)
        logger.info(f"FWT 随机基线：1/{n_classes_per_task} = {random_baseline:.4f}")

        for task_idx, task_data in enumerate(task_loaders):
            logger.info(f"=== 开始训练 Task {task_idx}/{n_tasks - 1} ===")
            self.model.reset_for_task(task_idx)

            # 前向评估：训练任务 task_idx 之前，先评估该任务（填充 R[task_idx][task_idx-1]）
            if task_idx > 0:
                acc_pre = self._evaluate(task_data.val_loader, task_data.class_ids)
                self.metrics.update(task_idx, task_idx - 1, acc_pre)
                logger.info(
                    f"  Task {task_idx} 训练前准确率: {acc_pre:.4f}"
                    f"（FWT 贡献: {acc_pre - random_baseline:+.4f}）"
                )

            # 训练当前任务
            self._train_task(task_data.train_loader, task_idx)

            # EWC: 任务结束后估计 Fisher 信息并保存参数
            if hasattr(self.model, "consolidate"):
                self.model.consolidate(task_data.val_loader)

            # 回测所有已见任务（填充 R 矩阵对角线以下）
            for eval_idx in range(task_idx + 1):
                eval_data = task_loaders[eval_idx]
                acc = self._evaluate(eval_data.val_loader, eval_data.class_ids)
                self.metrics.update(eval_idx, task_idx, acc)
                logger.info(
                    f"  Task {eval_idx} acc after Task {task_idx}: {acc:.4f}"
                )

            # 记录当前指标
            summary = self.metrics.summary()
            logger.info(f"  当前指标: {summary}")
            if self._wandb:
                self._wandb.log({f"task_{task_idx}/{k}": v for k, v in summary.items()})

        # 语义指标（仅 wiki_facts 或 split_newsgroups 数据集，Hopfield 模型）
        dataset_name = getattr(self.cfg.dataset, "name", "")
        if dataset_name.startswith(("wiki_facts", "split_newsgroups")):
            all_val_loaders = [td.val_loader for td in task_loaders]
            # 计算训练总样本数，作为压缩比的分母（不依赖模型内部计数器）
            n_total_training = sum(
                len(td.train_loader.dataset) for td in task_loaders
            ) * self.n_epochs
            self.semantic_metrics = compute_semantic_metrics(
                self.model, all_val_loaders,
                device=torch.device(self.device),
                n_total_training=n_total_training,
            )
            if self.semantic_metrics is not None:
                logger.info(
                    f"语义指标："
                    f"semantic_similarity={self.semantic_metrics.semantic_similarity:.4f}，"
                    f"compression_ratio={self.semantic_metrics.compression_ratio:.4f}"
                )

        logger.info(f"训练完成：{self.metrics}")
        if hasattr(self.model, "save_diagnostics"):
            try:
                out_dir = Path(HydraConfig.get().runtime.output_dir)
            except ValueError:
                out_dir = Path("outputs") / "diagnostics"
            self.model.save_diagnostics(out_dir / "diagnostics.csv")
        return self.metrics

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _train_task(self, loader: DataLoader, task_idx: int) -> None:
        """训练单个任务 n_epochs 轮。"""
        self.model.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x, y in tqdm(
                loader,
                desc=f"Task {task_idx} Epoch {epoch + 1}/{self.n_epochs}",
                leave=False,
            ):
                if isinstance(x, (tuple, list)):
                    x = tuple(xi.to(self.device) for xi in x)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                logits, energy = self.model(x, update=True, labels=y)
                cls_loss = self.criterion(logits, y)
                energy_reg = self.energy_lambda * energy.mean()
                loss = cls_loss + energy_reg

                # Memory replay（ForgetGate 选择性保留的旧任务样本回放）
                if self.replay_size > 0 and hasattr(self.model, "sample_replay"):
                    replay = self.model.sample_replay(
                        self.replay_size, energy_priority=self.energy_priority
                    )
                    if replay is not None:
                        rx, ry = replay
                        rx, ry = rx.to(self.device), ry.to(self.device)
                        r_logits, _ = self.model(rx, update=False)
                        loss = loss + self.replay_lambda * self.criterion(r_logits, ry)

                # EWC 正则项（首个任务时 _fisher 为空，返回 0）
                if hasattr(self.model, "ewc_loss"):
                    loss = loss + self.model.ewc_loss()

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

                # DMHN 需在每步更新后强制对称性约束
                if hasattr(self.model, "enforce_constraints"):
                    self.model.enforce_constraints()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"  Task {task_idx} Epoch {epoch + 1} loss={avg_loss:.4f}")

    def _evaluate(
        self,
        loader: DataLoader,
        task_class_ids: list[int] | None = None,
    ) -> float:
        """在给定 DataLoader 上计算分类准确率。

        Args:
            loader: 验证集 DataLoader。
            task_class_ids: 当前任务的全局类别 ID 列表（如 [0, 1]）。
                提供时使用 task-oracle 评估：只在该任务的类别上做 argmax，
                防止跨任务类别干扰，正确衡量记忆保留能力。
                为 None 时退化为全类别 argmax（class-incremental 模式）。
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                if isinstance(x, (tuple, list)):
                    x = tuple(xi.to(self.device) for xi in x)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                logits, _ = self.model(x, update=False)

                if task_class_ids is not None:
                    # Task-oracle 评估：只在本任务的类别切片上取 argmax
                    task_logits = logits[:, task_class_ids]       # (B, K_task)
                    local_preds = task_logits.argmax(dim=-1)       # 0-based
                    preds = torch.tensor(
                        [task_class_ids[i] for i in local_preds.tolist()],
                        device=y.device,
                    )
                else:
                    preds = logits.argmax(dim=-1)

                correct += int((preds == y).sum().item())
                total += y.shape[0]

        return correct / max(total, 1)
