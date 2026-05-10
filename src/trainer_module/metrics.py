"""持续学习评估指标：AA、BWT、FWT。

标准持续学习三指标定义（Lopez-Paz & Ranzato 2017）：
    - AA  (Average Accuracy)   ：所有任务最终平均准确率
    - BWT (Backward Transfer)  ：新任务对旧任务的负面影响
    - FWT (Forward Transfer)   ：旧任务对新任务的正面迁移

指标矩阵 R[i][j] = 在学完任务 j 后，任务 i 的准确率。
"""
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ContinualMetrics:
    """持续学习指标计算器。

    维护一个 n_tasks × n_tasks 的准确率矩阵 R，
    其中 R[i][j] = 训练完任务 j 后，在任务 i 上的准确率。

    Args:
        n_tasks: 总任务数。
    """

    def __init__(self, n_tasks: int, random_baseline: float = 0.0) -> None:
        self.n_tasks = n_tasks
        # 初始化为 -1 表示未测量
        self._R = torch.full((n_tasks, n_tasks), -1.0)
        # 随机基线（FWT 计算用）：通常取 1/n_classes_per_task
        self._random_baseline: float = random_baseline

    def update(self, task_i: int, after_task_j: int, accuracy: float) -> None:
        """记录一条准确率数据。

        Args:
            task_i: 评估的任务编号。
            after_task_j: 训练完哪个任务后做的评估。
            accuracy: 准确率 ∈ [0, 1]。
        """
        self._R[task_i][after_task_j] = accuracy

    def average_accuracy(self) -> float:
        """AA = (1/T) Σ_i R[i][T-1]（训练结束后所有任务准确率均值）。"""
        final_accs = [
            float(self._R[i][self.n_tasks - 1].item())
            for i in range(self.n_tasks)
            if self._R[i][self.n_tasks - 1] >= 0
        ]
        if not final_accs:
            return 0.0
        return sum(final_accs) / len(final_accs)

    def backward_transfer(self) -> float:
        """BWT = (1/(T-1)) Σ_{i=1}^{T-1} [R[i][T-1] - R[i][i]]。

        负值表示发生了遗忘，0 表示无遗忘，正值表示正向迁移。
        """
        if self.n_tasks < 2:
            return 0.0

        bwt_sum = 0.0
        count = 0
        for i in range(self.n_tasks - 1):
            r_final = float(self._R[i][self.n_tasks - 1].item())
            r_diag = float(self._R[i][i].item())
            if r_final >= 0 and r_diag >= 0:
                bwt_sum += r_final - r_diag
                count += 1

        return bwt_sum / max(count, 1)

    def forward_transfer(self, random_baseline: Optional[float] = None) -> float:
        """FWT = (1/(T-1)) Σ_{i=1}^{T-1} [R[i][i-1] - b_i]。

        R[i][i-1] = 学完任务 i-1 后（但未学任务 i 时）在任务 i 上的准确率。
        b_i = 随机基线（通常 1/n_classes_per_task）。

        Args:
            random_baseline: 覆盖初始化时的随机基线；None 时使用 self._random_baseline。
        """
        if self.n_tasks < 2:
            return 0.0

        b = random_baseline if random_baseline is not None else self._random_baseline
        fwt_sum = 0.0
        count = 0
        for i in range(1, self.n_tasks):
            r_pre = float(self._R[i][i - 1].item())
            if r_pre >= 0:
                fwt_sum += r_pre - b
                count += 1

        return fwt_sum / max(count, 1)

    def summary(self) -> dict[str, float]:
        """返回所有指标的汇总字典。"""
        return {
            "AA": self.average_accuracy(),
            "BWT": self.backward_transfer(),
            "FWT": self.forward_transfer(),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return f"ContinualMetrics(AA={s['AA']:.4f}, BWT={s['BWT']:.4f}, FWT={s['FWT']:.4f})"
