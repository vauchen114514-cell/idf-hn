# IDF-HN：输入依赖选择性遗忘 Hopfield 网络

**Input-Dependent Selective Forgetting Hopfield Network**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89b4fa)](https://hydra.cc/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目概述

IDF-HN 是一个面向持续学习的 Modern Hopfield Network 扩展框架，通过**输入依赖的选择性遗忘门**解决 MHN 在非平稳数据流中的灾难性遗忘问题。

### 核心问题

Modern Hopfield Networks（Ramsauer et al., ICLR 2021）具备指数级理论容量，但面对持续到来的新任务时，缺乏选择性遗忘机制，会逐渐覆盖旧记忆（catastrophic forgetting）。

### 方法

IDF-HN 扩展了 MHN 的能量函数，引入输入依赖的遗忘门 γ(u, t)：

$$E_{\text{IDF}}(\xi; \mathcal{M}) = -\text{lse}(\beta, \mathcal{M}^T \xi) + \tfrac{1}{2}\|\xi\|^2 + \lambda \cdot \Omega_{\text{eff}}(\mathcal{M}, u(t))$$

**遗忘门**由当前输入 u 与记忆的冲突度 ΔE 驱动：

$$\gamma(u, t) = \gamma_0 + \Delta\gamma \cdot \sigma\!\left(\Delta E(u, \mathcal{M}) - \tau\right)$$

**冲突度**（Decision 1：ΔE 能量差度量）：

$$\Delta E = E(\xi; \mathcal{M} \cup \{u\}) - E(\xi; \mathcal{M})$$

### 主要特性

| 特性 | 说明 |
|------|------|
| 输入依赖遗忘门 | 高冲突输入触发更强遗忘，低冲突输入保留旧记忆 |
| 自适应阈值 τ | 基于历史冲突分布的百分位数自动估计 τ，无需手动调参 |
| Prototype Clustering | Mini-Batch K-Means 将密度估计从 O(N²) 降至 O(K)，K=50 时实测加速 47.5× |
| 双缓冲区设计 | `_mem`（能量管理）与 `_replay_buf`（回放）完全分离，防止 ForgetGate 衰减污染回放样本 |
| norm_min 驱逐策略 | 缓冲区满时驱逐被 ForgetGate 衰减最多（范数最小）的记忆槽 |
| 统一持续学习接口 | 所有模型实现 `forward(x, update)` + `reset_for_task()` |

---

## 安装

### 依赖要求

- Python ≥ 3.11
- CUDA（单卡 8GB+ 显存，推荐 RTX 3080 以上）
- [uv](https://github.com/astral-sh/uv) 包管理器

### 步骤

```bash
# 克隆仓库
git clone <repo-url>
cd idf-hn

# 安装运行时依赖
uv sync

# 安装开发依赖（测试、linting）
uv sync --extra dev
```

### 主要依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | ≥ 2.2.0 | 深度学习框架 |
| torchvision | ≥ 0.17.0 | 数据集与预训练模型 |
| hydra-core | ≥ 1.3.2 | 配置管理 |
| faiss-cpu | ≥ 1.8.0 | 近似最近邻（可选加速） |
| scikit-learn | ≥ 1.4.0 | K-Means 聚类 |
| wandb | ≥ 0.16.0 | 实验追踪（可选） |

---

## 使用

### 快速开始

```bash
# 默认：IDF-HN on Split-MNIST
uv run python run/main.py

# 指定模型和数据集
uv run python run/main.py model=idf_hn dataset=split_cifar100

# 覆盖超参数
uv run python run/main.py model=idf_hn dataset=split_mnist \
    model.beta=2.0 model.forget_gate.tau=0.5

# 运行基线
uv run python run/main.py model=baselines/er dataset=split_mnist
uv run python run/main.py model=baselines/ewc dataset=split_mnist
uv run python run/main.py model=baselines/gss dataset=split_cifar100
```

### 多模型对比实验

```bash
# 在 Split-MNIST 上对比所有模型
for model in idf_hn baselines/er baselines/gss baselines/ewc baselines/classical_hn baselines/clipping_hn baselines/sparse_memory baselines/dmhn; do
    uv run python run/main.py model=$model dataset=split_mnist
done
```

### 输出结果

训练完成后，终端输出格式：

```
==================================================
最终结果 (idf_hn on split_mnist):
  AA  = 0.9838    # 平均准确率（↑ 越高越好）
  BWT = -0.0046   # 反向迁移（接近 0 越好）
  FWT =  0.0062   # 正向迁移（↑ 越高越好）
==================================================
```

实验输出保存至 `outputs/<run_name>/`（Hydra 自动管理）。

---

## 项目结构

```
idf-hn/
├── run/
│   ├── main.py                    # 实验入口
│   └── conf/                      # Hydra 配置
│       ├── config.yaml            # 主配置（模型 + 数据集 + 训练器）
│       ├── model/
│       │   ├── idf_hn.yaml        # IDF-HN 超参
│       │   └── baselines/         # ER、GSS、EWC、DMHN、Classical HN、Clipping HN、SparseMemory
│       ├── dataset/
│       │   ├── split_mnist.yaml
│       │   ├── split_cifar100.yaml
│       │   ├── permuted_mnist.yaml
│       │   ├── split_newsgroups.yaml
│       │   └── wiki_facts.yaml
│       └── trainer/
│           └── continual.yaml
│
├── src/
│   ├── data_module/               # 数据集注册表（Factory & Registry）
│   │   └── dataset/
│   │       ├── base_dataset.py    # 抽象基类 BaseContinualDataset
│   │       ├── split_mnist.py     # Split-MNIST（5 任务，每任务 2 类）
│   │       ├── permuted_mnist.py  # Permuted-MNIST（随机像素排列）
│   │       ├── split_cifar100.py  # Split-CIFAR-100（ResNet-18 特征缓存）
│   │       ├── split_newsgroups.py # Split-20Newsgroups（5 任务，TF-IDF/SVD 或 sentence embedding）
│   │       └── wiki_facts.py      # WikiFacts / DBpedia14（5 任务）
│   │
│   ├── model_module/              # 模型注册表
│   │   ├── hopfield/
│   │   │   ├── energy.py          # lse / mhn_energy / delta_energy
│   │   │   ├── base_hopfield.py   # 抽象基类 BaseHopfieldNetwork
│   │   │   └── mhn_layer.py       # ModernHopfieldLayer（双缓冲区：_mem + _replay_buf）
│   │   ├── forget_gate/
│   │   │   ├── conflict_detector.py  # ΔE 冲突检测 + 自适应 τ
│   │   │   └── forget_gate.py        # Sigmoid 调制遗忘率 + EMA 平滑
│   │   ├── memory/
│   │   │   ├── memory_bank.py     # 抽象基类 BaseMemoryBank
│   │   │   ├── prototype_bank.py  # Mini-Batch K-Means Prototype 库（O(K) 密度估计）
│   │   │   ├── exact_density_bank.py # 精确 RBF 密度（消融对照，O(N)）
│   │   │   └── faiss_density_bank.py # FAISS ANN 加速（可选，N>50K 时有优势）
│   │   ├── dreaming/
│   │   │   ├── semantic_filter.py    # 余弦相似度语义过滤
│   │   │   └── dreaming_scheduler.py # 周期性 Dreaming 触发（默认禁用）
│   │   ├── idf_hn/
│   │   │   ├── idf_hopfield.py    # IDFHopfieldNetwork（主模型）
│   │   │   └── update_rule.py     # idf_update_step（在线增量更新）
│   │   └── baselines/
│   │       ├── classical_hn.py    # 无遗忘 MHN（FIFO 淘汰）
│   │       ├── clipping_hn.py     # Norm Clipping + Dreaming（Marinari et al., 2026）
│   │       ├── dmhn.py            # Dynamic Manifold HN（Li et al., 2025）
│   │       ├── er.py              # Experience Replay（Reservoir Sampling）
│   │       ├── ewc.py             # Elastic Weight Consolidation
│   │       ├── gss.py             # Gradient-based Sample Selection
│   │       └── sparse_memory.py   # Hopfield + 常数全局衰减（消融对照）
│   │
│   ├── trainer_module/
│   │   ├── continual_trainer.py   # 多任务序贯训练 + 回测
│   │   ├── metrics.py             # AA / BWT / FWT 指标矩阵
│   │   └── semantic_metrics.py    # SemanticSim / CompRatio（文本数据集专用）
│   └── utils/
│       ├── seed.py                # 随机种子（实验可复现）
│       └── logging_utils.py       # 日志配置
│
└── tests/
    ├── test_energy.py             # 能量函数数学性质验证
    ├── test_mhn_layer.py          # ModernHopfieldLayer 单元测试
    ├── test_forget_gate.py        # 遗忘门单元测试
    ├── test_metrics.py            # 持续学习指标单元测试
    ├── test_dmhn.py               # DMHN 基线单元测试
    ├── test_er.py                 # ER 基线单元测试
    └── test_gss.py                # GSS 基线单元测试
```

---

## 模型配置

### IDF-HN 关键超参（`run/conf/model/idf_hn.yaml`）

```yaml
beta: 1.0                    # 逆温度参数（检索锐度）
memory_size: 1000            # 记忆矩阵最大容量 N

forget_gate:
  gamma_0: 0.1               # 基础遗忘率下界（0.01 时 norm_min 驱逐等价于随机，需 ≥ 0.1）
  delta_gamma: 0.5           # 最大额外遗忘率
  tau: 0.3                   # 冲突阈值（自适应模式下仅作回退值）
  adaptive_tau: true         # 启用自适应 τ
  tau_percentile: 75         # 历史冲突分布的第 75 百分位
  ema_alpha: 0.9             # EMA 平滑系数（有效窗口 ≈ 10 步）

memory_bank:
  type: prototype            # prototype（默认）或 exact（消融用）
  n_prototypes: 50           # Prototype 簇数量 K
  warmup_steps: 50           # 前 50 步收集样本，完成后初始化 K-Means

dreaming:
  enabled: false             # 默认禁用（RNG 修复后效果中性，禁用更稳定）
  freq: 500                  # 每 500 步触发一次（仅 enabled=true 时有效）
  n_dream_samples: 50        # 每次随机采样的候选记忆数

eviction_policy: norm_min    # 缓冲区满时驱逐范数最小的槽位
write_threshold: -0.1        # ΔE > -0.1 时才写入（过滤冗余同任务样本）
forget_mode: input_dependent # 遗忘机制（消融可选 time_decay / static_density / none）

lambda_omega: 0.01           # Ω_eff 正则化系数
```

### 数据集配置

| 数据集 | 任务数 | 每任务类别 | 输入维度 | 显存 |
|--------|--------|-----------|---------|------|
| split_mnist | 5 | 2 | 784 | < 1 GB |
| permuted_mnist | 10 | 10（全） | 784 | < 1 GB |
| split_cifar100 | 20 | 5 | 512（ResNet-18 特征） | ~2 GB |
| split_newsgroups | 5 | 4 | 512（TF-IDF/SVD）或 768（sentence embedding） | < 1 GB |
| wiki_facts | 5 | 3 | 512（TF-IDF/SVD）或 768（sentence embedding） | < 1 GB |

> **注意**：Split-CIFAR-100 首次运行时会自动提取并缓存 ResNet-18 特征（约 2 分钟），后续运行直接加载缓存。

---

## 基线模型

| 模型 | 范式 | 遗忘机制 | 参考文献 |
|------|------|---------|---------|
| `classical_hn` | 显式记忆 | 无（FIFO 淘汰） | Ramsauer et al., 2021 |
| `clipping_hn` | 显式记忆 | Norm Clipping + 随机 Dreaming | Marinari et al., 2026 |
| `dmhn` | 动态流形 | 无（梯度遗忘） | Li et al., arXiv:2506.01303 |
| `er` | 显式记忆 | Reservoir Replay | — |
| `ewc` | 参数保护 | Fisher 信息矩阵 | Kirkpatrick et al., 2017 |
| `gss` | 显式记忆 | 梯度多样性驱动缓冲区选择 | Aljundi et al., 2019 |
| `sparse_memory` | 显式记忆 | 常数全局衰减（消融对照） | — |
| **`idf_hn`** | **显式记忆** | **输入依赖选择性遗忘** | **本工作** |

> EWC 与显式记忆类模型属于不同范式（参数保护 vs 显式记忆），实验中作参考基线。

---

## 实验结果

所有结果：3 seeds（42/123/456）mean ± std，n_epochs=2。

### Split-MNIST（5 任务，memory_size=1000）

| 模型 | AA (mean ± std) | BWT (mean ± std) |
|------|----------------|-----------------|
| **IDF-HN** | **0.9838 ± 0.0026** | **-0.0046 ± 0.0033** |
| ER | 0.9827 ± 0.0020 | -0.0052 ± 0.0029 |
| SparseMemory | 0.9833 ± 0.0010 | -0.0049 ± 0.0008 |
| GSS | 0.9750 ± 0.0044 | -0.0160 ± 0.0062 |
| classical_hn | 0.8886 ± 0.0178 | -0.1173 ± 0.0219 |
| EWC | 0.8749 ± 0.0161 | -0.1508 ± 0.0204 |

### Split-CIFAR-100（20 任务，memory_size=5000）

| 模型 | AA (mean ± std) | BWT (mean ± std) |
|------|----------------|-----------------|
| ER | 0.9471 ± 0.0009 | -0.0065 ± 0.0021 |
| GSS | 0.9417 ± 0.0036 | -0.0102 ± 0.0054 |
| **IDF-HN** | **0.9325 ± 0.0024** | **-0.0191 ± 0.0031** |
| SparseMemory | 0.9305 ± 0.0023 | -0.0218 ± 0.0006 |
| EWC | 0.6811 ± 0.0167 | -0.2719 ± 0.0201 |
| classical_hn | 0.4246 ± 0.0019 | -0.2680 ± 0.0038 |
| DMHN | 0.3849 ± 0.0120 | -0.4230 ± 0.0191 |

### Permuted-MNIST（10 任务，memory_size=1000）

| 模型 | AA (mean ± std) | BWT (mean ± std) |
|------|----------------|-----------------|
| ER | 0.7691 ± 0.0022 | -0.1557 ± 0.0031 |
| **IDF-HN** | **0.7637 ± 0.0046** | **-0.1629 ± 0.0059** |
| EWC | 0.7478 ± 0.0101 | -0.2515 ± 0.0106 |
| GSS | 0.5814 ± 0.0074 | -0.3669 ± 0.0080 |
| DMHN | 0.4064 ± 0.0448 | -0.4731 ± 0.0607 |
| classical_hn | 0.2143 ± 0.0078 | -0.6405 ± 0.0033 |

---

## 评估指标

持续学习三指标（Lopez-Paz & Ranzato, 2017），基于准确率矩阵 R[i][j]（训练完任务 j 后在任务 i 上的准确率）：

| 指标 | 公式 | 含义 |
|------|------|------|
| **AA** | (1/T) Σ R[i][T-1] | 最终所有任务平均准确率（↑） |
| **BWT** | (1/(T-1)) Σ (R[i][T-1] - R[i][i]) | 新任务对旧任务的负面影响（接近 0） |
| **FWT** | (1/(T-1)) Σ (R[i][i-1] - b_i) | 旧任务对新任务的正迁移（↑） |

---

## 开发

### 运行测试

```bash
# 全部测试（含覆盖率报告）
uv run pytest

# 单独运行某模块测试
uv run pytest tests/test_energy.py -v

# 跳过慢速测试
uv run pytest -m "not slow"
```

### 代码检查

```bash
# Linting
uv run ruff check src/ tests/

# 类型检查
uv run mypy src/
```

### 添加新数据集

1. 继承 `BaseContinualDataset`，实现 `_setup()`、`get_input_dim()`、`get_n_classes_total()`
2. 用 `@register_dataset("your_name")` 装饰类
3. 在 `src/data_module/dataset/__init__.py` 中导入
4. 创建 `run/conf/dataset/your_name.yaml`

### 添加新模型

1. 继承 `nn.Module`，实现 `forward(x, update)` 和 `reset_for_task(task_id)`
2. 用 `@register_model("your_name")` 装饰类
3. 在 `src/model_module/baselines/__init__.py` 中导入
4. 创建 `run/conf/model/baselines/your_name.yaml`

---

## 论文参考

- **IDF-HN（本工作）**：投稿目标 NeurIPS 2026
- Ramsauer et al., *Hopfield Networks is All You Need*, ICLR 2021
- Kirkpatrick et al., *Overcoming Catastrophic Forgetting in NNs*, PNAS 2017
- Aljundi et al., *Gradient Based Sample Selection for Online Continual Learning*, NeurIPS 2019
- Li et al., *Dynamic Manifold Hopfield Networks*, arXiv:2506.01303, 2025
- Marinari et al., *Dreaming in Hopfield Networks*, 2026
- Santos et al., *Hopfield-Fenchel-Young Networks*, arXiv:2411.08590, 2024

---

## 许可证

MIT License
