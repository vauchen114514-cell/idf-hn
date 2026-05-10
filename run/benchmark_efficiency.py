"""IDF-HN 效率基准测试。

覆盖两个实验：
  1. 规模验证（N=500/1K/5K/10K）：per-sample retrieve+store 时间、内存占用
  2. O(N²) naive pairwise density  vs  Prototype O(K) density 速度对比

使用 D=512 合成特征（对应 CIFAR-100 ResNet-18 特征维度），
所有测试在 GPU 上运行（若可用），否则自动回退到 CPU。

运行：
    cd idf-hn
    uv run python run/benchmark_efficiency.py
    uv run python run/benchmark_efficiency.py --device cpu
"""
import argparse
import time
import gc
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor

# 导入注册表触发 @register_model 装饰器
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_module.hopfield.mhn_layer import ModernHopfieldLayer
from src.model_module.memory.prototype_bank import PrototypeBank
from src.model_module.memory.faiss_density_bank import FaissDensityBank, _FAISS_AVAILABLE

# -----------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------

N_VALUES   = [500, 1_000, 5_000, 10_000]
D          = 512      # CIFAR-100 ResNet-18 特征维度
K          = 50       # Prototype 数量（固定，不随 N 变化）
SIGMA      = 1.0      # RBF 核带宽
THETA      = 0.5      # naive pairwise 相似度阈值
N_WARMUP   = 10       # 预热次数（不计入统计）
N_REPEATS  = 50       # 正式计时次数


# -----------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------

def gpu_mem_mb(device: torch.device) -> float:
    """返回当前 GPU 已分配显存（MB）；CPU 模式返回 0。"""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6
    return 0.0


def timeit(fn, n_warmup: int, n_repeats: int, device: torch.device) -> float:
    """计时 fn()，返回平均耗时（毫秒）。"""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(n_repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000  # → ms
    return elapsed / n_repeats


# -----------------------------------------------------------------------
# 实验 1：规模验证 N=500/1K/5K/10K
# -----------------------------------------------------------------------

def benchmark_scale(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("实验 1：规模验证 — per-sample retrieve + store 时间 & 内存")
    print(f"{'N':>7} | {'fill(ms)':>9} | {'retrieve(ms)':>13} | {'store(ms)':>10} | {'mem(MB)':>8}")
    print("-" * 65)

    for N in N_VALUES:
        mhn = ModernHopfieldLayer(
            input_dim=D, beta=1.0, max_memories=N, eviction_policy="norm_min"
        ).to(device)

        # 预填充缓冲区：先写满 N 条
        fill_data = torch.randn(N, D, device=device)
        t_fill = timeit(
            lambda: [mhn.store(fill_data[i]) for i in range(min(32, N))],
            n_warmup=2, n_repeats=5, device=device
        ) / min(32, N)   # per-sample

        # 重新填满供后续测试
        mhn.reset_memory()
        for i in range(N):
            mhn.store(fill_data[i], label=i % 100)

        u = torch.randn(D, device=device)

        # retrieve 计时（buffer 已满，稳态）
        t_retrieve = timeit(
            lambda: mhn.retrieve(u, n_steps=1),
            n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=device
        )

        # store with norm_min 计时（buffer 已满，每次驱逐）
        t_store = timeit(
            lambda: mhn.store(u, label=0),
            n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=device
        )

        # 内存：两个缓冲区 _mem + _replay_buf，各 (N, D) float32 + label (N,) int64
        mem_mb = (
            2 * N * D * 4       # float32 特征
            + 2 * N * 8         # int64 标签
        ) / 1e6

        print(
            f"{N:>7,} | {t_fill:>9.3f} | {t_retrieve:>13.3f} | {t_store:>10.3f} | {mem_mb:>8.2f}"
        )

        del mhn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()


# -----------------------------------------------------------------------
# 实验 2：O(N²) naive pairwise density  vs  Prototype O(K) density
# -----------------------------------------------------------------------

def naive_pairwise_density(M: Tensor, theta: float = THETA) -> Tensor:
    """O(N²·D) 全量成对相似度密度（原始 Ω 公式）。"""
    M_norm = F.normalize(M, dim=-1)        # (N, D)
    sim_mat = M_norm @ M_norm.T            # (N, N) ← O(N²) 瓶颈
    return (sim_mat > theta).float().mean()


def prototype_density_fn(
    u: Tensor, centers: Tensor, counts: Tensor, sigma: float = SIGMA
) -> Tensor:
    """O(K·D) Prototype 密度（K 固定，与 N 无关）。"""
    diff = u.unsqueeze(0) - centers        # (K, D)
    dist_sq = (diff * diff).sum(-1)        # (K,)
    rbf = torch.exp(-dist_sq / (2.0 * sigma ** 2))
    weights = counts.float() / (counts.float().sum() + 1e-8)
    return (weights * rbf).sum()


def benchmark_density(device: torch.device) -> None:
    print("=" * 75)
    print("实验 2：O(N²) naive pairwise density  vs  Prototype O(K) density")
    print(f"Prototype 数量 K={K}（固定），D={D}")
    print()
    print(
        f"{'N':>7} | {'naive_Ω(ms)':>12} | {'proto_Ω(ms)':>12} | {'speedup':>9} | {'理论加速':>10}"
    )
    print("-" * 75)

    # 预构造 Prototype（与 N 无关，固定 K=50 个中心）
    centers = F.normalize(torch.randn(K, D, device=device), dim=-1)
    counts  = torch.ones(K, dtype=torch.long, device=device) * 100
    u = torch.randn(D, device=device)

    # Prototype 密度只测一次（与 N 无关）
    t_proto = timeit(
        lambda: prototype_density_fn(u, centers, counts),
        n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=device
    )

    for N in N_VALUES:
        M = torch.randn(N, D, device=device)

        t_naive = timeit(
            lambda: naive_pairwise_density(M),
            n_warmup=max(1, N_WARMUP // (N // 500)),
            n_repeats=max(5, N_REPEATS // (N // 500)),
            device=device
        )

        speedup = t_naive / t_proto
        theoretical = N ** 2 / K  # N²/K

        print(
            f"{N:>7,} | {t_naive:>12.3f} | {t_proto:>12.3f} | "
            f"{speedup:>9.1f}× | {theoretical:>10,.0f}×"
        )

        del M
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print(f"  Prototype 密度（K={K}）：{t_proto:.3f} ms（与 N 无关，O(1)）")
    print()


# -----------------------------------------------------------------------
# 实验 3：双缓冲区 IDF-HN per-sample 完整更新步耗时
# -----------------------------------------------------------------------

def benchmark_full_update_step(device: torch.device) -> None:
    """模拟 idf_update_step 的核心操作：retrieve + conflict + forget + store。"""
    print("=" * 65)
    print("实验 3：IDF-HN 完整 per-sample 更新步耗时（双缓冲区）")
    print(f"{'N':>7} | {'update_step(ms)':>16} | {'batch32(ms)':>12} | {'batch32(s/task)':>15}")
    print("-" * 65)

    SAMPLES_PER_TASK = 2500
    EPOCHS = 2

    for N in N_VALUES:
        mhn = ModernHopfieldLayer(
            input_dim=D, beta=1.0, max_memories=N, eviction_policy="norm_min"
        ).to(device)

        # 预填充
        for i in range(N):
            mhn.store(torch.randn(D, device=device), label=i % 100)

        u = torch.randn(D, device=device)

        def one_step():
            xi = mhn.retrieve(u, n_steps=1)
            e = mhn.energy(xi)
            mhn.forget(gamma=0.1)
            mhn.store(u, label=0)

        t_step = timeit(one_step, n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=device)
        t_batch32 = t_step * 32          # 一个 batch 的估计耗时（ms）
        n_batches = SAMPLES_PER_TASK * EPOCHS / 32
        t_task_s = t_batch32 * n_batches / 1000  # 一个任务的估计耗时（秒）

        print(
            f"{N:>7,} | {t_step:>16.3f} | {t_batch32:>12.1f} | {t_task_s:>15.1f}"
        )

        del mhn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()


# -----------------------------------------------------------------------
# 实验 4：FAISS ANN vs Prototype O(K) vs O(N) exact 密度查询速度对比
# -----------------------------------------------------------------------

N_VALUES_FAISS = [1_000, 5_000, 10_000, 50_000]


def benchmark_faiss(device: torch.device) -> None:
    """对比三种密度估计方法的查询速度。

    - Prototype O(K): K=50 聚类中心（与 N 无关，O(1)）
    - FAISS FlatL2:   精确 L2 检索（O(N·D) 但 FAISS 高度优化）
    - FAISS IVF:      近似检索（O(n_probe · n_cells · D)，适合大规模）
    """
    if not _FAISS_AVAILABLE:
        print("\n  FAISS 未安装，跳过实验 4（uv add faiss-cpu）")
        return

    print("=" * 85)
    print("实验 4：FAISS ANN vs Prototype O(K) vs O(N) exact 密度查询速度对比")
    print(f"查询维度 D={D}，k={K}（近邻数 / Prototype 数），重复 n={N_REPEATS}")
    print()
    print(
        f"{'N':>8} | {'Prototype(ms)':>14} | {'FAISS-Flat(ms)':>15} | "
        f"{'FAISS-IVF(ms)':>14} | {'Flat加速':>8} | {'IVF加速':>8}"
    )
    print("-" * 85)

    # Prototype 密度只测一次（与 N 无关）
    centers = F.normalize(torch.randn(K, D), dim=-1)
    counts = torch.ones(K, dtype=torch.long) * 100
    u_cpu = torch.randn(D)
    t_proto = timeit(
        lambda: prototype_density_fn(u_cpu, centers, counts),
        n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=torch.device("cpu")
    )

    for N in N_VALUES_FAISS:
        # 构建 FaissDensityBank（FlatL2，精确）
        bank_flat = FaissDensityBank(
            input_dim=D, n_prototypes=K, warmup_steps=0, use_ivf=False
        )
        # 构建 FaissDensityBank（IVF，近似）
        bank_ivf = FaissDensityBank(
            input_dim=D, n_prototypes=K, warmup_steps=0,
            use_ivf=True, n_lists=max(10, min(100, N // 100)),
        )

        # 预填充向量
        import numpy as np
        data = np.random.randn(N, D).astype(np.float32)
        for vec in data:
            bank_flat._vectors.append(vec)
            bank_ivf._vectors.append(vec)
        bank_flat._rebuild_index()
        bank_ivf._rebuild_index()

        u_tensor = torch.randn(D)

        t_flat = timeit(
            lambda: bank_flat.density(u_tensor),
            n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=torch.device("cpu")
        )
        t_ivf = timeit(
            lambda: bank_ivf.density(u_tensor),
            n_warmup=N_WARMUP, n_repeats=N_REPEATS, device=torch.device("cpu")
        )

        speedup_flat = t_flat / t_proto if t_proto > 0 else float("inf")
        speedup_ivf = t_ivf / t_proto if t_proto > 0 else float("inf")

        print(
            f"{N:>8,} | {t_proto:>14.3f} | {t_flat:>15.3f} | "
            f"{t_ivf:>14.3f} | {speedup_flat:>7.1f}× | {speedup_ivf:>7.1f}×"
        )

        del bank_flat, bank_ivf
        gc.collect()

    print()
    note = "注：Prototype 密度与 N 无关（O(1) 常数），FAISS 加速比 = FAISS耗时/Prototype耗时（<1 表示比 Prototype 慢）"
    print(f"  {note}")
    print()


# -----------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IDF-HN 效率基准测试")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--faiss-only", action="store_true", help="只运行 FAISS 对比实验")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nIDF-HN 效率基准测试")
    print(f"设备: {device}  |  特征维度 D={D}  |  计时重复 n={N_REPEATS}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    if args.faiss_only:
        benchmark_faiss(device)
    else:
        benchmark_scale(device)
        benchmark_density(device)
        benchmark_full_update_step(device)
        benchmark_faiss(device)

    print("=" * 65)
    print("基准测试完成。")


if __name__ == "__main__":
    main()
