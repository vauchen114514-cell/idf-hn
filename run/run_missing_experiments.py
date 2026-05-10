"""补全缺失实验的批量运行脚本。

需补实验（共 24 次）：
  Split-MNIST    : ER, GSS(重跑对齐 mem=1000), IDF-HN(重跑双缓冲)
  Permuted-MNIST : ER, GSS, DMHN, IDF-HN(重跑双缓冲)
  Split-CIFAR-100: GSS(重跑 n_epochs=2 对齐其他模型)

所有 MNIST 系列使用 n_epochs=2, memory_size=1000（与 CIFAR-100 实验对齐）。

用法：
    cd idf-hn
    uv run python run/run_missing_experiments.py
    uv run python run/run_missing_experiments.py --dry-run   # 仅打印命令不执行
    uv run python run/run_missing_experiments.py --start 5   # 从第5个实验开始（断点续跑）
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 实验定义
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]

# 每组实验：(model_override, dataset, 额外override列表)
# model_override 对应 run/conf/model/ 下的路径
EXPERIMENT_GROUPS = [
    # Split-MNIST：补 ER
    ("baselines/er",  "split_mnist",    ["model.memory_size=1000", "trainer.n_epochs=2"]),
    # Split-MNIST：GSS 重跑（对齐 memory_size=1000, n_epochs=2）
    ("baselines/gss", "split_mnist",    ["model.memory_size=1000", "trainer.n_epochs=2"]),
    # Split-MNIST：IDF-HN 重跑（双缓冲区最终设计）
    ("idf_hn",        "split_mnist",    ["trainer.n_epochs=2"]),
    # Permuted-MNIST：补 ER
    ("baselines/er",  "permuted_mnist", ["model.memory_size=1000", "trainer.n_epochs=2"]),
    # Permuted-MNIST：补 GSS
    ("baselines/gss", "permuted_mnist", ["model.memory_size=1000", "trainer.n_epochs=2"]),
    # Permuted-MNIST：补 DMHN
    ("baselines/dmhn","permuted_mnist", ["trainer.n_epochs=2"]),
    # Permuted-MNIST：IDF-HN 重跑（双缓冲区最终设计）
    ("idf_hn",        "permuted_mnist", ["trainer.n_epochs=2"]),
    # Split-CIFAR-100：GSS 重跑（n_epochs=2 对齐所有其他模型）
    ("baselines/gss", "split_cifar100", ["trainer.n_epochs=2"]),
]

# 展开为每条独立实验（model, dataset, seed, overrides）
EXPERIMENTS: list[tuple[str, str, int, list[str]]] = []
for model, dataset, overrides in EXPERIMENT_GROUPS:
    for seed in SEEDS:
        EXPERIMENTS.append((model, dataset, seed, overrides))


# ---------------------------------------------------------------------------
# 运行逻辑
# ---------------------------------------------------------------------------

def build_cmd(model: str, dataset: str, seed: int, overrides: list[str]) -> list[str]:
    cmd = [
        "uv", "run", "python", "run/main.py",
        f"model={model}",
        f"dataset={dataset}",
        f"seed={seed}",
    ] + overrides
    return cmd


def run_experiment(
    idx: int,
    total: int,
    model: str,
    dataset: str,
    seed: int,
    overrides: list[str],
    dry_run: bool,
) -> bool:
    cmd = build_cmd(model, dataset, seed, overrides)
    label = f"[{idx}/{total}] {model} | {dataset} | seed={seed}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ 失败（returncode={result.returncode}，耗时={elapsed:.0f}s）")
        return False
    else:
        print(f"  ✅ 完成（耗时={elapsed:.0f}s）")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="批量补全缺失实验")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际运行")
    parser.add_argument("--start", type=int, default=1, metavar="N",
                        help="从第 N 个实验开始（1-indexed，断点续跑）")
    args = parser.parse_args()

    total = len(EXPERIMENTS)
    print(f"共 {total} 个实验，从第 {args.start} 个开始")

    if args.dry_run:
        print("（DRY-RUN 模式，不实际执行）\n")

    failed: list[int] = []
    for i, (model, dataset, seed, overrides) in enumerate(EXPERIMENTS, start=1):
        if i < args.start:
            continue
        ok = run_experiment(i, total, model, dataset, seed, overrides, args.dry_run)
        if not ok:
            failed.append(i)
            print(f"  ⚠️  实验 {i} 失败，继续执行后续实验...")

    print(f"\n{'='*60}")
    print(f"全部完成：{total - len(failed)}/{total} 成功")
    if failed:
        print(f"失败的实验序号：{failed}")
        print(f"可用 --start {min(failed)} 从失败处重新开始")
        sys.exit(1)
    else:
        print("所有实验成功！运行 collect_results.py 查看汇总结果。")


if __name__ == "__main__":
    main()
