"""补做实验批量运行脚本（proposal 缺口实验）。

覆盖三个缺口：
  1. Classical HN + Clipping（Marinari 2026）：3 个主数据集 × 3 seeds = 9 次
  2. WikiFacts 全模型对比：7 模型 × 3 seeds = 21 次
  3. FAISS ANN 效率消融：1 次（无 seed，benchmark 脚本）

用法：
    cd idf-hn
    uv run python run/run_supplement_experiments.py               # 全部运行
    uv run python run/run_supplement_experiments.py --dry-run     # 预览命令
    uv run python run/run_supplement_experiments.py --group clip  # 只跑 clipping_hn
    uv run python run/run_supplement_experiments.py --group wiki  # 只跑 wiki_facts
    uv run python run/run_supplement_experiments.py --group faiss # 只跑 FAISS benchmark
    uv run python run/run_supplement_experiments.py --start 5     # 断点续跑
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]
PROJECT_ROOT = Path(__file__).parent.parent

# (label, model, dataset, overrides)
SUPPLEMENT_GROUPS: list[tuple[str, str, str, list[str]]] = [
    # ── 1. Classical HN + Clipping（3 数据集）──
    (
        "clip/split_mnist",
        "baselines/clipping_hn", "split_mnist",
        ["trainer.n_epochs=2"],
    ),
    (
        "clip/split_cifar100",
        "baselines/clipping_hn", "split_cifar100",
        ["trainer.n_epochs=2", "model.memory_size=5000"],
    ),
    (
        "clip/permuted_mnist",
        "baselines/clipping_hn", "permuted_mnist",
        ["trainer.n_epochs=2"],
    ),

    # ── 2. WikiFacts 全模型对比（7 模型）──
    (
        "wiki/idf_hn",
        "idf_hn", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
    (
        "wiki/er",
        "baselines/er", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
    (
        "wiki/gss",
        "baselines/gss", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
    (
        "wiki/sparse_memory",
        "baselines/sparse_memory", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
    (
        "wiki/classical_hn",
        "baselines/classical_hn", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
    (
        "wiki/ewc",
        "baselines/ewc", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
    (
        "wiki/clipping_hn",
        "baselines/clipping_hn", "wiki_facts",
        ["trainer.n_epochs=5"],
    ),
]

# 展开 seed（仅主实验组需要多 seed）
EXPERIMENTS: list[tuple[str, str, str, int, list[str]]] = []
for label, model, dataset, overrides in SUPPLEMENT_GROUPS:
    for seed in SEEDS:
        EXPERIMENTS.append((label, model, dataset, seed, overrides))

# FAISS benchmark 作为单独项（无 seed）
FAISS_CMD = [
    "uv", "run", "python", "run/benchmark_efficiency.py", "--faiss-only"
]


def build_cmd(model: str, dataset: str, seed: int, overrides: list[str]) -> list[str]:
    return [
        "uv", "run", "python", "run/main.py",
        f"model={model}",
        f"dataset={dataset}",
        f"seed={seed}",
    ] + overrides


def run_experiment(
    idx: int, total: int,
    label: str, model: str, dataset: str, seed: int, overrides: list[str],
    dry_run: bool,
) -> bool:
    cmd = build_cmd(model, dataset, seed, overrides)
    print(f"\n{'='*70}")
    print(f"  [{idx}/{total}] {label} | seed={seed}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode}  elapsed={elapsed:.0f}s")
        return False
    print(f"  [OK] elapsed={elapsed:.0f}s")
    return True


def run_faiss_benchmark(dry_run: bool) -> bool:
    print(f"\n{'='*70}")
    print("  [FAISS] FAISS ANN 效率基准测试")
    print(f"  CMD: {' '.join(FAISS_CMD)}")
    print(f"{'='*70}")

    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(FAISS_CMD, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode}  elapsed={elapsed:.0f}s")
        return False
    print(f"  [OK] elapsed={elapsed:.0f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="补做实验批量运行")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=1, metavar="N")
    parser.add_argument(
        "--group", type=str, default=None,
        help="只运行标签包含指定字符串的组（clip / wiki / faiss）"
    )
    args = parser.parse_args()

    exps = EXPERIMENTS
    run_faiss = True

    if args.group:
        if args.group == "faiss":
            exps = []
        else:
            exps = [(l, m, d, s, o) for l, m, d, s, o in exps if args.group in l]
            run_faiss = False

    total = len(exps) + (1 if run_faiss else 0)
    print(f"\n补做实验共 {total} 项（筛选: {args.group or '全部'}），从第 {args.start} 项开始")
    if args.dry_run:
        print("（DRY-RUN 模式）\n")

    failed: list[int] = []
    for i, (label, model, dataset, seed, overrides) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_experiment(
            i, total, label, model, dataset, seed, overrides, args.dry_run
        )
        if not ok:
            failed.append(i)
            print(f"  [WARN] 实验 {i} 失败，继续执行...")

    # FAISS benchmark（最后一项）
    if run_faiss:
        faiss_idx = len(exps) + 1
        if faiss_idx >= args.start:
            ok = run_faiss_benchmark(args.dry_run)
            if not ok:
                failed.append(faiss_idx)

    print(f"\n{'='*70}")
    print(f"完成：{total - len(failed)}/{total} 成功")
    if failed:
        print(f"失败序号：{failed}")
        print(f"可用 --start {min(failed)} 断点续跑")
        sys.exit(1)
    else:
        print("所有补做实验成功！")


if __name__ == "__main__":
    main()
