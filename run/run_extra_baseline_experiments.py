"""额外基线与 Newsgroups 实验批量运行脚本。

实验内容：
  1. SparseMemory 基线（3 数据集 × 3 seeds = 9 次）
     - split_mnist, split_cifar100, permuted_mnist
  2. Split-20Newsgroups 全模型对比（7 模型 × 3 seeds = 21 次）
     - idf_hn, er, gss, ewc, classical_hn, dmhn, sparse_memory
  3. ForgetGate ON/OFF 多 seed 补跑（CIFAR-100，seeds 123+456 = 2 次）
     - ForgetGate OFF (gamma_0=0, delta_gamma=0)

总计：9 + 21 + 2 = 32 次运行

用法：
    cd idf-hn
    uv run python run/run_extra_baseline_experiments.py
    uv run python run/run_extra_baseline_experiments.py --dry-run
    uv run python run/run_extra_baseline_experiments.py --section sparse
    uv run python run/run_extra_baseline_experiments.py --section newsgroups
    uv run python run/run_extra_baseline_experiments.py --section forgegate
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

# ── Section 1: SparseMemory 基线（3 数据集） ──
SPARSE_MEMORY_GROUPS: list[tuple[str, str, list[str]]] = [
    ("baselines/sparse_memory", "split_mnist",    ["trainer.n_epochs=2"]),
    ("baselines/sparse_memory", "split_cifar100", ["trainer.n_epochs=2"]),
    ("baselines/sparse_memory", "permuted_mnist", ["trainer.n_epochs=2"]),
]

# ── Section 2: Split-20Newsgroups 全模型对比 ──
NEWSGROUPS_MODELS: list[tuple[str, list[str]]] = [
    ("idf_hn",            ["trainer.n_epochs=5"]),
    ("baselines/er",      ["model.memory_size=1000", "trainer.n_epochs=5"]),
    ("baselines/gss",     ["model.memory_size=1000", "trainer.n_epochs=5"]),
    ("baselines/ewc",     ["trainer.n_epochs=5"]),
    ("baselines/classical_hn", ["trainer.n_epochs=5"]),
    ("baselines/dmhn",    ["trainer.n_epochs=5"]),
    ("baselines/sparse_memory", ["trainer.n_epochs=5"]),
]

# ── Section 3: ForgetGate OFF 多 seed 补跑（CIFAR-100） ──
# ForgetGate ON（默认）的 seeds 123/456 已运行过（见 findings.md）
# ForgetGate OFF seeds 123/456 缺失
FORGEGATE_MISSING: list[tuple[str, str, int, list[str]]] = [
    ("idf_hn", "split_cifar100", 123,
     ["model.forget_gate.gamma_0=0", "model.forget_gate.delta_gamma=0",
      "trainer.n_epochs=2"]),
    ("idf_hn", "split_cifar100", 456,
     ["model.forget_gate.gamma_0=0", "model.forget_gate.delta_gamma=0",
      "trainer.n_epochs=2"]),
]


def _expand_experiments(
    groups: list[tuple[str, str, list[str]]],
) -> list[tuple[str, str, int, list[str]]]:
    exps = []
    for model, dataset, overrides in groups:
        for seed in SEEDS:
            exps.append((model, dataset, seed, overrides))
    return exps


def _expand_newsgroups() -> list[tuple[str, str, int, list[str]]]:
    exps = []
    for model, overrides in NEWSGROUPS_MODELS:
        for seed in SEEDS:
            exps.append((model, "split_newsgroups", seed, overrides))
    return exps


SECTION_MAP: dict[str, list[tuple[str, str, int, list[str]]]] = {
    "sparse":     _expand_experiments(SPARSE_MEMORY_GROUPS),
    "newsgroups": _expand_newsgroups(),
    "forgegate":  FORGEGATE_MISSING,
}

ALL_EXPERIMENTS: list[tuple[str, str, int, list[str]]] = (
    SECTION_MAP["sparse"] + SECTION_MAP["newsgroups"] + SECTION_MAP["forgegate"]
)


def build_cmd(model: str, dataset: str, seed: int, overrides: list[str]) -> list[str]:
    return [
        "uv", "run", "python", "run/main.py",
        f"model={model}",
        f"dataset={dataset}",
        f"seed={seed}",
    ] + overrides


def run_experiment(
    idx: int, total: int,
    model: str, dataset: str, seed: int, overrides: list[str],
    dry_run: bool,
) -> bool:
    cmd = build_cmd(model, dataset, seed, overrides)
    label = f"[{idx}/{total}] {model} | {dataset} | seed={seed}"
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*65}")

    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode}  elapsed={elapsed:.0f}s")
        return False
    print(f"  [OK] elapsed={elapsed:.0f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="额外基线与 Newsgroups 实验")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=1, metavar="N")
    parser.add_argument("--section", choices=list(SECTION_MAP.keys()) + ["all"],
                        default="all",
                        help="只运行指定 section (sparse/newsgroups/forgegate/all)")
    args = parser.parse_args()

    if args.section == "all":
        exps = ALL_EXPERIMENTS
    else:
        exps = SECTION_MAP[args.section]

    total = len(exps)
    print(f"共 {total} 个实验（section={args.section}），从第 {args.start} 个开始")
    if args.dry_run:
        print("（DRY-RUN 模式）\n")

    failed: list[int] = []
    for i, (model, dataset, seed, overrides) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_experiment(i, total, model, dataset, seed, overrides, args.dry_run)
        if not ok:
            failed.append(i)
            print(f"  [WARN] 实验 {i} 失败，继续执行后续...")

    print(f"\n{'='*65}")
    print(f"完成：{total - len(failed)}/{total} 成功")
    if failed:
        print(f"失败序号：{failed}")
        print(f"可用 --start {min(failed)} 断点续跑")
        sys.exit(1)
    else:
        print("所有实验成功！")


if __name__ == "__main__":
    main()
