"""消融实验批量运行脚本。

消融维度（均在 Split-MNIST 上作为锚定数据集，3 seeds）：

  维度 1：遗忘机制（forget mechanism）
    - none:           gamma_0=0, delta_gamma=0（无遗忘，Vanilla MHN）
    - time_decay:     delta_gamma=0（固定 gamma_0，无输入依赖）
    - static_density: forget_mode=static_density（按槽位密度逐槽衰减）
    - input_dependent: 默认 IDF-HN（已有结果，不重跑）

  维度 2：Dreaming
    - random:   dreaming.enabled=true, semantic_threshold=2.0（全部候选可遗忘）
    - semantic: dreaming.enabled=true, semantic_threshold=0.5（语义过滤）
    - none: 默认禁用（已有结果）

  维度 3：效率（密度计算方式）
    - exact: memory_bank.type=exact（精确 O(N) 密度）
    - prototype: 默认（已有结果）

  τ sweep（idf_hn 默认 adaptive_tau=true，固定 τ 消融）：
    - tau=0.3, 0.5, 0.7, 0.9（关闭 adaptive_tau）

总计：(3+2+1+4) × 3 seeds = 30 次运行

用法：
    cd idf-hn
    uv run python run/run_ablation_experiments.py
    uv run python run/run_ablation_experiments.py --dry-run
    uv run python run/run_ablation_experiments.py --start 10
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

# (label, model, dataset, overrides)
ABLATION_GROUPS: list[tuple[str, str, str, list[str]]] = [
    # ── 维度 1：遗忘机制 ──
    # none：gamma_0=0, delta_gamma=0 → IDF-HN 无遗忘（等价 Vanilla MHN + write_threshold）
    (
        "forget=none",
        "idf_hn", "split_mnist",
        ["model.forget_gate.gamma_0=0", "model.forget_gate.delta_gamma=0",
         "trainer.n_epochs=2"],
    ),
    # time_decay：delta_gamma=0 → 固定遗忘率
    (
        "forget=time_decay",
        "idf_hn", "split_mnist",
        ["model.forget_gate.delta_gamma=0", "trainer.n_epochs=2"],
    ),
    # static_density：按槽位 Prototype 密度逐槽衰减
    (
        "forget=static_density",
        "idf_hn", "split_mnist",
        ["model.forget_mode=static_density", "trainer.n_epochs=2"],
    ),

    # ── 维度 2：Dreaming ──
    # random：开启 Dreaming，semantic_threshold=2.0 → 所有候选均可遗忘（随机）
    (
        "dream=random",
        "idf_hn", "split_mnist",
        ["model.dreaming.enabled=true",
         "model.dreaming.semantic_threshold=2.0",
         "trainer.n_epochs=2"],
    ),
    # semantic：开启 Dreaming，使用默认语义过滤阈值 0.5
    (
        "dream=semantic",
        "idf_hn", "split_mnist",
        ["model.dreaming.enabled=true",
         "model.dreaming.semantic_threshold=0.5",
         "trainer.n_epochs=2"],
    ),

    # ── 维度 3：效率 ──
    # exact：精确 O(N) 密度（ExactDensityBank，不用 Prototype 聚类）
    (
        "efficiency=exact",
        "idf_hn", "split_mnist",
        ["model.memory_bank.type=exact", "trainer.n_epochs=2"],
    ),

    # ── 维度 4：τ sweep（关闭 adaptive_tau，固定 τ）──
    (
        "tau=0.3",
        "idf_hn", "split_mnist",
        ["model.forget_gate.adaptive_tau=false",
         "model.forget_gate.tau=0.3",
         "trainer.n_epochs=2"],
    ),
    (
        "tau=0.5",
        "idf_hn", "split_mnist",
        ["model.forget_gate.adaptive_tau=false",
         "model.forget_gate.tau=0.5",
         "trainer.n_epochs=2"],
    ),
    (
        "tau=0.7",
        "idf_hn", "split_mnist",
        ["model.forget_gate.adaptive_tau=false",
         "model.forget_gate.tau=0.7",
         "trainer.n_epochs=2"],
    ),
    (
        "tau=0.9",
        "idf_hn", "split_mnist",
        ["model.forget_gate.adaptive_tau=false",
         "model.forget_gate.tau=0.9",
         "trainer.n_epochs=2"],
    ),
]

# 展开为每条独立实验 (label, model, dataset, seed, overrides)
EXPERIMENTS: list[tuple[str, str, str, int, list[str]]] = []
for label, model, dataset, overrides in ABLATION_GROUPS:
    for seed in SEEDS:
        EXPERIMENTS.append((label, model, dataset, seed, overrides))


def build_cmd(model: str, dataset: str, seed: int, overrides: list[str]) -> list[str]:
    cmd = [
        "uv", "run", "python", "run/main.py",
        f"model={model}",
        f"dataset={dataset}",
        f"seed={seed}",
    ] + overrides
    return cmd


def run_experiment(
    idx: int, total: int,
    label: str, model: str, dataset: str, seed: int, overrides: list[str],
    dry_run: bool,
) -> bool:
    cmd = build_cmd(model, dataset, seed, overrides)
    print(f"\n{'='*65}")
    print(f"  [{idx}/{total}] {label} | {dataset} | seed={seed}")
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
    parser = argparse.ArgumentParser(description="批量运行消融实验")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=1, metavar="N")
    parser.add_argument("--group", type=str, default=None,
                        help="只运行标签包含指定字符串的组（如 'forget' / 'dream' / 'tau'）")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.group:
        exps = [(l, m, d, s, o) for l, m, d, s, o in exps if args.group in l]

    total = len(exps)
    print(f"消融实验共 {total} 条（筛选: {args.group or '全部'}），从第 {args.start} 条开始")
    if args.dry_run:
        print("（DRY-RUN 模式）\n")

    failed: list[int] = []
    for i, (label, model, dataset, seed, overrides) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_experiment(i, total, label, model, dataset, seed, overrides, args.dry_run)
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
        print("所有消融实验成功！")


if __name__ == "__main__":
    main()
