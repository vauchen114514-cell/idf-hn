"""ForgetGate 能量优先级回放对比实验（Split-CIFAR-100，3 seeds）。

对比 replay_strategy=random（基线）vs replay_strategy=energy_priority（新方案）。
每种策略 3 seeds，共 6 次实验。

用法：
    cd idf-hn
    uv run python run/run_priority_replay_experiments.py
    uv run python run/run_priority_replay_experiments.py --dry-run
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

STRATEGIES = [
    ("random",          "random"),
    ("energy_priority", "energy_priority"),
]


def run_one(
    strategy_label: str,
    replay_strategy: str,
    seed: int,
    idx: int,
    total: int,
    dry_run: bool,
) -> bool:
    cmd = [
        "uv", "run", "python", "run/main.py",
        "model=idf_hn",
        "dataset=split_cifar100",
        f"seed={seed}",
        "trainer.n_epochs=2",
        f"trainer.replay_strategy={replay_strategy}",
        # 保持与主实验一致的 ForgetGate 配置
        "model.forget_gate.gamma_0=0.1",
        "model.forget_gate.delta_gamma=0.5",
        "model.memory_size=5000",
    ]
    print(f"\n{'='*65}")
    print(f"  [{idx}/{total}] strategy={strategy_label} | seed={seed}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    exps = [
        (label, strategy, seed)
        for label, strategy in STRATEGIES
        for seed in SEEDS
    ]
    total = len(exps)
    print(f"共 {total} 次实验 (random×3 + energy_priority×3)")
    if args.dry_run:
        print("（DRY-RUN）\n")

    failed = []
    for i, (label, strategy, seed) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_one(label, strategy, seed, i, total, args.dry_run)
        if not ok:
            failed.append(i)

    print(f"\n{'='*65}")
    print(f"完成：{total - len(failed)}/{total} 成功")
    if failed:
        print(f"失败序号：{failed}，可用 --start {min(failed)} 续跑")
        sys.exit(1)
    else:
        print("全部成功！")


if __name__ == "__main__":
    main()
