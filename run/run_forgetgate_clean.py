"""ForgetGate ON vs OFF 干净对比实验（CIFAR-100，双缓冲区，3 seeds）。

目的：解决第十四节 ForgetGate ON/OFF 数字混乱问题。
同批次、同代码版本跑 6 次，确保对比公平。

用法：
    cd idf-hn
    uv run python run/run_forgetgate_clean.py
    uv run python run/run_forgetgate_clean.py --dry-run
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

EXPERIMENTS = [
    # (label, gamma_0, delta_gamma)
    ("ON",  "0.1", "0.5"),
    ("OFF", "0.0", "0.0"),
]


def run_one(label: str, seed: int, gamma_0: str, delta_gamma: str,
            idx: int, total: int, dry_run: bool) -> bool:
    cmd = [
        "uv", "run", "python", "run/main.py",
        "model=idf_hn",
        "dataset=split_cifar100",
        f"seed={seed}",
        "trainer.n_epochs=2",
        f"model.forget_gate.gamma_0={gamma_0}",
        f"model.forget_gate.delta_gamma={delta_gamma}",
    ]
    print(f"\n{'='*65}")
    print(f"  [{idx}/{total}] ForgetGate {label} | seed={seed} | g0={gamma_0} dg={delta_gamma}")
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

    exps = [(label, seed, g0, dg)
            for label, g0, dg in EXPERIMENTS
            for seed in SEEDS]
    total = len(exps)
    print(f"共 {total} 次实验 (ForgetGate ON×3 + OFF×3)")
    if args.dry_run:
        print("（DRY-RUN）\n")

    failed = []
    for i, (label, seed, g0, dg) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_one(label, seed, g0, dg, i, total, args.dry_run)
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
