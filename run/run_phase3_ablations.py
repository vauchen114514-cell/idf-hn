"""Phase 3 OCL 消融实验脚本：IDF-HN Transformer 组件贡献验证。

消融维度（均在 OCL 设置 n_epochs=1 下）：
  1. ForgetGate OFF (forget_mode=none)：去掉 IDF 遗忘门，验证其对 OCL 的贡献
  2. FIFO 驱逐 (eviction_policy=fifo)：对比 norm_min 智能驱逐策略

实验规模：2 维度 × 2 数据集 × 3 seeds = 12 次

用法：
    cd idf-hn
    uv run python run/run_phase3_ablations.py --dry-run
    uv run python run/run_phase3_ablations.py
    uv run python run/run_phase3_ablations.py --start 3
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

DATASETS = [
    "split_newsgroups_text",
    "wiki_facts_text",
]

# 消融变体：(ablation_name, extra_overrides)
ABLATIONS: list[tuple[str, list[str]]] = [
    ("forget_off",  ["model.forget_mode=none",              "trainer.n_epochs=1"]),
    ("fifo_evict",  ["model.eviction_policy=fifo",          "trainer.n_epochs=1"]),
]


def build_experiments() -> list[tuple[str, str, int, list[str]]]:
    """生成 (ablation, dataset, seed, overrides) 元组列表。"""
    exps: list[tuple[str, str, int, list[str]]] = []
    for ablation, overrides in ABLATIONS:
        for dataset in DATASETS:
            for seed in SEEDS:
                exps.append((ablation, dataset, seed, overrides))
    return exps


def run_one(
    ablation: str,
    dataset: str,
    seed: int,
    overrides: list[str],
    idx: int,
    total: int,
    dry_run: bool,
) -> bool:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"outputs/phase3_ablation_{ablation}_{dataset}_{seed}_{ts}"
    cmd = [
        "uv", "run", "python", "run/main.py",
        "model=idf_hn_transformer",
        f"dataset={dataset}",
        "trainer=continual_distilbert",
        f"seed={seed}",
        f"hydra.run.dir={out_dir}",
    ] + overrides

    print(f"\n{'='*70}")
    print(f"  [{idx}/{total}] ablation={ablation} | {dataset} | seed={seed}")
    print(f"  overrides: {' '.join(overrides)}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

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
    parser = argparse.ArgumentParser(description="Phase 3 OCL 消融实验")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际运行")
    parser.add_argument("--start", type=int, default=1, help="从第 N 个实验续跑")
    args = parser.parse_args()

    exps = build_experiments()
    total = len(exps)

    print(f"\nPhase 3 OCL 消融实验")
    print(f"总计：{total} 次实验（2 维度 × 2 数据集 × 3 seeds）")
    print(f"Seeds：{SEEDS}")
    print(f"消融维度：{[a for a, _ in ABLATIONS]}")
    if args.dry_run:
        print("（DRY-RUN 模式，不实际运行）\n")

    failed: list[int] = []
    for i, (ablation, dataset, seed, overrides) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_one(ablation, dataset, seed, overrides, i, total, args.dry_run)
        if not ok:
            failed.append(i)

    print(f"\n{'='*70}")
    print(f"完成：{total - len(failed)}/{total} 成功")
    if failed:
        print(f"失败序号：{failed}，可用 --start {min(failed)} 续跑")
        sys.exit(1)
    else:
        print("全部成功！")


if __name__ == "__main__":
    main()
