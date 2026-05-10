"""句子嵌入（all-mpnet-base-v2）实验批量运行脚本。

实验内容：
  1. Split-20Newsgroups（sentence_emb）：全 7 模型 × 3 seeds = 21 次
  2. WikiFacts（sentence_emb）：主要 5 模型 × 3 seeds = 15 次
总计：36 次运行

注意：首次运行会下载 all-mpnet-base-v2 模型并提取特征，之后从缓存加载。
  - Newsgroups 特征提取：GPU 约 2-5 分钟，CPU 约 15-30 分钟
  - WikiFacts 特征提取：GPU 约 10-20 分钟（560K 条），CPU 极慢（不推荐）

用法：
    cd idf-hn
    uv run python run/run_sentemb_experiments.py
    uv run python run/run_sentemb_experiments.py --dry-run
    uv run python run/run_sentemb_experiments.py --section newsgroups
    uv run python run/run_sentemb_experiments.py --section wikifacts
    uv run python run/run_sentemb_experiments.py --start 5   # 从第 5 个续跑
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

# ── Section 1: Split-20Newsgroups + 句子嵌入 ──
NEWSGROUPS_MODELS: list[tuple[str, list[str]]] = [
    ("idf_hn",                 ["trainer.n_epochs=5"]),
    ("baselines/er",           ["model.memory_size=1000", "trainer.n_epochs=5"]),
    ("baselines/gss",          ["model.memory_size=1000", "trainer.n_epochs=5"]),
    ("baselines/ewc",          ["trainer.n_epochs=5"]),
    ("baselines/classical_hn", ["trainer.n_epochs=5"]),
    ("baselines/dmhn",         ["trainer.n_epochs=5"]),
    ("baselines/sparse_memory",["trainer.n_epochs=5"]),
]

# ── Section 2: WikiFacts + 句子嵌入 ──
WIKIFACTS_MODELS: list[tuple[str, list[str]]] = [
    ("idf_hn",                 ["trainer.n_epochs=5", "model.memory_size=1000"]),
    ("baselines/er",           ["model.memory_size=1000", "trainer.n_epochs=5"]),
    ("baselines/gss",          ["model.memory_size=1000", "trainer.n_epochs=5"]),
    ("baselines/sparse_memory",["trainer.n_epochs=5"]),
    ("baselines/ewc",          ["trainer.n_epochs=5"]),
    ("baselines/classical_hn", ["trainer.n_epochs=5"]),
]


def build_experiments(section: str) -> list[tuple[str, str, int, list[str]]]:
    """生成 (model, dataset, seed, extra_overrides) 元组列表。"""
    exps = []
    if section in ("newsgroups", "all"):
        for model, overrides in NEWSGROUPS_MODELS:
            for seed in SEEDS:
                exps.append((model, "split_newsgroups_sentemb", seed, overrides))
    if section in ("wikifacts", "all"):
        for model, overrides in WIKIFACTS_MODELS:
            for seed in SEEDS:
                exps.append((model, "wiki_facts_sentemb", seed, overrides))
    return exps


def run_one(
    model: str,
    dataset: str,
    seed: int,
    extra: list[str],
    idx: int,
    total: int,
    dry_run: bool,
) -> bool:
    model_short = model.split("/")[-1]
    cmd = [
        "uv", "run", "python", "run/main.py",
        f"model={model}",
        f"dataset={dataset}",
        f"seed={seed}",
    ] + extra

    print(f"\n{'='*65}")
    print(f"  [{idx}/{total}] {model_short} | {dataset} | seed={seed}")
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
    parser.add_argument("--section", choices=["newsgroups", "wikifacts", "all"],
                        default="all")
    parser.add_argument("--start", type=int, default=1, help="从第 N 个实验续跑")
    args = parser.parse_args()

    exps = build_experiments(args.section)
    total = len(exps)
    print(f"共 {total} 次实验（section={args.section}）")
    if args.dry_run:
        print("（DRY-RUN）\n")

    failed = []
    for i, (model, dataset, seed, extra) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_one(model, dataset, seed, extra, i, total, args.dry_run)
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
