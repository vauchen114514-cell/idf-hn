"""Phase 3 实验批量运行脚本：IDF-HN Transformer 嵌入实验。

实验内容（来自 research-proposal.md Phase 3）：
  1. 标准实验（n_epochs=3, Transformer 微调）：
       4 模型 × 2 数据集 × 3 seeds = 24 次

  2. OCL 单轮评估（n_epochs=1，在线持续学习 benchmark）：
       2 模型 × 2 数据集 × 3 seeds = 12 次

模型对比框架：
  - idf_hn_transformer:    IDF-KV 交叉注意力（Phase 3 主模型）
  - distilbert_er:         DistilBERT + Experience Replay（上界参考）
  - kv_cache_distilbert:   DistilBERT + KV-Cache（无 IDF，对照）
  - distilbert_finetune:   纯顺序微调（灾难性遗忘下界）

数据集：
  - split_newsgroups_text:  20 Newsgroups，5 任务，DistilBERT tokenized
  - wiki_facts_text:        WikiFacts（DBpedia14），5 任务，DistilBERT tokenized

用法：
    cd idf-hn
    uv run python run/run_phase3_experiments.py --dry-run
    uv run python run/run_phase3_experiments.py --section main
    uv run python run/run_phase3_experiments.py --section ocl
    uv run python run/run_phase3_experiments.py
    uv run python run/run_phase3_experiments.py --start 5
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 123, 456]

# 标准 Transformer 微调实验（n_epochs=3, trainer=continual_distilbert）
MAIN_MODELS: list[tuple[str, list[str]]] = [
    ("idf_hn_transformer",   []),
    ("distilbert_er",        []),
    ("kv_cache_distilbert",  []),
    ("distilbert_finetune",  []),
]

MAIN_DATASETS = [
    "split_newsgroups_text",
    "wiki_facts_text",
]

# OCL 单轮评估（n_epochs=1）
OCL_MODELS: list[tuple[str, list[str]]] = [
    ("idf_hn_transformer",  ["trainer.n_epochs=1"]),
    ("distilbert_er",       ["trainer.n_epochs=1"]),
]

OCL_DATASETS = [
    "split_newsgroups_text",
    "wiki_facts_text",
]


def build_experiments(section: str) -> list[tuple[str, str, int, list[str]]]:
    """生成 (model, dataset, seed, extra_overrides) 元组列表。"""
    exps: list[tuple[str, str, int, list[str]]] = []

    if section in ("main", "all"):
        for model, overrides in MAIN_MODELS:
            for dataset in MAIN_DATASETS:
                for seed in SEEDS:
                    exps.append((model, dataset, seed, overrides))

    if section in ("ocl", "all"):
        for model, overrides in OCL_MODELS:
            for dataset in OCL_DATASETS:
                for seed in SEEDS:
                    exps.append((model, dataset, seed, overrides))

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
    """运行单次实验，返回是否成功。"""
    cmd = [
        "uv", "run", "python", "run/main.py",
        f"model={model}",
        f"dataset={dataset}",
        f"trainer=continual_distilbert",
        f"seed={seed}",
    ] + extra

    print(f"\n{'='*70}")
    print(f"  [{idx}/{total}] {model} | {dataset} | seed={seed}")
    if extra:
        print(f"  override: {' '.join(extra)}")
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
    parser = argparse.ArgumentParser(
        description="Phase 3：IDF-HN Transformer 嵌入实验"
    )
    parser.add_argument(
        "--section",
        choices=["main", "ocl", "all"],
        default="all",
        help="实验分组：main=标准3轮, ocl=在线单轮, all=全部",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际运行")
    parser.add_argument("--start", type=int, default=1, help="从第 N 个实验续跑")
    args = parser.parse_args()

    exps = build_experiments(args.section)
    total = len(exps)

    print(f"\nPhase 3 实验规划（section={args.section}）")
    print(f"总计：{total} 次实验")
    print(f"Seeds：{SEEDS}")
    if args.dry_run:
        print("（DRY-RUN 模式，不实际运行）\n")

    failed: list[int] = []
    for i, (model, dataset, seed, extra) in enumerate(exps, start=1):
        if i < args.start:
            continue
        ok = run_one(model, dataset, seed, extra, i, total, args.dry_run)
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
