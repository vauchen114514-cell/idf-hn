"""收集补做实验结果（clipping_hn + wiki_facts）。

从 outputs/ 目录递归扫描符合条件的实验结果，输出：
  1. Clipping HN vs 其他基线（3 个主数据集）
  2. WikiFacts 全模型对比（含语义相似度、压缩比）

用法：
    cd idf-hn
    uv run python run/collect_supplement_results.py
    uv run python run/collect_supplement_results.py --verbose
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SEEDS = [42, 123, 456]

CLIPPING_DATASETS = ["split_mnist", "split_cifar100", "permuted_mnist"]
WIKI_MODELS = [
    "idf_hn", "er", "gss", "sparse_memory",
    "classical_hn", "ewc", "clipping_hn",
]


def parse_run_dir(run_dir: Path) -> dict | None:
    """解析 outputs/<name>/ 目录，返回指标字典或 None。"""
    log_file = run_dir / "main.log"
    hydra_cfg = run_dir / ".hydra" / "config.yaml"

    if not log_file.exists() or not hydra_cfg.exists():
        return None

    cfg_text = hydra_cfg.read_text(encoding="utf-8", errors="replace")

    seed_m = re.search(r"^seed:\s*(\d+)", cfg_text, re.MULTILINE)
    model_m = re.search(r"^\s+name:\s*(\w+)", cfg_text, re.MULTILINE)
    dataset_m = re.search(
        r"name:\s*(split_mnist|split_cifar100|permuted_mnist|split_newsgroups|wiki_facts)",
        cfg_text,
    )

    if not (seed_m and model_m and dataset_m):
        return None

    seed = int(seed_m.group(1))
    model_raw = model_m.group(1)
    dataset = dataset_m.group(1)

    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    aa_m = re.search(r"AA\s*=\s*([\d.]+)", log_text)
    bwt_m = re.search(r"BWT\s*=\s*(-?[\d.]+)", log_text)
    fwt_m = re.search(r"FWT\s*=\s*(-?[\d.]+)", log_text)

    if not (aa_m and bwt_m and fwt_m):
        return None

    rec: dict = {
        "model":   model_raw,
        "dataset": dataset,
        "seed":    seed,
        "AA":      float(aa_m.group(1)),
        "BWT":     float(bwt_m.group(1)),
        "FWT":     float(fwt_m.group(1)),
        "run_dir": str(run_dir.name),
    }

    # 语义指标（仅 wiki_facts / split_newsgroups 中的 Hopfield 模型）
    sem_m = re.search(r"SemanticSim\s*=\s*([\d.]+)", log_text)
    comp_m = re.search(r"CompressRatio\s*=\s*([\d.]+)", log_text)
    if sem_m:
        rec["semantic_similarity"] = float(sem_m.group(1))
    if comp_m:
        rec["compression_ratio"] = float(comp_m.group(1))

    return rec


def load_all_results() -> list[dict]:
    """加载所有已完成实验，每个 (model, dataset, seed) 只保留最新一条。"""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        rec = parse_run_dir(run_dir)
        if rec is None:
            continue
        key = (rec["model"], rec["dataset"], rec["seed"])
        grouped[key].append(rec)

    results = []
    for recs in grouped.values():
        results.append(max(recs, key=lambda r: r["run_dir"]))
    return results


def collect_model_dataset(
    all_results: list[dict], model: str, dataset: str
) -> dict | None:
    """收集 model × dataset 的 3 seed 均值/标准差。"""
    rows = [
        r for r in all_results
        if r["model"] == model and r["dataset"] == dataset
        and r["seed"] in SEEDS
    ]
    # 每个 seed 只取一条（已在 load_all_results 去重）
    by_seed = {r["seed"]: r for r in rows}
    rows = [by_seed[s] for s in SEEDS if s in by_seed]

    if len(rows) < len(SEEDS):
        return None

    aa_vals  = [r["AA"]  for r in rows]
    bwt_vals = [r["BWT"] for r in rows]
    fwt_vals = [r["FWT"] for r in rows]

    out: dict = {
        "aa_mean":  float(np.mean(aa_vals)),
        "aa_std":   float(np.std(aa_vals)),
        "bwt_mean": float(np.mean(bwt_vals)),
        "bwt_std":  float(np.std(bwt_vals)),
        "fwt_mean": float(np.mean(fwt_vals)),
        "fwt_std":  float(np.std(fwt_vals)),
        "n_seeds":  len(rows),
    }

    sem_vals  = [r["semantic_similarity"] for r in rows if "semantic_similarity" in r]
    comp_vals = [r["compression_ratio"]   for r in rows if "compression_ratio"   in r]
    if sem_vals:
        out["sem_sim_mean"] = float(np.mean(sem_vals))
        out["sem_sim_std"]  = float(np.std(sem_vals))
    if comp_vals:
        out["comp_ratio_mean"] = float(np.mean(comp_vals))
        out["comp_ratio_std"]  = float(np.std(comp_vals))

    return out


def print_table_clipping(all_results: list[dict], verbose: bool) -> None:
    """打印 Clipping HN 对比表（3 个主数据集）。"""
    compare_models = ["classical_hn", "sparse_memory", "er", "idf_hn", "clipping_hn"]

    for dataset in CLIPPING_DATASETS:
        print(f"\n{'='*72}")
        print(f"  Clipping HN 对比 — {dataset}")
        print(f"{'='*72}")
        print(f"{'模型':<18} | {'AA':>20} | {'BWT':>22} |")
        print(f"{'':18} | {'mean ± std':>20} | {'mean ± std':>22} |")
        print("-" * 72)

        for model in compare_models:
            r = collect_model_dataset(all_results, model, dataset)
            if r is None:
                mark = "(缺少数据)" if model == "clipping_hn" else ""
                print(f"  {model:<16} | {'N/A':>20} | {'N/A':>22} | {mark}")
            else:
                aa_str  = f"{r['aa_mean']:.4f} ± {r['aa_std']:.4f}"
                bwt_str = f"{r['bwt_mean']:.4f} ± {r['bwt_std']:.4f}"
                flag = " ←" if model == "clipping_hn" else ""
                print(f"  {model:<16} | {aa_str:>20} | {bwt_str:>22} |{flag}")


def print_table_wiki(all_results: list[dict], verbose: bool) -> None:
    """打印 WikiFacts 全模型对比表（含语义指标）。"""
    print(f"\n{'='*90}")
    print("  WikiFacts（DBpedia14）全模型对比")
    print(f"{'='*90}")
    print(
        f"{'模型':<18} | {'AA mean±std':>20} | {'BWT mean±std':>22} | "
        f"{'SemanticSim':>12} | {'CompRatio':>10}"
    )
    print("-" * 90)

    for model in WIKI_MODELS:
        r = collect_model_dataset(all_results, model, "wiki_facts")
        if r is None:
            print(f"  {model:<16} | {'N/A':>20} | {'N/A':>22} | {'N/A':>12} | {'N/A':>10}")
            continue

        aa_str   = f"{r['aa_mean']:.4f} ± {r['aa_std']:.4f}"
        bwt_str  = f"{r['bwt_mean']:.4f} ± {r['bwt_std']:.4f}"
        sem_str  = f"{r['sem_sim_mean']:.4f}"  if "sem_sim_mean"   in r else "N/A"
        comp_str = f"{r['comp_ratio_mean']:.4f}" if "comp_ratio_mean" in r else "N/A"
        print(
            f"  {model:<16} | {aa_str:>20} | {bwt_str:>22} | "
            f"{sem_str:>12} | {comp_str:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="收集补做实验结果")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n[补做实验结果收集]")
    print(f"输出目录：{OUTPUTS_DIR}")

    all_results = load_all_results()
    print(f"共加载 {len(all_results)} 条已完成实验记录\n")

    print_table_clipping(all_results, args.verbose)
    print_table_wiki(all_results, args.verbose)

    print("\n[完成]")


if __name__ == "__main__":
    main()
