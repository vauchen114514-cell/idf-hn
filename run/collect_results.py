"""从 outputs/ 目录收集所有实验结果并打印对比表。

用法：
    cd idf-hn
    uv run python run/collect_results.py                      # 打印所有结果
    uv run python run/collect_results.py --dataset split_mnist  # 只看某数据集
    uv run python run/collect_results.py --missing            # 只列出缺失实验
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# model 名称规范化（output 目录前缀 → 论文用名）
MODEL_DISPLAY = {
    "idf_hn":        "IDF-HN",
    "er":            "ER",
    "gss":           "GSS",
    "ewc":           "EWC",
    "classical_hn":  "classical_hn",
    "dmhn":          "DMHN",
    "mhn":           "MHN",
    "sparse_memory": "SparseMemory",
}

DATASETS = ["split_mnist", "split_cifar100", "permuted_mnist", "split_newsgroups"]
MODELS_ORDER = ["IDF-HN", "ER", "GSS", "EWC", "DMHN", "classical_hn", "MHN", "SparseMemory"]


# ---------------------------------------------------------------------------
# 解析单个实验目录
# ---------------------------------------------------------------------------

def parse_run_dir(run_dir: Path) -> dict | None:
    """解析 outputs/<name>/ 目录，返回 {model, dataset, seed, AA, BWT, FWT} 或 None。"""
    log_file = run_dir / "main.log"
    hydra_cfg = run_dir / ".hydra" / "config.yaml"

    if not log_file.exists() or not hydra_cfg.exists():
        return None

    # 从 .hydra/config.yaml 读 seed、model.name、dataset.name、trainer.n_epochs
    cfg_text = hydra_cfg.read_text(encoding="utf-8", errors="replace")

    seed_m = re.search(r"^seed:\s*(\d+)", cfg_text, re.MULTILINE)
    model_m = re.search(r"^\s+name:\s*(\w+)", cfg_text, re.MULTILINE)
    dataset_m = re.search(
        r"name:\s*(split_mnist|split_cifar100|permuted_mnist|split_newsgroups)", cfg_text
    )
    epochs_m = re.search(r"n_epochs:\s*(\d+)", cfg_text)
    memsize_m = re.search(r"memory_size:\s*(\d+)", cfg_text)

    if not (seed_m and model_m and dataset_m):
        return None

    seed = int(seed_m.group(1))
    model_raw = model_m.group(1)
    dataset = dataset_m.group(1)
    n_epochs = int(epochs_m.group(1)) if epochs_m else None
    memory_size = int(memsize_m.group(1)) if memsize_m else None

    # 从 main.log 读取最终指标
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    aa_m = re.search(r"AA\s*=\s*([\d.]+)", log_text)
    bwt_m = re.search(r"BWT\s*=\s*(-?[\d.]+)", log_text)
    fwt_m = re.search(r"FWT\s*=\s*(-?[\d.]+)", log_text)

    if not (aa_m and bwt_m and fwt_m):
        return None  # 实验未完成

    return {
        "model":       model_raw,
        "dataset":     dataset,
        "seed":        seed,
        "AA":          float(aa_m.group(1)),
        "BWT":         float(bwt_m.group(1)),
        "FWT":         float(fwt_m.group(1)),
        "n_epochs":    n_epochs,
        "memory_size": memory_size,
        "run_dir":     str(run_dir.name),
    }


# ---------------------------------------------------------------------------
# 选取每个 (model, dataset, seed) 的最新实验（防止重跑后有多条）
# ---------------------------------------------------------------------------

def load_all_results(outputs_dir: Path) -> list[dict]:
    """加载所有已完成实验，每个 (model, dataset, seed) 只保留最新一条。"""
    # key: (model, dataset, seed) → list of results（按目录名排序=时间序）
    grouped: dict[tuple, list[dict]] = defaultdict(list)

    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rec = parse_run_dir(run_dir)
        if rec is None:
            continue
        key = (rec["model"], rec["dataset"], rec["seed"])
        grouped[key].append(rec)

    # 每组保留最新（目录名字典序最大 = 时间最晚）
    results = []
    for recs in grouped.values():
        results.append(max(recs, key=lambda r: r["run_dir"]))

    return results


# ---------------------------------------------------------------------------
# 统计均值 ± 标准差
# ---------------------------------------------------------------------------

def compute_stats(records: list[dict]) -> dict[str, tuple[float, float]]:
    """输入同一 (model, dataset) 的多 seed 记录，返回各指标 (mean, std)。"""
    import statistics
    stats = {}
    for metric in ["AA", "BWT", "FWT"]:
        vals = [r[metric] for r in records]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[metric] = (mean, std)
    return stats


# ---------------------------------------------------------------------------
# 打印对比表
# ---------------------------------------------------------------------------

def print_table(results: list[dict], dataset: str) -> None:
    ds_results = [r for r in results if r["dataset"] == dataset]
    if not ds_results:
        print(f"  {dataset}: 无已完成实验")
        return

    # 按 (model, seed) 分组
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in ds_results:
        by_model[r["model"]].append(r)

    print(f"\n{'='*72}")
    print(f"  {dataset.upper().replace('_', '-')}")
    print(f"{'='*72}")
    print(f"  {'模型':<16} {'AA (mean±std)':<20} {'BWT (mean±std)':<20} {'Seeds':<12} {'epochs'}")
    print(f"  {'-'*68}")

    # 按指定顺序排列
    ordered_models = [m for m in MODELS_ORDER
                      if any(k == m.lower().replace("-", "_") or
                             MODEL_DISPLAY.get(k, k) == m
                             for k in by_model)]
    # 追加未在顺序列表的模型
    extra = [k for k in by_model if MODEL_DISPLAY.get(k, k) not in MODELS_ORDER]
    ordered_keys = []
    for m in MODELS_ORDER:
        for k in by_model:
            if MODEL_DISPLAY.get(k, k) == m and k not in ordered_keys:
                ordered_keys.append(k)
    for k in extra:
        if k not in ordered_keys:
            ordered_keys.append(k)

    for model_key in ordered_keys:
        recs = sorted(by_model[model_key], key=lambda r: r["seed"])
        display = MODEL_DISPLAY.get(model_key, model_key)
        seeds_done = [r["seed"] for r in recs]
        epochs = recs[0]["n_epochs"] if recs else "?"

        if len(recs) >= 2:
            stats = compute_stats(recs)
            aa_str  = f"{stats['AA'][0]:.4f} ± {stats['AA'][1]:.4f}"
            bwt_str = f"{stats['BWT'][0]:.4f} ± {stats['BWT'][1]:.4f}"
        elif len(recs) == 1:
            r = recs[0]
            aa_str  = f"{r['AA']:.4f} (1 seed)"
            bwt_str = f"{r['BWT']:.4f} (1 seed)"
        else:
            aa_str = bwt_str = "—"

        print(f"  {display:<16} {aa_str:<20} {bwt_str:<20} {str(seeds_done):<12} {epochs}")

    print()


def print_missing(results: list[dict]) -> None:
    """打印哪些 (model, dataset, seed) 组合还未完成。"""
    REQUIRED = {
        # ── 三主数据集：主实验 ──
        ("idf_hn",        "split_mnist"):    [42, 123, 456],
        ("er",            "split_mnist"):    [42, 123, 456],
        ("gss",           "split_mnist"):    [42, 123, 456],
        ("classical_hn",  "split_mnist"):    [42, 123, 456],
        ("ewc",           "split_mnist"):    [42, 123, 456],
        ("sparse_memory", "split_mnist"):    [42, 123, 456],
        ("idf_hn",        "split_cifar100"): [42, 123, 456],
        ("er",            "split_cifar100"): [42, 123, 456],
        ("gss",           "split_cifar100"): [42, 123, 456],
        ("ewc",           "split_cifar100"): [42, 123, 456],
        ("classical_hn",  "split_cifar100"): [42, 123, 456],
        ("dmhn",          "split_cifar100"): [42, 123, 456],
        ("sparse_memory", "split_cifar100"): [42, 123, 456],
        ("idf_hn",        "permuted_mnist"): [42, 123, 456],
        ("er",            "permuted_mnist"): [42, 123, 456],
        ("gss",           "permuted_mnist"): [42, 123, 456],
        ("ewc",           "permuted_mnist"): [42, 123, 456],
        ("classical_hn",  "permuted_mnist"): [42, 123, 456],
        ("dmhn",          "permuted_mnist"): [42, 123, 456],
        ("sparse_memory", "permuted_mnist"): [42, 123, 456],
        # ── Split-20Newsgroups（WikiFacts 代理） ──
        ("idf_hn",        "split_newsgroups"): [42, 123, 456],
        ("er",            "split_newsgroups"): [42, 123, 456],
        ("gss",           "split_newsgroups"): [42, 123, 456],
        ("ewc",           "split_newsgroups"): [42, 123, 456],
        ("classical_hn",  "split_newsgroups"): [42, 123, 456],
        ("dmhn",          "split_newsgroups"): [42, 123, 456],
        ("sparse_memory", "split_newsgroups"): [42, 123, 456],
    }

    done: set[tuple] = set()
    for r in results:
        done.add((r["model"], r["dataset"], r["seed"]))

    any_missing = False
    for (model, dataset), seeds in sorted(REQUIRED.items()):
        missing_seeds = [s for s in seeds if (model, dataset, s) not in done]
        if missing_seeds:
            display = MODEL_DISPLAY.get(model, model)
            print(f"  [MISS] {display:<16} {dataset:<20} seeds={missing_seeds}")
            any_missing = True

    if not any_missing:
        print("  [OK] All required experiments complete!")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="收集并展示所有实验结果")
    parser.add_argument("--dataset", choices=DATASETS, help="只显示指定数据集")
    parser.add_argument("--missing", action="store_true", help="只列出缺失实验")
    args = parser.parse_args()

    results = load_all_results(OUTPUTS_DIR)
    print(f"共加载 {len(results)} 条已完成实验记录")

    if args.missing:
        print("\n缺失实验：")
        print_missing(results)
        return

    if args.dataset:
        print_table(results, args.dataset)
    else:
        for ds in DATASETS:
            print_table(results, ds)

    print("\n缺失实验检查：")
    print_missing(results)


if __name__ == "__main__":
    main()
