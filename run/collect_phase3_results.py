"""收集 Phase 3 实验结果（IDF-HN Transformer 嵌入实验）。

从 Hydra 输出目录的 metrics.json 中读取 idf_hn_transformer / distilbert_er /
kv_cache_distilbert / distilbert_finetune 在文本数据集上的结果，
生成与 sentence_emb 实验的对比表。

用法：
    cd idf-hn
    uv run python run/collect_phase3_results.py
    uv run python run/collect_phase3_results.py --section main
    uv run python run/collect_phase3_results.py --section ocl
    uv run python run/collect_phase3_results.py --dataset newsgroups
"""
import argparse
import json
import statistics
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

MODEL_DISPLAY = {
    "idf_hn_transformer":   "IDF-HN-Transformer",
    "distilbert_er":        "DistilBERT-ER",
    "kv_cache_distilbert":  "KV-Cache-DistilBERT",
    "distilbert_finetune":  "DistilBERT-Finetune",
}

MODELS_ORDER = [
    "IDF-HN-Transformer",
    "DistilBERT-ER",
    "KV-Cache-DistilBERT",
    "DistilBERT-Finetune",
]

TEXT_DATASETS = {"split_newsgroups_text", "wiki_facts_text"}

# Sentence-emb 参考基线（findings.md 第16节，冻结 encoder + all-mpnet-base-v2）
SENTEMB_BASELINE = {
    "split_newsgroups": {
        "IDF-HN":       {"aa": 0.7583, "bwt": -0.0238},
        "ER":           {"aa": 0.7875, "bwt": -0.0050},
        "GSS":          {"aa": 0.7938, "bwt": +0.0047},
        "SparseMemory": {"aa": 0.7918, "bwt": +0.0015},
        "EWC":          {"aa": 0.7579, "bwt": -0.0632},
    },
    "wiki_facts": {
        "IDF-HN":       {"aa": 0.9769, "bwt": -0.0210},
        "ER":           {"aa": 0.9910, "bwt": -0.0027},
        "SparseMemory": {"aa": 0.9908, "bwt": -0.0030},
        "EWC":          {"aa": 0.9528, "bwt": -0.0537},
    },
}


def collect_results(
    section: str,
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """返回 {section: {dataset: {model: {"aa": [...], "bwt": [...]}}}}"""
    results: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        "main": {"split_newsgroups_text": {}, "wiki_facts_text": {}},
        "ocl":  {"split_newsgroups_text": {}, "wiki_facts_text": {}},
    }

    seen: set[tuple[str, str, str, int]] = set()

    for metrics_path in sorted(OUTPUTS_DIR.rglob("metrics.json")):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        dataset_raw = data.get("dataset", "")
        if dataset_raw not in TEXT_DATASETS:
            continue

        model_raw = data.get("model", "")
        model = MODEL_DISPLAY.get(model_raw)
        if model is None:
            continue

        seed = data.get("seed", -1)
        aa = data.get("aa")
        bwt = data.get("bwt")
        n_epochs = data.get("n_epochs", 3)
        if aa is None or bwt is None:
            continue

        # 用 n_epochs 区分 main vs ocl
        exp_section = "ocl" if n_epochs == 1 else "main"
        dedup_key = (exp_section, dataset_raw, model, seed)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        bucket = results[exp_section][dataset_raw]
        if model not in bucket:
            bucket[model] = {"aa": [], "bwt": []}
        bucket[model]["aa"].append(aa)
        bucket[model]["bwt"].append(bwt)

    return results


def _canonical(dataset_raw: str) -> str:
    """split_newsgroups_text → split_newsgroups"""
    return dataset_raw.replace("_text", "")


def print_table(
    exp_section: str,
    dataset_raw: str,
    model_results: dict[str, dict[str, list[float]]],
) -> None:
    canonical = _canonical(dataset_raw)
    baseline = SENTEMB_BASELINE.get(canonical, {})
    dataset_label = canonical.replace("_", "-").title()
    epochs_label = "n_epochs=1 (OCL)" if exp_section == "ocl" else "n_epochs=3"

    print(f"\n{'='*80}")
    print(f"[{exp_section.upper()}] {dataset_label}  DistilBERT 端到端微调 ({epochs_label})")
    print(f"{'='*80}")
    hdr = f"{'模型':<25} {'AA':<22} {'BWT':<22} {'sentemb BWT (ref)'}"
    print(hdr)
    print("-" * 80)

    ordered = [m for m in MODELS_ORDER if m in model_results]
    others = [m for m in model_results if m not in ordered]

    for model in ordered + others:
        d = model_results.get(model, {})
        n = len(d.get("aa", []))
        if n > 0:
            aa_m = statistics.mean(d["aa"])
            bwt_m = statistics.mean(d["bwt"])
            aa_s = statistics.stdev(d["aa"]) if n > 1 else 0.0
            bwt_s = statistics.stdev(d["bwt"]) if n > 1 else 0.0
            aa_str = f"{aa_m:.4f}±{aa_s:.4f} (n={n})"
            bwt_str = f"{bwt_m:.4f}±{bwt_s:.4f}"
        else:
            aa_str, bwt_str, bwt_m = "—", "—", None

        # sentemb 参考值（IDF-HN-Transformer → 对应 IDF-HN，其余类推）
        ref_name = {
            "IDF-HN-Transformer":   "IDF-HN",
            "DistilBERT-ER":        "ER",
            "KV-Cache-DistilBERT":  None,
            "DistilBERT-Finetune":  None,
        }.get(model)
        ref = baseline.get(ref_name) if ref_name else None
        ref_str = f"{ref['bwt']:+.4f}" if ref else "—"

        print(f"{model:<25} {aa_str:<22} {bwt_str:<22} {ref_str}")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["main", "ocl", "all"], default="all")
    parser.add_argument("--dataset", choices=["newsgroups", "wikifacts", "all"], default="all")
    args = parser.parse_args()

    all_results = collect_results(args.section)

    sections = ["main", "ocl"] if args.section == "all" else [args.section]
    datasets = []
    if args.dataset in ("newsgroups", "all"):
        datasets.append("split_newsgroups_text")
    if args.dataset in ("wikifacts", "all"):
        datasets.append("wiki_facts_text")

    any_result = False
    for sec in sections:
        for ds in datasets:
            model_results = all_results[sec].get(ds, {})
            if model_results:
                print_table(sec, ds, model_results)
                any_result = True

    if not any_result:
        print("\n未找到 Phase 3 结果。请先运行：")
        print("  cd idf-hn && uv run python run/run_phase3_experiments.py --dry-run")


if __name__ == "__main__":
    main()
