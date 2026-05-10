"""收集句子嵌入实验结果（split_newsgroups_sentemb & wiki_facts_sentemb）。

从 Hydra 输出目录的 metrics.json 中读取结果，
打印每个数据集的 AA / BWT 均值 ± 标准差对比表，
并与 TF-IDF/SVD 基线（findings.md 第 14.7 / 15.2 节）对比。

用法：
    cd idf-hn
    uv run python run/collect_sentemb_results.py
    uv run python run/collect_sentemb_results.py --dataset newsgroups
    uv run python run/collect_sentemb_results.py --dataset wikifacts
"""
import argparse
import json
import statistics
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

MODEL_DISPLAY = {
    "idf_hn":        "IDF-HN",
    "er":            "ER",
    "gss":           "GSS",
    "ewc":           "EWC",
    "classical_hn":  "classical_hn",
    "dmhn":          "DMHN",
    "sparse_memory": "SparseMemory",
}

MODELS_ORDER = ["IDF-HN", "ER", "GSS", "SparseMemory", "EWC", "DMHN", "classical_hn"]

# TF-IDF/SVD 基线（findings.md 历史结果，用于对比）
TFIDF_BASELINE = {
    "split_newsgroups": {
        "DMHN":        {"aa": 0.7375, "bwt": -0.0105},
        "EWC":         {"aa": 0.7372, "bwt": -0.0228},
        "GSS":         {"aa": 0.7312, "bwt": +0.0042},
        "ER":          {"aa": 0.7216, "bwt": -0.0058},
        "SparseMemory":{"aa": 0.7220, "bwt": -0.0069},
        "IDF-HN":      {"aa": 0.6813, "bwt": -0.0441},
    },
    "wiki_facts": {
        "ER":          {"aa": 0.9780, "bwt": -0.0058},
        "classical_hn":{"aa": 0.4891, "bwt": -0.3963},
        "EWC":         {"aa": 0.9718, "bwt": -0.0192},
        "SparseMemory":{"aa": 0.9704, "bwt": -0.0145},
        "IDF-HN":      {"aa": 0.9700, "bwt": -0.0160},
        "GSS":         {"aa": 0.9674, "bwt": -0.0201},
    },
}

SENTEMB_DATASET_MAP = {
    "split_newsgroups_sentemb": "split_newsgroups",
    "wiki_facts_sentemb": "wiki_facts",
}


def collect_results() -> dict[str, dict[str, dict[str, list[float]]]]:
    """返回 {canonical_dataset: {model_display: {"aa": [...], "bwt": [...]}}}"""
    results: dict[str, dict[str, dict[str, list[float]]]] = {
        "split_newsgroups": {},
        "wiki_facts": {},
    }
    # 去重：(canonical, model, seed) → 只保留第一条（按路径排序）
    seen: set[tuple[str, str, int]] = set()

    for metrics_path in sorted(OUTPUTS_DIR.rglob("metrics.json")):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # 用 feature_type 字段区分（新格式），兼容旧格式（通过 dataset 名称）
        feature_type = data.get("feature_type", "tfidf_svd")
        dataset_raw = data.get("dataset", "")
        if feature_type == "sentence_emb" and dataset_raw in ("split_newsgroups", "wiki_facts"):
            canonical = dataset_raw
        else:
            canonical = SENTEMB_DATASET_MAP.get(dataset_raw)
        if canonical is None:
            continue

        model_raw = data.get("model", "")
        model = MODEL_DISPLAY.get(model_raw, model_raw)
        seed = data.get("seed", -1)
        aa = data.get("aa")
        bwt = data.get("bwt")
        if aa is None or bwt is None:
            continue

        dedup_key = (canonical, model, seed)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        if model not in results[canonical]:
            results[canonical][model] = {"aa": [], "bwt": []}
        results[canonical][model]["aa"].append(aa)
        results[canonical][model]["bwt"].append(bwt)

    return results


def print_table(
    canonical_dataset: str,
    results: dict[str, dict[str, list[float]]],
) -> None:
    baseline = TFIDF_BASELINE.get(canonical_dataset, {})
    label = canonical_dataset.replace("_", "-").title()
    print(f"\n{'='*80}")
    print(f"{label}  (sentence_emb: all-mpnet-base-v2, 768 dim)")
    print(f"{'='*80}")
    hdr = f"{'模型':<16} {'AA sentemb':<22} {'BWT sentemb':<22} {'BWT tfidf_svd':<16} {'ΔBWT'}"
    print(hdr)
    print("-" * 80)

    ordered = [m for m in MODELS_ORDER if m in results or m in baseline]
    other = [m for m in results if m not in ordered]
    for model in ordered + other:
        sentemb = results.get(model, {})
        n = len(sentemb.get("aa", []))
        if n > 0:
            aa_m = statistics.mean(sentemb["aa"])
            bwt_m = statistics.mean(sentemb["bwt"])
            aa_s = statistics.stdev(sentemb["aa"]) if n > 1 else 0.0
            bwt_s = statistics.stdev(sentemb["bwt"]) if n > 1 else 0.0
            aa_str = f"{aa_m:.4f}±{aa_s:.4f} (n={n})"
            bwt_str = f"{bwt_m:.4f}±{bwt_s:.4f}"
        else:
            aa_str = "—"
            bwt_str = "—"
            bwt_m = None

        ref = baseline.get(model)
        ref_str = f"{ref['bwt']:+.4f}" if ref else "—"
        if ref and bwt_m is not None:
            delta = bwt_m - ref["bwt"]
            delta_str = f"{delta:+.4f} ({'↑' if delta > 0 else '↓'})"
        else:
            delta_str = "—"

        print(f"{model:<16} {aa_str:<22} {bwt_str:<22} {ref_str:<16} {delta_str}")

    print("=" * 80)
    print("  ΔBWT > 0 = BWT 恶化（更负）；ΔBWT < 0 = BWT 改善（更接近 0）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["newsgroups", "wikifacts", "all"], default="all"
    )
    args = parser.parse_args()

    results = collect_results()

    if args.dataset in ("newsgroups", "all"):
        print_table("split_newsgroups", results["split_newsgroups"])
    if args.dataset in ("wikifacts", "all"):
        print_table("wiki_facts", results["wiki_facts"])


if __name__ == "__main__":
    main()
