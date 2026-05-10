"""收集 Phase 3 OCL 消融实验结果。

消融维度：
  - forget_off:  forget_mode=none（无遗忘门，对照完整 IDF-HN-Transformer）
  - fifo_evict:  eviction_policy=fifo（FIFO 驱逐，对照 norm_min 智能驱逐）

用法：
    cd idf-hn
    uv run python run/collect_phase3_ablations.py
"""
import json
import statistics
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# OCL 主实验参考值（从 collect_phase3_results.py 结果中读取）
# forget_mode=input_dependent, eviction_policy=norm_min, n_epochs=1
OCL_REFERENCE = {
    "split_newsgroups_text": {"aa": 0.7615, "bwt": 0.0280},
    "wiki_facts_text":       {"aa": 0.9941, "bwt": -0.0039},
}

ABLATION_DISPLAY = {
    "none":    "ForgetGate-OFF (forget_mode=none)",
    "fifo":    "FIFO-Eviction  (eviction_policy=fifo)",
}

DATASET_DISPLAY = {
    "split_newsgroups_text": "Split-Newsgroups",
    "wiki_facts_text":       "Wiki-Facts",
}


def collect() -> dict:
    """返回 {dataset: {ablation_key: {"aa": [...], "bwt": [...]}}}"""
    results: dict = {ds: {} for ds in DATASET_DISPLAY}
    seen: set = set()

    for metrics_path in sorted(OUTPUTS_DIR.rglob("metrics.json")):
        if "phase3_ablation" not in str(metrics_path):
            continue
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("model") != "idf_hn_transformer":
            continue
        if data.get("n_epochs") != 1:
            continue

        dataset = data.get("dataset", "")
        if dataset not in results:
            continue

        forget_mode = data.get("forget_mode")
        eviction_policy = data.get("eviction_policy")
        seed = data.get("seed", -1)
        aa = data.get("aa")
        bwt = data.get("bwt")
        if aa is None or bwt is None:
            continue

        # 确定消融 key：优先 forget_mode 变体，其次 eviction_policy 变体
        if forget_mode == "none":
            ablation_key = "none"
        elif eviction_policy == "fifo":
            ablation_key = "fifo"
        else:
            continue  # 完整配置，跳过（由主实验处理）

        dedup = (dataset, ablation_key, seed)
        if dedup in seen:
            continue
        seen.add(dedup)

        bucket = results[dataset]
        if ablation_key not in bucket:
            bucket[ablation_key] = {"aa": [], "bwt": []}
        bucket[ablation_key]["aa"].append(aa)
        bucket[ablation_key]["bwt"].append(bwt)

    return results


def print_table(dataset: str, ablation_results: dict) -> None:
    ref = OCL_REFERENCE.get(dataset, {})
    ds_label = DATASET_DISPLAY.get(dataset, dataset)

    print(f"\n{'='*85}")
    print(f"[OCL Ablation] {ds_label}  (n_epochs=1, model=idf_hn_transformer)")
    print(f"{'='*85}")
    print(f"{'变体':<45} {'AA':<20} {'BWT':<20}")
    print("-" * 85)

    # 先打印参考行
    ref_aa = ref.get("aa", float("nan"))
    ref_bwt = ref.get("bwt", float("nan"))
    print(f"{'IDF-HN-Transformer (完整，参考)':<45} {ref_aa:.4f}               {ref_bwt:+.4f}")
    print("-" * 85)

    for ablation_key, label in ABLATION_DISPLAY.items():
        d = ablation_results.get(ablation_key, {})
        n = len(d.get("aa", []))
        if n > 0:
            aa_m = statistics.mean(d["aa"])
            bwt_m = statistics.mean(d["bwt"])
            aa_s = statistics.stdev(d["aa"]) if n > 1 else 0.0
            bwt_s = statistics.stdev(d["bwt"]) if n > 1 else 0.0
            aa_str = f"{aa_m:.4f}±{aa_s:.4f} (n={n})"
            bwt_str = f"{bwt_m:+.4f}±{bwt_s:.4f}"
            delta_bwt = bwt_m - ref_bwt
            print(f"{label:<45} {aa_str:<20} {bwt_str:<20} ΔBWT={delta_bwt:+.4f}")
        else:
            print(f"{label:<45} {'—':<20} {'—':<20}")

    print("=" * 85)


def main() -> None:
    all_results = collect()

    any_result = False
    for dataset in DATASET_DISPLAY:
        ablation_results = all_results.get(dataset, {})
        if ablation_results:
            print_table(dataset, ablation_results)
            any_result = True

    if not any_result:
        print("\n未找到 Phase 3 消融结果。请先运行：")
        print("  uv run python run/run_phase3_ablations.py")


if __name__ == "__main__":
    main()
