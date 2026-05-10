"""收集优先级回放对比实验结果（Split-CIFAR-100）。

从 Hydra 输出目录解析 metrics.json，汇总 random vs energy_priority
两种策略的 AA / BWT 均值和标准差。

用法：
    cd idf-hn
    uv run python run/collect_priority_replay_results.py
"""
import json
import statistics
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
STRATEGIES = ["random", "energy_priority"]


def collect() -> None:
    results: dict[str, dict[str, list[float]]] = {
        s: {"aa": [], "bwt": []} for s in STRATEGIES
    }
    seen: set[tuple[str, int]] = set()  # (strategy, seed)

    for metrics_path in sorted(OUTPUTS_DIR.rglob("metrics.json")):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("dataset") != "split_cifar100":
            continue
        if data.get("model") != "idf_hn":
            continue

        strategy = data.get("replay_strategy", "random")
        if strategy not in STRATEGIES:
            continue

        aa = data.get("aa")
        bwt = data.get("bwt")
        seed = data.get("seed", -1)
        if aa is None or bwt is None:
            continue

        dedup_key = (strategy, seed)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        results[strategy]["aa"].append(aa)
        results[strategy]["bwt"].append(bwt)

    print("=" * 62)
    print("ForgetGate 优先级回放对比：Split-CIFAR-100（idf_hn）")
    print("=" * 62)
    print(f"{'策略':<22} {'AA (mean±std)':<24} {'BWT (mean±std)':<24} n")
    print("-" * 62)
    for strategy in STRATEGIES:
        aa_list = results[strategy]["aa"]
        bwt_list = results[strategy]["bwt"]
        n = len(aa_list)
        if n == 0:
            print(f"{strategy:<22} {'—':<24} {'—':<24} 0")
            continue
        aa_mean = statistics.mean(aa_list)
        bwt_mean = statistics.mean(bwt_list)
        aa_std = statistics.stdev(aa_list) if n > 1 else 0.0
        bwt_std = statistics.stdev(bwt_list) if n > 1 else 0.0
        print(
            f"{strategy:<22} "
            f"{aa_mean:.4f} ± {aa_std:.4f}       "
            f"{bwt_mean:.4f} ± {bwt_std:.4f}       "
            f"{n}"
        )

    print("=" * 62)

    random_bwt = results["random"]["bwt"]
    priority_bwt = results["energy_priority"]["bwt"]
    if random_bwt and priority_bwt:
        delta = statistics.mean(priority_bwt) - statistics.mean(random_bwt)
        direction = "改善（↑）" if delta > 0 else "退化（↓）"
        rel = abs(delta) / abs(statistics.mean(random_bwt)) * 100
        print(f"\nBWT 变化：energy_priority vs random")
        print(f"  绝对差：{delta:+.4f}  方向：{direction}")
        print(f"  相对改善：{rel:.1f}%")


if __name__ == "__main__":
    collect()
