"""收集并展示消融实验结果。

从 outputs/ 目录中读取配置，按 forget_mode / dreaming 设置分组，
打印消融对比表。

用法：
    cd idf-hn
    uv run python run/collect_ablation_results.py
    uv run python run/collect_ablation_results.py --dim forget
    uv run python run/collect_ablation_results.py --dim dreaming
    uv run python run/collect_ablation_results.py --dim efficiency
    uv run python run/collect_ablation_results.py --dim tau
"""
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
SEEDS = [42, 123, 456]


def parse_run_dir(run_dir: Path) -> dict | None:
    """解析实验目录，提取完整配置 + 指标。"""
    log_file = run_dir / "main.log"
    hydra_cfg = run_dir / ".hydra" / "config.yaml"

    if not log_file.exists() or not hydra_cfg.exists():
        return None

    cfg_text = hydra_cfg.read_text(encoding="utf-8", errors="replace")
    log_text = log_file.read_text(encoding="utf-8", errors="replace")

    seed_m = re.search(r"^seed:\s*(\d+)", cfg_text, re.MULTILINE)
    model_m = re.search(r"^\s+name:\s*(\w+)", cfg_text, re.MULTILINE)
    dataset_m = re.search(r"name:\s*(split_mnist|split_cifar100|permuted_mnist|split_newsgroups)", cfg_text)
    aa_m = re.search(r"AA\s*=\s*([\d.]+)", log_text)
    bwt_m = re.search(r"BWT\s*=\s*(-?[\d.]+)", log_text)

    if not (seed_m and model_m and dataset_m and aa_m and bwt_m):
        return None

    seed = int(seed_m.group(1))
    model_raw = model_m.group(1)
    dataset = dataset_m.group(1)

    if model_raw != "idf_hn":
        return None  # 消融实验只关注 idf_hn 变体

    # 提取消融相关字段
    forget_mode_m = re.search(r"forget_mode:\s*(\S+)", cfg_text)
    gamma_0_m = re.search(r"gamma_0:\s*([\d.]+)", cfg_text)
    delta_gamma_m = re.search(r"delta_gamma:\s*([\d.]+)", cfg_text)
    dreaming_enabled_m = re.search(r"enabled:\s*(true|false)", cfg_text)
    sem_thresh_m = re.search(r"semantic_threshold:\s*([\d.]+)", cfg_text)
    bank_type_m = re.search(r"type:\s*(prototype|exact|faiss)", cfg_text)
    adaptive_tau_m = re.search(r"adaptive_tau:\s*(true|false)", cfg_text)
    tau_fixed_m = re.search(r"^    tau:\s*([\d.]+)", cfg_text, re.MULTILINE)
    epochs_m = re.search(r"n_epochs:\s*(\d+)", cfg_text)

    forget_mode = forget_mode_m.group(1) if forget_mode_m else "input_dependent"
    gamma_0 = float(gamma_0_m.group(1)) if gamma_0_m else 0.1
    delta_gamma = float(delta_gamma_m.group(1)) if delta_gamma_m else 0.5
    dreaming_enabled = dreaming_enabled_m.group(1) == "true" if dreaming_enabled_m else False
    sem_thresh = float(sem_thresh_m.group(1)) if sem_thresh_m else 0.5
    bank_type = bank_type_m.group(1) if bank_type_m else "prototype"
    adaptive_tau = adaptive_tau_m.group(1) == "true" if adaptive_tau_m else True
    tau_fixed = float(tau_fixed_m.group(1)) if tau_fixed_m else 0.3
    n_epochs = int(epochs_m.group(1)) if epochs_m else 2

    # 判断消融维度标签
    if forget_mode == "static_density":
        ablation_label = "static_density"
    elif delta_gamma == 0.0 and gamma_0 == 0.0:
        ablation_label = "none"
    elif delta_gamma == 0.0:
        ablation_label = "time_decay"
    else:
        ablation_label = "input_dependent"

    if dreaming_enabled:
        if sem_thresh >= 2.0:
            dream_label = "random"
        else:
            dream_label = "semantic"
    else:
        dream_label = "none"

    efficiency_label = bank_type

    if not adaptive_tau:
        tau_label = f"tau={tau_fixed}"
    else:
        tau_label = "adaptive"

    return {
        "model":              model_raw,
        "dataset":            dataset,
        "seed":               seed,
        "forget_mode":        ablation_label,
        "dream_mode":         dream_label,
        "efficiency":         efficiency_label,
        "tau_label":          tau_label,
        "n_epochs":           n_epochs,
        "AA":                 float(aa_m.group(1)),
        "BWT":                float(bwt_m.group(1)),
        "run_dir":            str(run_dir.name),
    }


def load_ablation_results() -> list[dict]:
    """加载所有 idf_hn 实验，每个 (forget_mode, dream_mode, efficiency, tau, dataset, seed) 保留最新。"""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        rec = parse_run_dir(run_dir)
        if rec is None:
            continue
        key = (rec["forget_mode"], rec["dream_mode"], rec["efficiency"],
               rec["tau_label"], rec["dataset"], rec["seed"])
        grouped[key].append(rec)

    results = []
    for recs in grouped.values():
        results.append(max(recs, key=lambda r: r["run_dir"]))
    return results


def mean_std(vals: list[float]) -> str:
    if len(vals) >= 2:
        m = statistics.mean(vals)
        s = statistics.stdev(vals)
        return f"{m:.4f}±{s:.4f}"
    elif len(vals) == 1:
        return f"{vals[0]:.4f} (1seed)"
    return "—"


def print_forget_mechanism(results: list[dict], dataset: str = "split_mnist") -> None:
    """消融维度 1：遗忘机制。"""
    ORDER = ["none", "time_decay", "static_density", "input_dependent"]
    ds_results = [r for r in results if r["dataset"] == dataset
                  and r["dream_mode"] == "none"
                  and r["efficiency"] == "prototype"
                  and r["tau_label"] == "adaptive"]

    print(f"\n{'='*65}")
    print(f"  遗忘机制消融 | {dataset}")
    print(f"{'='*65}")
    print(f"  {'forget_mode':<20} {'AA':<20} {'BWT':<20} seeds")
    print(f"  {'-'*60}")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in ds_results:
        by_mode[r["forget_mode"]].append(r)

    for mode in ORDER:
        recs = sorted(by_mode.get(mode, []), key=lambda r: r["seed"])
        seeds = [r["seed"] for r in recs]
        aa_str = mean_std([r["AA"] for r in recs])
        bwt_str = mean_std([r["BWT"] for r in recs])
        print(f"  {mode:<20} {aa_str:<20} {bwt_str:<20} {seeds}")


def print_dreaming_ablation(results: list[dict], dataset: str = "split_mnist") -> None:
    """消融维度 2：Dreaming。"""
    ORDER = ["none", "random", "semantic"]
    ds_results = [r for r in results if r["dataset"] == dataset
                  and r["forget_mode"] == "input_dependent"
                  and r["efficiency"] == "prototype"
                  and r["tau_label"] == "adaptive"]

    print(f"\n{'='*65}")
    print(f"  Dreaming 消融 | {dataset}")
    print(f"{'='*65}")
    print(f"  {'dream_mode':<15} {'AA':<20} {'BWT':<20} seeds")
    print(f"  {'-'*60}")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in ds_results:
        by_mode[r["dream_mode"]].append(r)

    for mode in ORDER:
        recs = sorted(by_mode.get(mode, []), key=lambda r: r["seed"])
        seeds = [r["seed"] for r in recs]
        aa_str = mean_std([r["AA"] for r in recs])
        bwt_str = mean_std([r["BWT"] for r in recs])
        print(f"  {mode:<15} {aa_str:<20} {bwt_str:<20} {seeds}")


def print_efficiency_ablation(results: list[dict], dataset: str = "split_mnist") -> None:
    """消融维度 3：效率（密度计算方式）。"""
    ORDER = ["exact", "prototype"]
    ds_results = [r for r in results if r["dataset"] == dataset
                  and r["forget_mode"] == "input_dependent"
                  and r["dream_mode"] == "none"
                  and r["tau_label"] == "adaptive"]

    print(f"\n{'='*65}")
    print(f"  效率消融 | {dataset}")
    print(f"{'='*65}")
    print(f"  {'efficiency':<15} {'AA':<20} {'BWT':<20} seeds")
    print(f"  {'-'*60}")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in ds_results:
        by_mode[r["efficiency"]].append(r)

    for mode in ORDER:
        recs = sorted(by_mode.get(mode, []), key=lambda r: r["seed"])
        seeds = [r["seed"] for r in recs]
        aa_str = mean_std([r["AA"] for r in recs])
        bwt_str = mean_std([r["BWT"] for r in recs])
        print(f"  {mode:<15} {aa_str:<20} {bwt_str:<20} {seeds}")


def print_tau_sweep(results: list[dict], dataset: str = "split_mnist") -> None:
    """消融维度 4：τ sweep。"""
    ds_results = [r for r in results if r["dataset"] == dataset
                  and r["forget_mode"] == "input_dependent"
                  and r["dream_mode"] == "none"
                  and r["efficiency"] == "prototype"]

    # 只保留 tau=xxx 标签
    tau_results = [r for r in ds_results if r["tau_label"].startswith("tau=")]
    # 加入 adaptive（默认设置）
    adaptive_results = [r for r in ds_results if r["tau_label"] == "adaptive"]

    print(f"\n{'='*65}")
    print(f"  τ sweep 消融 | {dataset}")
    print(f"{'='*65}")
    print(f"  {'tau':<15} {'AA':<20} {'BWT':<20} seeds")
    print(f"  {'-'*60}")

    by_tau: dict[str, list[dict]] = defaultdict(list)
    for r in tau_results + adaptive_results:
        by_tau[r["tau_label"]].append(r)

    ordered_keys = sorted(
        [k for k in by_tau if k.startswith("tau=")],
        key=lambda t: float(t.split("=")[1])
    )
    if "adaptive" in by_tau:
        ordered_keys.insert(0, "adaptive")

    for tau_key in ordered_keys:
        recs = sorted(by_tau[tau_key], key=lambda r: r["seed"])
        seeds = [r["seed"] for r in recs]
        aa_str = mean_std([r["AA"] for r in recs])
        bwt_str = mean_std([r["BWT"] for r in recs])
        print(f"  {tau_key:<15} {aa_str:<20} {bwt_str:<20} {seeds}")


def main() -> None:
    parser = argparse.ArgumentParser(description="收集并展示消融实验结果")
    parser.add_argument("--dim", choices=["forget", "dreaming", "efficiency", "tau", "all"],
                        default="all", help="消融维度")
    parser.add_argument("--dataset", default="split_mnist",
                        choices=["split_mnist", "split_cifar100", "permuted_mnist", "split_newsgroups"])
    args = parser.parse_args()

    results = load_ablation_results()
    print(f"共加载 {len(results)} 条 idf_hn 消融实验记录")

    if args.dim in ("forget", "all"):
        print_forget_mechanism(results, args.dataset)
    if args.dim in ("dreaming", "all"):
        print_dreaming_ablation(results, args.dataset)
    if args.dim in ("efficiency", "all"):
        print_efficiency_ablation(results, args.dataset)
    if args.dim in ("tau", "all"):
        print_tau_sweep(results, args.dataset)


if __name__ == "__main__":
    main()
