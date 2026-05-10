"""Post-hoc analysis for completed continual-learning runs.

This script is intentionally read-only with respect to training outputs. It
parses Hydra run directories, reconstructs task accuracy matrices from logs,
and writes compact CSV/PNG artifacts under analysis/.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
ANALYSIS_DIR = ROOT / "analysis"

DATASETS = [
    "split_mnist",
    "split_cifar100",
    "permuted_mnist",
    "split_newsgroups",
    "wiki_facts",
]

MODEL_DISPLAY = {
    "idf_hn": "IDF-HN",
    "er": "ER",
    "gss": "GSS",
    "ewc": "EWC",
    "classical_hn": "ClassicalHN",
    "dmhn": "DMHN",
    "sparse_memory": "SparseMemory",
    "clipping_hn": "ClippingHN",
    "kv_cache": "KVCache",
    "idf_hn_distilbert": "IDF-HN-DistilBERT",
    "idf_hn_transformer": "IDF-HN-KV",
    "kv_cache_distilbert": "KVCache-DistilBERT",
    "distilbert_finetune": "DistilBERT",
    "distilbert_er": "DistilBERT-ER",
}


def parse_config(run_dir: Path) -> dict | None:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8", errors="replace")

    seed_m = re.search(r"^seed:\s*(\d+)", text, re.MULTILINE)
    model_m = re.search(r"^model:\s*\n(?:.*\n)*?\s+name:\s*([\w_]+)", text, re.MULTILINE)
    dataset_m = re.search(r"^dataset:\s*\n(?:.*\n)*?\s+name:\s*([\w_]+)", text, re.MULTILINE)
    epochs_m = re.search(r"^\s+n_epochs:\s*(\d+)", text, re.MULTILINE)
    mem_m = re.search(r"^\s+memory_size:\s*(\d+)", text, re.MULTILINE)
    feature_m = re.search(r"^\s+feature_type:\s*([\w_]+)", text, re.MULTILINE)
    replay_m = re.search(r"^\s+replay_size:\s*(\d+)", text, re.MULTILINE)
    replay_strategy_m = re.search(r"^\s+replay_strategy:\s*\"?([\w_]+)\"?", text, re.MULTILINE)
    gamma0_m = re.search(r"^\s+gamma_0:\s*([\d.]+)", text, re.MULTILINE)
    delta_gamma_m = re.search(r"^\s+delta_gamma:\s*([\d.]+)", text, re.MULTILINE)
    write_threshold_m = re.search(r"^\s+write_threshold:\s*(-?[\d.]+|null|None)", text, re.MULTILINE)
    forget_mode_m = re.search(r"^\s+forget_mode:\s*([\w_]+)", text, re.MULTILINE)

    if not (seed_m and model_m and dataset_m):
        return None

    return {
        "seed": int(seed_m.group(1)),
        "model": model_m.group(1),
        "dataset": dataset_m.group(1),
        "n_epochs": int(epochs_m.group(1)) if epochs_m else None,
        "memory_size": int(mem_m.group(1)) if mem_m else None,
        "feature_type": feature_m.group(1) if feature_m else "",
        "replay_size": int(replay_m.group(1)) if replay_m else None,
        "replay_strategy": replay_strategy_m.group(1) if replay_strategy_m else "random",
        "gamma_0": float(gamma0_m.group(1)) if gamma0_m else None,
        "delta_gamma": float(delta_gamma_m.group(1)) if delta_gamma_m else None,
        "write_threshold": write_threshold_m.group(1) if write_threshold_m else "",
        "forget_mode": forget_mode_m.group(1) if forget_mode_m else "",
    }


def parse_log(run_dir: Path) -> dict | None:
    log_path = run_dir / "main.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")

    aa_m = re.search(r"AA\s*=\s*([\d.]+)", text)
    bwt_m = re.search(r"BWT\s*=\s*(-?[\d.]+)", text)
    fwt_m = re.search(r"FWT\s*=\s*(-?[\d.]+)", text)
    if not (aa_m and bwt_m and fwt_m):
        return None

    entries: list[tuple[int, int, float]] = []
    for m in re.finditer(r"Task\s+(\d+)\s+acc after Task\s+(\d+):\s+([\d.]+)", text):
        entries.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))

    pre_entries: list[tuple[int, int, float]] = []
    for m in re.finditer(r"Task\s+(\d+)\s+训练前准确率:\s+([\d.]+)", text):
        task = int(m.group(1))
        pre_entries.append((task, task - 1, float(m.group(2))))

    n_tasks = 0
    if entries:
        n_tasks = max(max(i, j) for i, j, _ in entries) + 1
    R = np.full((n_tasks, n_tasks), np.nan, dtype=float)
    for i, j, value in entries:
        R[i, j] = value

    R_pre = np.full((n_tasks, n_tasks), np.nan, dtype=float)
    for i, j, value in pre_entries:
        if 0 <= i < n_tasks and 0 <= j < n_tasks:
            R_pre[i, j] = value

    return {
        "AA": float(aa_m.group(1)),
        "BWT": float(bwt_m.group(1)),
        "FWT": float(fwt_m.group(1)),
        "R": R,
        "R_pre": R_pre,
    }


def load_runs(outputs_dir: Path) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)

    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg = parse_config(run_dir)
        log = parse_log(run_dir)
        if cfg is None or log is None:
            continue
        rec = {**cfg, **log, "run_dir": run_dir.name, "mtime": run_dir.stat().st_mtime}
        # Keep experimental variants separate. Later ablations often reuse the
        # same model/dataset/seed directory prefix, and taking only the newest
        # run would overwrite the paper's main configuration.
        key = (
            rec["model"],
            rec["dataset"],
            rec["seed"],
            rec["feature_type"] or "",
            rec["n_epochs"],
            rec["memory_size"],
            rec["replay_size"],
            rec["replay_strategy"],
            rec["gamma_0"],
            rec["delta_gamma"],
            rec["write_threshold"],
            rec["forget_mode"],
        )
        grouped[key].append(rec)

    latest = []
    for recs in grouped.values():
        latest.append(max(recs, key=lambda r: (r["mtime"], r["run_dir"])))
    return latest


def variant_signature(r: dict) -> str:
    return (
        f"epochs={r['n_epochs']};mem={r['memory_size']};replay={r['replay_size']};"
        f"strategy={r['replay_strategy']};gamma0={r['gamma_0']};"
        f"dgamma={r['delta_gamma']};wt={r['write_threshold']};mode={r['forget_mode']}"
    )


def is_canonical_main(r: dict) -> bool:
    """Select the paper-facing main configuration from all completed variants."""
    expected_mem = {
        "split_mnist": 1000,
        "split_cifar100": 5000,
        "permuted_mnist": 1000,
        "split_newsgroups": 1000,
        "wiki_facts": 1000,
    }
    expected_epochs = {
        "split_mnist": 2,
        "split_cifar100": 2,
        "permuted_mnist": 2,
        "split_newsgroups": 5,
        "wiki_facts": 5,
    }

    dataset = r["dataset"]
    model = r["model"]

    if dataset not in expected_epochs:
        return False
    if r["n_epochs"] != expected_epochs[dataset]:
        return False
    if r["replay_strategy"] != "random":
        return False

    # Text feature experiments are kept as their own canonical rows.
    if r["feature_type"] not in ("", "tfidf_svd", "sentence_emb"):
        return False

    if model in {"idf_hn", "er", "gss", "classical_hn", "dmhn", "sparse_memory", "clipping_hn"}:
        if r["memory_size"] is not None and r["memory_size"] != expected_mem[dataset]:
            return False

    if model == "idf_hn":
        if r["gamma_0"] is not None and abs(r["gamma_0"] - 0.1) > 1e-9:
            return False
        if r["delta_gamma"] is not None and abs(r["delta_gamma"] - 0.5) > 1e-9:
            return False

    # Exclude text finetuning Phase-3 models from the pre-Phase-3 canonical table.
    if model in {"idf_hn_distilbert", "idf_hn_transformer", "kv_cache_distilbert", "distilbert_finetune", "distilbert_er"}:
        return False

    return True


def latest_by_main_key(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for r in results:
        if not is_canonical_main(r):
            continue
        feature = r["feature_type"] or "default"
        grouped[(r["dataset"], feature, r["model"], r["seed"])].append(r)
    return [max(recs, key=lambda r: (r["mtime"], r["run_dir"])) for recs in grouped.values()]


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_summary(
    results: list[dict],
    out_dir: Path,
    filename: str = "paired_seed_differences.csv",
) -> None:
    pairs = [
        ("idf_hn", "er"),
        ("idf_hn", "gss"),
        ("idf_hn", "sparse_memory"),
        ("idf_hn", "ewc"),
        ("idf_hn", "classical_hn"),
        ("idf_hn", "dmhn"),
        ("er", "sparse_memory"),
    ]

    rows = []
    by_key: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for r in results:
        feature = r["feature_type"] or ""
        by_key[(r["dataset"], r["model"], feature)][r["seed"]] = r

    for dataset in DATASETS:
        features = sorted({r["feature_type"] or "" for r in results if r["dataset"] == dataset})
        for feature in features:
            for a, b in pairs:
                a_by_seed = by_key.get((dataset, a, feature), {})
                b_by_seed = by_key.get((dataset, b, feature), {})
                seeds = sorted(set(a_by_seed) & set(b_by_seed))
                if not seeds:
                    continue
                for metric in ("AA", "BWT", "FWT"):
                    diffs = [a_by_seed[s][metric] - b_by_seed[s][metric] for s in seeds]
                    mean, std = mean_std(diffs)
                    se = std / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
                    ci95 = 1.96 * se
                    better = "a" if mean > 0 else "b" if mean < 0 else "tie"
                    rows.append({
                        "dataset": dataset,
                        "feature_type": feature,
                        "model_a": a,
                        "model_b": b,
                        "metric": metric,
                        "seeds": " ".join(str(s) for s in seeds),
                        "n": len(seeds),
                        "mean_diff_a_minus_b": f"{mean:.6f}",
                        "std_diff": f"{std:.6f}",
                        "ci95_half_width": f"{ci95:.6f}",
                        "direction": better,
                    })

    out_path = out_dir / filename
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def canonical_paired_summary(results: list[dict], out_dir: Path) -> None:
    canonical = latest_by_main_key(results)
    paired_summary(canonical, out_dir, filename="canonical_paired_seed_differences.csv")


def write_run_summary(results: list[dict], out_dir: Path) -> None:
    rows = []
    for r in sorted(results, key=lambda x: (x["dataset"], x["model"], x["seed"], x["feature_type"])):
        rows.append({
            "dataset": r["dataset"],
            "feature_type": r["feature_type"],
            "variant": variant_signature(r),
            "model": r["model"],
            "seed": r["seed"],
            "AA": f"{r['AA']:.6f}",
            "BWT": f"{r['BWT']:.6f}",
            "FWT": f"{r['FWT']:.6f}",
            "n_epochs": r["n_epochs"],
            "memory_size": r["memory_size"],
            "replay_size": r["replay_size"],
            "run_dir": r["run_dir"],
        })
    out_path = out_dir / "latest_run_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_matrices(results: list[dict], out_dir: Path) -> None:
    matrix_dir = out_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        R = r["R"]
        if R.size == 0:
            continue
        stem = f"{r['dataset']}__{r['feature_type'] or 'default'}__{r['model']}__seed{r['seed']}"
        np.savetxt(matrix_dir / f"{stem}__R.csv", R, delimiter=",", fmt="%.6f")

        final = R[:, -1]
        diag = np.diag(R)
        forgetting = final - diag
        with (matrix_dir / f"{stem}__forgetting.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "diag_acc", "final_acc", "final_minus_diag"])
            for i in range(R.shape[0]):
                writer.writerow([i, f"{diag[i]:.6f}", f"{final[i]:.6f}", f"{forgetting[i]:.6f}"])


def plot_heatmaps(results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[WARN] matplotlib unavailable, skipping heatmaps: {exc}")
        return

    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        ("split_cifar100", "", "idf_hn"),
        ("split_cifar100", "", "er"),
        ("split_cifar100", "", "sparse_memory"),
        ("permuted_mnist", "", "idf_hn"),
        ("permuted_mnist", "", "er"),
        ("split_newsgroups", "tfidf_svd", "idf_hn"),
        ("split_newsgroups", "sentence_emb", "idf_hn"),
        ("wiki_facts", "tfidf_svd", "idf_hn"),
        ("wiki_facts", "sentence_emb", "idf_hn"),
    ]

    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_key[(r["dataset"], r["feature_type"] or "", r["model"])].append(r)

    for dataset, feature, model in selected:
        recs = by_key.get((dataset, feature, model), [])
        if not recs:
            continue
        shape = recs[0]["R"].shape
        same_shape = [r for r in recs if r["R"].shape == shape and r["R"].size > 0]
        if not same_shape:
            continue
        mean_R = np.nanmean(np.stack([r["R"] for r in same_shape]), axis=0)
        title = f"{MODEL_DISPLAY.get(model, model)} on {dataset}"
        if feature:
            title += f" ({feature})"

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mean_R, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("After task")
        ax.set_ylabel("Eval task")
        ax.set_xticks(range(mean_R.shape[1]))
        ax.set_yticks(range(mean_R.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        safe_feature = feature or "default"
        fig.savefig(heatmap_dir / f"{dataset}__{safe_feature}__{model}__R_mean.png", dpi=180)
        plt.close(fig)

        final = mean_R[:, -1]
        diag = np.diag(mean_R)
        forgetting = final - diag
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(np.arange(len(forgetting)), forgetting, color="#476A6F")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Final minus diagonal: {title}")
        ax.set_xlabel("Task")
        ax.set_ylabel("Final - initial")
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{dataset}__{safe_feature}__{model}__forgetting.png", dpi=180)
        plt.close(fig)


def write_markdown_summary(results: list[dict], out_dir: Path) -> None:
    by_group: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_group[(r["dataset"], r["feature_type"] or "", r["model"])].append(r)

    lines = ["# Existing Results Analysis", ""]
    lines.append(f"Parsed latest completed runs: {len(results)}")
    lines.append("")
    lines.append("## Mean Metrics")
    lines.append("")
    lines.append("| Dataset | Feature | Model | n | AA | BWT | FWT |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for (dataset, feature, model), recs in sorted(by_group.items()):
        aa_m, aa_s = mean_std([r["AA"] for r in recs])
        bwt_m, bwt_s = mean_std([r["BWT"] for r in recs])
        fwt_m, fwt_s = mean_std([r["FWT"] for r in recs])
        lines.append(
            f"| {dataset} | {feature or 'default'} | {MODEL_DISPLAY.get(model, model)} | "
            f"{len(recs)} | {aa_m:.4f} +/- {aa_s:.4f} | "
            f"{bwt_m:.4f} +/- {bwt_s:.4f} | {fwt_m:.4f} +/- {fwt_s:.4f} |"
        )

    canonical = latest_by_main_key(results)
    if canonical:
        lines.append("")
        lines.append("## Canonical Main Metrics")
        lines.append("")
        lines.append("| Dataset | Feature | Model | n | AA | BWT | FWT |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        by_canon: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for r in canonical:
            by_canon[(r["dataset"], r["feature_type"] or "default", r["model"])].append(r)
        for (dataset, feature, model), recs in sorted(by_canon.items()):
            aa_m, aa_s = mean_std([r["AA"] for r in recs])
            bwt_m, bwt_s = mean_std([r["BWT"] for r in recs])
            fwt_m, fwt_s = mean_std([r["FWT"] for r in recs])
            lines.append(
                f"| {dataset} | {feature} | {MODEL_DISPLAY.get(model, model)} | "
                f"{len(recs)} | {aa_m:.4f} +/- {aa_s:.4f} | "
                f"{bwt_m:.4f} +/- {bwt_s:.4f} | {fwt_m:.4f} +/- {fwt_s:.4f} |"
            )

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Use `paired_seed_differences.csv` for seed-paired comparisons; with n=3, CI is descriptive rather than decisive.")
    lines.append("- Use `canonical_paired_seed_differences.csv` for paper-facing main configurations. `latest_run_metrics.csv` intentionally keeps all ablation variants.")
    lines.append("- Accuracy matrices in `matrices/` reconstruct R[i,j] from log lines, where i is eval task and j is the task just trained.")
    lines.append("- Heatmaps in `heatmaps/` average available seeds for selected model/dataset pairs.")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed continual-learning outputs")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR / "existing_results")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = load_runs(args.outputs_dir)
    if not results:
        raise SystemExit("No completed runs found.")

    write_run_summary(results, args.out_dir)
    paired_summary(results, args.out_dir)
    canonical_paired_summary(results, args.out_dir)
    write_matrices(results, args.out_dir)
    plot_heatmaps(results, args.out_dir)
    write_markdown_summary(results, args.out_dir)

    print(f"Parsed {len(results)} latest completed runs")
    print(f"Wrote analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
