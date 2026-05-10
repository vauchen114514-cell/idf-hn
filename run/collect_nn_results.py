"""Collect Neural Networks journal experiment artifacts.

This collector reads Hydra output directories and writes paper-facing CSV/MD
tables under analysis/neural_networks/. It is intentionally tolerant of partial
completion so that long experiment groups can be inspected incrementally.
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
OUT_DIR = ROOT / "analysis" / "neural_networks"

MODEL_DISPLAY = {
    "idf_hn": "IDF-HN",
    "er": "ER",
    "gss": "GSS",
    "ewc": "EWC",
    "classical_hn": "ClassicalHN",
    "clipping_hn": "ClippingHN",
    "dmhn": "DMHN",
    "sparse_memory": "SparseMemory",
}


def _search(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else default


def parse_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    log_path = run_dir / "main.log"
    if not cfg_path.exists() or not log_path.exists():
        return None

    cfg = cfg_path.read_text(encoding="utf-8", errors="replace")
    log = log_path.read_text(encoding="utf-8", errors="replace")

    aa = _search(r"AA\s*=\s*([\d.]+)", log)
    bwt = _search(r"BWT\s*=\s*(-?[\d.]+)", log)
    fwt = _search(r"FWT\s*=\s*(-?[\d.]+)", log)
    if not (aa and bwt and fwt):
        return None

    model = _search(r"^model:\s*\n(?:.*\n)*?\s+name:\s*([\w_]+)", cfg)
    dataset = _search(r"^dataset:\s*\n(?:.*\n)*?\s+name:\s*([\w_]+)", cfg)
    seed = _search(r"^seed:\s*(\d+)", cfg)
    if not (model and dataset and seed):
        return None

    rec = {
        "run_dir": run_dir.name,
        "mtime": run_dir.stat().st_mtime,
        "model": model,
        "dataset": dataset,
        "seed": int(seed),
        "AA": float(aa),
        "BWT": float(bwt),
        "FWT": float(fwt),
        "n_epochs": int(_search(r"^\s+n_epochs:\s*(\d+)", cfg, "0")),
        "memory_size": int(_search(r"^\s+memory_size:\s*(\d+)", cfg, "0") or "0"),
        "replay_size": int(_search(r"^\s+replay_size:\s*(\d+)", cfg, "0") or "0"),
        "replay_strategy": _search(r"^\s+replay_strategy:\s*\"?([\w_]+)\"?", cfg, "random"),
        "feature_type": _search(r"^\s+feature_type:\s*([\w_]+)", cfg, ""),
        "forget_mode": _search(r"^\s+forget_mode:\s*([\w_]+)", cfg, ""),
        "eviction_policy": _search(r"^\s+eviction_policy:\s*([\w_]+)", cfg, ""),
        "dual_buffer": _search(r"^\s+dual_buffer:\s*(true|false)", cfg, ""),
        "gamma_0": float(_search(r"^\s+gamma_0:\s*([\d.]+)", cfg, "nan")),
        "delta_gamma": float(_search(r"^\s+delta_gamma:\s*([\d.]+)", cfg, "nan")),
        "bank_type": _search(r"^\s+type:\s*([\w_]+)", cfg, ""),
        "diagnostics": (run_dir / "diagnostics.csv").exists(),
        "nn_group": _search(r"^nn_group:\s*([\w_]+)", cfg, ""),
        "nn_label": _search(r"^nn_label:\s*([\w_\-=.]+)", cfg, ""),
    }

    if rec["model"] == "idf_hn":
        if rec["forget_mode"] == "static_density":
            rec["forget_label"] = "static_density"
        elif rec["forget_mode"] == "none" or (
            abs(rec["gamma_0"]) < 1e-12 and abs(rec["delta_gamma"]) < 1e-12
        ):
            rec["forget_label"] = "none"
        elif abs(rec["delta_gamma"]) < 1e-12:
            rec["forget_label"] = "time_decay"
        else:
            suffix = "_fifo" if rec["eviction_policy"] == "fifo" else ""
            rec["forget_label"] = f"input_dependent{suffix}"
    else:
        rec["forget_label"] = ""

    return rec


def load_runs(outputs_dir: Path) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rec = parse_run(run_dir)
        if rec is None:
            continue
        key = (
            rec["model"],
            rec["dataset"],
            rec["seed"],
            rec["n_epochs"],
            rec["memory_size"],
            rec["replay_size"],
            rec["replay_strategy"],
            rec["feature_type"],
            rec["forget_label"],
            rec["eviction_policy"],
            rec["dual_buffer"],
            rec["bank_type"],
            rec["diagnostics"],
        )
        grouped[key].append(rec)
    return [max(recs, key=lambda r: (r["mtime"], r["run_dir"])) for recs in grouped.values()]


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)

    out = []
    for key, recs in sorted(groups.items()):
        aa_m, aa_s = mean_std([r["AA"] for r in recs])
        bwt_m, bwt_s = mean_std([r["BWT"] for r in recs])
        fwt_m, fwt_s = mean_std([r["FWT"] for r in recs])
        row = {k: v for k, v in zip(keys, key)}
        row.update({
            "n": len(recs),
            "seeds": " ".join(str(r["seed"]) for r in sorted(recs, key=lambda x: x["seed"])),
            "AA_mean": f"{aa_m:.6f}",
            "AA_std": f"{aa_s:.6f}",
            "BWT_mean": f"{bwt_m:.6f}",
            "BWT_std": f"{bwt_s:.6f}",
            "FWT_mean": f"{fwt_m:.6f}",
            "FWT_std": f"{fwt_s:.6f}",
        })
        out.append(row)
    return out


def write_markdown(out_dir: Path, tables: dict[str, list[dict]]) -> None:
    lines = ["# Neural Networks Experiment Summary", ""]
    for name, rows in tables.items():
        lines += [f"## {name}", ""]
        if not rows:
            lines += ["No rows yet.", ""]
            continue
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def collect_diagnostics(rows: list[dict], out_dir: Path) -> list[dict]:
    diag_rows = []
    for r in rows:
        if not r["diagnostics"]:
            continue
        path = OUTPUTS_DIR / r["run_dir"] / "diagnostics.csv"
        if not path.exists():
            continue
        vals = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals.append(row)
        if not vals:
            continue
        conflicts = np.asarray([float(v["conflict"]) for v in vals], dtype=float)
        gammas = np.asarray([float(v["gamma"]) for v in vals], dtype=float)
        energy_norm = np.asarray([float(v["energy_mean_norm"]) for v in vals], dtype=float)
        replay_norm = np.asarray([float(v["replay_mean_norm"]) for v in vals], dtype=float)
        diag_rows.append({
            "dataset": r["dataset"],
            "forget_label": r["forget_label"],
            "seed": r["seed"],
            "run_dir": r["run_dir"],
            "n_steps": len(vals),
            "conflict_mean": f"{float(conflicts.mean()):.6f}",
            "conflict_p95": f"{float(np.quantile(conflicts, 0.95)):.6f}",
            "gamma_mean": f"{float(gammas.mean()):.6f}",
            "gamma_p95": f"{float(np.quantile(gammas, 0.95)):.6f}",
            "energy_norm_final": f"{float(energy_norm[-1]):.6f}",
            "replay_norm_final": f"{float(replay_norm[-1]):.6f}",
        })

    write_csv(out_dir / "diagnostics_summary.csv", diag_rows)
    return diag_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Neural Networks experiment results")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.outputs_dir)
    write_csv(args.out_dir / "all_latest_variants.csv", rows)

    nn_main = [r for r in rows if r["nn_group"] == "main"]
    main_rows = nn_main if nn_main else [
        r for r in rows
        if r["dataset"] in {"split_cifar100", "permuted_mnist", "split_mnist"}
        and r["n_epochs"] == 2
        and r["replay_strategy"] == "random"
        and r["feature_type"] in {"", "tfidf_svd"}
        and not r["diagnostics"]
        and r["nn_group"] == ""
    ]
    main_summary = summarize(main_rows, ["dataset", "model", "memory_size"])
    write_csv(args.out_dir / "main_hopfield_family.csv", main_summary)

    nn_forget = [r for r in rows if r["nn_group"] == "forget"]
    forget_rows = nn_forget if nn_forget else [
        r for r in rows
        if r["model"] == "idf_hn"
        and r["dataset"] in {"split_cifar100", "permuted_mnist"}
        and r["n_epochs"] == 2
        and r["memory_size"] in {1000, 5000}
        and r["replay_strategy"] == "random"
        and not r["diagnostics"]
        and r["nn_group"] == ""
    ]
    forget_summary = summarize(forget_rows, ["dataset", "forget_label", "eviction_policy", "dual_buffer", "memory_size"])
    write_csv(args.out_dir / "forget_mechanism_ablation.csv", forget_summary)

    nn_memory = [r for r in rows if r["nn_group"] == "memory"]
    memory_rows = nn_memory if nn_memory else [
        r for r in rows
        if r["dataset"] == "split_cifar100"
        and r["model"] in {"idf_hn", "er", "sparse_memory", "classical_hn"}
        and r["n_epochs"] == 2
        and r["memory_size"] in {100, 200, 500, 1000, 2000, 5000}
        and not r["diagnostics"]
        and r["nn_group"] == ""
    ]
    memory_summary = summarize(memory_rows, ["dataset", "model", "memory_size"])
    write_csv(args.out_dir / "memory_budget_sweep.csv", memory_summary)

    nn_buffer = [r for r in rows if r["nn_group"] == "buffer"]
    buffer_rows = nn_buffer if nn_buffer else [
        r for r in rows
        if r["model"] == "idf_hn"
        and r["dataset"] in {"split_mnist", "permuted_mnist"}
        and r["n_epochs"] == 2
        and r["memory_size"] == 1000
        and r["dual_buffer"] in {"true", "false"}
        and not r["diagnostics"]
        and r["nn_group"] == ""
    ]
    buffer_summary = summarize(buffer_rows, ["dataset", "dual_buffer"])
    write_csv(args.out_dir / "dual_buffer_ablation.csv", buffer_summary)

    diag_summary = collect_diagnostics(rows, args.out_dir)

    write_markdown(args.out_dir, {
        "Main Hopfield Family": main_summary,
        "Forgetting Mechanism Ablation": forget_summary,
        "Memory Budget Sweep": memory_summary,
        "Dual Buffer Ablation": buffer_summary,
        "Diagnostics Summary": diag_summary,
    })

    print(f"Loaded {len(rows)} latest variant rows")
    print(f"Wrote Neural Networks analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
