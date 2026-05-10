"""Synthetic drift diagnostics for input-dependent forgetting.

The goal is not to optimize benchmark accuracy. This experiment isolates the
mechanism: does a Hopfield-style conflict signal rise when the data stream
shifts, and does the induced gamma track drift strength?
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "synthetic_drift"


@dataclass
class DriftConfig:
    seed: int
    drift_type: str
    overlap: str
    dim: int = 32
    n_tasks: int = 6
    steps_per_task: int = 250
    memory_size: int = 400
    beta: float = 10.0
    gamma_0: float = 0.002
    delta_gamma: float = 0.04
    tau_percentile: float = 75.0
    tau_warmup: int = 50
    noise: float = 0.18


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def overlap_angle(name: str) -> float:
    if name == "low":
        return math.radians(85)
    if name == "medium":
        return math.radians(45)
    if name == "high":
        return math.radians(15)
    raise ValueError(f"unknown overlap: {name}")


def make_centers(cfg: DriftConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 1009)
    base = rng.normal(size=cfg.dim)
    base /= np.linalg.norm(base)

    ortho = rng.normal(size=cfg.dim)
    ortho -= ortho.dot(base) * base
    ortho /= np.linalg.norm(ortho)

    angle = overlap_angle(cfg.overlap)
    centers = []
    for t in range(cfg.n_tasks):
        if cfg.drift_type == "abrupt":
            theta = t * angle
        elif cfg.drift_type == "gradual":
            theta = t * angle / max(cfg.n_tasks - 1, 1)
        elif cfg.drift_type == "cyclic":
            theta = (t % 3) * angle
        else:
            raise ValueError(f"unknown drift_type: {cfg.drift_type}")
        c = math.cos(theta) * base + math.sin(theta) * ortho
        centers.append(c)
    return np.asarray(centers, dtype=float)


def sample_stream(cfg: DriftConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    centers = make_centers(cfg)
    xs = []
    labels = []
    for task in range(cfg.n_tasks):
        for _ in range(cfg.steps_per_task):
            x = centers[task] + cfg.noise * rng.normal(size=cfg.dim)
            xs.append(x)
            labels.append(task)
    X = l2_normalize(np.asarray(xs, dtype=float))
    y = np.asarray(labels, dtype=int)
    return X, y


class SyntheticIDFMemory:
    def __init__(self, cfg: DriftConfig) -> None:
        self.cfg = cfg
        self.mem = np.zeros((cfg.memory_size, cfg.dim), dtype=float)
        self.labels = np.full(cfg.memory_size, -1, dtype=int)
        self.n_stored = 0
        self.conflicts: list[float] = []
        self.write_count = 0

    def retrieve(self, u: np.ndarray) -> np.ndarray:
        if self.n_stored == 0:
            return u.copy()
        M = self.mem[: self.n_stored]
        scores = self.cfg.beta * (M @ u)
        weights = softmax(scores)
        return weights @ M

    def conflict(self, u: np.ndarray, xi: np.ndarray) -> float:
        # A bounded, interpretable conflict proxy: 1 - cosine(retrieved, input).
        denom = max(np.linalg.norm(xi) * np.linalg.norm(u), 1e-12)
        return float(1.0 - np.dot(xi, u) / denom)

    def tau(self) -> float:
        if len(self.conflicts) < self.cfg.tau_warmup:
            return 0.25
        return float(np.percentile(self.conflicts, self.cfg.tau_percentile))

    def gamma(self, conflict: float) -> float:
        tau = self.tau()
        return self.cfg.gamma_0 + self.cfg.delta_gamma * sigmoid(conflict - tau)

    def update(self, u: np.ndarray, label: int) -> dict:
        if self.n_stored == 0:
            conflict = 0.0
            gamma = self.cfg.gamma_0
            tau = self.tau()
            xi = u
            self.store(u, label)
        else:
            xi = self.retrieve(u)
            conflict = self.conflict(u, xi)
            tau = self.tau()
            gamma = self.gamma(conflict)
            self.mem[: self.n_stored] *= 1.0 - gamma
            self.store(u, label)

        self.conflicts.append(conflict)
        return {
            "conflict": conflict,
            "gamma": gamma,
            "tau": tau,
            "retrieval_cosine": 1.0 - conflict,
            "n_stored": self.n_stored,
        }

    def store(self, u: np.ndarray, label: int) -> None:
        self.write_count += 1
        if self.n_stored < self.cfg.memory_size:
            slot = self.n_stored
            self.n_stored += 1
        else:
            norms = np.linalg.norm(self.mem, axis=1)
            slot = int(np.argmin(norms))
        self.mem[slot] = u
        self.labels[slot] = label

    def memory_label_counts(self) -> dict[int, int]:
        labels = self.labels[: self.n_stored]
        return {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}

    def mean_norm(self) -> float:
        if self.n_stored == 0:
            return 0.0
        return float(np.linalg.norm(self.mem[: self.n_stored], axis=1).mean())


def run_one(cfg: DriftConfig) -> tuple[list[dict], list[dict]]:
    X, y = sample_stream(cfg)
    memory = SyntheticIDFMemory(cfg)

    step_rows: list[dict] = []
    task_rows: list[dict] = []

    for step, (u, label) in enumerate(zip(X, y, strict=True)):
        stats = memory.update(u, int(label))
        boundary = step > 0 and y[step] != y[step - 1]
        row = {
            "seed": cfg.seed,
            "drift_type": cfg.drift_type,
            "overlap": cfg.overlap,
            "step": step,
            "task": int(label),
            "is_boundary": int(boundary),
            **stats,
            "mean_mem_norm": memory.mean_norm(),
        }
        step_rows.append(row)

        if (step + 1) % cfg.steps_per_task == 0:
            counts = memory.memory_label_counts()
            current_task = int(label)
            old_count = sum(v for k, v in counts.items() if k < current_task)
            current_count = counts.get(current_task, 0)
            task_rows.append({
                "seed": cfg.seed,
                "drift_type": cfg.drift_type,
                "overlap": cfg.overlap,
                "task": current_task,
                "mean_conflict": np.mean([r["conflict"] for r in step_rows[-cfg.steps_per_task:]]),
                "mean_gamma": np.mean([r["gamma"] for r in step_rows[-cfg.steps_per_task:]]),
                "boundary_conflict": step_rows[step - cfg.steps_per_task + 1]["conflict"],
                "boundary_gamma": step_rows[step - cfg.steps_per_task + 1]["gamma"],
                "memory_current_task_count": current_count,
                "memory_old_task_count": old_count,
                "memory_n_stored": memory.n_stored,
                "mean_mem_norm": memory.mean_norm(),
            })

    return step_rows, task_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_task_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (row["drift_type"], row["overlap"], int(row["task"]))
        groups.setdefault(key, []).append(row)

    out = []
    for (drift_type, overlap, task), recs in sorted(groups.items()):
        for metric in [
            "mean_conflict",
            "mean_gamma",
            "boundary_conflict",
            "boundary_gamma",
            "memory_current_task_count",
            "memory_old_task_count",
            "mean_mem_norm",
        ]:
            vals = np.asarray([float(r[metric]) for r in recs], dtype=float)
            out.append({
                "drift_type": drift_type,
                "overlap": overlap,
                "task": task,
                "metric": metric,
                "mean": f"{vals.mean():.6f}",
                "std": f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:.6f}",
                "n": len(vals),
            })
    return out


def plot(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable, skipping plots: {exc}")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot one representative seed per scenario for readability.
    for drift_type in sorted({r["drift_type"] for r in rows}):
        for overlap in sorted({r["overlap"] for r in rows}):
            recs = [
                r for r in rows
                if r["drift_type"] == drift_type and r["overlap"] == overlap and int(r["seed"]) == 42
            ]
            if not recs:
                continue
            x = np.asarray([int(r["step"]) for r in recs])
            conflict = np.asarray([float(r["conflict"]) for r in recs])
            gamma = np.asarray([float(r["gamma"]) for r in recs])
            task = np.asarray([int(r["task"]) for r in recs])
            boundaries = x[np.r_[False, task[1:] != task[:-1]]]

            fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            axes[0].plot(x, conflict, linewidth=0.9, color="#345995")
            axes[0].set_ylabel("Conflict")
            axes[0].set_title(f"Synthetic drift: {drift_type}, overlap={overlap}, seed=42")
            axes[1].plot(x, gamma, linewidth=0.9, color="#D1495B")
            axes[1].set_ylabel("Gamma")
            axes[1].set_xlabel("Step")
            for ax in axes:
                for b in boundaries:
                    ax.axvline(b, color="black", alpha=0.25, linewidth=0.8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{drift_type}__{overlap}__seed42.png", dpi=180)
            plt.close(fig)


def write_summary(task_rows: list[dict], agg_rows: list[dict], out_dir: Path) -> None:
    lines = ["# Synthetic Drift Diagnostics", ""]
    lines.append("This experiment isolates the conflict/gamma mechanism on controlled Gaussian streams.")
    lines.append("")
    lines.append("## Boundary Response")
    lines.append("")
    lines.append("| Drift | Overlap | Mean boundary conflict | Mean boundary gamma | Mean task conflict |")
    lines.append("|---|---|---:|---:|---:|")

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in task_rows:
        if int(row["task"]) == 0:
            continue
        groups.setdefault((row["drift_type"], row["overlap"]), []).append(row)

    for (drift_type, overlap), recs in sorted(groups.items()):
        bc = np.asarray([float(r["boundary_conflict"]) for r in recs])
        bg = np.asarray([float(r["boundary_gamma"]) for r in recs])
        mc = np.asarray([float(r["mean_conflict"]) for r in recs])
        lines.append(
            f"| {drift_type} | {overlap} | {bc.mean():.4f} | {bg.mean():.4f} | {mc.mean():.4f} |"
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `step_metrics.csv`: per-step conflict, gamma, tau, and memory norm.")
    lines.append("- `task_metrics.csv`: per-task memory composition and boundary response.")
    lines.append("- `task_metric_summary.csv`: mean/std aggregation across seeds.")
    lines.append("- `plots/`: representative seed=42 conflict/gamma traces.")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic drift diagnostics")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--steps-per-task", type=int, default=250)
    args = parser.parse_args()

    step_rows: list[dict] = []
    task_rows: list[dict] = []
    for seed in args.seeds:
        for drift_type in ["abrupt", "gradual"]:
            for overlap in ["low", "medium", "high"]:
                cfg = DriftConfig(
                    seed=seed,
                    drift_type=drift_type,
                    overlap=overlap,
                    dim=args.dim,
                    steps_per_task=args.steps_per_task,
                )
                s_rows, t_rows = run_one(cfg)
                step_rows.extend(s_rows)
                task_rows.extend(t_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "step_metrics.csv", step_rows)
    write_csv(args.out_dir / "task_metrics.csv", task_rows)
    agg_rows = aggregate_task_rows(task_rows)
    write_csv(args.out_dir / "task_metric_summary.csv", agg_rows)
    plot(step_rows, args.out_dir)
    write_summary(task_rows, agg_rows, args.out_dir)

    print(f"Wrote synthetic drift diagnostics to {args.out_dir}")
    print(f"Rows: step={len(step_rows)}, task={len(task_rows)}")


if __name__ == "__main__":
    main()
