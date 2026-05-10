"""Run the Neural Networks journal experiment package.

The matrix is intentionally broader than the minimal submission package. It is
organized around the evidence Neural Networks reviewers are likely to expect:

1. Hopfield-family comparisons on canonical continual-learning benchmarks.
2. Forgetting-mechanism ablations with fixed training/replay settings.
3. Memory-budget sweeps for capacity/stability trade-offs.
4. Dual-buffer vs single-buffer failure-mode validation.
5. Low-cost diagnostics runs that write conflict/gamma/norm traces.

Usage:
    uv run python run/run_nn_experiments.py --dry-run
    uv run python run/run_nn_experiments.py --group forget --dry-run
    uv run python run/run_nn_experiments.py --group main --start 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 456, 789, 2026]


@dataclass(frozen=True)
class Experiment:
    group: str
    label: str
    model: str
    dataset: str
    seed: int
    overrides: tuple[str, ...]


def _mem_for_dataset(dataset: str, memory_size: int | None = None) -> int:
    if memory_size is not None:
        return memory_size
    if dataset == "split_cifar100":
        return 5000
    return 1000


def _common_overrides(dataset: str, memory_size: int | None = None) -> list[str]:
    mem = _mem_for_dataset(dataset, memory_size)
    epochs = 2 if dataset in {"split_mnist", "split_cifar100", "permuted_mnist"} else 5
    return [f"trainer.n_epochs={epochs}", f"model.memory_size={mem}"]


def _make(
    group: str,
    label: str,
    model: str,
    dataset: str,
    seeds: list[int],
    overrides: list[str],
) -> list[Experiment]:
    return [
        Experiment(group, label, model, dataset, seed, tuple(overrides))
        for seed in seeds
    ]


def build_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # 1) Hopfield-family and reference baselines. These are the main paper rows.
    main_models = [
        ("idf_hn", "IDF-HN"),
        ("baselines/sparse_memory", "SparseMemory"),
        ("baselines/classical_hn", "ClassicalHN"),
        ("baselines/clipping_hn", "ClippingHN"),
        ("baselines/dmhn", "DMHN"),
        ("baselines/er", "ER-reference"),
        ("baselines/gss", "GSS-reference"),
        ("baselines/ewc", "EWC-reference"),
    ]
    for dataset in ["split_cifar100", "permuted_mnist", "split_mnist"]:
        for model, label in main_models:
            overrides = _common_overrides(dataset)
            if model in {"baselines/er", "baselines/gss"}:
                overrides = _common_overrides(dataset)
            exps.extend(_make("main", label, model, dataset, SEEDS, overrides))

    # 2) Forgetting-mechanism ablations. These isolate input-dependent forgetting.
    forget_variants = [
        ("none", [
            "model.forget_mode=none",
            "model.forget_gate.gamma_0=0",
            "model.forget_gate.delta_gamma=0",
        ]),
        ("time_decay", [
            "model.forget_mode=time_decay",
            "model.forget_gate.delta_gamma=0",
        ]),
        ("static_density", [
            "model.forget_mode=static_density",
        ]),
        ("input_dependent", [
            "model.forget_mode=input_dependent",
        ]),
        ("input_dependent_fifo", [
            "model.forget_mode=input_dependent",
            "model.eviction_policy=fifo",
        ]),
    ]
    for dataset in ["split_cifar100", "permuted_mnist"]:
        for label, extra in forget_variants:
            overrides = _common_overrides(dataset) + extra
            exps.extend(_make("forget", label, "idf_hn", dataset, SEEDS, overrides))

    # 3) Memory budget sweep. Capacity/stability trade-off for the central claim.
    memory_grid = [100, 200, 500, 1000, 2000, 5000]
    sweep_models = [
        ("idf_hn", "IDF-HN"),
        ("baselines/er", "ER"),
        ("baselines/sparse_memory", "SparseMemory"),
        ("baselines/classical_hn", "ClassicalHN"),
    ]
    for mem in memory_grid:
        for model, label in sweep_models:
            overrides = _common_overrides("split_cifar100", memory_size=mem)
            exps.extend(_make("memory", f"{label}_mem{mem}", model, "split_cifar100", SEEDS, overrides))

    # 4) Dual-buffer failure-mode validation. This directly supports the paper's
    # architectural argument and should be cheap relative to CIFAR sweeps.
    for dataset in ["split_mnist", "permuted_mnist"]:
        for label, dual in [("dual_buffer", "true"), ("single_buffer", "false")]:
            overrides = _common_overrides(dataset) + [f"model.dual_buffer={dual}"]
            exps.extend(_make("buffer", label, "idf_hn", dataset, SEEDS, overrides))

    # 5) Diagnostics traces. Use 3 seeds to avoid excessive CSV volume.
    diag_seeds = [42, 123, 456]
    diag_variants = [
        ("diag_input_dependent", [
            "model.forget_mode=input_dependent",
            "model.diagnostics_enabled=true",
        ]),
        ("diag_none", [
            "model.forget_mode=none",
            "model.forget_gate.gamma_0=0",
            "model.forget_gate.delta_gamma=0",
            "model.diagnostics_enabled=true",
        ]),
    ]
    for label, extra in diag_variants:
        overrides = _common_overrides("split_cifar100") + extra
        exps.extend(_make("diagnostics", label, "idf_hn", "split_cifar100", diag_seeds, overrides))

    return exps


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in value)


def build_cmd(exp: Experiment) -> list[str]:
    model_name = safe_name(exp.model.replace("baselines/", ""))
    label = safe_name(exp.label)
    run_name = (
        f"nn_{exp.group}_{label}_{model_name}_{exp.dataset}_s{exp.seed}_"
        "${now:%Y%m%d_%H%M%S}"
    )
    return [
        "uv",
        "run",
        "python",
        "run/main.py",
        f"model={exp.model}",
        f"dataset={exp.dataset}",
        f"seed={exp.seed}",
        f"+nn_group={exp.group}",
        f"+nn_label={exp.label}",
        f"run_name={run_name}",
        *exp.overrides,
    ]


def run_one(idx: int, total: int, exp: Experiment, dry_run: bool) -> bool:
    cmd = build_cmd(exp)
    print("\n" + "=" * 80)
    print(f"[{idx}/{total}] group={exp.group} label={exp.label} dataset={exp.dataset} seed={exp.seed}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)
    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] returncode={result.returncode} elapsed={elapsed:.0f}s")
        return False
    print(f"[OK] elapsed={elapsed:.0f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neural Networks journal experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--group", default=None, help="Run one group: main/forget/memory/buffer/diagnostics")
    parser.add_argument("--start", type=int, default=1, help="1-indexed start position after filtering")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs after filtering")
    args = parser.parse_args()

    exps = build_experiments()
    if args.group:
        exps = [e for e in exps if e.group == args.group]
    if args.limit is not None:
        exps = exps[: args.limit]

    print(f"Neural Networks experiment package: {len(exps)} runs")
    if args.group:
        print(f"Filtered group: {args.group}")
    if args.dry_run:
        print("DRY-RUN: commands will not be executed")

    failed: list[int] = []
    for idx, exp in enumerate(exps, start=1):
        if idx < args.start:
            continue
        ok = run_one(idx, len(exps), exp, args.dry_run)
        if not ok:
            failed.append(idx)
            print("[WARN] continuing after failed run")

    print("\n" + "=" * 80)
    print(f"Completed {len(exps) - len(failed)}/{len(exps)} runs")
    if failed:
        print("Failed run indices:", failed)
        print(f"Resume with --start {min(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
