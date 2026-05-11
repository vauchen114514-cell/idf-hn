"""Run only Neural Networks experiments missing from imported/cloud results.

This is intended for a single local GPU worker. It compares the full experiment
matrix from ``run_nn_experiments.py`` against:

1. an imported ``all_latest_variants.csv`` from a cloud run, and
2. any local runs already present under ``outputs/``.

Then it launches only the missing experiments sequentially.

Examples:
    uv run python run/run_missing_nn_local.py --dry-run
    uv run python run/run_missing_nn_local.py --group buffer
    uv run python run/run_missing_nn_local.py --group main --dataset split_mnist
    uv run python run/run_missing_nn_local.py --group memory --limit 2
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from collect_nn_results import load_runs
from run_nn_experiments import Experiment, build_cmd, build_experiments


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPORTED_CSV = (
    ROOT
    / "imported_nn_results_partial_20260511"
    / "analysis"
    / "neural_networks"
    / "all_latest_variants.csv"
)


def _model_name(value: str) -> str:
    return value.replace("baselines/", "")


def _key(group: str, label: str, model: str, dataset: str, seed: int | str) -> tuple[str, str, str, str, str]:
    return (group or "", label or "", _model_name(model or ""), dataset or "", str(seed))


def _experiment_key(exp: Experiment) -> tuple[str, str, str, str, str]:
    return _key(exp.group, exp.label, exp.model, exp.dataset, exp.seed)


def load_completed_from_csv(path: Path) -> set[tuple[str, str, str, str, str]]:
    completed: set[tuple[str, str, str, str, str]] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group = row.get("nn_group", "")
            label = row.get("nn_label", "")
            if not group:
                continue
            completed.add(
                _key(
                    group=group,
                    label=label,
                    model=row.get("model", ""),
                    dataset=row.get("dataset", ""),
                    seed=row.get("seed", ""),
                )
            )
    return completed


def load_completed_from_outputs(outputs_dir: Path) -> set[tuple[str, str, str, str, str]]:
    completed: set[tuple[str, str, str, str, str]] = set()
    if not outputs_dir.exists():
        return completed

    for row in load_runs(outputs_dir):
        group = row.get("nn_group", "")
        label = row.get("nn_label", "")
        if not group:
            continue
        completed.add(
            _key(
                group=group,
                label=label,
                model=row.get("model", ""),
                dataset=row.get("dataset", ""),
                seed=row.get("seed", ""),
            )
        )
    return completed


def select_missing(
    completed: set[tuple[str, str, str, str, str]],
    group: str | None,
    dataset: str | None,
    start: int,
    limit: int | None,
) -> list[tuple[int, Experiment]]:
    missing: list[tuple[int, Experiment]] = []
    for idx, exp in enumerate(build_experiments(), start=1):
        if idx < start:
            continue
        if group and exp.group != group:
            continue
        if dataset and exp.dataset != dataset:
            continue
        if _experiment_key(exp) in completed:
            continue
        missing.append((idx, exp))

    if limit is not None:
        missing = missing[:limit]
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing Neural Networks experiments on a local GPU")
    parser.add_argument("--completed-csv", type=Path, default=DEFAULT_IMPORTED_CSV)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--group", default=None, help="buffer/forget/memory/diagnostics/main")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. split_mnist")
    parser.add_argument("--start", type=int, default=1, help="Global experiment index lower bound")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    completed = load_completed_from_csv(args.completed_csv)
    completed |= load_completed_from_outputs(args.outputs_dir)
    missing = select_missing(
        completed=completed,
        group=args.group,
        dataset=args.dataset,
        start=args.start,
        limit=args.limit,
    )

    print(f"Completed keys loaded: {len(completed)}")
    print(f"Missing experiments selected: {len(missing)}")
    if args.completed_csv.exists():
        print(f"Imported CSV: {args.completed_csv}")
    else:
        print(f"Imported CSV not found: {args.completed_csv}")

    failed: list[int] = []
    for pos, (idx, exp) in enumerate(missing, start=1):
        cmd = build_cmd(exp)
        print("\n" + "=" * 100)
        print(
            f"[{pos}/{len(missing)}] global_idx={idx} "
            f"group={exp.group} label={exp.label} dataset={exp.dataset} seed={exp.seed}"
        )
        print("CMD:", " ".join(cmd))
        print("=" * 100)
        if args.dry_run:
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[FAIL] global_idx={idx} returncode={result.returncode} elapsed={elapsed/60:.1f}m")
            failed.append(idx)
            continue
        print(f"[OK] global_idx={idx} elapsed={elapsed/60:.1f}m")

    if failed:
        print("\nFailed global indices:", " ".join(str(i) for i in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
