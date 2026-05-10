"""Parallel GPU scheduler for Neural Networks experiment package.

This launches independent Hydra runs concurrently, assigning one experiment to
one GPU via CUDA_VISIBLE_DEVICES. It is designed for multi-GPU cloud machines.

Examples:
    uv run python run/run_parallel_nn_experiments.py --dry-run --group buffer --gpus 0,1
    uv run python run/run_parallel_nn_experiments.py --group forget --gpus 0,1,2,3
    uv run python run/run_parallel_nn_experiments.py --group memory --gpus 0,1,2,3 --start 21

Notes:
    - This is run-level parallelism, not distributed training.
    - Each child process sees exactly one GPU as cuda:0.
    - Child stdout/stderr are saved under outputs/parallel_logs/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from run_nn_experiments import Experiment, build_cmd, build_experiments


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "parallel_logs"


@dataclass
class RunningJob:
    idx: int
    exp: Experiment
    gpu: str
    process: subprocess.Popen
    log_path: Path
    start_time: float


def parse_gpus(raw: str) -> list[str]:
    gpus = [g.strip() for g in raw.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id, e.g. 0 or 0,1,2,3")
    return gpus


def filter_experiments(
    group: str | None,
    start: int,
    limit: int | None,
) -> list[tuple[int, Experiment]]:
    exps = build_experiments()
    indexed = list(enumerate(exps, start=1))
    if group:
        indexed = [(i, e) for i, e in indexed if e.group == group]
    indexed = [(i, e) for i, e in indexed if i >= start]
    if limit is not None:
        indexed = indexed[:limit]
    return indexed


def log_name(idx: int, exp: Experiment, gpu: str) -> str:
    label = exp.label.replace("/", "_").replace("=", "-")
    model = exp.model.replace("baselines/", "")
    return f"{idx:04d}__gpu{gpu}__{exp.group}__{label}__{model}__{exp.dataset}__s{exp.seed}.log"


def launch(idx: int, exp: Experiment, gpu: str, dry_run: bool) -> RunningJob | None:
    cmd = build_cmd(exp)
    print("\n" + "=" * 100)
    print(f"LAUNCH idx={idx} gpu={gpu} group={exp.group} label={exp.label} dataset={exp.dataset} seed={exp.seed}")
    print("CMD:", " ".join(cmd))
    print("=" * 100)

    if dry_run:
        return None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / log_name(idx, exp, gpu)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # Avoid CPU thread oversubscription when multiple processes run at once.
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    f = path.open("w", encoding="utf-8")
    f.write("CMD: " + " ".join(cmd) + "\n")
    f.write(f"CUDA_VISIBLE_DEVICES={gpu}\n\n")
    f.flush()
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    # Keep file handle alive via process object attribute so Windows does not close it early.
    process._codex_log_handle = f  # type: ignore[attr-defined]
    return RunningJob(idx, exp, gpu, process, path, time.time())


def close_job(job: RunningJob) -> None:
    handle = getattr(job.process, "_codex_log_handle", None)
    if handle is not None:
        handle.flush()
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neural Networks experiments across GPUs")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--group", default=None, help="Run one group: main/forget/memory/buffer/diagnostics")
    parser.add_argument("--start", type=int, default=1, help="Global 1-indexed experiment start")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll", type=float, default=10.0, help="Polling interval in seconds")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    queue = filter_experiments(args.group, args.start, args.limit)
    print(f"Queued {len(queue)} experiments; GPUs={gpus}; group={args.group or 'all'}")
    if args.dry_run:
        for k, (idx, exp) in enumerate(queue):
            launch(idx, exp, gpus[k % len(gpus)], dry_run=True)
        return

    running: list[RunningJob] = []
    failed: list[tuple[int, int, Path]] = []
    completed = 0

    next_job = 0
    while next_job < len(queue) or running:
        while next_job < len(queue) and len(running) < len(gpus):
            used = {job.gpu for job in running}
            free = next(g for g in gpus if g not in used)
            idx, exp = queue[next_job]
            job = launch(idx, exp, free, dry_run=False)
            if job is not None:
                running.append(job)
            next_job += 1

        time.sleep(args.poll)

        still_running: list[RunningJob] = []
        for job in running:
            rc = job.process.poll()
            if rc is None:
                still_running.append(job)
                continue
            elapsed = time.time() - job.start_time
            close_job(job)
            completed += 1
            if rc == 0:
                print(f"[OK] idx={job.idx} gpu={job.gpu} elapsed={elapsed/60:.1f}m log={job.log_path}")
            else:
                print(f"[FAIL] idx={job.idx} gpu={job.gpu} rc={rc} elapsed={elapsed/60:.1f}m log={job.log_path}")
                failed.append((job.idx, rc, job.log_path))
        running = still_running
        print(f"Progress: completed={completed}/{len(queue)} running={len(running)} queued={len(queue)-next_job}")

    print("\n" + "=" * 100)
    print(f"All done: completed={completed}/{len(queue)} failed={len(failed)}")
    if failed:
        print("Failures:")
        for idx, rc, path in failed:
            print(f"  idx={idx} rc={rc} log={path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
