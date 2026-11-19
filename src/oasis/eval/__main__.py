"""Command-line interface for OASIS benchmark execution and reproduction."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

from oasis.db.models import Database
from oasis.eval.runner import DEFAULT_MATRIX_ARMS, BenchmarkRunner
from oasis.eval.stats import compute_metrics_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
MODEL_ENV_DEFAULTS = {
    "OASIS_MODEL_DEFAULT": "gpt-4o-mini-2024-07-18",
    "OASIS_MODEL_CHEAP": "gpt-4o-mini-2024-07-18",
    "OASIS_MODEL_STRONG": "gpt-4o-2024-11-20",
    "OASIS_MODEL_JUDGE": "gpt-4o-mini-2024-07-18",
}
MANIFEST_DISTRIBUTIONS = (
    "oasis",
    "datasets",
    "fastapi",
    "litellm",
    "networkx",
    "numpy",
    "pandas",
    "pydantic",
    "python-dotenv",
    "python-ulid",
    "scipy",
    "sentence-transformers",
    "spacy",
    "uvicorn",
)


def build_result_manifest(
    *,
    job_id: str,
    mode: str,
    arms: Sequence[str],
    seeds: Sequence[int],
    task_ids: Sequence[str],
    run_ids: Sequence[str],
    planned_cells: int,
    observed_models: Sequence[str],
) -> dict[str, object]:
    """Capture the code, configuration, model, and dependency versions for a result."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(CONFIG_DIR.rglob("*.yaml"))
    }

    return {
        "job_id": job_id,
        "mode": mode,
        "arms": list(arms),
        "seeds": list(seeds),
        "task_ids": list(task_ids),
        "run_ids": list(run_ids),
        "planned_cells": planned_cells,
        "recorded_cells": len(run_ids),
        "git_commit": commit,
        "python_version": platform.python_version(),
        "library_versions": {
            distribution: importlib.metadata.version(distribution)
            for distribution in MANIFEST_DISTRIBUTIONS
        },
        "uv_lock_sha256": hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "model_identifiers": {
            "configured": {
                name: os.environ.get(name, default)
                for name, default in MODEL_ENV_DEFAULTS.items()
            },
            "observed": sorted(set(observed_models)),
        },
        "config_sha256": config_hashes,
    }


def require_clean_worktree() -> None:
    """Prevent associating results with a commit that does not contain the code run."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "Reproduction requires a clean git worktree so the result manifest "
            "identifies the exact code that ran."
        )


async def run_reproduction(
    job_id: str,
    mode: str,
    output_dir: str,
    arms: Sequence[str] = DEFAULT_MATRIX_ARMS,
    seeds: Sequence[int] = (1, 2, 3),
    max_tasks: int | None = None,
) -> None:
    """Execute a selected ablation matrix using only ingested evaluation tasks."""
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be a positive integer")
    db = Database()
    tasks = db.list_tasks(split="eval")
    if not tasks:
        raise RuntimeError(
            "No evaluation-split tasks found. Ingest the benchmark datasets before "
            "running a reproduction job."
        )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if db.list_runs(job_id=job_id):
        raise ValueError(
            f"Job ID {job_id!r} already has runs; choose a new job ID to preserve "
            "the existing results."
        )

    out_path = Path(output_dir)
    summary_file = out_path / f"{job_id}_summary.json"
    manifest_file = out_path / f"{job_id}_manifest.json"
    for path in (summary_file, manifest_file):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result file: {path}")

    require_clean_worktree()
    out_path.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(db=db, mode=mode)
    print("=== OASIS Benchmark Reproduction ===")
    print(f"Job ID: {job_id}")
    print(f"Mode:   {mode}")
    print(f"Tasks:  {len(tasks)} evaluation-split task(s)")
    print(f"Arms:   {', '.join(arms)}")
    print("====================================")

    runs = await runner.run_matrix(
        tasks=tasks,
        arms=arms,
        seeds=seeds,
        job_id=job_id,
    )
    print(f"Matrix execution completed: {len(runs)} run(s) recorded.")

    db_runs = db.list_runs(job_id=job_id)
    if not db_runs:
        raise RuntimeError(
            "The matrix produced no run records; no result files were written."
        )
    raw_runs = [
        {
            "run_id": run.run_id,
            "task_id": run.task_id,
            "arm": run.arm,
            "seed": run.seed,
            "quality_composite": run.quality_composite,
            "total_cost_usd": run.total_cost_usd,
            "total_tokens": run.total_tokens,
            "wall_ms": run.wall_ms,
            "supervision_tokens": run.supervision_tokens,
            "compliant_all": run.compliant_all,
        }
        for run in db_runs
    ]
    summary = compute_metrics_summary(raw_runs)
    planned_cells = len(tasks) * len(arms) * len(seeds)
    summary["execution"] = {
        "planned_cells": planned_cells,
        "recorded_cells": len(db_runs),
        "complete": len(db_runs) == planned_cells,
    }

    manifest = build_result_manifest(
        job_id=job_id,
        mode=mode,
        arms=arms,
        seeds=seeds,
        task_ids=[task.task_id for task in tasks],
        run_ids=[run.run_id for run in db_runs],
        planned_cells=planned_cells,
        observed_models=[
            agent.model
            for run in db_runs
            for agent in db.get_agent_instances(run.run_id)
        ],
    )
    summary_file.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest_file.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Statistical summary written to: {summary_file}")
    print(f"Result manifest written to: {manifest_file}")
    print(f"Ablation arms evaluated: {list(summary['arms'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OASIS Evaluation & Benchmark Runner")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Run the implemented ablation arms against ingested evaluation tasks",
    )
    reproduce_parser.add_argument("--job-id", type=str, default="ablation-v3")
    reproduce_parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    reproduce_parser.add_argument("--output-dir", type=str, default="artifacts/results")
    reproduce_parser.add_argument(
        "--arms",
        nargs="+",
        choices=(
            *DEFAULT_MATRIX_ARMS,
            "captainagent",
            "monitor_only",
            "full",
            "oracle",
        ),
        default=list(DEFAULT_MATRIX_ARMS),
        help="Ablation arms to execute (unsupported mechanisms fail explicitly)",
    )
    reproduce_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Random seeds for each task/arm cell",
    )
    reproduce_parser.add_argument(
        "--max-tasks",
        type=int,
        help="Limit to the first N evaluation tasks for a smoke run",
    )

    args = parser.parse_args()
    if args.subcommand == "reproduce":
        asyncio.run(
            run_reproduction(
                args.job_id,
                args.mode,
                args.output_dir,
                arms=args.arms,
                seeds=args.seeds,
                max_tasks=args.max_tasks,
            )
        )


if __name__ == "__main__":
    main()
