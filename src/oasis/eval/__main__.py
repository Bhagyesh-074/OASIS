"""Command-line interface for OASIS benchmark execution and reproduction."""

import argparse
import asyncio
import json
from pathlib import Path

from oasis.db.models import Database, TaskRecord
from oasis.eval.runner import BenchmarkRunner
from oasis.eval.stats import compute_metrics_summary


async def run_reproduction(job_id: str, mode: str, output_dir: str) -> None:
    """Execute reproduction run and output summary tables (FR-30)."""
    db = Database()
    runner = BenchmarkRunner(db=db, mode=mode)

    print("=== OASIS Benchmark Reproduction ===")
    print(f"Job ID: {job_id}")
    print(f"Mode:   {mode}")
    print("====================================")

    # 1. Fetch or create benchmark tasks
    tasks = db.list_tasks()
    if not tasks:
        tasks = [
            TaskRecord(
                task_id=f"task_{i:03d}",
                domain="research_qa",
                source="authored",
                statement=f"Evaluate comparative evidence on clinical topic {i}",
                complexity_label="medium",
                split="eval",
            )
            for i in range(1, 6)
        ]
        for t in tasks:
            db.insert_task(t)

    # 2. Clean up previous runs for this job if any, then execute matrix
    with db.get_connection() as conn:
        conn.execute("DELETE FROM run WHERE job_id = ?;", (job_id,))

    runs = await runner.run_matrix(tasks=tasks, job_id=job_id)
    print(f"Matrix execution completed: {len(runs)} run(s) recorded.")

    # 3. Gather run statistics
    db_runs = db.list_runs(job_id=job_id)
    raw_runs = [
        {
            "run_id": r.run_id,
            "arm": r.arm,
            "seed": r.seed,
            "quality_composite": r.quality_composite or 0.0,
            "total_cost_usd": r.total_cost_usd,
            "total_tokens": r.total_tokens,
            "wall_ms": r.wall_ms,
            "supervision_tokens": r.supervision_tokens,
            "compliant_all": r.compliant_all,
            "violation": not r.compliant_all,
        }
        for r in db_runs
    ]

    summary = compute_metrics_summary(raw_runs)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_file = out_path / f"{job_id}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"Statistical summary written to: {summary_file}")
    print(f"Ablation Arms Evaluated: {list(summary.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OASIS Evaluation & Benchmark Runner")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    reproduce_parser = subparsers.add_parser("reproduce", help="Reproduce benchmark ablation matrix")
    reproduce_parser.add_argument("--job-id", type=str, default="ablation-v3")
    reproduce_parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    reproduce_parser.add_argument("--output-dir", type=str, default="artifacts/results")

    args = parser.parse_args()
    if args.subcommand == "reproduce":
        asyncio.run(run_reproduction(args.job_id, args.mode, args.output_dir))


if __name__ == "__main__":
    main()
