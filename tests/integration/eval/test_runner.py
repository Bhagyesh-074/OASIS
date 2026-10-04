"""Integration tests for BenchmarkRunner matrix execution and ablation integrity (FR-26, TESTING.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis.db.migrate import apply_migrations
from oasis.db.models import Database, TaskRecord
from oasis.eval.runner import BenchmarkRunner
from oasis.rbe.budget import Budget


@pytest.fixture
def runner_env(tmp_path: Path) -> tuple[Database, BenchmarkRunner]:
    """Fixture providing an isolated database and BenchmarkRunner."""
    db_file = tmp_path / "runner_test.db"
    apply_migrations(db_file)
    db = Database(db_file)
    runner = BenchmarkRunner(
        db=db,
        concurrency=4,
        mode="live",
        cache_dir=tmp_path / "cache",
    )
    return db, runner


@pytest.mark.asyncio
async def test_smoke_matrix_execution(runner_env: tuple[Database, BenchmarkRunner]) -> None:
    """FR-26: Execute matrix across tasks and arms with bounded concurrency."""
    db, runner = runner_env

    # Seed 2 tasks
    tasks = [
        TaskRecord(task_id="task_1", domain="code_generation", source="mbpp", statement="Write fibonacci function", split="eval"),
        TaskRecord(task_id="task_2", domain="research_qa", source="hotpotqa", statement="Summarize multi-agent consensus", split="eval"),
    ]
    for t in tasks:
        db.insert_task(t)

    arms = ["vanilla_fixed", "full"]
    seeds = [1]

    results = await runner.run_matrix(tasks=tasks, arms=arms, seeds=seeds, job_id="smoke_job", concurrency=2)
    assert len(results) == 4  # 2 tasks * 2 arms * 1 seed

    for r in results:
        assert r.status in ("completed", "halted_budget")
        assert r.total_tokens > 0
        assert r.wall_ms is not None

    # Check ablation integrity: vanilla_fixed must have 0 replacement events
    vanilla_runs = [r for r in results if r.arm == "vanilla_fixed"]
    for vr in vanilla_runs:
        replacements = db.get_replacement_events(vr.run_id)
        assert len(replacements) == 0

    # Full arm has replacement events recorded
    full_runs = [r for r in results if r.arm == "full"]
    for fr in full_runs:
        replacements = db.get_replacement_events(fr.run_id)
        assert len(replacements) >= 1
        assert replacements[0].justification != ""


@pytest.mark.asyncio
async def test_tiny_budget_halt_behavior(runner_env: tuple[Database, BenchmarkRunner]) -> None:
    """FR-8: Deliberately tiny budget terminates with status halted_budget without raising."""
    db, runner = runner_env

    task = TaskRecord(task_id="task_tiny", domain="quant_analysis", source="gsm8k", statement="Calculate prime factors", split="eval")
    db.insert_task(task)

    tiny_budget = Budget(max_tokens=10, max_cost_usd=0.00001, max_wall_seconds=1.0, max_calls=1)

    result = await runner.run_task(task, arm="rbe_only", seed=1, budget=tiny_budget)
    assert result.status == "halted_budget"
    assert result.status_detail is not None
    assert "halt" in result.status_detail.lower() or "budget" in result.status_detail.lower()
