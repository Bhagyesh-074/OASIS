"""Integration tests for oracle sweep execution engine (FR-27, ADR-011, FR-3, NFR-3).

Tests:
- Calibration task refusal (FR-3)
- Resumability and idempotency (second run executes 0 runs)
- Concurrency bounded strictly by semaphore limit
- Spend check stopping mid-sweep without partial saves (NFR-3)
- Retry on transient error followed by success
- Permanent failure reported in report.failed and excluded from store
- Determinism across multiple runs
- Tranche callbacks and summarise() reporting
"""

from __future__ import annotations

import asyncio

import pytest

from oasis.eval.oracle_sweep import (
    InMemoryStore,
    OracleSweepConfig,
    RunResult,
    SweepReport,
    TaskRef,
    run_sweep,
    summarise,
)


@pytest.mark.asyncio
async def test_calibration_task_refused() -> None:
    """FR-3: Tasks with split='calibration' must be refused with ValueError naming the task."""
    tasks = [
        TaskRef(task_id="eval_task_01", split="eval"),
        TaskRef(task_id="calib_task_01", split="calibration"),
    ]
    store = InMemoryStore()
    config = OracleSweepConfig(k_values=[1], seeds=[1])

    async def fake_run(task_id: str, k: int, seed: int) -> RunResult:
        return RunResult(quality=0.9, cost_usd=0.01, tokens=100, wall_ms=50)

    with pytest.raises(ValueError, match="calib_task_01.*FR-3"):
        await run_sweep(
            tasks=tasks,
            run_fn=fake_run,
            store=store,
            config=config,
            spend_check=lambda: False,
        )


@pytest.mark.asyncio
async def test_resumability_zero_runs() -> None:
    """Second call to run_sweep with populated store performs zero runs."""
    tasks = [TaskRef(task_id="eval_01", split="eval")]
    store = InMemoryStore()
    config = OracleSweepConfig(k_values=[1, 2], seeds=[1, 2])

    call_count = 0

    async def fake_run(task_id: str, k: int, seed: int) -> RunResult:
        nonlocal call_count
        call_count += 1
        return RunResult(quality=0.85, cost_usd=0.005, tokens=150, wall_ms=80)

    # First pass: runs all 4 (2 k * 2 seeds)
    r1 = await run_sweep(
        tasks=tasks,
        run_fn=fake_run,
        store=store,
        config=config,
        spend_check=lambda: False,
    )
    assert r1.completed == 4
    assert r1.skipped_existing == 0
    assert call_count == 4

    # Second pass: all runs exist in store -> 0 attempted, 4 skipped
    r2 = await run_sweep(
        tasks=tasks,
        run_fn=fake_run,
        store=store,
        config=config,
        spend_check=lambda: False,
    )
    assert r2.completed == 0
    assert r2.attempted == 0
    assert r2.skipped_existing == 4
    assert call_count == 4  # No additional calls made


@pytest.mark.asyncio
async def test_concurrency_bounded() -> None:
    """Concurrency never exceeds config.concurrency limit."""
    tasks = [TaskRef(task_id=f"t_{i}", split="eval") for i in range(5)]
    store = InMemoryStore()
    # 5 tasks * 2 k * 2 seeds = 20 runs
    config = OracleSweepConfig(k_values=[1, 2], seeds=[1, 2], concurrency=3)

    active_count = 0
    max_active = 0

    async def fake_run(task_id: str, k: int, seed: int) -> RunResult:
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.01)
        active_count -= 1
        return RunResult(quality=0.8, cost_usd=0.001, tokens=50, wall_ms=10)

    report = await run_sweep(
        tasks=tasks,
        run_fn=fake_run,
        store=store,
        config=config,
        spend_check=lambda: False,
    )

    assert report.completed == 20
    assert max_active <= config.concurrency
    assert max_active > 1


@pytest.mark.asyncio
async def test_spend_check_stops_mid_sweep() -> None:
    """NFR-3: When spend_check() returns True, sweep stops cleanly with no partial writes."""
    tasks = [TaskRef(task_id=f"t_{i}", split="eval") for i in range(4)]
    store = InMemoryStore()
    config = OracleSweepConfig(
        k_values=[1], seeds=[1], tranche_size=2, concurrency=1
    )

    runs_completed = 0

    def spend_check() -> bool:
        # Trip the ceiling after 2 runs
        return runs_completed >= 2

    async def fake_run(task_id: str, k: int, seed: int) -> RunResult:
        nonlocal runs_completed
        runs_completed += 1
        return RunResult(quality=0.9, cost_usd=0.01, tokens=100, wall_ms=50)

    report = await run_sweep(
        tasks=tasks,
        run_fn=fake_run,
        store=store,
        config=config,
        spend_check=spend_check,
    )

    assert report.stopped_reason == "spend_ceiling_reached"
    assert report.completed == 2
    assert len(store.existing_keys()) == 2


@pytest.mark.asyncio
async def test_retry_then_success() -> None:
    """Transient failures are retried up to max_retries, recorded, and stored on success."""
    tasks = [TaskRef(task_id="t_retry", split="eval")]
    store = InMemoryStore()
    config = OracleSweepConfig(k_values=[1], seeds=[1], max_retries=2)

    attempts = 0

    async def flaky_run(task_id: str, k: int, seed: int) -> RunResult:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("Transient gateway network glitch")
        return RunResult(quality=0.88, cost_usd=0.005, tokens=200, wall_ms=100)

    report = await run_sweep(
        tasks=tasks,
        run_fn=flaky_run,
        store=store,
        config=config,
        spend_check=lambda: False,
    )

    assert report.completed == 1
    assert report.retried == 1
    assert len(report.failed) == 0
    assert ("t_retry", 1, 1) in store.existing_keys()


@pytest.mark.asyncio
async def test_permanent_failure_reported_and_not_stored() -> None:
    """Runs that fail past max_retries are appended to report.failed and never stored."""
    tasks = [
        TaskRef(task_id="t_fail", split="eval"),
        TaskRef(task_id="t_succeed", split="eval"),
    ]
    store = InMemoryStore()
    config = OracleSweepConfig(k_values=[1], seeds=[1], max_retries=1)

    async def run_fn(task_id: str, k: int, seed: int) -> RunResult:
        if task_id == "t_fail":
            raise ValueError("Deterministic test failure")
        return RunResult(quality=0.95, cost_usd=0.01, tokens=100, wall_ms=50)

    report = await run_sweep(
        tasks=tasks,
        run_fn=run_fn,
        store=store,
        config=config,
        spend_check=lambda: False,
    )

    assert report.completed == 1
    assert report.retried == 1
    assert len(report.failed) == 1
    assert report.failed[0].task_id == "t_fail"
    assert report.failed[0] == ("t_fail", 1, 1)

    # Failed run must NOT be in store
    assert ("t_fail", 1, 1) not in store.existing_keys()
    assert ("t_succeed", 1, 1) in store.existing_keys()


@pytest.mark.asyncio
async def test_determinism_across_runs() -> None:
    """Sweep execution produces identical results across two identical runs."""
    tasks = [TaskRef(task_id=f"t_{i}", split="eval") for i in [3, 1, 2]]
    config = OracleSweepConfig(k_values=[2, 1], seeds=[1, 2])

    execution_order_1: list[tuple[str, int, int]] = []
    execution_order_2: list[tuple[str, int, int]] = []

    async def run_record_1(task_id: str, k: int, seed: int) -> RunResult:
        execution_order_1.append((task_id, k, seed))
        return RunResult(quality=0.8, cost_usd=0.01, tokens=100, wall_ms=10)

    async def run_record_2(task_id: str, k: int, seed: int) -> RunResult:
        execution_order_2.append((task_id, k, seed))
        return RunResult(quality=0.8, cost_usd=0.01, tokens=100, wall_ms=10)

    rep1 = await run_sweep(
        tasks=tasks,
        run_fn=run_record_1,
        store=InMemoryStore(),
        config=config,
        spend_check=lambda: False,
    )
    rep2 = await run_sweep(
        tasks=tasks,
        run_fn=run_record_2,
        store=InMemoryStore(),
        config=config,
        spend_check=lambda: False,
    )

    assert rep1.completed == rep2.completed
    assert rep1.cumulative_cost_usd == pytest.approx(rep2.cumulative_cost_usd)
    # Strict deterministic ordering (sorted task_id, sorted k, sorted seeds)
    assert execution_order_1 == execution_order_2


@pytest.mark.asyncio
async def test_on_tranche_callback_and_summarise() -> None:
    """Tranche callback receives report with cumulative cost and summarise groups results."""
    tasks = [
        TaskRef(task_id="t_complete", split="eval"),
        TaskRef(task_id="t_incomplete", split="eval"),
    ]
    store = InMemoryStore()
    config = OracleSweepConfig(k_values=[1, 2], seeds=[1], tranche_size=1)

    tranche_reports: list[SweepReport] = []

    def on_tranche(rep: SweepReport) -> None:
        tranche_reports.append(
            SweepReport(
                attempted=rep.attempted,
                completed=rep.completed,
                cumulative_cost_usd=rep.cumulative_cost_usd,
            )
        )

    # Run only for t_complete
    async def fake_run(task_id: str, k: int, seed: int) -> RunResult:
        return RunResult(quality=0.85, cost_usd=0.01, tokens=100, wall_ms=50)

    await run_sweep(
        tasks=[tasks[0]],
        run_fn=fake_run,
        store=store,
        config=config,
        spend_check=lambda: False,
        on_tranche=on_tranche,
    )

    # 2 runs (k=1, k=2), tranche_size=1 -> 2 tranche calls
    assert len(tranche_reports) == 2
    assert tranche_reports[-1].completed == 2
    assert tranche_reports[-1].cumulative_cost_usd == pytest.approx(0.02)

    # summarise() across both tasks
    summary = summarise(store=store, tasks=tasks, config=config)
    assert "t_complete" in summary
    assert "t_incomplete" not in summary
    assert summary.incomplete == ["t_incomplete"]
    assert summary["t_complete"].oracle_k == 1
