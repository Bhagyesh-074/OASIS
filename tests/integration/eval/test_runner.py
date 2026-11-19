"""Integration tests for BenchmarkRunner matrix execution and ablation integrity (FR-26, TESTING.md)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from oasis.db.migrate import apply_migrations
from oasis.db.models import Database, TaskRecord
from oasis.eval.runner import (
    BenchmarkRunner,
    ReplayCacheMissError,
    _LiteLLMBackend,
    _ReplayLLMBackend,
)
from oasis.gateway._hash import hash_request
from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
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
        fake_llm=FakeLLM(),
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

    arms = ["vanilla_fixed", "rbe_only"]
    seeds = [1]

    results = await runner.run_matrix(tasks=tasks, arms=arms, seeds=seeds, job_id="smoke_job", concurrency=2)
    assert len(results) == 4  # 2 tasks * 2 arms * 1 seed

    for r in results:
        assert r.status in ("completed", "halted_budget")
        assert r.total_tokens > 0
        assert r.wall_ms is not None

    for run in results:
        assert db.get_replacement_events(run.run_id) == []
        assert run.quality_composite is None


@pytest.mark.asyncio
async def test_unimplemented_arms_are_rejected_before_run_creation(
    runner_env: tuple[Database, BenchmarkRunner],
) -> None:
    """Do not include hard-coded monitor/replacement behavior in experiment results."""
    db, runner = runner_env
    task = TaskRecord(
        task_id="task_full",
        domain="code_generation",
        source="mbpp",
        statement="Write a sorting function",
        split="eval",
    )
    db.insert_task(task)

    with pytest.raises(NotImplementedError, match="RTPM and replacement pipelines"):
        await runner.run_task(task, arm="full", seed=1)
    with pytest.raises(NotImplementedError, match="RTPM and replacement pipelines"):
        await runner.run_matrix([task], arms=["vanilla_fixed", "full"])

    assert db.list_runs() == []
    assert runner.fake_llm is not None
    assert runner.fake_llm.call_count == 0


@pytest.mark.asyncio
async def test_unknown_and_oracle_arms_are_rejected(
    runner_env: tuple[Database, BenchmarkRunner],
) -> None:
    db, runner = runner_env
    task = TaskRecord(
        task_id="task_unknown_arm",
        domain="code_generation",
        source="mbpp",
        statement="Write a sorting function",
        split="eval",
    )
    db.insert_task(task)

    with pytest.raises(ValueError, match="Unsupported benchmark arm"):
        await runner.run_task(task, arm="unknown_arm")
    with pytest.raises(NotImplementedError, match="oracle team-size sweep"):
        await runner.run_task(task, arm="oracle")

    assert db.list_runs() == []


@pytest.mark.asyncio
async def test_matrix_calls_are_concurrent_but_bounded(
    runner_env: tuple[Database, BenchmarkRunner],
) -> None:
    """The matrix semaphore allows parallel provider waits without exceeding its limit."""
    db, runner = runner_env

    class SlowFakeLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self.active_calls = 0
            self.max_active_calls = 0

        def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float = 0.0,
            tool_schema: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            with self._lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                time.sleep(0.05)
                return super().complete(model, messages, temperature, tool_schema)
            finally:
                with self._lock:
                    self.active_calls -= 1

    fake_llm = SlowFakeLLM()
    runner.fake_llm = fake_llm
    tasks = [
        TaskRecord(
            task_id=f"parallel_{idx}",
            domain="code_generation",
            source="mbpp",
            statement=f"Write function {idx}",
            split="eval",
        )
        for idx in range(2)
    ]
    for task in tasks:
        db.insert_task(task)

    results = await runner.run_matrix(
        tasks=tasks,
        arms=["single_agent"],
        seeds=[1],
        job_id="parallel_job",
        concurrency=2,
    )

    assert all(result.status == "completed" for result in results)
    assert fake_llm.max_active_calls == 2


def test_litellm_backend_normalizes_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live provider responses are adapted to the gateway accounting shape."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        model="gpt-4o-mini-2024-07-18",
    )
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: response)
    monkeypatch.setattr(
        litellm,
        "completion_cost",
        lambda *, completion_response: 0.001,
    )

    result = _LiteLLMBackend().complete(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result["content"] == "answer"
    assert result["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    assert result["cost_usd"] == 0.001


def test_replay_backend_uses_prompt_hash_and_reports_cache_miss(tmp_path: Path) -> None:
    """Replay returns recorded responses and makes a cache miss explicit."""
    model = "gpt-4o-mini-2024-07-18"
    messages = [{"role": "user", "content": "hello"}]
    response = {
        "content": "cached answer",
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        "cost_usd": 0.00004,
        "latency_ms": 3.0,
    }
    sink = JSONLSink(tmp_path / "cache.jsonl")
    sink.write(
        prompt_sha=hash_request(model, messages, 0.0, None),
        model=model,
        messages=messages,
        temperature=0.0,
        tool_schema=None,
        response=response,
        tokens_in=2,
        tokens_out=2,
        cost_usd=0.00004,
        latency_ms=3.0,
    )
    backend = _ReplayLLMBackend(sink)

    assert backend.complete(model, messages) == response
    with pytest.raises(ReplayCacheMissError, match="No cached response"):
        backend.complete(model, [{"role": "user", "content": "uncached"}])


@pytest.mark.asyncio
async def test_replay_run_reads_cache_without_overwriting_it(tmp_path: Path) -> None:
    """A replay run reads the live cache and writes its audit log separately."""
    db_file = tmp_path / "replay_test.db"
    apply_migrations(db_file)
    db = Database(db_file)
    cache_dir = tmp_path / "cache"
    task = TaskRecord(
        task_id="task_replay",
        domain="code_generation",
        source="mbpp",
        statement="Write fibonacci function",
        split="eval",
    )
    db.insert_task(task)

    model = "gpt-4o-mini-2024-07-18"
    messages = [
        {
            "role": "system",
            "content": "You are the generalist agent. Work on the task and use prior agent output where helpful.",
        },
        {"role": "user", "content": "Task:\nWrite fibonacci function"},
    ]
    response = {
        "content": "cached fibonacci implementation",
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "cost_usd": 0.00015,
        "latency_ms": 4.0,
    }
    source_sink = JSONLSink(cache_dir / "replay-job.jsonl")
    source_sink.write(
        prompt_sha=hash_request(model, messages, 0.2, None),
        model=model,
        messages=messages,
        temperature=0.2,
        tool_schema=None,
        response=response,
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.00015,
        latency_ms=4.0,
    )

    runner = BenchmarkRunner(db=db, mode="replay", cache_dir=cache_dir)
    result = await runner.run_task(
        task,
        arm="single_agent",
        seed=1,
        job_id="replay-job",
    )

    assert result.status == "completed"
    assert result.total_calls == 1
    assert source_sink.all_hashes() == [hash_request(model, messages, 0.2, None)]
    stored = db.get_agent_outputs(result.run_id)
    assert len(stored) == 1
    assert stored[0].content == response["content"]


@pytest.mark.asyncio
async def test_replay_cache_miss_is_recorded_as_failed_run(tmp_path: Path) -> None:
    """A missing replay entry is surfaced in persisted run status, not left running."""
    db_file = tmp_path / "replay_miss_test.db"
    apply_migrations(db_file)
    db = Database(db_file)
    task = TaskRecord(
        task_id="task_replay_miss",
        domain="code_generation",
        source="mbpp",
        statement="Write a binary search function",
        split="eval",
    )
    db.insert_task(task)
    runner = BenchmarkRunner(
        db=db,
        mode="replay",
        cache_dir=tmp_path / "empty_cache",
    )

    result = await runner.run_task(
        task,
        arm="single_agent",
        seed=1,
        job_id="missing-job",
    )

    assert result.status == "failed"
    assert result.status_detail is not None
    assert result.status_detail.startswith("cache_miss_in_replay:")


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
