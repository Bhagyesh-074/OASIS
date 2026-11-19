"""Invariant test: Replay mode issues exactly zero network calls (FR-30, TESTING.md).

FR-30: 'A replay mode shall execute the full pipeline from cache with zero
network calls. Verified by CI test asserting network calls equal zero.'
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from oasis.db.migrate import apply_migrations
from oasis.db.models import Database, TaskRecord
from oasis.eval.runner import BenchmarkRunner
from oasis.gateway._hash import hash_request
from oasis.gateway.jsonl_sink import JSONLSink


@pytest.fixture
def replay_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Database, BenchmarkRunner]:
    """Fixture with socket.socket patched to forbid any outbound network connection."""
    # Guard against network calls
    def forbidden_connect(*args, **kwargs):
        raise RuntimeError("FR-30 Invariant Violation: Network call detected in replay mode!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)

    db_file = tmp_path / "replay_test.db"
    apply_migrations(db_file)
    db = Database(db_file)

    runner = BenchmarkRunner(
        db=db,
        concurrency=2,
        mode="replay",
        cache_dir=tmp_path / "replay_cache",
    )
    return db, runner


@pytest.mark.asyncio
async def test_replay_mode_zero_network_calls_invariant(
    replay_env: tuple[Database, BenchmarkRunner],
    tmp_path: Path,
) -> None:
    """Assert full matrix execution in replay mode makes 0 network calls (FR-30)."""
    db, runner = replay_env

    task = TaskRecord(
        task_id="t_replay",
        domain="code_generation",
        source="mbpp",
        statement="Write a fast sort function",
        split="eval",
    )
    db.insert_task(task)

    model = "gpt-4o-mini-2024-07-18"
    messages = [
        {
            "role": "system",
            "content": "You are the generalist agent. Work on the task and use prior agent output where helpful.",
        },
        {"role": "user", "content": "Task:\nWrite a fast sort function"},
    ]
    JSONLSink(tmp_path / "replay_cache" / "replay_job.jsonl").write(
        prompt_sha=hash_request(model, messages, 0.2, None),
        model=model,
        messages=messages,
        temperature=0.2,
        tool_schema=None,
        response={
            "content": "A fast sort implementation.",
            "model": model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.00015,
            "latency_ms": 4.0,
        },
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.00015,
        latency_ms=4.0,
    )

    # Must serve the recorded response without tripping the network guard.
    results = await runner.run_matrix(
        tasks=[task],
        arms=["single_agent"],
        seeds=[1],
        job_id="replay_job",
        concurrency=1,
    )

    assert len(results) == 1
    for r in results:
        assert r.mode == "replay"
        assert r.status == "completed"
        assert r.total_calls == 1
