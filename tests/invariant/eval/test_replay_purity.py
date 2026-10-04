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
from oasis.gateway.fake_llm import FakeLLM


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

    fake_llm = FakeLLM()
    runner = BenchmarkRunner(
        db=db,
        concurrency=2,
        mode="replay",
        cache_dir=tmp_path / "replay_cache",
        fake_llm=fake_llm,
    )
    return db, runner


@pytest.mark.asyncio
async def test_replay_mode_zero_network_calls_invariant(
    replay_env: tuple[Database, BenchmarkRunner],
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

    # Must complete cleanly without tripping the monkeypatched socket.connect error
    results = await runner.run_matrix(
        tasks=[task],
        arms=["vanilla_fixed", "full"],
        seeds=[1],
        job_id="replay_job",
        concurrency=1,
    )

    assert len(results) == 2
    for r in results:
        assert r.mode == "replay"
        assert r.status == "completed"
