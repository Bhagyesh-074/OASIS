"""Unit tests for oasis.gateway.litellm_hooks — pre/post call hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import (
    get_call_count,
    post_call_hook,
    pre_call_hook,
    reset_call_counter,
)


class TestPreCallHook:
    """pre_call_hook — RBE admission at the gateway."""

    def test_pre_call_hook_returns_verdict(self) -> None:
        """Hook returns a dict containing a verdict key."""
        req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        verdict = pre_call_hook(req)
        assert isinstance(verdict, dict)
        assert "verdict" in verdict
        assert verdict["verdict"] == "allow"
        assert verdict == "allow"
        assert verdict.verdict == "allow"

    def test_pre_call_hook_with_custom_admission_fn(self) -> None:
        """Admission check override is called if provided."""
        req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        verdict = pre_call_hook(req, admission_fn=lambda _: {"verdict": "downgrade"})
        assert verdict == "downgrade"
        assert verdict["verdict"] == "downgrade"

    def test_pre_call_hook_increments_call_counter(self) -> None:
        """Every pre_call_hook invocation increments the in-memory call counter."""
        reset_call_counter()
        count_before = get_call_count()
        req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        pre_call_hook(req)
        assert get_call_count() == count_before + 1


class TestPostCallHook:
    """post_call_hook — token/cost accounting."""

    def test_post_call_hook_updates_budget_state(self) -> None:
        """Budget state is updated with token/cost from the response."""
        class MockBudgetState:
            def __init__(self) -> None:
                self.records: list[dict[str, Any]] = []

            def record(
                self,
                *,
                tokens: int = 0,
                cost_usd: float = 0.0,
                wall_seconds: float = 0.0,
                calls: int = 0,
            ) -> None:
                self.records.append({
                    "tokens": tokens,
                    "cost_usd": cost_usd,
                    "wall_seconds": wall_seconds,
                    "calls": calls,
                })

        state = MockBudgetState()
        req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        resp = {
            "content": "ok",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            "cost_usd": 0.0002,
            "latency_ms": 50.0,
        }

        accounting = post_call_hook(req, resp, run_budget_state=state)
        assert accounting["tokens_in"] == 12
        assert accounting["tokens_out"] == 8
        assert accounting["cost_usd"] == 0.0002
        assert accounting["latency_ms"] == 50.0

        assert len(state.records) == 1
        rec = state.records[0]
        assert rec["tokens"] == 20
        assert rec["cost_usd"] == 0.0002
        assert rec["wall_seconds"] == 0.05
        assert rec["calls"] == 1

    def test_post_call_hook_writes_to_sink(self, tmp_path: Path) -> None:
        """post_call_hook writes a record to the provided JSONLSink."""
        sink_path = tmp_path / "sink.jsonl"
        sink = JSONLSink(sink_path)

        req = {
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "ping"}],
            "run_id": "r-1",
            "agent_id": "a-1",
            "purpose": "productive",
        }
        resp = {
            "content": "pong",
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
            "cost_usd": 0.00008,
            "latency_ms": 15.0,
        }

        accounting = post_call_hook(req, resp, sink=sink)
        assert accounting["tokens_in"] == 6
        assert accounting["tokens_out"] == 2

        lines = sink_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model"] == "gpt-4o-2024-08-06"
        assert record["tokens_in"] == 6
        assert record["tokens_out"] == 2
        assert record["cost_usd"] == 0.00008
        assert record["latency_ms"] == 15.0
        assert record["run_id"] == "r-1"
        assert record["agent_id"] == "a-1"
        assert record["purpose"] == "productive"

    def test_post_call_hook_does_not_increment_call_counter(self) -> None:
        """post_call_hook must never increment the call counter."""
        reset_call_counter()
        count = get_call_count()
        req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        resp = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "cost_usd": 0.00001}
        post_call_hook(req, resp)
        assert get_call_count() == count

