"""Integration tests for FR-8: on halt, run terminates with status halted_budget

and returns the best output produced so far rather than raising an error.
Tested with FakeLLM and a deliberately tiny budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway
from oasis.rbe.budget import Budget, BudgetState


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_halt_calls.jsonl"


@pytest.fixture
def sink(log_path: Path) -> JSONLSink:
    return JSONLSink(log_path)


class TestHaltBehaviorWithTinyBudget:
    """FR-8 integration test: tiny budget leads to halted_budget and best output returned."""

    def test_halt_returns_best_output_without_raising(
        self, sink: JSONLSink
    ) -> None:
        """A tiny budget allows the first call and halts the second, returning best output so far."""
        # Budget allows exactly 1 call
        budget = Budget(
            max_tokens=1000,
            max_cost_usd=0.01,
            max_wall_seconds=10.0,
            max_calls=1,
        )
        budget_state = BudgetState(budget)

        model = "gpt-4o-2024-08-06"
        msgs_1 = [{"role": "user", "content": "First step"}]
        msgs_2 = [{"role": "user", "content": "Second step"}]

        resp_1 = {
            "content": "Step 1 complete with high quality analysis.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "cost_usd": 0.0002,
            "latency_ms": 50.0,
        }

        fake = FakeLLM()
        fake.script(model=model, messages=msgs_1, response=resp_1)

        gw = Gateway(
            sink=sink,
            llm=fake,
            budget_state=budget_state,
            return_on_halt=True,
            run_id="run-halt-test-1",
        )

        outputs_so_far: list[dict[str, str]] = []

        # Call 1: should be admitted and succeed
        res_1 = gw.call(model=model, messages=msgs_1, outputs_so_far=outputs_so_far)
        assert res_1["_verdict"] == "allow"
        assert res_1["content"] == "Step 1 complete with high quality analysis."
        outputs_so_far.append({"step": "1", "content": res_1["content"], "score": 0.92})

        # At this point, 1 call consumed, max_calls was 1. Budget is exhausted for calls.
        assert budget_state.calls == 1
        assert budget_state.is_exhausted()

        # Call 2: should be halted, NOT raise an error, and return best output so far
        res_2 = gw.call(model=model, messages=msgs_2, outputs_so_far=outputs_so_far)

        assert isinstance(res_2, dict)
        assert res_2["status"] == "halted_budget"
        assert res_2["best_output"] == {"step": "1", "content": res_1["content"], "score": 0.92}
        assert res_2["outputs_count"] == 1
        assert "exhausted" in res_2["halt_reason"] or "calls" in res_2["halt_reason"]

    def test_halt_on_zero_budget_immediate_return(
        self, sink: JSONLSink
    ) -> None:
        """A budget with 0 tokens halts immediately without executing any calls."""
        budget = Budget(
            max_tokens=0,
            max_cost_usd=1.0,
            max_wall_seconds=10.0,
            max_calls=10,
        )
        budget_state = BudgetState(budget)

        gw = Gateway(
            sink=sink,
            llm=FakeLLM(),
            budget_state=budget_state,
            return_on_halt=True,
            run_id="run-zero-budget",
        )

        prior_outputs = [{"id": "fallback-output", "score": 0.5}]
        res = gw.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            outputs_so_far=prior_outputs,
        )

        assert res["status"] == "halted_budget"
        assert res["best_output"] == {"id": "fallback-output", "score": 0.5}
        assert res["outputs_count"] == 1
        assert gw.call_count == 1  # Admission check was executed and counted

    def test_halt_leaves_sink_unpolluted_by_unexecuted_call(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """The halted call never executed, so sink only contains records for allowed calls."""
        budget = Budget(
            max_tokens=1000,
            max_cost_usd=0.01,
            max_wall_seconds=10.0,
            max_calls=1,
        )
        budget_state = BudgetState(budget)
        model = "gpt-4o"
        fake = FakeLLM()
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "first"}],
            response={"content": "ok", "usage": {"prompt_tokens": 5, "completion_tokens": 5}, "cost_usd": 0.0001},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=budget_state, return_on_halt=True)

        gw.call(model=model, messages=[{"role": "user", "content": "first"}])
        gw.call(model=model, messages=[{"role": "user", "content": "second"}])

        lines = [line for line in log_path.read_text(encoding="utf-8").strip().splitlines() if line]
        # Exactly 1 line in sink because call 2 was halted before execution
        assert len(lines) == 1
