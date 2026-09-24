"""Invariant tests for oasis.rbe — guards claims made in the paper.

These are the highest-value tests in the project (TESTING.md).  Each
guards a claim that, if violated, silently corrupts results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway
from oasis.rbe.budget import Budget, BudgetState


@pytest.fixture
def sink(tmp_path: Path) -> JSONLSink:
    return JSONLSink(tmp_path / "invariants.jsonl")


class TestCallCountReconciliation:
    """Gateway calls == budget event log count (FR-7)."""

    def test_no_call_bypasses_enforcement(self, sink: JSONLSink) -> None:
        """Every call through the gateway appears in the budget event log."""
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_wall_seconds=60.0,
            max_calls=5,
        )
        budget_state = BudgetState(budget)
        fake = FakeLLM()
        fake.script(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
            response={"content": "pong", "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=budget_state)

        for _ in range(3):
            gw.call(model="gpt-4o", messages=[{"role": "user", "content": "ping"}])

        assert gw.call_count == 3
        assert len(gw.admission_controller.events) == 3
        assert gw.call_count == len(gw.admission_controller.events)


class TestAccountingReconciliation:
    """Run totals == Σ per-output costs; supervision included (FR-9, ADR-006)."""

    def test_totals_equal_sum_of_per_call_costs(self, sink: JSONLSink) -> None:
        """No cost leakage: run total matches sum of individual records."""
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_wall_seconds=60.0,
            max_calls=10,
        )
        budget_state = BudgetState(budget)
        fake = FakeLLM()
        fake.script(
            model="gpt-4o",
            messages=[{"role": "user", "content": "a"}],
            response={"content": "res a", "usage": {"prompt_tokens": 10, "completion_tokens": 15}, "cost_usd": 0.00025, "latency_ms": 20.0},
        )
        fake.script(
            model="gpt-4o",
            messages=[{"role": "user", "content": "b"}],
            response={"content": "res b", "usage": {"prompt_tokens": 20, "completion_tokens": 30}, "cost_usd": 0.00050, "latency_ms": 40.0},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=budget_state)
        gw.call(model="gpt-4o", messages=[{"role": "user", "content": "a"}])
        gw.call(model="gpt-4o", messages=[{"role": "user", "content": "b"}])

        expected_tokens = sum(
            r["tokens_in"] + r["tokens_out"] for r in gw.accounting_log
        )
        expected_cost = sum(r["cost_usd"] for r in gw.accounting_log)

        assert budget_state.tokens == expected_tokens
        assert pytest.approx(budget_state.cost_usd, rel=1e-6) == expected_cost
        assert budget_state.calls == 2

    def test_supervision_tokens_included_in_budget(self, sink: JSONLSink) -> None:
        """ADR-006: purpose=supervision tokens counted, not excluded."""
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_wall_seconds=60.0,
            max_calls=10,
        )
        budget_state = BudgetState(budget)
        fake = FakeLLM()
        fake.script(
            model="gpt-4o",
            messages=[{"role": "user", "content": "supervise"}],
            response={"content": "evaluation", "usage": {"prompt_tokens": 50, "completion_tokens": 25}, "cost_usd": 0.00075},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=budget_state)
        gw.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "supervise"}],
            purpose="supervision",
        )

        assert budget_state.tokens == 75
        assert pytest.approx(budget_state.cost_usd, rel=1e-6) == 0.00075
        assert gw.accounting_log[0]["purpose"] == "supervision"
