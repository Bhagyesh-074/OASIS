"""Invariant test: Gateway call count == Budget event log count (FR-7).

FR-7: "Every outbound model call shall pass an admission check returning one
of allow, downgrade, truncate, halt. No call may bypass the check. Verified
by an integration test asserting call count at the gateway equals call count
in the budget event log."
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
    return JSONLSink(tmp_path / "test_reconciliation.jsonl")


class TestCallCountReconciliationInvariant:
    """Every call through the gateway increments the call counter and emits a budget event."""

    def test_reconciliation_across_allowed_calls(self, sink: JSONLSink) -> None:
        """Call count at gateway equals budget event count for multiple normal calls."""
        budget = Budget(
            max_tokens=100_000,
            max_cost_usd=10.0,
            max_wall_seconds=100.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        fake = FakeLLM()
        model = "gpt-4o"
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "1"}],
            response={"content": "resp 1", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0001},
        )
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "2"}],
            response={"content": "resp 2", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0001},
        )
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "3"}],
            response={"content": "resp 3", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0001},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=state)

        for i in range(1, 4):
            gw.call(model=model, messages=[{"role": "user", "content": str(i)}])

        # Invariant: gateway.call_count == len(admission_controller.events)
        assert gw.call_count == 3
        assert gw.admission_controller is not None
        assert len(gw.admission_controller.events) == 3
        assert gw.call_count == len(gw.admission_controller.events)

    def test_reconciliation_includes_halted_calls(self, sink: JSONLSink) -> None:
        """Halted calls that never execute still pass through admission and increment count."""
        budget = Budget(
            max_tokens=1000,
            max_cost_usd=0.01,
            max_wall_seconds=10.0,
            max_calls=2,
        )
        state = BudgetState(budget)
        fake = FakeLLM()
        model = "gpt-4o"
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "1"}],
            response={"content": "resp 1", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0001},
        )
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "2"}],
            response={"content": "resp 2", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0001},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=state, return_on_halt=True)

        # Call 1 (allow)
        gw.call(model=model, messages=[{"role": "user", "content": "1"}])
        # Call 2 (allow)
        gw.call(model=model, messages=[{"role": "user", "content": "2"}])
        # Call 3 (halt - budget max_calls was 2)
        gw.call(model=model, messages=[{"role": "user", "content": "3"}])

        assert gw.call_count == 3
        assert len(gw.admission_controller.events) == 3
        assert gw.call_count == len(gw.admission_controller.events)

        # First two were allowed, last was halted
        actions = [e["action"] for e in gw.admission_controller.events]
        assert actions == ["allow", "allow", "halt"]

    def test_reconciliation_zero_empty_justification_invariants(self, sink: JSONLSink) -> None:
        """Every single recorded event must have a non-empty justification string."""
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_wall_seconds=50.0,
            max_calls=5,
        )
        state = BudgetState(budget)
        fake = FakeLLM()
        model = "gpt-4o"
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            response={"content": "done"},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=state)
        gw.call(model=model, messages=[{"role": "user", "content": "test"}])

        events = gw.admission_controller.events
        assert len(events) == 1
        for event in events:
            assert isinstance(event["justification"], str)
            assert len(event["justification"].strip()) > 0
