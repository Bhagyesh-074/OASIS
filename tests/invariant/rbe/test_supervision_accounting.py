"""Invariant test for FR-9: supervision calls are tagged purpose=supervision

and counted against the same budget as productive calls.
Run totals must equal productive-tagged costs + supervision-tagged costs.
Supervision calls trigger halt just like productive calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway, GatewayHaltError
from oasis.rbe.admission import AdmissionController, AdmissionVerdict
from oasis.rbe.budget import Budget, BudgetState


@pytest.fixture
def sink(tmp_path: Path) -> JSONLSink:
    return JSONLSink(tmp_path / "supervision_test.jsonl")


class TestSupervisionAccounting:
    """FR-9 / ADR-006: Supervision calls counted against unified budget."""

    def test_run_totals_equal_productive_plus_supervision(self, sink: JSONLSink) -> None:
        """Run total tokens and cost equal sum of productive + supervision parts."""
        budget = Budget(
            max_tokens=100_000,
            max_cost_usd=10.0,
            max_wall_seconds=100.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        model = "gpt-4o-2024-08-06"

        fake = FakeLLM()
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "task 1"}],
            response={"content": "result 1", "usage": {"prompt_tokens": 50, "completion_tokens": 50}, "cost_usd": 0.001},
        )
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "judge 1"}],
            response={"content": "judged score", "usage": {"prompt_tokens": 30, "completion_tokens": 20}, "cost_usd": 0.0005},
        )
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "task 2"}],
            response={"content": "result 2", "usage": {"prompt_tokens": 40, "completion_tokens": 60}, "cost_usd": 0.001},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=state)

        # Call 1: productive
        gw.call(model=model, messages=[{"role": "user", "content": "task 1"}], purpose="productive")
        # Call 2: supervision
        gw.call(model=model, messages=[{"role": "user", "content": "judge 1"}], purpose="supervision")
        # Call 3: productive
        gw.call(model=model, messages=[{"role": "user", "content": "task 2"}], purpose="productive")

        # Invariant 1: Total tokens == productive_tokens + supervision_tokens
        assert state.tokens == state.productive_tokens + state.supervision_tokens
        assert state.productive_tokens == (100 + 100)
        assert state.supervision_tokens == 50
        assert state.tokens == 250

        # Invariant 2: Total cost == productive_cost_usd + supervision_cost_usd
        assert pytest.approx(state.cost_usd, rel=1e-6) == state.productive_cost_usd + state.supervision_cost_usd
        assert pytest.approx(state.productive_cost_usd, rel=1e-6) == 0.002
        assert pytest.approx(state.supervision_cost_usd, rel=1e-6) == 0.0005
        assert pytest.approx(state.cost_usd, rel=1e-6) == 0.0025

        # Invariant 3: Accounting log contains the exact matching tags
        purposes = [record["purpose"] for record in gw.accounting_log]
        assert purposes == ["productive", "supervision", "productive"]

    def test_supervision_call_triggers_halt_on_budget_exhaustion(self, sink: JSONLSink) -> None:
        """Supervision calls are never exempted and trigger halt if budget exhausted."""
        budget = Budget(
            max_tokens=1000,
            max_cost_usd=1.0,
            max_wall_seconds=10.0,
            max_calls=1,
        )
        state = BudgetState(budget)
        model = "gpt-4o-2024-08-06"

        fake = FakeLLM()
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "work"}],
            response={"content": "work done", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0002},
        )

        gw = Gateway(sink=sink, llm=fake, budget_state=state)

        # Call 1 consumes the only call allowed in budget
        gw.call(model=model, messages=[{"role": "user", "content": "work"}], purpose="productive")
        assert state.calls == 1

        # Call 2 is a supervision call — MUST NOT be exempted; must trigger halt!
        with pytest.raises(GatewayHaltError, match="halt"):
            gw.call(
                model=model,
                messages=[{"role": "user", "content": "judge work"}],
                purpose="supervision",
            )

    def test_admission_controller_treats_supervision_identically(self) -> None:
        """Direct check on AdmissionController asserting zero special-casing for supervision."""
        budget = Budget(
            max_tokens=100,
            max_cost_usd=0.01,
            max_wall_seconds=10.0,
            max_calls=1,
        )
        state = BudgetState(budget)
        state.record(tokens=100, cost_usd=0.01, calls=1)

        controller = AdmissionController(state)
        # Check productive
        verdict_prod = controller.check({
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "prod"}],
            "purpose": "productive",
        })
        assert verdict_prod == AdmissionVerdict.HALT

        # Check supervision: must be HALT, not ALLOW
        verdict_sup = controller.check({
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "judge"}],
            "purpose": "supervision",
        })
        assert verdict_sup == AdmissionVerdict.HALT

    def test_invalid_purpose_rejected(self, sink: JSONLSink) -> None:
        """Only 'productive' and 'supervision' are accepted (matching DATABASE.md)."""
        gw = Gateway(sink=sink, llm=FakeLLM())
        with pytest.raises(ValueError, match="Invalid purpose"):
            gw.call(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": "test"}],
                purpose="internal_eval",
            )
