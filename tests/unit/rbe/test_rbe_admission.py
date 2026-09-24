"""Unit tests for oasis.rbe.admission — per-call admission control (FR-7/FR-8)."""

from __future__ import annotations

from oasis.rbe.admission import AdmissionController, AdmissionVerdict
from oasis.rbe.budget import Budget, BudgetState


class TestAdmissionVerdict:
    """AdmissionVerdict enum coverage."""

    def test_exactly_four_verdicts(self) -> None:
        """FR-7 / ADR-004: exactly allow, downgrade, truncate, halt."""
        assert len(AdmissionVerdict) == 4
        expected = {"allow", "downgrade", "truncate", "halt"}
        assert {v.value for v in AdmissionVerdict} == expected


class TestAdmissionController:
    """AdmissionController.check — verdict boundaries."""

    def test_allow_when_ample_budget(self) -> None:
        """Plenty of budget → allow."""
        budget = Budget(
            max_tokens=100_000,
            max_cost_usd=10.0,
            max_wall_seconds=300.0,
            max_calls=100,
        )
        state = BudgetState(budget)
        controller = AdmissionController(state, run_id="run-1")

        call_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, world!"}],
        }
        verdict = controller.check(call_data)
        assert verdict == AdmissionVerdict.ALLOW
        assert verdict == "allow"
        assert len(controller.events) == 1
        assert controller.events[0]["action"] == "allow"

    def test_halt_when_budget_exhausted(self) -> None:
        """FR-8: exhausted budget → halt (not an error)."""
        budget = Budget(
            max_tokens=100,
            max_cost_usd=0.01,
            max_wall_seconds=10.0,
            max_calls=1,
        )
        state = BudgetState(budget)
        state.record(tokens=100, cost_usd=0.01, wall_seconds=10.0, calls=1)
        controller = AdmissionController(state, run_id="run-1")

        call_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Another call"}],
        }
        verdict = controller.check(call_data)
        assert verdict == AdmissionVerdict.HALT
        assert verdict == "halt"
        assert len(controller.events) == 1
        assert controller.events[0]["action"] == "halt"
        assert "exhausted" in controller.events[0]["justification"]

    def test_downgrade_near_limit(self) -> None:
        """Close to limit on cost (> 70%) → downgrade to cheaper model."""
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=0.01,
            max_wall_seconds=100.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        # Remaining cost = 0.01. Projected cost = 0.008 (> 70% of 0.01)
        controller = AdmissionController(state, run_id="run-1")
        call_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Task"}],
            "projected_cost_usd": 0.008,
            "projected_tokens": 100,
        }
        verdict = controller.check(call_data)
        assert verdict == AdmissionVerdict.DOWNGRADE
        assert verdict == "downgrade"
        assert len(controller.events) == 1
        assert controller.events[0]["action"] == "downgrade"
        assert controller.events[0]["model_after"] == "gpt-4o-mini"

    def test_truncate_near_token_limit(self) -> None:
        """Close to limit on tokens (> 80%) → truncate context."""
        budget = Budget(
            max_tokens=1000,
            max_cost_usd=1.0,
            max_wall_seconds=100.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        # Remaining tokens = 1000. Projected tokens = 850 (> 80% of 1000)
        # Projected cost = 0.001 (<< 70% of 1.0)
        controller = AdmissionController(state, run_id="run-1")
        call_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Long context"}],
            "projected_tokens": 850,
            "projected_cost_usd": 0.001,
        }
        verdict = controller.check(call_data)
        assert verdict == AdmissionVerdict.TRUNCATE
        assert verdict == "truncate"
        assert len(controller.events) == 1
        assert controller.events[0]["action"] == "truncate"


class TestHandoffBudget:
    """AdmissionController.request_handoff_budget — ADR-007."""

    def test_granted_when_sufficient(self) -> None:
        """Enough remaining → returns True and records consumption."""
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_wall_seconds=60.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        controller = AdmissionController(state)

        granted = controller.request_handoff_budget(
            estimated_tokens=500,
            estimated_cost_usd=0.05,
        )
        assert granted is True
        assert state.tokens == 500
        assert state.cost_usd == 0.05

    def test_denied_when_insufficient(self) -> None:
        """Not enough → returns False (replacement_denied_budget)."""
        budget = Budget(
            max_tokens=100,
            max_cost_usd=0.01,
            max_wall_seconds=60.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        controller = AdmissionController(state)

        # Request exceeds budget
        granted = controller.request_handoff_budget(
            estimated_tokens=500,
            estimated_cost_usd=0.05,
        )
        assert granted is False
        assert state.tokens == 0
        assert state.cost_usd == 0.0
