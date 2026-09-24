"""Unit tests for NFR-3 cost ceilings: per-run hard ceiling and pure kill-switch calculations."""

from __future__ import annotations

import pytest

from oasis.rbe.admission import (
    AdmissionController,
    AdmissionVerdict,
    admission_check,
    check_kill_switch,
    check_spend_ceiling,
)
from oasis.rbe.budget import Budget, BudgetState


class TestGlobalKillSwitchPureCalculations:
    """NFR-3 pure kill-switch and spend-ceiling interface for Vidish (db/ and api/)."""

    def test_check_kill_switch_strictly_at_and_above_120_percent(self) -> None:
        """check_kill_switch() returns True only at/above 120% of given ceiling and False below it."""
        ceiling = 250.0  # e.g. OASIS_SPEND_CEILING_USD

        # Below 120% (ceiling * 1.2 = 300.0)
        assert check_kill_switch(0.0, ceiling) is False
        assert check_kill_switch(249.99, ceiling) is False
        assert check_kill_switch(250.0, ceiling) is False
        assert check_kill_switch(299.99, ceiling) is False

        # At exactly 120%
        assert check_kill_switch(300.0, ceiling) is True

        # Above 120%
        assert check_kill_switch(300.01, ceiling) is True
        assert check_kill_switch(500.0, ceiling) is True

    def test_check_spend_ceiling_at_100_percent(self) -> None:
        """check_spend_ceiling() returns True at/above 100% of ceiling to block run creation."""
        ceiling = 250.0

        assert check_spend_ceiling(100.0, ceiling) is False
        assert check_spend_ceiling(249.99, ceiling) is False
        assert check_spend_ceiling(250.0, ceiling) is True
        assert check_spend_ceiling(251.0, ceiling) is True


class TestPerRunHardCeiling:
    """NFR-3 per-run hard ceiling (OASIS_PER_RUN_MAX_COST_USD)."""

    def test_per_run_ceiling_triggers_halt_regardless_of_generous_budget(self) -> None:
        """Even if the declared budget allows $100.00, OASIS_PER_RUN_MAX_COST_USD=$2.00 halts when reached."""
        # Declared budget is very generous ($100.00, 100,000 tokens)
        budget = Budget(
            max_tokens=100_000,
            max_cost_usd=100.0,
            max_wall_seconds=1000.0,
            max_calls=1000,
        )
        state = BudgetState(budget)
        # Already consumed $1.99 in this run
        state.record(tokens=1000, cost_usd=1.99, calls=1)

        # per_run_max_cost_usd is set to $2.00
        controller = AdmissionController(state, per_run_max_cost_usd=2.00)

        # Next call projects $0.05 spend -> total would be $2.04 > $2.00 limit
        call_data = {
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "hello"}],
            "projected_cost_usd": 0.05,
        }
        verdict = controller.check(call_data)

        assert verdict == AdmissionVerdict.HALT
        assert len(controller.events) == 1
        event = controller.events[0]
        assert event["action"] == "halt"
        assert event["dimension"] == "cost"
        assert "OASIS_PER_RUN_MAX_COST_USD" in event["justification"]

    def test_per_run_ceiling_from_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Controller picks up OASIS_PER_RUN_MAX_COST_USD from os.environ when not explicitly passed."""
        monkeypatch.setenv("OASIS_PER_RUN_MAX_COST_USD", "1.50")

        budget = Budget(
            max_tokens=100_000,
            max_cost_usd=50.0,
            max_wall_seconds=500.0,
            max_calls=50,
        )
        state = BudgetState(budget)
        state.record(tokens=1000, cost_usd=1.50, calls=1)

        controller = AdmissionController(state)
        call_data = {
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "next call"}],
            "projected_cost_usd": 0.01,
        }
        verdict = controller.check(call_data)
        assert verdict == AdmissionVerdict.HALT
        assert "OASIS_PER_RUN_MAX_COST_USD" in controller.events[0]["justification"]

    def test_admission_check_direct_with_per_run_max_cost(self) -> None:
        """admission_check pure function evaluates per_run_max_cost_usd."""
        remaining = {"tokens": 100_000, "cost": 50.0, "wall": 100.0, "calls": 50}
        projected = {"tokens": 100, "cost": 0.50, "wall": 1.0, "calls": 1}

        # Current cost = 1.60, limit = 2.00, projected = 0.50 -> 2.10 > 2.00
        verdict, dim, justification = admission_check(
            remaining,
            projected,
            per_run_max_cost_usd=2.00,
            current_cost_usd=1.60,
        )
        assert verdict == AdmissionVerdict.HALT
        assert dim == "cost"
        assert "Per-run cost ceiling" in justification
