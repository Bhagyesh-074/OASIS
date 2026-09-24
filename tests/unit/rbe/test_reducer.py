"""Unit tests for oasis.rbe.reducer — FR-6 pre-execution team reduction."""

from __future__ import annotations

from typing import Any

import pytest

from oasis.rbe.budget import Budget, BudgetDimension
from oasis.rbe.reducer import BudgetInfeasibleError, TeamReducer


@pytest.fixture
def proposed_four_agent_team() -> list[dict[str, Any]]:
    """Fixture team of 4 roles with equal cost estimates.

    Initial projection:
      tokens: 40,000
      cost_usd: 0.40
      wall_seconds: 40.0
      calls: 20
    """
    return [
        {
            "role": "researcher",
            "est_tokens": 10_000,
            "est_cost_usd": 0.10,
            "est_wall_seconds": 10.0,
            "est_calls": 5,
        },
        {
            "role": "analyst",
            "est_tokens": 10_000,
            "est_cost_usd": 0.10,
            "est_wall_seconds": 10.0,
            "est_calls": 5,
        },
        {
            "role": "coder",
            "est_tokens": 10_000,
            "est_cost_usd": 0.10,
            "est_wall_seconds": 10.0,
            "est_calls": 5,
        },
        {
            "role": "reviewer",
            "est_tokens": 10_000,
            "est_cost_usd": 0.10,
            "est_wall_seconds": 10.0,
            "est_calls": 5,
        },
    ]


class TestTeamReducer:
    """FR-6: Pre-execution team cost projection and iterative reduction."""

    def test_budget_already_satisfied_triggers_zero_reductions(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """A budget already satisfied by the proposed team triggers zero reductions."""
        budget = Budget(
            max_tokens=50_000,
            max_cost_usd=0.50,
            max_wall_seconds=50.0,
            max_calls=25,
        )
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(proposed_four_agent_team, budget)

        assert len(reduced_team) == 4
        assert len(reduction_log) == 0

    def test_budget_forcing_exactly_one_reduction(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """A budget forcing exactly 1 reduction ends with correct final team size (3)
        and one logged binding-constraint entry."""
        # Initial is 40,000 tokens. 1 reduction brings it to 35,000 tokens.
        budget = Budget(
            max_tokens=36_000,
            max_cost_usd=1.00,
            max_wall_seconds=100.0,
            max_calls=50,
        )
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(proposed_four_agent_team, budget)

        assert len(reduced_team) == 3
        assert len(reduction_log) == 1

        entry = reduction_log[0]
        assert entry["from_size"] == 4
        assert entry["to_size"] == 3
        assert entry["binding_constraint"] == "max_tokens"
        assert len(entry["merged_roles"]) == 2
        assert "max_tokens" in entry["justification"]

    def test_budget_forcing_exactly_two_reductions(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """A budget forcing exactly 2 reductions ends with team size 2 and 2 logged entries."""
        # 1 reduction -> 35k. 2 reductions -> 30k. Budget is 31k.
        budget = Budget(
            max_tokens=31_000,
            max_cost_usd=1.00,
            max_wall_seconds=100.0,
            max_calls=50,
        )
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(proposed_four_agent_team, budget)

        assert len(reduced_team) == 2
        assert len(reduction_log) == 2

        assert reduction_log[0]["from_size"] == 4
        assert reduction_log[0]["to_size"] == 3
        assert reduction_log[0]["binding_constraint"] == "max_tokens"

        assert reduction_log[1]["from_size"] == 3
        assert reduction_log[1]["to_size"] == 2
        assert reduction_log[1]["binding_constraint"] == "max_tokens"

    def test_budget_forcing_exactly_three_reductions(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """A budget forcing exactly 3 reductions ends with team size 1 and 3 logged entries."""
        # 1 reduction -> 35k. 2 reductions -> 30k. 3 reductions -> 25k. Budget is 26k.
        budget = Budget(
            max_tokens=26_000,
            max_cost_usd=1.00,
            max_wall_seconds=100.0,
            max_calls=50,
        )
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(proposed_four_agent_team, budget)


        assert len(reduced_team) == 1
        assert len(reduction_log) == 3

        for i, entry in enumerate(reduction_log):
            assert entry["from_size"] == 4 - i
            assert entry["to_size"] == 3 - i
            assert entry["binding_constraint"] == "max_tokens"

    def test_logged_entries_name_real_dimensions_not_placeholders(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """Each logged entry names a real one of the four dimensions, not a placeholder."""
        valid_dimension_values = {d.value for d in BudgetDimension}

        # Case 1: Bound by cost
        cost_budget = Budget(
            max_tokens=100_000,
            max_cost_usd=0.36,  # 40k cost is $0.40 -> breaches
            max_wall_seconds=100.0,
            max_calls=100,
        )
        reducer = TeamReducer()
        _, log_cost = reducer.reduce(proposed_four_agent_team, cost_budget)
        assert len(log_cost) == 1
        assert log_cost[0]["binding_constraint"] == "max_cost_usd"
        assert log_cost[0]["binding_constraint"] in valid_dimension_values

        # Case 2: Bound by wall_seconds
        sec_budget = Budget(
            max_tokens=100_000,
            max_cost_usd=10.0,
            max_wall_seconds=36.0,  # 4-role sec is 40.0 -> breaches
            max_calls=100,
        )
        _, log_sec = reducer.reduce(proposed_four_agent_team, sec_budget)
        assert len(log_sec) == 1
        assert log_sec[0]["binding_constraint"] == "max_wall_seconds"
        assert log_sec[0]["binding_constraint"] in valid_dimension_values

        # Case 3: Bound by calls
        calls_budget = Budget(
            max_tokens=100_000,
            max_cost_usd=10.0,
            max_wall_seconds=100.0,
            max_calls=19,  # 4-role calls is 20 -> breaches
        )
        _, log_calls = reducer.reduce(proposed_four_agent_team, calls_budget)
        assert len(log_calls) == 1
        assert log_calls[0]["binding_constraint"] == "max_calls"
        assert log_calls[0]["binding_constraint"] in valid_dimension_values

    def test_infeasible_budget_raises_budget_infeasible_error(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """A budget too tight even for a single-agent team raises BudgetInfeasibleError."""
        impossible_budget = Budget(
            max_tokens=5_000,  # Min 1-agent team needs ~22.5k tokens
            max_cost_usd=1.00,
            max_wall_seconds=100.0,
            max_calls=100,
        )
        reducer = TeamReducer()
        with pytest.raises(BudgetInfeasibleError) as exc_info:
            reducer.reduce(proposed_four_agent_team, impossible_budget)

        assert exc_info.value.binding_constraint == "max_tokens"
        assert len(exc_info.value.reduced_team) == 1
        assert len(exc_info.value.reduction_log) == 3

    def test_pluggable_similarity_function_influences_merge_order(
        self, proposed_four_agent_team: list[dict[str, Any]]
    ) -> None:
        """Custom similarity function controls which roles are merged first."""
        # Prioritize merging researcher and coder
        def custom_sim(a: dict[str, Any], b: dict[str, Any]) -> float:
            pair = {a.get("role"), b.get("role")}
            if pair == {"researcher", "coder"}:
                return 1.0
            return 0.1

        budget = Budget(
            max_tokens=36_000,
            max_cost_usd=1.00,
            max_wall_seconds=100.0,
            max_calls=50,
        )
        reducer = TeamReducer(similarity_fn=custom_sim)
        _, log = reducer.reduce(proposed_four_agent_team, budget)

        assert len(log) == 1
        assert set(log[0]["merged_roles"]) == {"researcher", "coder"}
