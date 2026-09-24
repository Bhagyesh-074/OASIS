"""Unit tests for oasis.rbe.reducer — pre-execution team reduction (FR-6)."""

from __future__ import annotations

from oasis.rbe.budget import Budget
from oasis.rbe.reducer import TeamReducer


class TestTeamReducer:
    """TeamReducer — cost projection and iterative merge/prune."""

    def test_no_reduction_when_within_budget(self) -> None:
        """Team that already fits → returned unchanged, empty log."""
        team = [
            {"role": "researcher", "est_tokens": 1000, "est_cost_usd": 0.01},
            {"role": "coder", "est_tokens": 1000, "est_cost_usd": 0.01},
        ]
        budget = Budget(max_tokens=10_000, max_cost_usd=1.0, max_wall_seconds=60.0, max_calls=20)
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(team, budget)
        assert len(reduced_team) == 2
        assert len(reduction_log) == 0

    def test_single_reduction_logs_binding_constraint(self) -> None:
        """One merge → log records the binding constraint (FR-6)."""
        team = [
            {"role": "researcher", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
            {"role": "analyst", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
            {"role": "coder", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
        ]
        # 30k total tokens. 1 merge reduces to ~25k tokens. Budget 26k.
        budget = Budget(max_tokens=26_000, max_cost_usd=1.0, max_wall_seconds=60.0, max_calls=20)
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(team, budget)
        assert len(reduced_team) == 2
        assert len(reduction_log) == 1
        assert reduction_log[0]["binding_constraint"] == "max_tokens"
        assert reduction_log[0]["from_size"] == 3
        assert reduction_log[0]["to_size"] == 2

    def test_multiple_reductions_all_logged(self) -> None:
        """Three merges → three log entries with binding constraints."""
        team = [
            {"role": "researcher", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
            {"role": "analyst", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
            {"role": "coder", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
            {"role": "reviewer", "est_tokens": 10_000, "est_cost_usd": 0.10, "est_wall_seconds": 10.0, "est_calls": 5},
        ]
        # 40k -> 35k -> 30k -> 22.5k. Budget 25k forces 3 reductions to size 1.
        budget = Budget(max_tokens=25_000, max_cost_usd=1.0, max_wall_seconds=60.0, max_calls=20)
        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(team, budget)
        assert len(reduced_team) == 1
        assert len(reduction_log) == 3
        for entry in reduction_log:
            assert entry["binding_constraint"] == "max_tokens"

