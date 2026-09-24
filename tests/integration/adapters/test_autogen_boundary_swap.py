"""Integration tests for AutoGen boundary-only swap timing (FR-19, ADR-008).

Exact requirement:
  "FR-19 (S, RE) On AutoGen, replacement shall occur at the next task boundary;
   the adapter shall declare this capability truthfully. Verified by capability
   contract test."

Key assertion:
  A swap requested mid-task does NOT take effect until the task boundary is reached;
  the old agent still produces the next in-task output and the new agent only appears
  after the boundary.
"""

from __future__ import annotations

import pytest

from oasis.adapters.autogen_adapter import AutoGenAdapter


class TestAutoGenBoundarySwapIntegration:
    """Verify that swaps requested mid-task take effect strictly at the task boundary."""

    @pytest.fixture
    def adapter(self) -> AutoGenAdapter:
        return AutoGenAdapter()

    def test_swap_requested_mid_task_takes_effect_only_at_task_boundary(
        self, adapter: AutoGenAdapter
    ) -> None:
        """A swap requested mid-task does NOT take effect until the task boundary is reached;
        assert the old agent still produces the next in-task output and the new agent only
        appears after the boundary (FR-19).
        """
        # Configure team with 2 roles and 2 turns per task/subtask
        config = {
            "roles": ["researcher", "critic"],
            "turns_per_task": 2,
        }
        runnable = adapter.build(config)

        # Multi-phase task with two distinct subtasks
        task = {
            "subtasks": ["subtask_1_investigate", "subtask_2_synthesize"],
            "turns_per_subtask": 2,
        }

        gen = adapter.run(runnable, task)

        # Turn 1: researcher produces output for subtask 1
        turn1 = next(gen)
        assert turn1["agent"] == "researcher"
        assert turn1["task"] == "subtask_1_investigate"
        assert turn1["turn"] == 1

        # Mid-task swap requested: replace researcher with lead_researcher
        swap_success = adapter.swap(
            runnable=runnable,
            out_agent="researcher",
            in_agent="lead_researcher",
            handoff={"findings": "Preliminary research complete"},
        )
        assert swap_success is True, "Swap request should be successfully accepted/queued"

        # Assert swap is queued and NOT yet applied to active roster
        assert len(runnable.pending_swaps) == 1
        assert "researcher" in runnable.roster
        assert "lead_researcher" not in runnable.roster

        # Turn 2: still within subtask 1 (before task boundary)!
        # CRITICAL ASSERTION: The OLD agent ("researcher") must still produce the next in-task output!
        turn2 = next(gen)
        assert turn2["agent"] == "researcher", (
            f"Expected old agent 'researcher' to produce next in-task output before boundary, "
            f"but got '{turn2['agent']}'"
        )
        assert turn2["task"] == "subtask_1_investigate"
        assert turn2["turn"] == 2

        # Turn 3: transition to subtask 2 (task boundary crossed)!
        # CRITICAL ASSERTION: The NEW agent ("lead_researcher") only appears after the boundary!
        turn3 = next(gen)
        assert turn3["agent"] == "lead_researcher", (
            f"Expected replacement agent 'lead_researcher' to produce output after task boundary, "
            f"but got '{turn3['agent']}'"
        )
        assert turn3["task"] == "subtask_2_synthesize"
        assert turn3["turn"] == 3

        # Confirm roster was updated at the boundary
        assert "lead_researcher" in runnable.roster
        assert "researcher" not in runnable.roster
        assert len(runnable.pending_swaps) == 0

        # Turn 4: second turn of subtask 2 proceeds with team
        turn4 = next(gen)
        assert turn4["turn"] == 4
        assert turn4["task"] == "subtask_2_synthesize"

        # No more turns expected
        with pytest.raises(StopIteration):
            next(gen)

    def test_swap_with_string_task_boundary(self, adapter: AutoGenAdapter) -> None:
        """Verify boundary swap behavior when task is specified as delimited string."""
        config = {
            "roles": ["analyst", "verifier"],
            "turns_per_task": 2,
        }
        runnable = adapter.build(config)

        # Two tasks separated by boundary delimiter
        task_str = "TaskPhase1: DataGathering\n---\nTaskPhase2: Analysis"
        gen = adapter.run(runnable, task_str)

        # Turn 1: in Phase 1
        t1 = next(gen)
        assert t1["agent"] == "analyst"

        # Request swap mid-phase
        adapter.swap(runnable, "analyst", "senior_analyst")

        # Turn 2: still Phase 1, old agent must speak
        t2 = next(gen)
        assert t2["agent"] == "analyst"

        # Turn 3: Phase 2 (after boundary), replacement agent must speak
        t3 = next(gen)
        assert t3["agent"] == "senior_analyst"

    def test_handoff_context_retained_on_boundary_swap(
        self, adapter: AutoGenAdapter
    ) -> None:
        """Verify handoff data is preserved and accessible to the replacement agent."""
        config = {"roles": ["agent_a", "agent_b"], "turns_per_task": 2}
        runnable = adapter.build(config)

        task = {"subtasks": ["step_1", "step_2"], "turns_per_subtask": 2}
        gen = adapter.run(runnable, task)

        next(gen)
        handoff_payload = {"summary": "critical context", "metrics": [1.0, 2.5]}
        adapter.swap(runnable, "agent_a", "agent_replacement", handoff=handoff_payload)

        # Next turn in step 1
        next(gen)
        # Next turn in step 2 (boundary crossed)
        next(gen)

        assert runnable.handoffs["agent_replacement"] == handoff_payload

    def test_truthful_capabilities_boundary_only(self, adapter: AutoGenAdapter) -> None:
        """Assert adapter capabilities match actual boundary-only behavior (FR-19)."""
        caps = adapter.capabilities()
        assert caps.mid_run_swap is False
        assert caps.boundary_swap is True
        assert caps.supports_mid_run_swap is False
        assert caps.supports_boundary_swap is True

    def test_manual_trigger_boundary(self, adapter: AutoGenAdapter) -> None:
        """Calling trigger_boundary manually applies pending swaps immediately."""
        config = {"roles": ["agent_x", "agent_y"]}
        runnable = adapter.build(config)

        adapter.swap(runnable, "agent_x", "agent_z")
        assert "agent_x" in runnable.roster
        assert "agent_z" not in runnable.roster

        applied = runnable.trigger_boundary()
        assert len(applied) == 1
        assert applied[0]["out_agent"] == "agent_x"
        assert applied[0]["in_agent"] == "agent_z"
        assert "agent_z" in runnable.roster
        assert "agent_x" not in runnable.roster

    def test_teardown_closes_groupchat_safely(self, adapter: AutoGenAdapter) -> None:
        """Teardown marks runnable as torn down and clears queues."""
        config = {"roles": ["agent_1"]}
        runnable = adapter.build(config)
        adapter.swap(runnable, "agent_1", "agent_2")

        adapter.teardown()
        assert runnable.is_torn_down is True
        assert len(runnable.pending_swaps) == 0

        # Attempting to run a torn down runnable raises RuntimeError
        with pytest.raises(RuntimeError, match="torn down"):
            list(adapter.run(runnable, "task"))
