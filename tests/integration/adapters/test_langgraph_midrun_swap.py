"""Integration tests for LangGraph mid-run swap & state preservation (FR-18, ADR-008).

Exact requirement:
  "FR-18 (M, RE) On LangGraph, replacement shall occur without restarting the run,
   preserving accumulated state. Verified by integration test asserting pre-swap
   state survives."

Key assertions:
  Construct a graph, execute partway, accumulate state, swap an agent,
  and assert the pre-swap accumulated state is still present and correct after
  the swap, and the run continues without restarting.
"""

from __future__ import annotations

import pytest

from oasis.adapters.langgraph_adapter import CompiledStateGraph, LangGraphAdapter


class TestLangGraphMidRunSwapIntegration:
    """Verify that mid-run swaps preserve state and continue execution without restarting."""

    @pytest.fixture
    def adapter(self) -> LangGraphAdapter:
        return LangGraphAdapter()

    def test_midrun_swap_preserves_accumulated_state_without_restarting(
        self, adapter: LangGraphAdapter
    ) -> None:
        """Execute partway, accumulate state, swap an agent, and assert pre-swap
        accumulated state is still present and correct after the swap, and the
        run continues without restarting (FR-18).
        """
        # Configure team with 2 roles
        config = {
            "roles": ["researcher", "analyst"],
            "turns_per_role": 1,
        }
        runnable = adapter.build(config)
        assert isinstance(runnable, CompiledStateGraph)
        original_runnable_id = id(runnable)

        task = "Analyze AI safety alignment mechanisms and summarize findings."
        gen = adapter.run(runnable, task)

        # Step 1: Supervisor routes to researcher
        ev1 = next(gen)
        assert ev1["agent"] == "supervisor"
        assert ev1["node"] == "supervisor"
        assert ev1["step"] == 1

        # Step 2: Researcher executes and produces output
        ev2 = next(gen)
        assert ev2["agent"] == "researcher"
        assert ev2["node"] == "researcher"
        assert ev2["step"] == 2
        assert "researcher" in ev2["content"]

        # Snapshot accumulated pre-swap state
        pre_swap_messages = list(runnable.state["messages"])
        pre_swap_step = runnable.step_count
        pre_swap_data = list(runnable.state["accumulated_data"])

        assert len(pre_swap_messages) == 2  # supervisor event + researcher event
        assert pre_swap_step == 2
        assert len(pre_swap_data) == 1
        assert "researcher" in pre_swap_data[0]

        # ==================================================================
        # MID-RUN SWAP OCCURS HERE
        # ==================================================================
        handoff_payload = {
            "tokens": 150,
            "cost_usd": 0.0045,
            "context": "Preliminary research complete; focus on verification.",
            "summary": "AI safety alignment survey draft",
        }
        swap_success = adapter.swap(
            runnable=runnable,
            out_agent="researcher",
            in_agent="senior_researcher",
            handoff=handoff_payload,
        )
        assert swap_success is True

        # CRITICAL ASSERTION 1: Runnable identity is unchanged (no graph rebuild/restart)
        assert id(runnable) == original_runnable_id

        # CRITICAL ASSERTION 2: Pre-swap accumulated state survives intact
        assert runnable.state["messages"][:len(pre_swap_messages)] == pre_swap_messages
        assert runnable.state["accumulated_data"] == pre_swap_data
        assert runnable.state["step_count"] == pre_swap_step, "Step count must NOT reset to 0"

        # CRITICAL ASSERTION 3: Handoff payload is ingested into accumulated state
        assert runnable.state["handoffs"]["senior_researcher"] == handoff_payload

        # ==================================================================
        # RESUME EXECUTION (Run continues forward without restarting)
        # ==================================================================
        # Step 3: Supervisor routes to analyst
        ev3 = next(gen)
        assert ev3["agent"] == "supervisor"
        assert ev3["step"] == 3
        assert ev3["step"] > pre_swap_step, "Run must continue forward, not restart"

        # Step 4: Analyst executes
        ev4 = next(gen)
        assert ev4["agent"] == "analyst"
        assert ev4["step"] == 4

        # Run completes naturally
        with pytest.raises(StopIteration):
            next(gen)

        # Final state check: pre-swap messages still present, plus post-swap messages
        all_messages = runnable.state["messages"]
        assert len(all_messages) == 4
        assert all_messages[0]["agent"] == "supervisor"
        assert all_messages[1]["agent"] == "researcher"
        assert all_messages[2]["agent"] == "supervisor"
        assert all_messages[3]["agent"] == "analyst"

    def test_midrun_swap_rewrites_future_turns_to_replacement(
        self, adapter: LangGraphAdapter
    ) -> None:
        """When an agent with remaining turns is swapped mid-run, the replacement
        agent executes the subsequent turns for that role, reading prior state.
        """
        config = {
            "roles": ["investigator"],
            "turns_per_role": 2,  # 2 turns allocated for investigator role
        }
        runnable = adapter.build(config)
        gen = adapter.run(runnable, "Investigate transaction anomaly")

        # Turn 1: Supervisor routes to investigator
        next(gen)
        # Turn 2: First investigator turn completes
        ev_turn1 = next(gen)
        assert ev_turn1["agent"] == "investigator"

        # Mid-run swap: replace investigator with forensic_investigator
        handoff = {"reason": "escalation", "evidence_ids": [101, 102]}
        adapter.swap(
            runnable,
            out_agent="investigator",
            in_agent="forensic_investigator",
            handoff=handoff,
        )

        # Pre-swap state has 2 messages
        assert len(runnable.state["messages"]) == 2
        assert runnable.state["messages"][1]["agent"] == "investigator"

        # Turn 3: Supervisor routes next turn to the replacement!
        ev_sup = next(gen)
        assert ev_sup["agent"] == "supervisor"
        assert "forensic_investigator" in ev_sup["content"]

        # Turn 4: Replacement agent executes turn 2
        ev_turn2 = next(gen)
        assert ev_turn2["agent"] == "forensic_investigator"
        assert "forensic_investigator" in ev_turn2["content"]

        # Final state contains outputs from BOTH investigator and forensic_investigator
        agents_in_state = [m["agent"] for m in runnable.state["messages"]]
        assert "investigator" in agents_in_state
        assert "forensic_investigator" in agents_in_state

    def test_swap_on_torn_down_runnable_fails_gracefully(
        self, adapter: LangGraphAdapter
    ) -> None:
        """swap on a torn down runnable returns False without raising."""
        runnable = adapter.build({"roles": ["worker"]})
        adapter.teardown()
        assert runnable.is_torn_down is True
        res = adapter.swap(runnable, "worker", "replacement")
        assert res is False

    def test_truthful_capabilities_reports_mid_run_swap(
        self, adapter: LangGraphAdapter
    ) -> None:
        """Truthful capability reporting matches ADR-008 and ARCHITECTURE.md §5."""
        caps = adapter.capabilities()
        assert caps.mid_run_swap is True
        assert caps.boundary_swap is True
        assert caps.supports_mid_run_swap is True
        assert caps.supports_boundary_swap is True
