"""Contract tests for LangGraphAdapter — shared conformance suite & capability assertions.

FR-18 (M, RE):
  "On LangGraph, replacement shall occur without restarting the run,
   preserving accumulated state. Verified by integration test asserting
   pre-swap state survives."

ARCHITECTURE.md §5 & DECISIONS.md (ADR-008):
  "LangGraph reports full mid-run swap."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from oasis.adapters.base import NOT_SUPPORTED_SIGNAL, AdapterCapabilities
from oasis.adapters.langgraph_adapter import CompiledStateGraph, LangGraphAdapter

# Ensure repo root is on sys.path so 'tests' package is resolvable
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.contract.adapters.test_contract_shared import BaseAdapterConformanceSuite


class TestLangGraphAdapterConformance(BaseAdapterConformanceSuite):
    """LangGraph adapter plugged into the shared conformance suite."""

    __test__ = True

    @pytest.fixture
    def adapter(self) -> LangGraphAdapter:
        return LangGraphAdapter()


class TestLangGraphAdapterContract:
    """Explicit capability declarations and contract tests for LangGraph (ADR-008, FR-18)."""

    @pytest.fixture
    def adapter(self) -> LangGraphAdapter:
        return LangGraphAdapter()

    def test_langgraph_capabilities_truthfully_declares_mid_run_swap(
        self, adapter: LangGraphAdapter
    ) -> None:
        """LangGraph truthfully declares mid_run_swap=True and boundary_swap=True."""
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.mid_run_swap is True, "LangGraph must claim mid-run swap"
        assert caps.boundary_swap is True, "LangGraph must claim boundary swap"
        assert caps.supports_mid_run_swap is True
        assert caps.supports_boundary_swap is True

    def test_langgraph_build_returns_compiled_state_graph(
        self, adapter: LangGraphAdapter
    ) -> None:
        """build(config) constructs a CompiledStateGraph runnable."""
        config = {
            "roles": ["researcher", "critic"],
            "turns_per_role": 1,
        }
        runnable = adapter.build(config)
        assert runnable is not None
        assert isinstance(runnable, CompiledStateGraph)
        assert hasattr(runnable, "nodes")
        assert hasattr(runnable, "supervisor")
        assert hasattr(runnable, "state")
        assert "researcher" in runnable.nodes
        assert "critic" in runnable.nodes

    def test_langgraph_swap_without_runnable_returns_not_supported(
        self, adapter: LangGraphAdapter
    ) -> None:
        """Calling swap without a runnable returns not-supported signal."""
        res = adapter.swap(
            runnable=None,
            out_agent="agent_a",
            in_agent="agent_b",
        )
        assert res is False
        assert res is NOT_SUPPORTED_SIGNAL

    def test_langgraph_swap_with_runnable_succeeds(
        self, adapter: LangGraphAdapter
    ) -> None:
        """Calling swap on a valid runnable succeeds and rewrites edges."""
        runnable = adapter.build({"roles": ["worker", "reviewer"]})
        result = adapter.swap(
            runnable=runnable,
            out_agent="worker",
            in_agent="senior_worker",
            handoff={"notes": "pre-swap briefing", "tokens": 50},
        )
        assert result is True
        assert "senior_worker" in runnable.nodes
        assert runnable.supervisor.route_table.get("worker") == "senior_worker"

    def test_langgraph_teardown_cleans_resources(
        self, adapter: LangGraphAdapter
    ) -> None:
        """teardown releases graph resources and marks runnables as torn down."""
        runnable = adapter.build({"roles": ["worker"]})
        adapter.teardown()
        assert runnable.is_torn_down is True
        assert adapter.swap(runnable, "worker", "replacement") is False
