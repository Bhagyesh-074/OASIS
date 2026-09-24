"""Contract tests for AutoGenAdapter — shared conformance suite & capability assertions.

FR-19 (S, RE):
  "On AutoGen, replacement shall occur at the next task boundary; the adapter
   shall declare this capability truthfully. Verified by capability contract test."

ARCHITECTURE.md §5 & DECISIONS.md (ADR-008):
  "AutoGen reports task-boundary swap only."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from oasis.adapters.autogen_adapter import AutoGenAdapter
from oasis.adapters.base import NOT_SUPPORTED_SIGNAL, AdapterCapabilities

# Ensure repo root is on sys.path so 'tests' package is resolvable
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.contract.adapters.test_contract_shared import BaseAdapterConformanceSuite


class TestAutoGenAdapterConformance(BaseAdapterConformanceSuite):
    """AutoGen adapter plugged into the shared conformance suite."""

    __test__ = True

    @pytest.fixture
    def adapter(self) -> AutoGenAdapter:
        return AutoGenAdapter()


class TestAutoGenAdapterContract:
    """Explicit capability declarations and contract tests for AutoGen (FR-19)."""

    @pytest.fixture
    def adapter(self) -> AutoGenAdapter:
        return AutoGenAdapter()

    def test_autogen_capabilities_truthfully_declares_boundary_only(
        self, adapter: AutoGenAdapter
    ) -> None:
        """FR-19: Must truthfully declare mid_run_swap=False and boundary_swap=True."""
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.mid_run_swap is False, "AutoGen must NOT claim mid-run swap"
        assert caps.boundary_swap is True, "AutoGen must claim boundary swap"
        assert caps.supports_mid_run_swap is False
        assert caps.supports_boundary_swap is True

    def test_autogen_swap_without_runnable_returns_not_supported(
        self, adapter: AutoGenAdapter
    ) -> None:
        """Calling swap without a runnable returns not-supported signal."""
        res = adapter.swap(
            runnable=None,
            out_agent="agent_a",
            in_agent="agent_b",
        )
        assert res is False
        assert res is NOT_SUPPORTED_SIGNAL

    def test_autogen_build_returns_runnable(self, adapter: AutoGenAdapter) -> None:
        """build(config) returns an AutoGenRunnable."""
        config = {
            "roles": ["researcher", "reviewer"],
            "turns_per_task": 2,
        }
        runnable = adapter.build(config)
        assert runnable is not None
        assert hasattr(runnable, "groupchat")
        assert hasattr(runnable, "roster")
        assert "researcher" in runnable.roster
        assert "reviewer" in runnable.roster

    def test_autogen_swap_with_runnable_queues_successfully(
        self, adapter: AutoGenAdapter
    ) -> None:
        """Calling swap on a valid runnable successfully queues the boundary swap."""
        runnable = adapter.build({"roles": ["agent_1", "agent_2"]})
        result = adapter.swap(
            runnable=runnable,
            out_agent="agent_1",
            in_agent="replacement_1",
            handoff={"key": "value"},
        )
        assert result is True
        assert len(runnable.pending_swaps) == 1
        assert runnable.pending_swaps[0].out_agent == "agent_1"
        assert runnable.pending_swaps[0].in_agent == "replacement_1"

    def test_teardown_cleans_resources(self, adapter: AutoGenAdapter) -> None:
        """teardown releases all runnable resources."""
        runnable = adapter.build({"roles": ["agent_1", "agent_2"]})
        adapter.teardown()
        assert runnable.is_torn_down is True
        # Swapping on a torn down runnable returns False
        assert adapter.swap(runnable, "agent_1", "agent_2") is False
