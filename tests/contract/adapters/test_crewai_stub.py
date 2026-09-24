"""Contract tests for the CrewAI stub adapter.

ARCHITECTURE.md §5 & DECISIONS.md (ADR-008):
  "CrewAI in particular binds agents to tasks at crew construction,
   making mid-run swapping impractical without forking... the CrewAI
   adapter exists as a stub returning not-supported, documenting the
   limitation honestly."

TESTING.md (FR-19):
  "the CrewAI stub must claim and return not-supported (FR-19)."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from oasis.adapters.base import NOT_SUPPORTED_SIGNAL, AdapterCapabilities
from oasis.adapters.crewai_stub import CrewAIStub

# Ensure repo root is on sys.path so 'tests' package is resolvable
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.contract.adapters.test_contract_shared import BaseAdapterConformanceSuite


class TestCrewAIStubConformance(BaseAdapterConformanceSuite):
    """CrewAI stub plugged directly into the shared conformance suite."""

    __test__ = True

    @pytest.fixture
    def adapter(self) -> CrewAIStub:
        """Provide an instance of CrewAIStub for the conformance suite."""
        return CrewAIStub()


class TestCrewAIStubSpecifics:
    """Specific assertions verifying CrewAI stub's honest limitations (ADR-008)."""

    @pytest.fixture
    def stub(self) -> CrewAIStub:
        return CrewAIStub()

    def test_capabilities_truthfully_declares_no_swaps(self, stub: CrewAIStub) -> None:
        """capabilities() says no mid-run swap and no boundary swap."""
        caps = stub.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.mid_run_swap is False
        assert caps.boundary_swap is False
        assert caps.supports_mid_run_swap is False
        assert caps.supports_boundary_swap is False

    def test_swap_returns_not_supported_never_raises(self, stub: CrewAIStub) -> None:
        """Calling swap() returns not-supported rather than raising or silently succeeding."""
        # Must return False / NOT_SUPPORTED_SIGNAL without raising
        result = stub.swap(
            runnable=None,
            out_agent="agent_alpha",
            in_agent="agent_beta",
            handoff=None,
        )
        assert result is False
        assert result is NOT_SUPPORTED_SIGNAL
        assert result is not True

    def test_swap_with_arbitrary_inputs_never_raises_or_succeeds(
        self, stub: CrewAIStub
    ) -> None:
        """swap() returns not-supported under various edge case inputs."""
        cases = [
            (None, "", "", None),
            (object(), "out", "in", {"history": ["a", "b"]}),
            ("dummy_runnable", "worker_1", "worker_2", {}),
        ]
        for runnable, out_a, in_a, handoff in cases:
            res = stub.swap(runnable, out_a, in_a, handoff)
            assert res is False
            assert res is not True

    def test_teardown_is_safe_noop(self, stub: CrewAIStub) -> None:
        """teardown() completes cleanly without raising any exception."""
        stub.teardown()

    def test_build_documented_not_supported(self, stub: CrewAIStub) -> None:
        """build() raises documented NotImplementedError (ADR-008)."""
        with pytest.raises(NotImplementedError, match="CrewAI adapter is a documented stub"):
            stub.build({"roles": ["agent1"]})

    def test_run_documented_not_supported(self, stub: CrewAIStub) -> None:
        """run() raises documented NotImplementedError (ADR-008)."""
        with pytest.raises(NotImplementedError, match="CrewAI adapter is a documented stub"):
            stub.run(None, "sample task")
