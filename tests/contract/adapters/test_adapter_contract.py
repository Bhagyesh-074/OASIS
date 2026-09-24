"""Contract tests for adapters — five-operation conformance suite.

Every adapter must pass these tests.  Key assertions (TESTING.md):
  - capabilities() matches actual behaviour
  - CrewAI stub must claim and return not-supported (FR-19, ADR-008)
"""

from __future__ import annotations

from oasis.adapters.base import AdapterBase, AdapterCapabilities


class TestAdapterConformance:
    """Shared conformance suite run against every adapter (via conftest fixture)."""

    def test_is_adapter_base_subclass(self, adapter) -> None:
        """Every adapter is a subclass of AdapterBase."""
        assert isinstance(adapter, AdapterBase)

    def test_capabilities_returns_adapter_capabilities(self, adapter) -> None:
        """capabilities() returns an AdapterCapabilities dataclass."""
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)

    def test_capabilities_mid_run_swap_is_bool(self, adapter) -> None:
        """mid_run_swap is a boolean."""
        assert isinstance(adapter.capabilities().mid_run_swap, bool)

    def test_capabilities_boundary_swap_is_bool(self, adapter) -> None:
        """boundary_swap is a boolean."""
        assert isinstance(adapter.capabilities().boundary_swap, bool)


class TestCrewAIStubContract:
    """CrewAI stub must honestly report not-supported (ADR-008)."""

    def test_crewai_capabilities_no_mid_run_swap(self) -> None:
        """CrewAI reports mid_run_swap=False."""
        from oasis.adapters.crewai_stub import CrewAIStub

        stub = CrewAIStub()
        assert stub.capabilities().mid_run_swap is False

    def test_crewai_capabilities_no_boundary_swap(self) -> None:
        """CrewAI reports boundary_swap=False."""
        from oasis.adapters.crewai_stub import CrewAIStub

        stub = CrewAIStub()
        assert stub.capabilities().boundary_swap is False

    def test_crewai_swap_returns_false(self) -> None:
        """swap() returns False (not-supported signal)."""
        from oasis.adapters.crewai_stub import CrewAIStub

        stub = CrewAIStub()
        assert stub.swap(None, "a", "b") is False

    def test_crewai_teardown_is_noop(self) -> None:
        """teardown() completes without error."""
        from oasis.adapters.crewai_stub import CrewAIStub

        stub = CrewAIStub()
        stub.teardown()  # should not raise


class TestLangGraphCapabilities:
    """LangGraph adapter capability declarations."""

    def test_langgraph_reports_mid_run_swap(self) -> None:
        """LangGraph claims mid_run_swap=True."""
        from oasis.adapters.langgraph_adapter import LangGraphAdapter

        adapter = LangGraphAdapter()
        assert adapter.capabilities().mid_run_swap is True

    def test_langgraph_reports_boundary_swap(self) -> None:
        """LangGraph claims boundary_swap=True."""
        from oasis.adapters.langgraph_adapter import LangGraphAdapter

        adapter = LangGraphAdapter()
        assert adapter.capabilities().boundary_swap is True


class TestAutoGenCapabilities:
    """AutoGen adapter capability declarations."""

    def test_autogen_no_mid_run_swap(self) -> None:
        """AutoGen claims mid_run_swap=False."""
        from oasis.adapters.autogen_adapter import AutoGenAdapter

        adapter = AutoGenAdapter()
        assert adapter.capabilities().mid_run_swap is False

    def test_autogen_reports_boundary_swap(self) -> None:
        """AutoGen claims boundary_swap=True."""
        from oasis.adapters.autogen_adapter import AutoGenAdapter

        adapter = AutoGenAdapter()
        assert adapter.capabilities().boundary_swap is True
