"""Shared fixtures for contract/adapters tests."""

from __future__ import annotations

import pytest

from oasis.adapters.autogen_adapter import AutoGenAdapter
from oasis.adapters.crewai_stub import CrewAIStub
from oasis.adapters.langgraph_adapter import LangGraphAdapter


@pytest.fixture(params=["langgraph", "autogen", "crewai"])
def adapter(request: pytest.FixtureRequest):
    """Parametrised fixture yielding each adapter for conformance testing."""
    adapters = {
        "langgraph": LangGraphAdapter,
        "autogen": AutoGenAdapter,
        "crewai": CrewAIStub,
    }
    return adapters[request.param]()
