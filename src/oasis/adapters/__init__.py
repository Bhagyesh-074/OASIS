"""OASIS Framework Adapters.

Framework-agnostic adapter layer (NFR-7, ADR-008).  Every adapter
implements the five-operation contract defined in :mod:`~oasis.adapters.base`:
``build``, ``run``, ``swap``, ``capabilities``, ``teardown``.

Adding a new framework adapter shall require no changes to TCE, RBE,
RTPM, RE or CM (NFR-7).

Current adapters:
  - **LangGraph** — deep integration: compiled StateGraph, mid-run swap.
  - **AutoGen** — shallow integration: GroupChat + custom speaker
    selector, boundary swap only.
  - **CrewAI** — documented not-supported stub (ADR-008).
"""

from oasis.adapters.autogen_adapter import AutoGenAdapter
from oasis.adapters.base import (
    NOT_SUPPORTED_SIGNAL,
    AdapterBase,
    AdapterCapabilities,
    BaseAdapter,
)
from oasis.adapters.crewai_stub import CrewAIStub
from oasis.adapters.langgraph_adapter import LangGraphAdapter

__all__ = [
    "NOT_SUPPORTED_SIGNAL",
    "AdapterBase",
    "AdapterCapabilities",
    "AutoGenAdapter",
    "BaseAdapter",
    "CrewAIStub",
    "LangGraphAdapter",
]
