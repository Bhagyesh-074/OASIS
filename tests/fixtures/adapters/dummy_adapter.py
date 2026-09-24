"""Dummy framework adapter fixture for NFR-7 isolation testing.

NFR-7 (M):
  "Adding a new framework adapter shall require no changes to TCE, RBE,
   RTPM, RE or CM."
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from oasis.adapters.base import AdapterBase, AdapterCapabilities


class DummyIsolatedAdapter(AdapterBase):
    """A fourth dummy adapter proving that adding an adapter requires zero changes to core modules."""

    def __init__(self) -> None:
        self.is_torn_down: bool = False
        self.swapped_agents: list[tuple[str, str]] = []

    def build(self, config: dict[str, Any]) -> Any:
        """Construct a minimal dummy runnable."""
        return {"runnable": "dummy_runnable", "config": config}

    def run(self, runnable: Any, task: str) -> Generator[dict[str, Any], None, None]:
        """Yield a dummy output event."""
        yield {
            "agent": "dummy_agent",
            "content": f"Dummy output for task: {task}",
        }

    def swap(
        self,
        runnable: Any,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Record swap and return success."""
        self.swapped_agents.append((out_agent, in_agent))
        return True

    def capabilities(self) -> AdapterCapabilities:
        """Report mid-run and boundary swap capability."""
        return AdapterCapabilities(mid_run_swap=True, boundary_swap=True)

    def teardown(self) -> None:
        """Clean up dummy adapter resources."""
        self.is_torn_down = True
