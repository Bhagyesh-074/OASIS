"""CrewAI documented not-supported stub (ADR-008, ARCHITECTURE.md §5).

CrewAI binds agents to tasks at crew construction, making mid-run
swapping impractical without forking the framework.  This adapter exists
to document the limitation honestly (ADR-008):
  "CrewAI in particular binds agents to tasks at crew construction,
   making mid-run swapping impractical without forking... the CrewAI
   adapter exists as a stub returning not-supported, documenting the
   limitation honestly."

Results are reported separately by capability rather than averaged over
frameworks with different powers (ADR-008).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from oasis.adapters.base import NOT_SUPPORTED_SIGNAL, AdapterBase, AdapterCapabilities


class CrewAIStub(AdapterBase):
    """CrewAI stub — documents the limitation; swap() returns not-supported."""

    def build(self, config: dict[str, Any]) -> Any:
        """Not supported — CrewAI dynamic integration is out of scope for OASIS.

        ARCHITECTURE.md §5: 'build(config) returns a runnable'
        ADR-008: CrewAI adapter exists as a documented stub.

        Raises
        ------
        NotImplementedError
            CrewAI adapter is a documented stub (ADR-008).
        """
        raise NotImplementedError("CrewAI adapter is a documented stub (ADR-008)")

    def run(self, runnable: Any, task: str) -> Generator[dict[str, Any], None, None]:
        """Not supported — execution is not implemented for the CrewAI stub.

        ARCHITECTURE.md §5: 'run(runnable, task) yields output events'
        ADR-008: CrewAI adapter exists as a documented stub.

        Raises
        ------
        NotImplementedError
            CrewAI adapter is a documented stub (ADR-008).
        """
        raise NotImplementedError("CrewAI adapter is a documented stub (ADR-008)")

    def swap(
        self,
        runnable: Any,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Return not-supported signal. CrewAI cannot swap agents mid-run.

        ARCHITECTURE.md §5: 'swap(runnable, out_agent, in_agent, handoff) returns success or a not-supported signal'
        ADR-008: CrewAI binds agents to tasks at crew construction; mid-run swap is impractical.

        Returns
        -------
        Always ``False`` (NOT_SUPPORTED_SIGNAL). Never raises, never pretends to succeed.
        """
        return NOT_SUPPORTED_SIGNAL

    def capabilities(self) -> AdapterCapabilities:
        """Truthfully declare that CrewAI supports neither mid-run nor boundary swap.

        ARCHITECTURE.md §5: 'capabilities() declares whether mid-run swap is available'
        FR-19: Capability contract test prevents over-claiming.
        """
        return AdapterCapabilities(mid_run_swap=False, boundary_swap=False)

    def teardown(self) -> None:
        """No-op — nothing to release."""


__all__ = ["CrewAIStub"]
