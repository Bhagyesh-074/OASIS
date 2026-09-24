"""The five-operation adapter contract: build / run / swap / capabilities / teardown.

Every adapter implements the same five operations (ARCHITECTURE.md §5):
  - ``build(config)`` → returns a runnable
  - ``run(runnable, task)`` → yields output events
  - ``swap(runnable, out_agent, in_agent, handoff)`` → success or
    not-supported signal
  - ``capabilities()`` → declares whether mid-run swap is available
  - ``teardown()``

Contract tests enforce that ``capabilities()`` matches actual behaviour:
an adapter claiming mid-run swap must demonstrate it, and the CrewAI stub
must claim and return not-supported (TESTING.md, FR-19).

Adding a new adapter shall require no changes to TCE, RBE, RTPM, RE or
CM (NFR-7).
"""

from __future__ import annotations

import abc
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

# Canonical signal returned by swap() when mid-run swapping is not supported
NOT_SUPPORTED_SIGNAL: bool = False


@dataclass(frozen=True)
class AdapterCapabilities:
    """What this adapter can actually do.

    Contract tests verify that claimed capabilities match real behaviour (TESTING.md).
    """

    mid_run_swap: bool = False
    """Can agents be swapped without restarting the run?"""

    boundary_swap: bool = False
    """Can agents be swapped at task boundaries?"""

    @property
    def supports_mid_run_swap(self) -> bool:
        """Alias for mid_run_swap."""
        return self.mid_run_swap

    @property
    def supports_boundary_swap(self) -> bool:
        """Alias for boundary_swap."""
        return self.boundary_swap


class AdapterBase(abc.ABC):
    """Abstract base for all framework adapters.

    Subclasses must implement all five operations (ARCHITECTURE.md §5):
      build, run, swap, capabilities, teardown.
    """

    # ------------------------------------------------------------------
    # 1. Build: build(config) returns a runnable
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build(self, config: dict[str, Any]) -> Any:
        """Translate an OASIS team configuration into a framework-native runnable.

        ARCHITECTURE.md §5: 'build(config) returns a runnable'

        Parameters
        ----------
        config:
            Team configuration dict (roles, topology, model assignments).

        Returns
        -------
        A framework-native runnable object.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        ...

    # ------------------------------------------------------------------
    # 2. Run: run(runnable, task) yields output events
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def run(self, runnable: Any, task: str) -> Generator[dict[str, Any], None, None]:
        """Execute the task and yield output events.

        ARCHITECTURE.md §5: 'run(runnable, task) yields output events'

        Parameters
        ----------
        runnable:
            The object returned by :meth:`build`.
        task:
            The task statement string.

        Yields
        ------
        dict
            Output event dicts with at minimum
            ``{"agent": str, "content": str}``.
        """
        ...

    # ------------------------------------------------------------------
    # 3. Swap: swap(runnable, out_agent, in_agent, handoff) returns success or a not-supported signal
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def swap(
        self,
        runnable: Any,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Replace *out_agent* with *in_agent* in the live runnable.

        ARCHITECTURE.md §5: 'swap(runnable, out_agent, in_agent, handoff) returns success or a not-supported signal'

        Parameters
        ----------
        runnable:
            The live framework-native runnable.
        out_agent:
            Identifier of the agent being removed.
        in_agent:
            Identifier of the replacement agent.
        handoff:
            Optional context-handoff payload.

        Returns
        -------
        ``True`` if the swap succeeded, or ``False`` (NOT_SUPPORTED_SIGNAL) if not supported.
        Must never raise an exception when mid-run swap is not supported.
        """
        ...

    # ------------------------------------------------------------------
    # 4. Capabilities: capabilities() declares whether mid-run swap is available
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Declare what this adapter can actually do.

        ARCHITECTURE.md §5: 'capabilities() declares whether mid-run swap is available'

        Contract tests verify the return matches real behaviour.
        """
        ...

    # ------------------------------------------------------------------
    # 5. Teardown: teardown()
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release any resources held by the adapter.

        ARCHITECTURE.md §5: 'teardown()'
        """
        ...


# Alias for BaseAdapter
BaseAdapter = AdapterBase

__all__ = [
    "NOT_SUPPORTED_SIGNAL",
    "AdapterBase",
    "AdapterCapabilities",
    "BaseAdapter",
]
