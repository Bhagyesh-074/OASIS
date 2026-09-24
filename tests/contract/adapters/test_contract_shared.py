"""Shared reusable conformance test suite for OASIS framework adapters.

ARCHITECTURE.md §5:
  "Every adapter implements the same five operations: build(config) returns a
   runnable, run(runnable, task) yields output events, swap(runnable, out_agent,
   in_agent, handoff) returns success or a not-supported signal, capabilities()
   declares whether mid-run swap is available, and teardown()."

TESTING.md:
  "Every adapter is run against a shared conformance suite asserting the
   five-operation contract and, critically, that capabilities() matches actual
   behaviour — an adapter claiming mid-run swap must demonstrate it, and the
   CrewAI stub must claim and return not-supported (FR-19)."

This module provides ``BaseAdapterConformanceSuite``, a reusable base test
class designed to be subclassed or parameterized per adapter. Prompts 11 and 12
plug their respective adapters (LangGraph and AutoGen) into this suite.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from oasis.adapters.base import (
    NOT_SUPPORTED_SIGNAL,
    AdapterBase,
    AdapterCapabilities,
    BaseAdapter,
)

# The required five operations per ARCHITECTURE.md §5
REQUIRED_FIVE_OPERATIONS: tuple[str, ...] = (
    "build",
    "run",
    "swap",
    "capabilities",
    "teardown",
)


class BaseAdapterConformanceSuite:
    """Reusable conformance suite for testing any OASIS framework adapter.

    Subclasses must define an ``adapter`` fixture returning an instance of the
    adapter under test. Pytest ignores this base class directly due to
    ``__test__ = False``.
    """

    __test__ = False

    def test_implements_adapter_base(self, adapter: AdapterBase) -> None:
        """Every adapter must be an instance of AdapterBase (and BaseAdapter alias)."""
        assert isinstance(adapter, AdapterBase), (
            f"{adapter.__class__.__name__} must be an instance of AdapterBase"
        )
        assert isinstance(adapter, BaseAdapter), (
            f"{adapter.__class__.__name__} must be an instance of BaseAdapter"
        )

    def test_has_all_five_operations(self, adapter: AdapterBase) -> None:
        """Every adapter must implement the exact five operations (ARCHITECTURE.md §5):
        build, run, swap, capabilities, teardown.
        """
        for op_name in REQUIRED_FIVE_OPERATIONS:
            assert hasattr(adapter, op_name), (
                f"{adapter.__class__.__name__} is missing required operation: {op_name}"
            )
            attr = getattr(adapter, op_name)
            assert callable(attr), (
                f"{adapter.__class__.__name__}.{op_name} must be callable"
            )

    def test_operation_signatures(self, adapter: AdapterBase) -> None:
        """Verify the five operations accept the expected parameters."""
        # 1. build(config)
        build_sig = inspect.signature(adapter.build)
        assert "config" in build_sig.parameters, "build() must accept 'config'"

        # 2. run(runnable, task)
        run_sig = inspect.signature(adapter.run)
        assert "runnable" in run_sig.parameters, "run() must accept 'runnable'"
        assert "task" in run_sig.parameters, "run() must accept 'task'"

        # 3. swap(runnable, out_agent, in_agent, handoff)
        swap_sig = inspect.signature(adapter.swap)
        for param in ("runnable", "out_agent", "in_agent"):
            assert param in swap_sig.parameters, f"swap() must accept '{param}'"

        # 4. capabilities()
        cap_sig = inspect.signature(adapter.capabilities)
        # Should take only self (or no extra required args)
        required_cap_params = [
            p for p in cap_sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required_cap_params) == 0, "capabilities() must not require arguments"

        # 5. teardown()
        teardown_sig = inspect.signature(adapter.teardown)
        required_teardown_params = [
            p for p in teardown_sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required_teardown_params) == 0, "teardown() must not require arguments"

    def test_capabilities_returns_adapter_capabilities(self, adapter: AdapterBase) -> None:
        """capabilities() returns an AdapterCapabilities instance."""
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities), (
            f"Expected AdapterCapabilities, got {type(caps)}"
        )

    def test_capabilities_fields_are_booleans(self, adapter: AdapterBase) -> None:
        """capabilities() declares boolean flags for mid-run and boundary swaps."""
        caps = adapter.capabilities()
        assert isinstance(caps.mid_run_swap, bool), "mid_run_swap must be a bool"
        assert isinstance(caps.boundary_swap, bool), "boundary_swap must be a bool"
        assert isinstance(caps.supports_mid_run_swap, bool), "supports_mid_run_swap must be a bool"
        assert isinstance(caps.supports_boundary_swap, bool), "supports_boundary_swap must be a bool"
        assert caps.mid_run_swap == caps.supports_mid_run_swap
        assert caps.boundary_swap == caps.supports_boundary_swap

    def test_capabilities_truthfulness_when_mid_run_unsupported(
        self, adapter: AdapterBase
    ) -> None:
        """Contract enforcement (TESTING.md, FR-19):
        When an adapter declares mid_run_swap=False, swap() MUST return not-supported (False),
        must NEVER return True (pretend to succeed), and must NEVER raise an exception.
        """
        caps = adapter.capabilities()
        if not caps.mid_run_swap:
            # Calling swap should cleanly return False / NOT_SUPPORTED_SIGNAL without raising
            result = adapter.swap(
                runnable=None,
                out_agent="agent_out",
                in_agent="agent_in",
                handoff={"context": "handoff_data"},
            )
            assert result is False, (
                f"{adapter.__class__.__name__} declares mid_run_swap=False but swap() returned "
                f"{result!r} instead of False (not-supported signal)."
            )
            assert result is NOT_SUPPORTED_SIGNAL


# ======================================================================
# Standalone Contract Tests for AdapterBase Abstract Class
# ======================================================================


def test_cannot_instantiate_adapter_base_directly() -> None:
    """AdapterBase and BaseAdapter are abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AdapterBase()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        BaseAdapter()  # type: ignore[abstract]


def test_cannot_instantiate_incomplete_adapter_subclass() -> None:
    """A subclass omitting any of the five operations cannot be instantiated."""
    class IncompleteAdapter(AdapterBase):
        def build(self, config: dict[str, Any]) -> Any:
            return None
        # missing: run, swap, capabilities, teardown

    with pytest.raises(TypeError):
        IncompleteAdapter()  # type: ignore[abstract]


def test_adapter_capabilities_defaults() -> None:
    """AdapterCapabilities defaults to False for both swap flags."""
    caps = AdapterCapabilities()
    assert caps.mid_run_swap is False
    assert caps.boundary_swap is False
    assert caps.supports_mid_run_swap is False
    assert caps.supports_boundary_swap is False


def test_not_supported_signal_constant() -> None:
    """NOT_SUPPORTED_SIGNAL is False."""
    assert NOT_SUPPORTED_SIGNAL is False
