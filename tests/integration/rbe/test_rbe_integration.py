"""Integration tests for oasis.rbe — budget enforcement pipeline with FakeLLM."""

from __future__ import annotations


class TestRBEIntegration:
    """Full budget lifecycle: configure → admit calls → halt."""

    def test_tiny_budget_produces_halted_status(self, fake_llm) -> None:
        """FR-8: deliberately tiny budget → halted_budget, best-so-far returned."""
        # TODO: implement

    def test_replacement_denied_when_handoff_exceeds_budget(self, fake_llm) -> None:
        """FR-17 / ADR-007: handoff cost > remaining → replacement_denied_budget."""
        # TODO: implement
