"""Unit tests for oasis.rbe.budget — four-dimension budget schema (FR-5)."""

from __future__ import annotations

import pytest

from oasis.rbe.budget import Budget, BudgetDimension, BudgetState


class TestBudgetDimension:
    """BudgetDimension enum coverage."""

    def test_exactly_four_dimensions(self) -> None:
        """FR-5 / ADR-002: exactly max_tokens, max_cost_usd, max_wall_seconds, max_calls."""
        assert len(BudgetDimension) == 4
        expected = {"max_tokens", "max_cost_usd", "max_wall_seconds", "max_calls"}
        assert {d.value for d in BudgetDimension} == expected


class TestBudgetValidation:
    """Budget.validate_dimensions — FR-5 contract test entry-point."""

    def test_rejects_gpu_dimension(self) -> None:
        """FR-5: gpu is not a valid budget dimension."""
        with pytest.raises(ValueError, match="gpu"):
            Budget.validate_dimensions({
                "max_tokens": 10_000,
                "max_cost_usd": 1.00,
                "max_wall_seconds": 60.0,
                "max_calls": 50,
                "gpu": 1,
            })

    def test_rejects_ram_dimension(self) -> None:
        """ADR-002: ram is not a valid budget dimension."""
        with pytest.raises(ValueError, match="ram"):
            Budget.validate_dimensions({
                "max_tokens": 10_000,
                "max_cost_usd": 1.00,
                "max_wall_seconds": 60.0,
                "max_calls": 50,
                "ram": "16GB",
            })

    def test_accepts_valid_dimensions(self, sample_budget_kwargs: dict) -> None:
        """All four valid dimensions pass without error."""
        Budget.validate_dimensions(sample_budget_kwargs)
        budget = Budget(**sample_budget_kwargs)
        assert budget.max_tokens == sample_budget_kwargs["max_tokens"]
        assert budget.max_cost_usd == sample_budget_kwargs["max_cost_usd"]
        assert budget.max_wall_seconds == sample_budget_kwargs["max_wall_seconds"]
        assert budget.max_calls == sample_budget_kwargs["max_calls"]


class TestBudgetState:
    """BudgetState — mutable consumption tracker."""

    def test_remaining_decreases_after_record(self, sample_budget_kwargs: dict) -> None:
        """Recording consumption reduces remaining capacity."""
        budget = Budget(**sample_budget_kwargs)
        state = BudgetState(budget)

        assert state.remaining(BudgetDimension.MAX_TOKENS) == 10_000
        assert state.remaining(BudgetDimension.MAX_COST_USD) == 1.00
        assert state.remaining(BudgetDimension.MAX_WALL_SECONDS) == 60.0
        assert state.remaining(BudgetDimension.MAX_CALLS) == 50

        state.record(tokens=2_000, cost_usd=0.20, wall_seconds=5.0, calls=1)

        assert state.remaining(BudgetDimension.MAX_TOKENS) == 8_000
        assert state.remaining(BudgetDimension.MAX_COST_USD) == pytest.approx(0.80)
        assert state.remaining(BudgetDimension.MAX_WALL_SECONDS) == pytest.approx(55.0)
        assert state.remaining(BudgetDimension.MAX_CALLS) == 49

    def test_is_exhausted_when_any_dimension_reaches_zero(
        self, sample_budget_kwargs: dict
    ) -> None:
        """Exhaustion on any single dimension triggers True."""
        budget = Budget(**sample_budget_kwargs)
        state = BudgetState(budget)
        assert not state.is_exhausted()

        # Exhaust tokens
        state.record(tokens=10_000)
        assert state.is_exhausted()

