"""Contract tests for oasis.rbe.budget — FR-5 four-dimension budget schema.

FR-5 (REQUIREMENTS.md):
"The system shall accept a budget over exactly four dimensions:
 max_tokens, max_cost_usd, max_wall_seconds, max_calls. GPU and RAM
 are not budget dimensions. Verified by API contract test rejecting
 unknown dimensions."

ADR-002 (DECISIONS.md):
"Budget is now exactly tokens, dollars, wall-clock seconds and call count
 — all directly measurable at the gateway...the API rejects unknown
 dimensions rather than accepting them and ignoring them."
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oasis.rbe.budget import Budget

VALID_BUDGET_PAYLOAD = {
    "max_tokens": 200_000,
    "max_cost_usd": 0.75,
    "max_wall_seconds": 300.0,
    "max_calls": 60,
}


class TestBudgetSchemaContract:
    """FR-5 API contract tests for budget schema validation."""

    def test_valid_four_dimension_budget_constructs_successfully(self) -> None:
        """Valid four-dimension budget constructs successfully."""
        budget = Budget(**VALID_BUDGET_PAYLOAD)
        assert budget.max_tokens == 200_000
        assert budget.max_cost_usd == 0.75
        assert budget.max_wall_seconds == 300.0
        assert budget.max_calls == 60

        # Also verify construction via model_validate and from_dict
        validated = Budget.model_validate(VALID_BUDGET_PAYLOAD)
        assert validated == budget

        from_dict_budget = Budget.from_dict(VALID_BUDGET_PAYLOAD)
        assert from_dict_budget == budget

    def test_budget_dict_containing_gpu_is_rejected(self) -> None:
        """FR-5: A budget dict containing 'gpu' is rejected."""
        payload_with_gpu = {**VALID_BUDGET_PAYLOAD, "gpu": 1}

        with pytest.raises((ValueError, ValidationError)):
            Budget(**payload_with_gpu)

        with pytest.raises(ValueError, match="gpu"):
            Budget.validate_dimensions(payload_with_gpu)

        with pytest.raises((ValueError, ValidationError)):
            Budget.from_dict(payload_with_gpu)

    def test_budget_dict_containing_ram_is_rejected(self) -> None:
        """ADR-002: A budget dict containing 'ram' is rejected."""
        payload_with_ram = {**VALID_BUDGET_PAYLOAD, "ram": "16GB"}

        with pytest.raises((ValueError, ValidationError)):
            Budget(**payload_with_ram)

        with pytest.raises(ValueError, match="ram"):
            Budget.validate_dimensions(payload_with_ram)

        with pytest.raises((ValueError, ValidationError)):
            Budget.from_dict(payload_with_ram)

    def test_budget_dict_containing_gpu_and_ram_is_rejected(self) -> None:
        """FR-5 / ADR-002: A budget dict containing both 'gpu' and 'ram' is rejected."""
        payload_with_both = {**VALID_BUDGET_PAYLOAD, "gpu": 2, "ram": "32GB"}

        with pytest.raises((ValueError, ValidationError)):
            Budget(**payload_with_both)

        with pytest.raises(ValueError):
            Budget.validate_dimensions(payload_with_both)

    @pytest.mark.parametrize(
        "missing_key",
        ["max_tokens", "max_cost_usd", "max_wall_seconds", "max_calls"],
    )
    def test_missing_one_of_the_four_required_dimensions_is_rejected(
        self, missing_key: str
    ) -> None:
        """Missing one of the four required dimensions is rejected."""
        incomplete_payload = {k: v for k, v in VALID_BUDGET_PAYLOAD.items() if k != missing_key}

        with pytest.raises((ValueError, ValidationError)):
            Budget(**incomplete_payload)

        with pytest.raises(ValueError, match=missing_key):
            Budget.validate_dimensions(incomplete_payload)

        with pytest.raises((ValueError, ValidationError)):
            Budget.from_dict(incomplete_payload)
