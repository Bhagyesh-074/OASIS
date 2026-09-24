"""Shared fixtures for unit/rbe tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_budget_kwargs() -> dict:
    """Keyword arguments for a typical four-dimension budget."""
    return {
        "max_tokens": 10_000,
        "max_cost_usd": 1.00,
        "max_wall_seconds": 60.0,
        "max_calls": 50,
    }
