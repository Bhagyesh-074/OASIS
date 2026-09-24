"""Shared fixtures for unit/gateway tests."""

from __future__ import annotations

import pytest

from oasis.gateway.fake_llm import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    """An empty FakeLLM instance (no scripted responses)."""
    return FakeLLM()
