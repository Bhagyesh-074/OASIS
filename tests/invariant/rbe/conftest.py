"""Shared fixtures for invariant/rbe tests."""

from __future__ import annotations

import pytest

from oasis.gateway.fake_llm import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    """An empty FakeLLM for invariant tests."""
    return FakeLLM()
