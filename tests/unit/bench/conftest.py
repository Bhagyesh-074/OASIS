"""Pytest configuration and markers for benchmark unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for benchmark tests."""
    config.addinivalue_line(
        "markers",
        "real_download: tests that require downloading real datasets from HuggingFace (skipped by default)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip real_download tests by default unless explicitly requested with -m real_download."""
    markexpr = config.getoption("-m", default="")
    if "real_download" not in markexpr:
        skip_download = pytest.mark.skip(
            reason="Real dataset download test skipped by default. Run with '-m real_download' to execute."
        )
        for item in items:
            if "real_download" in item.keywords:
                item.add_marker(skip_download)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return Path to local benchmark sample fixture files."""
    return Path(__file__).resolve().parents[2] / "fixtures" / "bench"
