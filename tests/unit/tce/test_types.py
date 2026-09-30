"""Unit tests for TCE data types and schema validation (FR-1, FR-2, API_SPEC)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oasis.tce.types import Estimate, SubScores, WeightedContribution


def test_subscores_creation_and_bounds() -> None:
    """FR-2: SubScores validates bounds (subtask_count >= 0, dep_density in [0, 1])."""
    subscores = SubScores(subtask_count=4.0, skill_clusters=3.0, dep_density=0.42)
    assert subscores.subtask_count == 4.0
    assert subscores.skill_clusters == 3.0
    assert subscores.dep_density == 0.42

    # dep_density > 1 rejected
    with pytest.raises(ValidationError):
        SubScores(subtask_count=4.0, skill_clusters=3.0, dep_density=1.5)

    # subtask_count < 0 rejected
    with pytest.raises(ValidationError):
        SubScores(subtask_count=-1.0, skill_clusters=3.0, dep_density=0.5)


def test_weighted_contribution_creation() -> None:
    """FR-2: WeightedContribution stores per-subscore weighted contributions."""
    wc = WeightedContribution(
        subtask_count=0.16,
        skill_clusters=0.16,
        dep_density=0.084,
    )
    assert wc.subtask_count == 0.16
    assert wc.skill_clusters == 0.16
    assert wc.dep_density == 0.084


def test_estimate_schema_matches_api_spec() -> None:
    """FR-1, FR-2: Estimate matches POST /v1/estimate schema from API_SPEC.md."""
    subscores = SubScores(subtask_count=4.0, skill_clusters=3.0, dep_density=0.42)
    weights = {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2}
    contributions = WeightedContribution(
        subtask_count=0.16,
        skill_clusters=0.16,
        dep_density=0.084,
    )
    subtasks = ["retrieve papers", "extract claims", "compare", "format table"]

    estimate = Estimate(
        mvts=3,
        subscores=subscores,
        weights=weights,
        contributions=contributions,
        subtasks=subtasks,
        justification="MVTS=3 derived from 4 subtasks across 3 skill clusters with dep_density 0.42",
        latency_ms=63.0,
    )

    data = estimate.model_dump()
    assert data["mvts"] == 3
    assert data["subscores"]["subtask_count"] == 4.0
    assert data["subscores"]["skill_clusters"] == 3.0
    assert data["subscores"]["dep_density"] == 0.42
    assert data["weights"] == weights
    assert data["contributions"]["subtask_count"] == 0.16
    assert data["subtasks"] == subtasks
    assert "MVTS=3" in data["justification"]
    assert data["latency_ms"] == 63.0


def test_estimate_mvts_bounds() -> None:
    """FR-1: Estimate enforces mvts in range [1, 8]."""
    subscores = SubScores(subtask_count=1.0, skill_clusters=1.0, dep_density=0.1)
    weights = {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2}
    contributions = WeightedContribution(
        subtask_count=0.4,
        skill_clusters=0.4,
        dep_density=0.2,
    )

    # mvts < 1 rejected
    with pytest.raises(ValidationError):
        Estimate(
            mvts=0,
            subscores=subscores,
            weights=weights,
            contributions=contributions,
            subtasks=["step 1"],
            justification="invalid",
            latency_ms=10.0,
        )

    # mvts > 8 rejected
    with pytest.raises(ValidationError):
        Estimate(
            mvts=9,
            subscores=subscores,
            weights=weights,
            contributions=contributions,
            subtasks=["step 1"],
            justification="invalid",
            latency_ms=10.0,
        )


def test_estimate_empty_justification_rejected() -> None:
    """FR-24: Estimate requires non-empty justification string."""
    subscores = SubScores(subtask_count=1.0, skill_clusters=1.0, dep_density=0.1)
    weights = {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2}
    contributions = WeightedContribution(
        subtask_count=0.4,
        skill_clusters=0.4,
        dep_density=0.2,
    )

    with pytest.raises(ValidationError):
        Estimate(
            mvts=1,
            subscores=subscores,
            weights=weights,
            contributions=contributions,
            subtasks=["step 1"],
            justification="",  # empty!
            latency_ms=10.0,
        )
