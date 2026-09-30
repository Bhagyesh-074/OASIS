"""Data types for Task Complexity Estimation (FR-1, FR-2, FR-3, FR-24, NFR-1).

Matches the /v1/estimate response shape defined in docs/API_SPEC.md.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SubScores(BaseModel):
    """Named sub-scores comprising task complexity (FR-2).

    Attributes
    ----------
    subtask_count:
        Number of candidate subtasks identified in the problem statement.
    skill_clusters:
        Number of distinct skill/competency clusters after embedding clustering.
    dep_density:
        Dependency graph edge density in [0.0, 1.0].
    """

    model_config = ConfigDict(frozen=True)

    subtask_count: int = Field(
        ...,
        description="Candidate subtask count (FR-2)",
        ge=0,
    )
    skill_clusters: int = Field(
        ...,
        description="Skill-diversity cluster count (FR-2)",
        ge=0,
    )
    dep_density: float = Field(
        ...,
        description="Dependency-graph density in [0.0, 1.0] (FR-2)",
        ge=0.0,
        le=1.0,
    )


class WeightedContribution(BaseModel):
    """Weighted contribution of each sub-score to composite complexity (FR-2).

    Attributes
    ----------
    subtask_count:
        Contribution of subtask count (weight * normalized score).
    skill_clusters:
        Contribution of skill-diversity clusters (weight * normalized score).
    dep_density:
        Contribution of dependency-graph density (weight * normalized score).
    """

    model_config = ConfigDict(frozen=True)

    subtask_count: float = Field(
        ...,
        description="Weighted contribution from subtask count",
    )
    skill_clusters: float = Field(
        ...,
        description="Weighted contribution from skill-diversity clusters",
    )
    dep_density: float = Field(
        ...,
        description="Weighted contribution from dependency graph density",
    )


class Estimate(BaseModel):
    """Complexity estimate matching POST /v1/estimate in docs/API_SPEC.md (FR-1, FR-2).

    Attributes
    ----------
    mvts:
        Minimum Viable Team Size integer in range 1-8 (FR-1).
    subscores:
        Named sub-scores (subtask_count, skill_clusters, dep_density) (FR-2).
    weights:
        Configured weights dictionary loaded from config/complexity.yaml (FR-2, FR-3).
    contributions:
        Weighted contribution of each sub-score to composite score (FR-2).
    subtasks:
        List of candidate subtask text segments.
    justification:
        Human-readable explanation of MVTS derivation (FR-24).
    latency_ms:
        Supervisory estimation compute latency in milliseconds (NFR-1).
    """

    model_config = ConfigDict(frozen=True)

    mvts: int = Field(
        ...,
        description="Minimum Viable Team Size integer in range 1-8 (FR-1)",
        ge=1,
        le=8,
    )
    subscores: SubScores = Field(
        ...,
        description="Named sub-scores (FR-2)",
    )
    weights: dict[str, float] = Field(
        ...,
        description="Weights applied to sub-scores (FR-2, FR-3)",
    )
    contributions: WeightedContribution = Field(
        ...,
        description="Weighted contribution of each sub-score (FR-2)",
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Extracted candidate subtasks",
    )
    justification: str = Field(
        ...,
        description="Human-readable justification string (FR-24)",
        min_length=1,
    )
    latency_ms: float = Field(
        ...,
        description="Estimation compute latency in ms (NFR-1)",
        ge=0.0,
    )
