"""Configuration loader and schema for Task Complexity Estimator (FR-1, FR-3, NFR-4, NFR-5).

Loads weights, MVTS range, score-to-MVTS thresholds, and SBERT model settings
from config/complexity.yaml. Exposes the config file SHA-256 digest for run manifest
reproducibility (NFR-4).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WeightsConfig(BaseModel):
    """Sub-score weights for composite complexity calculation (FR-2, FR-3).

    Attributes
    ----------
    subtask_count:
        Weight for candidate subtask count.
    skill_clusters:
        Weight for skill-diversity clusters.
    dep_density:
        Weight for dependency graph edge density.
    """

    model_config = ConfigDict(frozen=True)

    subtask_count: float = Field(
        ...,
        ge=0.0,
        description="Weight for subtask count (FR-2, FR-3)",
    )
    skill_clusters: float = Field(
        ...,
        ge=0.0,
        description="Weight for skill-diversity clusters (FR-2, FR-3)",
    )
    dep_density: float = Field(
        ...,
        ge=0.0,
        description="Weight for dependency graph density (FR-2, FR-3)",
    )

    @model_validator(mode="after")
    def validate_sum_to_one(self) -> WeightsConfig:
        """Validate that weights are non-negative and sum to 1.0 (FR-3)."""
        total = self.subtask_count + self.skill_clusters + self.dep_density
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0; got {total:.6f}")
        return self

    def to_dict(self) -> dict[str, float]:
        """Return weights as a primitive dictionary (FR-2)."""
        return {
            "subtask_count": self.subtask_count,
            "skill_clusters": self.skill_clusters,
            "dep_density": self.dep_density,
        }


class ComplexityConfig(BaseModel):
    """Frozen configuration for TCE complexity calculation (FR-1, FR-3, NFR-4, NFR-5).

    Attributes
    ----------
    version:
        Configuration version string.
    weights:
        Validated sub-score weights.
    mvts_range:
        Permissible Minimum Viable Team Size range [min, max], within 1-8 (FR-1).
    thresholds:
        Monotonic score thresholds mapping composite complexity score to integer MVTS.
        Keyed by integer team size.
    sbert_model:
        Pinned SBERT model identifier (NFR-5: no aliases such as 'latest').
    cluster_distance_threshold:
        Distance threshold for grouping subtasks into skill clusters.
    sha256:
        Hex-encoded SHA-256 hash of the configuration file (NFR-4).
    """

    model_config = ConfigDict(frozen=True)

    version: str = Field(
        ...,
        min_length=1,
        description="Configuration version tag",
    )
    weights: WeightsConfig = Field(
        ...,
        description="Sub-score weights (FR-3)",
    )
    mvts_range: tuple[int, int] = Field(
        ...,
        description="Permissible [min, max] MVTS range within 1-8 (FR-1)",
    )
    thresholds: dict[int, float] = Field(
        ...,
        description="Monotonic score thresholds for each team size in mvts_range (FR-3)",
    )
    sbert_model: str = Field(
        ...,
        min_length=1,
        description="Pinned SBERT model name (NFR-5)",
    )
    cluster_distance_threshold: float = Field(
        ...,
        gt=0.0,
        description="Cluster distance threshold for skill grouping",
    )
    sha256: str = Field(
        default="",
        description="SHA-256 digest of config file for run manifest (NFR-4)",
    )

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, v: Any) -> WeightsConfig:
        """Coerce raw dictionary to validated WeightsConfig (FR-3)."""
        if isinstance(v, WeightsConfig):
            return v
        if isinstance(v, dict):
            return WeightsConfig(**v)
        raise ValueError(f"weights must be WeightsConfig or dict; got {type(v).__name__}")

    @field_validator("mvts_range", mode="before")
    @classmethod
    def _coerce_mvts_range(cls, v: Any) -> tuple[int, int]:
        """Ensure mvts_range is a 2-tuple of integers within [1, 8] (FR-1)."""
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"mvts_range must have exactly 2 elements [min, max]; got {v}")
            min_v, max_v = int(v[0]), int(v[1])
            if not (1 <= min_v <= max_v <= 8):
                raise ValueError(
                    f"mvts_range must satisfy 1 <= min <= max <= 8; got [{min_v}, {max_v}]"
                )
            return (min_v, max_v)
        raise ValueError(f"mvts_range must be a list or tuple; got {type(v).__name__}")

    @field_validator("sbert_model")
    @classmethod
    def _validate_sbert_model_pinned(cls, v: str) -> str:
        """Enforce pinned model identifier without 'latest' alias (NFR-5)."""
        if "latest" in v.lower():
            raise ValueError(f"NFR-5 violation: sbert_model cannot use 'latest' alias; got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_thresholds_monotonic(self) -> ComplexityConfig:
        """Validate that thresholds cover mvts_range and are strictly monotonic (FR-1, FR-3)."""
        min_mvts, max_mvts = self.mvts_range
        expected_keys = list(range(min_mvts, max_mvts + 1))
        actual_keys = sorted(self.thresholds.keys())

        if actual_keys != expected_keys:
            raise ValueError(
                f"Thresholds keys must cover all team sizes in mvts_range {self.mvts_range}; "
                f"expected {expected_keys}, got {actual_keys}"
            )

        for i in range(len(actual_keys) - 1):
            k1, k2 = actual_keys[i], actual_keys[i + 1]
            t1, t2 = self.thresholds[k1], self.thresholds[k2]
            if t1 >= t2:
                raise ValueError(
                    f"Thresholds must be strictly monotonic increasing; "
                    f"team size {k1} threshold ({t1}) >= team size {k2} threshold ({t2})"
                )

        return self

    def score_to_mvts(self, score: float) -> int:
        """Map a continuous composite complexity score to an integer MVTS in mvts_range (FR-1).

        Parameters
        ----------
        score:
            Continuous composite complexity score.

        Returns
        -------
        int
            Integer Minimum Viable Team Size in range [min, max] (FR-1).
        """
        min_mvts, max_mvts = self.mvts_range
        chosen = min_mvts
        for k in sorted(self.thresholds.keys()):
            if score >= self.thresholds[k]:
                chosen = k
        return max(min_mvts, min(max_mvts, chosen))


def _resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve the path to complexity.yaml."""
    if path is not None:
        return Path(path).resolve()
    cwd_path = Path("config/complexity.yaml").resolve()
    if cwd_path.is_file():
        return cwd_path
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "config" / "complexity.yaml").resolve()


def load_complexity_config(path: str | Path | None = None) -> ComplexityConfig:
    """Load and validate the complexity configuration from YAML (FR-3, NFR-4).

    Parameters
    ----------
    path:
        Optional file path to complexity.yaml. If omitted, resolves config/complexity.yaml
        relative to the repository root.

    Returns
    -------
    ComplexityConfig
        Validated, frozen configuration object with sha256 digest populated.

    Raises
    ------
    FileNotFoundError
        If the configuration file cannot be found.
    ValueError
        If the configuration file contains invalid YAML, negative weights,
        non-monotonic thresholds, or violates NFR-5 / FR-1 / FR-3.
    """
    config_path = _resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"TCE complexity config not found at: {config_path}")

    raw_bytes = config_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        data = yaml.safe_load(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML from {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected top-level mapping in {config_path}, got {type(data).__name__}"
        )

    # Ensure required top-level keys are present for clear error messages
    required_keys = {"weights", "mvts_range", "thresholds", "sbert_model", "cluster_distance_threshold"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing required configuration key(s) in {config_path}: {sorted(missing)}")

    return ComplexityConfig(
        version=data.get("version", "0.1.0-uncalibrated"),
        weights=data["weights"],
        mvts_range=data["mvts_range"],
        thresholds=data["thresholds"],
        sbert_model=data["sbert_model"],
        cluster_distance_threshold=float(data["cluster_distance_threshold"]),
        sha256=file_sha256,
    )


def compute_complexity_config_hash(path: str | Path | None = None) -> str:
    """Compute the SHA-256 digest of the complexity configuration file (NFR-4).

    Parameters
    ----------
    path:
        Optional file path to complexity.yaml. Defaults to repository config/complexity.yaml.

    Returns
    -------
    str
        Hex-encoded 64-character SHA-256 digest string for the run manifest.
    """
    config_path = _resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"TCE complexity config not found at: {config_path}")
    return hashlib.sha256(config_path.read_bytes()).hexdigest()
