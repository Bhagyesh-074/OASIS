"""Benchmark configuration loading and validation (FR-3, NFR-4).

Loads and validates config/benchmark.yaml using Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkConfig(BaseModel):
    """Configuration schema for benchmark ingestion and split assignment (FR-3).

    Attributes
    ----------
    seed:
        RNG seed for deterministic dataset sampling and split assignment.
    calibration_fraction:
        Fraction of tasks in each domain reserved for the calibration split.
    domain_counts:
        Target number of tasks to ingest per domain.
    source_allocation:
        Per-source task counts within each domain.
    datasets:
        HuggingFace dataset configuration parameters.
    output_path:
        Default destination path for ingested task JSONL records.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = Field(default=42, description="Seeded RNG base value")
    calibration_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of tasks assigned to held-out calibration split (FR-3)",
    )
    domain_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "code_generation": 20,
            "research_qa": 20,
            "quant_analysis": 20,
        },
        description="Task counts per domain",
    )
    source_allocation: dict[str, dict[str, int]] = Field(
        default_factory=lambda: {
            "code_generation": {"mbpp": 10, "humaneval": 10},
            "research_qa": {"hotpotqa": 20},
            "quant_analysis": {"gsm8k": 20},
        },
        description="Task counts per source within domain",
    )
    datasets: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="HuggingFace dataset parameters",
    )
    output_path: str = Field(
        default="data/tasks.jsonl",
        description="Output JSONL destination",
    )


def load_benchmark_config(config_path: Path | str | None = None) -> BenchmarkConfig:
    """Load and validate BenchmarkConfig from YAML file (FR-3, NFR-4).

    Parameters
    ----------
    config_path:
        Optional path to YAML config. Defaults to config/benchmark.yaml at repo root.

    Returns
    -------
    BenchmarkConfig
        Validated configuration instance.
    """
    if config_path is None:
        # Locate config/benchmark.yaml relative to repository root
        repo_root = Path(__file__).resolve().parents[3]
        target = repo_root / "config" / "benchmark.yaml"
    else:
        target = Path(config_path)

    if not target.exists():
        # Fall back to default configuration
        return BenchmarkConfig()

    raw_bytes = target.read_bytes()
    # Normalize CRLF to LF per NFR-4
    text = raw_bytes.replace(b"\r\n", b"\n").decode("utf-8")
    data = yaml.safe_load(text) or {}
    return BenchmarkConfig.model_validate(data)
