"""Data models for benchmark tasks (FR-3, FR-13, docs/DATABASE.md).

Matches the `task` SQLite table schema in docs/DATABASE.md:
  task_id           TEXT PRIMARY KEY
  domain            TEXT NOT NULL CHECK (domain IN (...))
  source            TEXT NOT NULL
  source_ref        TEXT
  statement         TEXT NOT NULL
  reference_answer  TEXT
  verifier_type     TEXT
  verifier_spec     TEXT
  complexity_label  TEXT
  split             TEXT NOT NULL CHECK (split IN ('calibration','eval'))
  created_at        TEXT NOT NULL
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskDomain = Literal[
    "code_generation",
    "research_qa",
    "quant_analysis",
    "support_triage",
    "content_generation",
]

TaskSource = Literal[
    "mbpp",
    "humaneval",
    "hotpotqa",
    "gsm8k",
    "authored",
]

TaskVerifierType = Literal[
    "pytest",
    "citation_resolve",
    "numeric_consistency",
]

TaskComplexityLabel = Literal[
    "low",
    "medium",
    "high",
]

TaskSplit = Literal[
    "calibration",
    "eval",
]


class TaskRecord(BaseModel):
    """Pydantic model representing a benchmark task row (docs/DATABASE.md).

    Attributes
    ----------
    task_id:
        Unique task identifier (e.g. 'mbpp_0001', 'gsm8k_0001').
    domain:
        Benchmark task domain.
    source:
        Origin dataset ('mbpp', 'humaneval', 'hotpotqa', 'gsm8k', 'authored').
    source_ref:
        Identifier or key within the source dataset.
    statement:
        The task prompt or problem statement given to the agent team.
    reference_answer:
        Gold answer or canonical reference solution.
    verifier_type:
        L3 verifier kind ('pytest', 'citation_resolve', 'numeric_consistency').
    verifier_spec:
        JSON string containing test code, expected values, tolerances (FR-13).
    complexity_label:
        Optional coarse complexity categorization ('low', 'medium', 'high').
    split:
        Dataset partition ('calibration' or 'eval') (FR-3).
    created_at:
        ISO 8601 creation timestamp string.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique task primary key")
    domain: TaskDomain = Field(..., description="Task evaluation domain")
    source: str = Field(..., description="Source dataset or 'authored'")
    source_ref: str | None = Field(default=None, description="Reference ID in source dataset")
    statement: str = Field(..., min_length=1, description="Problem statement text")
    reference_answer: str | None = Field(default=None, description="Gold/reference answer")
    verifier_type: TaskVerifierType | None = Field(
        default=None, description="L3 verifier kind (FR-13)"
    )
    verifier_spec: str | None = Field(
        default=None, description="JSON string with verifier execution parameters"
    )
    complexity_label: TaskComplexityLabel | None = Field(
        default=None, description="Coarse complexity label ('low', 'medium', 'high')"
    )
    split: TaskSplit = Field(..., description="Held-out partition ('calibration' or 'eval') (FR-3)")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")

    def parsed_verifier_spec(self) -> dict[str, Any]:
        """Return the verifier_spec deserialized from JSON."""
        if not self.verifier_spec:
            return {}
        try:
            parsed = json.loads(self.verifier_spec)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except (json.JSONDecodeError, TypeError):
            return {}
