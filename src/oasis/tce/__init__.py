"""Task Complexity Estimator (TCE) package.

Owns team sizing estimation (heuristic and LLM-planner), subtask segmentation,
skill clustering, and dependency graph density calculation (FR-1, FR-2, FR-3, FR-4).
"""

from oasis.tce import llm_planner
from oasis.tce._log import log_tce_decision
from oasis.tce.config import (
    ComplexityConfig,
    NormalizationConfig,
    WeightsConfig,
    compute_complexity_config_hash,
    load_complexity_config,
)
from oasis.tce.heuristic import estimate, similarity
from oasis.tce.llm_planner import LLMPlanner
from oasis.tce.llm_planner import estimate as estimate_llm
from oasis.tce.types import Estimate, SubScores, WeightedContribution

__all__ = [
    "ComplexityConfig",
    "Estimate",
    "LLMPlanner",
    "NormalizationConfig",
    "SubScores",
    "WeightedContribution",
    "WeightsConfig",
    "compute_complexity_config_hash",
    "estimate",
    "estimate_llm",
    "llm_planner",
    "load_complexity_config",
    "log_tce_decision",
    "similarity",
]
