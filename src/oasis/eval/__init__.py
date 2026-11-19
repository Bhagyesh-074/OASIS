"""OASIS Evaluation, statistical analysis, and benchmark runner package."""

from oasis.eval.runner import BenchmarkRunner, load_budget_profile
from oasis.eval.stats import (
    bootstrap_ci,
    compute_metrics_summary,
    compute_recall_by_type_and_severity,
    holm_bonferroni,
    rank_biserial_effect_size,
    wilcoxon_paired_test,
)

__all__ = [
    "BenchmarkRunner",
    "bootstrap_ci",
    "compute_metrics_summary",
    "compute_recall_by_type_and_severity",
    "holm_bonferroni",
    "load_budget_profile",
    "rank_biserial_effect_size",
    "wilcoxon_paired_test",
]
