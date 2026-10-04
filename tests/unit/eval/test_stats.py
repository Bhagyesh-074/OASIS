"""Unit tests for statistics module: bootstrap CI, Wilcoxon, effect sizes, and Holm correction (FR-29)."""

from __future__ import annotations

import numpy as np
import pytest

from oasis.eval.stats import (
    bootstrap_ci,
    compute_metrics_summary,
    compute_recall_by_type_and_severity,
    holm_bonferroni,
    rank_biserial_effect_size,
    wilcoxon_paired_test,
)


class TestBootstrapCI:
    """Validate bootstrap confidence interval calculation."""

    def test_constant_array_returns_constant(self) -> None:
        """Bootstrap on invariant array returns the exact value."""
        data = [5.0] * 20
        pt, low, high = bootstrap_ci(data, n_resamples=500)
        assert pt == 5.0
        assert low == 5.0
        assert high == 5.0

    def test_empty_and_single_element(self) -> None:
        """Handle edge cases of empty and single-element inputs."""
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
        assert bootstrap_ci([42.0]) == (42.0, 42.0, 42.0)

    def test_symmetric_interval(self) -> None:
        """Point estimate lies within the 95% bootstrap confidence interval."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        pt, low, high = bootstrap_ci(data, stat_fn=np.median, confidence=0.95, n_resamples=1000, seed=123)
        assert pt == 5.0
        assert low <= pt <= high
        assert low >= 1.0
        assert high <= 9.0


class TestWilcoxonPaired:
    """Validate paired Wilcoxon signed-rank testing."""

    def test_identical_samples_returns_p_value_one(self) -> None:
        """Zero difference returns null p-value of 1.0."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = wilcoxon_paired_test(a, b)
        assert p == 1.0

    def test_strictly_dominant_sample_is_significant(self) -> None:
        """Large consistent positive differences yield p < 0.05."""
        a = [10.0, 12.0, 15.0, 14.0, 18.0, 20.0, 22.0, 25.0]
        b = [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0]
        _, p = wilcoxon_paired_test(a, b)
        assert p < 0.05

    def test_length_mismatch_raises_error(self) -> None:
        """Sample length mismatch raises ValueError."""
        with pytest.raises(ValueError):
            wilcoxon_paired_test([1.0, 2.0], [1.0])


class TestRankBiserialEffectSize:
    """Validate rank-biserial effect size correlation."""

    def test_maximal_positive_effect(self) -> None:
        """When all differences are strictly positive, r = +1.0."""
        a = [5.0, 6.0, 7.0, 8.0]
        b = [1.0, 2.0, 3.0, 4.0]
        r = rank_biserial_effect_size(a, b)
        assert r == 1.0

    def test_maximal_negative_effect(self) -> None:
        """When all differences are strictly negative, r = -1.0."""
        a = [1.0, 2.0, 3.0, 4.0]
        b = [5.0, 6.0, 7.0, 8.0]
        r = rank_biserial_effect_size(a, b)
        assert r == -1.0

    def test_zero_effect_on_identical(self) -> None:
        """Zero differences yield r = 0.0."""
        a = [3.0, 3.0, 3.0]
        b = [3.0, 3.0, 3.0]
        r = rank_biserial_effect_size(a, b)
        assert r == 0.0


class TestHolmBonferroni:
    """Validate family-wise error rate correction."""

    def test_known_analytical_case(self) -> None:
        """Test with analytically hand-computed Holm adjustments.

        Raw: p1=0.01, p2=0.04, p3=0.03
        Sorted:
          0.01 * 3 = 0.03
          0.03 * 2 = 0.06
          0.04 * 1 = 0.04 -> max(0.06, 0.04) = 0.06
        Returned in original order: [0.03, 0.06, 0.06]
        """
        raw = [0.01, 0.04, 0.03]
        adj = holm_bonferroni(raw)
        assert pytest.approx(adj[0], rel=1e-5) == 0.03
        assert pytest.approx(adj[1], rel=1e-5) == 0.06
        assert pytest.approx(adj[2], rel=1e-5) == 0.06

    def test_caps_at_one(self) -> None:
        """Adjusted p-values cannot exceed 1.0."""
        raw = [0.8, 0.9]
        adj = holm_bonferroni(raw)
        assert all(p <= 1.0 for p in adj)


class TestMetricsSummaryAndRecall:
    """Validate aggregate summary generation."""

    def test_compute_recall_by_type_and_severity(self) -> None:
        """Recall disaggregated by (failure_type, severity)."""
        records = [
            {"failure_type": "topical_drift", "severity": "subtle", "detected": 1},
            {"failure_type": "topical_drift", "severity": "subtle", "detected": 0},
            {"failure_type": "numeric_perturb", "severity": "gross", "detected": 1},
        ]
        recall = compute_recall_by_type_and_severity(records)
        assert recall["topical_drift"]["subtle"] == 0.5
        assert recall["numeric_perturb"]["gross"] == 1.0

    def test_compute_metrics_summary(self) -> None:
        """Full metrics table generation with cross-arm comparison."""
        runs = [
            # Arm: vanilla_fixed
            {
                "arm": "vanilla_fixed",
                "task_id": "t1",
                "seed": 1,
                "quality_composite": 0.75,
                "total_cost_usd": 0.40,
                "total_tokens": 10000,
                "wall_ms": 1200,
                "supervision_tokens": 1000,
                "compliant_all": 1,
                "replacements": 0,
                "replacements_denied": 0,
            },
            {
                "arm": "vanilla_fixed",
                "task_id": "t2",
                "seed": 1,
                "quality_composite": 0.80,
                "total_cost_usd": 0.50,
                "total_tokens": 12000,
                "wall_ms": 1500,
                "supervision_tokens": 1200,
                "compliant_all": 1,
                "replacements": 0,
                "replacements_denied": 0,
            },
            # Arm: full
            {
                "arm": "full",
                "task_id": "t1",
                "seed": 1,
                "quality_composite": 0.88,
                "total_cost_usd": 0.35,
                "total_tokens": 8500,
                "wall_ms": 1100,
                "supervision_tokens": 900,
                "compliant_all": 1,
                "replacements": 1,
                "replacements_denied": 1,
            },
            {
                "arm": "full",
                "task_id": "t2",
                "seed": 1,
                "quality_composite": 0.92,
                "total_cost_usd": 0.38,
                "total_tokens": 9000,
                "wall_ms": 1250,
                "supervision_tokens": 1000,
                "compliant_all": 1,
                "replacements": 1,
                "replacements_denied": 0,
            },
        ]

        summary = compute_metrics_summary(runs, reference_arm="vanilla_fixed", n_resamples=500)
        assert "arms" in summary
        assert "vanilla_fixed" in summary["arms"]
        assert "full" in summary["arms"]

        full_stats = summary["arms"]["full"]
        assert full_stats["quality"]["median"] == 0.9
        assert full_stats["compliance_rate"] == 1.0
        assert full_stats["replacements"] == 2
        assert full_stats["replacements_denied"] == 1
        assert full_stats["replacement_suppression_rate"] == pytest.approx(1 / 3, rel=1e-2)
        assert full_stats["wilcoxon_p_value"] is not None
