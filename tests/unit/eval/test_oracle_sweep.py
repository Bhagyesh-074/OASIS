"""Unit tests for oracle sweep algorithms, types, storage, and statistics.

Covers:
- compute_oracle_k hand-computed cases (ties, exact tolerance, even/odd seeds, incomplete tasks)
- JsonlStore round trip with truncated lines & InMemoryStore
- Pearson and Spearman correlation helpers (hand-computed, ties, constant series, < 3 points)
- Estimator evaluation (clamping, exact match rate, size and quality regret)
- Configuration validation (k in 1..5, positive seeds, tolerances)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from oasis.eval.oracle_sweep import (
    InMemoryStore,
    JsonlStore,
    OracleResult,
    OracleSweepConfig,
    RunResult,
    compute_oracle_k,
    evaluate_estimator,
    load_oracle_sweep_config,
    pearson_correlation,
    spearman_correlation,
)

# ============================================================================
# compute_oracle_k Tests
# ============================================================================


def test_compute_oracle_k_standard() -> None:
    """Hand-computed standard case where oracle_k is smaller than best k."""
    results_by_k = {
        1: [0.60, 0.62, 0.64],  # median = 0.62
        2: [0.75, 0.77, 0.76],  # median = 0.76
        3: [0.81, 0.80, 0.82],  # median = 0.81
        4: [0.82, 0.83, 0.81],  # median = 0.82 (best)
        5: [0.81, 0.82, 0.83],  # median = 0.82
    }
    # Tolerance 0.03 -> threshold = 0.82 - 0.03 = 0.79
    # k=3 median is 0.81 >= 0.79, so oracle_k = 3
    res = compute_oracle_k(results_by_k, tolerance=0.03)
    assert res is not None
    assert res.best_median == pytest.approx(0.82)
    assert res.oracle_k == 3
    assert res.medians[1] == pytest.approx(0.62)
    assert res.medians[3] == pytest.approx(0.81)


def test_compute_oracle_k_tie() -> None:
    """When multiple k achieve the identical best quality, smallest k is chosen."""
    results_by_k = {
        1: [0.70, 0.70, 0.70],  # median = 0.70
        2: [0.90, 0.90, 0.90],  # median = 0.90 (tie for best)
        3: [0.85, 0.85, 0.85],  # median = 0.85
        4: [0.90, 0.90, 0.90],  # median = 0.90 (tie for best)
        5: [0.88, 0.88, 0.88],  # median = 0.88
    }
    # With tolerance 0.01, threshold = 0.89. Smallest k meeting it is k=2.
    res = compute_oracle_k(results_by_k, tolerance=0.01)
    assert res is not None
    assert res.oracle_k == 2
    assert res.best_median == pytest.approx(0.90)


def test_compute_oracle_k_exact_tolerance() -> None:
    """A median exactly at best_median - tolerance is inclusive and selected."""
    results_by_k = {
        1: [0.77, 0.77, 0.77],  # median = 0.77 (exactly 0.80 - 0.03)
        2: [0.76, 0.76, 0.76],  # median = 0.76
        3: [0.80, 0.80, 0.80],  # median = 0.80 (best)
    }
    # tolerance = 0.03 -> 0.80 - 0.03 = 0.77. Inclusive epsilon allows k=1.
    res = compute_oracle_k(results_by_k, tolerance=0.03)
    assert res is not None
    assert res.oracle_k == 1
    assert res.best_median == pytest.approx(0.80)


def test_compute_oracle_k_even_seeds() -> None:
    """Even number of seeds takes the mean of the two middle elements."""
    results_by_k = {
        1: [0.70, 0.90],  # median = (0.70 + 0.90) / 2 = 0.80
        2: [0.82, 0.88],  # median = 0.85 (best)
    }
    # tolerance = 0.04 -> threshold = 0.81 -> k=1 (0.80) does not qualify -> k=2
    res_tight = compute_oracle_k(results_by_k, tolerance=0.04, expected_seeds=2)
    assert res_tight is not None
    assert res_tight.oracle_k == 2

    # tolerance = 0.05 -> threshold = 0.80 -> k=1 qualifies
    res_loose = compute_oracle_k(results_by_k, tolerance=0.05, expected_seeds=2)
    assert res_loose is not None
    assert res_loose.oracle_k == 1


def test_compute_oracle_k_odd_seeds() -> None:
    """Odd number of seeds takes the exact middle element."""
    results_by_k = {
        1: [0.50, 0.75, 0.90],  # median = 0.75
        2: [0.80, 0.85, 0.90],  # median = 0.85
    }
    res = compute_oracle_k(results_by_k, tolerance=0.05, expected_seeds=3)
    assert res is not None
    assert res.best_median == pytest.approx(0.85)
    assert res.oracle_k == 2


def test_compute_oracle_k_incomplete_task() -> None:
    """Tasks with fewer seeds than expected or missing configured k return None."""
    # Fewer seeds than expected
    results_incomplete_seeds = {
        1: [0.8, 0.8, 0.8],
        2: [0.85, 0.85],  # only 2 seeds, expected 3
    }
    assert (
        compute_oracle_k(results_incomplete_seeds, tolerance=0.03, expected_seeds=3)
        is None
    )

    # Missing a configured k
    results_missing_k = {
        1: [0.8, 0.8, 0.8],
        2: [0.85, 0.85, 0.85],
    }
    assert (
        compute_oracle_k(
            results_missing_k,
            tolerance=0.03,
            expected_seeds=3,
            k_values=[1, 2, 3, 4, 5],
        )
        is None
    )

    # Empty results
    assert compute_oracle_k({}, tolerance=0.03) is None


# ============================================================================
# SweepStore Tests
# ============================================================================


def test_in_memory_store_operations() -> None:
    """InMemoryStore records runs and returns existing keys."""
    store = InMemoryStore()
    assert store.existing_keys() == set()

    res = RunResult(quality=0.85, cost_usd=0.002, tokens=150, wall_ms=250)
    store.save("task_01", 1, 1, res)

    assert store.existing_keys() == {("task_01", 1, 1)}
    loaded = store.load_all()
    assert loaded[("task_01", 1, 1)] == res


def test_jsonl_store_roundtrip_with_truncated_line(tmp_path: Path) -> None:
    """JsonlStore correctly writes LF rows and skips a truncated/corrupted last line."""
    log_file = tmp_path / "sweep.jsonl"
    store = JsonlStore(log_file)

    r1 = RunResult(quality=0.75, cost_usd=0.01, tokens=500, wall_ms=1000)
    r2 = RunResult(quality=0.85, cost_usd=0.02, tokens=750, wall_ms=1500)

    store.save("task_01", 1, 1, r1)
    store.save("task_01", 2, 1, r2)

    # Simulate an interrupted partial append (truncated line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write('{"task_id": "task_02", "k": 1, "seed": 1, "quality": 0.90, "co\n')

    # Load back: should recover r1 and r2 and skip the broken line
    loaded = store.load_all()
    assert len(loaded) == 2
    assert loaded[("task_01", 1, 1)] == r1
    assert loaded[("task_01", 2, 1)] == r2
    assert store.existing_keys() == {("task_01", 1, 1), ("task_01", 2, 1)}


# ============================================================================
# Statistics & Correlation Helpers Tests
# ============================================================================


def test_pearson_correlation_known_values() -> None:
    """Test Pearson correlation against perfect positive, negative, and hand-computed."""
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    assert pearson_correlation(x, y) == pytest.approx(1.0)

    y_inv = [6.0, 4.0, 2.0]
    assert pearson_correlation(x, y_inv) == pytest.approx(-1.0)


def test_pearson_correlation_constant_and_small() -> None:
    """Pearson correlation returns None (not NaN) for constant series or < 3 points."""
    # Constant series (variance is zero)
    assert pearson_correlation([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None
    assert pearson_correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None

    # Fewer than 3 points
    assert pearson_correlation([1.0, 2.0], [2.0, 4.0]) is None
    assert pearson_correlation([], []) is None


def test_spearman_correlation_with_ties() -> None:
    """Spearman correlation uses fractional average ranks for tied values."""
    # x has a tie at 2.0: ranks should be [1.0, 2.5, 2.5, 4.0]
    # y: [1.0, 2.0, 3.0, 4.0]
    # Hand-computed: cov = 4.5, var_x = 4.5, var_y = 5.0 -> 4.5 / sqrt(22.5) ≈ 0.9486833
    x = [1.0, 2.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0, 4.0]
    res = spearman_correlation(x, y)
    assert res is not None
    expected = 4.5 / math.sqrt(4.5 * 5.0)
    assert res == pytest.approx(expected, rel=1e-5)


def test_spearman_correlation_constant_and_small() -> None:
    """Spearman returns None for constant series or < 3 points."""
    assert spearman_correlation([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert spearman_correlation([1.0, 2.0], [2.0, 3.0]) is None


# ============================================================================
# Estimator Evaluation & Regret Tests (ADR-011)
# ============================================================================


def test_evaluate_estimator_clamping_and_regret() -> None:
    """Test clamping to max configured k, size regret, quality regret, and exact match."""
    oracle = {
        "t1": OracleResult(
            oracle_k=3,
            best_median=0.90,
            medians={1: 0.60, 2: 0.80, 3: 0.88, 4: 0.90, 5: 0.90},
        ),
        "t2": OracleResult(
            oracle_k=5,
            best_median=0.85,
            medians={1: 0.50, 2: 0.60, 3: 0.70, 4: 0.80, 5: 0.85},
        ),
        "t3": OracleResult(
            oracle_k=2,
            best_median=0.80,
            medians={1: 0.70, 2: 0.80, 3: 0.80, 4: 0.80, 5: 0.80},
        ),
    }

    # Predicted MVTS: t1 predicts 6 (exceeds max 5), t2 predicts 5, t3 predicts 1
    predicted = {"t1": 6, "t2": 5, "t3": 1}

    report = evaluate_estimator(predicted, oracle, max_k=5)

    # Clamping: t1 (6 -> 5) is clamped. Total clamped = 1
    assert report.clamped_count == 1
    assert report.task_count == 3

    # Clamped predictions: t1: 5, t2: 5, t3: 1
    # Oracle k:            t1: 3, t2: 5, t3: 2
    # Size regrets:        (5-3)=2, (5-5)=0, (1-2)=-1 -> mean = (2 + 0 - 1) / 3 = 1/3
    assert report.mean_size_regret == pytest.approx(1.0 / 3.0)

    # Quality regrets:
    # t1: best=0.90, median at k=5 is 0.90 -> regret = 0.0
    # t2: best=0.85, median at k=5 is 0.85 -> regret = 0.0
    # t3: best=0.80, median at k=1 is 0.70 -> regret = 0.10
    # Mean quality regret: (0 + 0 + 0.10) / 3 = 0.10 / 3
    assert report.mean_quality_regret == pytest.approx(0.10 / 3.0)

    # Exact match: t2 matches (5 == 5). Rate = 1/3
    assert report.exact_match_rate == pytest.approx(1.0 / 3.0)

    # Properties
    assert report.pearson_correlation == report.pearson
    assert report.spearman_correlation == report.spearman


# ============================================================================
# Configuration Validation Tests (FR-3, NFR-3)
# ============================================================================


def test_oracle_sweep_config_validation() -> None:
    """Config enforces k between 1 and 5, positive seeds, and valid numeric bounds."""
    # Valid default config
    cfg = OracleSweepConfig()
    assert cfg.k_values == [1, 2, 3, 4, 5]
    assert cfg.seeds == [1, 2, 3]
    assert cfg.tolerance == 0.03

    # k outside 1..5 raises ValueError per SQLite DB check constraint
    with pytest.raises(ValueError, match="k must be between 1 and 5"):
        OracleSweepConfig(k_values=[1, 2, 6])

    # Empty k_values
    with pytest.raises(ValueError, match="k_values must not be empty"):
        OracleSweepConfig(k_values=[])

    # Non-positive seed
    with pytest.raises(ValueError, match="seeds must be positive"):
        OracleSweepConfig(seeds=[0, 1])

    # Negative tolerance
    with pytest.raises(ValueError):
        OracleSweepConfig(tolerance=-0.01)


def test_load_oracle_sweep_config_file(tmp_path: Path) -> None:
    """Config loads accurately from YAML file."""
    yaml_path = tmp_path / "custom_sweep.yaml"
    yaml_path.write_text(
        "k_values: [1, 2, 3]\nseeds: [10, 20]\ntolerance: 0.05\ntranche_size: 5\nconcurrency: 2\nmax_retries: 1\n",
        encoding="utf-8",
    )
    loaded = load_oracle_sweep_config(yaml_path)
    assert loaded.k_values == [1, 2, 3]
    assert loaded.seeds == [10, 20]
    assert loaded.tolerance == 0.05
    assert loaded.tranche_size == 5
    assert loaded.concurrency == 2
    assert loaded.max_retries == 1
