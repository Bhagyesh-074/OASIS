"""Statistical protocol and significance testing module (FR-29, TESTING.md).

Implements:
- 10,000-resample bootstrap 95% confidence intervals (BCa / percentile).
- Paired Wilcoxon signed-rank tests across tasks against a reference arm.
- Rank-biserial effect sizes for paired non-parametric differences.
- Holm-Bonferroni step-down correction for family-wise multiple testing.
- Disaggregated failure detection recall by failure_type and severity.
- Aggregate metrics summary table generator for benchmark jobs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(
    data: Sequence[float] | np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.median,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap point estimate and confidence interval.

    Parameters
    ----------
    data:
        1D sequence of numerical observations.
    stat_fn:
        Statistic function applied to each resample (default: np.median).
    confidence:
        Confidence level in (0, 1) (default: 0.95 for 95% CI).
    n_resamples:
        Number of bootstrap iterations (default: 10,000 per TESTING.md).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val

    point_est = float(stat_fn(arr))

    # All identical values case
    if np.all(arr == arr[0]):
        return point_est, point_est, point_est

    rng = np.random.default_rng(seed)
    # Generate bootstrap replicates efficiently with vectorized sampling
    resamples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    resample_stats = np.apply_along_axis(stat_fn, 1, resamples)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(resample_stats, lower_pct))
    ci_upper = float(np.percentile(resample_stats, upper_pct))

    return point_est, ci_lower, ci_upper


def wilcoxon_paired_test(
    sample_a: Sequence[float] | np.ndarray,
    sample_b: Sequence[float] | np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Compute paired Wilcoxon signed-rank test across tasks.

    Parameters
    ----------
    sample_a, sample_b:
        Paired observation vectors of equal length.
    alternative:
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    tuple[float, float]
        (statistic, p_value)
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)

    if len(arr_a) != len(arr_b):
        raise ValueError(
            f"Paired samples must have identical length: {len(arr_a)} != {len(arr_b)}"
        )

    # Filter out pairs with NaNs
    valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
    a_valid = arr_a[valid]
    b_valid = arr_b[valid]

    diffs = a_valid - b_valid
    non_zero = diffs[diffs != 0]

    # If no non-zero differences or fewer than 2 pairs, difference is null
    if len(non_zero) == 0:
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(a_valid, b_valid, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        # Handles edge cases in small sample tests
        return 0.0, 1.0


def rank_biserial_effect_size(
    sample_a: Sequence[float] | np.ndarray,
    sample_b: Sequence[float] | np.ndarray,
) -> float:
    """Compute rank-biserial correlation effect size for paired samples.

    Effect size: r = (W+ - W-) / (W+ + W-) in [-1.0, 1.0].
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)

    if len(arr_a) != len(arr_b):
        raise ValueError("Samples must have equal length for paired effect size")

    diffs = arr_a - arr_b
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        return 0.0

    abs_diffs = np.abs(non_zero)
    ranks = stats.rankdata(abs_diffs)

    w_plus = float(np.sum(ranks[non_zero > 0]))
    w_minus = float(np.sum(ranks[non_zero < 0]))
    total = w_plus + w_minus

    if total == 0:
        return 0.0
    return (w_plus - w_minus) / total


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Compute Holm-Bonferroni step-down adjusted p-values.

    Preserves original sequence ordering in output.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [float(p_values[0])]

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: list[tuple[int, float]] = []

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        multiplier = m - rank
        raw_adj = min(1.0, p_val * multiplier)
        adj_val = max(running_max, raw_adj)
        running_max = adj_val
        adjusted.append((orig_idx, min(1.0, adj_val)))

    # Restore original ordering
    adjusted.sort(key=lambda x: x[0])
    return [adj[1] for adj in adjusted]


def compute_recall_by_type_and_severity(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute detection recall disaggregated by failure_type and severity.

    TESTING.md: 'Recall is always reported per failure type and severity,
    never aggregated alone'.
    """
    if not records:
        return {}

    df = pd.DataFrame(records)
    if (
        "failure_type" not in df.columns
        or "severity" not in df.columns
        or "detected" not in df.columns
    ):
        return {}

    grouped = (
        df.groupby(["failure_type", "severity"])["detected"]
        .agg(["sum", "count"])
        .reset_index()
    )
    result: dict[str, dict[str, float]] = {}

    for _, row in grouped.iterrows():
        ftype = str(row["failure_type"])
        sev = str(row["severity"])
        recall = float(row["sum"]) / float(row["count"]) if row["count"] > 0 else 0.0

        if ftype not in result:
            result[ftype] = {}
        result[ftype][sev] = round(recall, 4)

    return result


def compute_metrics_summary(
    runs: Sequence[dict[str, Any]],
    reference_arm: str = "vanilla_fixed",
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Compute comprehensive aggregate metrics table matching GET /v1/metrics/summary.

    Computes for each arm:
    - Medians and 95% bootstrap CIs for quality, tokens, cost, wall_ms, supervision_share
    - Compliance rate (all declared dimensions satisfied)
    - Replacement counts, denials, and suppression rate
    - Paired Wilcoxon p-value against reference_arm on quality
    - Rank-biserial effect size
    - Holm-Bonferroni adjusted p-values
    """
    if not runs:
        return {"arms": {}, "reference_arm": reference_arm}

    df = pd.DataFrame(runs)
    if "arm" not in df.columns:
        return {"arms": {}, "reference_arm": reference_arm}

    arms = sorted(df["arm"].unique().tolist())
    summary: dict[str, Any] = {"arms": {}, "reference_arm": reference_arm}

    ref_df = (
        df[df["arm"] == reference_arm] if reference_arm in df["arm"].values else None
    )

    arm_p_values: list[tuple[str, float]] = []

    for arm in arms:
        arm_df = df[df["arm"] == arm]
        total_runs = len(arm_df)

        # 1. Quality
        q_vals = arm_df["quality_composite"].dropna().values
        q_med, q_low, q_high = bootstrap_ci(
            q_vals, stat_fn=np.median, n_resamples=n_resamples
        )

        # 2. Cost USD
        c_vals = arm_df["total_cost_usd"].dropna().values
        c_med, c_low, c_high = bootstrap_ci(
            c_vals, stat_fn=np.median, n_resamples=n_resamples
        )

        # 3. Tokens
        t_vals = arm_df["total_tokens"].dropna().values
        t_med, t_low, t_high = bootstrap_ci(
            t_vals, stat_fn=np.median, n_resamples=n_resamples
        )

        # 4. Wall Time ms
        w_vals = arm_df["wall_ms"].dropna().values
        w_med, w_low, w_high = bootstrap_ci(
            w_vals, stat_fn=np.median, n_resamples=n_resamples
        )

        # 5. Supervision token share
        sup_vals = (
            (arm_df["supervision_tokens"] / arm_df["total_tokens"].replace(0, 1))
            .dropna()
            .values
            if "supervision_tokens" in arm_df.columns
            else np.array([])
        )
        sup_med, sup_low, sup_high = bootstrap_ci(
            sup_vals, stat_fn=np.median, n_resamples=n_resamples
        )

        # 6. Compliance rate
        compliant_count = (
            int(arm_df["compliant_all"].sum())
            if "compliant_all" in arm_df.columns
            else 0
        )
        compliance_rate = compliant_count / total_runs if total_runs > 0 else 0.0

        # 7. Replacements & suppression rate
        replacements = (
            int(arm_df["replacements"].sum())
            if "replacements" in arm_df.columns
            else 0
        )
        denials = (
            int(arm_df["replacements_denied"].sum())
            if "replacements_denied" in arm_df.columns
            else 0
        )
        total_attempts = replacements + denials
        suppression_rate = (
            (denials / total_attempts) if total_attempts > 0 else 0.0
        )

        # 8. Paired Wilcoxon and effect size vs reference arm
        p_val = 1.0
        effect_size = 0.0

        if ref_df is not None and arm != reference_arm:
            # Pair on (task_id, seed) if available
            if "task_id" in arm_df.columns and "seed" in arm_df.columns:
                merged = pd.merge(
                    arm_df[["task_id", "seed", "quality_composite"]],
                    ref_df[["task_id", "seed", "quality_composite"]],
                    on=["task_id", "seed"],
                    suffixes=("_arm", "_ref"),
                ).dropna()
                if len(merged) > 0:
                    _, p_val = wilcoxon_paired_test(
                        merged["quality_composite_arm"].values,
                        merged["quality_composite_ref"].values,
                    )
                    effect_size = rank_biserial_effect_size(
                        merged["quality_composite_arm"].values,
                        merged["quality_composite_ref"].values,
                    )
            arm_p_values.append((arm, p_val))

        summary["arms"][arm] = {
            "total_runs": total_runs,
            "quality": {
                "median": round(q_med, 4),
                "ci_95": [round(q_low, 4), round(q_high, 4)],
            },
            "cost_usd": {
                "median": round(c_med, 4),
                "ci_95": [round(c_low, 4), round(c_high, 4)],
            },
            "tokens": {
                "median": round(t_med, 1),
                "ci_95": [round(t_low, 1), round(t_high, 1)],
            },
            "wall_ms": {
                "median": round(w_med, 1),
                "ci_95": [round(w_low, 1), round(w_high, 1)],
            },
            "supervision_share": {
                "median": round(sup_med, 4),
                "ci_95": [round(sup_low, 4), round(sup_high, 4)],
            },
            "compliance_rate": round(compliance_rate, 4),
            "replacements": replacements,
            "replacements_denied": denials,
            "replacement_suppression_rate": round(suppression_rate, 4),
            "wilcoxon_p_value": round(p_val, 6) if arm != reference_arm else None,
            "effect_size": round(effect_size, 4) if arm != reference_arm else None,
            "holm_adjusted_p_value": None,  # Will populate below
        }

    # Apply Holm-Bonferroni correction across arms
    if arm_p_values:
        raw_p = [p for _, p in arm_p_values]
        adj_p = holm_bonferroni(raw_p)
        for (arm, _), adj in zip(arm_p_values, adj_p, strict=False):
            summary["arms"][arm]["holm_adjusted_p_value"] = round(adj, 6)

    return summary
