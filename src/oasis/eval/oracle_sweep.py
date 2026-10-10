"""Oracle team-size sweep, store, analysis, and estimator evaluation.

Key Requirements and Architecture Citations:
- FR-27: Ground-truth Minimum Viable Team Size (MVTS) established by running
  each benchmark task across team sizes k = 1..5.
- ADR-011: Evaluates complexity estimator as a cheap predictor of oracle_k,
  quantified through correlation (Pearson, Spearman) and regret (size, quality).
- FR-3: Data leakage prevention: only 'eval' split tasks are swept; calibration
  split tasks must be rejected with ValueError.
- NFR-3: Strict API spend control via spend_check() ceiling assertions prior to
  each tranche and run.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# 1. Configuration (FR-27, ADR-011, FR-3, NFR-3)
# ============================================================================


class OracleSweepConfig(BaseModel):
    """Validated configuration for oracle team-size sweep.

    Enforces SQLite DB schema constraints (k BETWEEN 1 AND 5, docs/DATABASE.md).
    No hard-coded values in execution code.
    """

    model_config = ConfigDict(frozen=True)

    k_values: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    tolerance: float = Field(default=0.03, ge=0.0)
    tranche_size: int = Field(default=10, gt=0)
    concurrency: int = Field(default=4, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("k_values must not be empty")
        for k in v:
            if not (1 <= k <= 5):
                raise ValueError(
                    f"k must be between 1 and 5 per database schema constraint, got {k}"
                )
        return sorted(set(v))

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("seeds must not be empty")
        for s in v:
            if s <= 0:
                raise ValueError(f"seeds must be positive integers, got {s}")
        return sorted(set(v))


def load_oracle_sweep_config(path: str | Path | None = None) -> OracleSweepConfig:
    """Load and validate OracleSweepConfig from a YAML file or defaults."""
    target_path = Path(path) if path is not None else Path("config/oracle_sweep.yaml")
    if not target_path.exists():
        if path is not None:
            raise FileNotFoundError(f"Oracle sweep config not found at: {target_path}")
        return OracleSweepConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return OracleSweepConfig.model_validate(data)


# ============================================================================
# 2. Types matching SQLite Schema (docs/DATABASE.md)
# ============================================================================


@dataclass(frozen=True)
class TaskRef:
    """Reference to a task identifying its id and partition split."""

    task_id: str
    split: str


@dataclass(frozen=True)
class RunResult:
    """Execution metrics for an oracle sweep run matching the oracle_sweep table."""

    quality: float
    cost_usd: float
    tokens: int
    wall_ms: int


@dataclass(frozen=True)
class OracleResult:
    """Derived ground-truth result for a task across swept team sizes (ADR-011)."""

    oracle_k: int
    best_median: float
    medians: dict[int, float]


# ============================================================================
# 3. Pure Calculation: compute_oracle_k (ADR-011, docs/DATABASE.md)
# ============================================================================


def compute_oracle_k(
    results_by_k: dict[int, list[float]],
    tolerance: float,
    expected_seeds: int | None = None,
    k_values: list[int] | None = None,
) -> OracleResult | None:
    """Compute ground-truth oracle_k for a task over multi-seed sweep results.

    oracle_k is defined in docs/DATABASE.md (Derived quantities) and ADR-011:
    The smallest team size k whose median quality lies within tolerance
    of the best observed median quality across all evaluated k.

    Inclusive boundary comparison with 1e-9 epsilon prevents floating-point noise
    from flipping decision boundaries.

    Returns None if any configured k has fewer results than configured seeds,
    reporting the task as incomplete.
    """
    if not results_by_k:
        return None

    # Check completeness against expected k_values if provided
    if k_values is not None:
        for k in k_values:
            if k not in results_by_k:
                return None

    # Determine required seed count per team size k
    if expected_seeds is not None:
        required_count = expected_seeds
        for qualities in results_by_k.values():
            if len(qualities) < required_count:
                return None
    else:
        # If expected_seeds is not explicitly passed, check non-empty and consistent seed counts
        first_len = len(next(iter(results_by_k.values())))
        if first_len == 0:
            return None
        for qualities in results_by_k.values():
            if len(qualities) != first_len or len(qualities) == 0:
                return None

    # Compute median quality per k over seeds
    medians: dict[int, float] = {
        k: float(statistics.median(qualities))
        for k, qualities in sorted(results_by_k.items())
    }

    best_median = max(medians.values())
    epsilon = 1e-9
    threshold = best_median - tolerance - epsilon

    # Smallest k whose median quality is within tolerance of best median
    for k in sorted(medians.keys()):
        if medians[k] >= threshold:
            return OracleResult(
                oracle_k=k,
                best_median=best_median,
                medians=medians,
            )

    # Fallback to smallest k with best median
    min_k = min(medians.keys())
    return OracleResult(oracle_k=min_k, best_median=best_median, medians=medians)


# ============================================================================
# 4. Storage Protocols & Implementations (InMemoryStore, JsonlStore)
# ============================================================================


class SweepStore(Protocol):
    """Protocol for persisting and loading oracle sweep run results."""

    def existing_keys(self) -> set[tuple[str, int, int]]:
        """Return set of (task_id, k, seed) keys already stored."""
        ...

    def save(self, task_id: str, k: int, seed: int, result: RunResult) -> None:
        """Save a single run result."""
        ...

    def load_all(self) -> dict[tuple[str, int, int], RunResult]:
        """Load all saved run results."""
        ...


class InMemoryStore:
    """In-memory implementation of SweepStore for unit testing."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, int, int], RunResult] = {}

    def existing_keys(self) -> set[tuple[str, int, int]]:
        return set(self._store.keys())

    def save(self, task_id: str, k: int, seed: int, result: RunResult) -> None:
        self._store[(task_id, k, seed)] = result

    def load_all(self) -> dict[tuple[str, int, int], RunResult]:
        return dict(self._store)


class JsonlStore:
    """Append-only JSONL implementation of SweepStore.

    Uses LF newlines, one row per line. Skips truncated lines during reading
    to ensure crash recovery resilience.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def existing_keys(self) -> set[tuple[str, int, int]]:
        return set(self.load_all().keys())

    def save(self, task_id: str, k: int, seed: int, result: RunResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "k": k,
            "seed": seed,
            "quality": result.quality,
            "cost_usd": result.cost_usd,
            "tokens": result.tokens,
            "wall_ms": result.wall_ms,
        }
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload) + "\n")

    def load_all(self) -> dict[tuple[str, int, int], RunResult]:
        if not self.path.exists():
            return {}

        results: dict[tuple[str, int, int], RunResult] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    task_id = str(data["task_id"])
                    k = int(data["k"])
                    seed = int(data["seed"])
                    res = RunResult(
                        quality=float(data["quality"]),
                        cost_usd=float(data["cost_usd"]),
                        tokens=int(data["tokens"]),
                        wall_ms=int(data["wall_ms"]),
                    )
                    results[(task_id, k, seed)] = res
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    # Skip truncated or malformed records
                    continue

        return results


# ============================================================================
# 5. Sweep Runner: run_sweep (FR-27, FR-3, NFR-3)
# ============================================================================


@dataclass
class FailedRun:
    """Representation of an unrecoverable run that exceeded max_retries."""

    task_id: str
    k: int
    seed: int
    error: str = ""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            if len(other) == 3:
                return (self.task_id, self.k, self.seed) == other
            if len(other) == 4:
                return (self.task_id, self.k, self.seed, self.error) == other
        if isinstance(other, FailedRun):
            return (self.task_id, self.k, self.seed, self.error) == (
                other.task_id,
                other.k,
                other.seed,
                other.error,
            )
        return False


@dataclass
class SweepReport:
    """Comprehensive summary report returned by run_sweep."""

    attempted: int = 0
    skipped_existing: int = 0
    completed: int = 0
    retried: int = 0
    failed: list[FailedRun] = field(default_factory=list)
    stopped_reason: str | None = None
    cumulative_cost_usd: float = 0.0


async def run_sweep(
    tasks: Iterable[TaskRef],
    run_fn: Callable[[str, int, int], Awaitable[RunResult]],
    store: SweepStore,
    config: OracleSweepConfig,
    spend_check: Callable[[], bool],
    on_tranche: Callable[[SweepReport], None] | None = None,
) -> SweepReport:
    """Execute the oracle sweep across tasks, team sizes k, and seeds.

    Guarantees:
    - FR-3: Rejects any task whose split != 'eval' with ValueError.
    - Determinism: Iterates tasks in sorted task_id order, then k, then seed.
    - Idempotency & Resumability: Skips keys present in store.existing_keys().
    - Bounded concurrency: asyncio.Semaphore(config.concurrency).
    - NFR-3 Spend safety: Calls spend_check() before each tranche and before each
      run starts. If True, stops cleanly without saving partial results.
    - Retries: Retries failed runs up to config.max_retries. Unrecoverable failures
      are appended to report.failed and never stored in SweepStore.
    - Tranche callbacks: Calls on_tranche(report) after each tranche.
    """
    # Enforce FR-3: Refuse any calibration task
    task_list = list(tasks)
    for t in task_list:
        if t.split != "eval":
            raise ValueError(
                f"Task '{t.task_id}' has split '{t.split}'. Oracle sweep only permits 'eval' split tasks (FR-3)."
            )

    # Sort tasks, k values, and seeds for execution determinism
    sorted_tasks = sorted(task_list, key=lambda x: x.task_id)
    sorted_k = sorted(config.k_values)
    sorted_seeds = sorted(config.seeds)

    existing = store.existing_keys()
    to_run: list[tuple[str, int, int]] = []
    skipped_existing = 0

    for task in sorted_tasks:
        for k in sorted_k:
            for seed in sorted_seeds:
                key = (task.task_id, k, seed)
                if key in existing:
                    skipped_existing += 1
                else:
                    to_run.append(key)

    report = SweepReport(
        attempted=0,
        skipped_existing=skipped_existing,
        completed=0,
        retried=0,
        failed=[],
        stopped_reason=None,
        cumulative_cost_usd=0.0,
    )

    if not to_run:
        return report

    semaphore = asyncio.Semaphore(config.concurrency)
    lock = asyncio.Lock()
    stopped = False

    async def execute_single_run(task_id: str, k: int, seed: int) -> None:
        nonlocal stopped
        if stopped:
            return

        async with semaphore:
            if stopped:
                return

            # Check spend ceiling immediately before run begins
            if spend_check():
                async with lock:
                    stopped = True
                    report.stopped_reason = "spend_ceiling_reached"
                return

            async with lock:
                report.attempted += 1

            success = False
            last_error = ""
            attempt = 0

            while attempt <= config.max_retries:
                try:
                    result = await run_fn(task_id, k, seed)
                    # Successful run: persist to store
                    store.save(task_id, k, seed, result)
                    async with lock:
                        report.completed += 1
                        report.cumulative_cost_usd += result.cost_usd
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if attempt < config.max_retries:
                        async with lock:
                            report.retried += 1
                        attempt += 1
                    else:
                        break

            if not success:
                async with lock:
                    report.failed.append(
                        FailedRun(task_id=task_id, k=k, seed=seed, error=last_error)
                    )

    # Process in tranches of config.tranche_size
    tranche_size = config.tranche_size
    for i in range(0, len(to_run), tranche_size):
        # Check spend ceiling before starting new tranche
        if spend_check():
            report.stopped_reason = "spend_ceiling_reached"
            break

        tranche = to_run[i : i + tranche_size]
        await asyncio.gather(
            *(execute_single_run(tid, k_val, s_val) for tid, k_val, s_val in tranche)
        )

        if on_tranche is not None:
            on_tranche(report)

        if stopped:
            break

    return report


# ============================================================================
# 6. Summary: summarise (ADR-011)
# ============================================================================


class SummaryResult(dict[str, OracleResult]):
    """Mapping of complete task_id to OracleResult, with incomplete tasks listed separately.

    Inherits from dict[str, OracleResult] for direct dictionary access and
    compatibility with evaluate_estimator.
    """

    def __init__(
        self, complete: dict[str, OracleResult], incomplete: list[str]
    ) -> None:
        super().__init__(complete)
        self.complete = complete
        self.incomplete = sorted(incomplete)


def summarise(
    store: SweepStore,
    tasks: Iterable[TaskRef],
    config: OracleSweepConfig,
) -> SummaryResult:
    """Summarise store contents into OracleResult per task.

    Tasks with fewer results than configured seeds across any k are listed
    separately in result.incomplete.
    """
    all_runs = store.load_all()
    grouped: dict[str, dict[int, list[float]]] = {}

    for (task_id, k, seed), result in all_runs.items():
        grouped.setdefault(task_id, {}).setdefault(k, []).append(result.quality)

    complete_map: dict[str, OracleResult] = {}
    incomplete_list: list[str] = []

    for task in tasks:
        task_data = grouped.get(task.task_id, {})
        res = compute_oracle_k(
            results_by_k=task_data,
            tolerance=config.tolerance,
            expected_seeds=len(config.seeds),
            k_values=config.k_values,
        )
        if res is None:
            incomplete_list.append(task.task_id)
        else:
            complete_map[task.task_id] = res

    return SummaryResult(complete=complete_map, incomplete=incomplete_list)


# ============================================================================
# 7. Estimator Evaluation & Regret Metrics (ADR-011)
# ============================================================================


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient in pure Python.

    Returns None (never NaN or an exception) if either series is constant,
    has variance < 1e-12, or has fewer than 3 points.
    """
    n = len(x)
    if n != len(y) or n < 3:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x < 1e-12 or var_y < 1e-12:
        return None

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return None

    val = cov / denom
    if math.isnan(val):
        return None
    return float(val)


def _rank_data(values: list[float]) -> list[float]:
    """Compute 1-indexed fractional (average) ranks for ties."""
    n = len(values)
    if n == 0:
        return []

    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0

    while i < n:
        j = i
        while j < n - 1 and math.isclose(
            values[indexed[j + 1]], values[indexed[i]], rel_tol=1e-9, abs_tol=1e-12
        ):
            j += 1
        # Average rank for identical values from rank i+1 to j+1
        avg_rank = (i + j + 2) / 2.0
        for idx in range(i, j + 1):
            ranks[indexed[idx]] = avg_rank
        i = j + 1

    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman rank correlation using average ranks for ties.

    Returns None if either series is constant or has fewer than 3 points.
    """
    n = len(x)
    if n != len(y) or n < 3:
        return None

    ranks_x = _rank_data(x)
    ranks_y = _rank_data(y)

    return pearson_correlation(ranks_x, ranks_y)


@dataclass(frozen=True)
class EstimatorReport:
    """Evaluation report comparing predicted MVTS against oracle ground truth (ADR-011).

    Regret definitions:
    - Size Regret: (predicted_k - oracle_k).
      Positive values reflect over-provisioning (larger team than needed);
      negative values reflect under-provisioning.
    - Quality Regret: (best_median_quality - median_quality_at_predicted_k).
      Measures quality penalty incurred by configuring team size at predicted_k
      instead of optimal oracle_k. Always >= 0 (within float precision).
    """

    pearson: float | None
    spearman: float | None
    mean_size_regret: float
    mean_quality_regret: float
    exact_match_rate: float
    clamped_count: int
    task_count: int

    @property
    def pearson_correlation(self) -> float | None:
        return self.pearson

    @property
    def spearman_correlation(self) -> float | None:
        return self.spearman


def evaluate_estimator(
    predicted_mvts: dict[str, int],
    oracle: dict[str, OracleResult],
    medians: dict[str, dict[int, float]] | None = None,
    max_k: int | None = None,
) -> EstimatorReport:
    """Evaluate team-size estimator predictions against oracle ground truth.

    Predictions exceeding the maximum configured k are clamped to max_k,
    and the number of clamped predictions is recorded.

    Regret definitions (ADR-011):
    - Size Regret: (predicted_k - oracle_k)
    - Quality Regret: (best_median_quality - median_quality_at_predicted_k)
    """
    eval_tasks = sorted(set(predicted_mvts.keys()) & set(oracle.keys()))
    if not eval_tasks:
        return EstimatorReport(
            pearson=None,
            spearman=None,
            mean_size_regret=0.0,
            mean_quality_regret=0.0,
            exact_match_rate=0.0,
            clamped_count=0,
            task_count=0,
        )

    # Determine maximum configured k
    if max_k is None:
        observed_ks = [k for res in oracle.values() for k in res.medians]
        max_k_val = max(observed_ks) if observed_ks else 5
    else:
        max_k_val = max_k

    clamped_count = 0
    clamped_preds: list[float] = []
    oracle_ks: list[float] = []
    size_regrets: list[float] = []
    quality_regrets: list[float] = []
    exact_matches = 0

    for tid in eval_tasks:
        raw_pred = predicted_mvts[tid]
        if raw_pred > max_k_val:
            clamped_pred = max_k_val
            clamped_count += 1
        else:
            clamped_pred = raw_pred

        orc = oracle[tid]
        oracle_k = orc.oracle_k

        clamped_preds.append(float(clamped_pred))
        oracle_ks.append(float(oracle_k))

        # Size regret = predicted_k - oracle_k
        size_regrets.append(float(clamped_pred - oracle_k))

        # Quality regret = best_median - median_at_predicted_k
        if medians is not None and tid in medians and clamped_pred in medians[tid]:
            median_at_pred = medians[tid][clamped_pred]
        elif clamped_pred in orc.medians:
            median_at_pred = orc.medians[clamped_pred]
        else:
            median_at_pred = 0.0

        q_regret = max(0.0, orc.best_median - median_at_pred)
        quality_regrets.append(q_regret)

        if clamped_pred == oracle_k:
            exact_matches += 1

    n = len(eval_tasks)
    p_corr = pearson_correlation(clamped_preds, oracle_ks)
    s_corr = spearman_correlation(clamped_preds, oracle_ks)
    mean_sz = sum(size_regrets) / n
    mean_ql = sum(quality_regrets) / n
    exact_rate = exact_matches / n

    return EstimatorReport(
        pearson=p_corr,
        spearman=s_corr,
        mean_size_regret=mean_sz,
        mean_quality_regret=mean_ql,
        exact_match_rate=exact_rate,
        clamped_count=clamped_count,
        task_count=n,
    )
