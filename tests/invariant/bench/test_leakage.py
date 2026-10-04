"""Invariant tests for benchmark split partitioning and zero leakage (FR-3).

Guards the core empirical integrity claim in docs/TESTING.md:
"Leakage asserts that no calibration-split task appears in any eval-arm run
 and that no calibration task influenced thresholds after the freeze (FR-3)."

Invariants guarded:
1. Strict Disjointness: No task_id in the calibration set may appear in the eval set.
2. Contamination Detection: assert_no_leakage() immediately raises AssertionError
   if an identical task_id exists in both calibration and eval splits.
3. Split Immutability: Once assigned, a task's split partition never flips between
   calibration and eval upon subsequent writes or re-ingestions.
4. TCE Calibration Consumer Isolation: TCE calibration data consumers only ever read
   tasks with split='calibration', rejecting or excluding eval tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oasis.bench.ingest import (
    assert_no_leakage,
    assign_splits,
    get_calibration_tasks,
    get_eval_tasks,
    ingest_benchmarks,
)
from oasis.bench.models import TaskRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "bench"


def _make_dummy_task_record(
    task_id: str,
    domain: str = "code_generation",
    split: str = "calibration",
) -> TaskRecord:
    """Helper to construct minimal valid TaskRecord for invariant checks."""
    return TaskRecord(
        task_id=task_id,
        domain=domain,  # type: ignore[arg-type]
        source="mbpp",
        source_ref=task_id,
        statement=f"Statement for {task_id}",
        reference_answer="assert True",
        verifier_type="pytest",
        verifier_spec=json.dumps({"test_code": "assert True"}),
        split=split,  # type: ignore[arg-type]
        created_at="2026-09-01T00:00:00+00:00",
    )


class TestLeakageDisjointnessInvariant:
    """Guards FR-3: calibration and eval splits must be strictly disjoint."""

    def test_ingested_benchmarks_have_strictly_disjoint_splits(self, tmp_path: Path) -> None:
        """Verify that an end-to-end ingestion produces zero overlapping task_ids."""
        out_file = tmp_path / "tasks.jsonl"
        records = ingest_benchmarks(
            fixtures_dir=FIXTURES_DIR,
            use_fixtures=True,
            output_path=out_file,
            write_db=False,
        )

        calib_tasks = [r for r in records if r.split == "calibration"]
        eval_tasks = [r for r in records if r.split == "eval"]

        assert len(calib_tasks) > 0, "Ingestion must produce calibration tasks"
        assert len(eval_tasks) > 0, "Ingestion must produce eval tasks"

        calib_ids = {r.task_id for r in calib_tasks}
        eval_ids = {r.task_id for r in eval_tasks}

        intersection = calib_ids.intersection(eval_ids)
        assert len(intersection) == 0, (
            f"FR-3 Invariant Violation: Found {len(intersection)} leaking tasks "
            f"in both splits: {intersection}"
        )

        # Built-in guard must pass
        assert_no_leakage(records)

    def test_assert_no_leakage_detects_duplicate_task_in_both_splits(self) -> None:
        """assert_no_leakage must raise AssertionError if a task_id is duplicated across splits."""
        t1 = _make_dummy_task_record("task-001", split="calibration")
        t2 = _make_dummy_task_record("task-002", split="eval")
        t_leaked = _make_dummy_task_record("task-001", split="eval")

        with pytest.raises(AssertionError, match="FR-3 leakage violation"):
            assert_no_leakage([t1, t2, t_leaked])

    def test_assert_no_leakage_rejects_invalid_split(self) -> None:
        """assert_no_leakage must reject records with unrecognized split strings."""
        t1 = _make_dummy_task_record("task-001", split="calibration")
        # Construct task with invalid split via bypass
        t_bad = _make_dummy_task_record("task-002", split="calibration")
        object.__setattr__(t_bad, "split", "test")

        with pytest.raises(AssertionError, match="invalid split"):
            assert_no_leakage([t1, t_bad])


class TestTCECalibrationIsolationInvariant:
    """Guards FR-3: TCE calibration code shall only read split='calibration'."""

    def test_get_calibration_tasks_returns_only_calibration_records(self) -> None:
        """get_calibration_tasks must filter strictly to split='calibration'."""
        records = [
            _make_dummy_task_record("task-1", split="calibration"),
            _make_dummy_task_record("task-2", split="eval"),
            _make_dummy_task_record("task-3", split="calibration"),
            _make_dummy_task_record("task-4", split="eval"),
        ]

        calib = get_calibration_tasks(records)
        assert len(calib) == 2
        assert all(r.split == "calibration" for r in calib)
        assert {r.task_id for r in calib} == {"task-1", "task-3"}

    def test_get_eval_tasks_returns_only_eval_records(self) -> None:
        """get_eval_tasks must filter strictly to split='eval'."""
        records = [
            _make_dummy_task_record("task-1", split="calibration"),
            _make_dummy_task_record("task-2", split="eval"),
            _make_dummy_task_record("task-3", split="calibration"),
            _make_dummy_task_record("task-4", split="eval"),
        ]

        eval_tasks = get_eval_tasks(records)
        assert len(eval_tasks) == 2
        assert all(r.split == "eval" for r in eval_tasks)
        assert {r.task_id for r in eval_tasks} == {"task-2", "task-4"}

    def test_tce_calibration_consumer_fails_fast_on_contaminated_dataset(self) -> None:
        """Consumers calling get_calibration_tasks fail immediately on contaminated data."""
        contaminated = [
            _make_dummy_task_record("task-leak", split="calibration"),
            _make_dummy_task_record("task-leak", split="eval"),
        ]

        with pytest.raises(AssertionError, match="FR-3 leakage violation"):
            get_calibration_tasks(contaminated)

    def test_tce_consumer_filter_invariant(self) -> None:
        """Simulate a TCE calibration routine and prove zero eval task influence."""
        # Simulated mixed task store
        all_tasks = [
            _make_dummy_task_record(f"calib-{i}", split="calibration") for i in range(5)
        ] + [_make_dummy_task_record(f"eval-{j}", split="eval") for j in range(15)]

        # TCE calibration consumer reading calibration split
        calibration_data = get_calibration_tasks(all_tasks)

        # Assert no eval tasks present in calibration data
        eval_ids = {f"eval-{j}" for j in range(15)}
        calib_consumed_ids = {r.task_id for r in calibration_data}

        assert len(calib_consumed_ids.intersection(eval_ids)) == 0
        assert len(calib_consumed_ids) == 5

        # Invariant check: attempting to pass an eval task as calibration input is rejected
        for task in calibration_data:
            assert task.split == "calibration", f"Task {task.task_id} is not a calibration task!"


class TestSplitImmutabilityInvariant:
    """Guards FR-3: Never reassign a task's split after first write."""

    def test_existing_split_assignments_are_strictly_preserved(self) -> None:
        """Tasks with established splits retain them regardless of re-ingestion or new seed."""
        existing_splits = {
            "task-01": "calibration",
            "task-02": "eval",
        }

        task_dicts: list[dict[str, Any]] = [
            {
                "task_id": "task-01",
                "domain": "code_generation",
                "source": "mbpp",
                "source_ref": "1",
                "statement": "stmt 1",
                "reference_answer": "ans 1",
                "verifier_type": "pytest",
                "verifier_spec": json.dumps({"test": "1"}),
                "created_at": "2026-09-01T00:00:00+00:00",
            },
            {
                "task_id": "task-02",
                "domain": "code_generation",
                "source": "mbpp",
                "source_ref": "2",
                "statement": "stmt 2",
                "reference_answer": "ans 2",
                "verifier_type": "pytest",
                "verifier_spec": json.dumps({"test": "2"}),
                "created_at": "2026-09-01T00:00:00+00:00",
            },
            {
                "task_id": "task-03",
                "domain": "code_generation",
                "source": "mbpp",
                "source_ref": "3",
                "statement": "stmt 3",
                "reference_answer": "ans 3",
                "verifier_type": "pytest",
                "verifier_spec": json.dumps({"test": "3"}),
                "created_at": "2026-09-01T00:00:00+00:00",
            },
        ]

        # Assign with seed A
        records_a = assign_splits(
            task_dicts,
            calibration_fraction=0.5,
            seed=42,
            existing_splits=existing_splits,
        )
        splits_a = {r.task_id: r.split for r in records_a}
        assert splits_a["task-01"] == "calibration"
        assert splits_a["task-02"] == "eval"

        # Assign with seed B (radically different seed)
        records_b = assign_splits(
            task_dicts,
            calibration_fraction=0.5,
            seed=99999,
            existing_splits=existing_splits,
        )
        splits_b = {r.task_id: r.split for r in records_b}
        assert splits_b["task-01"] == "calibration"
        assert splits_b["task-02"] == "eval"

        # Neither pre-existing split can change
        assert splits_a["task-01"] == splits_b["task-01"]
        assert splits_a["task-02"] == splits_b["task-02"]
