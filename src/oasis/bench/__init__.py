"""OASIS Benchmark Ingestion and Task Management (FR-3, FR-13, docs/DATABASE.md)."""

from typing import TYPE_CHECKING, Any

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.models import (
    TaskComplexityLabel,
    TaskDomain,
    TaskRecord,
    TaskSource,
    TaskSplit,
    TaskVerifierType,
)

if TYPE_CHECKING:
    from oasis.bench.ingest import (
        assert_no_leakage,
        assign_splits,
        get_calibration_tasks,
        get_eval_tasks,
        ingest_benchmarks,
        read_task_records,
        save_task_records,
        write_to_db_adapter,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "assert_no_leakage",
        "assign_splits",
        "get_calibration_tasks",
        "get_eval_tasks",
        "ingest_benchmarks",
        "read_task_records",
        "save_task_records",
        "write_to_db_adapter",
    }:
        import oasis.bench.ingest as ingest_mod

        return getattr(ingest_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkConfig",
    "TaskComplexityLabel",
    "TaskDomain",
    "TaskRecord",
    "TaskSource",
    "TaskSplit",
    "TaskVerifierType",
    "assert_no_leakage",
    "assign_splits",
    "get_calibration_tasks",
    "get_eval_tasks",
    "ingest_benchmarks",
    "load_benchmark_config",
    "read_task_records",
    "save_task_records",
    "write_to_db_adapter",
]
