"""OASIS Benchmark Ingestion and Task Management (FR-3, FR-13, docs/DATABASE.md)."""

from typing import TYPE_CHECKING, Any

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.models import (
    CONTEXT_MARKER,
    CitationResolveSpec,
    TaskComplexityLabel,
    TaskDomain,
    TaskRecord,
    TaskSource,
    TaskSplit,
    TaskVerifierType,
)

if TYPE_CHECKING:
    from oasis.bench.authored import load_authored_tasks
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
    if name == "load_authored_tasks":
        import oasis.bench.authored as authored_mod

        return getattr(authored_mod, name)
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
    "CONTEXT_MARKER",
    "BenchmarkConfig",
    "CitationResolveSpec",
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
    "load_authored_tasks",
    "load_benchmark_config",
    "read_task_records",
    "save_task_records",
    "write_to_db_adapter",
]
