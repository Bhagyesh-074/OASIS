"""Benchmark orchestration and metrics summary endpoints: /v1/benchmark/jobs and /v1/metrics/summary."""

from __future__ import annotations

import json
import logging
from typing import Any

import ulid
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from oasis.api.schemas import BenchmarkJobRequest, BenchmarkJobResponse
from oasis.db.models import Database
from oasis.eval.stats import compute_metrics_summary

router = APIRouter(tags=["Benchmark"])
logger = logging.getLogger(__name__)

# In-memory registry of submitted benchmark jobs
_BENCHMARK_JOBS: dict[str, dict[str, Any]] = {}


async def _execute_benchmark_job(
    job_id: str,
    request: BenchmarkJobRequest,
    tasks: list[Any],
) -> None:
    """Run a submitted matrix job and persist its observable progress."""
    job = _BENCHMARK_JOBS[job_id]
    job["status"] = "running"

    try:
        from oasis.eval.runner import BenchmarkRunner, load_budget_profile

        runner = BenchmarkRunner(
            concurrency=request.concurrency,
            mode=request.mode,
        )
        results = await runner.run_matrix(
            tasks=tasks,
            arms=request.arms,
            seeds=request.seeds,
            job_id=job_id,
            concurrency=request.concurrency,
            budget=load_budget_profile(request.budget_profile),
        )
    except NotImplementedError as exc:
        logger.info("Benchmark job %s rejected by runner: %s", job_id, exc)
        job["status"] = "failed"
        job["status_detail"] = str(exc)
        job["failed_runs"] = job["total_runs"]
        return
    except Exception as exc:  # noqa: BLE001 — background jobs must report terminal failure.
        logger.error("Benchmark job %s failed before completion (%s)", job_id, type(exc).__name__)
        job["status"] = "failed"
        job["status_detail"] = f"execution_error: {type(exc).__name__}"
        job["failed_runs"] = job["total_runs"] - job["completed_runs"]
        return

    completed = sum(run.status in ("completed", "halted_budget") for run in results)
    failed = len(results) - completed
    job["completed_runs"] = completed
    job["failed_runs"] = failed + max(0, job["total_runs"] - len(results))
    job["spend_usd"] = round(sum(run.total_cost_usd for run in results), 4)
    job["status"] = (
        "completed"
        if len(results) == job["total_runs"] and job["failed_runs"] == 0
        else "failed"
    )
    if job["status"] == "failed" and job["failed_runs"]:
        job["status_detail"] = "One or more matrix runs failed or were not started"


@router.post("/v1/benchmark/jobs", response_model=BenchmarkJobResponse)
def submit_benchmark_job(
    req: BenchmarkJobRequest,
    background_tasks: BackgroundTasks,
) -> BenchmarkJobResponse:
    """Submit a benchmark ablation matrix job (API_SPEC.md)."""
    db = Database()
    tasks = db.list_tasks(split=req.task_split)
    if not tasks:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_error",
                "message": f"No tasks are available in the {req.task_split!r} split",
            },
        )

    total_runs = len(tasks) * len(req.arms) * len(req.seeds)
    job_id = str(ulid.ULID())

    job_data: dict[str, Any] = {
        "job_id": job_id,
        "name": req.name,
        "status": "queued",
        "total_runs": total_runs,
        "completed_runs": 0,
        "failed_runs": 0,
        "spend_usd": 0.0,
        "status_detail": None,
        "arms": req.arms,
        "seeds": req.seeds,
        "mode": req.mode,
    }
    _BENCHMARK_JOBS[job_id] = job_data

    response = BenchmarkJobResponse(
        job_id=job_id,
        name=req.name,
        status="queued",
        total_runs=total_runs,
        completed_runs=0,
        failed_runs=0,
        spend_usd=0.0,
    )
    background_tasks.add_task(_execute_benchmark_job, job_id, req, tasks)
    return response


@router.get("/v1/benchmark/jobs/{job_id}", response_model=BenchmarkJobResponse)
def get_benchmark_job(job_id: str) -> BenchmarkJobResponse:
    """Get status and spend progress of a benchmark job (API_SPEC.md)."""
    if job_id not in _BENCHMARK_JOBS:
        # Check if there are runs in DB with this job_id
        db = Database()
        db_runs = db.list_runs(job_id=job_id)
        if not db_runs:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": f"Job '{job_id}' not found"},
            )
        total = len(db_runs)
        completed = sum(1 for r in db_runs if r.status in ("completed", "halted_budget"))
        failed = sum(1 for r in db_runs if r.status in ("failed", "cancelled"))
        spend = sum(r.total_cost_usd for r in db_runs)
        return BenchmarkJobResponse(
            job_id=job_id,
            name="recovered_job",
            status="completed" if completed + failed >= total else "running",
            total_runs=total,
            completed_runs=completed,
            failed_runs=failed,
            spend_usd=round(spend, 4),
        )

    info = _BENCHMARK_JOBS[job_id]
    db = Database()
    runs = db.list_runs(job_id=job_id)
    completed = sum(1 for r in runs if r.status in ("completed", "halted_budget"))
    failed = sum(1 for r in runs if r.status in ("failed", "cancelled"))
    spend = sum(r.total_cost_usd for r in runs)

    if info["status"] in ("queued", "running"):
        info["completed_runs"] = completed
        info["failed_runs"] = failed
        info["spend_usd"] = round(spend, 4)
        if completed + failed >= info["total_runs"] and info["total_runs"] > 0:
            info["status"] = "failed" if failed else "completed"

    return BenchmarkJobResponse(**info)


@router.get("/v1/metrics/summary")
def get_metrics_summary(
    job_id: str | None = Query(None, description="Benchmark job ID to summarize"),
    reference_arm: str = Query("vanilla_fixed", description="Reference arm for Wilcoxon comparison"),
) -> dict[str, Any]:
    """Retrieve aggregate evaluation metrics table (FR-29, API_SPEC.md).

    Computes per-arm median quality, agent count, cost, tokens, compliance rate,
    detection recall, replacement counts and denials, supervision share,
    bootstrap 95% CIs, and paired Wilcoxon p-values against reference_arm.
    """
    db = Database()
    runs = db.list_runs(job_id=job_id)

    run_dicts: list[dict[str, Any]] = []
    for r in runs:
        if r.status not in ("completed", "halted_budget"):
            continue
        budget_dict = (
            json.loads(r.budget_json)
            if isinstance(r.budget_json, str)
            else r.budget_json
        )
        replacements = db.get_replacement_events(r.run_id)
        exec_swaps = sum(1 for re in replacements if re.outcome == "executed")
        denied_swaps = sum(1 for re in replacements if re.outcome.startswith("denied"))

        run_dicts.append({
            "run_id": r.run_id,
            "task_id": r.task_id,
            "job_id": r.job_id,
            "arm": r.arm,
            "seed": r.seed,
            "quality_composite": r.quality_composite,
            "total_cost_usd": r.total_cost_usd,
            "total_tokens": r.total_tokens,
            "wall_ms": r.wall_ms or 0,
            "supervision_tokens": r.supervision_tokens,
            "supervision_cost_usd": r.supervision_cost_usd,
            "compliant_all": r.compliant_all or 0,
            "replacements": exec_swaps,
            "replacements_denied": denied_swaps,
            "budget": budget_dict,
        })

    return compute_metrics_summary(
        run_dicts,
        reference_arm=reference_arm,
        n_resamples=10000,
    )
