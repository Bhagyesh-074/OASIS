"""Run lifecycle, streaming events (SSE), XAI decisions, and cancellation (API_SPEC.md)."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

import ulid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from oasis.api.schemas import (
    CancelRunResponse,
    DecisionListResponse,
    DecisionRecordResponse,
    RoleConfigModel,
    RunComplianceModel,
    RunCreateRequest,
    RunCreateResponse,
    RunDetailResponse,
    RunQualityModel,
    RunTotalsModel,
    TeamConfigModel,
)
from oasis.db.models import Database, RunRecord, TaskRecord

router = APIRouter(tags=["Runs"])

# In-memory pub/sub queues for live SSE broadcasting: run_id -> list of asyncio.Queue
_RUN_EVENT_SUBSCRIBERS: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)


def broadcast_run_event(run_id: str, event_type: str, data: dict[str, Any]) -> None:
    """Broadcast an execution event to all active SSE subscribers."""
    payload = {"event": event_type, "data": data}
    queues = _RUN_EVENT_SUBSCRIBERS.get(run_id, [])
    for q in queues:
        q.put_nowait(payload)


@router.post("/v1/runs", status_code=202, response_model=RunCreateResponse)
def create_run(req: RunCreateRequest) -> RunCreateResponse:
    """Start or enqueue a run. Returns immediately with 202 Accepted (API_SPEC.md)."""
    db = Database()

    # Check global spend ceiling (NFR-3)
    ceiling = float(os.environ.get("OASIS_SPEND_CEILING_USD", "250.0"))
    if db.get_cumulative_spend() >= ceiling:
        raise HTTPException(
            status_code=429,
            detail={"code": "global_spend_limit", "message": "Global API spend ceiling reached"},
        )

    # Resolve or create task record
    task_id = req.task_id
    if not task_id:
        if not req.statement or not req.domain:
            raise HTTPException(
                status_code=422,
                detail={"code": "validation_error", "message": "Either task_id or (statement + domain) must be provided"},
            )
        task_id = f"custom_{ulid.ULID()}"
        task_rec = TaskRecord(
            task_id=task_id,
            domain=req.domain,
            source="authored",
            statement=req.statement,
            split="eval",
        )
        db.insert_task(task_rec)
    else:
        # Check task existence
        existing_task = db.get_task(task_id)
        if not existing_task:
            if req.statement and req.domain:
                db.insert_task(
                    TaskRecord(
                        task_id=task_id,
                        domain=req.domain,
                        source="authored",
                        statement=req.statement,
                        split="eval",
                    )
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "unknown_task", "message": f"Task '{task_id}' not found"},
                )

    run_id = str(ulid.ULID())
    budget_dict = req.budget.model_dump()

    run_rec = RunRecord(
        run_id=run_id,
        task_id=task_id,
        arm=req.arm,
        seed=req.seed,
        framework=req.framework,
        mode=req.mode,
        budget_json=budget_dict,
        status="queued",
        started_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    db.create_run(run_rec)

    return RunCreateResponse(
        run_id=run_id,
        status="queued",
        events_url=f"/v1/runs/{run_id}/events",
    )


@router.get("/v1/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: str) -> RunDetailResponse:
    """Get full state and metrics for a run (API_SPEC.md)."""
    db = Database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"code": "run_not_found", "message": f"Run '{run_id}' not found"},
        )

    # Retrieve associated records
    tc = db.get_team_config(run_id)
    replacements = db.get_replacement_events(run_id)
    executed_swaps = sum(1 for r in replacements if r.outcome == "executed")
    denied_swaps = sum(1 for r in replacements if r.outcome.startswith("denied"))

    # Parse budget
    budget_dict = (
        json.loads(run.budget_json)
        if isinstance(run.budget_json, str)
        else run.budget_json
    )

    # Compliance
    max_tokens = budget_dict.get("max_tokens", float("inf"))
    max_cost = budget_dict.get("max_cost_usd", float("inf"))
    max_wall = budget_dict.get("max_wall_seconds", float("inf"))
    max_calls = budget_dict.get("max_calls", float("inf"))

    tokens_ok = run.total_tokens <= max_tokens
    cost_ok = run.total_cost_usd <= max_cost
    wall_sec = (run.wall_ms or 0) / 1000.0
    wall_ok = wall_sec <= max_wall
    calls_ok = run.total_calls <= max_calls
    all_ok = tokens_ok and cost_ok and wall_ok and calls_ok

    sup_share = (
        (run.supervision_tokens / run.total_tokens)
        if run.total_tokens > 0
        else 0.0
    )

    team_config_model: TeamConfigModel | None = None
    if tc:
        roles_data = (
            json.loads(tc.roles_json)
            if isinstance(tc.roles_json, str)
            else tc.roles_json
        )
        team_config_model = TeamConfigModel(
            config_id=tc.config_id,
            team_size=tc.team_size,
            origin=tc.origin,  # type: ignore[arg-type]
            topology=tc.topology,  # type: ignore[arg-type]
            roles=[RoleConfigModel(**r) for r in roles_data],
            justification="",
        )

    status_literal = (
        run.status
        if run.status in ("queued", "running", "completed", "halted_budget", "failed", "cancelled")
        else "completed"
    )

    return RunDetailResponse(
        run_id=run.run_id,
        status=status_literal,  # type: ignore[arg-type]
        status_detail=run.status_detail,
        arm=run.arm,
        seed=run.seed,
        framework=run.framework,
        config=team_config_model,
        totals=RunTotalsModel(
            tokens=run.total_tokens,
            cost_usd=round(run.total_cost_usd, 4),
            wall_seconds=round(wall_sec, 2),
            calls=run.total_calls,
            supervision_tokens=run.supervision_tokens,
            supervision_share=round(sup_share, 4),
        ),
        compliance=RunComplianceModel(
            tokens=tokens_ok,
            cost=cost_ok,
            wall=wall_ok,
            calls=calls_ok,
            all=all_ok,
        ),
        quality=RunQualityModel(
            composite=run.quality_composite,
        ),
        replacements=executed_swaps,
        replacements_denied=denied_swaps,
        final_output_ref=f"artifacts/{run.run_id}.md",
    )


@router.get("/v1/runs/{run_id}/decisions", response_model=DecisionListResponse)
def get_run_decisions(run_id: str) -> DecisionListResponse:
    """Retrieve full XAI decision trail for a run (FR-24, API_SPEC.md)."""
    db = Database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"code": "run_not_found", "message": f"Run '{run_id}' not found"},
        )

    records = db.get_decisions(run_id)
    decisions = []
    for r in records:
        inputs_dict = (
            json.loads(r.inputs_json)
            if isinstance(r.inputs_json, str)
            else r.inputs_json
        )
        decisions.append(
            DecisionRecordResponse(
                decision_id=r.decision_id,
                ts=r.created_at,
                component=r.component,
                decision=r.decision,
                inputs=inputs_dict if isinstance(inputs_dict, dict) else {},
                justification=r.justification,
            )
        )

    return DecisionListResponse(decisions=decisions)


@router.post("/v1/runs/{run_id}/cancel", response_model=CancelRunResponse)
def cancel_run(run_id: str) -> CancelRunResponse:
    """Cancel a running or queued run (API_SPEC.md)."""
    db = Database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"code": "run_not_found", "message": f"Run '{run_id}' not found"},
        )

    db.update_run_status(run_id, status="cancelled", status_detail="Cancelled by user request")
    broadcast_run_event(run_id, "run_completed", {"run_id": run_id, "status": "cancelled"})
    return CancelRunResponse(run_id=run_id, status="cancelled")


@router.get("/v1/runs/{run_id}/events")
async def stream_run_events(run_id: str) -> StreamingResponse:
    """Server-Sent Events (SSE) stream for real-time run monitoring (FR-25)."""
    db = Database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"code": "run_not_found", "message": f"Run '{run_id}' not found"},
        )

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _RUN_EVENT_SUBSCRIBERS[run_id].append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. First replay existing recorded events for this run
            budget_events = db.get_budget_events(run_id)
            for be in budget_events:
                yield f"event: budget_event\ndata: {json.dumps({'event_id': be.event_id, 'action': be.action, 'dimension': be.dimension, 'justification': be.justification})}\n\n"

            replacements = db.get_replacement_events(run_id)
            for re in replacements:
                yield f"event: replacement\ndata: {json.dumps({'event_id': re.event_id, 'outcome': re.outcome, 'justification': re.justification})}\n\n"

            scores = db.get_scores_for_run(run_id)
            for sc in scores:
                yield f"event: score_computed\ndata: {json.dumps({'score_id': sc.score_id, 'layer': sc.layer, 'composite': sc.composite})}\n\n"

            # If already completed or cancelled, send termination event immediately
            current_run = db.get_run(run_id)
            if current_run and current_run.status in ("completed", "halted_budget", "failed", "cancelled"):
                yield f"event: run_completed\ndata: {json.dumps({'run_id': run_id, 'status': current_run.status})}\n\n"
                return

            # 2. Stream live events as they arrive
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20.0)
                    evt_name = msg.get("event", "message")
                    evt_data = json.dumps(msg.get("data", {}))
                    yield f"event: {evt_name}\ndata: {evt_data}\n\n"
                    if evt_name == "run_completed":
                        break
                except TimeoutError:
                    # Keep-alive heartbeat comment
                    yield ": keep-alive\n\n"
        finally:
            if queue in _RUN_EVENT_SUBSCRIBERS[run_id]:
                _RUN_EVENT_SUBSCRIBERS[run_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
