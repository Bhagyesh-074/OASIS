"""Estimation and team planning routes: /v1/estimate and /v1/plan (API_SPEC.md)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

import ulid
import yaml
from fastapi import APIRouter, HTTPException

from oasis.api.schemas import (
    EnforcementProjectionModel,
    EnforcementResultModel,
    EstimateRequest,
    EstimateResponse,
    MemoryHitModel,
    PlanRequest,
    PlanResponse,
    ReductionRecordModel,
    RoleConfigModel,
    TeamConfigModel,
)
from oasis.db.models import Database
from oasis.rbe.budget import Budget
from oasis.rbe.reducer import BudgetInfeasibleError, TeamReducer

router = APIRouter(tags=["Estimation & Planning"])

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"


def _load_complexity_config() -> dict[str, Any]:
    """Load weights and mvts_range from config/complexity.yaml."""
    cfg_file = CONFIG_DIR / "complexity.yaml"
    if cfg_file.exists():
        return yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    return {
        "weights": {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2},
        "mvts_range": [1, 8],
    }


def _estimate_mvts(statement: str, domain: str) -> tuple[int, dict[str, float], list[str], str]:
    """Compute MVTS and sub-scores (FR-1, FR-2)."""
    cfg = _load_complexity_config()
    weights = cfg.get("weights", {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2})
    mvts_min, mvts_max = cfg.get("mvts_range", [1, 8])

    # Clause / punctuation segmentation heuristic
    raw_parts = [p.strip() for p in statement.replace(";", ".").replace("\n", ".").split(".") if p.strip()]
    if not raw_parts:
        raw_parts = [statement.strip()]

    subtasks = raw_parts[:6]
    n_subtasks = max(1, len(subtasks))
    skill_clusters = max(1, min(n_subtasks, 4))
    dep_density = round(min(0.9, 0.15 * n_subtasks), 2)

    subscores = {
        "subtask_count": float(n_subtasks),
        "skill_clusters": float(skill_clusters),
        "dep_density": float(dep_density),
    }

    raw_mvts = (
        weights.get("subtask_count", 0.4) * n_subtasks
        + weights.get("skill_clusters", 0.4) * skill_clusters
        + weights.get("dep_density", 0.2) * (dep_density * 4.0)
    )
    mvts = max(mvts_min, min(mvts_max, round(raw_mvts)))
    justification = f"MVTS={mvts} from subtasks={n_subtasks}, skill_clusters={skill_clusters}, dep_density={dep_density}"

    return mvts, subscores, subtasks, justification


@router.post("/v1/estimate", response_model=EstimateResponse)
def estimate_complexity(req: EstimateRequest) -> EstimateResponse:
    """Complexity estimation only, no side effects (FR-1, FR-2, API_SPEC.md)."""
    t0 = time.perf_counter()
    mvts, subscores, subtasks, justification = _estimate_mvts(req.statement, req.domain)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    weights = _load_complexity_config().get(
        "weights", {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2}
    )

    return EstimateResponse(
        mvts=mvts,
        subscores=subscores,
        weights=weights,
        subtasks=subtasks,
        justification=justification,
        latency_ms=round(latency_ms, 2),
    )


@router.post("/v1/plan", response_model=PlanResponse)
def plan_team(req: PlanRequest) -> PlanResponse:
    """Memory lookup, estimation, and pre-execution budget reduction (FR-6, API_SPEC.md)."""
    db = Database()

    # Global spend ceiling check (NFR-3)
    ceiling = float(os.environ.get("OASIS_SPEND_CEILING_USD", "250.0"))
    if db.get_cumulative_spend() >= ceiling:
        raise HTTPException(
            status_code=429,
            detail={"code": "global_spend_limit", "message": "Global API spend ceiling reached"},
        )

    # 1. Check Configuration Memory
    memory_hit: MemoryHitModel | None = None
    seed_config: dict[str, Any] | None = None

    if req.use_memory:
        mem_records = db.list_config_memory(domain=req.domain)
        if mem_records:
            best = max(mem_records, key=lambda m: m.quality)
            if best.quality >= 0.80:
                memory_hit = MemoryHitModel(
                    mem_id=best.mem_id,
                    similarity=0.91,
                    prior_quality=best.quality,
                )
                seed_config = (
                    yaml.safe_load(best.config_json)
                    if isinstance(best.config_json, str)
                    else best.config_json
                )

    # 2. Team Generation (either from memory or TCE estimate)
    roles: list[dict[str, Any]] = []
    origin: Literal["estimator", "memory", "baseline", "oracle"] = (
        "memory" if seed_config else "estimator"
    )

    if seed_config and "roles" in seed_config:
        roles = list(seed_config["roles"])
        justification = f"Seeded from memory ({memory_hit.mem_id if memory_hit else ''})"
    else:
        mvts, _, _, justification = _estimate_mvts(req.statement, req.domain)
        role_names = ["researcher", "analyst", "planner", "executor", "reviewer", "synthesizer", "editor", "verifier"]
        for i in range(mvts):
            r_name = role_names[i % len(role_names)]
            roles.append({
                "role": r_name,
                "template_id": "fast_cheap",
                "model": "gpt-4o-mini-2024-07-18",
                "temperature": 0.2,
                "est_tokens": 20000,
                "est_cost_usd": 0.05,
                "est_wall_seconds": 30.0,
                "est_calls": 8,
            })

    # 3. RBE Pre-Execution Reduction (FR-6)
    rbe_budget = Budget(
        max_tokens=req.budget.max_tokens,
        max_cost_usd=req.budget.max_cost_usd,
        max_wall_seconds=req.budget.max_wall_seconds,
        max_calls=req.budget.max_calls,
    )
    reducer = TeamReducer()

    try:
        reduced_team, reduction_log = reducer.reduce(roles, rbe_budget)
    except BudgetInfeasibleError as err:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "budget_infeasible",
                "message": str(err),
                "details": {"binding": "budget"},
            },
        ) from err

    # 4. Assemble Final Response
    reductions: list[ReductionRecordModel] = []
    for r in reduction_log:
        reductions.append(
            ReductionRecordModel(
                from_size=r["from_size"],
                to_size=r["to_size"],
                binding=r["binding_constraint"],
                merged=r["merged_roles"],
                justification=r["justification"],
            )
        )

    projected = reducer.project_cost(reduced_team, rbe_budget)
    projection = EnforcementProjectionModel(
        tokens=int(projected["max_tokens"]),
        cost_usd=round(projected["max_cost_usd"], 4),
        wall_seconds=round(projected["max_wall_seconds"], 1),
        calls=int(projected["max_calls"]),
    )

    enforcement_result = EnforcementResultModel(
        feasible=True,
        reductions=reductions,
        projection=projection,
    )

    final_roles = [RoleConfigModel(**r) for r in reduced_team]
    config_model = TeamConfigModel(
        config_id=str(ulid.ULID()),
        team_size=len(reduced_team),
        origin=origin,
        topology="hub_and_spoke",
        roles=final_roles,
        justification=justification,
    )

    return PlanResponse(
        config=config_model,
        memory_hit=memory_hit,
        enforcement=enforcement_result,
    )
