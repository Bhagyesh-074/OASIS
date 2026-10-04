"""Pydantic schemas and serialization models for OASIS REST API (API_SPEC.md).

Conventions:
- All identifiers are ULIDs.
- All timestamps are UTC RFC 3339.
- Error envelope follows single pattern: {"error": {"code": ..., "message": ..., "details": ...}}.
- Budget strictly accepts exactly four dimensions: max_tokens, max_cost_usd, max_wall_seconds, max_calls (FR-5).
"""

from __future__ import annotations

import datetime
from typing import Any, Literal

import ulid
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =====================================================================
# Error Envelope
# =====================================================================


class ErrorDetail(BaseModel):
    """Detailed error object."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """Single unified error envelope across all endpoints (API_SPEC.md)."""

    error: ErrorDetail


# =====================================================================
# Budget & Team Configuration
# =====================================================================


class BudgetModel(BaseModel):
    """Budget over exactly four dimensions (FR-5, ADR-002).

    extra='forbid' ensures GPU, RAM, or unknown dimensions are strictly rejected.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0, description="Max total tokens")
    max_cost_usd: float = Field(gt=0.0, description="Max dollar cost")
    max_wall_seconds: float = Field(gt=0.0, description="Max wall-clock seconds")
    max_calls: int = Field(gt=0, description="Max provider calls")


class RoleConfigModel(BaseModel):
    """Specification of an agent role in a team configuration."""

    role: str
    template_id: str = "fast_cheap"
    model: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.2
    est_tokens: int = 1000
    est_cost_usd: float = 0.001
    est_wall_seconds: float = 5.0
    est_calls: int = 1


class TeamConfigModel(BaseModel):
    """Team configuration returned by plan/estimator."""

    config_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    team_size: int
    origin: Literal["estimator", "memory", "baseline", "oracle"] = "estimator"
    topology: Literal["linear", "hub_and_spoke", "parallel"] = "hub_and_spoke"
    roles: list[RoleConfigModel] = Field(default_factory=list)
    justification: str = ""


# =====================================================================
# Estimation and Planning
# =====================================================================


class EstimateRequest(BaseModel):
    """Request for complexity estimation (POST /v1/estimate)."""

    statement: str
    domain: Literal[
        "code_generation",
        "research_qa",
        "quant_analysis",
        "support_triage",
        "content_generation",
    ]
    estimator: Literal["heuristic", "llm_planner"] = "heuristic"


class EstimateResponse(BaseModel):
    """Response from complexity estimation (POST /v1/estimate)."""

    mvts: int
    subscores: dict[str, float]
    weights: dict[str, float]
    subtasks: list[str] = Field(default_factory=list)
    justification: str
    latency_ms: float = 0.0


class PlanRequest(BaseModel):
    """Request for team planning with memory & pre-execution enforcement."""

    statement: str
    domain: Literal[
        "code_generation",
        "research_qa",
        "quant_analysis",
        "support_triage",
        "content_generation",
    ]
    budget: BudgetModel
    use_memory: bool = True
    framework: Literal["langgraph", "autogen", "crewai"] = "langgraph"
    pareto: bool = False


class MemoryHitModel(BaseModel):
    """Matched record from Configuration Memory."""

    mem_id: str
    similarity: float
    prior_quality: float


class ReductionRecordModel(BaseModel):
    """Record of role merge/prune forced by a budget dimension."""

    from_size: int
    to_size: int
    binding: Literal["tokens", "cost", "wall", "calls"]
    merged: list[str]
    justification: str


class EnforcementProjectionModel(BaseModel):
    """Cost/resource projection after enforcement."""

    tokens: int
    cost_usd: float
    wall_seconds: float
    calls: int


class EnforcementResultModel(BaseModel):
    """Result of RBE pre-execution reduction."""

    feasible: bool
    reductions: list[ReductionRecordModel] = Field(default_factory=list)
    projection: EnforcementProjectionModel


class PlanResponse(BaseModel):
    """Response from POST /v1/plan."""

    config: TeamConfigModel
    memory_hit: MemoryHitModel | None = None
    enforcement: EnforcementResultModel
    budget_frontier: list[dict[str, Any]] | None = None


# =====================================================================
# Runs
# =====================================================================

VALID_ARMS = (
    "vanilla_fixed",
    "single_agent",
    "captainagent",
    "tce_only",
    "rbe_only",
    "monitor_only",
    "tce_rbe",
    "full",
    "oracle",
)


class InjectionConfigModel(BaseModel):
    """Configuration for failure injection (FR-28)."""

    enabled: bool = False
    rate: float = 0.0
    types: list[str] = Field(default_factory=list)


class RunCreateRequest(BaseModel):
    """Request to initiate a run (POST /v1/runs)."""

    budget: BudgetModel
    task_id: str | None = None
    statement: str | None = None
    domain: (
        Literal[
            "code_generation",
            "research_qa",
            "quant_analysis",
            "support_triage",
            "content_generation",
        ]
        | None
    ) = None
    arm: str = "full"
    framework: Literal["langgraph", "autogen", "crewai"] = "langgraph"
    seed: int = 1
    mode: Literal["live", "replay"] = "live"
    injection: InjectionConfigModel = Field(default_factory=InjectionConfigModel)

    @field_validator("arm")
    @classmethod
    def validate_arm(cls, v: str) -> str:
        if v not in VALID_ARMS:
            raise ValueError(f"Invalid arm: {v}. Must be one of {VALID_ARMS}")
        return v


class RunCreateResponse(BaseModel):
    """Immediate 202 Accepted response for run creation."""

    run_id: str
    status: Literal["queued", "running"] = "queued"
    events_url: str


class RunTotalsModel(BaseModel):
    """Final run resource consumption totals."""

    tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    calls: int = 0
    supervision_tokens: int = 0
    supervision_share: float = 0.0


class RunComplianceModel(BaseModel):
    """Compliance boolean flags against declared budget dimensions."""

    tokens: bool = True
    cost: bool = True
    wall: bool = True
    calls: bool = True
    all: bool = True


class RunQualityModel(BaseModel):
    """Multi-layer quality evaluations."""

    composite: float | None = None
    l3_verifier: float | None = None
    l2_judge: float | None = None
    judge_model: str = "gpt-4o-mini-2024-07-18"
    rubric_version: str = "r3"


class RunDetailResponse(BaseModel):
    """Detailed run state (GET /v1/runs/{id})."""

    run_id: str
    status: Literal[
        "queued", "running", "completed", "halted_budget", "failed", "cancelled"
    ]
    status_detail: str | None = None
    arm: str
    seed: int
    framework: str
    config: TeamConfigModel | None = None
    totals: RunTotalsModel
    compliance: RunComplianceModel
    quality: RunQualityModel
    replacements: int = 0
    replacements_denied: int = 0
    final_output_ref: str | None = None


class DecisionRecordResponse(BaseModel):
    """Single decision log entry in GET /v1/runs/{id}/decisions."""

    decision_id: str
    ts: str
    component: str
    decision: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    justification: str


class DecisionListResponse(BaseModel):
    """Full decision list response for a run."""

    decisions: list[DecisionRecordResponse]


class CancelRunResponse(BaseModel):
    """Response after cancelling a run."""

    run_id: str
    status: Literal["cancelled"] = "cancelled"


# =====================================================================
# Memory, Templates, Benchmark & Ops
# =====================================================================


class MemorySearchItem(BaseModel):
    """Memory search result item."""

    mem_id: str
    domain: str
    statement: str
    similarity: float
    prior_quality: float
    prior_cost_usd: float
    config: dict[str, Any]


class TemplateItem(BaseModel):
    """Agent template specification from Agent Template Library."""

    template_id: str
    model: str
    temperature: float
    system_prompt_hash: str = ""
    mean_historical_cost: float = 0.001


class BenchmarkJobRequest(BaseModel):
    """Launch benchmark matrix job (POST /v1/benchmark/jobs)."""

    name: str
    task_split: Literal["calibration", "eval"] = "eval"
    arms: list[str] = Field(default_factory=lambda: list(VALID_ARMS[:-1]))
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    budget_profile: Literal["tight", "moderate", "generous"] = "moderate"
    concurrency: int = 8
    mode: Literal["live", "replay"] = "live"


class BenchmarkJobResponse(BaseModel):
    """Job creation acknowledgement."""

    job_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    name: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    total_runs: int
    completed_runs: int = 0
    failed_runs: int = 0
    spend_usd: float = 0.0
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class HealthResponse(BaseModel):
    """Health readiness check (/healthz)."""

    status: Literal["ok", "degraded", "down"]
    components: dict[str, bool]
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class VersionResponse(BaseModel):
    """Version metadata (/v1/version)."""

    version: str = "0.1.0"
    git_commit: str
    config_hashes: dict[str, str]
    pinned_models: dict[str, str]


class SpendResponse(BaseModel):
    """Cumulative spend against global ceiling (/v1/spend)."""

    cumulative_spend_usd: float
    ceiling_usd: float
    kill_multiplier: float
    kill_ceiling_usd: float
    ceiling_exceeded: bool
