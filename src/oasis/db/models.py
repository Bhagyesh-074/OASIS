"""Typed accessors and repository over the SQLite schema in docs/DATABASE.md.

Implements all 13 tables:
task, run, team_config, agent_instance, agent_output, score, budget_event,
replacement_event, config_memory, decision_log, injected_failure, oracle_sweep,
judge_validation.

Operational constraints (docs/DATABASE.md):
- WAL mode, foreign_keys=ON, busy_timeout=5000.
- Serialized writes via a threading lock / queue for safe multi-worker execution.
- Score layers (L1, L2, L3) are never collapsed before storage (FR-15).
- Every decision has a non-empty justification string (FR-24).
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import ulid
from pydantic import BaseModel, ConfigDict, Field

from oasis.db.migrate import get_db_path, init_connection

# =====================================================================
# Typed Records (Pydantic models)
# =====================================================================


class TaskRecord(BaseModel):
    """Benchmark task definition (docs/DATABASE.md)."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str
    domain: str
    source: str
    statement: str
    split: str
    source_ref: str | None = None
    reference_answer: str | None = None
    verifier_type: str | None = None
    verifier_spec: str | dict[str, Any] | None = None
    complexity_label: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class RunRecord(BaseModel):
    """Run tracking record (docs/DATABASE.md)."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    task_id: str
    arm: str
    seed: int
    framework: str
    mode: str
    budget_json: str | dict[str, Any]
    status: str = "queued"
    job_id: str | None = None
    status_detail: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    wall_ms: int | None = None
    supervision_tokens: int = 0
    supervision_cost_usd: float = 0.0
    orchestration_ms: int | None = None
    quality_composite: float | None = None
    compliant_all: int | None = None
    manifest_json: str | dict[str, Any] = "{}"


class TeamConfigRecord(BaseModel):
    """Configured team composition for a run."""

    model_config = ConfigDict(from_attributes=True)

    config_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    team_size: int
    origin: str
    topology: str
    roles_json: str | list[dict[str, Any]]
    mvts_proposed: int | None = None
    subscores_json: str | dict[str, Any] | None = None
    memory_mem_id: str | None = None
    memory_similarity: float | None = None


class AgentInstanceRecord(BaseModel):
    """Active or swapped agent in a run."""

    model_config = ConfigDict(from_attributes=True)

    agent_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    role: str
    template_id: str
    model: str
    temperature: float
    active_from_step: int
    active_to_step: int | None = None
    replaces_agent_id: str | None = None


class AgentOutputRecord(BaseModel):
    """Output message produced by an agent."""

    model_config = ConfigDict(from_attributes=True)

    output_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    agent_id: str
    step: int
    content: str
    content_sha: str
    prompt_sha: str
    purpose: str = "productive"
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


class ScoreRecord(BaseModel):
    """Multi-layer evaluation score (L1/L2/L3 stored separately per FR-15)."""

    model_config = ConfigDict(from_attributes=True)

    score_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    output_id: str
    layer: str
    relevance: float | None = None
    factuality: float | None = None
    adherence: float | None = None
    completeness: float | None = None
    efficiency: float | None = None
    composite: float | None = None
    scorer_model: str | None = None
    rubric_version: str | None = None
    cost_usd: float = 0.0
    computed_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class BudgetEventRecord(BaseModel):
    """Runtime budget admission or reduction event."""

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    dimension: str
    action: str
    justification: str
    step: int | None = None
    projected: float | None = None
    remaining: float | None = None
    model_before: str | None = None
    model_after: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class ReplacementEventRecord(BaseModel):
    """Agent replacement execution or denial event."""

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    step: int
    out_agent_id: str
    outcome: str
    trigger_dimension: str
    justification: str
    in_agent_id: str | None = None
    trigger_value: float | None = None
    threshold: float | None = None
    window_size: int | None = None
    handoff_tokens: int | None = None
    handoff_cost_usd: float | None = None
    quality_before: float | None = None
    quality_after: float | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class ConfigMemoryRecord(BaseModel):
    """Stored validated configuration in Configuration Memory."""

    model_config = ConfigDict(from_attributes=True)

    mem_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    source_run_id: str
    domain: str
    statement: str
    embedding: bytes
    embed_model: str
    complexity_json: str | dict[str, Any]
    budget_json: str | dict[str, Any]
    config_json: str | dict[str, Any]
    quality: float
    cost_usd: float
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class DecisionLogRecord(BaseModel):
    """XAI Audit decision record (FR-24). Non-empty justification strictly required."""

    model_config = ConfigDict(from_attributes=True)

    decision_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    component: str
    decision: str
    inputs_json: str | dict[str, Any]
    justification: str
    run_id: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class InjectedFailureRecord(BaseModel):
    """Ground truth for injected failure detection (FR-28)."""

    model_config = ConfigDict(from_attributes=True)

    inj_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    run_id: str
    agent_id: str
    step: int
    failure_type: str
    severity: str
    output_id: str | None = None
    detected: int = 0
    detected_at_step: int | None = None
    detected_by_layer: str | None = None


class OracleSweepRecord(BaseModel):
    """Oracle team-size sweep point (ADR-011)."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str
    k: int
    seed: int
    quality: float
    cost_usd: float
    tokens: int | None = None
    wall_ms: int | None = None


class JudgeValidationRecord(BaseModel):
    """Human-judge validation point for Cohen's kappa (ADR-013)."""

    model_config = ConfigDict(from_attributes=True)

    jv_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    output_id: str
    domain: str
    human_a: float
    human_b: float
    human_a_id: str
    human_b_id: str
    judge_score: float
    judge_model: str
    rubric_version: str
    labelled_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


# =====================================================================
# Database Repository & Writer Management
# =====================================================================


def _to_json_str(val: Any) -> str:
    """Normalize dict, list, or string to JSON string."""
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val)


class Database:
    """Thread-safe SQLite repository with WAL mode and serialized writes."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = get_db_path(db_path)
        self._write_lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and WAL mode."""
        return init_connection(self.db_path)

    # -----------------------------------------------------------------
    # Tasks
    # -----------------------------------------------------------------

    def insert_task(self, task: TaskRecord) -> None:
        """Insert or replace a benchmark task."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task (
                    task_id, domain, source, source_ref, statement, reference_answer,
                    verifier_type, verifier_spec, complexity_label, split, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    task.task_id,
                    task.domain,
                    task.source,
                    task.source_ref,
                    task.statement,
                    task.reference_answer,
                    task.verifier_type,
                    _to_json_str(task.verifier_spec) if task.verifier_spec else None,
                    task.complexity_label,
                    task.split,
                    task.created_at,
                ),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Fetch task by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM task WHERE task_id = ?;", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return TaskRecord(**dict(row))

    def list_tasks(
        self, domain: str | None = None, split: str | None = None
    ) -> list[TaskRecord]:
        """List tasks with optional domain and split filters."""
        query = "SELECT * FROM task WHERE 1=1"
        params: list[Any] = []
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if split:
            query += " AND split = ?"
            params.append(split)
        query += " ORDER BY task_id ASC;"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [TaskRecord(**dict(row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Runs
    # -----------------------------------------------------------------

    def create_run(self, run: RunRecord) -> None:
        """Record a new run."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO run (
                    run_id, task_id, job_id, arm, seed, framework, mode,
                    budget_json, status, status_detail, started_at, ended_at,
                    total_tokens, total_cost_usd, total_calls, wall_ms,
                    supervision_tokens, supervision_cost_usd, orchestration_ms,
                    quality_composite, compliant_all, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run.run_id,
                    run.task_id,
                    run.job_id,
                    run.arm,
                    run.seed,
                    run.framework,
                    run.mode,
                    _to_json_str(run.budget_json),
                    run.status,
                    run.status_detail,
                    run.started_at,
                    run.ended_at,
                    run.total_tokens,
                    run.total_cost_usd,
                    run.total_calls,
                    run.wall_ms,
                    run.supervision_tokens,
                    run.supervision_cost_usd,
                    run.orchestration_ms,
                    run.quality_composite,
                    run.compliant_all,
                    _to_json_str(run.manifest_json),
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        """Fetch run by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM run WHERE run_id = ?;", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return RunRecord(**dict(row))

    def update_run_status(
        self,
        run_id: str,
        status: str,
        status_detail: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        """Update run status and optional status detail / completion timestamp."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                UPDATE run
                SET status = ?,
                    status_detail = COALESCE(?, status_detail),
                    ended_at = COALESCE(?, ended_at)
                WHERE run_id = ?;
                """,
                (status, status_detail, ended_at, run_id),
            )

    def finalize_run(
        self,
        run_id: str,
        *,
        totals: dict[str, Any],
        compliant_all: bool,
        quality_composite: float | None = None,
        status: str = "completed",
        status_detail: str | None = None,
        wall_ms: int | None = None,
        orchestration_ms: int | None = None,
    ) -> None:
        """Finalize run with final totals, compliance status, and quality score."""
        now_utc = datetime.datetime.now(datetime.UTC).isoformat()
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                UPDATE run
                SET status = ?,
                    status_detail = ?,
                    ended_at = ?,
                    total_tokens = ?,
                    total_cost_usd = ?,
                    total_calls = ?,
                    wall_ms = ?,
                    supervision_tokens = ?,
                    supervision_cost_usd = ?,
                    orchestration_ms = ?,
                    quality_composite = ?,
                    compliant_all = ?
                WHERE run_id = ?;
                """,
                (
                    status,
                    status_detail,
                    now_utc,
                    totals.get("tokens", 0),
                    totals.get("cost_usd", 0.0),
                    totals.get("calls", 0),
                    wall_ms,
                    totals.get("supervision_tokens", 0),
                    totals.get("supervision_cost_usd", 0.0),
                    orchestration_ms,
                    quality_composite,
                    1 if compliant_all else 0,
                    run_id,
                ),
            )

    def list_runs(
        self, job_id: str | None = None, arm: str | None = None
    ) -> list[RunRecord]:
        """List runs with optional job_id and arm filters."""
        query = "SELECT * FROM run WHERE 1=1"
        params: list[Any] = []
        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        if arm:
            query += " AND arm = ?"
            params.append(arm)
        query += " ORDER BY started_at DESC;"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [RunRecord(**dict(row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Team Configuration & Agent Instances
    # -----------------------------------------------------------------

    def insert_team_config(self, config: TeamConfigRecord) -> None:
        """Insert team configuration record."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO team_config (
                    config_id, run_id, team_size, origin, topology, roles_json,
                    mvts_proposed, subscores_json, memory_mem_id, memory_similarity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    config.config_id,
                    config.run_id,
                    config.team_size,
                    config.origin,
                    config.topology,
                    _to_json_str(config.roles_json),
                    config.mvts_proposed,
                    _to_json_str(config.subscores_json)
                    if config.subscores_json
                    else None,
                    config.memory_mem_id,
                    config.memory_similarity,
                ),
            )

    def get_team_config(self, run_id: str) -> TeamConfigRecord | None:
        """Fetch team configuration for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM team_config WHERE run_id = ?;", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return TeamConfigRecord(**dict(row))

    def insert_agent_instance(self, agent: AgentInstanceRecord) -> None:
        """Insert agent instance record."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_instance (
                    agent_id, run_id, role, template_id, model, temperature,
                    active_from_step, active_to_step, replaces_agent_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    agent.agent_id,
                    agent.run_id,
                    agent.role,
                    agent.template_id,
                    agent.model,
                    agent.temperature,
                    agent.active_from_step,
                    agent.active_to_step,
                    agent.replaces_agent_id,
                ),
            )

    def deactivate_agent_instance(self, agent_id: str, active_to_step: int) -> None:
        """Mark an agent instance as deactivated at step."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                "UPDATE agent_instance SET active_to_step = ? WHERE agent_id = ?;",
                (active_to_step, agent_id),
            )

    def get_agent_instances(self, run_id: str) -> list[AgentInstanceRecord]:
        """List all agent instances for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_instance WHERE run_id = ? ORDER BY active_from_step ASC;",
                (run_id,),
            )
            return [AgentInstanceRecord(**dict(row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Outputs and Separate Score Layers (FR-15)
    # -----------------------------------------------------------------

    def insert_agent_output(self, output: AgentOutputRecord) -> None:
        """Record an agent's output."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_output (
                    output_id, run_id, agent_id, step, content, content_sha,
                    prompt_sha, purpose, tokens_in, tokens_out, cost_usd, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    output.output_id,
                    output.run_id,
                    output.agent_id,
                    output.step,
                    output.content,
                    output.content_sha,
                    output.prompt_sha,
                    output.purpose,
                    output.tokens_in,
                    output.tokens_out,
                    output.cost_usd,
                    output.latency_ms,
                ),
            )

    def get_agent_outputs(self, run_id: str) -> list[AgentOutputRecord]:
        """Fetch all outputs for a run ordered by step."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_output WHERE run_id = ? ORDER BY step ASC;",
                (run_id,),
            )
            return [AgentOutputRecord(**dict(row)) for row in cursor.fetchall()]

    def insert_score(self, score: ScoreRecord) -> None:
        """Insert a score layer. Never collapses layers (FR-15)."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO score (
                    score_id, output_id, layer, relevance, factuality, adherence,
                    completeness, efficiency, composite, scorer_model, rubric_version,
                    cost_usd, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    score.score_id,
                    score.output_id,
                    score.layer,
                    score.relevance,
                    score.factuality,
                    score.adherence,
                    score.completeness,
                    score.efficiency,
                    score.composite,
                    score.scorer_model,
                    score.rubric_version,
                    score.cost_usd,
                    score.computed_at,
                ),
            )

    def get_scores_for_output(self, output_id: str) -> list[ScoreRecord]:
        """Get all separate score layer records for a given output."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM score WHERE output_id = ?;", (output_id,))
            return [ScoreRecord(**dict(row)) for row in cursor.fetchall()]

    def get_scores_for_run(self, run_id: str) -> list[ScoreRecord]:
        """Get all score records across all outputs of a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.* FROM score s
                JOIN agent_output o ON s.output_id = o.output_id
                WHERE o.run_id = ?
                ORDER BY o.step ASC, s.layer ASC;
                """,
                (run_id,),
            )
            return [ScoreRecord(**dict(row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Budget Events & Replacements
    # -----------------------------------------------------------------

    def insert_budget_event(self, event: BudgetEventRecord) -> None:
        """Record an admission check or reduction event."""
        if not event.justification or not event.justification.strip():
            raise ValueError("FR-24 Invariant: justification string cannot be empty")
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO budget_event (
                    event_id, run_id, step, dimension, action, projected, remaining,
                    model_before, model_after, justification, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.step,
                    event.dimension,
                    event.action,
                    event.projected,
                    event.remaining,
                    event.model_before,
                    event.model_after,
                    event.justification,
                    event.created_at,
                ),
            )

    def get_budget_events(self, run_id: str) -> list[BudgetEventRecord]:
        """Get all budget events for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM budget_event WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [BudgetEventRecord(**dict(row)) for row in cursor.fetchall()]

    def insert_replacement_event(self, event: ReplacementEventRecord) -> None:
        """Record a replacement event (executed or denied)."""
        if not event.justification or not event.justification.strip():
            raise ValueError("FR-24 Invariant: justification string cannot be empty")
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO replacement_event (
                    event_id, run_id, step, out_agent_id, in_agent_id, outcome,
                    trigger_dimension, trigger_value, threshold, window_size,
                    handoff_tokens, handoff_cost_usd, quality_before, quality_after,
                    justification, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.step,
                    event.out_agent_id,
                    event.in_agent_id,
                    event.outcome,
                    event.trigger_dimension,
                    event.trigger_value,
                    event.threshold,
                    event.window_size,
                    event.handoff_tokens,
                    event.handoff_cost_usd,
                    event.quality_before,
                    event.quality_after,
                    event.justification,
                    event.created_at,
                ),
            )

    def get_replacement_events(self, run_id: str) -> list[ReplacementEventRecord]:
        """Get all replacement events for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM replacement_event WHERE run_id = ? ORDER BY step ASC;",
                (run_id,),
            )
            return [ReplacementEventRecord(**dict(row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Decision Log (FR-24)
    # -----------------------------------------------------------------

    def insert_decision(self, decision: DecisionLogRecord) -> None:
        """Insert an automated decision record (FR-24)."""
        if not decision.justification or not decision.justification.strip():
            raise ValueError("FR-24 Invariant: justification string cannot be empty")
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO decision_log (
                    decision_id, run_id, component, decision, inputs_json,
                    justification, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    decision.decision_id,
                    decision.run_id,
                    decision.component,
                    decision.decision,
                    _to_json_str(decision.inputs_json),
                    decision.justification,
                    decision.created_at,
                ),
            )

    def get_decisions(self, run_id: str) -> list[DecisionLogRecord]:
        """Get the full XAI decision trail for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM decision_log WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [DecisionLogRecord(**dict(row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Configuration Memory (FR-21..FR-23)
    # -----------------------------------------------------------------

    def insert_config_memory(self, memory: ConfigMemoryRecord) -> None:
        """Store validated configuration into memory."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO config_memory (
                    mem_id, source_run_id, domain, statement, embedding,
                    embed_model, complexity_json, budget_json, config_json,
                    quality, cost_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    memory.mem_id,
                    memory.source_run_id,
                    memory.domain,
                    memory.statement,
                    memory.embedding,
                    memory.embed_model,
                    _to_json_str(memory.complexity_json),
                    _to_json_str(memory.budget_json),
                    _to_json_str(memory.config_json),
                    memory.quality,
                    memory.cost_usd,
                    memory.created_at,
                ),
            )

    def list_config_memory(
        self, domain: str | None = None, exclude_run_id: str | None = None
    ) -> list[ConfigMemoryRecord]:
        """Retrieve memory entries, optionally filtering by domain or excluding a run."""
        query = "SELECT * FROM config_memory WHERE 1=1"
        params: list[Any] = []
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if exclude_run_id:
            query += " AND source_run_id != ?"
            params.append(exclude_run_id)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [ConfigMemoryRecord(**dict(row)) for row in cursor.fetchall()]

    def delete_config_memory(self) -> int:
        """Reset memory store (FR-23 / API_SPEC DELETE /v1/memory)."""
        with self._write_lock, self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM config_memory;")
            return cursor.rowcount

    # -----------------------------------------------------------------
    # Injected Failures (FR-28) & Oracle Sweep (ADR-011)
    # -----------------------------------------------------------------

    def insert_injected_failure(self, inj: InjectedFailureRecord) -> None:
        """Record an injected failure ground truth item."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO injected_failure (
                    inj_id, run_id, output_id, agent_id, step, failure_type,
                    severity, detected, detected_at_step, detected_by_layer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    inj.inj_id,
                    inj.run_id,
                    inj.output_id,
                    inj.agent_id,
                    inj.step,
                    inj.failure_type,
                    inj.severity,
                    inj.detected,
                    inj.detected_at_step,
                    inj.detected_by_layer,
                ),
            )

    def update_injected_failure_detection(
        self,
        inj_id: str,
        detected: bool,
        detected_at_step: int,
        detected_by_layer: str,
    ) -> None:
        """Record detection outcome for an injected failure."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                UPDATE injected_failure
                SET detected = ?, detected_at_step = ?, detected_by_layer = ?
                WHERE inj_id = ?;
                """,
                (1 if detected else 0, detected_at_step, detected_by_layer, inj_id),
            )

    def get_injected_failures(self, run_id: str) -> list[InjectedFailureRecord]:
        """Fetch all injected failures for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM injected_failure WHERE run_id = ? ORDER BY step ASC;",
                (run_id,),
            )
            return [InjectedFailureRecord(**dict(row)) for row in cursor.fetchall()]

    def insert_oracle_sweep(self, sweep: OracleSweepRecord) -> None:
        """Insert oracle sweep ground-truth point."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO oracle_sweep (
                    task_id, k, seed, quality, cost_usd, tokens, wall_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    sweep.task_id,
                    sweep.k,
                    sweep.seed,
                    sweep.quality,
                    sweep.cost_usd,
                    sweep.tokens,
                    sweep.wall_ms,
                ),
            )

    def insert_judge_validation(self, jv: JudgeValidationRecord) -> None:
        """Insert human-judge validation score."""
        with self._write_lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO judge_validation (
                    jv_id, output_id, domain, human_a, human_b,
                    human_a_id, human_b_id, judge_score, judge_model,
                    rubric_version, labelled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    jv.jv_id,
                    jv.output_id,
                    jv.domain,
                    jv.human_a,
                    jv.human_b,
                    jv.human_a_id,
                    jv.human_b_id,
                    jv.judge_score,
                    jv.judge_model,
                    jv.rubric_version,
                    jv.labelled_at,
                ),
            )

    # -----------------------------------------------------------------
    # Global Spend Guard (NFR-3) & Aggregate Metrics
    # -----------------------------------------------------------------

    def get_cumulative_spend(self) -> float:
        """Sum total_cost_usd across all completed or halted runs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(SUM(total_cost_usd), 0.0) FROM run;")
            row = cursor.fetchone()
            return float(row[0]) if row else 0.0

    def get_detection_recall(self) -> dict[tuple[str, str], float]:
        """Compute detection recall disaggregated by failure_type and severity.

        Never collapsed into a single aggregate (TESTING.md detection recall validity).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT failure_type, severity,
                       CAST(SUM(detected) AS FLOAT) / COUNT(*) as recall
                FROM injected_failure
                GROUP BY failure_type, severity;
                """
            )
            return {(row[0], row[1]): float(row[2]) for row in cursor.fetchall()}

    def get_replacement_suppression_rate(self, run_id: str | None = None) -> float:
        """Compute replacement suppression rate: denied events / total replacement attempts."""
        query = "SELECT outcome FROM replacement_event WHERE 1=1"
        params: list[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            outcomes = [row[0] for row in cursor.fetchall()]
            if not outcomes:
                return 0.0
            denied_count = sum(1 for o in outcomes if o.startswith("denied"))
            return denied_count / len(outcomes)
