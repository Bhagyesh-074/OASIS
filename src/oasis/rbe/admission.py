"""FR-7 / FR-8 — per-call admission control.

Every outbound model call shall pass an admission check returning one of
**allow**, **downgrade**, **truncate**, **halt** (FR-7).  No call may
bypass the check.

On *halt*, the run shall terminate with status ``halted_budget`` and
return the best output produced so far rather than raising an error
(FR-8).

Supervision calls are tagged ``purpose=supervision`` and counted against
the same budget as productive calls (FR-9, ADR-006).
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from oasis.rbe.budget import BudgetDimension, BudgetState
from oasis.rbe.decision_log_client import log_decision

# ------------------------------------------------------------------
# Dimension and Action constants matching DATABASE.md
# ------------------------------------------------------------------

DIMENSION_MAP: dict[str, str] = {
    BudgetDimension.MAX_TOKENS.value: "tokens",
    BudgetDimension.MAX_COST_USD.value: "cost",
    BudgetDimension.MAX_WALL_SECONDS.value: "wall",
    BudgetDimension.MAX_CALLS.value: "calls",
    "tokens": "tokens",
    "cost": "cost",
    "wall": "wall",
    "calls": "calls",
}

DEFAULT_TOKEN_RATES: dict[str, float] = {
    "gpt-4o": 0.00001,
    "gpt-4o-2024-08-06": 0.00001,
    "gpt-4o-mini": 0.000001,
    "claude-3-5-sonnet": 0.000015,
    "claude-3-haiku": 0.000002,
}

DOWNGRADE_MODEL_MAP: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4o-2024-08-06": "gpt-4o-mini",
    "claude-3-5-sonnet": "claude-3-haiku",
}


def get_downgrade_model(model: str) -> str:
    """Return a cheaper model alternative for downgrades."""
    if model in DOWNGRADE_MODEL_MAP:
        return DOWNGRADE_MODEL_MAP[model]
    for prefix, cheaper in DOWNGRADE_MODEL_MAP.items():
        if prefix in model:
            return cheaper
    return f"{model}-mini"


class AdmissionVerdict(str, enum.Enum):
    """The four possible per-call admission verdicts (FR-7, ADR-004)."""

    ALLOW = "allow"
    DOWNGRADE = "downgrade"
    TRUNCATE = "truncate"
    HALT = "halt"


def project_call_cost(call_data: dict[str, Any]) -> dict[str, float]:
    """Project resource requirements for a single model call across the 4 dimensions.

    Parameters
    ----------
    call_data:
        LiteLLM call kwargs — messages, model, temperature, etc.

    Returns
    -------
    dict with keys 'tokens', 'cost', 'wall', 'calls'.
    """
    # 1. Tokens projection
    if "projected_tokens" in call_data:
        projected_tokens = float(call_data["projected_tokens"])
    elif "estimated_tokens" in call_data:
        projected_tokens = float(call_data["estimated_tokens"])
    else:
        messages = call_data.get("messages", [])
        total_chars = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                total_chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and "text" in part:
                        total_chars += len(str(part["text"]))
        prompt_tokens = max(10, total_chars // 4)
        completion_tokens = 100
        projected_tokens = float(prompt_tokens + completion_tokens)

    # 2. Cost projection
    if "projected_cost_usd" in call_data:
        projected_cost = float(call_data["projected_cost_usd"])
    elif "estimated_cost_usd" in call_data:
        projected_cost = float(call_data["estimated_cost_usd"])
    else:
        model = call_data.get("model", "")
        rate = 0.00001
        for m_key, m_rate in DEFAULT_TOKEN_RATES.items():
            if m_key in model:
                rate = m_rate
                break
        projected_cost = float(projected_tokens * rate)

    # 3. Wall seconds projection
    if "projected_wall_seconds" in call_data:
        projected_wall = float(call_data["projected_wall_seconds"])
    elif "estimated_wall_seconds" in call_data:
        projected_wall = float(call_data["estimated_wall_seconds"])
    else:
        projected_wall = 1.0

    # 4. Calls projection
    if "projected_calls" in call_data:
        projected_calls = float(call_data["projected_calls"])
    elif "estimated_calls" in call_data:
        projected_calls = float(call_data["estimated_calls"])
    else:
        projected_calls = 1.0

    return {
        "tokens": projected_tokens,
        "cost": projected_cost,
        "wall": projected_wall,
        "calls": projected_calls,
    }


def check_kill_switch(
    cumulative_spend_usd: float,
    ceiling_usd: float,
    *,
    kill_multiplier: float = 1.2,
) -> bool:
    """NFR-3 global kill switch: True if cumulative spend reaches or exceeds 120% of ceiling.

    Intended for Vidish (db/ and api/ owner) to call before admitting run creation or in the global watchdog.
    """
    is_killed = float(cumulative_spend_usd) >= (float(ceiling_usd) * float(kill_multiplier))
    if is_killed:
        log_decision(
            component="RBE",
            decision="global_kill_switch_halt",
            inputs={
                "cumulative_spend_usd": cumulative_spend_usd,
                "ceiling_usd": ceiling_usd,
                "kill_multiplier": kill_multiplier,
            },
            justification=(
                f"Global cumulative spend ${cumulative_spend_usd:.2f} reached or exceeded "
                f"{kill_multiplier*100:.0f}% of ceiling (${ceiling_usd:.2f}); process hard-exit required."
            ),
        )
    return is_killed


def check_spend_ceiling(
    cumulative_spend_usd: float,
    ceiling_usd: float,
) -> bool:
    """NFR-3 ceiling check: True if cumulative spend reaches or exceeds 100% of ceiling.

    Intended for Vidish (db/ and api/ owner) to block run-creating endpoints.
    """
    is_reached = float(cumulative_spend_usd) >= float(ceiling_usd)
    if is_reached:
        log_decision(
            component="RBE",
            decision="spend_ceiling_reached",
            inputs={
                "cumulative_spend_usd": cumulative_spend_usd,
                "ceiling_usd": ceiling_usd,
            },
            justification=(
                f"Global cumulative spend ${cumulative_spend_usd:.2f} reached or exceeded "
                f"declared ceiling (${ceiling_usd:.2f}); all run-creating endpoints blocked."
            ),
        )
    return is_reached


def admission_check(
    remaining_budget: dict[str, float] | BudgetState,
    projected_call_cost: dict[str, float],
    *,
    per_run_max_cost_usd: float | None = None,
    current_cost_usd: float | None = None,
) -> tuple[AdmissionVerdict, str, str]:
    """Pure boundary logic for per-call admission control.

    Parameters
    ----------
    remaining_budget:
        Either a dictionary mapping dimension name ('tokens','cost','wall','calls'
        or 'max_tokens', etc.) to remaining capacity, or a :class:`BudgetState`.
    projected_call_cost:
        Dictionary mapping dimension name to projected consumption.
    per_run_max_cost_usd:
        Optional per-run cost ceiling override (NFR-3).
    current_cost_usd:
        Optional current cumulative cost for the run (used with per_run_max_cost_usd).

    Returns
    -------
    tuple of (AdmissionVerdict, binding_dimension, justification)
    """
    if isinstance(remaining_budget, BudgetState):
        rem = {
            "tokens": float(remaining_budget.remaining(BudgetDimension.MAX_TOKENS)),
            "cost": float(remaining_budget.remaining(BudgetDimension.MAX_COST_USD)),
            "wall": float(remaining_budget.remaining(BudgetDimension.MAX_WALL_SECONDS)),
            "calls": float(remaining_budget.remaining(BudgetDimension.MAX_CALLS)),
        }
        if current_cost_usd is None:
            current_cost_usd = float(remaining_budget.cost_usd)
    else:
        rem = {}
        for k, v in remaining_budget.items():
            mapped_dim = DIMENSION_MAP.get(k, k)
            rem[mapped_dim] = float(v)

    # Standardize projected costs
    proj = {
        "tokens": float(projected_call_cost.get("tokens", projected_call_cost.get("max_tokens", 0.0))),
        "cost": float(projected_call_cost.get("cost", projected_call_cost.get("max_cost_usd", 0.0))),
        "wall": float(projected_call_cost.get("wall", projected_call_cost.get("max_wall_seconds", 0.0))),
        "calls": float(projected_call_cost.get("calls", projected_call_cost.get("max_calls", 1.0))),
    }

    # 0. NFR-3 Per-Run Cost Ceiling Check (highest priority hard ceiling)
    if per_run_max_cost_usd is not None:
        cost_so_far = float(current_cost_usd or 0.0)
        proj_cost = proj.get("cost", 0.0)
        if cost_so_far >= per_run_max_cost_usd:
            return (
                AdmissionVerdict.HALT,
                "cost",
                f"Per-run cost ceiling reached: spend ${cost_so_far:.4f} >= limit ${per_run_max_cost_usd:.4f} (OASIS_PER_RUN_MAX_COST_USD)",
            )
        if (cost_so_far + proj_cost) > per_run_max_cost_usd:
            return (
                AdmissionVerdict.HALT,
                "cost",
                f"Per-run cost ceiling exceeded: spend ${cost_so_far + proj_cost:.4f} > limit ${per_run_max_cost_usd:.4f} (OASIS_PER_RUN_MAX_COST_USD)",
            )

    # 1. HALT Check (declared budget exhaustion)
    # If any dimension is already exhausted (<= 0) or projected exceeds remaining
    for dim in ["tokens", "cost", "wall", "calls"]:
        r = rem.get(dim, float("inf"))
        p = proj.get(dim, 0.0)
        if r <= 0:
            return (
                AdmissionVerdict.HALT,
                dim,
                f"Budget exhausted on dimension '{dim}' (remaining: {r:.4f})",
            )
        if p > r:
            return (
                AdmissionVerdict.HALT,
                dim,
                f"Projected consumption ({p:.4f}) exceeds remaining budget ({r:.4f}) on dimension '{dim}'",
            )

    # 2. DOWNGRADE Check (near limit on cost)
    # If projected cost > 70% of remaining cost
    rem_cost = rem.get("cost", float("inf"))
    proj_cost = proj.get("cost", 0.0)
    if rem_cost > 0 and proj_cost > (0.70 * rem_cost):
        return (
            AdmissionVerdict.DOWNGRADE,
            "cost",
            f"Projected cost ({proj_cost:.6f}) exceeds 70% of remaining cost ({rem_cost:.6f}); model downgrade recommended",
        )

    # 3. TRUNCATE Check (near limit on tokens)
    # If projected tokens > 80% of remaining tokens
    rem_tokens = rem.get("tokens", float("inf"))
    proj_tokens = proj.get("tokens", 0.0)
    if rem_tokens > 0 and proj_tokens > (0.80 * rem_tokens):
        return (
            AdmissionVerdict.TRUNCATE,
            "tokens",
            f"Projected tokens ({proj_tokens:.0f}) exceeds 80% of remaining tokens ({rem_tokens:.0f}); context truncation recommended",
        )

    # 4. ALLOW Check
    return (
        AdmissionVerdict.ALLOW,
        "tokens",
        "Projected consumption within remaining budget limits across all dimensions",
    )


def handle_halt(
    outputs: list[Any] | None = None,
    *,
    run_id: str = "",
    budget_event: dict[str, Any] | None = None,
    quality_fn: Callable[[Any], float] | None = None,
    halt_reason: str | None = None,
) -> dict[str, Any]:
    """Terminate with status 'halted_budget' and return the best output produced so far (FR-8).

    Guarantees no error is raised and returns the best artifact/output available.

    Parameters
    ----------
    outputs:
        Outputs produced so far by agent(s) during the run.
    run_id:
        Current run identifier.
    budget_event:
        Optional budget event dict that triggered the halt.
    quality_fn:
        Optional function to score output quality.
    halt_reason:
        Optional human-readable explanation.

    Returns
    -------
    dict
        Status record containing 'status' ('halted_budget'), 'best_output',
        'outputs_count', 'halt_reason', 'budget_event', and 'run_id'.
    """
    outputs_list = list(outputs or [])

    best = None
    if outputs_list:
        if quality_fn is not None:
            best = max(outputs_list, key=quality_fn)
        elif all(isinstance(x, dict) and ("score" in x or "quality" in x) for x in outputs_list):
            best = max(outputs_list, key=lambda x: x.get("score", x.get("quality", 0.0)))
        else:
            best = outputs_list[-1]

    reason = halt_reason
    if not reason and budget_event:
        reason = budget_event.get("justification", "Budget exhausted")
    if not reason:
        reason = "Budget exhausted (FR-8)"

    log_decision(
        component="RBE",
        decision="run_halt",
        inputs={
            "outputs_count": len(outputs_list),
            "has_best_output": best is not None,
        },
        justification=f"Run terminated with status halted_budget: {reason}.",
        run_id=run_id,
    )

    return {
        "status": "halted_budget",
        "best_output": best,
        "outputs_count": len(outputs_list),
        "halt_reason": reason,
        "budget_event": budget_event,
        "run_id": run_id,
    }


class AdmissionController:
    """Stateful per-call admission gate for a single run.

    Instantiated at run start with a live :class:`~oasis.rbe.budget.BudgetState`.
    Called from the LiteLLM pre-call hook on every outbound model call.
    Maintains an in-memory audit log of all ``budget_event`` records.
    """

    def __init__(
        self,
        budget_state: Any,
        *,
        run_id: str = "",
        per_run_max_cost_usd: float | None = None,
    ) -> None:
        """Bind this controller to a live budget state.

        Parameters
        ----------
        budget_state:
            A :class:`~oasis.rbe.budget.BudgetState` tracking
            consumption for the current run.
        run_id:
            Optional run identifier associated with this controller.
        per_run_max_cost_usd:
            Optional hard per-run cost ceiling in USD (NFR-3). If None,
            reads OASIS_PER_RUN_MAX_COST_USD from environment if set.
        """
        self._budget_state = budget_state
        self._run_id = run_id
        self._per_run_max_cost_usd = per_run_max_cost_usd
        self._step: int = 0
        self._events: list[dict[str, Any]] = []

    @property
    def budget_state(self) -> Any:
        return self._budget_state

    @property
    def events(self) -> list[dict[str, Any]]:
        """List of all budget events emitted by this admission controller."""
        return list(self._events)

    def check(
        self,
        call_data: dict[str, Any],
    ) -> AdmissionVerdict:
        """Return an admission verdict for the proposed call and record a budget event.

        The controller projects the call's cost, compares against
        remaining budget across all four dimensions, appends a ``budget_event``
        record matching the SQLite schema in DATABASE.md, and returns the
        verdict.

        Parameters
        ----------
        call_data:
            LiteLLM call kwargs — must include ``model``, ``messages``
            and optionally ``purpose``, ``run_id``, ``step``.

        Returns
        -------
        AdmissionVerdict
        """
        import os

        # FR-9: Validate purpose field ('productive' | 'supervision').
        # Supervision calls are counted against the exact same budget and never exempted.
        purpose = call_data.get("purpose", "productive")
        if purpose not in ("productive", "supervision"):
            raise ValueError(
                f"Invalid purpose {purpose!r}; must be 'productive' or 'supervision'"
            )

        per_run_limit = self._per_run_max_cost_usd
        if per_run_limit is None:
            env_val = os.environ.get("OASIS_PER_RUN_MAX_COST_USD")
            if env_val is not None and env_val != "":
                try:
                    per_run_limit = float(env_val)
                except ValueError:
                    per_run_limit = None

        projected = project_call_cost(call_data)
        verdict, dim, justification = admission_check(
            self._budget_state,
            projected,
            per_run_max_cost_usd=per_run_limit,
        )

        model_before = str(call_data.get("model", ""))
        if verdict == AdmissionVerdict.DOWNGRADE:
            model_after = get_downgrade_model(model_before)
        else:
            model_after = model_before

        run_id = call_data.get("run_id") or self._run_id or "default-run"
        step = call_data.get("step", self._step)
        self._step = step + 1

        # Calculate remaining for the binding dimension
        if hasattr(self._budget_state, "remaining"):
            rev_dim = {
                "tokens": BudgetDimension.MAX_TOKENS,
                "cost": BudgetDimension.MAX_COST_USD,
                "wall": BudgetDimension.MAX_WALL_SECONDS,
                "calls": BudgetDimension.MAX_CALLS,
            }.get(dim, dim)
            remaining_val = float(self._budget_state.remaining(rev_dim))
        elif isinstance(self._budget_state, dict):
            remaining_val = float(self._budget_state.get(dim, 0.0))
        else:
            remaining_val = 0.0

        event: dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "run_id": str(run_id),
            "step": int(step),
            "dimension": dim,
            "action": verdict.value,
            "projected": float(projected.get(dim, 0.0)),
            "remaining": remaining_val,
            "model_before": model_before,
            "model_after": model_after,
            "justification": justification,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._events.append(event)

        # FR-24: Every automated decision writes to decision_log
        log_decision(
            component="RBE",
            decision=f"admit_{verdict.value}",
            inputs={
                "model": model_before,
                "dimension": dim,
                "projected": float(projected.get(dim, 0.0)),
                "remaining": remaining_val,
                "step": int(step),
                "purpose": purpose,
            },
            justification=justification,
            run_id=str(run_id),
        )

        return verdict

    def request_handoff_budget(
        self,
        estimated_tokens: int,
        estimated_cost_usd: float,
    ) -> bool:
        """Reserve budget for a replacement handoff (ADR-007).

        Called by the Replacement Engine before executing a swap.
        Returns ``True`` if sufficient budget remains and the
        reservation is recorded, ``False`` (logged as
        ``replacement_denied_budget``) otherwise.

        Parameters
        ----------
        estimated_tokens:
            Estimated token cost of the context handoff.
        estimated_cost_usd:
            Estimated dollar cost of the context handoff.

        Returns
        -------
        bool
        """
        if hasattr(self._budget_state, "remaining"):
            rem_tokens = self._budget_state.remaining(BudgetDimension.MAX_TOKENS)
            rem_cost = self._budget_state.remaining(BudgetDimension.MAX_COST_USD)
        elif isinstance(self._budget_state, dict):
            rem_tokens = self._budget_state.get("tokens", self._budget_state.get("max_tokens", 0))
            rem_cost = self._budget_state.get("cost", self._budget_state.get("max_cost_usd", 0.0))
        else:
            return False

        if rem_tokens >= estimated_tokens and rem_cost >= estimated_cost_usd:
            # Record reservation as consumption
            if hasattr(self._budget_state, "record"):
                self._budget_state.record(
                    tokens=estimated_tokens,
                    cost_usd=estimated_cost_usd,
                )
            log_decision(
                component="RBE",
                decision="handoff_budget_granted",
                inputs={
                    "estimated_tokens": estimated_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                },
                justification=(
                    f"Handoff budget reservation approved: "
                    f"{estimated_tokens} tokens and ${estimated_cost_usd:.4f} allocated for replacement."
                ),
                run_id=self._run_id,
            )
            return True

        log_decision(
            component="RBE",
            decision="replacement_denied_budget",
            inputs={
                "estimated_tokens": estimated_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "remaining_tokens": rem_tokens,
                "remaining_cost": rem_cost,
            },
            justification=(
                f"Handoff budget reservation denied: requested {estimated_tokens} tokens and "
                f"${estimated_cost_usd:.4f} exceeds remaining budget."
            ),
            run_id=self._run_id,
        )
        return False
