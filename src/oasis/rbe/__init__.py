"""OASIS Resource Budget Enforcer (RBE).

Two-phase enforcement (ADR-004):
  1. Pre-execution: project team cost, reduce by merging/pruning (FR-6).
  2. Runtime: per-call admission control — allow/downgrade/truncate/halt
     (FR-7 / FR-8).

Budget is exactly four dimensions: max_tokens, max_cost_usd,
max_wall_seconds, max_calls (FR-5, ADR-002).
"""

from oasis.rbe.admission import (
    AdmissionController,
    AdmissionVerdict,
    admission_check,
    check_kill_switch,
    check_spend_ceiling,
    handle_halt,
    project_call_cost,
)
from oasis.rbe.budget import Budget, BudgetDimension, BudgetState
from oasis.rbe.decision_log_client import (
    DecisionLogClient,
    DecisionRecord,
    clear_decision_log,
    get_decision_log,
    log_decision,
)
from oasis.rbe.reducer import BudgetInfeasibleError, TeamReducer

__all__ = [
    "AdmissionController",
    "AdmissionVerdict",
    "Budget",
    "BudgetDimension",
    "BudgetInfeasibleError",
    "BudgetState",
    "DecisionLogClient",
    "DecisionRecord",
    "TeamReducer",
    "admission_check",
    "check_kill_switch",
    "check_spend_ceiling",
    "clear_decision_log",
    "get_decision_log",
    "handle_halt",
    "log_decision",
    "project_call_cost",
]


