"""FR-6 — pre-execution team-size reduction.

Pre-execution, the system shall project team cost and reduce the team by
merging or pruning roles until the projection satisfies all four budget
dimensions, logging the binding constraint for each reduction.

The reducer operates on the proposed team from TCE and the budget from the
run request.  It iteratively merges the two most semantically similar
roles until the cost projection fits within all four budget dimensions.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

from oasis.rbe.budget import Budget, BudgetDimension
from oasis.rbe.decision_log_client import log_decision

# ------------------------------------------------------------------
# Netra's TCE Interface Contract (Role Semantic Similarity)
# ------------------------------------------------------------------
# Interface contract needed from Netra's TCE module (complexity / role modeling):
#
#   def similarity(role_a: dict[str, Any] | str, role_b: dict[str, Any] | str) -> float:
#       """Compute semantic similarity in [0.0, 1.0] between two roles.
#
#       Parameters
#       ----------
#       role_a, role_b:
#           Role specification dicts (with 'role' key) or role name strings.
#
#       Returns
#       -------
#       float
#           Cosine similarity or semantic score in [0.0, 1.0].
#       """
#
# NOTE: Confirm this contract with Netra before relying on the TCE implementation.
# In the meantime, TeamReducer uses a local string-based placeholder default.


def default_role_similarity(
    role_a: dict[str, Any] | str,
    role_b: dict[str, Any] | str,
) -> float:
    """Placeholder similarity based on role name string similarity.

    Used until Netra's TCE module provides real embedding-based similarity.
    """
    name_a = role_a.get("role", "") if isinstance(role_a, dict) else str(role_a)
    name_b = role_b.get("role", "") if isinstance(role_b, dict) else str(role_b)
    if name_a == name_b:
        return 1.0
    return difflib.SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()


class BudgetInfeasibleError(ValueError):
    """Raised when no reduction can make the team fit the budget (FR-6)."""

    def __init__(
        self,
        message: str,
        *,
        binding_constraint: str = "",
        projected: dict[str, float] | None = None,
        reduced_team: list[dict[str, Any]] | None = None,
        reduction_log: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.binding_constraint = binding_constraint
        self.projected = projected or {}
        self.reduced_team = reduced_team or []
        self.reduction_log = reduction_log or []


class TeamReducer:
    """Project cost and reduce team size to fit the budget (FR-6, ARCHITECTURE.md).

    The reduction loop:
      1. Estimate per-role token cost from role dict or historical means.
      2. While projected cost exceeds any budget dimension:
         a. Identify the binding constraint that triggered the merge.
         b. Find the two most semantically similar roles.
         c. Merge them.
         d. Log the reduction step with the binding constraint.
      3. If no reduction can make the team fit (size 1 still exceeds budget),
         raise BudgetInfeasibleError.
    """

    def __init__(
        self,
        per_role_token_estimates: dict[str, int] | None = None,
        *,
        similarity_fn: Callable[[Any, Any], float] | None = None,
        merge_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialise with optional historical per-role token estimates.

        Parameters
        ----------
        per_role_token_estimates:
            ``{role_name: estimated_tokens}``.  If ``None``, estimates
            are read from the role dicts or default to 1,000.
        similarity_fn:
            Pluggable pairwise similarity function ``similarity(role_a, role_b) -> float``.
            Defaults to ``default_role_similarity`` (name matching).
        merge_fn:
            Pluggable role merger function ``merge(role_a, role_b) -> merged_role``.
            Defaults to ``self._default_merge``.
        """
        self._estimates = per_role_token_estimates or {}
        self._similarity_fn = similarity_fn or default_role_similarity
        self._merge_fn = merge_fn or self._default_merge

    def _extract_role_metrics(self, role: dict[str, Any]) -> tuple[int, float, float, int]:
        """Extract (tokens, cost_usd, wall_seconds, calls) for a role."""
        role_name = role.get("role", "")
        tokens = int(
            role.get("est_tokens")
            or role.get("tokens")
            or self._estimates.get(role_name, 1_000)
        )
        cost_usd = float(
            role.get("est_cost_usd")
            if role.get("est_cost_usd") is not None
            else role.get("cost_usd", tokens * 0.00001)
        )
        wall_seconds = float(
            role.get("est_wall_seconds")
            if role.get("est_wall_seconds") is not None
            else role.get("wall_seconds", max(1.0, tokens / 100.0))
        )
        calls = int(
            role.get("est_calls")
            or role.get("calls")
            or 1
        )
        return tokens, cost_usd, wall_seconds, calls

    def _default_merge(
        self,
        role_a: dict[str, Any],
        role_b: dict[str, Any],
    ) -> dict[str, Any]:
        """Default role merger: combines two roles into one agent.

        Tokens, cost, wall-seconds, and calls reflect shared context and
        reduced inter-agent coordination overhead.
        """
        tok_a, cost_a, sec_a, calls_a = self._extract_role_metrics(role_a)
        tok_b, cost_b, sec_b, calls_b = self._extract_role_metrics(role_b)

        merged_tokens = max(tok_a, tok_b) + int(0.5 * min(tok_a, tok_b))
        merged_cost = max(cost_a, cost_b) + 0.5 * min(cost_a, cost_b)
        merged_sec = max(sec_a, sec_b) + 0.5 * min(sec_a, sec_b)
        merged_calls = max(calls_a, calls_b) + max(0, min(calls_a, calls_b) - 1)

        name_a = role_a.get("role", "role_a")
        name_b = role_b.get("role", "role_b")

        merged_from: list[str] = []
        for r in (role_a, role_b):
            if "merged_from" in r:
                merged_from.extend(r["merged_from"])
            else:
                merged_from.append(r.get("role", "unknown"))

        return {
            "role": f"{name_a}+{name_b}",
            "est_tokens": merged_tokens,
            "est_cost_usd": round(merged_cost, 6),
            "est_wall_seconds": round(merged_sec, 2),
            "est_calls": max(1, merged_calls),
            "merged_from": merged_from,
            "model": role_a.get("model") or role_b.get("model", ""),
            "temperature": role_a.get("temperature", 0.0),
        }

    def project_cost(
        self,
        team: list[dict[str, Any]],
        budget: Any = None,
    ) -> dict[str, float]:
        """Project total cost across all four budget dimensions.

        Returns
        -------
        dict mapping ``BudgetDimension`` value strings to projected totals.
        """
        tot_tokens = 0
        tot_cost = 0.0
        tot_sec = 0.0
        tot_calls = 0

        for r in team:
            t, c, s, n = self._extract_role_metrics(r)
            tot_tokens += t
            tot_cost += c
            tot_sec += s
            tot_calls += n

        return {
            BudgetDimension.MAX_TOKENS.value: float(tot_tokens),
            BudgetDimension.MAX_COST_USD.value: round(tot_cost, 6),
            BudgetDimension.MAX_WALL_SECONDS.value: round(tot_sec, 2),
            BudgetDimension.MAX_CALLS.value: float(tot_calls),
        }

    def _ensure_budget(self, budget: Any) -> Budget:
        """Convert input budget to Budget model if necessary."""
        if isinstance(budget, Budget):
            return budget
        if isinstance(budget, dict):
            return Budget.from_dict(budget)
        raise TypeError(f"Expected Budget or dict, got {type(budget).__name__}")

    def _find_binding_constraint(
        self,
        projected: dict[str, float],
        budget: Budget,
    ) -> str | None:
        """Identify which dimension is the binding constraint causing budget breach."""
        breached: list[tuple[float, str]] = []
        for dim in BudgetDimension:
            limit = budget.limit(dim)
            if limit is not None:
                val = projected[dim.value]
                if limit == 0 and val > 0:
                    breached.append((float("inf"), dim.value))
                elif val > limit:
                    ratio = val / limit
                    breached.append((ratio, dim.value))

        if not breached:
            return None

        # Sort by breach ratio descending
        breached.sort(key=lambda x: x[0], reverse=True)
        return breached[0][1]

    def _find_most_similar_pair(
        self,
        team: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Find index pair (i, j) with highest semantic similarity."""
        best_pair = (0, 1)
        best_sim = -1.0

        n = len(team)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._similarity_fn(team[i], team[j])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)

        return best_pair

    def reduce(
        self,
        team: list[dict[str, Any]],
        budget: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Reduce *team* until the projection fits within *budget*.

        Returns
        -------
        (reduced_team, reduction_log)
            ``reduction_log`` is a list of dicts, each with at minimum
            ``{"merged_roles": [...], "binding_constraint": "..."}``
            per FR-6.

        Raises
        ------
        BudgetInfeasibleError
            If even after all possible reductions, the team cannot satisfy the budget.
        """
        b = self._ensure_budget(budget)
        current_team = [dict(r) for r in team]
        reduction_log: list[dict[str, Any]] = []

        while True:
            projected = self.project_cost(current_team, b)
            binding = self._find_binding_constraint(projected, b)

            # If within budget across all four dimensions, done
            if binding is None:
                return current_team, reduction_log

            # If team cannot be reduced further (size == 1)
            if len(current_team) <= 1:
                infeasible_msg = (
                    f"No team of size >=1 fits budget: {binding} projected={projected[binding]} "
                    f"> limit={b.limit(binding)}"
                )
                log_decision(
                    component="RBE",
                    decision="reduce_infeasible",
                    inputs={
                        "binding_constraint": binding,
                        "projected": projected,
                        "team_size": len(current_team),
                    },
                    justification=infeasible_msg,
                )
                raise BudgetInfeasibleError(
                    infeasible_msg,
                    binding_constraint=binding,
                    projected=projected,
                    reduced_team=current_team,
                    reduction_log=reduction_log,
                )

            # Find two most similar roles
            idx_a, idx_b = self._find_most_similar_pair(current_team)
            role_a = current_team[idx_a]
            role_b = current_team[idx_b]

            # Merge roles
            merged_role = self._merge_fn(role_a, role_b)

            name_a = role_a.get("role", "role_a")
            name_b = role_b.get("role", "role_b")

            log_entry = {
                "from_size": len(current_team),
                "to_size": len(current_team) - 1,
                "binding": binding,
                "binding_constraint": binding,
                "merged": [name_a, name_b],
                "merged_roles": [name_a, name_b],
                "projected_before": projected,
                "justification": (
                    f"Projected {binding} ({projected[binding]}) exceeded limit "
                    f"({b.limit(binding)}); merged {name_a} and {name_b}"
                ),
            }
            reduction_log.append(log_entry)
            log_decision(
                component="RBE",
                decision="reduce_merge",
                inputs={
                    "from_size": len(current_team),
                    "to_size": len(current_team) - 1,
                    "binding_constraint": binding,
                    "merged_roles": [name_a, name_b],
                    "projected_before": projected,
                },
                justification=log_entry["justification"],
            )

            # Replace idx_a with merged_role and remove idx_b
            current_team.pop(idx_b)
            current_team[idx_a] = merged_role
