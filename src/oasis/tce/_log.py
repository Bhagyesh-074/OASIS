"""TCE decision logging client shim (FR-24).

Single import point for writing TCE automated decisions to the shared decision_log.
Maintains component="TCE" and can be redirected to the xai module once available
without modifying callers across the tce package.
"""

from __future__ import annotations

from typing import Any

from oasis.rbe.decision_log_client import DecisionRecord, log_decision


def log_tce_decision(
    decision: str,
    inputs: dict[str, Any],
    justification: str,
    run_id: str = "",
) -> DecisionRecord:
    """Log an automated TCE decision to the audit log (FR-24).

    Parameters
    ----------
    decision:
        Action or outcome name (e.g., 'estimate_mvts', 'propose_team_size').
    inputs:
        Context/inputs dictionary (e.g. statement, subscores, weights).
    justification:
        Human-readable explanation string (FR-24: never empty).
    run_id:
        Optional run identifier.

    Returns
    -------
    DecisionRecord
        The validated decision record appended to the log.
    """
    return log_decision(
        component="TCE",
        decision=decision,
        inputs=inputs,
        justification=justification,
        run_id=run_id,
    )
