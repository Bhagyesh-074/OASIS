"""Local shim for writing to the shared decision_log (FR-24).

FR-24: Every automated decision — sizing, reduction, escalation, flag,
swap, denial, halt — shall write a decision-log record with component,
decision, inputs, and a human-readable justification string.

Schema matching docs/DATABASE.md:
  decision_log(
    decision_id  TEXT PRIMARY KEY,
    run_id       TEXT REFERENCES run(run_id),
    component    TEXT NOT NULL CHECK (component IN ('CM','TCE','RBE','ASL','RTPM','RE')),
    decision     TEXT NOT NULL,
    inputs_json  TEXT NOT NULL,
    justification TEXT NOT NULL,  -- never NULL, never empty
    created_at   TEXT NOT NULL
  )

This module provides a local client and `log_decision` shim used by RBE and
Gateway components to record their automated decisions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

VALID_COMPONENTS = frozenset({"CM", "TCE", "RBE", "ASL", "RTPM", "RE"})


@dataclass(frozen=True)
class DecisionRecord:
    """A single decision-log entry matching DATABASE.md decision_log schema (FR-24).

    ``justification`` must never be empty or whitespace.
    ``component`` must be one of ('CM','TCE','RBE','ASL','RTPM','RE').
    """

    component: str
    decision: str
    inputs: dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def __post_init__(self) -> None:
        if not self.justification or not self.justification.strip():
            raise ValueError("FR-24 violation: justification must never be empty")
        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Invalid component {self.component!r}; must be one of {sorted(VALID_COMPONENTS)}"
            )

    @property
    def inputs_json(self) -> str:
        """Serialize inputs dict to JSON matching SQLite inputs_json column."""
        return json.dumps(self.inputs, sort_keys=True)


# Global in-memory list buffering decisions in this local shim
_GLOBAL_DECISION_LOG: list[DecisionRecord] = []


def log_decision(
    component: str,
    decision: str,
    inputs: dict[str, Any],
    justification: str,
    run_id: str = "",
) -> DecisionRecord:
    """Log an automated decision (FR-24).

    Parameters
    ----------
    component:
        Component name — for RBE/Gateway decisions, must be "RBE".
    decision:
        Action or outcome name (e.g. "reduce_merge", "admit_allow", "admit_halt").
    inputs:
        Context/inputs dictionary.
    justification:
        Human-readable explanation (non-empty string).
    run_id:
        Optional run identifier.

    Returns
    -------
    DecisionRecord
        The validated decision record appended to the log.
    """
    record = DecisionRecord(
        component=component,
        decision=decision,
        inputs=dict(inputs),
        justification=justification,
        run_id=run_id,
    )
    _GLOBAL_DECISION_LOG.append(record)
    return record


def get_decision_log() -> list[DecisionRecord]:
    """Return all decision records logged in this process."""
    return list(_GLOBAL_DECISION_LOG)


def clear_decision_log() -> None:
    """Clear in-memory decision log buffer (useful in tests)."""
    _GLOBAL_DECISION_LOG.clear()


class DecisionLogClient:
    """Local shim — manages decision records.

    Stores records in-memory and can be replaced by the real XAI-module
    writer once confirmed with Bhagyesh.
    """

    def __init__(self) -> None:
        self._records: list[DecisionRecord] = []

    def write(self, record: DecisionRecord) -> None:
        """Persist a decision record."""
        self._records.append(record)
        _GLOBAL_DECISION_LOG.append(record)

    def log(
        self,
        component: str,
        decision: str,
        inputs: dict[str, Any],
        justification: str,
        run_id: str = "",
    ) -> DecisionRecord:
        """Log a decision and persist to client buffer."""
        rec = log_decision(
            component=component,
            decision=decision,
            inputs=inputs,
            justification=justification,
            run_id=run_id,
        )
        self._records.append(rec)
        return rec

    def flush(self) -> list[DecisionRecord]:
        """Return all buffered records."""
        return list(self._records)

    def clear(self) -> None:
        """Clear client buffer."""
        self._records.clear()
