"""Unit tests for TCE decision log shim (FR-24)."""

from __future__ import annotations

import pytest

from oasis.rbe.decision_log_client import clear_decision_log, get_decision_log
from oasis.tce._log import log_tce_decision


def setup_function() -> None:
    """Clear in-memory decision log before each test."""
    clear_decision_log()


def teardown_function() -> None:
    """Clear in-memory decision log after each test."""
    clear_decision_log()


def test_log_tce_decision_component_tce() -> None:
    """FR-24: log_tce_decision logs a decision with component='TCE'."""
    inputs = {
        "statement": "Analyze benchmark results",
        "subscores": {"subtask_count": 3, "skill_clusters": 2, "dep_density": 0.25},
        "mvts": 2,
    }
    justification = "MVTS=2 based on subtask count 3, skill clusters 2, dep density 0.25"

    record = log_tce_decision(
        decision="estimate_mvts",
        inputs=inputs,
        justification=justification,
        run_id="run_001",
    )

    assert record.component == "TCE"
    assert record.decision == "estimate_mvts"
    assert record.inputs == inputs
    assert record.justification == justification
    assert record.run_id == "run_001"

    # Verify present in global decision log
    all_records = get_decision_log()
    assert len(all_records) == 1
    assert all_records[0].decision_id == record.decision_id
    assert all_records[0].component == "TCE"


def test_log_tce_decision_empty_justification_rejected() -> None:
    """FR-24: Empty or whitespace justification is rejected."""
    with pytest.raises(ValueError, match="FR-24 violation: justification must never be empty"):
        log_tce_decision(
            decision="estimate_mvts",
            inputs={"statement": "Test"},
            justification="",
        )

    with pytest.raises(ValueError, match="FR-24 violation: justification must never be empty"):
        log_tce_decision(
            decision="estimate_mvts",
            inputs={"statement": "Test"},
            justification="   ",
        )
