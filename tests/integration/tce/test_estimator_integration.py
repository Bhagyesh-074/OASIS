"""Integration test running BOTH TCE estimators on the same task (FR-4, REQUIREMENTS.md §FR-4).

FR-4 (S, TCE):
"The system shall support an alternative LLM-planner estimator behind the same interface,
 for use as a baseline arm. Verified by integration test running both estimators on the same task."

Verifies:
1. Both `heuristic.estimate` and `llm_planner.estimate` execute against the same statement.
2. Both return an instance of `Estimate` matching the schema in docs/API_SPEC.md.
3. Both outputs contain the identical set of top-level fields, subscores, and contributions.
4. Both log decisions to the TCE audit trail (FR-24).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway
from oasis.rbe.decision_log_client import clear_decision_log, get_decision_log
from oasis.tce import (
    Estimate,
    LLMPlanner,
)
from oasis.tce import (
    estimate as estimate_heuristic,
)
from oasis.tce import (
    estimate_llm as estimate_llm_func,
)
from oasis.tce.llm_planner import DEFAULT_MODEL, build_planner_messages


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Deterministic FakeLLM instance."""
    return FakeLLM()


@pytest.fixture
def gateway(tmp_path: Path, fake_llm: FakeLLM) -> Gateway:
    """Gateway configured with FakeLLM and isolated JSONLSink."""
    sink = JSONLSink(tmp_path / "integration_gateway.jsonl")
    return Gateway(sink=sink, llm=fake_llm, run_id="integration-run")


@pytest.fixture(autouse=True)
def _clear_logs() -> Iterator[None]:
    clear_decision_log()
    yield
    clear_decision_log()


def test_both_estimators_on_same_statement(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4 Integration: Run heuristic and LLM-planner on the same task; assert schema parity."""
    statement = (
        "Write a parser for CSV files, add unit tests, then document the API and "
        "finally benchmark it against pandas."
    )

    # 1. Script the FakeLLM response for the LLM planner
    messages = build_planner_messages(statement, domain="code_generation")
    scripted_payload = {
        "mvts": 4,
        "subtasks": [
            "Write a parser for CSV files",
            "Add comprehensive unit tests",
            "Document the API specifications",
            "Benchmark performance against pandas",
        ],
        "rationale": "Four distinct engineering stages requiring parser dev, QA, docs, and perf testing.",
    }
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={"content": json.dumps(scripted_payload)},
    )

    # 2. Run Heuristic Estimator
    est_heuristic = estimate_heuristic(statement, domain="code_generation")

    # 3. Run LLM Planner Estimator (via direct function)
    est_llm = estimate_llm_func(
        statement,
        domain="code_generation",
        gateway=gateway,
        model=DEFAULT_MODEL,
    )

    # 4. Run LLM Planner Estimator (via OO wrapper)
    planner_wrapper = LLMPlanner(gateway=gateway, model=DEFAULT_MODEL)
    est_llm_oo = planner_wrapper.estimate(statement, domain="code_generation")

    # 5. Parity & Validation Checks
    for name, est in [
        ("heuristic", est_heuristic),
        ("llm_planner", est_llm),
        ("llm_planner_oo", est_llm_oo),
    ]:
        assert isinstance(est, Estimate), f"{name} did not return an Estimate instance"
        assert 1 <= est.mvts <= 8, f"{name} mvts {est.mvts} out of bounds [1, 8]"
        assert est.subscores.subtask_count >= 1, f"{name} subtask_count < 1"
        assert est.subscores.skill_clusters >= 1, f"{name} skill_clusters < 1"
        assert 0.0 <= est.subscores.dep_density <= 1.0, f"{name} dep_density out of [0, 1]"
        assert est.latency_ms >= 0.0, f"{name} latency_ms negative"
        assert len(est.justification) > 0, f"{name} justification empty (FR-24 violation)"
        assert len(est.subtasks) >= 1, f"{name} subtasks empty"

    # 6. Field Set Parity Check
    heuristic_dump = est_heuristic.model_dump()
    llm_dump = est_llm.model_dump()

    # Top-level field keys must be identical
    assert set(heuristic_dump.keys()) == set(llm_dump.keys()), (
        f"Field mismatch between estimators: {set(heuristic_dump.keys()) ^ set(llm_dump.keys())}"
    )

    # SubScores field keys must be identical
    assert set(heuristic_dump["subscores"].keys()) == set(llm_dump["subscores"].keys())
    assert set(heuristic_dump["subscores"].keys()) == {
        "subtask_count",
        "skill_clusters",
        "dep_density",
    }

    # Contributions field keys must be identical
    assert set(heuristic_dump["contributions"].keys()) == set(llm_dump["contributions"].keys())
    assert set(heuristic_dump["contributions"].keys()) == {
        "subtask_count",
        "skill_clusters",
        "dep_density",
    }

    # Weights keys must be identical
    assert set(heuristic_dump["weights"].keys()) == set(llm_dump["weights"].keys())

    # 7. Audit log check (both logged decisions under FR-24)
    logs = get_decision_log()
    assert len(logs) >= 3  # heuristic + llm + llm_oo
    estimators_logged = {record.inputs.get("estimator", "heuristic") for record in logs}
    assert "llm_planner" in estimators_logged
