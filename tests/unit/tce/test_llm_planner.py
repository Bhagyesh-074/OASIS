"""Unit tests for LLM-Planner Task Complexity Estimator (FR-4, FR-24, NFR-1, NFR-5).

Verifies:
1. Valid JSON model replies parse into valid Estimate models.
2. JSON inside markdown code fences (with/without surrounding commentary) parses cleanly.
3. Out-of-range MVTS is clamped to config.mvts_range.
4. Malformed/unparseable replies fall back to documented defaults without raising.
5. Unpinned model identifiers are rejected (NFR-5).
6. LLMPlanner object-oriented wrapper works as an estimator callable.
7. Decision logging records rationale, clamp notes, and fallbacks (FR-24).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway, UnpinnedModelError
from oasis.rbe.decision_log_client import clear_decision_log, get_decision_log
from oasis.tce.config import load_complexity_config
from oasis.tce.llm_planner import (
    DEFAULT_MODEL,
    LLMPlanner,
    build_planner_messages,
    estimate,
    parse_llm_response,
)
from oasis.tce.types import Estimate


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Fresh FakeLLM instance for deterministic scripted responses."""
    return FakeLLM()


@pytest.fixture
def gateway(tmp_path: Path, fake_llm: FakeLLM) -> Gateway:
    """Gateway instance backed by FakeLLM and a temporary JSONLSink."""
    sink = JSONLSink(tmp_path / "gateway_test.jsonl")
    return Gateway(sink=sink, llm=fake_llm, run_id="unit-test-run")


@pytest.fixture(autouse=True)
def _clear_logs() -> Iterator[None]:
    """Clear decision log before and after each test."""
    clear_decision_log()
    yield
    clear_decision_log()


# ---------------------------------------------------------------------------
# 1. Valid JSON Reply (FR-4)
# ---------------------------------------------------------------------------
def test_valid_json_reply(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: Valid JSON output produces a complete Estimate model with sub-scores."""
    statement = "Write a CSV parser in Python, write unit tests, and document the API."
    messages = build_planner_messages(statement)

    scripted_payload = {
        "mvts": 3,
        "subtasks": [
            "Write a CSV parser in Python",
            "Write comprehensive unit tests",
            "Document the API specification",
        ],
        "rationale": "Requires development, testing, and technical documentation competencies.",
    }
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={"content": json.dumps(scripted_payload)},
    )

    est = estimate(statement, gateway=gateway, model=DEFAULT_MODEL)

    assert isinstance(est, Estimate)
    assert est.mvts == 3
    assert len(est.subtasks) == 3
    assert est.subscores.subtask_count == 3
    assert est.subscores.skill_clusters >= 1
    assert 0.0 <= est.subscores.dep_density <= 1.0
    assert "Requires development" in est.justification
    assert est.latency_ms >= 0.0

    # Verify decision log was written (FR-24)
    logs = get_decision_log()
    assert len(logs) >= 1
    last = logs[-1]
    assert last.decision == "estimate_mvts"
    assert last.inputs["estimator"] == "llm_planner"
    assert last.inputs["mvts"] == 3
    assert last.inputs["fallback_used"] is False


# ---------------------------------------------------------------------------
# 2. JSON Inside Code Fences & Surrounding Commentary (FR-4)
# ---------------------------------------------------------------------------
def test_json_inside_code_fences(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: Parser strips markdown ```json ... ``` code fences."""
    statement = "Clean dataset and train a random forest model."
    messages = build_planner_messages(statement)

    fenced_content = """```json
{
  "mvts": 2,
  "subtasks": [
    "Clean and normalize dataset",
    "Train and evaluate random forest model"
  ],
  "rationale": "Two serial data science stages."
}
```"""
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={"content": fenced_content},
    )

    est = estimate(statement, gateway=gateway, model=DEFAULT_MODEL)

    assert est.mvts == 2
    assert len(est.subtasks) == 2
    assert est.subscores.subtask_count == 2
    assert "Two serial data science stages." in est.justification


def test_json_with_surrounding_conversational_text(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: Parser tolerates conversational intro/outro text around the JSON block."""
    statement = "Audit security certificates and deploy renewal automation."
    messages = build_planner_messages(statement)

    chatter_content = """Certainly! Here is my complexity assessment:
```json
{
  "mvts": 4,
  "subtasks": ["Audit certificates", "Build renewal script", "Deploy cron", "Verify alerting"],
  "rationale": "High reliability requirements across devops and secops."
}
```
Please let me know if you need additional clarification."""

    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={"content": chatter_content},
    )

    est = estimate(statement, gateway=gateway, model=DEFAULT_MODEL)

    assert est.mvts == 4
    assert len(est.subtasks) == 4
    assert "High reliability requirements" in est.justification


# ---------------------------------------------------------------------------
# 3. Out-of-Range MVTS Clamping (FR-4)
# ---------------------------------------------------------------------------
def test_out_of_range_mvts_is_clamped(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: MVTS exceeding config.mvts_range is clamped to boundaries with justification."""
    cfg = load_complexity_config()
    min_mvts, max_mvts = cfg.mvts_range

    # 1. Test upper clamp
    stmt_high = "Build a global distributed cloud operating system from scratch."
    msgs_high = build_planner_messages(stmt_high)
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=msgs_high,
        response={
            "content": json.dumps({
                "mvts": 25,  # Exceeds max (8)
                "subtasks": ["Kernel", "Network", "Storage"],
                "rationale": "Massive engineering scope.",
            })
        },
    )

    est_high = estimate(stmt_high, gateway=gateway, config=cfg, model=DEFAULT_MODEL)
    assert est_high.mvts == max_mvts
    assert f"clamped from 25 to ({min_mvts}, {max_mvts})" in est_high.justification

    # 2. Test lower clamp (zero or negative)
    stmt_low = "Echo hello world."
    msgs_low = build_planner_messages(stmt_low)
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=msgs_low,
        response={
            "content": json.dumps({
                "mvts": 0,  # Below min (1)
                "subtasks": ["Echo text"],
                "rationale": "Trivial single step.",
            })
        },
    )

    est_low = estimate(stmt_low, gateway=gateway, config=cfg, model=DEFAULT_MODEL)
    assert est_low.mvts == min_mvts
    assert f"clamped from 0 to ({min_mvts}, {max_mvts})" in est_low.justification


# ---------------------------------------------------------------------------
# 4. Malformed Reply Fallback (Never Raises) (FR-4, FR-24)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "malformed_content",
    [
        "Internal server error: upstream timeout",
        "```json\n{ truncated json ...",
        "",
        "   \n\t  ",
        '{"unexpected_key": 42}',
        '{"mvts": "not-an-int", "subtasks": "not-a-list"}',
        "<html><body>502 Bad Gateway</body></html>",
    ],
)
def test_malformed_reply_falls_back_without_raising(
    gateway: Gateway,
    fake_llm: FakeLLM,
    malformed_content: str,
) -> None:
    """FR-4: Defensively falls back to documented default (mvts=1, 1 subtask) on bad replies."""
    statement = "Perform regression analysis on quarterly churn."
    messages = build_planner_messages(statement)

    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={"content": malformed_content},
    )

    est = estimate(statement, gateway=gateway, model=DEFAULT_MODEL)

    assert isinstance(est, Estimate)
    assert est.mvts == 1
    assert len(est.subtasks) >= 1
    assert est.subtasks[0] == statement
    assert "fallback" in est.justification.lower()

    # Decision log confirms fallback was audited (FR-24)
    logs = get_decision_log()
    assert len(logs) >= 1
    assert logs[-1].inputs["fallback_used"] is True


def test_parse_llm_response_direct_unit() -> None:
    """Unit test for parse_llm_response helper behavior directly."""
    cfg = load_complexity_config()
    clamped, raw, subtasks, rationale, fallback = parse_llm_response(
        raw_content='{"mvts": 5, "subtasks": ["A", "B"], "rationale": "ok"}',
        statement="original statement",
        config=cfg,
    )
    assert clamped == 5
    assert raw == 5
    assert subtasks == ["A", "B"]
    assert rationale == "ok"
    assert fallback is False


# ---------------------------------------------------------------------------
# 5. Unpinned Model Rejection (NFR-5)
# ---------------------------------------------------------------------------
def test_unpinned_model_is_rejected(gateway: Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR-5: Model identifiers with unpinned aliases like 'latest' or empty strings are rejected."""
    statement = "Write a sorting function."

    # Test explicit unpinned parameter
    with pytest.raises(UnpinnedModelError, match="unpinned"):
        estimate(statement, gateway=gateway, model="gpt-4o-latest")

    with pytest.raises(UnpinnedModelError, match="unpinned"):
        estimate(statement, gateway=gateway, model="claude-3-5-sonnet-latest")

    with pytest.raises(UnpinnedModelError, match="unpinned"):
        estimate(statement, gateway=gateway, model="")

    # Test via environment variable
    monkeypatch.setenv("OASIS_MODEL_CHEAP", "gpt-4o-latest")
    with pytest.raises(UnpinnedModelError, match="unpinned"):
        estimate(statement, gateway=gateway)


# ---------------------------------------------------------------------------
# 6. LLMPlanner Class Wrapper & Interchangeability (FR-4)
# ---------------------------------------------------------------------------
def test_llm_planner_class_wrapper(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: LLMPlanner provides an object-oriented 2-argument estimator callable."""
    statement = "Refactor payment processor."
    messages = build_planner_messages(statement, domain="fintech")

    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={
            "content": json.dumps({
                "mvts": 3,
                "subtasks": ["Stripe integration", "Webhook handler", "Idempotency test"],
                "rationale": "Three payments subtasks.",
            })
        },
    )

    planner = LLMPlanner(gateway=gateway, model=DEFAULT_MODEL)
    est = planner.estimate(statement, domain="fintech")

    assert isinstance(est, Estimate)
    assert est.mvts == 3
    assert len(est.subtasks) == 3


# ---------------------------------------------------------------------------
# 7. Gateway Call Purpose: Default "productive" & Override (FR-7, FR-9, ADR-006)
# ---------------------------------------------------------------------------
def test_gateway_call_purpose_default_and_override(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """The planner is an estimator, not supervision; default purpose is productive, override reaches Gateway."""
    statement = "Build an index over log files."
    messages = build_planner_messages(statement)

    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={
            "content": json.dumps({
                "mvts": 2,
                "subtasks": ["Parse logs", "Build inverted index"],
                "rationale": "Index building tasks.",
            })
        },
    )

    # 1. Default invocation has purpose="productive"
    estimate(statement, gateway=gateway, model=DEFAULT_MODEL)
    assert len(gateway.accounting_log) == 1
    assert gateway.accounting_log[-1]["purpose"] == "productive"

    logs = get_decision_log()
    assert logs[-1].inputs["purpose"] == "productive"

    # 2. Explicit override purpose="supervision" in estimate(...) reaches Gateway.call
    estimate(statement, gateway=gateway, model=DEFAULT_MODEL, purpose="supervision")
    assert len(gateway.accounting_log) == 2
    assert gateway.accounting_log[-1]["purpose"] == "supervision"

    logs = get_decision_log()
    assert logs[-1].inputs["purpose"] == "supervision"

    # 3. LLMPlanner constructor purpose override
    planner_supervision = LLMPlanner(gateway=gateway, model=DEFAULT_MODEL, purpose="supervision")
    planner_supervision.estimate(statement)
    assert len(gateway.accounting_log) == 3
    assert gateway.accounting_log[-1]["purpose"] == "supervision"

    # 4. LLMPlanner default constructor has purpose="productive"
    planner_default = LLMPlanner(gateway=gateway, model=DEFAULT_MODEL)
    planner_default.estimate(statement)
    assert len(gateway.accounting_log) == 4
    assert gateway.accounting_log[-1]["purpose"] == "productive"

    # 5. LLMPlanner per-call override
    planner_default.estimate(statement, purpose="supervision")
    assert len(gateway.accounting_log) == 5
    assert gateway.accounting_log[-1]["purpose"] == "supervision"


# ---------------------------------------------------------------------------
# 8. Context Splitting & Excluded Auxiliary Text (FR-1, FR-2, FR-3, FR-4, FR-24)
# ---------------------------------------------------------------------------
def test_context_marker_excludes_context_from_gateway_call_and_matches_instruction_alone(
    gateway: Gateway, fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """FR-4: A statement with long context after marker sends only instruction to Gateway and matches instruction alone."""
    instruction = "Write a CSV parser in Python, write unit tests, and document the API."
    paragraphs = [
        f"Paragraph {i}: Auxiliary background documentation details for data parsing and validation #{i}."
        for i in range(1, 11)
    ]
    context_text = "\n\n".join(paragraphs)
    statement_with_context = f"{instruction}\n\n### CONTEXT\n{context_text}"

    messages_instruction = build_planner_messages(instruction)
    scripted_payload = {
        "mvts": 3,
        "subtasks": [
            "Write a CSV parser in Python",
            "Write comprehensive unit tests",
            "Document the API specification",
        ],
        "rationale": "Three subtasks requiring development, testing, and documentation.",
    }
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages_instruction,
        response={"content": json.dumps(scripted_payload)},
    )

    # 1. Run estimate with context
    est_with_context = estimate(statement_with_context, gateway=gateway, model=DEFAULT_MODEL)

    # 2. Assert on the Gateway call log (JSONL sink)
    sink_file = tmp_path / "gateway_test.jsonl"
    records = [
        json.loads(line)
        for line in sink_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    logged_messages = records[0]["messages"]
    for msg in logged_messages:
        assert "Paragraph 1" not in msg["content"]
        assert "Auxiliary background" not in msg["content"]
        assert "### CONTEXT" not in msg["content"]
        if msg["role"] == "user":
            assert instruction in msg["content"]

    # 3. Run estimate on instruction alone (same scripted response will be matched)
    est_alone = estimate(instruction, gateway=gateway, model=DEFAULT_MODEL)

    # 4. Results are identical
    assert est_with_context.mvts == est_alone.mvts
    assert est_with_context.subscores == est_alone.subscores
    assert est_with_context.contributions == est_alone.contributions
    assert est_with_context.subtasks == est_alone.subtasks
    assert est_with_context.justification == est_alone.justification

    # 5. Check decision logs (FR-24)
    logs = get_decision_log()
    assert len(logs) >= 2
    log_context = logs[-2]
    log_alone = logs[-1]
    assert log_context.inputs["context_chars_excluded"] == len(context_text.strip())
    assert log_context.inputs["context_chars_excluded"] > 0
    assert log_alone.inputs["context_chars_excluded"] == 0


def test_absent_marker_leaves_behaviour_unchanged(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: Absent marker leaves behaviour unchanged and context_chars_excluded is 0."""
    statement = "Deploy a kubernetes cluster and configure ingress controllers."
    messages = build_planner_messages(statement)
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={
            "content": json.dumps({
                "mvts": 2,
                "subtasks": ["Deploy cluster", "Configure ingress"],
                "rationale": "Two infra stages.",
            })
        },
    )

    est = estimate(statement, gateway=gateway, model=DEFAULT_MODEL)
    assert est.mvts == 2
    assert "empty instruction before marker" not in est.justification

    logs = get_decision_log()
    assert logs[-1].inputs["context_chars_excluded"] == 0


def test_empty_instruction_falls_back_with_justification(gateway: Gateway, fake_llm: FakeLLM) -> None:
    """FR-4: Empty instruction before marker falls back to whole statement with justification."""
    statement = "   \n### CONTEXT\nBackground documentation text without preceding instruction."
    messages = build_planner_messages(statement)
    fake_llm.script(
        model=DEFAULT_MODEL,
        messages=messages,
        response={
            "content": json.dumps({
                "mvts": 1,
                "subtasks": ["Review background documentation"],
                "rationale": "Fallback task.",
            })
        },
    )

    est = estimate(statement, gateway=gateway, model=DEFAULT_MODEL)
    assert est.mvts == 1
    assert "empty instruction before marker, fell back to whole statement" in est.justification

    logs = get_decision_log()
    assert logs[-1].inputs["context_chars_excluded"] == 0

