"""Integration tests for gateway hooks wiring — pre→execute→post flow.

Tests the structural guarantee that every call goes through
pre_call_hook before post_call_hook, that the call counter increments
exactly once per call, and that accounting numbers match what FakeLLM
returned.

All tests use FakeLLM — no network, no API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis.gateway._hash import hash_request
from oasis.gateway.fake_llm import FakeLLM, FakeLLMCallError, FakeLLMError
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway, GatewayHaltError

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

MODEL = "gpt-4o-2024-08-06"
MSGS = [{"role": "user", "content": "What is 2+2?"}]
SCRIPTED_RESPONSE = {
    "content": "The answer is 4.",
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    "cost_usd": 0.00009,
    "latency_ms": 12.5,
}


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_calls.jsonl"


@pytest.fixture
def sink(log_path: Path) -> JSONLSink:
    return JSONLSink(log_path)


@pytest.fixture
def fake_llm() -> FakeLLM:
    fake = FakeLLM()
    fake.script(
        model=MODEL,
        messages=MSGS,
        response=SCRIPTED_RESPONSE,
    )
    return fake


@pytest.fixture
def gateway(sink: JSONLSink, fake_llm: FakeLLM) -> Gateway:
    return Gateway(sink=sink, llm=fake_llm, run_id="test-run-1", agent_id="agent-a")


# ==================================================================
# 1. Full pre→execute→post flow produces exactly one JSONL record
# ==================================================================


class TestFullCallFlow:
    """A call through Gateway.call() produces exactly one JSONL record."""

    def test_single_call_one_record(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """One call → exactly one line in the JSONL sink."""
        gateway.call(model=MODEL, messages=MSGS)

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_two_calls_two_records(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """Two calls → exactly two lines."""
        gateway.call(model=MODEL, messages=MSGS)
        gateway.call(model=MODEL, messages=MSGS)

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_record_prompt_sha_matches_hash_request(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """The prompt_sha in the JSONL record matches hash_request()."""
        expected_sha = hash_request(MODEL, MSGS)
        gateway.call(model=MODEL, messages=MSGS)

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["prompt_sha"] == expected_sha

    def test_record_has_run_and_agent_ids(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """The record carries the run_id and agent_id from the gateway."""
        gateway.call(model=MODEL, messages=MSGS)

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["run_id"] == "test-run-1"
        assert record["agent_id"] == "agent-a"

    def test_fallback_response_also_recorded(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Un-scripted (fallback) calls are also recorded in the sink."""
        fake = FakeLLM()  # no scripts, fallback enabled
        gw = Gateway(sink=sink, llm=fake)
        gw.call(model="some-model", messages=[{"role": "user", "content": "hi"}])

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_response_includes_verdict_metadata(
        self, gateway: Gateway
    ) -> None:
        """The response dict includes _verdict and _prompt_sha metadata."""
        resp = gateway.call(model=MODEL, messages=MSGS)
        assert resp["_verdict"] == "allow"
        assert resp["_prompt_sha"] == hash_request(MODEL, MSGS)


# ==================================================================
# 2. Call counter increments exactly once per call
# ==================================================================


class TestCallCounter:
    """The call counter increments exactly once per pre_call_hook."""

    def test_starts_at_zero(self, gateway: Gateway) -> None:
        assert gateway.call_count == 0

    def test_one_call_increments_to_one(self, gateway: Gateway) -> None:
        gateway.call(model=MODEL, messages=MSGS)
        assert gateway.call_count == 1

    def test_three_calls_increments_to_three(self, gateway: Gateway) -> None:
        for _ in range(3):
            gateway.call(model=MODEL, messages=MSGS)
        assert gateway.call_count == 3

    def test_counter_equals_jsonl_record_count(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """FR-7 invariant: call_count == number of JSONL records."""
        for _ in range(5):
            gateway.call(model=MODEL, messages=MSGS)

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert gateway.call_count == len(lines) == 5

    def test_counter_increments_even_on_error(
        self, sink: JSONLSink
    ) -> None:
        """Counter increments in pre-hook, before execution — so even
        a failing call (FakeLLMError) is counted."""
        fake = FakeLLM()
        fake.script(
            model=MODEL,
            messages=MSGS,
            response=FakeLLMError(error_type="timeout"),
        )
        gw = Gateway(sink=sink, llm=fake)

        with pytest.raises(FakeLLMCallError):
            gw.call(model=MODEL, messages=MSGS)

        # Counter incremented (the call was attempted)
        assert gw.call_count == 1

    def test_counter_increments_on_halt(
        self, sink: JSONLSink
    ) -> None:
        """Counter increments even when admission returns halt."""
        fake = FakeLLM()
        gw = Gateway(
            sink=sink,
            llm=fake,
            admission_fn=lambda _: {"verdict": "halt"},
        )

        with pytest.raises(GatewayHaltError):
            gw.call(model=MODEL, messages=MSGS)

        assert gw.call_count == 1

    def test_counter_increments_per_call_not_per_hook(
        self, gateway: Gateway
    ) -> None:
        """The counter increments exactly once per call, not per hook (pre + post)."""
        initial_count = gateway.call_count
        gateway.call(model=MODEL, messages=MSGS)
        # Even though pre_call_hook AND post_call_hook both ran, count increased by 1
        assert gateway.call_count == initial_count + 1


# ==================================================================
# 3. Accounting numbers match what FakeLLM returned
# ==================================================================


class TestAccountingAccuracy:
    """post_call_hook accounting numbers match FakeLLM's response exactly."""

    def test_tokens_match(self, gateway: Gateway) -> None:
        resp = gateway.call(model=MODEL, messages=MSGS)
        acct = resp["_accounting"]
        assert acct["tokens_in"] == SCRIPTED_RESPONSE["usage"]["prompt_tokens"]
        assert acct["tokens_out"] == SCRIPTED_RESPONSE["usage"]["completion_tokens"]

    def test_cost_matches(self, gateway: Gateway) -> None:
        resp = gateway.call(model=MODEL, messages=MSGS)
        acct = resp["_accounting"]
        assert acct["cost_usd"] == SCRIPTED_RESPONSE["cost_usd"]

    def test_latency_matches(self, gateway: Gateway) -> None:
        resp = gateway.call(model=MODEL, messages=MSGS)
        acct = resp["_accounting"]
        assert acct["latency_ms"] == SCRIPTED_RESPONSE["latency_ms"]

    def test_model_matches(self, gateway: Gateway) -> None:
        resp = gateway.call(model=MODEL, messages=MSGS)
        acct = resp["_accounting"]
        assert acct["model"] == MODEL

    def test_accounting_matches_jsonl_record(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """Accounting in the response matches what was written to the sink."""
        resp = gateway.call(model=MODEL, messages=MSGS)
        acct = resp["_accounting"]

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["tokens_in"] == acct["tokens_in"]
        assert record["tokens_out"] == acct["tokens_out"]
        assert record["cost_usd"] == acct["cost_usd"]
        assert record["latency_ms"] == acct["latency_ms"]

    def test_no_silent_transformation(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """The token counts in the JSONL record are exactly what FakeLLM
        returned — no rounding, scaling, or other transformation."""
        gateway.call(model=MODEL, messages=MSGS)

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["tokens_in"] == 5   # exact match, not transformed
        assert record["tokens_out"] == 4
        assert record["cost_usd"] == 0.00009
        assert record["latency_ms"] == 12.5

    def test_purpose_tag_recorded(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Purpose tag (FR-9, ADR-006) is recorded in the JSONL record."""
        fake = FakeLLM()
        fake.script(model=MODEL, messages=MSGS, response=SCRIPTED_RESPONSE)
        gw = Gateway(sink=sink, llm=fake, purpose="supervision")
        gw.call(model=MODEL, messages=MSGS)

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["purpose"] == "supervision"

    def test_purpose_override_per_call(
        self, gateway: Gateway, log_path: Path
    ) -> None:
        """Per-call purpose override takes precedence over gateway default."""
        gateway.call(model=MODEL, messages=MSGS, purpose="supervision")

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["purpose"] == "supervision"


# ==================================================================
# 4. Accounting log
# ==================================================================


class TestAccountingLog:
    """Gateway.accounting_log tracks all calls."""

    def test_initially_empty(self, gateway: Gateway) -> None:
        assert gateway.accounting_log == []

    def test_records_entries(self, gateway: Gateway) -> None:
        gateway.call(model=MODEL, messages=MSGS)
        assert len(gateway.accounting_log) == 1
        assert gateway.accounting_log[0]["model"] == MODEL

    def test_returns_copy(self, gateway: Gateway) -> None:
        gateway.call(model=MODEL, messages=MSGS)
        log = gateway.accounting_log
        log.clear()
        assert len(gateway.accounting_log) == 1


# ==================================================================
# 5. Halt verdict
# ==================================================================


class TestHaltVerdict:
    """GatewayHaltError is raised on halt verdict."""

    def test_halt_raises_error(self, sink: JSONLSink) -> None:
        fake = FakeLLM()
        gw = Gateway(
            sink=sink,
            llm=fake,
            admission_fn=lambda _: {"verdict": "halt"},
        )
        with pytest.raises(GatewayHaltError, match="halt"):
            gw.call(model=MODEL, messages=MSGS)

    def test_halt_does_not_write_to_sink(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """On halt, no JSONL record is written (the call never executed)."""
        fake = FakeLLM()
        gw = Gateway(
            sink=sink,
            llm=fake,
            admission_fn=lambda _: {"verdict": "halt"},
        )
        with pytest.raises(GatewayHaltError):
            gw.call(model=MODEL, messages=MSGS)

        assert not log_path.exists() or log_path.read_text().strip() == ""

    def test_halt_error_carries_call_data(self, sink: JSONLSink) -> None:
        fake = FakeLLM()
        gw = Gateway(
            sink=sink,
            llm=fake,
            admission_fn=lambda _: {"verdict": "halt"},
        )
        with pytest.raises(GatewayHaltError) as exc_info:
            gw.call(model=MODEL, messages=MSGS)

        assert exc_info.value.call_data["model"] == MODEL


# ==================================================================
# 6. Module-level functions still importable
# ==================================================================


class TestModuleLevelFunctions:
    """pre_call_hook and post_call_hook are still importable and callable."""

    def test_pre_call_hook_returns_allow(self) -> None:
        from oasis.gateway.litellm_hooks import pre_call_hook
        result = pre_call_hook({"model": MODEL, "messages": MSGS})
        assert result["verdict"] == "allow"

    def test_post_call_hook_extracts_accounting(self) -> None:
        from oasis.gateway.litellm_hooks import post_call_hook
        result = post_call_hook(
            {"model": MODEL, "messages": MSGS},
            SCRIPTED_RESPONSE,
        )
        assert result["tokens_in"] == 5
        assert result["tokens_out"] == 4
        assert result["cost_usd"] == 0.00009
        assert result["latency_ms"] == 12.5

    def test_pre_call_hook_verdict_string_and_dict_compatibility(self) -> None:
        from oasis.gateway.litellm_hooks import pre_call_hook
        result = pre_call_hook({"model": MODEL, "messages": MSGS})
        assert result == "allow"
        assert result["verdict"] == "allow"
        assert result.verdict == "allow"

    def test_pre_call_hook_increments_counter_post_call_hook_does_not(self) -> None:
        from oasis.gateway.litellm_hooks import (
            get_call_count,
            post_call_hook,
            pre_call_hook,
            reset_call_counter,
        )
        reset_call_counter()
        assert get_call_count() == 0
        pre_call_hook({"model": MODEL, "messages": MSGS})
        assert get_call_count() == 1
        # Calling post_call_hook must NOT increment call counter
        post_call_hook({"model": MODEL, "messages": MSGS}, SCRIPTED_RESPONSE)
        assert get_call_count() == 1

    def test_post_call_hook_writes_to_jsonl_sink(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        from oasis.gateway.litellm_hooks import post_call_hook
        req = {
            "model": MODEL,
            "messages": MSGS,
            "run_id": "test-run-standalone",
            "agent_id": "agent-standalone",
            "purpose": "supervision",
        }
        post_call_hook(req, SCRIPTED_RESPONSE, sink=sink)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["tokens_in"] == 5
        assert record["tokens_out"] == 4
        assert record["cost_usd"] == 0.00009
        assert record["latency_ms"] == 12.5
        assert record["run_id"] == "test-run-standalone"
        assert record["agent_id"] == "agent-standalone"
        assert record["purpose"] == "supervision"

