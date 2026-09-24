"""Unit tests for oasis.gateway.fake_llm — FakeLLM deterministic stub.

Covers:
  - Prompt hash determinism and sensitivity to each input field.
  - Pre-scripted response exact matching.
  - Deterministic fallback for un-scripted calls.
  - Response shape (content, usage, cost_usd, latency_ms, model).
  - Error-mode (FakeLLMError / FakeLLMCallError) for always-failing agents.
  - Call log inspection.
  - Strict mode (default_fallback=False).
  - hash_request single-sourced consistency.
"""

from __future__ import annotations

import pytest

from oasis.gateway._hash import hash_request
from oasis.gateway.fake_llm import (
    FakeLLM,
    FakeLLMCallError,
    FakeLLMError,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_MODEL = "gpt-4o-2024-08-06"
SAMPLE_MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
SAMPLE_TEMPERATURE = 0.0
SAMPLE_TOOL_SCHEMA: list[dict] = [
    {"type": "function", "function": {"name": "add", "parameters": {}}}
]

ALT_MODEL = "claude-3-5-sonnet-20241022"
ALT_MESSAGES = [{"role": "user", "content": "What is 3+3?"}]


@pytest.fixture
def fake() -> FakeLLM:
    """A default FakeLLM with fallback enabled."""
    return FakeLLM()


@pytest.fixture
def strict_fake() -> FakeLLM:
    """A FakeLLM with fallback disabled (strict mode)."""
    return FakeLLM(default_fallback=False)


# ==================================================================
# 1. Prompt hash — determinism
# ==================================================================


class TestPromptHashDeterminism:
    """Same input → same hash → reproducible (ADR-015)."""

    def test_same_inputs_same_hash(self, fake: FakeLLM) -> None:
        """Calling prompt_hash twice with identical args gives the same digest."""
        h1 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, SAMPLE_TEMPERATURE)
        h2 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, SAMPLE_TEMPERATURE)
        assert h1 == h2

    def test_hash_is_64_char_hex(self, fake: FakeLLM) -> None:
        """SHA-256 hex digest is exactly 64 lowercase hex characters."""
        h = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_stable_across_instances(self) -> None:
        """Two independent FakeLLM instances produce the same hash for the same input."""
        h1 = FakeLLM().prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        h2 = FakeLLM().prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert h1 == h2

    def test_tool_schema_none_vs_empty_list_same_hash(self, fake: FakeLLM) -> None:
        """None and [] for tool_schema produce the same hash (normalised)."""
        h1 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.0, None)
        h2 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.0, [])
        assert h1 == h2


# ==================================================================
# 2. Prompt hash — sensitivity to each field
# ==================================================================


class TestPromptHashSensitivity:
    """Different inputs → different hash."""

    def test_different_model(self, fake: FakeLLM) -> None:
        h1 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        h2 = fake.prompt_hash(ALT_MODEL, SAMPLE_MESSAGES)
        assert h1 != h2

    def test_different_messages(self, fake: FakeLLM) -> None:
        h1 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        h2 = fake.prompt_hash(SAMPLE_MODEL, ALT_MESSAGES)
        assert h1 != h2

    def test_different_temperature(self, fake: FakeLLM) -> None:
        h1 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.0)
        h2 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.7)
        assert h1 != h2

    def test_different_tool_schema(self, fake: FakeLLM) -> None:
        h1 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.0, None)
        h2 = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.0, SAMPLE_TOOL_SCHEMA)
        assert h1 != h2


# ==================================================================
# 3. Single-sourced hash consistency
# ==================================================================


class TestHashSingleSourced:
    """FakeLLM.prompt_hash must use the same implementation as hash_request."""

    def test_matches_hash_request(self) -> None:
        h_fake = FakeLLM.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.7, SAMPLE_TOOL_SCHEMA)
        h_util = hash_request(SAMPLE_MODEL, SAMPLE_MESSAGES, 0.7, SAMPLE_TOOL_SCHEMA)
        assert h_fake == h_util

    def test_matches_hash_request_defaults(self) -> None:
        h_fake = FakeLLM.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        h_util = hash_request(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert h_fake == h_util


# ==================================================================
# 4. Pre-scripted responses — exact match
# ==================================================================


class TestPreScripted:
    """Pre-scripted response is returned exactly when its hash matches."""

    def test_scripted_response_returned(self, fake: FakeLLM) -> None:
        """Scripted content is returned verbatim."""
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "The answer is 4."},
        )
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["content"] == "The answer is 4."

    def test_scripted_custom_fields_preserved(self, fake: FakeLLM) -> None:
        """Extra fields in the scripted response survive."""
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "ok", "custom_flag": True},
        )
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["custom_flag"] is True

    def test_scripted_fills_missing_usage(self, fake: FakeLLM) -> None:
        """If the script omits usage, FakeLLM fills it in."""
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "filled"},
        )
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert "usage" in resp
        assert resp["usage"]["total_tokens"] > 0

    def test_scripted_preserves_explicit_usage(self, fake: FakeLLM) -> None:
        """If the script provides usage, it is kept."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "exact", "usage": usage},
        )
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["usage"] == usage

    def test_script_returns_hash(self, fake: FakeLLM) -> None:
        """script() returns the prompt hash for test assertions."""
        h = fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "x"},
        )
        assert h == fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)

    def test_script_hash_direct(self, fake: FakeLLM) -> None:
        """script_hash() registers by pre-computed hash."""
        h = fake.prompt_hash(SAMPLE_MODEL, SAMPLE_MESSAGES)
        fake.script_hash(h, {"content": "direct"})
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["content"] == "direct"

    def test_scripted_does_not_mutate_original(self, fake: FakeLLM) -> None:
        """The original scripted dict is not modified by complete()."""
        original = {"content": "immutable"}
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response=original,
        )
        fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert "usage" not in original  # complete() added usage to a copy, not the original


# ==================================================================
# 5. Deterministic fallback — un-scripted calls
# ==================================================================


class TestFallback:
    """Un-scripted input returns a valid, well-shaped response (not an exception)."""

    def test_fallback_returns_dict(self, fake: FakeLLM) -> None:
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert isinstance(resp, dict)

    def test_fallback_has_content(self, fake: FakeLLM) -> None:
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert isinstance(resp["content"], str)
        assert len(resp["content"]) > 0

    def test_fallback_is_deterministic(self, fake: FakeLLM) -> None:
        """Same input → same fallback response."""
        r1 = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        r2 = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert r1["content"] == r2["content"]
        assert r1["usage"] == r2["usage"]

    def test_fallback_different_input_different_content(self, fake: FakeLLM) -> None:
        """Different inputs → different fallback content."""
        r1 = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        r2 = fake.complete(ALT_MODEL, ALT_MESSAGES)
        assert r1["content"] != r2["content"]


# ==================================================================
# 6. Response shape — realistic fields
# ==================================================================


class TestResponseShape:
    """Every response (scripted or fallback) has the expected shape."""

    @pytest.fixture(params=["fallback", "scripted"])
    def response(self, request: pytest.FixtureRequest) -> dict:
        fake = FakeLLM()
        if request.param == "scripted":
            fake.script(
                model=SAMPLE_MODEL,
                messages=SAMPLE_MESSAGES,
                response={"content": "scripted answer"},
            )
        return fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)

    def test_has_content(self, response: dict) -> None:
        assert "content" in response

    def test_has_model(self, response: dict) -> None:
        assert response["model"] == SAMPLE_MODEL

    def test_has_usage_with_token_counts(self, response: dict) -> None:
        usage = response["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_has_cost_usd(self, response: dict) -> None:
        assert isinstance(response["cost_usd"], float)
        assert response["cost_usd"] >= 0

    def test_has_latency_ms(self, response: dict) -> None:
        assert isinstance(response["latency_ms"], float)
        assert response["latency_ms"] > 0

    def test_token_counts_positive(self, response: dict) -> None:
        assert response["usage"]["prompt_tokens"] >= 1
        assert response["usage"]["completion_tokens"] >= 1


# ==================================================================
# 7. Error mode — FakeLLMError / FakeLLMCallError
# ==================================================================


class TestErrorMode:
    """FakeLLMError sentinel makes complete() raise FakeLLMCallError."""

    def test_error_raises_call_error(self, fake: FakeLLM) -> None:
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response=FakeLLMError(error_type="rate_limit", message="429 Too Many Requests"),
        )
        with pytest.raises(FakeLLMCallError, match="429 Too Many Requests"):
            fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)

    def test_error_type_carried(self, fake: FakeLLM) -> None:
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response=FakeLLMError(error_type="timeout"),
        )
        with pytest.raises(FakeLLMCallError) as exc_info:
            fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert exc_info.value.error_type == "timeout"

    def test_error_default_fields(self) -> None:
        """FakeLLMError defaults are reasonable."""
        err = FakeLLMError()
        assert err.error_type == "provider_error"
        assert "FakeLLM" in err.message

    def test_error_logged_in_call_log(self, fake: FakeLLM) -> None:
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response=FakeLLMError(),
        )
        with pytest.raises(FakeLLMCallError):
            fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert fake.call_count == 1
        assert "error" in fake.call_log[0]


# ==================================================================
# 8. Strict mode — default_fallback=False
# ==================================================================


class TestStrictMode:
    """With default_fallback=False, un-scripted calls raise KeyError."""

    def test_unscripted_raises_keyerror(self, strict_fake: FakeLLM) -> None:
        with pytest.raises(KeyError):
            strict_fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)

    def test_scripted_still_works(self, strict_fake: FakeLLM) -> None:
        strict_fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "strict ok"},
        )
        resp = strict_fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["content"] == "strict ok"


# ==================================================================
# 9. Call log
# ==================================================================


class TestCallLog:
    """Call log tracks all invocations for test assertions."""

    def test_initially_empty(self, fake: FakeLLM) -> None:
        assert fake.call_count == 0
        assert fake.call_log == []

    def test_records_calls(self, fake: FakeLLM) -> None:
        fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        fake.complete(ALT_MODEL, ALT_MESSAGES)
        assert fake.call_count == 2

    def test_log_entry_has_hash_and_model(self, fake: FakeLLM) -> None:
        fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        entry = fake.call_log[0]
        assert "prompt_hash" in entry
        assert entry["model"] == SAMPLE_MODEL

    def test_reset_clears_log_keeps_scripts(self, fake: FakeLLM) -> None:
        fake.script(
            model=SAMPLE_MODEL,
            messages=SAMPLE_MESSAGES,
            response={"content": "kept"},
        )
        fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert fake.call_count == 1

        fake.reset()
        assert fake.call_count == 0

        # Script is still there
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["content"] == "kept"

    def test_call_log_returns_copy(self, fake: FakeLLM) -> None:
        """Mutating the returned list doesn't affect the internal log."""
        fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        log = fake.call_log
        log.clear()
        assert fake.call_count == 1  # internal log unaffected


# ==================================================================
# 10. Constructor with pre-built scripted_responses dict
# ==================================================================


class TestConstructorScripting:
    """FakeLLM(scripted_responses={hash: resp}) works."""

    def test_constructor_scripted(self) -> None:
        h = hash_request(SAMPLE_MODEL, SAMPLE_MESSAGES)
        fake = FakeLLM(scripted_responses={h: {"content": "ctor"}})
        resp = fake.complete(SAMPLE_MODEL, SAMPLE_MESSAGES)
        assert resp["content"] == "ctor"

    def test_constructor_does_not_mutate_input(self) -> None:
        h = hash_request(SAMPLE_MODEL, SAMPLE_MESSAGES)
        original = {h: {"content": "orig"}}
        fake = FakeLLM(scripted_responses=original)
        # Registering a new script should not affect the original dict
        fake.script(model=ALT_MODEL, messages=ALT_MESSAGES, response={"content": "new"})
        assert len(original) == 1
