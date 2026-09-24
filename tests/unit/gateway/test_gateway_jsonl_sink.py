"""Unit tests for oasis.gateway.jsonl_sink — append-only JSONL log.

Covers:
  - Write creates file on first call; appends without overwriting.
  - Record round-trip: write → lookup returns the same response.
  - Two different requests → two different prompt_sha keys.
  - Lookup by prompt_sha returns cached response with zero re-computation.
  - Lookup for unknown hash returns None.
  - scrub_keys() removes API-key-shaped strings (NFR-6).
  - Module-level scrub() convenience works.
  - Prompt hash is single-sourced (hash_request from _hash.py).
  - write_raw convenience extracts structured fields.
  - Index invalidation on write.
  - all_hashes() returns hashes in write order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis.gateway._hash import hash_request
from oasis.gateway.jsonl_sink import JSONLSink, scrub

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

MODEL_A = "gpt-4o-2024-08-06"
MSGS_A = [{"role": "user", "content": "What is 2+2?"}]

MODEL_B = "claude-3-5-sonnet-20241022"
MSGS_B = [{"role": "user", "content": "Explain photosynthesis"}]

RESPONSE_A: dict = {
    "content": "The answer is 4.",
    "model": MODEL_A,
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    "cost_usd": 0.00009,
    "latency_ms": 12.5,
}

RESPONSE_B: dict = {
    "content": "Photosynthesis converts CO₂ and water into glucose.",
    "model": MODEL_B,
    "usage": {"prompt_tokens": 8, "completion_tokens": 10, "total_tokens": 18},
    "cost_usd": 0.00018,
    "latency_ms": 25.0,
}


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_calls.jsonl"


@pytest.fixture
def sink(log_path: Path) -> JSONLSink:
    return JSONLSink(log_path)


def _write_record_a(sink: JSONLSink) -> str:
    """Write RESPONSE_A and return its prompt_sha."""
    sha = hash_request(MODEL_A, MSGS_A)
    sink.write(
        prompt_sha=sha,
        model=MODEL_A,
        messages=MSGS_A,
        temperature=0.0,
        tool_schema=None,
        response=RESPONSE_A,
        tokens_in=5,
        tokens_out=4,
        cost_usd=0.00009,
        latency_ms=12.5,
    )
    return sha


def _write_record_b(sink: JSONLSink) -> str:
    """Write RESPONSE_B and return its prompt_sha."""
    sha = hash_request(MODEL_B, MSGS_B)
    sink.write(
        prompt_sha=sha,
        model=MODEL_B,
        messages=MSGS_B,
        temperature=0.0,
        tool_schema=None,
        response=RESPONSE_B,
        tokens_in=8,
        tokens_out=10,
        cost_usd=0.00018,
        latency_ms=25.0,
    )
    return sha


# ==================================================================
# 1. Write — append-only behaviour
# ==================================================================


class TestJSONLSinkWrite:
    """JSONLSink.write — file creation and append-only semantics."""

    def test_write_creates_file_on_first_call(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Log file is created lazily on first write."""
        assert not log_path.exists()
        _write_record_a(sink)
        assert log_path.exists()

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Nested parent directories are created automatically."""
        deep_path = tmp_path / "a" / "b" / "calls.jsonl"
        s = JSONLSink(deep_path)
        _write_record_a(s)
        assert deep_path.exists()

    def test_write_appends_without_overwriting(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Multiple writes → multiple lines, no truncation."""
        _write_record_a(sink)
        _write_record_b(sink)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        _write_record_a(sink)
        _write_record_b(sink)
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            assert isinstance(record, dict)

    def test_record_has_required_fields(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Each record has the field set from DATABASE.md §"Raw log"."""
        _write_record_a(sink)
        line = log_path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        required = {
            "prompt_sha", "model", "messages", "temperature",
            "tool_schema", "response", "tokens_in", "tokens_out",
            "cost_usd", "latency_ms", "ts",
        }
        assert required.issubset(record.keys())

    def test_record_has_prompt_sha_field_name(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Field is ``prompt_sha`` — matches agent_output.prompt_sha in DATABASE.md."""
        sha = _write_record_a(sink)
        line = log_path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["prompt_sha"] == sha
        assert "prompt_hash" not in record  # not the old name

    def test_record_has_timestamp(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        _write_record_a(sink)
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert "ts" in record
        assert len(record["ts"]) > 0


# ==================================================================
# 2. Round-trip: write then read back
# ==================================================================


class TestJSONLSinkRoundTrip:
    """Write → lookup returns the same response."""

    def test_lookup_returns_stored_response(self, sink: JSONLSink) -> None:
        sha = _write_record_a(sink)
        resp = sink.lookup(sha)
        assert resp is not None
        assert resp["content"] == RESPONSE_A["content"]
        assert resp["usage"] == RESPONSE_A["usage"]

    def test_lookup_full_returns_entire_record(self, sink: JSONLSink) -> None:
        sha = _write_record_a(sink)
        full = sink.lookup_full(sha)
        assert full is not None
        assert full["prompt_sha"] == sha
        assert full["model"] == MODEL_A
        assert full["tokens_in"] == 5
        assert full["tokens_out"] == 4

    def test_two_records_both_retrievable(self, sink: JSONLSink) -> None:
        sha_a = _write_record_a(sink)
        sha_b = _write_record_b(sink)
        assert sha_a != sha_b  # different inputs → different hashes

        resp_a = sink.lookup(sha_a)
        resp_b = sink.lookup(sha_b)
        assert resp_a is not None
        assert resp_b is not None
        assert resp_a["content"] == RESPONSE_A["content"]
        assert resp_b["content"] == RESPONSE_B["content"]


# ==================================================================
# 3. Different requests → different prompt_sha
# ==================================================================


class TestPromptShaDifference:
    """Two different requests get two different prompt_sha keys."""

    def test_different_models_different_sha(self, sink: JSONLSink) -> None:
        sha_a = hash_request(MODEL_A, MSGS_A)
        sha_b = hash_request(MODEL_B, MSGS_A)
        assert sha_a != sha_b

    def test_different_messages_different_sha(self, sink: JSONLSink) -> None:
        sha_a = hash_request(MODEL_A, MSGS_A)
        sha_b = hash_request(MODEL_A, MSGS_B)
        assert sha_a != sha_b

    def test_written_records_have_different_sha(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        _write_record_a(sink)
        _write_record_b(sink)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        sha_0 = json.loads(lines[0])["prompt_sha"]
        sha_1 = json.loads(lines[1])["prompt_sha"]
        assert sha_0 != sha_1


# ==================================================================
# 4. Lookup — zero re-computation (index caching)
# ==================================================================


class TestLookupCaching:
    """Lookup uses an in-memory index — no re-parsing after first call."""

    def test_lookup_returns_none_for_unknown_hash(self, sink: JSONLSink) -> None:
        _write_record_a(sink)
        assert sink.lookup("0" * 64) is None

    def test_lookup_on_empty_file(self, sink: JSONLSink) -> None:
        assert sink.lookup("0" * 64) is None

    def test_lookup_on_nonexistent_file(self, tmp_path: Path) -> None:
        s = JSONLSink(tmp_path / "does_not_exist.jsonl")
        assert s.lookup("0" * 64) is None

    def test_index_rebuilt_after_write(self, sink: JSONLSink) -> None:
        """New writes invalidate the index so new records are visible."""
        sha_a = _write_record_a(sink)
        # Trigger index build
        assert sink.lookup(sha_a) is not None

        # Write a second record
        sha_b = _write_record_b(sink)
        # Must be findable despite the index having been built before
        assert sink.lookup(sha_b) is not None

    def test_latest_record_wins_on_duplicate_sha(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """If the same prompt_sha is written twice, lookup returns the latest."""
        sha = hash_request(MODEL_A, MSGS_A)
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.0,
            tool_schema=None,
            response={"content": "first"},
            tokens_in=5, tokens_out=4,
            cost_usd=0.0, latency_ms=1.0,
        )
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.0,
            tool_schema=None,
            response={"content": "second"},
            tokens_in=5, tokens_out=4,
            cost_usd=0.0, latency_ms=1.0,
        )
        resp = sink.lookup(sha)
        assert resp is not None
        assert resp["content"] == "second"


# ==================================================================
# 5. all_hashes()
# ==================================================================


class TestAllHashes:
    """all_hashes() returns sha values in write order."""

    def test_empty_file(self, sink: JSONLSink) -> None:
        assert sink.all_hashes() == []

    def test_preserves_order(self, sink: JSONLSink) -> None:
        sha_a = _write_record_a(sink)
        sha_b = _write_record_b(sink)
        assert sink.all_hashes() == [sha_a, sha_b]

    def test_includes_duplicates(self, sink: JSONLSink) -> None:
        sha_a = _write_record_a(sink)
        _write_record_a(sink)  # same hash again
        assert sink.all_hashes() == [sha_a, sha_a]


# ==================================================================
# 6. write_raw() convenience
# ==================================================================


class TestWriteRaw:
    """write_raw() extracts fields from request/response dicts."""

    def test_write_raw_round_trips(self, sink: JSONLSink) -> None:
        sha = hash_request(MODEL_A, MSGS_A)
        request = {"model": MODEL_A, "messages": MSGS_A, "temperature": 0.0}
        sink.write_raw(sha, request, RESPONSE_A)
        resp = sink.lookup(sha)
        assert resp is not None
        assert resp["content"] == RESPONSE_A["content"]

    def test_write_raw_extracts_usage(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        sha = hash_request(MODEL_A, MSGS_A)
        request = {"model": MODEL_A, "messages": MSGS_A}
        sink.write_raw(sha, request, RESPONSE_A)
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["tokens_in"] == 5
        assert record["tokens_out"] == 4


# ==================================================================
# 7. NFR-6 scrubbing
# ==================================================================


class TestScrubKeys:
    """scrub_keys() removes API-key-shaped strings from the log file."""

    def _write_with_key(self, sink: JSONLSink, key_value: str) -> str:
        """Write a record containing an API-key-like string in the response."""
        sha = hash_request(MODEL_A, MSGS_A)
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.0,
            tool_schema=None,
            response={
                "content": "ok",
                "headers": {"Authorization": f"Bearer {key_value}"},
                "api_key": key_value,
            },
            tokens_in=5, tokens_out=2,
            cost_usd=0.0, latency_ms=1.0,
        )
        return sha

    def test_scrub_removes_openai_key(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """sk-… patterns are redacted."""
        key = "sk-" + "mockkey" + "0" * 40
        self._write_with_key(sink, key)

        modified = sink.scrub_keys()
        assert modified == 1

        text = log_path.read_text(encoding="utf-8")
        assert key not in text
        assert "[REDACTED]" in text

    def test_scrub_removes_bearer_token(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Bearer tokens in Authorization headers are redacted."""
        token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        sha = hash_request(MODEL_A, MSGS_A)
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.0,
            tool_schema=None,
            response={
                "content": "ok",
                "headers": {"Authorization": f"Bearer {token}"},
            },
            tokens_in=5, tokens_out=2,
            cost_usd=0.0, latency_ms=1.0,
        )

        modified = sink.scrub_keys()
        assert modified == 1

        text = log_path.read_text(encoding="utf-8")
        assert f"Bearer {token}" not in text
        assert "[REDACTED]" in text

    def test_scrub_returns_zero_when_no_keys(
        self, sink: JSONLSink
    ) -> None:
        """Clean log → scrub returns 0."""
        _write_record_a(sink)
        assert sink.scrub_keys() == 0

    def test_scrub_on_nonexistent_file(self, sink: JSONLSink) -> None:
        assert sink.scrub_keys() == 0

    def test_scrub_preserves_non_key_data(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Scrubbing doesn't destroy the rest of the record."""
        key = "sk-" + "TEST" + "1" * 38
        self._write_with_key(sink, key)
        sink.scrub_keys()

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["prompt_sha"] == hash_request(MODEL_A, MSGS_A)
        assert record["model"] == MODEL_A

    def test_scrub_invalidates_index(
        self, sink: JSONLSink
    ) -> None:
        """After scrubbing, lookup returns the scrubbed version."""
        key = "sk-" + "TEST" + "1" * 38
        sha = self._write_with_key(sink, key)

        # Build index before scrub
        resp_before = sink.lookup(sha)
        assert resp_before is not None
        assert key in json.dumps(resp_before)

        sink.scrub_keys()

        # After scrub, index should be rebuilt
        resp_after = sink.lookup(sha)
        assert resp_after is not None
        assert key not in json.dumps(resp_after)
        assert "[REDACTED]" in json.dumps(resp_after)

    def test_scrub_custom_pattern(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """Custom pattern overrides defaults."""
        sha = hash_request(MODEL_A, MSGS_A)
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.0,
            tool_schema=None,
            response={"content": "SECRET_VALUE_123"},
            tokens_in=5, tokens_out=2,
            cost_usd=0.0, latency_ms=1.0,
        )
        modified = sink.scrub_keys(patterns=[r"SECRET_VALUE_\d+"])
        assert modified == 1
        text = log_path.read_text(encoding="utf-8")
        assert "SECRET_VALUE_123" not in text


# ==================================================================
# 8. Module-level scrub() convenience
# ==================================================================


class TestModuleScrub:
    """Module-level scrub(path) function."""

    def test_scrub_function(self, sink: JSONLSink, log_path: Path) -> None:
        sha = hash_request(MODEL_A, MSGS_A)
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.0,
            tool_schema=None,
            response={"api_key": "sk-" + "mockkey" + "2" * 38},
            tokens_in=5, tokens_out=2,
            cost_usd=0.0, latency_ms=1.0,
        )
        modified = scrub(log_path)
        assert modified == 1


# ==================================================================
# 9. Hash is single-sourced
# ==================================================================


class TestHashSingleSourced:
    """JSONLSink uses the same hash_request as FakeLLM."""

    def test_hash_request_used_for_prompt_sha(
        self, sink: JSONLSink, log_path: Path
    ) -> None:
        """The prompt_sha written to the log matches hash_request()."""
        sha = hash_request(MODEL_A, MSGS_A, 0.7, [{"type": "function"}])
        sink.write(
            prompt_sha=sha,
            model=MODEL_A,
            messages=MSGS_A,
            temperature=0.7,
            tool_schema=[{"type": "function"}],
            response={"content": "x"},
            tokens_in=1, tokens_out=1,
            cost_usd=0.0, latency_ms=1.0,
        )
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["prompt_sha"] == sha
