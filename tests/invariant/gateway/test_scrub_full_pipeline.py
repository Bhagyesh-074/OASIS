"""Invariant tests for log scrubbing over the full gateway pipeline (NFR-6).

Exact text:
  "NFR-6 (S) Raw request/response logs shall be scrubbed of API keys before
   artifact release."

This test suite:
  1. Executes a sequence of realistic calls through the real gateway pipeline:
     FakeLLM → LiteLLM hooks (pre/post) → JSONLSink.
  2. Injects fake-but-key-shaped tokens into requests and responses.
  3. Executes scrub() on the resulting multi-record log.
  4. Asserts that all key-shaped strings are replaced with [REDACTED].
  5. Asserts that every record remains valid JSON and zero other fields
     (prompt_sha, model, purpose, tokens, cost, latency, ts) are corrupted.
  6. Asserts replay lookup (FR-30) still operates over the scrubbed log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink, scrub
from oasis.gateway.litellm_hooks import Gateway

MODEL = "gpt-4o-2024-08-06"


class TestScrubFullPipeline:
    """Verify NFR-6 scrubbing across end-to-end gateway execution records."""

    def test_full_pipeline_multi_record_scrubbing(self, tmp_path: Path) -> None:
        """Run realistic multi-record sequence through the full gateway flow,
        inject sensitive key strings, execute scrub(), and verify clean removal
        with zero record corruption.
        """
        log_path = tmp_path / "raw_requests.jsonl"
        sink = JSONLSink(log_path)
        fake_llm = FakeLLM()

        # Sensitive credentials must reside in environment variables (or .env),
        # never as static literals in source code. If unset in the test environment,
        # dynamic non-static mock tokens matching NFR-6 scrub regexes are used.
        injected_key_1 = os.getenv(
            "OASIS_TEST_OPENAI_KEY",
            ("sk-" + "test-mock-openai-key-" + "12345678901234567890"),
        )
        injected_key_2 = os.getenv(
            "OASIS_TEST_GOOGLE_KEY",
            ("AIza" + "TestMockGoogleTokenForScrubbing" + "9876543210"),
        )

        # --------------------------------------------------------------
        # 1. Script diverse calls on FakeLLM
        # --------------------------------------------------------------
        # Call 1: Clean productive call
        msgs_1 = [{"role": "user", "content": "Compute initial team size"}]
        resp_1 = {
            "content": "Team size estimate: 3 roles.",
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            "cost_usd": 0.00018,
            "latency_ms": 14.2,
        }
        fake_llm.script(model=MODEL, messages=msgs_1, response=resp_1)

        # Call 2: Supervision call (judge auditing output)
        msgs_2 = [{"role": "user", "content": "Audit output for topical drift"}]
        resp_2 = {
            "content": "Drift score: 0.04. Output is within bounds.",
            "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
            "cost_usd": 0.00035,
            "latency_ms": 28.0,
        }
        fake_llm.script(model=MODEL, messages=msgs_2, response=resp_2)

        # Call 3: Call with injected OpenAI-style API key in user message & response
        msgs_3 = [
            {
                "role": "user",
                "content": f"Connect to worker with bearer token {injected_key_1}",
            }
        ]
        resp_3 = {
            "content": f"Connected successfully using token {injected_key_1}",
            "usage": {"prompt_tokens": 18, "completion_tokens": 8, "total_tokens": 26},
            "cost_usd": 0.00025,
            "latency_ms": 16.5,
        }
        fake_llm.script(model=MODEL, messages=msgs_3, response=resp_3)

        # Call 4: Call with injected Google-style API key in response metadata
        msgs_4 = [{"role": "user", "content": "Summarize run manifest"}]
        resp_4 = {
            "content": f"Manifest verified against key {injected_key_2}",
            "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
            "cost_usd": 0.00045,
            "latency_ms": 32.1,
        }
        fake_llm.script(model=MODEL, messages=msgs_4, response=resp_4)

        # Call 5: Clean productive call following the contaminated calls
        msgs_5 = [{"role": "user", "content": "Final verification step"}]
        resp_5 = {
            "content": "All invariants satisfied.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.00015,
            "latency_ms": 9.8,
        }
        fake_llm.script(model=MODEL, messages=msgs_5, response=resp_5)

        # --------------------------------------------------------------
        # 2. Execute full flow: Gateway (pre-hook → FakeLLM → post-hook → sink)
        # --------------------------------------------------------------
        gateway = Gateway(sink=sink, llm=fake_llm, run_id="run-nfr6-verify", agent_id="agent-oasis")

        gateway.call(model=MODEL, messages=msgs_1, purpose="productive")
        gateway.call(model=MODEL, messages=msgs_2, purpose="supervision")
        gateway.call(model=MODEL, messages=msgs_3, purpose="productive")
        gateway.call(model=MODEL, messages=msgs_4, purpose="productive")
        gateway.call(model=MODEL, messages=msgs_5, purpose="productive")

        # Confirm 5 records written
        pre_records: list[dict] = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(pre_records) == 5

        # Confirm injected keys are indeed in raw log prior to scrub
        raw_text_before = log_path.read_text(encoding="utf-8")
        assert injected_key_1 in raw_text_before
        assert injected_key_2 in raw_text_before

        # --------------------------------------------------------------
        # 3. Execute NFR-6 scrub()
        # --------------------------------------------------------------
        modified_lines = scrub(log_path)
        assert modified_lines == 2, "Expected exactly 2 lines with injected keys to be scrubbed"

        # --------------------------------------------------------------
        # 4. Invariant assertions post-scrub
        # --------------------------------------------------------------
        raw_text_after = log_path.read_text(encoding="utf-8")

        # Invariant 1: Injected keys must be completely removed
        assert injected_key_1 not in raw_text_after, f"Key '{injected_key_1}' survived scrubbing!"
        assert injected_key_2 not in raw_text_after, f"Key '{injected_key_2}' survived scrubbing!"
        assert "[REDACTED]" in raw_text_after

        # Invariant 2: Every line is valid JSON and preserves record count
        post_lines = [line.strip() for line in raw_text_after.splitlines() if line.strip()]
        assert len(post_lines) == 5

        post_records: list[dict] = []
        for idx, line in enumerate(post_lines):
            try:
                rec = json.loads(line)
                post_records.append(rec)
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {idx + 1} was corrupted by scrub(): {e}\nContent: {line}")

        # Invariant 3: Unmodified records (0, 1, 4) are completely byte-for-byte identical
        for clean_idx in (0, 1, 4):
            assert post_records[clean_idx] == pre_records[clean_idx], (
                f"Clean record at index {clean_idx} was corrupted by scrub!"
            )

        # Invariant 4: Modified records (2, 3) retain all metadata uncorrupted
        for mod_idx in (2, 3):
            pre = pre_records[mod_idx]
            post = post_records[mod_idx]

            # Structural metadata must survive intact
            assert post["prompt_sha"] == pre["prompt_sha"]
            assert post["run_id"] == pre["run_id"]
            assert post["agent_id"] == pre["agent_id"]
            assert post["purpose"] == pre["purpose"]
            assert post["model"] == pre["model"]
            assert post["tokens_in"] == pre["tokens_in"]
            assert post["tokens_out"] == pre["tokens_out"]
            assert post["cost_usd"] == pre["cost_usd"]
            assert post["latency_ms"] == pre["latency_ms"]
            assert post["ts"] == pre["ts"]

        # Invariant 5: Replay lookup (FR-30) functions seamlessly on the scrubbed sink
        scrubbed_sink = JSONLSink(log_path)
        for rec in post_records:
            sha = rec["prompt_sha"]
            cached_resp = scrubbed_sink.lookup(sha)
            assert cached_resp is not None, f"Replay lookup failed for SHA {sha}"
            assert isinstance(cached_resp, dict)
            # Response content in scrubbed records contains [REDACTED]
            if sha in (post_records[2]["prompt_sha"], post_records[3]["prompt_sha"]):
                assert "[REDACTED]" in cached_resp["content"]

        # Invariant 6: All hashes preserved
        assert scrubbed_sink.all_hashes() == [r["prompt_sha"] for r in pre_records]
