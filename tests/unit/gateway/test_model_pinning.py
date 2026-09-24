"""Unit tests for NFR-5: Model identifiers must be pinned to explicit versions;

aliases such as 'latest' are prohibited and rejected at the gateway.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import (
    Gateway,
    UnpinnedModelError,
    is_pinned_model,
    pre_call_hook,
    validate_model_pinning,
)


@pytest.fixture
def sink(tmp_path: Path) -> JSONLSink:
    return JSONLSink(tmp_path / "model_pinning.jsonl")


class TestModelPinningPureChecks:
    """Validation logic for NFR-5 model strings."""

    def test_pinned_model_strings_pass(self) -> None:
        """Pinned model versions pass validation without error."""
        pinned_models = [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-11-20",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ]
        for model in pinned_models:
            assert is_pinned_model(model) is True
            # Does not raise
            validate_model_pinning(model)

    def test_unpinned_models_with_latest_are_rejected(self) -> None:
        """Any model containing 'latest' is recognized as unpinned."""
        unpinned_models = [
            "gpt-4o-latest",
            "chatgpt-4o-latest",
            "claude-3-5-sonnet-latest",
            "gpt-4-latest",
            "model:latest",
        ]
        for model in unpinned_models:
            assert is_pinned_model(model) is False
            with pytest.raises(UnpinnedModelError, match="NFR-5 prohibits aliases"):
                validate_model_pinning(model)

    def test_empty_or_whitespace_model_rejected(self) -> None:
        """Empty or blank model strings are rejected."""
        assert is_pinned_model("") is False
        assert is_pinned_model("   ") is False
        with pytest.raises(UnpinnedModelError):
            validate_model_pinning("")


class TestGatewayModelPinningEnforcement:
    """Pre-call hook and Gateway.call enforcement of model pinning."""

    def test_pre_call_hook_rejects_latest_before_call(self) -> None:
        """pre_call_hook raises UnpinnedModelError when model contains 'latest'."""
        req = {
            "model": "gpt-4o-latest",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(UnpinnedModelError, match="unpinned"):
            pre_call_hook(req)

    def test_gateway_call_rejects_latest_without_executing_or_logging(
        self, sink: JSONLSink, tmp_path: Path
    ) -> None:
        """Gateway.call rejects unpinned model before execution and leaves sink empty."""
        fake = FakeLLM()
        gw = Gateway(sink=sink, llm=fake)

        with pytest.raises(UnpinnedModelError, match="NFR-5"):
            gw.call(
                model="gpt-4o-latest",
                messages=[{"role": "user", "content": "hello"}],
            )

        # Call counter was not incremented
        assert gw.call_count == 0
        # No sink file created or empty
        log_path = tmp_path / "model_pinning.jsonl"
        assert not log_path.exists() or log_path.read_text().strip() == ""

    def test_gateway_call_succeeds_with_pinned_model(self, sink: JSONLSink) -> None:
        """Gateway.call proceeds normally when model is pinned."""
        model = "gpt-4o-2024-08-06"
        fake = FakeLLM()
        fake.script(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            response={"content": "hello from pinned model", "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
        )

        gw = Gateway(sink=sink, llm=fake)
        resp = gw.call(model=model, messages=[{"role": "user", "content": "hello"}])
        assert resp["content"] == "hello from pinned model"
        assert gw.call_count == 1
