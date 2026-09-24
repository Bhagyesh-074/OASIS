"""OASIS LLM Gateway — intercepts all model traffic via LiteLLM.

Pre-call hook performs RBE admission control; post-call hook performs
token/cost accounting.  Prompt-hash JSONL sink enables deterministic
replay (FR-30).  FakeLLM provides a scripted stub for the entire test
suite (no network, no key).
"""

from oasis.gateway._hash import hash_request
from oasis.gateway.fake_llm import FakeLLM, FakeLLMCallError, FakeLLMError
from oasis.gateway.jsonl_sink import JSONLSink, scrub
from oasis.gateway.litellm_hooks import (
    Gateway,
    GatewayHaltError,
    UnpinnedModelError,
    Verdict,
    get_call_count,
    is_pinned_model,
    post_call_hook,
    pre_call_hook,
    reset_call_counter,
    validate_model_pinning,
)

__all__ = [
    "FakeLLM",
    "FakeLLMCallError",
    "FakeLLMError",
    "Gateway",
    "GatewayHaltError",
    "JSONLSink",
    "UnpinnedModelError",
    "Verdict",
    "get_call_count",
    "hash_request",
    "is_pinned_model",
    "post_call_hook",
    "pre_call_hook",
    "reset_call_counter",
    "scrub",
    "validate_model_pinning",
]

