"""Scripted, prompt-hash keyed stub provider — no network, no key.

What every other component's tests run against.  Responses are looked up
by a SHA-256 hash over (model, messages, temperature, tool_schema) so
that the same logical request always returns the same scripted response.

This is the foundation of the determinism strategy (ADR-015) and the
replay-purity invariant test (FR-30).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oasis.gateway._hash import hash_request

# ------------------------------------------------------------------
# Error-mode sentinel
# ------------------------------------------------------------------

@dataclass(frozen=True)
class FakeLLMError:
    """Sentinel value that can be registered as a scripted response to
    make :meth:`FakeLLM.complete` raise a :class:`FakeLLMCallError`.

    Use this for integration tests that need "an always-failing agent"
    (TESTING.md).

    Parameters
    ----------
    error_type:
        A short tag like ``"rate_limit"``, ``"timeout"``,
        ``"content_filter"`` — carried on the raised exception.
    message:
        Human-readable error message.
    """

    error_type: str = "provider_error"
    message: str = "Simulated provider error from FakeLLM"


class FakeLLMCallError(Exception):
    """Raised by :meth:`FakeLLM.complete` when the scripted response
    for a hash is a :class:`FakeLLMError` sentinel.
    """

    def __init__(self, error: FakeLLMError) -> None:
        self.error_type = error.error_type
        super().__init__(error.message)


# ------------------------------------------------------------------
# Response helpers
# ------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-family heuristic).

    Deterministic — no randomness.
    """
    return max(1, len(text) // 4)


def _build_response(
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> dict[str, Any]:
    """Build a realistically-shaped LLM response dict.

    Shape mirrors the common fields teammates' cost/quality code will
    read: ``content``, ``usage.{prompt_tokens, completion_tokens,
    total_tokens}``, ``cost_usd``, ``latency_ms``, ``model``.

    All numeric values are deterministic (derived from input lengths),
    never random.
    """
    total_tokens = prompt_tokens + completion_tokens
    # Fake cost: $0.01 per 1k tokens (deterministic, plausible order of magnitude)
    cost_usd = round(total_tokens * 0.00001, 8)
    # Fake latency: 10ms per 100 completion tokens (deterministic)
    latency_ms = max(1.0, completion_tokens / 100.0 * 10.0)

    return {
        "content": content,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "cost_usd": cost_usd,
        "latency_ms": round(latency_ms, 2),
    }


# ------------------------------------------------------------------
# FakeLLM
# ------------------------------------------------------------------

class FakeLLM:
    """Deterministic LLM stub for testing — no network, no API key.

    Three modes of operation:

    1. **Pre-scripted**: Register exact responses for specific prompt
       hashes (or input tuples).  When ``complete()`` sees a matching
       hash, the pre-scripted response is returned verbatim.

    2. **Error mode**: Register a :class:`FakeLLMError` sentinel for a
       hash.  ``complete()`` will raise :class:`FakeLLMCallError` —
       useful for "always-failing agent" fixtures (TESTING.md).

    3. **Deterministic fallback**: If no script matches, a
       well-shaped response is generated deterministically from the
       hash itself (echo-style), so tests never crash on un-scripted
       calls.

    Usage
    -----
    >>> fake = FakeLLM()
    >>> resp = fake.complete(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": "hi"}])
    >>> resp["content"]  # deterministic fallback
    '...'
    >>> resp["usage"]["total_tokens"] > 0
    True

    Pre-scripting:

    >>> fake.script(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": "hi"}],
    ...             response={"content": "hello!", "custom_field": 42})
    >>> fake.complete(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": "hi"}])["content"]
    'hello!'
    """

    def __init__(
        self,
        scripted_responses: dict[str, Any] | None = None,
        *,
        default_fallback: bool = True,
    ) -> None:
        """Initialise with an optional mapping of prompt-hash → response.

        Parameters
        ----------
        scripted_responses:
            ``{sha256_hex: response_dict_or_FakeLLMError}``.
        default_fallback:
            If ``True`` (default), un-scripted calls return a
            deterministic fallback response.  If ``False``, un-scripted
            calls raise ``KeyError`` (strict mode for tests that need
            to assert every call is pre-scripted).
        """
        self._responses: dict[str, Any] = dict(scripted_responses or {})
        self._default_fallback = default_fallback
        self._call_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Hashing (delegates to the single-sourced helper)
    # ------------------------------------------------------------------

    @staticmethod
    def prompt_hash(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tool_schema: list[dict[str, Any]] | None = None,
    ) -> str:
        """Compute the canonical SHA-256 prompt hash.

        Delegates to :func:`oasis.gateway._hash.hash_request` — the
        single-sourced implementation shared with ``JSONLSink`` and
        replay infrastructure.

        Parameters
        ----------
        model:
            Exact model identifier (no aliases — NFR-5).
        messages:
            Full message list.
        temperature:
            Call temperature (default 0.0).
        tool_schema:
            Optional tool/function schemas.

        Returns
        -------
        64-character lowercase hex SHA-256 digest.
        """
        return hash_request(model, messages, temperature, tool_schema)

    # ------------------------------------------------------------------
    # Scripting
    # ------------------------------------------------------------------

    def script(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tool_schema: list[dict[str, Any]] | None = None,
        response: dict[str, Any] | FakeLLMError,
    ) -> str:
        """Register a scripted response for specific call parameters.

        This is a convenience wrapper around direct hash-keyed
        registration: it computes the hash for you.

        Parameters
        ----------
        model, messages, temperature, tool_schema:
            The call parameters to match.
        response:
            The response dict to return (or a :class:`FakeLLMError`
            to trigger an error).

        Returns
        -------
        The prompt hash that was registered (useful for assertions).
        """
        h = self.prompt_hash(model, messages, temperature, tool_schema)
        self._responses[h] = response
        return h

    def script_hash(self, prompt_hash: str, response: dict[str, Any] | FakeLLMError) -> None:
        """Register a scripted response by pre-computed hash.

        Parameters
        ----------
        prompt_hash:
            The SHA-256 hex digest to match.
        response:
            The response dict or error sentinel.
        """
        self._responses[prompt_hash] = response

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tool_schema: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the scripted (or fallback) response — never makes a network call.

        Resolution order:

        1. If a scripted :class:`FakeLLMError` is registered for the
           hash, raise :class:`FakeLLMCallError`.
        2. If a scripted response dict is registered, return it
           (after ensuring ``usage``, ``cost_usd`` and ``latency_ms``
           fields are present with defaults if the script omitted them).
        3. If ``default_fallback`` is ``True``, build a deterministic
           echo-style response from the hash.
        4. Otherwise raise ``KeyError``.

        Parameters
        ----------
        model:
            Exact model identifier.
        messages:
            Full message list.
        temperature:
            Call temperature.
        tool_schema:
            Optional tool/function schemas.

        Returns
        -------
        dict with at minimum ``content``, ``model``, ``usage``,
        ``cost_usd``, ``latency_ms``.

        Raises
        ------
        FakeLLMCallError
            If the scripted response is a :class:`FakeLLMError`.
        KeyError
            If ``default_fallback=False`` and no script matches.
        """
        h = self.prompt_hash(model, messages, temperature, tool_schema)

        # Compute prompt_tokens from the input messages (deterministic).
        prompt_text = "".join(
            m.get("content", "") or "" for m in messages
        )
        prompt_tokens = _estimate_tokens(prompt_text)

        # Check for a scripted response.
        if h in self._responses:
            scripted = self._responses[h]

            # Error mode
            if isinstance(scripted, FakeLLMError):
                self._call_log.append({
                    "prompt_hash": h,
                    "model": model,
                    "error": scripted.error_type,
                })
                raise FakeLLMCallError(scripted)

            # Scripted dict — fill in any missing structural fields so
            # teammates' code always sees the expected shape.
            result = dict(scripted)  # shallow copy to avoid mutating the script
            result.setdefault("model", model)
            content = result.get("content", "")
            completion_tokens = _estimate_tokens(content) if content else 1
            result.setdefault("usage", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })
            usage = result["usage"]
            usage.setdefault("prompt_tokens", prompt_tokens)
            usage.setdefault("completion_tokens", completion_tokens)
            usage.setdefault(
                "total_tokens",
                usage["prompt_tokens"] + usage["completion_tokens"],
            )
            total = usage["total_tokens"]
            result.setdefault("cost_usd", round(total * 0.00001, 8))
            result.setdefault("latency_ms", max(1.0, usage["completion_tokens"] / 100.0 * 10.0))

        elif self._default_fallback:
            # Deterministic fallback: content is derived from the hash.
            fallback_content = f"[FakeLLM fallback] hash={h[:16]}"
            completion_tokens = _estimate_tokens(fallback_content)
            result = _build_response(
                content=fallback_content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
            )
        else:
            raise KeyError(
                f"No scripted response for prompt hash {h} and "
                f"default_fallback is disabled"
            )

        self._call_log.append({
            "prompt_hash": h,
            "model": model,
            "response": result,
        })
        return result

    # ------------------------------------------------------------------
    # Inspection (test helpers)
    # ------------------------------------------------------------------

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Chronological log of all calls made to this instance.

        Each entry is a dict with ``prompt_hash``, ``model``, and
        either ``response`` or ``error``.
        """
        return list(self._call_log)

    @property
    def call_count(self) -> int:
        """Total number of calls (including errored ones)."""
        return len(self._call_log)

    def reset(self) -> None:
        """Clear the call log (scripted responses are preserved)."""
        self._call_log.clear()
