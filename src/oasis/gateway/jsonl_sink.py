"""Append-only raw request/response log, keyed by SHA-256 prompt hash.

Enables replay mode (FR-30) — a reviewer can reproduce every table
from the shipped cache with no API key.  NFR-6 scrubbing applies here:
API keys must be removed before the log is included in an artifact
release.

Each line is a JSON object with the record schema defined in
DATABASE.md §"Raw log"::

    {prompt_sha, run_id, agent_id, purpose, model, temperature,
     messages, tool_schema, response, tokens_in, tokens_out,
     cost_usd, latency_ms, ts}

``prompt_sha`` is the join key into the ``agent_output`` table in
Vidish's ``db/`` schema — the field name must stay exactly
``prompt_sha`` for that join to work.

The hash is computed by :func:`oasis.gateway._hash.hash_request`
(single-sourced with :class:`~oasis.gateway.fake_llm.FakeLLM`).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Default NFR-6 scrubbing patterns
# ------------------------------------------------------------------

# Matches common API-key shaped strings:
#   - OpenAI: sk-...  (51+ chars)
#   - Anthropic: sk-ant-...
#   - Bearer tokens in Authorization headers
#   - Generic api_key / api-key / apikey fields
#
# NOTE: This is a minimal first pass.  Revisit once real provider
# responses are flowing through, to ensure coverage of all key
# formats (e.g. Azure deployment keys, GCP service-account tokens).
_DEFAULT_SCRUB_PATTERNS: list[str] = [
    r"sk-[A-Za-z0-9_-]{20,}",              # OpenAI / Anthropic style
    r"Bearer\s+[A-Za-z0-9_.\-/+=]{20,}",   # Authorization header values
    r"key-[A-Za-z0-9_-]{20,}",             # Generic key- prefix
    r"AIza[A-Za-z0-9_-]{30,}",             # Google API keys
]


class JSONLSink:
    """Append-only JSONL writer and reader for raw LLM request/response pairs.

    Writer: appends one JSON line per call.
    Reader: builds an in-memory index ``{prompt_sha → response}`` from
    the file on first lookup, then serves from the index.  This is the
    replay-mode read path (FR-30, ADR-015).

    Parameters
    ----------
    path:
        File path for the JSONL log.  Created on first write.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        # Lazy index: built on first lookup(), invalidated on write().
        self._index: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        prompt_sha: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tool_schema: list[dict[str, Any]] | None,
        response: dict[str, Any],
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float,
        run_id: str = "",
        agent_id: str = "",
        purpose: str = "productive",
    ) -> None:
        """Append a single request/response record.

        The field names and types match DATABASE.md §"Raw log" so that
        ``prompt_sha`` can join directly to ``agent_output.prompt_sha``
        in Vidish's db/ code.

        Parameters
        ----------
        prompt_sha:
            SHA-256 hex digest produced by
            :func:`oasis.gateway._hash.hash_request`.
        model:
            Exact pinned model identifier (NFR-5).
        messages:
            Full message list sent to the provider.
        temperature:
            Temperature used for the call.
        tool_schema:
            Tool/function schemas, or ``None``.
        response:
            Full provider response dict.
        tokens_in:
            Prompt token count.
        tokens_out:
            Completion token count.
        cost_usd:
            Dollar cost of the call.
        latency_ms:
            End-to-end latency in milliseconds.
        run_id:
            OASIS run identifier (empty string if not yet assigned).
        agent_id:
            Agent instance identifier.
        purpose:
            ``"productive"`` or ``"supervision"`` (FR-9, ADR-006).
        """
        record: dict[str, Any] = {
            "prompt_sha": prompt_sha,
            "run_id": run_id,
            "agent_id": agent_id,
            "purpose": purpose,
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "tool_schema": tool_schema or [],
            "response": response,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "ts": datetime.now(UTC).isoformat(),
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=False, ensure_ascii=True))
            f.write("\n")

        # Invalidate the index so the next lookup() sees the new record.
        self._index = None

    def write_raw(
        self,
        prompt_sha: str,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Convenience wrapper — extracts structured fields from *request*.

        Parameters
        ----------
        prompt_sha:
            SHA-256 hex digest.
        request:
            Full call kwargs dict — must contain ``model`` and
            ``messages``; ``temperature`` and ``tool_schema`` are
            optional.
        response:
            Provider response dict — ``usage``, ``cost_usd``,
            ``latency_ms`` are read if present.
        """
        usage = response.get("usage", {})
        self.write(
            prompt_sha=prompt_sha,
            model=request.get("model", ""),
            messages=request.get("messages", []),
            temperature=request.get("temperature", 0.0),
            tool_schema=request.get("tool_schema"),
            response=response,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            cost_usd=response.get("cost_usd", 0.0),
            latency_ms=response.get("latency_ms", 0.0),
            run_id=request.get("run_id", ""),
            agent_id=request.get("agent_id", ""),
            purpose=request.get("purpose", "productive"),
        )

    # ------------------------------------------------------------------
    # Reading / replay index
    # ------------------------------------------------------------------

    def _build_index(self) -> dict[str, dict[str, Any]]:
        """Scan the JSONL file and build ``{prompt_sha → full record}``.

        Later occurrences of the same ``prompt_sha`` overwrite earlier
        ones, so the index always reflects the latest recording.
        """
        index: dict[str, dict[str, Any]] = {}
        if not self._path.exists():
            return index
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sha = record.get("prompt_sha", "")
                if sha:
                    index[sha] = record
        return index

    def lookup(self, prompt_sha: str) -> dict[str, Any] | None:
        """Return the cached **response** dict for *prompt_sha*, or ``None``.

        This is the replay-mode read path (FR-30).  On first call the
        full JSONL file is scanned into an in-memory index; subsequent
        lookups are O(1) until the index is invalidated by a
        :meth:`write`.

        Parameters
        ----------
        prompt_sha:
            The SHA-256 hex digest to look up.

        Returns
        -------
        The ``response`` sub-dict from the stored record, or ``None``
        if no record matches.
        """
        if self._index is None:
            self._index = self._build_index()
        record = self._index.get(prompt_sha)
        if record is None:
            return None
        return record.get("response")

    def lookup_full(self, prompt_sha: str) -> dict[str, Any] | None:
        """Return the **full record** (not just the response) for *prompt_sha*.

        Useful when the caller needs metadata like ``tokens_in``,
        ``cost_usd``, etc.
        """
        if self._index is None:
            self._index = self._build_index()
        return self._index.get(prompt_sha)

    def all_hashes(self) -> list[str]:
        """Return all ``prompt_sha`` values in the log, in write order.

        Scans the file each time (does not use the index) to preserve
        order and duplicates.
        """
        hashes: list[str] = []
        if not self._path.exists():
            return hashes
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sha = record.get("prompt_sha", "")
                if sha:
                    hashes.append(sha)
        return hashes

    # ------------------------------------------------------------------
    # NFR-6 scrubbing
    # ------------------------------------------------------------------

    def scrub_keys(self, patterns: list[str] | None = None) -> int:
        """Remove API-key–like values from every record in the log.

        Reads the entire file, applies regex substitutions to every
        line, and atomically rewrites the file only if modifications
        were made.

        Parameters
        ----------
        patterns:
            Regex patterns to match sensitive values.  Defaults to
            :data:`_DEFAULT_SCRUB_PATTERNS` covering ``sk-…``,
            ``Bearer …``, ``key-…`` and Google ``AIza…`` prefixes.

        Returns
        -------
        Number of lines that were modified.

        Notes
        -----
        This is a minimal first-pass implementation.  It needs
        revisiting once real provider responses (with actual API
        keys embedded in request headers or error payloads) are
        flowing through.  In FakeLLM-based tests no real keys ever
        appear, so the patterns are only exercised against
        deliberately-planted test fixtures.
        """
        if patterns is None:
            patterns = _DEFAULT_SCRUB_PATTERNS

        compiled = [re.compile(p) for p in patterns]

        if not self._path.exists():
            return 0

        lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        modified_count = 0
        new_lines: list[str] = []

        for line in lines:
            original = line
            for rx in compiled:
                line = rx.sub("[REDACTED]", line)
            if line != original:
                modified_count += 1
            new_lines.append(line)

        if modified_count > 0:
            # Atomic write: write to temp file then rename, so a crash
            # mid-scrub doesn't corrupt the log.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                suffix=".jsonl.tmp",
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                # On Windows, must remove dst first if it exists.
                dest = self._path
                tmp = Path(tmp_path)
                if dest.exists():
                    dest.unlink()
                tmp.rename(dest)
            except BaseException:
                # Clean up temp file on failure.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Invalidate the index since file contents changed.
            self._index = None

        return modified_count


# ------------------------------------------------------------------
# Module-level convenience (thin wrappers)
# ------------------------------------------------------------------

def scrub(path: Path | str, patterns: list[str] | None = None) -> int:
    """Scrub API keys from a JSONL log file (NFR-6).

    Module-level convenience that creates a temporary :class:`JSONLSink`
    and delegates to :meth:`JSONLSink.scrub_keys`.

    Parameters
    ----------
    path:
        Path to the JSONL file.
    patterns:
        Optional override for scrub patterns.

    Returns
    -------
    Number of lines modified.
    """
    return JSONLSink(path).scrub_keys(patterns)
