"""Single-sourced canonical prompt hash (ARCHITECTURE.md §6, ADR-015).

Every request is hashed over model identifier, full message list,
temperature and tool schema.  This module is the single definition of
that hash — ``FakeLLM``, ``JSONLSink``, and any replay logic must all
import :func:`hash_request` from here so the format can never diverge.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_request(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    tool_schema: list[dict[str, Any]] | None = None,
) -> str:
    """Compute the canonical SHA-256 hex digest for a model call.

    The hash is computed over a JSON-serialised list of the four fields
    specified in ARCHITECTURE.md §6::

        [model, messages, temperature, tool_schema]

    Serialisation uses ``sort_keys=True`` and ``separators=(",", ":")``
    (compact, no whitespace) so the hash is independent of dict ordering
    and Python version.

    Parameters
    ----------
    model:
        The exact model identifier (no aliases — NFR-5).
    messages:
        The full message list, each element a dict with at least
        ``role`` and ``content``.
    temperature:
        Temperature setting for the call (default 0.0).
    tool_schema:
        Optional list of tool/function schemas.  ``None`` is normalised
        to ``[]`` before hashing.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(
        [model, messages, temperature, tool_schema or []],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
