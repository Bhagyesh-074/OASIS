"""LLM-Planner Task Complexity Estimator (FR-4, FR-24, NFR-1, NFR-5).

Alternative task complexity estimator behind the SAME interface as heuristic.estimate:
    estimate(statement, domain=None, ...) -> Estimate
Enabling the ablation harness to evaluate heuristic team sizing against an LLM-planner
baseline arm.

Architectural Design & Seams
----------------------------
1. Gateway Seam:
   Calls the LLM exclusively through `oasis.gateway.Gateway` (FR-7, FR-9).
   Tagged with `purpose="productive"` by default (configurable via `purpose`).
   The planner is an estimator, not part of the RTPM supervision loop, so
   ADR-006's supervision tagging does not apply by default.
   Model identifier is resolved from env `OASIS_MODEL_CHEAP` (default
   `gpt-4o-mini-2024-07-18`) and strictly validated against `validate_model_pinning`
   (NFR-5) to prohibit unpinned aliases like 'latest'. Temperature is pinned to 0.0.

2. Defensive Parsing:
   Prompts the model for JSON matching:
       {"mvts": int, "subtasks": [str], "rationale": str}
   Defensively strips markdown code fences, tolerates conversational commentary,
   clamps `mvts` to `config.mvts_range` (not hard-coded 1-8), and falls back to
   a documented default (mvts=config.mvts_range[0], 1 subtask) on malformed output.
   Never raises on malformed model replies.

3. Sub-Scores & Comparability Across Estimators:
   The Estimate schema requires three named sub-scores (FR-2). For this estimator:
   - `mvts` is predicted directly by the LLM (clamped to `config.mvts_range`).
   - `subtask_count` is derived from the model's decomposed subtasks (len(subtasks)).
   - `skill_clusters` and `dep_density` are computed by reusing the heuristic estimator's
     clustering and dependency-graph functions (`compute_skill_clusters`,
     `compute_dependency_density`) on those subtasks, so sub-scores remain directly
     comparable across estimators in the ablation harness.
   - Note on Contributions: `WeightedContribution` values are calculated from the
     configured weights and sub-score normalization for schema conformance and diagnostic
     reporting; MVTS is NOT derived from contributions in this estimator.

4. Audit Logging & Latency:
   Calls `log_tce_decision` for every estimate (FR-24) with a non-empty justification
   incorporating model rationale, clamping notes, and any fallbacks used.
   Measures compute duration in milliseconds (`latency_ms`, NFR-1).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway, validate_model_pinning
from oasis.tce._log import log_tce_decision
from oasis.tce.config import ComplexityConfig, load_complexity_config
from oasis.tce.heuristic import (
    compute_dependency_density,
    compute_skill_clusters,
)
from oasis.tce.types import Estimate, SubScores, WeightedContribution

# Default pinned cheap model identifier (NFR-5)
DEFAULT_MODEL: str = "gpt-4o-mini-2024-07-18"

# Module-level default gateway singleton / factory
_DEFAULT_GATEWAY: Gateway | None = None
_DEFAULT_GATEWAY_FACTORY: Callable[[], Gateway] | None = None


def get_planner_model(model_override: str | None = None) -> str:
    """Resolve and validate the pinned model identifier for TCE LLM planner (NFR-5).

    Parameters
    ----------
    model_override:
        Optional explicit model string overriding env and defaults.

    Returns
    -------
    str
        Validated pinned model identifier.

    Raises
    ------
    UnpinnedModelError
        If model identifier is empty or contains unpinned aliases like 'latest' (NFR-5).
    """
    if model_override is not None:
        model = model_override
    else:
        model = os.environ.get("OASIS_MODEL_CHEAP", DEFAULT_MODEL)
    validate_model_pinning(model)
    return model


def build_planner_messages(statement: str, domain: str | None = None) -> list[dict[str, str]]:
    """Build the prompt message list for the LLM complexity estimator.

    Parameters
    ----------
    statement:
        Raw problem statement text.
    domain:
        Optional evaluation domain identifier.

    Returns
    -------
    list[dict[str, str]]
        List of system and user message dictionaries.
    """
    system_content = (
        "You are a task complexity estimator. Analyze the problem statement and determine "
        "the Minimum Viable Team Size (MVTS), decompose the task into logical candidate subtasks, "
        "and provide a concise rationale.\n"
        "Respond with valid JSON ONLY in this format:\n"
        '{"mvts": <int>, "subtasks": [<str>, ...], "rationale": "<str>"}'
    )
    user_content = (
        f"Domain: {domain}\nProblem Statement:\n{statement}\n\nEstimate complexity and return JSON only."
        if domain
        else f"Problem Statement:\n{statement}\n\nEstimate complexity and return JSON only."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_llm_response(
    raw_content: str,
    statement: str,
    config: ComplexityConfig,
) -> tuple[int, int | None, list[str], str, bool]:
    """Parse the raw LLM output into (clamped_mvts, raw_mvts, subtasks, rationale, fallback_used).

    Defensively handles:
    - Markdown code fences (e.g., ```json ... ``` or ``` ... ```)
    - Surrounding conversational commentary/text before or after JSON
    - Clamping mvts to config.mvts_range (not hard-coded 1-8)
    - Missing or malformed keys
    - Unparseable strings, falling back safely without raising

    Fallback behavior:
    If parsing fails completely, defaults to:
    - mvts: config.mvts_range[0] (minimum team size, e.g. 1)
    - raw_mvts: None
    - subtasks: [statement.strip() or "Execute task"]
    - rationale: "Fallback default due to unparseable model response"
    - fallback_used: True

    Parameters
    ----------
    raw_content:
        Raw string returned by the model completion.
    statement:
        Original problem statement (used as fallback subtask).
    config:
        Validated ComplexityConfig instance.

    Returns
    -------
    tuple[int, int | None, list[str], str, bool]
        (clamped_mvts, raw_mvts, subtasks, rationale, fallback_used)
    """
    min_mvts, max_mvts = config.mvts_range
    fallback_mvts = min_mvts
    fallback_subtasks = [statement.strip() if statement.strip() else "Execute task"]
    fallback_rationale = "Fallback default due to unparseable LLM output"

    if not raw_content or not isinstance(raw_content, str):
        return fallback_mvts, None, fallback_subtasks, fallback_rationale, True

    cleaned = raw_content.strip()

    # 1. Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    candidate_json = fence_match.group(1).strip() if fence_match else cleaned

    # 2. Extract outermost JSON object if surrounding commentary exists
    data: Any = None
    try:
        data = json.loads(candidate_json)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", candidate_json)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return fallback_mvts, None, fallback_subtasks, fallback_rationale, True

    # 3. Extract and clamp mvts
    raw_mvts_val = data.get("mvts")
    raw_mvts: int | None = None
    fallback_used = False

    if raw_mvts_val is not None:
        try:
            raw_mvts = int(raw_mvts_val)
            clamped_mvts = max(min_mvts, min(max_mvts, raw_mvts))
        except (ValueError, TypeError):
            clamped_mvts = fallback_mvts
            fallback_used = True
    else:
        clamped_mvts = fallback_mvts
        fallback_used = True

    # 4. Extract subtasks
    raw_subtasks = data.get("subtasks")
    subtasks: list[str] = []
    if isinstance(raw_subtasks, list):
        for item in raw_subtasks:
            s = str(item).strip()
            if s:
                subtasks.append(s)
    if not subtasks:
        subtasks = fallback_subtasks

    # 5. Extract rationale
    raw_rationale = data.get("rationale")
    if raw_rationale and str(raw_rationale).strip():
        rationale = str(raw_rationale).strip()
    else:
        rationale = "No rationale provided by model" if not fallback_used else fallback_rationale

    return clamped_mvts, raw_mvts, subtasks, rationale, fallback_used


def set_default_gateway(gateway: Gateway | None) -> None:
    """Set the process-wide default Gateway instance."""
    global _DEFAULT_GATEWAY
    _DEFAULT_GATEWAY = gateway


def set_default_gateway_factory(factory: Callable[[], Gateway] | None) -> None:
    """Set the process-wide default Gateway factory callable."""
    global _DEFAULT_GATEWAY_FACTORY
    _DEFAULT_GATEWAY_FACTORY = factory


def get_default_gateway() -> Gateway:
    """Retrieve or initialize the default Gateway instance."""
    global _DEFAULT_GATEWAY
    if _DEFAULT_GATEWAY is None:
        sink_path = Path("artifacts/llm_calls/tce_gateway_calls.jsonl")
        sink = JSONLSink(sink_path)
        _DEFAULT_GATEWAY = Gateway(sink=sink, purpose="productive")
    return _DEFAULT_GATEWAY


def estimate(
    statement: str,
    domain: str | None = None,
    config: ComplexityConfig | None = None,
    gateway: Gateway | None = None,
    gateway_factory: Callable[[], Gateway] | None = None,
    model: str | None = None,
    purpose: str = "productive",
) -> Estimate:
    """Estimate task complexity using an LLM planner (FR-4, FR-24, NFR-1, NFR-5).

    Matches the SAME interface as `heuristic.estimate` so the ablation harness
    can swap between heuristic and LLM planner.

    Parameters
    ----------
    statement:
        Problem statement text to estimate.
    domain:
        Optional task domain identifier (e.g., 'code_generation', 'research_qa').
    config:
        Optional ComplexityConfig. Defaults to repository config/complexity.yaml.
    gateway:
        Optional Gateway instance. If None, uses gateway_factory or default Gateway.
    gateway_factory:
        Optional callable producing a Gateway instance.
    model:
        Optional model identifier overriding env OASIS_MODEL_CHEAP (NFR-5).
    purpose:
        Gateway call purpose tag (default "productive"). The planner is an estimator,
        not part of the RTPM supervision loop, so ADR-006's supervision tagging does
        not apply by default, but callers can override it (e.g. to "supervision").

    Returns
    -------
    Estimate
        Validated Estimate response matching POST /v1/estimate in docs/API_SPEC.md.

    Raises
    ------
    UnpinnedModelError
        If resolved model identifier contains 'latest' or is unpinned (NFR-5).
    """
    start_time = time.perf_counter()
    cfg = config or load_complexity_config()

    # 1. Resolve and validate model identifier (NFR-5)
    model_id = get_planner_model(model)

    # 2. Resolve Gateway dependency
    if gateway is not None:
        gw = gateway
    elif gateway_factory is not None:
        gw = gateway_factory()
    elif _DEFAULT_GATEWAY is not None:
        gw = _DEFAULT_GATEWAY
    elif _DEFAULT_GATEWAY_FACTORY is not None:
        gw = _DEFAULT_GATEWAY_FACTORY()
    else:
        gw = get_default_gateway()

    # 3. Build prompt and execute call through Gateway seam
    messages = build_planner_messages(statement, domain=domain)
    response = gw.call(
        model=model_id,
        messages=messages,
        temperature=0.0,
        purpose=purpose,
    )

    raw_content = response.get("content", "") if isinstance(response, dict) else ""

    # 4. Defensive parsing: strip fences, clamp MVTS, handle fallbacks
    clamped_mvts, raw_mvts, subtasks, rationale, fallback_used = parse_llm_response(
        raw_content=raw_content,
        statement=statement,
        config=cfg,
    )

    # 5. Compute comparable sub-scores by reusing heuristic functions (FR-2)
    subtask_count = len(subtasks)

    try:
        skill_clusters = compute_skill_clusters(subtasks, config=cfg)
    except (RuntimeError, ValueError, TypeError, OSError):
        skill_clusters = 1

    try:
        dep_density = compute_dependency_density(subtasks)
    except (RuntimeError, ValueError, TypeError, OSError):
        dep_density = 0.0

    subscores = SubScores(
        subtask_count=subtask_count,
        skill_clusters=skill_clusters,
        dep_density=dep_density,
    )

    # 6. Informational contributions for schema conformance and diagnostic parity
    norm_cfg = cfg.normalization
    subtask_range = max(1, norm_cfg.max_subtasks - norm_cfg.min_subtasks)
    norm_subtasks = max(
        0.0, min(1.0, (float(subtask_count) - norm_cfg.min_subtasks) / subtask_range)
    )

    cluster_range = max(1, norm_cfg.max_skill_clusters - norm_cfg.min_skill_clusters)
    norm_clusters = max(
        0.0, min(1.0, (float(skill_clusters) - norm_cfg.min_skill_clusters) / cluster_range)
    )

    norm_density = max(0.0, min(1.0, dep_density))

    contrib_subtasks = round(cfg.weights.subtask_count * norm_subtasks, 6)
    contrib_clusters = round(cfg.weights.skill_clusters * norm_clusters, 6)
    contrib_density = round(cfg.weights.dep_density * norm_density, 6)

    contributions = WeightedContribution(
        subtask_count=contrib_subtasks,
        skill_clusters=contrib_clusters,
        dep_density=contrib_density,
    )

    # 7. Construct non-empty justification and log decision (FR-24)
    if fallback_used:
        justification = (
            f"LLM planner fallback: unparseable model response; defaulted to mvts={clamped_mvts} "
            f"({subtask_count} subtask). Rationale: {rationale}"
        )
    else:
        clamp_str = (
            f" [clamped from {raw_mvts} to {cfg.mvts_range}]"
            if raw_mvts is not None and clamped_mvts != raw_mvts
            else ""
        )
        justification = (
            f"LLM planner: mvts={clamped_mvts}{clamp_str}, subtasks={subtask_count}. Rationale: {rationale}"
        )

    log_inputs: dict[str, Any] = {
        "statement": statement,
        "estimator": "llm_planner",
        "model": model_id,
        "purpose": purpose,
        "subscores": subscores.model_dump(),
        "weights": cfg.weights.to_dict(),
        "contributions": contributions.model_dump(),
        "mvts": clamped_mvts,
        "raw_mvts": raw_mvts,
        "fallback_used": fallback_used,
        "rationale": rationale,
    }
    if domain is not None:
        log_inputs["domain"] = domain

    log_tce_decision(
        decision="estimate_mvts",
        inputs=log_inputs,
        justification=justification,
    )

    # 8. Record latency in ms (NFR-1)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return Estimate(
        mvts=clamped_mvts,
        subscores=subscores,
        weights=cfg.weights.to_dict(),
        contributions=contributions,
        subtasks=subtasks,
        justification=justification,
        latency_ms=latency_ms,
    )


class LLMPlanner:
    """Object-oriented wrapper conforming to the 2-argument estimate callable (FR-4).

    Allows initializing once with a pre-configured Gateway or factory, then calling
    `planner.estimate(statement, domain=None)` identically to `heuristic.estimate`.
    """

    def __init__(
        self,
        gateway: Gateway | None = None,
        gateway_factory: Callable[[], Gateway] | None = None,
        config: ComplexityConfig | None = None,
        model: str | None = None,
        purpose: str = "productive",
    ) -> None:
        self.gateway = gateway
        self.gateway_factory = gateway_factory
        self.config = config
        self.model = model
        self.purpose = purpose

    def estimate(
        self,
        statement: str,
        domain: str | None = None,
        purpose: str | None = None,
    ) -> Estimate:
        """Estimate complexity for statement using bound gateway and configuration."""
        call_purpose = purpose if purpose is not None else self.purpose
        return estimate(
            statement=statement,
            domain=domain,
            config=self.config,
            gateway=self.gateway,
            gateway_factory=self.gateway_factory,
            model=self.model,
            purpose=call_purpose,
        )
