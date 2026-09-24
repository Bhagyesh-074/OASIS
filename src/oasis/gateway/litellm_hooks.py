"""Pre-call hook does admission (RBE); post-call hook does cost/token accounting.

The pre-call hook is invoked before every outbound model call, projects
the call's cost against remaining budget, and returns an admission verdict
from the RBE (allow / downgrade / truncate / halt).  No call may bypass
this check (FR-7).

The post-call hook records token counts and cost, tagged with
purpose=supervision where applicable (FR-9, ADR-006).

Both hooks are registered with LiteLLM's ``callbacks`` list at gateway
start-up.

Structural guarantee
--------------------
The :class:`Gateway` class enforces that **every** call flows through
the pre→execute→post sequence.  There is no public path to execute a
call without the pre-call hook running first and the post-call hook
running after.  This is the seam that FR-7's "no call may bypass the
check" requirement hangs on.

Admission placeholder
---------------------
The admission decision itself is a placeholder that always returns
``allow``.  Real admission logic (FR-7/FR-8) will be dropped in at
Prompt 7, after RBE's budget model and reducer exist.  The swap point
is :meth:`Gateway._admission_check` — replace its body without
changing any call site.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from oasis.gateway._hash import hash_request
from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink

# ------------------------------------------------------------------
# Admission verdict (matches rbe/admission.py enum values)
# ------------------------------------------------------------------

VERDICT_ALLOW = "allow"
VERDICT_DOWNGRADE = "downgrade"
VERDICT_TRUNCATE = "truncate"
VERDICT_HALT = "halt"

_VALID_VERDICTS = frozenset({VERDICT_ALLOW, VERDICT_DOWNGRADE, VERDICT_TRUNCATE, VERDICT_HALT})


class Verdict(dict):
    """Admission verdict returned by pre_call_hook.

    Behaves both as a dictionary (e.g. ``verdict["verdict"]``) and as a string
    (e.g. ``verdict == "allow"``), with a ``.verdict`` attribute for convenience.
    """

    def __init__(self, verdict: str = VERDICT_ALLOW, **extra: Any) -> None:
        super().__init__(verdict=verdict, **extra)

    @property
    def verdict(self) -> str:
        return self.get("verdict", VERDICT_ALLOW)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.get("verdict") == other
        return super().__eq__(other)

    def __str__(self) -> str:
        return str(self.get("verdict", VERDICT_ALLOW))

    def __repr__(self) -> str:
        return f"Verdict({super().__repr__()})"


# ------------------------------------------------------------------
# Model pinning validation (NFR-5)
# ------------------------------------------------------------------


class UnpinnedModelError(ValueError):
    """Raised when an unpinned model identifier or alias is used (NFR-5)."""


def is_pinned_model(model: str) -> bool:
    """Return True if model is a pinned version string, False if unpinned alias (NFR-5)."""
    if not model or not isinstance(model, str) or not model.strip():
        return False
    m = model.strip().lower()
    return "latest" not in m


def validate_model_pinning(model: str) -> None:
    """Validate that model is pinned to an explicit version (NFR-5).

    Raises
    ------
    UnpinnedModelError
        If model is empty or contains an alias such as 'latest'.
    """
    if not is_pinned_model(model):
        raise UnpinnedModelError(
            f"Model identifier {model!r} is unpinned. "
            f"NFR-5 prohibits aliases such as 'latest'; an explicit pinned version is required."
        )


# ------------------------------------------------------------------
# Global in-memory call counter
# ------------------------------------------------------------------

_CALL_COUNTER: int = 0


def get_call_count() -> int:
    """Return the global in-memory call count across all pre_call_hook invocations."""
    return _CALL_COUNTER


def reset_call_counter() -> None:
    """Reset the global in-memory call counter (useful in tests)."""
    global _CALL_COUNTER
    _CALL_COUNTER = 0


# ------------------------------------------------------------------
# Standalone hook functions
# ------------------------------------------------------------------

def pre_call_hook(
    request: dict[str, Any] | None = None,
    run_budget_state: Any = None,
    *,
    call_data: dict[str, Any] | None = None,
    admission_fn: Callable[[dict[str, Any]], Any] | None = None,
    admission_controller: Any = None,
    **kwargs: Any,
) -> Verdict:
    """Pre-call hook — runs admission check and increments call counter.

    Enforces NFR-5 model pinning before any call or counter increment,
    then runs admission check via AdmissionController (FR-7/FR-8), admission_fn,
    or returns "allow" if no budget is configured.

    Parameters
    ----------
    request:
        ``{model, messages, temperature, tool_schema, purpose, …}``
    run_budget_state:
        Optional live budget state (used in Prompt 7 by AdmissionController).
    call_data:
        Alias for request for backwards compatibility.
    admission_fn:
        Optional admission check callable.
    admission_controller:
        Optional :class:`~oasis.rbe.admission.AdmissionController`.

    Returns
    -------
    Verdict
        Verdict object representing the admission decision (default "allow").

    Raises
    ------
    UnpinnedModelError
        If the model identifier contains 'latest' or is unpinned (NFR-5).
    """
    req = request if request is not None else (call_data or {})

    # NFR-5: Model identifiers must be pinned to explicit versions
    model = req.get("model", "")
    validate_model_pinning(model)

    global _CALL_COUNTER
    _CALL_COUNTER += 1

    # Check if admission_controller or admission_fn is provided
    if admission_controller is not None:
        result = admission_controller.check(req)
    elif admission_fn is not None:
        result = admission_fn(req)
    elif run_budget_state is not None and hasattr(run_budget_state, "admission_controller"):
        result = run_budget_state.admission_controller.check(req)
    elif run_budget_state is not None and hasattr(run_budget_state, "check_admission"):
        result = run_budget_state.check_admission(req)
    elif run_budget_state is not None and hasattr(run_budget_state, "remaining"):
        from oasis.rbe.admission import AdmissionController

        if not hasattr(run_budget_state, "_admission_controller"):
            run_budget_state._admission_controller = AdmissionController(run_budget_state)
        result = run_budget_state._admission_controller.check(req)
    else:
        result = {"verdict": VERDICT_ALLOW}

    if isinstance(result, Verdict):
        verdict_val = result.verdict
        verdict_obj = result
    elif isinstance(result, dict):
        verdict_val = result.get("verdict", VERDICT_ALLOW)
        verdict_obj = Verdict(verdict=verdict_val, **{k: v for k, v in result.items() if k != "verdict"})
    elif hasattr(result, "value"):  # Enum like AdmissionVerdict
        verdict_val = str(result.value)
        verdict_obj = Verdict(verdict=verdict_val)
    else:
        verdict_val = str(result)
        verdict_obj = Verdict(verdict=verdict_val)

    if verdict_val not in _VALID_VERDICTS:
        raise ValueError(
            f"Invalid admission verdict {verdict_val!r}; "
            f"must be one of {sorted(_VALID_VERDICTS)}"
        )
    return verdict_obj


def post_call_hook(
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    sink: JSONLSink | None = None,
    run_budget_state: Any = None,
    *,
    call_data: dict[str, Any] | None = None,
    response_data: dict[str, Any] | None = None,
    prompt_sha: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Post-call hook — does cost/token accounting and writes to JSONLSink.

    Pulls tokens_in, tokens_out, cost_usd, latency_ms off the response
    (from FakeLLM's shape) and writes the record to jsonl_sink if provided.

    Parameters
    ----------
    request:
        The original call kwargs (model, messages, …).
    response:
        The LLM response including ``usage``, ``cost_usd``, ``latency_ms``.
    sink:
        Optional :class:`~oasis.gateway.jsonl_sink.JSONLSink` to record raw call.
    run_budget_state:
        Optional live budget state to update consumption.
    call_data:
        Alias for request for backwards compatibility.
    response_data:
        Alias for response for backwards compatibility.
    prompt_sha:
        Pre-computed SHA-256 hash. If None, computed from request.

    Returns
    -------
    dict with ``tokens_in``, ``tokens_out``, ``cost_usd``, ``latency_ms``,
    ``model``, ``purpose``, ``prompt_sha``.
    """
    req = request if request is not None else (call_data or {})
    resp = response if response is not None else (response_data or {})

    usage = resp.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    cost_usd = resp.get("cost_usd", 0.0)
    latency_ms = resp.get("latency_ms", 0.0)
    model = req.get("model", "")
    purpose = req.get("purpose", "productive")

    if prompt_sha is None:
        prompt_sha = hash_request(
            model=model,
            messages=req.get("messages", []),
            temperature=req.get("temperature", 0.0),
            tool_schema=req.get("tool_schema"),
        )

    accounting = {
        "prompt_sha": prompt_sha,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "purpose": purpose,
    }

    # Write to JSONL sink if provided
    target_sink = sink if sink is not None else (req.get("sink") or kwargs.get("sink"))
    if target_sink is not None:
        target_sink.write(
            prompt_sha=prompt_sha,
            model=model,
            messages=req.get("messages", []),
            temperature=req.get("temperature", 0.0),
            tool_schema=req.get("tool_schema"),
            response=resp,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            run_id=req.get("run_id", ""),
            agent_id=req.get("agent_id", ""),
            purpose=purpose,
        )

    # Update budget state if provided
    budget_state = run_budget_state if run_budget_state is not None else req.get("budget_state")
    if budget_state is not None:
        if hasattr(budget_state, "record"):
            total_tokens = tokens_in + tokens_out
            wall_sec = latency_ms / 1000.0 if latency_ms else 0.0
            try:
                budget_state.record(
                    tokens=total_tokens,
                    cost_usd=cost_usd,
                    wall_seconds=wall_sec,
                    calls=1,
                    purpose=purpose,
                )
            except TypeError:
                budget_state.record(
                    tokens=total_tokens,
                    cost_usd=cost_usd,
                    wall_seconds=wall_sec,
                    calls=1,
                )
        elif hasattr(budget_state, "record_call"):
            budget_state.record_call(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                purpose=purpose,
            )
        elif hasattr(budget_state, "update"):
            budget_state.update(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                purpose=purpose,
            )

    return accounting


# ------------------------------------------------------------------
# Gateway — the structural enforcement point
# ------------------------------------------------------------------

class Gateway:
    """LLM call gateway — enforces pre→execute→post on every call.

    This class is the single point through which **all** model traffic
    flows.  It owns:

    - A call counter (incremented exactly once per pre-call hook).
    - A JSONL sink for raw request/response logging (FR-30).
    - The admission-check plug point (placeholder now, real RBE in
      Prompt 7).

    Parameters
    ----------
    sink:
        A :class:`~oasis.gateway.jsonl_sink.JSONLSink` for recording
        raw call data.
    llm:
        The callable used to execute model calls.  In tests this is
        :class:`~oasis.gateway.fake_llm.FakeLLM`; in production it
        would be the LiteLLM completion function.
    admission_fn:
        Optional override for the admission check.  If ``None``, the
        built-in placeholder (always ``allow``) is used.  Prompt 7
        will supply the real RBE admission controller here.
    """

    def __init__(
        self,
        sink: JSONLSink,
        llm: FakeLLM | Any = None,
        *,
        admission_fn: Callable[[dict[str, Any]], Any] | None = None,
        admission_controller: Any = None,
        run_id: str = "",
        agent_id: str = "",
        purpose: str = "productive",
        budget_state: Any = None,
        return_on_halt: bool = False,
    ) -> None:
        self._sink = sink
        self._llm = llm
        self._admission_fn = admission_fn
        self._run_id = run_id
        self._agent_id = agent_id
        self._purpose = purpose
        self._budget_state = budget_state
        self._return_on_halt = return_on_halt

        if admission_controller is not None:
            self._admission_controller = admission_controller
        elif budget_state is not None:
            if hasattr(budget_state, "admission_controller"):
                self._admission_controller = budget_state.admission_controller
            elif hasattr(budget_state, "remaining"):
                from oasis.rbe.admission import AdmissionController

                self._admission_controller = AdmissionController(budget_state, run_id=run_id)
            else:
                self._admission_controller = None
        else:
            self._admission_controller = None

        # Call counter: incremented exactly once per pre_call_hook
        # invocation. Prompt 7's invariant test will compare this
        # against RBE's budget-event log count.
        self._call_count: int = 0

        # Records every call's accounting for inspection/reconciliation.
        self._accounting_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of calls that have passed through pre_call_hook.

        This is the number the FR-7 invariant test compares against the
        budget-event log count.
        """
        return self._call_count

    @property
    def accounting_log(self) -> list[dict[str, Any]]:
        """Chronological list of per-call accounting records."""
        return list(self._accounting_log)

    @property
    def admission_controller(self) -> Any:
        """The active admission controller, if any."""
        return self._admission_controller

    # ------------------------------------------------------------------
    # The three seams (pre / execute / post)
    # ------------------------------------------------------------------

    def pre_call_hook(self, request: dict[str, Any]) -> Verdict:
        """Pre-call hook invoked before every outbound call.

        Enforces NFR-5 model pinning before incrementing call counter.
        """
        validate_model_pinning(request.get("model", ""))
        self._call_count += 1
        return self._admission_check(request)

    def _admission_check(self, call_data: dict[str, Any]) -> Verdict:
        """Run the admission check on *call_data*.

        Parameters
        ----------
        call_data:
            ``{model, messages, temperature, tool_schema, purpose, …}``

        Returns
        -------
        Verdict with at least ``{"verdict": "allow"|"downgrade"|"truncate"|"halt"}``.
        """
        if self._admission_fn is not None:
            result = self._admission_fn(call_data)
        elif self._admission_controller is not None:
            result = self._admission_controller.check(call_data)
        elif self._budget_state is not None and hasattr(self._budget_state, "admission_controller"):
            result = self._budget_state.admission_controller.check(call_data)
        elif self._budget_state is not None and hasattr(self._budget_state, "remaining"):
            from oasis.rbe.admission import AdmissionController

            if not hasattr(self._budget_state, "_admission_controller"):
                self._budget_state._admission_controller = AdmissionController(
                    self._budget_state, run_id=self._run_id
                )
            result = self._budget_state._admission_controller.check(call_data)
        else:
            result = {"verdict": VERDICT_ALLOW}

        if isinstance(result, Verdict):
            verdict_val = result.verdict
            verdict_obj = result
        elif isinstance(result, dict):
            verdict_val = result.get("verdict", VERDICT_ALLOW)
            verdict_obj = Verdict(verdict=verdict_val, **{k: v for k, v in result.items() if k != "verdict"})
        elif hasattr(result, "value"):
            verdict_val = str(result.value)
            verdict_obj = Verdict(verdict=verdict_val)
        else:
            verdict_val = str(result)
            verdict_obj = Verdict(verdict=verdict_val)

        if verdict_val not in _VALID_VERDICTS:
            raise ValueError(
                f"Invalid admission verdict {verdict_val!r}; "
                f"must be one of {sorted(_VALID_VERDICTS)}"
            )
        return verdict_obj

    def _execute(
        self,
        call_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the model call via the configured LLM backend.

        Parameters
        ----------
        call_data:
            Must contain ``model`` and ``messages``.

        Returns
        -------
        The LLM response dict (FakeLLM shape: content, model, usage,
        cost_usd, latency_ms).
        """
        if self._llm is None:
            raise RuntimeError("No LLM backend configured on this Gateway")

        return self._llm.complete(
            model=call_data["model"],
            messages=call_data["messages"],
            temperature=call_data.get("temperature", 0.0),
            tool_schema=call_data.get("tool_schema"),
        )

    def post_call_hook(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        prompt_sha: str | None = None,
    ) -> dict[str, Any]:
        """Post-call hook invoked after every completed call.

        Extracts accounting fields, writes to JSONLSink, and records into
        the gateway's accounting log.
        """
        accounting = post_call_hook(
            request=request,
            response=response,
            sink=self._sink,
            run_budget_state=self._budget_state,
            prompt_sha=prompt_sha,
        )
        self._accounting_log.append(accounting)
        return accounting

    def _record_accounting(
        self,
        call_data: dict[str, Any],
        response: dict[str, Any],
        prompt_sha: str,
    ) -> dict[str, Any]:
        """Helper for recording accounting (delegates to post_call_hook)."""
        return self.post_call_hook(call_data, response, prompt_sha=prompt_sha)

    # ------------------------------------------------------------------
    # Public API — the ONLY way to make a call
    # ------------------------------------------------------------------

    def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tool_schema: list[dict[str, Any]] | None = None,
        purpose: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        return_on_halt: bool | None = None,
        outputs_so_far: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a model call through the full pre→execute→post pipeline.

        This is the **only** public entry-point for making LLM calls.
        It guarantees:

        1. Pre-call hook (admission check + call counter increment)
           runs first.
        2. If the verdict is ``halt``, either terminates with status
           ``halted_budget`` returning best output (FR-8) or raises
           ``GatewayHaltError`` (if return_on_halt is False).
        3. The model call executes (with model downgrade or context
           truncation applied if verdict is downgrade/truncate).
        4. Post-call hook (accounting + JSONL write) runs after.

        Parameters
        ----------
        model:
            Exact pinned model identifier (NFR-5).
        messages:
            Full message list.
        temperature:
            Call temperature (default 0.0).
        tool_schema:
            Optional tool/function schemas.
        purpose:
            ``"productive"`` (default) or ``"supervision"`` (FR-9).
        run_id:
            Override the gateway's default run_id for this call.
        agent_id:
            Override the gateway's default agent_id for this call.
        return_on_halt:
            If True, return status dict with best output on halt instead of
            raising GatewayHaltError (FR-8).
        outputs_so_far:
            Optional list of prior outputs produced so far in the run.

        Returns
        -------
        dict
            The full LLM response, augmented with ``_verdict`` and
            ``_prompt_sha`` metadata keys; or a halted_budget record on halt.

        Raises
        ------
        GatewayHaltError
            If the admission verdict is ``halt`` and return_on_halt is False.
        """
        try:
            validate_model_pinning(model)
        except UnpinnedModelError:
            from oasis.rbe.decision_log_client import log_decision

            log_decision(
                component="RBE",
                decision="reject_unpinned_model",
                inputs={"model": model},
                justification=f"Outbound call rejected: model identifier '{model}' is unpinned, violating NFR-5.",
                run_id=run_id or self._run_id,
            )
            raise

        call_purpose = purpose or self._purpose
        if call_purpose not in ("productive", "supervision"):
            raise ValueError(
                f"Invalid purpose {call_purpose!r}; must be 'productive' or 'supervision'"
            )

        call_data: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "tool_schema": tool_schema,
            "purpose": call_purpose,
            "run_id": run_id or self._run_id,
            "agent_id": agent_id or self._agent_id,
        }

        # ---- PRE-CALL HOOK ----
        # Increment the call counter EXACTLY once, before anything else.
        verdict = self.pre_call_hook(call_data)

        verdict_str = (
            verdict.verdict
            if isinstance(verdict, Verdict)
            else (verdict.get("verdict", "") if isinstance(verdict, dict) else str(verdict))
        )

        if verdict_str == VERDICT_HALT:
            should_return = (
                return_on_halt if return_on_halt is not None else self._return_on_halt
            )
            if should_return or outputs_so_far is not None:
                from oasis.rbe.admission import handle_halt

                last_event = None
                if (
                    self._admission_controller
                    and hasattr(self._admission_controller, "events")
                    and self._admission_controller.events
                ):
                    last_event = self._admission_controller.events[-1]
                return handle_halt(
                    outputs=outputs_so_far or [],
                    run_id=run_id or self._run_id,
                    budget_event=last_event,
                )
            raise GatewayHaltError(
                "Budget exhausted — admission verdict is halt (FR-8)",
                call_data=call_data,
            )

        # Handle DOWNGRADE (swap model) and TRUNCATE (trim messages)
        if verdict_str == VERDICT_DOWNGRADE:
            from oasis.rbe.admission import get_downgrade_model

            new_model = verdict.get("model_after") if isinstance(verdict, dict) else None
            call_data["model"] = new_model or get_downgrade_model(call_data["model"])
        elif verdict_str == VERDICT_TRUNCATE:
            msgs = call_data.get("messages", [])
            if len(msgs) > 1:
                truncated_msgs = []
                if msgs[0].get("role") == "system":
                    truncated_msgs.append(msgs[0])
                truncated_msgs.append(msgs[-1])
                call_data["messages"] = truncated_msgs

        # ---- EXECUTE ----
        prompt_sha = hash_request(
            call_data["model"],
            call_data["messages"],
            call_data.get("temperature", 0.0),
            call_data.get("tool_schema"),
        )
        response = self._execute(call_data)

        # ---- POST-CALL HOOK ----
        accounting = self.post_call_hook(call_data, response, prompt_sha=prompt_sha)

        # Augment the response with metadata for callers.
        response["_verdict"] = verdict_str
        response["_prompt_sha"] = prompt_sha
        response["_accounting"] = accounting

        return response


class GatewayHaltError(Exception):
    """Raised when the admission verdict is ``halt`` (FR-8).

    The run should terminate with status ``halted_budget`` and return
    the best output produced so far rather than propagating this error.
    """

    def __init__(self, message: str, *, call_data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.call_data = call_data or {}


__all__ = [
    "VERDICT_ALLOW",
    "VERDICT_DOWNGRADE",
    "VERDICT_HALT",
    "VERDICT_TRUNCATE",
    "Gateway",
    "GatewayHaltError",
    "UnpinnedModelError",
    "Verdict",
    "get_call_count",
    "is_pinned_model",
    "post_call_hook",
    "pre_call_hook",
    "reset_call_counter",
    "validate_model_pinning",
]
