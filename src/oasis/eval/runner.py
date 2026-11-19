"""Async benchmark runner and matrix execution harness (FR-26, FR-30, NFR-1, NFR-3).

Executes the eight-configuration ablation matrix:
1. vanilla_fixed
2. single_agent
3. captainagent
4. tce_only
5. rbe_only
6. monitor_only
7. tce_rbe
8. full

Operational constraints:
- Bounded concurrency via asyncio.Semaphore (default 8 per DEPLOYMENT.md).
- Global API spend ceiling checked after every tranche (NFR-3).
- Live mode uses LiteLLM; replay mode reads the recorded JSONL cache.
- FakeLLM is used only when explicitly injected by tests.
- Arms depending on the unimplemented monitor/replacement components fail explicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

import ulid
import yaml

from oasis.adapters.autogen_adapter import AutoGenAdapter
from oasis.adapters.base import AdapterBase
from oasis.adapters.crewai_stub import CrewAIStub
from oasis.adapters.langgraph_adapter import LangGraphAdapter
from oasis.api.routes_estimate import _estimate_mvts
from oasis.api.routes_runs import broadcast_run_event
from oasis.db.models import (
    AgentInstanceRecord,
    AgentOutputRecord,
    Database,
    RunRecord,
    TaskRecord,
    TeamConfigRecord,
)
from oasis.gateway._hash import hash_request
from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway, GatewayHaltError
from oasis.rbe.budget import Budget, BudgetState
from oasis.rbe.reducer import BudgetInfeasibleError, TeamReducer

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"

_EXECUTABLE_ARMS = frozenset({
    "vanilla_fixed",
    "single_agent",
    "tce_only",
    "rbe_only",
    "tce_rbe",
})
_UNAVAILABLE_ARMS = {
    "captainagent": "the CaptainAgent baseline adapter is not implemented",
    "monitor_only": "the RTPM scoring pipeline is not implemented",
    "full": "the RTPM and replacement pipelines are not implemented",
    "oracle": "the oracle team-size sweep is not implemented",
}
_CACHE_WRITE_LOCK = Lock()
logger = logging.getLogger(__name__)


class ReplayCacheMissError(RuntimeError):
    """Raised when replay mode cannot find a call in the recorded cache."""


class _SynchronizedJSONLSink(JSONLSink):
    """Serialize appends when concurrent runs share a job cache file."""

    def write(self, **kwargs: Any) -> None:
        with _CACHE_WRITE_LOCK:
            super().write(**kwargs)


class _LiteLLMBackend:
    """Adapt LiteLLM's response object to the gateway response contract."""

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tool_schema: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import litellm

        started = time.perf_counter()
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tool_schema,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):
            raise TypeError("LiteLLM returned a non-text response; OASIS expects text output")

        usage = response.usage
        prompt_tokens = int(usage.prompt_tokens or 0)
        completion_tokens = int(usage.completion_tokens or 0)
        return {
            "content": content,
            "model": response.model or model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.total_tokens or prompt_tokens + completion_tokens),
            },
            "cost_usd": float(litellm.completion_cost(completion_response=response)),
            "latency_ms": elapsed_ms,
        }


class _ReplayLLMBackend:
    """Serve responses from the recorded JSONL cache without network access."""

    def __init__(self, sink: JSONLSink) -> None:
        self._sink = sink

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tool_schema: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prompt_sha = hash_request(model, messages, temperature, tool_schema)
        record = self._sink.lookup_full(prompt_sha)
        if record is None:
            raise ReplayCacheMissError(
                f"No cached response for prompt hash {prompt_sha}"
            )
        return record["response"]


def load_budget_profile(profile_name: str = "moderate") -> Budget:
    """Load budget dimensions from config/budget_profiles.yaml."""
    cfg_file = CONFIG_DIR / "budget_profiles.yaml"
    if cfg_file.exists():
        content = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        if profile_name in content:
            return Budget(**content[profile_name])
    # Fallback default moderate profile
    return Budget(max_tokens=200000, max_cost_usd=0.75, max_wall_seconds=300.0, max_calls=60)


class BenchmarkRunner:
    """Async execution harness for the 8-arm ablation matrix."""

    def __init__(
        self,
        db: Database | None = None,
        concurrency: int = 8,
        mode: str = "live",
        spend_ceiling_usd: float = 250.0,
        cache_dir: Path | str = "./artifacts/llm_calls",
        fake_llm: FakeLLM | None = None,
    ) -> None:
        if mode not in ("live", "replay"):
            raise ValueError(f"Unsupported run mode: {mode!r}; expected 'live' or 'replay'")
        self.db = db or Database()
        self.concurrency = concurrency
        self.mode = mode
        self.spend_ceiling_usd = float(os.environ.get("OASIS_SPEND_CEILING_USD", spend_ceiling_usd))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fake_llm = fake_llm

    def _backend_and_sink(
        self, job_id: str | None
    ) -> tuple[Any, JSONLSink]:
        cache_path = self.cache_dir / f"{job_id or 'runs'}.jsonl"
        if self.fake_llm is not None:
            return self.fake_llm, _SynchronizedJSONLSink(cache_path)
        if self.mode == "live":
            return _LiteLLMBackend(), _SynchronizedJSONLSink(cache_path)

        replay_output = self.cache_dir / "replay" / f"{job_id or 'runs'}.jsonl"
        return _ReplayLLMBackend(JSONLSink(cache_path)), _SynchronizedJSONLSink(replay_output)

    def get_adapter(self, framework: str) -> AdapterBase:
        """Instantiate framework-specific adapter."""
        if framework == "langgraph":
            return LangGraphAdapter()
        if framework == "autogen":
            return AutoGenAdapter()
        if framework == "crewai":
            return CrewAIStub()
        raise ValueError(f"Unsupported framework: '{framework}'")

    def prepare_team_roles(
        self, task: TaskRecord, arm: str, budget: Budget
    ) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
        """Configure team roles and pre-execution reductions for the chosen arm."""
        all_roles = ["researcher", "analyst", "planner", "executor", "reviewer", "synthesizer"]
        roles: list[dict[str, Any]] = []
        origin = "baseline"
        reductions: list[dict[str, Any]] = []

        if arm == "single_agent":
            roles = [{
                "role": "generalist",
                "template_id": "fast_cheap",
                "model": "gpt-4o-mini-2024-07-18",
                "temperature": 0.2,
                "est_tokens": 10000,
                "est_cost_usd": 0.02,
                "est_wall_seconds": 15.0,
                "est_calls": 4,
            }]
        elif arm == "captainagent":
            origin = "baseline"
            roles = [
                {"role": "lead", "template_id": "accurate_slow", "model": "gpt-4o-2024-11-20", "temperature": 0.2, "est_tokens": 15000, "est_cost_usd": 0.04, "est_wall_seconds": 20.0, "est_calls": 5},
                {"role": "specialist", "template_id": "fast_cheap", "model": "gpt-4o-mini-2024-07-18", "temperature": 0.2, "est_tokens": 12000, "est_cost_usd": 0.03, "est_wall_seconds": 15.0, "est_calls": 4},
            ]
        elif arm in ("tce_only", "tce_rbe", "full"):
            origin = "estimator"
            mvts, _, _, _ = _estimate_mvts(task.statement, task.domain)
            for i in range(mvts):
                roles.append({
                    "role": all_roles[i % len(all_roles)],
                    "template_id": "fast_cheap",
                    "model": "gpt-4o-mini-2024-07-18",
                    "temperature": 0.2,
                    "est_tokens": 15000,
                    "est_cost_usd": 0.03,
                    "est_wall_seconds": 20.0,
                    "est_calls": 4,
                })
        else:
            # vanilla_fixed, rbe_only, monitor_only (default size 3)
            origin = "baseline"
            for r in ["researcher", "analyst", "reviewer"]:
                roles.append({
                    "role": r,
                    "template_id": "fast_cheap",
                    "model": "gpt-4o-mini-2024-07-18",
                    "temperature": 0.2,
                    "est_tokens": 15000,
                    "est_cost_usd": 0.03,
                    "est_wall_seconds": 20.0,
                    "est_calls": 4,
                })

        # Apply pre-execution reduction if arm includes RBE pre-execution
        if arm in ("rbe_only", "tce_rbe", "full"):
            reducer = TeamReducer()
            try:
                roles, reductions = reducer.reduce(roles, budget)
            except BudgetInfeasibleError:
                # Keep single role if infeasible
                roles = roles[:1]

        return roles, origin, reductions

    async def run_task(
        self,
        task: TaskRecord,
        arm: str = "full",
        seed: int = 1,
        budget: Budget | None = None,
        framework: str = "langgraph",
        job_id: str | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        """Execute a single benchmark run under the specified arm and budget."""
        if arm not in _EXECUTABLE_ARMS and arm not in _UNAVAILABLE_ARMS:
            raise ValueError(f"Unsupported benchmark arm: {arm!r}")
        if arm in _UNAVAILABLE_ARMS:
            raise NotImplementedError(
                f"Cannot execute arm {arm!r}: {_UNAVAILABLE_ARMS[arm]}"
            )

        run_id = run_id or str(ulid.ULID())
        run_budget = budget or load_budget_profile("moderate")

        # 1. Check cumulative spend ceiling (NFR-3)
        if self.db.get_cumulative_spend() >= self.spend_ceiling_usd:
            existing_run = self.db.get_run(run_id)
            if existing_run is not None:
                self.db.update_run_status(
                    run_id,
                    status="failed",
                    status_detail="global_spend_limit",
                )
                final_run = self.db.get_run(run_id)
                assert final_run is not None
                return final_run
            halted_run = RunRecord(
                run_id=run_id,
                task_id=task.task_id,
                job_id=job_id,
                arm=arm,
                seed=seed,
                framework=framework,
                mode=self.mode,
                budget_json=run_budget.model_dump(),
                status="failed",
                status_detail="global_spend_limit",
            )
            self.db.create_run(halted_run)
            return halted_run

        budget_state = BudgetState(run_budget)

        # 2. Pre-execution Team Configuration
        t0_orch = time.perf_counter()
        roles, origin, _reductions = self.prepare_team_roles(task, arm, run_budget)

        # 3. Create Run & Team Records in DB
        budget_dict = {
            "max_tokens": run_budget.max_tokens,
            "max_cost_usd": run_budget.max_cost_usd,
            "max_wall_seconds": run_budget.max_wall_seconds,
            "max_calls": run_budget.max_calls,
        }
        run_record = RunRecord(
            run_id=run_id,
            task_id=task.task_id,
            job_id=job_id,
            arm=arm,
            seed=seed,
            framework=framework,
            mode=self.mode,
            budget_json=budget_dict,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        existing_run = self.db.get_run(run_id)
        if existing_run is None:
            self.db.create_run(run_record)
        elif existing_run.status == "queued":
            self.db.update_run_status(run_id, status="running")
        else:
            raise ValueError(
                f"Run {run_id!r} cannot start from status {existing_run.status!r}"
            )

        # Record team config
        team_cfg = TeamConfigRecord(
            run_id=run_id,
            team_size=len(roles),
            origin=origin,
            topology="hub_and_spoke",
            roles_json=roles,
        )
        self.db.insert_team_config(team_cfg)

        # Record initial agent instances
        agent_instances: dict[str, str] = {}
        for idx, r in enumerate(roles):
            ag_id = f"ag_{run_id[-6:]}_{idx+1}"
            agent_instances[r["role"]] = ag_id
            self.db.insert_agent_instance(
                AgentInstanceRecord(
                    agent_id=ag_id,
                    run_id=run_id,
                    role=r["role"],
                    template_id=r.get("template_id", "fast_cheap"),
                    model=r.get("model", "gpt-4o-mini-2024-07-18"),
                    temperature=r.get("temperature", 0.2),
                    active_from_step=1,
                )
            )

        broadcast_run_event(run_id, "run_started", {"run_id": run_id, "arm": arm, "team_size": len(roles)})

        # 4. Set up Gateway & Sinks
        llm_backend, sink = self._backend_and_sink(job_id)
        gw = Gateway(
            sink=sink,
            llm=llm_backend,
            budget_state=budget_state,
            run_id=run_id,
            agent_id=next(iter(agent_instances.values())),
            purpose="productive",
        )

        adapter = self.get_adapter(framework)

        # 5. Execution Loop
        status = "completed"
        status_detail: str | None = None
        step = 0
        task_prompt = task.statement
        prior_outputs: list[tuple[str, str]] = []

        try:
            runnable = adapter.build({"roles": roles, "topology": "hub_and_spoke"})
            for event in adapter.run(runnable, task_prompt):
                role_name = event.get("agent", "researcher")
                if role_name == "supervisor":
                    continue
                step += 1
                current_ag_id = agent_instances.get(role_name)
                if current_ag_id is None:
                    raise RuntimeError(f"Adapter emitted unknown agent role {role_name!r}")

                role_config = next(
                    (role for role in roles if role.get("role") == role_name),
                    {},
                )
                system_prompt = role_config.get(
                    "system_prompt",
                    f"You are the {role_name} agent. Work on the task and use prior agent output where helpful.",
                )
                context = "\n\n".join(
                    f"{name}: {content}" for name, content in prior_outputs
                )
                user_content = f"Task:\n{task_prompt}"
                if event.get("task"):
                    user_content += f"\n\nCurrent subtask:\n{event['task']}"
                if context:
                    user_content += f"\n\nPrior agent outputs:\n{context}"
                call_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                model = role_config.get(
                    "model",
                    os.environ.get("OASIS_MODEL_DEFAULT", "gpt-4o-mini-2024-07-18"),
                )
                temperature = float(role_config.get("temperature", 0.2))

                try:
                    resp = await asyncio.to_thread(
                        gw.call,
                        model=model,
                        messages=call_messages,
                        temperature=temperature,
                        agent_id=current_ag_id,
                    )
                except GatewayHaltError:
                    status = "halted_budget"
                    status_detail = "Halted by RBE admission control"
                    break

                content = resp.get("content", "")
                accounting = resp["_accounting"]
                content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                prompt_sha = resp["_prompt_sha"]
                out_id = str(ulid.ULID())

                output_rec = AgentOutputRecord(
                    output_id=out_id,
                    run_id=run_id,
                    agent_id=current_ag_id,
                    step=step,
                    content=content,
                    content_sha=content_sha,
                    prompt_sha=prompt_sha,
                    tokens_in=accounting["tokens_in"],
                    tokens_out=accounting["tokens_out"],
                    cost_usd=accounting["cost_usd"],
                    latency_ms=accounting["latency_ms"],
                )
                self.db.insert_agent_output(output_rec)
                prior_outputs.append((role_name, content))

        except ReplayCacheMissError as exc:
            status = "failed"
            status_detail = f"cache_miss_in_replay: {exc}"
        except Exception as exc:  # noqa: BLE001 — provider and adapter errors must finalize the run.
            status = "failed"
            status_detail = f"execution_error: {type(exc).__name__}"
            logger.error("Run %s failed during execution (%s)", run_id, type(exc).__name__)

        finally:
            try:
                adapter.teardown()
            except Exception as exc:  # noqa: BLE001 — teardown failures must not leave the run active.
                status = "failed"
                status_detail = f"adapter_teardown_error: {type(exc).__name__}"
                logger.error("Run %s adapter teardown failed (%s)", run_id, type(exc).__name__)

        # 6. Finalize Run Record
        t_orch_end = time.perf_counter()
        elapsed_ms = int((t_orch_end - t0_orch) * 1000)
        provider_latency_ms = sum(
            float(record.get("latency_ms", 0.0)) for record in gw.accounting_log
        )
        if self.mode == "replay" and self.fake_llm is None:
            orch_ms = elapsed_ms
            wall_ms = elapsed_ms + int(provider_latency_ms)
        elif self.mode == "live" and self.fake_llm is None:
            wall_ms = elapsed_ms
            orch_ms = max(0, int(elapsed_ms - provider_latency_ms))
        else:
            wall_ms = elapsed_ms
            orch_ms = elapsed_ms

        tot_tokens = budget_state.consumed("max_tokens")
        tot_cost = budget_state.consumed("max_cost_usd")
        tot_calls = budget_state.consumed("max_calls")

        tokens_ok = tot_tokens <= run_budget.max_tokens
        cost_ok = tot_cost <= run_budget.max_cost_usd
        calls_ok = tot_calls <= run_budget.max_calls
        wall_ok = wall_ms / 1000.0 <= run_budget.max_wall_seconds
        compliant_all = tokens_ok and cost_ok and calls_ok and wall_ok

        self.db.finalize_run(
            run_id=run_id,
            totals={
                "tokens": tot_tokens,
                "cost_usd": tot_cost,
                "calls": tot_calls,
                "supervision_tokens": budget_state.supervision_tokens,
                "supervision_cost_usd": budget_state.supervision_cost_usd,
            },
            compliant_all=compliant_all,
            quality_composite=None,
            status=status,
            status_detail=status_detail,
            wall_ms=wall_ms,
            orchestration_ms=orch_ms,
        )

        broadcast_run_event(run_id, "run_completed", {"run_id": run_id, "status": status, "compliant": compliant_all})
        final_run = self.db.get_run(run_id)
        assert final_run is not None
        return final_run

    async def run_matrix(
        self,
        tasks: Sequence[TaskRecord],
        arms: Sequence[str] = ("vanilla_fixed", "full"),
        seeds: Sequence[int] = (1,),
        job_id: str | None = None,
        concurrency: int | None = None,
        budget: Budget | None = None,
        tranche_size: int = 10,
    ) -> list[RunRecord]:
        """Execute the matrix across tasks, arms, and seeds with bounded concurrency."""
        unknown = [arm for arm in arms if arm not in _EXECUTABLE_ARMS and arm not in _UNAVAILABLE_ARMS]
        if unknown:
            raise ValueError(f"Unsupported benchmark arm(s): {unknown}")
        unavailable = [arm for arm in arms if arm in _UNAVAILABLE_ARMS]
        if unavailable:
            arm = unavailable[0]
            raise NotImplementedError(
                f"Cannot execute matrix arm {arm!r}: {_UNAVAILABLE_ARMS[arm]}"
            )

        limit = concurrency or self.concurrency
        semaphore = asyncio.Semaphore(limit)
        all_results: list[RunRecord] = []

        # Build grid
        grid: list[tuple[TaskRecord, str, int]] = []
        for task in tasks:
            for arm in arms:
                for seed in seeds:
                    grid.append((task, arm, seed))

        async def worker(task_item: TaskRecord, arm_name: str, seed_num: int) -> RunRecord:
            async with semaphore:
                run_id = str(ulid.ULID())
                try:
                    return await self.run_task(
                        task_item,
                        arm=arm_name,
                        seed=seed_num,
                        budget=budget,
                        job_id=job_id,
                        run_id=run_id,
                    )
                except Exception as exc:  # noqa: BLE001 — each matrix cell must return a terminal record.
                    logger.error("Matrix run %s failed before finalization (%s)", run_id, type(exc).__name__)
                    existing = self.db.get_run(run_id)
                    if existing is None:
                        existing = RunRecord(
                            run_id=run_id,
                            task_id=task_item.task_id,
                            job_id=job_id,
                            arm=arm_name,
                            seed=seed_num,
                            framework="langgraph",
                            mode=self.mode,
                            budget_json=(budget or load_budget_profile("moderate")).model_dump(),
                            status="failed",
                            status_detail=f"execution_error: {type(exc).__name__}",
                            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                        self.db.create_run(existing)
                    else:
                        self.db.update_run_status(
                            run_id,
                            status="failed",
                            status_detail=f"execution_error: {type(exc).__name__}",
                            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    broadcast_run_event(
                        run_id,
                        "run_completed",
                        {"run_id": run_id, "status": "failed"},
                    )
                    failed_run = self.db.get_run(run_id)
                    assert failed_run is not None
                    return failed_run

        # Process in tranches
        for i in range(0, len(grid), tranche_size):
            # Check spend ceiling between tranches (NFR-3)
            if self.db.get_cumulative_spend() >= self.spend_ceiling_usd:
                break

            tranche = grid[i : i + tranche_size]
            tasks_coros = [worker(t, a, s) for t, a, s in tranche]
            tranche_results = await asyncio.gather(*tasks_coros)
            all_results.extend(tranche_results)

        return all_results
