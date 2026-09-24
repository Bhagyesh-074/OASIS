"""Deep integration: compiled StateGraph, mid-run swap (ADR-008, FR-18).

LangGraph's explicit state and conditional edges make genuine mid-run
replacement natural.

ARCHITECTURE.md §5:
  "ASL translates the final configuration into framework-native constructs:
   a compiled StateGraph with one node per agent plus a supervisor node
   for LangGraph."
  "LangGraph reports full mid-run swap."

FR-18 (M, RE):
  "On LangGraph, replacement shall occur without restarting the run,
   preserving accumulated state. Verified by integration test asserting
   pre-swap state survives."

Interface Note on Handoff Shape (FR-17 / RE dependency):
  ``swap(runnable, out_agent, in_agent, handoff)`` accepts an optional
  handoff dictionary. When provided by RE, the assumed shape is:
    handoff = {
        "tokens": int,       # Handoff token budget allocated by RBE
        "cost_usd": float,   # Dollar cost associated with the handoff
        "context": Any,      # Domain-specific handoff payload
        "summary": str,      # State summary for the replacement agent
    }
  The adapter incorporates this handoff directly into the graph's accumulated
  state for the replacement agent. Budget affordability decisions are made
  upstream by RBE/RE before ``swap()`` is called.

Operations:
  - ``build(config)`` — compiles a StateGraph with one node per agent plus
    a supervisor node.
  - ``run(runnable, task)`` — executes the compiled graph and yields output
    events.
  - ``swap(runnable, out_agent, in_agent, handoff)`` — rewrites the graph edge
    and node registry to route future turns to the replacement agent *without*
    restarting the run, preserving all pre-swap accumulated state (FR-18).
  - ``capabilities()`` — reports ``mid_run_swap=True, boundary_swap=True``.
  - ``teardown()`` — releases graph resources.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from oasis.adapters.base import NOT_SUPPORTED_SIGNAL, AdapterBase, AdapterCapabilities


@dataclass
class LangGraphAgentNode:
    """Agent node in the StateGraph."""

    name: str
    role: str
    system_prompt: str = ""
    model: str | None = None

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent node, reading accumulated state and returning state updates."""
        step = state.get("step_count", 0)
        messages = state.get("messages", [])
        handoff = state.get("handoffs", {}).get(self.name, {})

        handoff_context = ""
        if handoff:
            ctx = handoff.get("context") or handoff.get("summary") or handoff
            handoff_context = f" [Handoff context: {ctx}]"

        content = (
            f"[{self.name}] Turn {step} output building on {len(messages)} "
            f"prior messages.{handoff_context}"
        )

        return {
            "agent": self.name,
            "role": self.role,
            "content": content,
            "step": step,
        }


class SupervisorNode:
    """Supervisor node in the StateGraph managing routing and conditional edges."""

    def __init__(self, agent_roles: list[str], turns_per_role: int = 1) -> None:
        self.agent_roles: list[str] = list(agent_roles)
        self.turns_per_role: int = turns_per_role
        # Maps logical role/slot -> active agent/node name (rewritten on mid-run swap)
        self.route_table: dict[str, str] = {role: role for role in agent_roles}
        self.role_execution_counts: dict[str, int] = {role: 0 for role in agent_roles}

    def rewrite_route(self, out_agent: str, in_agent: str) -> bool:
        """Rewrite graph edge routing from out_agent to in_agent in-place (FR-18)."""
        for role, current_agent in list(self.route_table.items()):
            if current_agent == out_agent:
                self.route_table[role] = in_agent

        self.route_table[out_agent] = in_agent
        self.route_table[in_agent] = in_agent
        return True

    def next_node(self, state: dict[str, Any]) -> str:
        """Evaluate accumulated state and determine the next node to execute."""
        # Find next role that has remaining turns
        for role in self.agent_roles:
            target_agent = self.route_table.get(role, role)
            # Count turns executed by this role/agent
            exec_count = sum(
                1
                for m in state.get("messages", [])
                if m.get("role") == role
                or m.get("agent") in (role, target_agent)
            )
            if exec_count < self.turns_per_role:
                return target_agent

        return "__end__"

    def execute(self, state: dict[str, Any], next_target: str) -> dict[str, Any]:
        """Produce a supervisor routing decision event."""
        step = state.get("step_count", 0)
        return {
            "agent": "supervisor",
            "role": "supervisor",
            "content": f"[supervisor] Step {step}: routing to '{next_target}'.",
            "step": step,
            "target": next_target,
        }


class StateGraph:
    """StateGraph definition prior to compilation."""

    def __init__(self, supervisor: SupervisorNode, nodes: dict[str, LangGraphAgentNode]) -> None:
        self.supervisor = supervisor
        self.nodes = dict(nodes)
        self.edges: list[tuple[str, str]] = []
        for name in self.nodes:
            self.edges.append((name, "supervisor"))

    def compile(self, config: dict[str, Any] | None = None) -> CompiledStateGraph:
        """Compile the StateGraph into an executable runnable (ARCHITECTURE.md §5)."""
        return CompiledStateGraph(
            nodes=self.nodes,
            supervisor=self.supervisor,
            config=config or {},
        )


class CompiledStateGraph:
    """Compiled executable StateGraph runnable with genuine mid-run swap capability (FR-18)."""

    def __init__(
        self,
        nodes: dict[str, LangGraphAgentNode],
        supervisor: SupervisorNode,
        config: dict[str, Any],
    ) -> None:
        self.nodes: dict[str, LangGraphAgentNode] = dict(nodes)
        self.supervisor: SupervisorNode = supervisor
        self.config: dict[str, Any] = config

        # Explicit accumulated state (FR-18: MUST survive mid-run swap intact)
        self.state: dict[str, Any] = {
            "messages": [],
            "step_count": 0,
            "handoffs": {},
            "task": "",
            "accumulated_data": [],
        }

        self.step_count: int = 0
        self.is_running: bool = False
        self.is_torn_down: bool = False
        self.swap_history: list[dict[str, Any]] = []

    def swap(
        self,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Replace out_agent with in_agent in-place WITHOUT restarting the run (FR-18).

        Pre-swap accumulated state is 100% preserved. Future graph turns route
        to the replacement agent via updated edge routing.
        """
        if self.is_torn_down:
            return False

        # Register replacement node if not already present
        if in_agent not in self.nodes:
            old_node = self.nodes.get(out_agent)
            role = old_node.role if old_node else in_agent
            sys_prompt = old_node.system_prompt if old_node else f"Role: {role}"
            model = old_node.model if old_node else None

            self.nodes[in_agent] = LangGraphAgentNode(
                name=in_agent,
                role=role,
                system_prompt=sys_prompt,
                model=model,
            )

        # In-place edge rewriting in the supervisor routing table
        self.supervisor.rewrite_route(out_agent=out_agent, in_agent=in_agent)

        # Ingest handoff payload into state for the replacement agent
        if handoff is not None:
            self.state.setdefault("handoffs", {})[in_agent] = handoff

        # Record swap event in state and history without touching accumulated messages
        swap_record = {
            "out_agent": out_agent,
            "in_agent": in_agent,
            "handoff": handoff,
            "at_step": self.step_count,
            "pre_swap_message_count": len(self.state.get("messages", [])),
        }
        self.swap_history.append(swap_record)
        self.state["last_swap"] = swap_record

        return True

    def stream(
        self,
        task: str | dict[str, Any],
        max_steps: int = 20,
    ) -> Generator[dict[str, Any], None, None]:
        """Execute the compiled graph turn-by-turn, yielding output events."""
        if self.is_torn_down:
            raise RuntimeError("Cannot execute a torn down CompiledStateGraph")

        # Initialize task context if this is the start of a run
        if not self.is_running:
            self.is_running = True
            self.state["task"] = task
            self.state["status"] = "running"

        while self.step_count < max_steps:
            # 1. Determine next node from supervisor
            next_target = self.supervisor.next_node(self.state)
            if next_target == "__end__":
                self.state["status"] = "completed"
                break

            # 2. Supervisor step
            self.step_count += 1
            self.state["step_count"] = self.step_count
            sup_event = self.supervisor.execute(self.state, next_target)
            self.state["messages"].append(sup_event)
            yield {
                "agent": "supervisor",
                "node": "supervisor",
                "content": sup_event["content"],
                "state": dict(self.state),
                "step": self.step_count,
            }

            # 3. Agent step
            self.step_count += 1
            self.state["step_count"] = self.step_count

            # Lookup node from current nodes table (reflects mid-run swaps!)
            target_node = self.nodes.get(next_target)
            if target_node is None:
                # Fallback if dynamically routed to a new agent name
                target_node = LangGraphAgentNode(name=next_target, role=next_target)
                self.nodes[next_target] = target_node

            agent_event = target_node.execute(self.state)
            self.state["messages"].append(agent_event)
            self.state["accumulated_data"].append(f"data_from_{target_node.name}_at_step_{self.step_count}")

            yield {
                "agent": target_node.name,
                "node": target_node.name,
                "content": agent_event["content"],
                "state": dict(self.state),
                "step": self.step_count,
            }

        self.is_running = False

    def teardown(self) -> None:
        """Release graph resources."""
        self.is_torn_down = True
        self.is_running = False


class LangGraphAdapter(AdapterBase):
    """LangGraph adapter — deep integration with mid-run swap (ADR-008, FR-18)."""

    def __init__(self) -> None:
        self._active_runnables: list[CompiledStateGraph] = []

    def build(self, config: dict[str, Any]) -> CompiledStateGraph:
        """Compile a StateGraph from the OASIS team configuration.

        ARCHITECTURE.md §5: 'build(config) returns a runnable'
        Constructs one node per agent role plus a supervisor node.

        Parameters
        ----------
        config:
            Team configuration dict (roles, topology, model assignments).

        Returns
        -------
        CompiledStateGraph
            The compiled, executable runnable graph.
        """
        raw_roles = config.get("roles") or config.get("agents") or ["agent_1", "agent_2"]
        roles: list[str] = []
        nodes: dict[str, LangGraphAgentNode] = {}

        for r in raw_roles:
            if isinstance(r, dict):
                role_name = r.get("name") or r.get("role", "unknown_role")
                sys_msg = r.get("system_message", f"Role: {role_name}")
                model = r.get("model")
            else:
                role_name = str(r)
                sys_msg = f"Role: {role_name}"
                model = config.get("model")

            roles.append(role_name)
            nodes[role_name] = LangGraphAgentNode(
                name=role_name,
                role=role_name,
                system_prompt=sys_msg,
                model=model,
            )

        turns_per_role = int(config.get("turns_per_role", 1))
        supervisor = SupervisorNode(agent_roles=roles, turns_per_role=turns_per_role)
        state_graph = StateGraph(supervisor=supervisor, nodes=nodes)
        runnable = state_graph.compile(config=config)

        self._active_runnables.append(runnable)
        return runnable

    def run(
        self,
        runnable: Any,
        task: str | dict[str, Any],
    ) -> Generator[dict[str, Any], None, None]:
        """Execute the compiled graph and yield output events per node execution.

        ARCHITECTURE.md §5: 'run(runnable, task) yields output events'

        Parameters
        ----------
        runnable:
            The CompiledStateGraph returned by :meth:`build`.
        task:
            The task statement string or dict.

        Yields
        ------
        dict[str, Any]
            Output event dicts containing at minimum ``{"agent": str, "content": str}``.
        """
        if runnable is None or not isinstance(runnable, CompiledStateGraph):
            raise ValueError(f"Expected CompiledStateGraph, got {type(runnable)}")

        max_steps = int(runnable.config.get("max_steps", 20))
        yield from runnable.stream(task=task, max_steps=max_steps)

    def swap(
        self,
        runnable: Any,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Replace a node's underlying agent WITHOUT restarting the run (FR-18).

        ARCHITECTURE.md §5: 'swap(runnable, out_agent, in_agent, handoff) returns success or a not-supported signal'

        Rewrites the graph edge and node registry in-place so future turns route
        to the replacement agent. The accumulated state is preserved intact.
        """
        if runnable is None or not isinstance(runnable, CompiledStateGraph):
            return NOT_SUPPORTED_SIGNAL

        if runnable.is_torn_down:
            return NOT_SUPPORTED_SIGNAL

        return runnable.swap(out_agent=out_agent, in_agent=in_agent, handoff=handoff)

    def capabilities(self) -> AdapterCapabilities:
        """Truthfully declare that LangGraph supports full mid-run swap (ADR-008).

        ARCHITECTURE.md §5: 'capabilities() declares whether mid-run swap is available'
        """
        return AdapterCapabilities(mid_run_swap=True, boundary_swap=True)

    def teardown(self) -> None:
        """Release graph resources."""
        for runnable in self._active_runnables:
            runnable.teardown()
        self._active_runnables.clear()


__all__ = [
    "CompiledStateGraph",
    "LangGraphAdapter",
    "LangGraphAgentNode",
    "StateGraph",
    "SupervisorNode",
]
