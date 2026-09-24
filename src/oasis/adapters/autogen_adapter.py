"""Shallow integration: GroupChat + custom speaker selector, boundary swap (ADR-008).

AutoGen supports task-boundary swaps through a custom speaker selector
but not mid-run replacement.  The adapter shall declare this capability
truthfully (FR-19).

ARCHITECTURE.md §5:
  "ASL translates the final configuration into framework-native constructs:
   ...a GroupChat with a custom speaker selector for AutoGen."
  "AutoGen reports task-boundary swap only."

ADR-008:
  "AutoGen supports boundary-level swaps through a custom speaker selector.
   Adapters declare capabilities truthfully and results are reported by
   capability rather than averaged."

Operations:
  - ``build(config)`` — creates an AutoGen GroupChat with custom speaker selector.
  - ``run(runnable, task)`` — invokes the chat and yields output events.
  - ``swap(runnable, out_agent, in_agent, handoff)`` — queues replacement for the
    NEXT task boundary (does NOT replace mid-turn).
  - ``capabilities()`` — reports ``mid_run_swap=False``, ``boundary_swap=True``.
  - ``teardown()`` — releases chat resources.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from oasis.adapters.base import NOT_SUPPORTED_SIGNAL, AdapterBase, AdapterCapabilities


@dataclass
class GroupChatAgent:
    """Representation of an agent within an AutoGen GroupChat."""

    name: str
    system_message: str = ""
    description: str = ""
    model: str | None = None

    def generate_reply(self, message: str | dict[str, Any] | None = None) -> str:
        """Produce an in-task turn output from this agent."""
        return f"[{self.name}] Turn reply for: {message}"


@dataclass
class PendingSwap:
    """A replacement queued to take effect at the next task boundary."""

    out_agent: str
    in_agent: str
    handoff: dict[str, Any] | None = None
    queued_at_turn: int = 0


class CustomSpeakerSelector:
    """Custom speaker selector managing agent turns and task-boundary swaps (FR-19).

    Replacements queued via ``queue_swap`` do NOT affect the active roster
    immediately. They take effect only when ``apply_boundary_swaps`` is invoked
    at a task boundary.
    """

    def __init__(self, agent_names: list[str]) -> None:
        self.roster: list[str] = list(agent_names)
        self.pending_swaps: list[PendingSwap] = []
        self._current_index: int = 0
        self.active_speaker: str | None = None
        self.swap_history: list[dict[str, Any]] = []

    def queue_swap(
        self,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
        current_turn: int = 0,
    ) -> bool:
        """Queue a swap to be applied at the next task boundary."""
        self.pending_swaps.append(
            PendingSwap(
                out_agent=out_agent,
                in_agent=in_agent,
                handoff=handoff,
                queued_at_turn=current_turn,
            )
        )
        return True

    def apply_boundary_swaps(self) -> list[dict[str, Any]]:
        """Apply all queued swaps at the task boundary, updating the active roster."""
        executed: list[dict[str, Any]] = []
        for pending in self.pending_swaps:
            if pending.out_agent in self.roster:
                idx = self.roster.index(pending.out_agent)
                self.roster[idx] = pending.in_agent
            else:
                self.roster.append(pending.in_agent)

            # If the active speaker was swapped, update active_speaker reference
            if self.active_speaker == pending.out_agent:
                self.active_speaker = pending.in_agent

            record = {
                "out_agent": pending.out_agent,
                "in_agent": pending.in_agent,
                "handoff": pending.handoff,
                "applied_at": "task_boundary",
                "queued_at_turn": pending.queued_at_turn,
            }
            self.swap_history.append(record)
            executed.append(record)

        self.pending_swaps.clear()
        return executed

    def select_speaker(
        self,
        last_speaker: str | None = None,
        subtask: Any = None,
        turn_in_phase: int = 0,
        is_new_phase: bool = False,
    ) -> str:
        """Select the next speaker from the active roster.

        Within a task phase, the assigned speaker produces turns until the
        phase finishes. At a task boundary (is_new_phase=True), the next speaker
        from the roster is selected (reflecting any boundary swaps that occurred).
        """
        if not self.roster:
            raise ValueError("GroupChat roster is empty")

        # Explicit agent in subtask dict
        if isinstance(subtask, dict) and "agent" in subtask:
            specified = subtask["agent"]
            if specified in self.roster:
                self.active_speaker = specified
                return specified

        # Continuing within current phase: old speaker finishes the in-task turns
        if not is_new_phase and turn_in_phase > 0 and self.active_speaker in self.roster:
            return self.active_speaker

        # New phase: choose next from roster
        speaker = self.roster[self._current_index % len(self.roster)]
        self.active_speaker = speaker
        return speaker


class GroupChat:
    """GroupChat structure matching AutoGen/AG2 with custom speaker selection."""

    def __init__(
        self,
        agents: list[GroupChatAgent],
        speaker_selector: CustomSpeakerSelector,
        max_round: int = 10,
    ) -> None:
        self.agents: dict[str, GroupChatAgent] = {agent.name: agent for agent in agents}
        self.speaker_selector: CustomSpeakerSelector = speaker_selector
        self.max_round: int = max_round
        self.messages: list[dict[str, Any]] = []


class AutoGenRunnable:
    """Runnable representing an active AutoGen GroupChat session."""

    def __init__(
        self,
        groupchat: GroupChat,
        config: dict[str, Any],
        turns_per_task: int = 2,
    ) -> None:
        self.groupchat = groupchat
        self.config = config
        self.turns_per_task = turns_per_task
        self.current_turn: int = 0
        self.is_torn_down: bool = False
        self.active_agent: str | None = None
        self.handoffs: dict[str, Any] = {}

    @property
    def roster(self) -> list[str]:
        """The currently active agent roster."""
        return self.groupchat.speaker_selector.roster

    @property
    def pending_swaps(self) -> list[PendingSwap]:
        """Swaps awaiting the next task boundary."""
        return self.groupchat.speaker_selector.pending_swaps

    def queue_swap(
        self,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Queue a swap on the speaker selector.

        Returns True if the swap was successfully queued for the boundary.
        """
        if self.is_torn_down:
            return False

        # Register replacement agent in groupchat if not already present
        if in_agent not in self.groupchat.agents:
            self.groupchat.agents[in_agent] = GroupChatAgent(
                name=in_agent,
                system_message=f"Replacement agent for {out_agent}",
            )

        if handoff is not None:
            self.handoffs[in_agent] = handoff

        return self.groupchat.speaker_selector.queue_swap(
            out_agent=out_agent,
            in_agent=in_agent,
            handoff=handoff,
            current_turn=self.current_turn,
        )

    def trigger_boundary(self) -> list[dict[str, Any]]:
        """Trigger a task boundary and apply all pending swaps."""
        if self.is_torn_down:
            return []
        return self.groupchat.speaker_selector.apply_boundary_swaps()

    def teardown(self) -> None:
        """Tear down GroupChat resources."""
        self.is_torn_down = True
        self.groupchat.messages.clear()
        self.groupchat.speaker_selector.pending_swaps.clear()


class AutoGenAdapter(AdapterBase):
    """AutoGen adapter — shallow integration with boundary-only swap (FR-19)."""

    def __init__(self) -> None:
        self._active_runnables: list[AutoGenRunnable] = []

    def build(self, config: dict[str, Any]) -> AutoGenRunnable:
        """Create an AutoGen GroupChat with custom speaker selector.

        ARCHITECTURE.md §5: 'build(config) returns a runnable'

        Parameters
        ----------
        config:
            Team configuration dict (roles, topology, model assignments).

        Returns
        -------
        AutoGenRunnable
            Framework-native runnable wrapping the GroupChat and custom speaker selector.
        """
        raw_roles = config.get("roles") or config.get("agents") or ["agent_1", "agent_2"]
        agent_names: list[str] = []
        agents: list[GroupChatAgent] = []

        for role in raw_roles:
            if isinstance(role, dict):
                name = role.get("name") or role.get("role", "unknown_agent")
                sys_msg = role.get("system_message", "")
                model = role.get("model")
            else:
                name = str(role)
                sys_msg = f"Role: {name}"
                model = config.get("model")

            agent_names.append(name)
            agents.append(GroupChatAgent(name=name, system_message=sys_msg, model=model))

        speaker_selector = CustomSpeakerSelector(agent_names=agent_names)
        max_round = int(config.get("max_round", 10))
        turns_per_task = int(config.get("turns_per_task", 2))

        groupchat = GroupChat(
            agents=agents,
            speaker_selector=speaker_selector,
            max_round=max_round,
        )

        runnable = AutoGenRunnable(
            groupchat=groupchat,
            config=config,
            turns_per_task=turns_per_task,
        )
        self._active_runnables.append(runnable)
        return runnable

    def run(
        self,
        runnable: Any,
        task: str | dict[str, Any],
    ) -> Generator[dict[str, Any], None, None]:
        """Invoke the group chat and yield output events as agents produce turns.

        ARCHITECTURE.md §5: 'run(runnable, task) yields output events'

        Parameters
        ----------
        runnable:
            The AutoGenRunnable returned by :meth:`build`.
        task:
            The task statement (string or dict with subtasks).

        Yields
        ------
        dict[str, Any]
            Output event dicts with at minimum ``{"agent": str, "content": str}``.
        """
        if runnable is None or not isinstance(runnable, AutoGenRunnable):
            raise ValueError(f"Expected AutoGenRunnable, got {type(runnable)}")

        if runnable.is_torn_down:
            raise RuntimeError("Cannot run a torn down AutoGenRunnable")

        # Parse subtasks / phases
        if isinstance(task, dict):
            subtasks = (
                task.get("subtasks")
                or task.get("tasks")
                or task.get("steps")
                or [str(task.get("task", "task_1"))]
            )
            turns_per_subtask = int(
                task.get("turns_per_subtask")
                or task.get("turns_per_task")
                or runnable.turns_per_task
            )
        elif isinstance(task, str):
            if "\n---\n" in task:
                subtasks = [s.strip() for s in task.split("\n---\n") if s.strip()]
                turns_per_subtask = runnable.turns_per_task
            else:
                num_phases = int(runnable.config.get("num_phases", 2))
                subtasks = [f"{task}_phase_{i+1}" for i in range(num_phases)]
                turns_per_subtask = runnable.turns_per_task
        else:
            subtasks = ["default_task"]
            turns_per_subtask = runnable.turns_per_task

        for phase_idx, subtask in enumerate(subtasks):
            # When transitioning to a new subtask (phase > 0), a task boundary is reached!
            if phase_idx > 0:
                runnable.trigger_boundary()

            for turn_in_phase in range(turns_per_subtask):
                runnable.current_turn += 1
                speaker = runnable.groupchat.speaker_selector.select_speaker(
                    subtask=subtask,
                    turn_in_phase=turn_in_phase,
                    is_new_phase=(turn_in_phase == 0),
                )
                runnable.active_agent = speaker

                agent_obj = runnable.groupchat.agents.get(speaker)
                reply = (
                    agent_obj.generate_reply(subtask)
                    if agent_obj
                    else f"Response from {speaker}"
                )

                event = {
                    "agent": speaker,
                    "content": reply,
                    "task": subtask,
                    "turn": runnable.current_turn,
                    "turn_in_phase": turn_in_phase + 1,
                    "phase": phase_idx + 1,
                }
                runnable.groupchat.messages.append(event)
                yield event

    def swap(
        self,
        runnable: Any,
        out_agent: str,
        in_agent: str,
        handoff: dict[str, Any] | None = None,
    ) -> bool:
        """Replace an agent at the next task boundary (FR-19).

        ARCHITECTURE.md §5: 'swap(runnable, out_agent, in_agent, handoff) returns success or a not-supported signal'

        In AutoGen, mid-run swapping is not supported. The swap is queued
        and takes effect strictly at the next task boundary.
        """
        if runnable is None or not isinstance(runnable, AutoGenRunnable):
            return NOT_SUPPORTED_SIGNAL

        if runnable.is_torn_down:
            return NOT_SUPPORTED_SIGNAL

        return runnable.queue_swap(
            out_agent=out_agent,
            in_agent=in_agent,
            handoff=handoff,
        )

    def capabilities(self) -> AdapterCapabilities:
        """Declare capability: AutoGen supports boundary swap only (FR-19).

        ARCHITECTURE.md §5: 'capabilities() declares whether mid-run swap is available'
        FR-19: Must truthfully report boundary-only swap (mid_run_swap=False, boundary_swap=True).
        """
        return AdapterCapabilities(mid_run_swap=False, boundary_swap=True)

    def teardown(self) -> None:
        """Release chat resources."""
        for runnable in self._active_runnables:
            runnable.teardown()
        self._active_runnables.clear()


__all__ = [
    "AutoGenAdapter",
    "AutoGenRunnable",
    "CustomSpeakerSelector",
    "GroupChat",
    "GroupChatAgent",
    "PendingSwap",
]
