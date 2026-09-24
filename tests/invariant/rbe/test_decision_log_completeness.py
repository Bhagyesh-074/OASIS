"""Invariant test for FR-24: every automated decision writes a decision_log record

with component == 'RBE' and a non-empty human-readable justification.
Guards that zero decisions are logged without a justification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis.gateway.fake_llm import FakeLLM
from oasis.gateway.jsonl_sink import JSONLSink
from oasis.gateway.litellm_hooks import Gateway
from oasis.rbe.budget import Budget, BudgetState
from oasis.rbe.decision_log_client import (
    clear_decision_log,
    get_decision_log,
)
from oasis.rbe.reducer import TeamReducer


@pytest.fixture
def sink(tmp_path: Path) -> JSONLSink:
    return JSONLSink(tmp_path / "decision_log_test.jsonl")


@pytest.fixture(autouse=True)
def clean_decision_log() -> None:
    clear_decision_log()
    yield
    clear_decision_log()


class TestDecisionLogCompleteness:
    """FR-24 invariant test: Zero decision_log records with empty justification, all component == 'RBE'."""

    def test_full_pipeline_reduction_to_admission_to_halt(self, sink: JSONLSink) -> None:
        """Run full flow: reduction -> admission (allow) -> admission (halt).

        Asserts that:
        1. Both reduction and admission decisions are logged.
        2. Zero decisions have empty or missing justification.
        3. Every single decision from RBE/Gateway has component == 'RBE'.
        """
        # 1. Reduction Phase: team of 3 roles forced to reduce with a tight call/token budget
        initial_team = [
            {"role": "researcher", "description": "Gathers background information"},
            {"role": "analyst", "description": "Analyzes gathered information"},
            {"role": "writer", "description": "Writes up final draft"},
        ]

        # Budget forcing reduction to size 1 (1 role ~2000 tokens <= 3000; 2 roles ~3500 > 3000)
        budget = Budget(
            max_tokens=3000,
            max_cost_usd=0.05,
            max_wall_seconds=30.0,
            max_calls=1,  # Allows 1 call total during execution
        )

        reducer = TeamReducer()
        reduced_team, reduction_log = reducer.reduce(initial_team, budget)

        # Confirm reduction occurred
        assert len(reduced_team) < len(initial_team)
        assert len(reduction_log) >= 1

        # 2. Execution & Admission Phase: run calls through Gateway with FakeLLM
        budget_state = BudgetState(budget)
        model = "gpt-4o-2024-08-06"
        msgs_1 = [{"role": "user", "content": "Produce initial draft"}]
        msgs_2 = [{"role": "user", "content": "Review draft"}]

        fake = FakeLLM()
        fake.script(
            model=model,
            messages=msgs_1,
            response={"content": "draft v1", "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "cost_usd": 0.0002},
        )

        gw = Gateway(
            sink=sink,
            llm=fake,
            budget_state=budget_state,
            return_on_halt=True,
            run_id="run-decisions-1",
        )

        # Call 1: should be admitted and logged as admit_allow
        res_1 = gw.call(model=model, messages=msgs_1, outputs_so_far=[])
        assert res_1["_verdict"] == "allow"

        # Call 2: budget exhausted (max_calls was 1), should halt and be logged as admit_halt & run_halt
        res_2 = gw.call(model=model, messages=msgs_2, outputs_so_far=[{"content": "draft v1"}])
        assert res_2["status"] == "halted_budget"

        # 3. Decision Log Verification (FR-24 Invariant)
        decisions = get_decision_log()
        assert len(decisions) >= 3  # At least: reduce_merge, admit_allow, admit_halt, run_halt

        decision_names = [d.decision for d in decisions]
        assert "reduce_merge" in decision_names
        assert "admit_allow" in decision_names
        assert "admit_halt" in decision_names
        assert "run_halt" in decision_names

        for d in decisions:
            # Component invariant
            assert d.component == "RBE", f"Decision {d.decision} had component {d.component!r}; expected 'RBE'"

            # FR-24 Non-empty justification invariant
            assert isinstance(d.justification, str), f"Decision {d.decision} justification is not a string"
            assert len(d.justification.strip()) > 0, f"Decision {d.decision} had empty justification: {d.justification!r}"
            # Ensure it is a meaningful human-readable sentence (not just a word or code)
            assert " " in d.justification.strip(), f"Decision {d.decision} justification was not a sentence: {d.justification!r}"

            # Inputs dictionary invariant
            assert isinstance(d.inputs, dict)
            assert len(d.inputs) > 0

    def test_zero_decisions_with_empty_justification_enforced_by_schema(self) -> None:
        """Attempting to log a decision with an empty justification raises ValueError."""
        from oasis.rbe.decision_log_client import log_decision

        with pytest.raises(ValueError, match="justification must never be empty"):
            log_decision(
                component="RBE",
                decision="test_decision",
                inputs={"foo": "bar"},
                justification="",
            )

        with pytest.raises(ValueError, match="justification must never be empty"):
            log_decision(
                component="RBE",
                decision="test_decision",
                inputs={"foo": "bar"},
                justification="   ",
            )
