"""Unit tests for admission logic, projection, schema compliance, and halt handling (FR-7 / FR-8)."""

from __future__ import annotations

from oasis.rbe.admission import (
    AdmissionController,
    AdmissionVerdict,
    admission_check,
    get_downgrade_model,
    handle_halt,
    project_call_cost,
)
from oasis.rbe.budget import Budget, BudgetState


class TestAdmissionBoundaries:
    """Test pure boundary logic for admission_check."""

    def test_allow_when_all_dimensions_comfortably_fit(self) -> None:
        remaining = {"tokens": 10_000, "cost": 1.0, "wall": 50.0, "calls": 20}
        projected = {"tokens": 200, "cost": 0.002, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.ALLOW
        assert dim == "tokens"
        assert len(justification) > 0

    def test_halt_when_tokens_exhausted(self) -> None:
        remaining = {"tokens": 0, "cost": 1.0, "wall": 50.0, "calls": 20}
        projected = {"tokens": 50, "cost": 0.001, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.HALT
        assert dim == "tokens"
        assert "exhausted" in justification

    def test_halt_when_cost_exceeds_remaining(self) -> None:
        remaining = {"tokens": 1000, "cost": 0.005, "wall": 50.0, "calls": 20}
        projected = {"tokens": 50, "cost": 0.010, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.HALT
        assert dim == "cost"
        assert "exceeds" in justification

    def test_halt_when_calls_exhausted(self) -> None:
        remaining = {"tokens": 1000, "cost": 1.0, "wall": 50.0, "calls": 0}
        projected = {"tokens": 50, "cost": 0.001, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.HALT
        assert dim == "calls"
        assert len(justification) > 0

    def test_halt_when_wall_seconds_exhausted(self) -> None:
        remaining = {"tokens": 1000, "cost": 1.0, "wall": 0.5, "calls": 5}
        projected = {"tokens": 50, "cost": 0.001, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.HALT
        assert dim == "wall"
        assert len(justification) > 0

    def test_downgrade_when_cost_between_70_and_100_percent(self) -> None:
        remaining = {"tokens": 10_000, "cost": 0.010, "wall": 50.0, "calls": 10}
        projected = {"tokens": 100, "cost": 0.0075, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.DOWNGRADE
        assert dim == "cost"
        assert "downgrade" in justification

    def test_truncate_when_tokens_between_80_and_100_percent(self) -> None:
        remaining = {"tokens": 1000, "cost": 1.0, "wall": 50.0, "calls": 10}
        projected = {"tokens": 850, "cost": 0.001, "wall": 1.0, "calls": 1}
        verdict, dim, justification = admission_check(remaining, projected)
        assert verdict == AdmissionVerdict.TRUNCATE
        assert dim == "tokens"
        assert "truncation" in justification


class TestProjectCallCost:
    """Test resource projection for outbound calls."""

    def test_project_default_rates(self) -> None:
        call_data = {
            "model": "gpt-4o-2024-08-06",
            "messages": [{"role": "user", "content": "A" * 400}],  # ~100 tokens + 100 completion = 200 tokens
        }
        proj = project_call_cost(call_data)
        assert proj["tokens"] == 200.0
        assert proj["cost"] == 200.0 * 0.00001
        assert proj["wall"] == 1.0
        assert proj["calls"] == 1.0

    def test_project_explicit_overrides(self) -> None:
        call_data = {
            "model": "gpt-4o",
            "messages": [],
            "projected_tokens": 500,
            "projected_cost_usd": 0.05,
            "projected_wall_seconds": 2.5,
            "projected_calls": 1,
        }
        proj = project_call_cost(call_data)
        assert proj["tokens"] == 500.0
        assert proj["cost"] == 0.05
        assert proj["wall"] == 2.5
        assert proj["calls"] == 1.0


class TestDowngradeModelMap:
    """Test model downgrade selection."""

    def test_gpt4o_downgrades_to_mini(self) -> None:
        assert get_downgrade_model("gpt-4o") == "gpt-4o-mini"
        assert get_downgrade_model("gpt-4o-2024-08-06") == "gpt-4o-mini"

    def test_claude_downgrades_to_haiku(self) -> None:
        assert get_downgrade_model("claude-3-5-sonnet") == "claude-3-haiku"


class TestBudgetEventSchemaCompliance:
    """Validate that emitted budget_events match DATABASE.md line 112."""

    def test_event_has_all_required_columns(self) -> None:
        budget = Budget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_wall_seconds=60.0,
            max_calls=10,
        )
        state = BudgetState(budget)
        controller = AdmissionController(state, run_id="run-xyz")

        controller.check({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "test"}],
            "step": 3,
        })

        assert len(controller.events) == 1
        event = controller.events[0]

        # Check required columns from DATABASE.md
        required_cols = {
            "event_id",
            "run_id",
            "step",
            "dimension",
            "action",
            "projected",
            "remaining",
            "model_before",
            "model_after",
            "justification",
            "created_at",
        }
        assert set(event.keys()) == required_cols

        # Check CHECK constraints from DATABASE.md
        assert event["dimension"] in {"tokens", "cost", "wall", "calls"}
        assert event["action"] in {"allow", "downgrade", "truncate", "halt"}
        assert event["run_id"] == "run-xyz"
        assert event["step"] == 3
        assert isinstance(event["justification"], str)
        assert len(event["justification"]) > 0
        assert event["model_before"] == "gpt-4o"


class TestHandleHalt:
    """Test FR-8 handle_halt returns best output without raising errors."""

    def test_empty_outputs(self) -> None:
        result = handle_halt([], run_id="run-1")
        assert result["status"] == "halted_budget"
        assert result["best_output"] is None
        assert result["outputs_count"] == 0

    def test_outputs_with_scores(self) -> None:
        outputs = [
            {"text": "draft 1", "score": 0.6},
            {"text": "draft 2", "score": 0.95},
            {"text": "draft 3", "score": 0.4},
        ]
        result = handle_halt(outputs, run_id="run-2")
        assert result["status"] == "halted_budget"
        assert result["best_output"] == {"text": "draft 2", "score": 0.95}
        assert result["outputs_count"] == 3

    def test_outputs_with_custom_quality_fn(self) -> None:
        outputs = ["apple", "watermelon", "fig"]
        result = handle_halt(outputs, quality_fn=lambda x: len(x))
        assert result["status"] == "halted_budget"
        assert result["best_output"] == "watermelon"
