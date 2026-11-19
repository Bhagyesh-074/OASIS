"""Tests for safe evaluation reproduction setup."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oasis.db.models import RunRecord, TaskRecord
from oasis.eval import __main__ as eval_cli


@pytest.mark.asyncio
async def test_reproduction_rejects_nonpositive_task_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_tasks"):
        await eval_cli.run_reproduction(
            "invalid-limit",
            "replay",
            str(tmp_path),
            max_tasks=0,
        )


def test_manifest_requires_a_clean_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M src/oasis/eval/runner.py\n"),
    )

    with pytest.raises(RuntimeError, match="clean git worktree"):
        eval_cli.require_clean_worktree()


@pytest.mark.asyncio
async def test_reproduction_requires_ingested_evaluation_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class EmptyDatabase:
        def list_tasks(self, *, split: str) -> list[TaskRecord]:
            assert split == "eval"
            return []

    monkeypatch.setattr(eval_cli, "Database", EmptyDatabase)

    with pytest.raises(RuntimeError, match="No evaluation-split tasks"):
        await eval_cli.run_reproduction("empty-job", "replay", str(tmp_path))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_reproduction_writes_null_quality_and_run_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = TaskRecord(
        task_id="eval-1",
        domain="code_generation",
        source="mbpp",
        statement="Return the input unchanged",
        split="eval",
    )

    class FakeDatabase:
        runs: list[RunRecord]

        def __init__(self) -> None:
            self.runs = []

        def list_tasks(self, *, split: str) -> list[TaskRecord]:
            assert split == "eval"
            return [task]

        def list_runs(self, *, job_id: str) -> list[RunRecord]:
            assert job_id == "manifest-job"
            return self.runs

        def get_agent_instances(self, run_id: str) -> list[SimpleNamespace]:
            assert run_id == "run-1"
            return [SimpleNamespace(model="gpt-4o-mini-2024-07-18")]

    class FakeRunner:
        def __init__(self, *, db: FakeDatabase, mode: str) -> None:
            assert mode == "replay"
            self.db = db

        async def run_matrix(
            self,
            *,
            tasks: list[TaskRecord],
            arms: tuple[str, ...],
            seeds: tuple[int, ...],
            job_id: str,
        ) -> list[RunRecord]:
            run = RunRecord(
                run_id="run-1",
                task_id=tasks[0].task_id,
                job_id=job_id,
                arm=arms[0],
                seed=seeds[0],
                framework="langgraph",
                mode="replay",
                budget_json={},
                status="completed",
                quality_composite=None,
                compliant_all=1,
            )
            self.db.runs.append(run)
            return [run]

    monkeypatch.setattr(eval_cli, "Database", FakeDatabase)
    monkeypatch.setattr(eval_cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(eval_cli, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(
        eval_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="commit-sha\n"),
    )
    monkeypatch.setattr(
        eval_cli.importlib.metadata,
        "version",
        lambda distribution: f"{distribution}-version",
    )

    await eval_cli.run_reproduction(
        "manifest-job",
        "replay",
        str(tmp_path),
        arms=("vanilla_fixed",),
        seeds=(1,),
    )

    summary = json.loads((tmp_path / "manifest-job_summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest-job_manifest.json").read_text())
    assert summary["arms"]["vanilla_fixed"]["quality"] == {
        "median": None,
        "ci_95": [None, None],
    }
    assert summary["execution"] == {
        "planned_cells": 1,
        "recorded_cells": 1,
        "complete": True,
    }
    assert manifest["git_commit"] == "commit-sha"
    assert manifest["task_ids"] == ["eval-1"]
    assert manifest["run_ids"] == ["run-1"]
    assert manifest["model_identifiers"]["configured"]["OASIS_MODEL_DEFAULT"] == (
        "gpt-4o-mini-2024-07-18"
    )
    assert manifest["model_identifiers"]["observed"] == [
        "gpt-4o-mini-2024-07-18"
    ]
    assert manifest["config_sha256"]


@pytest.mark.asyncio
async def test_reproduction_preserves_existing_job_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = TaskRecord(
        task_id="eval-1",
        domain="code_generation",
        source="mbpp",
        statement="Return the input unchanged",
        split="eval",
    )

    class ExistingJobDatabase:
        def list_tasks(self, *, split: str) -> list[TaskRecord]:
            assert split == "eval"
            return [task]

        def list_runs(self, *, job_id: str) -> list[object]:
            assert job_id == "existing-job"
            return [object()]

    monkeypatch.setattr(eval_cli, "Database", ExistingJobDatabase)

    with pytest.raises(ValueError, match="choose a new job ID"):
        await eval_cli.run_reproduction(
            "existing-job",
            "replay",
            str(tmp_path),
        )

    assert list(tmp_path.iterdir()) == []
