"""Unit tests for OASIS benchmark manifest build and verify (FR-3, NFR-4).

Tests:
1. build then verify passes on fixture ingest (no network).
2. Changing a statement causes verify to fail with a clear message.
3. Moving a split causes verify to fail with a clear message.
4. Removing a task causes verify to fail with a clear message.
5. Changing a revision causes verify to fail with a clear message.
6. Offline check that authored tasks in src/oasis/bench/authored/*.yaml match
   the committed manifest (editing an authored task without rebuilding fails CI).
7. One test behind the real_download marker regenerates the benchmark from
   pinned revisions and verifies against the committed manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis.bench.authored import load_authored_tasks
from oasis.bench.config import load_benchmark_config
from oasis.bench.ingest import ingest_benchmarks
from oasis.bench.manifest import (
    BenchmarkManifest,
    build_manifest,
    compute_spec_sha256,
    compute_statement_sha256,
    main,
    save_manifest,
    verify_manifest,
)
from oasis.bench.models import TaskRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "benchmark.yaml"
MANIFEST_PATH = REPO_ROOT / "config" / "benchmark_manifest.json"


def test_build_then_verify_passes_on_fixture_ingest(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """build then verify passes cleanly on fixture ingest (FR-3, NFR-4)."""
    fixture_tasks_path = tmp_path / "fixture_tasks.jsonl"
    fixture_manifest_path = tmp_path / "fixture_manifest.json"

    # Ingest tasks using fixtures (zero network)
    records = ingest_benchmarks(
        use_fixtures=True,
        fixtures_dir=fixtures_dir,
        output_path=fixture_tasks_path,
    )
    assert len(records) > 0
    assert fixture_tasks_path.exists()

    # Build manifest
    manifest = build_manifest(
        tasks_path=fixture_tasks_path,
        config_path=CONFIG_PATH,
    )
    assert len(manifest.tasks) == len(records)
    save_manifest(manifest, fixture_manifest_path)
    assert fixture_manifest_path.exists()

    # Verify manifest programmatically
    diffs = verify_manifest(
        tasks_path=fixture_tasks_path,
        config_path=CONFIG_PATH,
        manifest_path=fixture_manifest_path,
    )
    assert diffs == [], f"Expected 0 differences on fixture ingest, got: {diffs}"

    # Verify manifest via CLI entrypoint
    ret = main(
        [
            "verify",
            "--tasks",
            str(fixture_tasks_path),
            "--config",
            str(CONFIG_PATH),
            "--manifest",
            str(fixture_manifest_path),
        ]
    )
    assert ret == 0, "CLI verify returned non-zero on fixture manifest"


def test_verify_fails_on_statement_changed(tmp_path: Path, fixtures_dir: Path) -> None:
    """Changing a statement causes verify to fail with a clear message (NFR-4)."""
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "manifest.json"

    # Generate fixture tasks and manifest
    records = ingest_benchmarks(
        use_fixtures=True,
        fixtures_dir=fixtures_dir,
        output_path=tasks_path,
    )
    manifest = build_manifest(tasks_path=tasks_path, config_path=CONFIG_PATH)
    save_manifest(manifest, manifest_path)

    # Modify statement of the first task
    target_task = records[0]
    tampered_tasks_path = tmp_path / "tasks_tampered.jsonl"
    with tampered_tasks_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            if r.task_id == target_task.task_id:
                # Alter problem statement text
                mod_r = TaskRecord(
                    task_id=r.task_id,
                    domain=r.domain,
                    source=r.source,
                    source_ref=r.source_ref,
                    statement=r.statement + " ### MODIFIED STATEMENT",
                    reference_answer=r.reference_answer,
                    verifier_type=r.verifier_type,
                    verifier_spec=r.verifier_spec,
                    complexity_label=r.complexity_label,
                    split=r.split,
                    created_at=r.created_at,
                )
                f.write(mod_r.model_dump_json() + "\n")
            else:
                f.write(r.model_dump_json() + "\n")

    diffs = verify_manifest(
        tasks_path=tampered_tasks_path,
        config_path=CONFIG_PATH,
        manifest_path=manifest_path,
    )
    assert len(diffs) > 0, "Expected verify to fail when statement is modified"
    statement_diffs = [
        d
        for d in diffs
        if f"task '{target_task.task_id}' statement SHA-256 changed" in d
    ]
    assert (
        len(statement_diffs) == 1
    ), f"Expected statement change diff for task '{target_task.task_id}', got: {diffs}"
    assert statement_diffs[0].startswith("changed:")

    # CLI exit code check
    ret = main(
        [
            "verify",
            "--tasks",
            str(tampered_tasks_path),
            "--config",
            str(CONFIG_PATH),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert ret != 0, "CLI verify must exit non-zero when statement is changed"


def test_verify_fails_on_split_moved(tmp_path: Path, fixtures_dir: Path) -> None:
    """Moving a split causes verify to fail with a clear message (FR-3)."""
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "manifest.json"

    records = ingest_benchmarks(
        use_fixtures=True,
        fixtures_dir=fixtures_dir,
        output_path=tasks_path,
    )
    manifest = build_manifest(tasks_path=tasks_path, config_path=CONFIG_PATH)
    save_manifest(manifest, manifest_path)

    # Find an eval task and flip its split to calibration
    eval_task = next(r for r in records if r.split == "eval")
    moved_tasks_path = tmp_path / "tasks_moved.jsonl"
    with moved_tasks_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            if r.task_id == eval_task.task_id:
                mod_r = TaskRecord(
                    task_id=r.task_id,
                    domain=r.domain,
                    source=r.source,
                    source_ref=r.source_ref,
                    statement=r.statement,
                    reference_answer=r.reference_answer,
                    verifier_type=r.verifier_type,
                    verifier_spec=r.verifier_spec,
                    complexity_label=r.complexity_label,
                    split="calibration",
                    created_at=r.created_at,
                )
                f.write(mod_r.model_dump_json() + "\n")
            else:
                f.write(r.model_dump_json() + "\n")

    diffs = verify_manifest(
        tasks_path=moved_tasks_path,
        config_path=CONFIG_PATH,
        manifest_path=manifest_path,
    )
    assert len(diffs) > 0, "Expected verify to fail when split is moved"
    split_diffs = [
        d
        for d in diffs
        if f"split moved: task '{eval_task.task_id}' moved from 'eval' to 'calibration'"
        in d
    ]
    assert (
        len(split_diffs) == 1
    ), f"Expected 'split moved' diff for task '{eval_task.task_id}', got: {diffs}"

    # CLI exit code check
    ret = main(
        [
            "verify",
            "--tasks",
            str(moved_tasks_path),
            "--config",
            str(CONFIG_PATH),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert ret != 0, "CLI verify must exit non-zero when split is moved"


def test_verify_fails_on_task_removed(tmp_path: Path, fixtures_dir: Path) -> None:
    """Removing a task causes verify to fail with a clear message (FR-3, NFR-4)."""
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "manifest.json"

    records = ingest_benchmarks(
        use_fixtures=True,
        fixtures_dir=fixtures_dir,
        output_path=tasks_path,
    )
    manifest = build_manifest(tasks_path=tasks_path, config_path=CONFIG_PATH)
    save_manifest(manifest, manifest_path)

    # Remove the first task
    removed_task = records[0]
    dropped_tasks_path = tmp_path / "tasks_dropped.jsonl"
    with dropped_tasks_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records[1:]:
            f.write(r.model_dump_json() + "\n")

    diffs = verify_manifest(
        tasks_path=dropped_tasks_path,
        config_path=CONFIG_PATH,
        manifest_path=manifest_path,
    )
    assert len(diffs) > 0, "Expected verify to fail when task is removed"
    removed_diffs = [
        d
        for d in diffs
        if f"removed: task '{removed_task.task_id}'" in d
    ]
    assert (
        len(removed_diffs) == 1
    ), f"Expected 'removed' diff for task '{removed_task.task_id}', got: {diffs}"

    # CLI exit code check
    ret = main(
        [
            "verify",
            "--tasks",
            str(dropped_tasks_path),
            "--config",
            str(CONFIG_PATH),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert ret != 0, "CLI verify must exit non-zero when task is removed"


def test_verify_fails_on_revision_changed(tmp_path: Path, fixtures_dir: Path) -> None:
    """Changing a dataset revision in config causes verify to fail with a clear message (NFR-4)."""
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    mod_config_path = tmp_path / "benchmark_mod.yaml"

    ingest_benchmarks(
        use_fixtures=True,
        fixtures_dir=fixtures_dir,
        output_path=tasks_path,
    )
    manifest = build_manifest(tasks_path=tasks_path, config_path=CONFIG_PATH)
    save_manifest(manifest, manifest_path)

    # Copy benchmark.yaml with an altered revision for mbpp
    raw_yaml = CONFIG_PATH.read_text(encoding="utf-8")
    tampered_rev = "0000000000000000000000000000000000000000"
    orig_rev = manifest.datasets["mbpp"].revision
    mod_yaml = raw_yaml.replace(orig_rev, tampered_rev)
    mod_config_path.write_text(mod_yaml, encoding="utf-8", newline="\n")

    diffs = verify_manifest(
        tasks_path=tasks_path,
        config_path=mod_config_path,
        manifest_path=manifest_path,
    )
    assert len(diffs) > 0, "Expected verify to fail when dataset revision changes"
    rev_diffs = [
        d
        for d in diffs
        if f"revision changed: dataset 'mbpp' revision '{orig_rev}' -> '{tampered_rev}'"
        in d
    ]
    assert (
        len(rev_diffs) == 1
    ), f"Expected revision changed diff for mbpp, got: {diffs}"

    # CLI exit code check
    ret = main(
        [
            "verify",
            "--tasks",
            str(tasks_path),
            "--config",
            str(mod_config_path),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert ret != 0, "CLI verify must exit non-zero when dataset revision changes"


def test_authored_tasks_match_committed_manifest() -> None:
    """Offline check that authored YAML tasks match the committed manifest (FR-3, NFR-4).

    Asserts that every authored task in src/oasis/bench/authored/*.yaml matches
    the hashes, splits, and specifications recorded in config/benchmark_manifest.json.
    Any edit to an authored YAML task without rebuilding the manifest fails CI.
    """
    assert MANIFEST_PATH.exists(), f"Committed manifest missing: {MANIFEST_PATH}"

    manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = BenchmarkManifest.model_validate(manifest_data)

    authored_dir = REPO_ROOT / "src" / "oasis" / "bench" / "authored"
    cfg = load_benchmark_config(CONFIG_PATH)
    authored_records = load_authored_tasks(authored_dir, config=cfg)

    # Filter authored tasks in committed manifest
    manifest_authored = {t.task_id: t for t in manifest.tasks if t.source == "authored"}

    assert len(authored_records) == 40, "Expected 40 authored tasks across support_triage & content_generation"
    assert len(manifest_authored) == len(
        authored_records
    ), f"Manifest authored count ({len(manifest_authored)}) differs from authored loader ({len(authored_records)})"

    for r in authored_records:
        assert r.task_id in manifest_authored, f"Authored task '{r.task_id}' not found in committed manifest"
        m_entry = manifest_authored[r.task_id]

        expected_stmt_sha = compute_statement_sha256(r.statement)
        expected_spec_sha = compute_spec_sha256(
            r.reference_answer,
            r.verifier_type,
            r.verifier_spec,
            r.complexity_label,
        )

        assert (
            m_entry.statement_sha256 == expected_stmt_sha
        ), f"Authored task '{r.task_id}' statement SHA-256 differs from committed manifest"
        assert (
            m_entry.spec_sha256 == expected_spec_sha
        ), f"Authored task '{r.task_id}' spec SHA-256 differs from committed manifest"
        assert (
            m_entry.split == r.split
        ), f"Authored task '{r.task_id}' split ({r.split}) differs from manifest ({m_entry.split})"
        assert (
            m_entry.domain == r.domain
        ), f"Authored task '{r.task_id}' domain ({r.domain}) differs from manifest ({m_entry.domain})"


@pytest.mark.real_download
def test_real_download_regenerates_and_verifies_against_manifest(
    tmp_path: Path,
) -> None:
    """Regenerate benchmark from pinned revisions and verify against manifest (FR-3, NFR-4).

    Only executes when explicitly requested via 'pytest -m real_download'.
    """
    real_tasks_path = tmp_path / "real_tasks.jsonl"
    cfg = load_benchmark_config(CONFIG_PATH)

    records = ingest_benchmarks(
        use_fixtures=False,
        config=cfg,
        output_path=real_tasks_path,
    )
    assert len(records) == 100, f"Expected 100 tasks on real ingestion, got {len(records)}"

    diffs = verify_manifest(
        tasks_path=real_tasks_path,
        config_path=CONFIG_PATH,
        manifest_path=MANIFEST_PATH,
    )
    assert diffs == [], f"Verification against committed manifest failed after real download: {diffs}"
