"""Invariant test for benchmark manifest determinism under PYTHONHASHSEED (FR-3, NFR-4).

Guards that benchmark manifest construction is 100% deterministic, byte-identical
across arbitrary PYTHONHASHSEED settings, LF-only without carriage returns,
and free of non-deterministic timestamps or un-ordered dictionaries.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_PATH = REPO_ROOT / "data" / "tasks.jsonl"
CONFIG_PATH = REPO_ROOT / "config" / "benchmark.yaml"


def test_manifest_build_determinism_across_pythonhashseeds(tmp_path: Path) -> None:
    """Verify byte-identical manifest output under PYTHONHASHSEED=1 vs PYTHONHASHSEED=2 (NFR-4)."""
    out1 = tmp_path / "manifest_seed1.json"
    out2 = tmp_path / "manifest_seed2.json"

    tasks_file = TASKS_PATH
    if not tasks_file.exists():
        tasks_file = tmp_path / "fixture_tasks.jsonl"
        fixtures_dir = REPO_ROOT / "tests" / "fixtures" / "bench"
        ingest_cmd = [
            sys.executable,
            "-m",
            "oasis.bench.ingest",
            "--all",
            "--use-fixtures",
            "--fixtures-dir",
            str(fixtures_dir),
            "--output",
            str(tasks_file),
        ]
        res = subprocess.run(
            ingest_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
        )
        assert res.returncode == 0, f"Fixture ingest failed:\n{res.stderr}"

    cmd = [
        sys.executable,
        "-m",
        "oasis.bench.manifest",
        "build",
        "--tasks",
        str(tasks_file),
        "--config",
        str(CONFIG_PATH),
    ]

    env1 = {**os.environ, "PYTHONHASHSEED": "1", "PYTHONPATH": str(REPO_ROOT / "src")}
    env2 = {**os.environ, "PYTHONHASHSEED": "2", "PYTHONPATH": str(REPO_ROOT / "src")}

    res1 = subprocess.run(
        [*cmd, "--output", str(out1)],
        cwd=str(REPO_ROOT),
        env=env1,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res1.returncode == 0, f"Manifest build with PYTHONHASHSEED=1 failed:\n{res1.stderr}"

    res2 = subprocess.run(
        [*cmd, "--output", str(out2)],
        cwd=str(REPO_ROOT),
        env=env2,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0, f"Manifest build with PYTHONHASHSEED=2 failed:\n{res2.stderr}"

    assert out1.exists(), "Manifest file for PYTHONHASHSEED=1 not created"
    assert out2.exists(), "Manifest file for PYTHONHASHSEED=2 not created"

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()

    assert len(bytes1) > 0, "Manifest output 1 is empty"
    assert len(bytes2) > 0, "Manifest output 2 is empty"

    assert b"\r" not in bytes1, (
        r"Carriage return \r detected in manifest output 1 (must use LF \n only) (NFR-4)"
    )
    assert b"\r" not in bytes2, (
        r"Carriage return \r detected in manifest output 2 (must use LF \n only) (NFR-4)"
    )

    assert bytes1 == bytes2, (
        f"Byte-identical manifest invariant violated under differing PYTHONHASHSEED!\n"
        f"Seed 1 length: {len(bytes1)}, Seed 2 length: {len(bytes2)}"
    )

    # Invariant: Verify structure, sorted task IDs, and absence of timestamps
    data = json.loads(bytes1.decode("utf-8"))
    assert data["manifest_version"] == "1.0.0"
    assert len(data["tasks"]) == 100 if tasks_file == TASKS_PATH else len(data["tasks"]) > 0

    task_ids = [t["task_id"] for t in data["tasks"]]
    assert task_ids == sorted(task_ids), "Manifest tasks must be strictly sorted by task_id"

    # Ensure no timestamp or creation metadata in manifest JSON
    text_content = bytes1.decode("utf-8")
    for forbidden_key in ["\"timestamp\"", "\"created_at\""]:
        assert forbidden_key not in text_content, (
            f"Non-deterministic field {forbidden_key} found in benchmark manifest (NFR-4)"
        )
