"""Invariant test for hash seed determinism under PYTHONHASHSEED (FR-13, FR-3, NFR-4).

Guards that task sampling and split assignment are 100% deterministic,
free of carriage returns (byte-identical across OSes), and strictly
independent of Python's randomized hash seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "bench"


def test_hash_seed_determinism_across_pythonhashseeds(tmp_path: Path) -> None:
    """Verify byte-identical ingestion output under PYTHONHASHSEED=1 vs PYTHONHASHSEED=2 (FR-3, FR-13, NFR-4)."""
    out1 = tmp_path / "tasks_seed1.jsonl"
    out2 = tmp_path / "tasks_seed2.jsonl"

    cmd = [
        sys.executable,
        "-m",
        "oasis.bench.ingest",
        "--all",
        "--use-fixtures",
        "--fixtures-dir",
        str(FIXTURES_DIR),
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
    assert res1.returncode == 0, f"Run with PYTHONHASHSEED=1 failed:\n{res1.stderr}"

    res2 = subprocess.run(
        [*cmd, "--output", str(out2)],
        cwd=str(REPO_ROOT),
        env=env2,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0, f"Run with PYTHONHASHSEED=2 failed:\n{res2.stderr}"

    assert out1.exists(), "Output file for PYTHONHASHSEED=1 not created"
    assert out2.exists(), "Output file for PYTHONHASHSEED=2 not created"

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()

    assert len(bytes1) > 0, "Output file 1 is empty"
    assert len(bytes2) > 0, "Output file 2 is empty"

    assert b"\r" not in bytes1, (
        r"Carriage return \r detected in task JSONL output for seed 1 (must use LF \n only) (FR-3, NFR-4)"
    )
    assert b"\r" not in bytes2, (
        r"Carriage return \r detected in task JSONL output for seed 2 (must use LF \n only) (FR-3, NFR-4)"
    )

    assert bytes1 == bytes2, (
        f"Byte-identical output invariant violated under differing PYTHONHASHSEED!\n"
        f"Seed 1 length: {len(bytes1)}, Seed 2 length: {len(bytes2)}"
    )

    # Verify presence and split distribution of authored domains in the ingested output
    lines = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines() if line.strip()]
    domain_counts = Counter(r["domain"] for r in lines)
    assert "support_triage" in domain_counts, "support_triage missing from --all output"
    assert "content_generation" in domain_counts, "content_generation missing from --all output"
    assert domain_counts["support_triage"] == 20
    assert domain_counts["content_generation"] == 20

    for domain in ["support_triage", "content_generation"]:
        domain_tasks = [r for r in lines if r["domain"] == domain]
        calib_tasks = [r for r in domain_tasks if r["split"] == "calibration"]
        eval_tasks = [r for r in domain_tasks if r["split"] == "eval"]
        assert len(calib_tasks) == 5, f"{domain} must have exactly 5 calibration tasks (FR-3)"
        assert len(eval_tasks) == 15, f"{domain} must have exactly 15 eval tasks (FR-3)"


def test_authored_loader_determinism_across_pythonhashseeds(tmp_path: Path) -> None:
    """Verify load_authored_tasks output is byte-identical under PYTHONHASHSEED=1 vs 2 (FR-3, FR-13)."""
    out1 = tmp_path / "authored_seed1.jsonl"
    out2 = tmp_path / "authored_seed2.jsonl"

    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from oasis.bench.authored import load_authored_tasks\n"
        "out = Path(sys.argv[1])\n"
        "records = load_authored_tasks('src/oasis/bench/authored')\n"
        "with out.open('w', encoding='utf-8', newline='\\n') as f:\n"
        "    for r in records:\n"
        "        f.write(r.model_dump_json() + '\\n')\n"
    )

    env1 = {**os.environ, "PYTHONHASHSEED": "1", "PYTHONPATH": str(REPO_ROOT / "src")}
    env2 = {**os.environ, "PYTHONHASHSEED": "2", "PYTHONPATH": str(REPO_ROOT / "src")}

    res1 = subprocess.run(
        [sys.executable, "-c", script, str(out1)],
        cwd=str(REPO_ROOT),
        env=env1,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res1.returncode == 0, f"load_authored_tasks with PYTHONHASHSEED=1 failed:\n{res1.stderr}"

    res2 = subprocess.run(
        [sys.executable, "-c", script, str(out2)],
        cwd=str(REPO_ROOT),
        env=env2,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0, f"load_authored_tasks with PYTHONHASHSEED=2 failed:\n{res2.stderr}"

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()

    assert len(bytes1) > 0, "Authored output file 1 is empty"
    assert len(bytes2) > 0, "Authored output file 2 is empty"

    assert b"\r" not in bytes1, r"Carriage return \r detected in authored output 1 (FR-3, NFR-4)"
    assert b"\r" not in bytes2, r"Carriage return \r detected in authored output 2 (FR-3, NFR-4)"

    assert bytes1 == bytes2, "Authored loader output differs under varying PYTHONHASHSEED"
