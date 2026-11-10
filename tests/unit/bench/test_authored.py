"""Unit tests for authored benchmark tasks and loader (FR-3, FR-13).

Guards:
1. Valid loading of support_triage.yaml and content_generation.yaml.
2. Rejection of duplicate IDs, missing fields, invalid complexity labels,
   checklists outside 4-8 lines or not starting with '- ', empty statements,
   prohibited email/phone patterns, and domain task count mismatch against config (FR-3).
3. Complexity label balance: exactly 7 low, 7 medium, 6 high per domain.
4. Stratified split counts: exactly 5 calibration and 15 eval per domain (FR-3).
5. Invariant: Null verifier_type and verifier_spec for authored tasks (FR-13).
6. File format invariants: no carriage returns (\\r) and presence of DRAFT review comment.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from oasis.bench.authored import load_authored_task_dicts, load_authored_tasks
from oasis.bench.config import BenchmarkConfig
from oasis.bench.models import CONTEXT_MARKER, TaskRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORED_DIR = REPO_ROOT / "src" / "oasis" / "bench" / "authored"
SUPPORT_TRIAGE_PATH = AUTHORED_DIR / "support_triage.yaml"
CONTENT_GEN_PATH = AUTHORED_DIR / "content_generation.yaml"


# ---------------------------------------------------------------------------
# 1. Valid File Ingestion & Model Attributes (FR-3, FR-13)
# ---------------------------------------------------------------------------
def test_valid_authored_support_triage_loads() -> None:
    """support_triage.yaml loads 20 valid TaskRecord instances with null verifiers (FR-13)."""
    records = load_authored_tasks(SUPPORT_TRIAGE_PATH)
    assert len(records) == 20
    for r in records:
        assert isinstance(r, TaskRecord)
        assert r.domain == "support_triage"
        assert r.source == "authored"
        assert r.source_ref is None
        assert r.verifier_type is None
        assert r.verifier_spec is None
        assert r.complexity_label in {"low", "medium", "high"}
        assert r.split in {"calibration", "eval"}
        assert r.task_id.startswith("support_triage_")


def test_valid_authored_content_generation_loads() -> None:
    """content_generation.yaml loads 20 valid TaskRecord instances with null verifiers (FR-13)."""
    records = load_authored_tasks(CONTENT_GEN_PATH)
    assert len(records) == 20
    for r in records:
        assert isinstance(r, TaskRecord)
        assert r.domain == "content_generation"
        assert r.source == "authored"
        assert r.source_ref is None
        assert r.verifier_type is None
        assert r.verifier_spec is None
        assert r.complexity_label in {"low", "medium", "high"}
        assert r.split in {"calibration", "eval"}
        assert r.task_id.startswith("content_generation_")


def test_valid_authored_directory_loads() -> None:
    """Loading the authored directory loads all 40 tasks across both domains (FR-3)."""
    records = load_authored_tasks(AUTHORED_DIR)
    assert len(records) == 40
    domains = {r.domain for r in records}
    assert domains == {"support_triage", "content_generation"}


def test_load_authored_task_dicts_returns_unassigned_dicts() -> None:
    """load_authored_task_dicts loads raw unassigned task property dictionaries (FR-3, FR-13)."""
    task_dicts = load_authored_task_dicts(SUPPORT_TRIAGE_PATH)
    assert len(task_dicts) == 20
    for d in task_dicts:
        assert isinstance(d, dict)
        assert "split" not in d
        assert d["domain"] == "support_triage"
        assert d["source"] == "authored"
        assert d["verifier_type"] is None
        assert d["verifier_spec"] is None


# ---------------------------------------------------------------------------
# 2. Complexity Label Balance: 7 Low / 7 Medium / 6 High
# ---------------------------------------------------------------------------
def test_complexity_label_balance_support_triage() -> None:
    """support_triage complexity labels are balanced: 7 low, 7 medium, 6 high."""
    records = load_authored_tasks(SUPPORT_TRIAGE_PATH)
    counts = Counter(r.complexity_label for r in records)
    assert counts["low"] == 7
    assert counts["medium"] == 7
    assert counts["high"] == 6


def test_complexity_label_balance_content_generation() -> None:
    """content_generation complexity labels are balanced: 7 low, 7 medium, 6 high."""
    records = load_authored_tasks(CONTENT_GEN_PATH)
    counts = Counter(r.complexity_label for r in records)
    assert counts["low"] == 7
    assert counts["medium"] == 7
    assert counts["high"] == 6


# ---------------------------------------------------------------------------
# 3. Stratified Split Counts: 5 Calibration / 15 Eval (FR-3)
# ---------------------------------------------------------------------------
def test_split_counts_support_triage() -> None:
    """support_triage receives exactly 5 calibration and 15 eval tasks (FR-3)."""
    records = load_authored_tasks(SUPPORT_TRIAGE_PATH)
    calib = [r for r in records if r.split == "calibration"]
    evaluation = [r for r in records if r.split == "eval"]
    assert len(calib) == 5
    assert len(evaluation) == 15


def test_split_counts_content_generation() -> None:
    """content_generation receives exactly 5 calibration and 15 eval tasks (FR-3)."""
    records = load_authored_tasks(CONTENT_GEN_PATH)
    calib = [r for r in records if r.split == "calibration"]
    evaluation = [r for r in records if r.split == "eval"]
    assert len(calib) == 5
    assert len(evaluation) == 15


# ---------------------------------------------------------------------------
# 4. Rejection Cases (Loader Validation Invariants)
# ---------------------------------------------------------------------------
def _build_valid_task_list(n: int = 20, domain: str = "support_triage") -> list[dict]:
    """Helper to generate a valid list of n task dictionaries."""
    tasks = []
    for i in range(1, n + 1):
        tasks.append(
            {
                "task_id": f"{domain}_{i:03d}",
                "statement": f"Perform support triage operation {i}.",
                "reference_answer": (
                    "- Step one requirement\n"
                    "- Step two requirement\n"
                    "- Step three requirement\n"
                    "- Step four requirement"
                ),
                "complexity_label": "low" if i <= 7 else ("medium" if i <= 14 else "high"),
            }
        )
    return tasks


def test_reject_non_dict_task_item(tmp_path: Path) -> None:
    """Loader raises TypeError if a task item in the YAML list is not a dictionary."""
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(["a string item", "another"]), encoding="utf-8")

    with pytest.raises(TypeError, match="must be a dictionary"):
        load_authored_tasks(yaml_file)


def test_reject_duplicate_task_ids(tmp_path: Path) -> None:
    """Loader rejects duplicate task IDs within domain."""
    tasks = _build_valid_task_list(20)
    tasks[1]["task_id"] = tasks[0]["task_id"]
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate task_id detected"):
        load_authored_tasks(yaml_file)


@pytest.mark.parametrize(
    "missing_field",
    ["task_id", "statement", "reference_answer", "complexity_label"],
)
def test_reject_missing_required_field(tmp_path: Path, missing_field: str) -> None:
    """Loader rejects tasks missing any mandatory field."""
    tasks = _build_valid_task_list(20)
    del tasks[0][missing_field]
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match=f"missing required field: '{missing_field}'"):
        load_authored_tasks(yaml_file)


def test_reject_invalid_complexity_label(tmp_path: Path) -> None:
    """Loader rejects unrecognized complexity labels."""
    tasks = _build_valid_task_list(20)
    tasks[0]["complexity_label"] = "extreme"
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid complexity_label 'extreme'"):
        load_authored_tasks(yaml_file)


@pytest.mark.parametrize("empty_stmt", ["", "   ", "\n\t"])
def test_reject_empty_statement(tmp_path: Path, empty_stmt: str) -> None:
    """Loader rejects empty or whitespace-only problem statements."""
    tasks = _build_valid_task_list(20)
    tasks[0]["statement"] = empty_stmt
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="has an empty statement"):
        load_authored_tasks(yaml_file)


def test_reject_checklist_too_short(tmp_path: Path) -> None:
    """Loader rejects reference answer checklist with fewer than 4 items."""
    tasks = _build_valid_task_list(20)
    tasks[0]["reference_answer"] = "- Item one\n- Item two\n- Item three"
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="checklist has 3 items; must be between 4 and 8 lines"):
        load_authored_tasks(yaml_file)


def test_reject_checklist_too_long(tmp_path: Path) -> None:
    """Loader rejects reference answer checklist with more than 8 items."""
    tasks = _build_valid_task_list(20)
    tasks[0]["reference_answer"] = "\n".join(f"- Item {i}" for i in range(1, 10))
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="checklist has 9 items; must be between 4 and 8 lines"):
        load_authored_tasks(yaml_file)


def test_reject_checklist_item_missing_dash_prefix(tmp_path: Path) -> None:
    """Loader rejects checklist items that do not begin with '- '."""
    tasks = _build_valid_task_list(20)
    tasks[0]["reference_answer"] = "- Item one\n* Item two with asterisk\n- Item three\n- Item four"
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="must start with '- '"):
        load_authored_tasks(yaml_file)


def test_reject_checklist_item_empty_body(tmp_path: Path) -> None:
    """Loader rejects checklist items with empty body after '- '."""
    tasks = _build_valid_task_list(20)
    tasks[0]["reference_answer"] = "- Item one\n- \n- Item three\n- Item four"
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="must start with '- ' and contain non-empty text"):
        load_authored_tasks(yaml_file)


@pytest.mark.parametrize(
    "prohibited_email",
    [
        "alice@example.com",
        "support@company.org",
        "first.last+tag@sub.domain.co",
    ],
)
def test_reject_email_pattern_in_statement(tmp_path: Path, prohibited_email: str) -> None:
    """Loader rejects text matching email patterns in statement."""
    tasks = _build_valid_task_list(20)
    tasks[0]["statement"] = f"Contact our agent at {prohibited_email} for assistance."
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="contains prohibited email pattern"):
        load_authored_tasks(yaml_file)


def test_reject_email_pattern_in_reference_answer(tmp_path: Path) -> None:
    """Loader rejects text matching email patterns in reference answer."""
    tasks = _build_valid_task_list(20)
    tasks[0]["reference_answer"] = (
        "- Verify customer details\n"
        "- Send confirmation to test@sample.net\n"
        "- Close incident ticket\n"
        "- Log audit entry"
    )
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="contains prohibited email pattern"):
        load_authored_tasks(yaml_file)


@pytest.mark.parametrize(
    "prohibited_phone",
    [
        "555-123-4567",
        "+1-800-555-0199",
        "(555) 012-3456",
        "5550123456",
    ],
)
def test_reject_phone_pattern_in_statement(tmp_path: Path, prohibited_phone: str) -> None:
    """Loader rejects text matching telephone patterns in statement."""
    tasks = _build_valid_task_list(20)
    tasks[0]["statement"] = f"Please dial {prohibited_phone} to verify identity."
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="contains prohibited phone pattern"):
        load_authored_tasks(yaml_file)


def test_reject_domain_count_mismatch(tmp_path: Path) -> None:
    """Loader rejects task file whose count differs from configured domain_counts (FR-3)."""
    # Create 19 tasks when config expects 20
    tasks = _build_valid_task_list(19)
    yaml_file = tmp_path / "support_triage.yaml"
    yaml_file.write_text(yaml.dump(tasks), encoding="utf-8")

    config = BenchmarkConfig(domain_counts={"support_triage": 20})
    with pytest.raises(ValueError, match="differs from configured domain_counts"):
        load_authored_tasks(yaml_file, config=config)


# ---------------------------------------------------------------------------
# 5. File Formatting and Metadata Invariants
# ---------------------------------------------------------------------------
def test_no_carriage_returns_in_authored_yaml_files() -> None:
    """Authored YAML files must contain LF (\\n) line endings, never CRLF (\\r\\n)."""
    for yaml_path in [SUPPORT_TRIAGE_PATH, CONTENT_GEN_PATH]:
        content_bytes = yaml_path.read_bytes()
        assert b"\r" not in content_bytes, f"Carriage return \\r detected in {yaml_path}"


def test_draft_review_comment_present_in_authored_yaml_files() -> None:
    """Both YAML files must carry the review DRAFT comment at the top."""
    expected_header = "# DRAFT: I will review and edit all 40 myself before they count as final."
    for yaml_path in [SUPPORT_TRIAGE_PATH, CONTENT_GEN_PATH]:
        text = yaml_path.read_text(encoding="utf-8")
        assert expected_header in text, f"Missing DRAFT comment header in {yaml_path}"


def test_context_marker_in_source_material_tasks() -> None:
    """Tasks providing source material properly incorporate CONTEXT_MARKER."""
    records = load_authored_tasks(SUPPORT_TRIAGE_PATH) + load_authored_tasks(CONTENT_GEN_PATH)
    tasks_with_context = [r for r in records if CONTEXT_MARKER in r.statement]
    # In both domains, the vast majority of realistic triage and generation tasks carry context
    assert len(tasks_with_context) >= 30, (
        f"Expected at least 30 tasks with '{CONTEXT_MARKER}', found {len(tasks_with_context)}"
    )
