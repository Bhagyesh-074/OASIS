"""Authored benchmark task loading, validation, and split assignment (FR-3, FR-13).

Provides the loader and validator for human-authored benchmark domains
('support_triage' and 'content_generation'). Enforces schema invariants,
checklist bounds, PII / contact pattern exclusions, and domain-count checks,
then assigns stratified calibration/eval splits using assign_splits (FR-3).

Authored domains have verifier_type and verifier_spec set to None (FR-13),
with output evaluation performed by an L2 judge against reference_answer checklists.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.ingest import DEFAULT_CREATED_AT, assign_splits
from oasis.bench.models import TaskRecord

# Regex patterns for detecting prohibited email addresses and telephone numbers
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b)"
)

REQUIRED_FIELDS = {"task_id", "statement", "reference_answer", "complexity_label"}
VALID_COMPLEXITY_LABELS = {"low", "medium", "high"}
VALID_DOMAINS = {
    "support_triage",
    "content_generation",
    "code_generation",
    "research_qa",
    "quant_analysis",
}


def _infer_domain(data: Any, file_path: Path, first_task_id: str | None = None) -> str:
    """Infer the benchmark domain from YAML contents or filename (FR-3)."""
    if isinstance(data, dict) and "domain" in data:
        domain = str(data["domain"]).strip()
        if domain in VALID_DOMAINS:
            return domain

    stem = file_path.stem.lower()
    if stem in VALID_DOMAINS:
        return stem

    if first_task_id:
        for known_domain in VALID_DOMAINS:
            if first_task_id.startswith(f"{known_domain}_"):
                return known_domain

    return stem


def load_authored_task_dicts(
    path: Path | str,
    config: BenchmarkConfig | None = None,
) -> list[dict[str, Any]]:
    """Load and validate authored task dictionaries from YAML file or directory (FR-3, FR-13).

    Validates:
    1. Rejects duplicate task IDs across the dataset.
    2. Rejects missing fields (task_id, statement, reference_answer, complexity_label).
    3. Rejects invalid complexity labels (must be 'low', 'medium', or 'high').
    4. Rejects empty or whitespace-only statements.
    5. Rejects reference answers with checklists outside 4-8 lines or not starting with '- '.
    6. Rejects text matching simple email or phone number patterns.
    7. Rejects domain task counts differing from config (FR-3).

    Parameters
    ----------
    path:
        Path to a YAML file or directory of YAML files containing authored tasks.
    config:
        Optional validated BenchmarkConfig instance.

    Returns
    -------
    list[dict[str, Any]]
        List of validated unassigned task dictionaries ready for split assignment.

    Raises
    ------
    FileNotFoundError:
        If path does not exist.
    ValueError:
        If any validation or schema invariant check fails.
    """
    cfg = config or load_benchmark_config()
    target_path = Path(path)

    if not target_path.exists():
        raise FileNotFoundError(f"Authored tasks path does not exist: {target_path}")

    if target_path.is_file():
        yaml_files = [target_path]
    elif target_path.is_dir():
        yaml_files = sorted(target_path.glob("*.yaml")) + sorted(target_path.glob("*.yml"))
        if not yaml_files:
            raise ValueError(f"No YAML task files found in directory: {target_path}")
    else:
        raise ValueError(f"Target path is neither file nor directory: {target_path}")

    seen_ids: set[str] = set()
    raw_tasks_by_domain: dict[str, list[dict[str, Any]]] = {}

    for file_path in yaml_files:
        raw_text = file_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw_text)

        if parsed is None:
            raise ValueError(f"Authored tasks file is empty: {file_path}")

        if isinstance(parsed, list):
            task_list = parsed
            first_tid = (
                task_list[0].get("task_id")
                if task_list and isinstance(task_list[0], dict)
                else None
            )
            inferred_domain = _infer_domain(
                parsed,
                file_path,
                first_task_id=first_tid,
            )
        elif isinstance(parsed, dict) and "tasks" in parsed:
            task_list = parsed["tasks"]
            first_tid = (
                task_list[0].get("task_id")
                if isinstance(task_list, list)
                and task_list
                and isinstance(task_list[0], dict)
                else None
            )
            inferred_domain = _infer_domain(
                parsed,
                file_path,
                first_task_id=first_tid,
            )
        else:
            raise ValueError(
                f"Invalid authored tasks structure in {file_path}: "
                f"expected list of tasks or dictionary with 'tasks' key."
            )

        if not isinstance(task_list, list) or len(task_list) == 0:
            raise ValueError(f"File {file_path} contains no task records.")

        for item in task_list:
            if not isinstance(item, dict):
                raise TypeError(f"Task record in {file_path} must be a dictionary: {item}")

            # Check for missing required fields
            for req_field in REQUIRED_FIELDS:
                if req_field not in item or item[req_field] is None:
                    tid_ref = item.get("task_id", "<unknown>")
                    raise ValueError(
                        f"Task '{tid_ref}' in {file_path} is missing required field: '{req_field}'"
                    )

            tid = str(item["task_id"]).strip()
            if not tid:
                raise ValueError(f"Task in {file_path} has an empty task_id")

            # Check for duplicate IDs
            if tid in seen_ids:
                raise ValueError(f"Duplicate task_id detected: '{tid}'")
            seen_ids.add(tid)

            # Validate statement
            statement = str(item["statement"]).strip()
            if not statement:
                raise ValueError(f"Task '{tid}' has an empty statement")

            # Validate complexity label
            complexity = str(item["complexity_label"]).strip().lower()
            if complexity not in VALID_COMPLEXITY_LABELS:
                raise ValueError(
                    f"Task '{tid}' has invalid complexity_label '{complexity}'; "
                    f"must be one of {sorted(VALID_COMPLEXITY_LABELS)}"
                )

            # Validate reference_answer checklist (4 to 8 lines, each starting with "- ")
            ref_ans = str(item["reference_answer"]).strip()
            checklist_lines = [
                line.strip() for line in ref_ans.splitlines() if line.strip()
            ]

            if len(checklist_lines) < 4 or len(checklist_lines) > 8:
                raise ValueError(
                    f"Task '{tid}' reference_answer checklist has {len(checklist_lines)} items; "
                    f"must be between 4 and 8 lines."
                )

            for raw_line in ref_ans.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith("- ") or not line[2:].strip():
                    raise ValueError(
                        f"Task '{tid}' reference_answer checklist item must start with '- ' "
                        f"and contain non-empty text: '{line}'"
                    )

            # Validate contact patterns (no emails or phone numbers)
            for field_name, text_val in [
                ("statement", statement),
                ("reference_answer", ref_ans),
            ]:
                email_match = EMAIL_PATTERN.search(text_val)
                if email_match:
                    raise ValueError(
                        f"Task '{tid}' {field_name} contains prohibited email pattern: "
                        f"'{email_match.group(0)}'"
                    )
                phone_match = PHONE_PATTERN.search(text_val)
                if phone_match:
                    raise ValueError(
                        f"Task '{tid}' {field_name} contains prohibited phone pattern: "
                        f"'{phone_match.group(0)}'"
                    )

            task_domain = item.get("domain", inferred_domain)
            if task_domain not in VALID_DOMAINS:
                raise ValueError(
                    f"Task '{tid}' has unknown or invalid domain: '{task_domain}'"
                )

            raw_tasks_by_domain.setdefault(task_domain, []).append(
                {
                    "task_id": tid,
                    "domain": task_domain,
                    "source": "authored",
                    "source_ref": None,
                    "statement": statement,
                    "reference_answer": ref_ans,
                    "verifier_type": None,
                    "verifier_spec": None,
                    "complexity_label": complexity,
                    "created_at": DEFAULT_CREATED_AT,
                }
            )

    # Check domain counts against config
    for domain, tasks in raw_tasks_by_domain.items():
        expected_count = cfg.domain_counts.get(domain)
        if expected_count is not None and len(tasks) != expected_count:
            raise ValueError(
                f"Domain '{domain}' task count ({len(tasks)}) differs from "
                f"configured domain_counts ({expected_count}) in benchmark.yaml (FR-3)"
            )

    all_task_dicts: list[dict[str, Any]] = []
    for domain in sorted(raw_tasks_by_domain.keys()):
        all_task_dicts.extend(raw_tasks_by_domain[domain])

    return all_task_dicts


def load_authored_tasks(
    path: Path | str,
    config: BenchmarkConfig | None = None,
    seed: int | None = None,
    calibration_fraction: float | None = None,
    existing_splits: Mapping[str, str] | None = None,
) -> list[TaskRecord]:
    """Load authored tasks and assign stratified calibration/eval splits (FR-3, FR-13).

    Reuses assign_splits to rank tasks within domain by sha256(f"{seed}:{task_id}"),
    assigning the first round(calibration_fraction * n) tasks to calibration (5 for n=20)
    and the remaining tasks to eval (15 for n=20).

    Authored tasks set verifier_type=None and verifier_spec=None (FR-13), evaluated
    via L2 judge against reference_answer checklists.

    Parameters
    ----------
    path:
        Path to authored YAML file or directory of YAML files.
    config:
        Optional validated BenchmarkConfig instance.
    seed:
        Optional RNG seed overriding config.seed.
    calibration_fraction:
        Optional fraction overriding config.calibration_fraction.
    existing_splits:
        Optional mapping of {task_id: split} preserving prior split assignments.

    Returns
    -------
    list[TaskRecord]
        List of validated TaskRecord instances with deterministic split assignments.
    """
    cfg = config or load_benchmark_config()
    seed_val = seed if seed is not None else cfg.seed
    calib_frac = (
        calibration_fraction
        if calibration_fraction is not None
        else cfg.calibration_fraction
    )

    task_dicts = load_authored_task_dicts(path=path, config=cfg)

    # Deterministic, stratified split assignment (FR-3)
    return assign_splits(
        task_dicts=task_dicts,
        calibration_fraction=calib_frac,
        seed=seed_val,
        existing_splits=existing_splits,
    )
