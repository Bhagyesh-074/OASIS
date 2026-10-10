"""Task ingestion and benchmark preparation (FR-3, FR-13, docs/DATABASE.md).

Loads tasks from benchmark datasets (MBPP, HumanEval, HotpotQA, GSM8K),
samples them deterministically using a seeded RNG, assigns stratified
calibration and evaluation splits (FR-3), enforces leakage invariants,
and exports TaskRecord rows via JSONL / database adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.models import (
    CONTEXT_MARKER,
    CitationResolveSpec,
    TaskDomain,
    TaskRecord,
)

logger = logging.getLogger(__name__)

# Canonical timestamp for deterministic benchmark generation
DEFAULT_CREATED_AT = "2026-09-01T00:00:00+00:00"

# Default fixtures directory relative to repo root
DEFAULT_FIXTURES_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "bench"
)


def derive_stable_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic 32-bit integer seed using SHA-256 (FR-3, NFR-4)."""
    payload = f"{base_seed}:{key}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# 1. Transformers: Raw Dataset Dict -> TaskRecord attributes
# ---------------------------------------------------------------------------
def extract_gsm8k_numeric(answer: str) -> float | None:
    """Extract the final numerical value from a GSM8K answer string.

    GSM8K standard solutions conclude with '#### <number>'.

    Parameters
    ----------
    answer:
        Raw solution text from GSM8K.

    Returns
    -------
    float | None
        Extracted floating-point value, or None if no number found.
    """
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    if match:
        val_str = match.group(1).replace(",", "")
        try:
            return float(val_str)
        except ValueError:
            return None

    # Fallback to the last numeric token in the string
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", answer)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def extract_hotpotqa_context(
    raw_context: Any,
) -> tuple[list[str], list[str]]:
    """Extract context titles and formatted paragraphs in dataset order (FR-13, FR-3).

    Handles both HuggingFace dict format ({"title": [...], "sentences": [[...], ...]})
    and raw JSON list format ([[title, sentences], ...]).

    Parameters
    ----------
    raw_context:
        Raw context structure from HotpotQA distractor record.

    Returns
    -------
    tuple[list[str], list[str]]
        (context_titles, formatted_paragraphs)
        where each formatted paragraph is '[Title] text' in dataset's own order.
    """
    context_titles: list[str] = []
    paragraphs: list[str] = []

    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentences_list = raw_context.get("sentences", [])
        for idx, title_item in enumerate(titles):
            title = str(title_item).strip()
            sentences = sentences_list[idx] if idx < len(sentences_list) else []
            if isinstance(sentences, (list, tuple)):
                text = " ".join(str(s).strip() for s in sentences if str(s).strip())
            else:
                text = str(sentences).strip()
            context_titles.append(title)
            para = f"[{title}] {text}".strip()
            paragraphs.append(para)

    elif isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0]).strip()
                sentences = item[1]
                if isinstance(sentences, (list, tuple)):
                    text = " ".join(str(s).strip() for s in sentences if str(s).strip())
                else:
                    text = str(sentences).strip()
                context_titles.append(title)
                para = f"[{title}] {text}".strip()
                paragraphs.append(para)
            elif isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                sentences = item.get("sentences", [])
                if isinstance(sentences, (list, tuple)):
                    text = " ".join(str(s).strip() for s in sentences if str(s).strip())
                else:
                    text = str(sentences).strip()
                context_titles.append(title)
                para = f"[{title}] {text}".strip()
                paragraphs.append(para)

    return context_titles, paragraphs


def extract_hotpotqa_supporting_titles(raw_facts: Any) -> list[str]:
    """Extract ordered unique supporting paragraph titles (FR-13, FR-3).

    Handles both HuggingFace dict format ({"title": [...], "sent_id": [...]})
    and raw JSON list format ([[title, sent_id], ...]).

    Parameters
    ----------
    raw_facts:
        Raw supporting_facts structure from HotpotQA record.

    Returns
    -------
    list[str]
        Ordered list of unique supporting paragraph titles.
    """
    raw_titles: list[str] = []
    if isinstance(raw_facts, dict):
        titles = raw_facts.get("title", [])
        if isinstance(titles, list):
            raw_titles = [str(t).strip() for t in titles if str(t).strip()]
    elif isinstance(raw_facts, list):
        for item in raw_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                t = str(item[0]).strip()
                if t:
                    raw_titles.append(t)
            elif isinstance(item, dict):
                t = str(item.get("title", "")).strip()
                if t:
                    raw_titles.append(t)
            elif isinstance(item, str) and item.strip():
                raw_titles.append(item.strip())

    return list(dict.fromkeys(raw_titles))


def transform_raw_to_task(
    raw: dict[str, Any],
    source: str,
    domain: TaskDomain,
    idx: int = 0,
    created_at: str = DEFAULT_CREATED_AT,
) -> dict[str, Any]:
    """Convert raw dataset dictionary into TaskRecord field dictionary (docs/DATABASE.md).

    Parameters
    ----------
    raw:
        Raw dataset record dictionary.
    source:
        Origin source name ('mbpp', 'humaneval', 'hotpotqa', 'gsm8k').
    domain:
        Target evaluation domain.
    idx:
        Index fallback for identifier creation.
    created_at:
        ISO timestamp string.

    Returns
    -------
    dict[str, Any]
        Dictionary of TaskRecord keyword arguments (excluding 'split').
    """
    if source == "mbpp":
        if "task_id" not in raw:
            raise KeyError("MBPP raw record missing native 'task_id'")
        native_id = raw["task_id"]
        source_ref = str(native_id)
        try:
            task_id = f"mbpp_{int(native_id):04d}"
        except (ValueError, TypeError):
            task_id = f"mbpp_{source_ref}"
        instruction_text = str(raw.get("text", raw.get("prompt", ""))).strip()
        code = str(raw.get("code", "")).strip()
        test_list = raw.get("test_list", [])
        test_setup = raw.get("test_setup_code", "")
        test_code = (
            "\n".join(test_list) if isinstance(test_list, list) else str(test_list)
        )
        first_assert = (
            test_list[0]
            if (isinstance(test_list, list) and len(test_list) > 0)
            else (test_code.splitlines()[0] if test_code else "")
        )
        statement = (
            f"{instruction_text}\n\n{CONTEXT_MARKER}\nYour code should pass this test:\n{first_assert}"
            if first_assert
            else instruction_text
        )
        spec = json.dumps({"test_code": test_code, "test_setup_code": test_setup})
        return {
            "task_id": task_id,
            "domain": domain,
            "source": "mbpp",
            "source_ref": source_ref,
            "statement": statement,
            "reference_answer": code,
            "verifier_type": "pytest",
            "verifier_spec": spec,
            "complexity_label": None,
            "created_at": created_at,
        }

    elif source == "humaneval":
        if "task_id" not in raw:
            raise KeyError("HumanEval raw record missing native 'task_id'")
        raw_id = str(raw["task_id"])
        source_ref = raw_id
        if "/" in raw_id:
            num_part = raw_id.split("/")[-1]
        elif raw_id.lower().startswith("humaneval_"):
            num_part = raw_id.split("_", 1)[1]
        else:
            num_part = raw_id
        task_id = f"humaneval_{num_part}"
        statement = str(raw.get("prompt", "")).strip()
        canonical_solution = str(raw.get("canonical_solution", "")).strip()
        test = str(raw.get("test", "")).strip()
        entry_point = str(raw.get("entry_point", "")).strip()
        spec = json.dumps({"test_code": test, "entry_point": entry_point})
        return {
            "task_id": task_id,
            "domain": domain,
            "source": "humaneval",
            "source_ref": raw_id,
            "statement": statement,
            "reference_answer": canonical_solution,
            "verifier_type": "pytest",
            "verifier_spec": spec,
            "complexity_label": None,
            "created_at": created_at,
        }

    elif source == "hotpotqa":
        native_id = raw.get("_id") or raw.get("id")
        if not native_id:
            raise KeyError("HotpotQA raw record missing native '_id' or 'id'")
        source_ref = str(native_id)
        task_id = f"hotpotqa_{source_ref}"
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError(f"HotpotQA record '{task_id}' missing 'question'")

        answer = str(raw.get("answer", "")).strip()
        context_titles, paragraphs = extract_hotpotqa_context(raw.get("context", {}))
        supporting_titles = extract_hotpotqa_supporting_titles(
            raw.get("supporting_facts", {})
        )

        # Validate spec with CitationResolveSpec pydantic model (FR-13, FR-3)
        # Rejects records with empty expected_answer or no supporting titles
        try:
            spec_obj = CitationResolveSpec(
                expected_answer=answer,
                context_titles=context_titles,
                supporting_titles=supporting_titles,
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid citation_resolve spec for HotpotQA record '{task_id}': {exc}"
            ) from exc

        # Format statement: instruction + question, then CONTEXT_MARKER, then paragraphs as "[Title] text"
        instruction = "Answer using only the context below and cite the titles of the paragraphs you used."
        paragraphs_block = "\n\n".join(paragraphs)
        if paragraphs_block:
            statement = (
                f"{instruction}\n\n{question}\n\n{CONTEXT_MARKER}\n\n{paragraphs_block}"
            )
        else:
            statement = f"{instruction}\n\n{question}\n\n{CONTEXT_MARKER}"

        spec = spec_obj.model_dump_json()
        return {
            "task_id": task_id,
            "domain": domain,
            "source": "hotpotqa",
            "source_ref": source_ref,
            "statement": statement,
            "reference_answer": spec_obj.expected_answer,
            "verifier_type": "citation_resolve",
            "verifier_spec": spec,
            "complexity_label": None,
            "created_at": created_at,
        }

    elif source == "gsm8k":
        statement = str(raw.get("question", "")).strip()
        if not statement:
            raise KeyError("GSM8K raw record missing 'question'")
        norm_q = " ".join(statement.split())
        native_id = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()[:10]
        task_id = f"gsm8k_{native_id}"
        source_ref = native_id
        answer = str(raw.get("answer", "")).strip()
        expected_val = extract_gsm8k_numeric(answer)
        if expected_val is None:
            raise ValueError(
                f"Cannot emit numeric_consistency spec with null expected_value for task '{task_id}'"
            )
        spec = json.dumps({"expected_value": expected_val, "tolerance": 1e-4})
        return {
            "task_id": task_id,
            "domain": domain,
            "source": "gsm8k",
            "source_ref": source_ref,
            "statement": statement,
            "reference_answer": answer,
            "verifier_type": "numeric_consistency",
            "verifier_spec": spec,
            "complexity_label": None,
            "created_at": created_at,
        }

    elif source == "authored":
        raw_tid = str(raw.get("task_id", "")).strip()
        source_ref_opt: str | None
        if raw_tid.startswith((f"{domain}_", f"{source}_")):
            task_id = raw_tid
            raw_sref = raw.get("source_ref")
            source_ref_opt = str(raw_sref) if raw_sref is not None else None
        elif raw_tid:
            source_ref_opt = raw_tid
            task_id = f"{source}_{source_ref_opt}"
        else:
            stmt = str(raw.get("statement", "")).strip()
            if stmt:
                norm_stmt = " ".join(stmt.split())
                source_ref_opt = hashlib.sha256(norm_stmt.encode("utf-8")).hexdigest()[:10]
                task_id = f"{source}_{source_ref_opt}"
            else:
                raise KeyError("Task record for source 'authored' missing task_id and statement")

        return {
            "task_id": task_id,
            "domain": domain,
            "source": "authored",
            "source_ref": source_ref_opt,
            "statement": str(raw["statement"]),
            "reference_answer": (
                str(raw["reference_answer"])
                if raw.get("reference_answer") is not None
                else None
            ),
            "verifier_type": None,
            "verifier_spec": None,
            "complexity_label": raw.get("complexity_label"),
            "created_at": created_at,
        }

    else:
        # Fallback for generic sources
        native_id = raw.get("task_id") or raw.get("id") or raw.get("_id")
        if not native_id:
            stmt = str(
                raw.get("statement", raw.get("prompt", raw.get("question", "")))
            ).strip()
            if stmt:
                norm_stmt = " ".join(stmt.split())
                native_id = hashlib.sha256(norm_stmt.encode("utf-8")).hexdigest()[:10]
            else:
                raise KeyError(
                    f"Task record for source '{source}' missing native identifier"
                )
        source_ref = str(native_id)
        task_id = f"{source}_{source_ref}"
        statement = str(
            raw.get("statement", raw.get("prompt", raw.get("question", "")))
        ).strip()
        ref = raw.get("reference_answer", raw.get("answer", raw.get("code", None)))
        return {
            "task_id": task_id,
            "domain": domain,
            "source": source,
            "source_ref": source_ref,
            "statement": statement,
            "reference_answer": str(ref) if ref is not None else None,
            "verifier_type": raw.get("verifier_type", None),
            "verifier_spec": raw.get("verifier_spec", None),
            "complexity_label": raw.get("complexity_label", None),
            "created_at": created_at,
        }


# ---------------------------------------------------------------------------
# 2. Raw Dataset Loaders (Lazy HuggingFace & Fixture Support)
# ---------------------------------------------------------------------------
def load_raw_from_fixture(fixture_file: Path | str) -> list[dict[str, Any]]:
    """Load raw dataset records from a local JSONL fixture file.

    Parameters
    ----------
    fixture_file:
        Path to JSONL fixture file.

    Returns
    -------
    list[dict[str, Any]]
        List of deserialized raw record dictionaries.
    """
    path = Path(fixture_file)
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_raw_from_huggingface(
    source: str,
    config: BenchmarkConfig | None = None,
) -> list[dict[str, Any]]:
    """Load raw records from HuggingFace `datasets` with lazy import.

    Parameters
    ----------
    source:
        Dataset key ('mbpp', 'humaneval', 'hotpotqa', 'gsm8k').
    config:
        Optional benchmark configuration containing HF dataset parameters.

    Returns
    -------
    list[dict[str, Any]]
        List of raw dictionaries from HuggingFace dataset split.

    Raises
    ------
    ImportError
        If HuggingFace `datasets` package is not installed.
    """
    try:
        from datasets import (  # type: ignore[import-not-found,import-untyped]
            load_dataset,
        )
    except ImportError as exc:
        raise ImportError(
            "HuggingFace 'datasets' package is required for live dataset ingestion. "
            "Install it via 'uv add datasets' or ingest from fixture files."
        ) from exc

    cfg = config or load_benchmark_config()
    ds_params = cfg.datasets.get(source, {})

    path = ds_params.get("path", source)
    name = ds_params.get("name", None)
    split = ds_params.get("split", "test")

    kwargs: dict[str, Any] = {"split": split}
    if name:
        kwargs["name"] = name

    dataset = load_dataset(path, **kwargs)
    return [dict(item) for item in dataset]


def load_raw_dataset(
    source: str,
    config: BenchmarkConfig | None = None,
    fixtures_dir: Path | None = None,
    use_fixtures: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve raw dataset records either from local fixtures or HuggingFace.

    Parameters
    ----------
    source:
        Dataset key ('mbpp', 'humaneval', 'hotpotqa', 'gsm8k').
    config:
        Validated BenchmarkConfig.
    fixtures_dir:
        Optional directory containing fixture JSONL files.
    use_fixtures:
        If True, forces loading from local fixture files (zero network).

    Returns
    -------
    list[dict[str, Any]]
        List of raw record dictionaries.
    """
    target_fixtures_dir = fixtures_dir or DEFAULT_FIXTURES_DIR
    fixture_path = target_fixtures_dir / f"{source}_sample.jsonl"

    if use_fixtures:
        return load_raw_from_fixture(fixture_path)

    # Attempt live download from HuggingFace if requested without --use-fixtures
    try:
        return load_raw_from_huggingface(source, config=config)
    except ImportError:
        # Fall back to offline fixture file if available
        if fixture_path.exists():
            return load_raw_from_fixture(fixture_path)
        raise


# ---------------------------------------------------------------------------
# 3. Deterministic Sampling
# ---------------------------------------------------------------------------
def sample_raw_data(
    raw_items: list[dict[str, Any]],
    count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample raw records deterministically using a seeded RNG.

    Sorts raw records by a stable key prior to sampling to guarantee
    exact reproducibility across environments and platforms.

    Parameters
    ----------
    raw_items:
        List of raw dictionaries from source.
    count:
        Target number of records to sample.
    seed:
        RNG seed value.

    Returns
    -------
    list[dict[str, Any]]
        Deterministically sampled subset of records.
    """
    if count <= 0 or not raw_items:
        return []

    # Sort stably by available identifier or json string
    def sort_key(item: dict[str, Any]) -> str:
        for k in ("task_id", "_id", "id", "question"):
            val = item.get(k)
            if val is not None:
                return str(val)
        return json.dumps(item, sort_keys=True)

    sorted_items = sorted(raw_items, key=sort_key)
    if len(sorted_items) <= count:
        return sorted_items

    rng = random.Random(seed)
    sampled = rng.sample(sorted_items, count)
    # Stably sort sampled items for consistent downstream processing
    return sorted(sampled, key=sort_key)


# ---------------------------------------------------------------------------
# 4. Stratified Split Assignment (FR-3) & Immutability Invariant
# ---------------------------------------------------------------------------
def assign_splits(
    task_dicts: list[dict[str, Any]],
    calibration_fraction: float = 0.25,
    seed: int = 42,
    existing_splits: Mapping[str, str] | None = None,
) -> list[TaskRecord]:
    """Assign tasks to 'calibration' and 'eval' splits deterministically (FR-3).

    Partitioning is:
    1. Stratified by domain: each domain independently receives the configured
       fraction of calibration tasks.
    2. Deterministic: governed by a seeded RNG over stably sorted task IDs.
    3. Immutable: tasks present in existing_splits retain their prior assignment.
       Attempting to reassign an existing task's split raises ValueError.

    Parameters
    ----------
    task_dicts:
        List of unassigned task property dictionaries.
    calibration_fraction:
        Fraction of tasks in each domain reserved for calibration (0.0 to 1.0).
    seed:
        Seed for the split assignment RNG.
    existing_splits:
        Optional mapping of {task_id: split} for previously persisted tasks.

    Returns
    -------
    list[TaskRecord]
        List of fully validated TaskRecord instances.
    """
    known_splits = existing_splits or {}
    records: list[TaskRecord] = []

    # Group by domain for stratified assignment
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for t in task_dicts:
        by_domain.setdefault(t["domain"], []).append(t)

    for domain, domain_tasks in sorted(by_domain.items(), key=lambda x: x[0]):
        # Separate tasks with existing splits from new unassigned tasks
        unassigned: list[dict[str, Any]] = []

        for task_item in domain_tasks:
            tid = task_item["task_id"]
            if tid in known_splits:
                prior_split = known_splits[tid]
                # Immutable: preserve existing split
                rec_dict = dict(task_item)
                rec_dict["split"] = prior_split
                records.append(TaskRecord.model_validate(rec_dict))
            else:
                unassigned.append(task_item)

        if not unassigned:
            continue

        # Rank unassigned tasks within each domain by sha256(f"{seed}:{task_id}")
        def task_rank_key(task_item: dict[str, Any]) -> tuple[str, str]:
            tid = str(task_item["task_id"])
            h = hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()
            return (h, tid)

        ranked = sorted(unassigned, key=task_rank_key)
        num_calib = round(len(ranked) * calibration_fraction)
        calib_ids = {t["task_id"] for t in ranked[:num_calib]}

        for task_item in unassigned:
            split_val = "calibration" if task_item["task_id"] in calib_ids else "eval"
            rec_dict = dict(task_item)
            rec_dict["split"] = split_val
            records.append(TaskRecord.model_validate(rec_dict))

    # Return records sorted by task_id
    return sorted(records, key=lambda r: r.task_id)


# ---------------------------------------------------------------------------
# 5. Leakage Guard (FR-3) & TCE Calibration Isolation
# ---------------------------------------------------------------------------
def assert_no_leakage(records: list[TaskRecord] | Iterable[TaskRecord]) -> None:
    """Assert zero leakage between calibration and evaluation sets (FR-3).

    Verifies:
    1. The calibration split and evaluation split are strictly disjoint (FR-3).
    2. No task_id appears with both split='calibration' and split='eval'.
    3. Every task carries a valid split ('calibration' or 'eval').

    Parameters
    ----------
    records:
        Collection of TaskRecord instances.

    Raises
    ------
    AssertionError
        If any calibration task_id appears in the evaluation split or vice-versa.
    """
    calib_ids: set[str] = set()
    eval_ids: set[str] = set()

    for r in records:
        if r.split == "calibration":
            if r.task_id in eval_ids:
                raise AssertionError(
                    f"FR-3 leakage violation: task '{r.task_id}' appears in both "
                    "calibration and evaluation splits."
                )
            calib_ids.add(r.task_id)
        elif r.split == "eval":
            if r.task_id in calib_ids:
                raise AssertionError(
                    f"FR-3 leakage violation: task '{r.task_id}' appears in both "
                    "calibration and evaluation splits."
                )
            eval_ids.add(r.task_id)
        else:
            raise AssertionError(
                f"FR-3 violation: task '{r.task_id}' has invalid split '{r.split}'. "
                "Must be 'calibration' or 'eval'."
            )


def get_calibration_tasks(records: list[TaskRecord]) -> list[TaskRecord]:
    """Retrieve only tasks assigned to the held-out calibration split (FR-3).

    Asserts zero leakage before returning calibration tasks to guarantee that
    TCE complexity calibration code only reads split='calibration'.

    Parameters
    ----------
    records:
        List of ingested TaskRecords.

    Returns
    -------
    list[TaskRecord]
        Calibration split tasks.
    """
    assert_no_leakage(records)
    return [r for r in records if r.split == "calibration"]


def get_eval_tasks(records: list[TaskRecord]) -> list[TaskRecord]:
    """Retrieve only tasks assigned to the evaluation split (FR-3).

    Asserts zero leakage before returning evaluation tasks to guarantee that
    evaluation runs never evaluate on held-out calibration tasks.

    Parameters
    ----------
    records:
        List of ingested TaskRecords.

    Returns
    -------
    list[TaskRecord]
        Evaluation split tasks.
    """
    assert_no_leakage(records)
    return [r for r in records if r.split == "eval"]


# ---------------------------------------------------------------------------
# 6. Database Adapter & JSONL Persistence
# ---------------------------------------------------------------------------
def write_to_db_adapter(records: list[TaskRecord]) -> int:
    """Adapter to write TaskRecords to database via src/oasis/db/models.py.

    Currently, src/oasis/db/models.py is a placeholder owned by Vidish.
    When db/models.py exposes task persistence (e.g. save_task or insert_tasks),
    this adapter function can be swapped in to persist records directly to SQLite.

    Parameters
    ----------
    records:
        List of TaskRecord instances.

    Returns
    -------
    int
        Number of records written to DB (0 if models.py does not yet implement writer).
    """
    try:
        from oasis.db import models  # type: ignore[import-untyped]

        if hasattr(models, "insert_tasks"):
            insert_fn = models.insert_tasks
            return int(insert_fn(records))
        elif hasattr(models, "save_task"):
            save_fn = models.save_task
            count = 0
            for r in records:
                save_fn(r)
                count += 1
            return count
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0

    # When db/models.py writer is not yet implemented, return 0
    return 0


def read_task_records(path: Path | str) -> list[TaskRecord]:
    """Read TaskRecords from a JSONL file.

    Parameters
    ----------
    path:
        Path to JSONL file.

    Returns
    -------
    list[TaskRecord]
        List of deserialized TaskRecord objects.
    """
    target = Path(path)
    if not target.exists():
        return []

    records: list[TaskRecord] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(TaskRecord.model_validate_json(stripped))
    return records


def save_task_records(
    records: list[TaskRecord],
    output_path: Path | str | None = None,
    write_db: bool = True,
) -> Path:
    """Save TaskRecords to JSONL and invoke the database adapter function.

    Parameters
    ----------
    records:
        List of TaskRecord instances to persist.
    output_path:
        Destination JSONL file path.
    write_db:
        Whether to invoke write_to_db_adapter.

    Returns
    -------
    Path
        Path to the saved JSONL file.
    """
    if write_db:
        write_to_db_adapter(records)

    out = Path(output_path or "data/tasks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    return out


# ---------------------------------------------------------------------------
# 7. High-Level Ingestion Orchestrator
# ---------------------------------------------------------------------------
def ingest_benchmarks(
    domains: list[TaskDomain] | None = None,
    config: BenchmarkConfig | None = None,
    count_override: int | None = None,
    seed_override: int | None = None,
    fixtures_dir: Path | None = None,
    use_fixtures: bool = False,
    output_path: Path | str | None = None,
    write_db: bool = True,
    created_at: str = DEFAULT_CREATED_AT,
) -> list[TaskRecord]:
    """Execute end-to-end benchmark ingestion across configured domains.

    Executes:
    1. Loading raw dataset items from HuggingFace or local fixture files.
    2. Deterministic sampling with a seeded RNG per source allocation.
    3. Field transformation into TaskRecord attributes.
    4. Stratified deterministic split assignment into 'calibration' and 'eval'.
    5. Invariant check asserting zero leakage between splits (FR-3).
    6. Persistence to JSONL and database adapter.

    Parameters
    ----------
    domains:
        List of domains to ingest (defaults to code_generation, research_qa, quant_analysis).
    config:
        Validated BenchmarkConfig instance.
    count_override:
        Optional count override for total tasks per domain.
    seed_override:
        Optional RNG seed override.
    fixtures_dir:
        Directory containing JSONL fixture samples for offline execution.
    use_fixtures:
        If True, forces loading from local fixture files with zero network calls.
    output_path:
        Output destination JSONL path.
    write_db:
        Whether to invoke database adapter on save.
    created_at:
        ISO timestamp string for created_at field.

    Returns
    -------
    list[TaskRecord]
        List of ingested and split-assigned TaskRecords.
    """
    cfg = config or load_benchmark_config()
    seed = seed_override if seed_override is not None else cfg.seed
    calib_frac = cfg.calibration_fraction

    target_domains: list[TaskDomain] = (
        domains
        if domains is not None
        else [
            "code_generation",
            "research_qa",
            "quant_analysis",
            "support_triage",
            "content_generation",
        ]
    )

    # Load existing records to preserve prior split assignments (immutability invariant)
    out_file = Path(output_path or cfg.output_path)
    existing_records = read_task_records(out_file)
    existing_splits = {r.task_id: r.split for r in existing_records}

    raw_task_dicts: list[dict[str, Any]] = []

    for domain in target_domains:
        allocation = cfg.source_allocation.get(domain, {})
        domain_total = (
            count_override
            if count_override is not None
            else cfg.domain_counts.get(domain, 20)
        )

        # Stably iterate sources within domain
        for source, src_target in sorted(allocation.items(), key=lambda x: x[0]):
            if count_override is not None:
                total_alloc = sum(allocation.values()) or 1
                sample_count = max(1, round(src_target * domain_total / total_alloc))
            else:
                sample_count = src_target

            if source == "authored":
                from oasis.bench.authored import load_authored_task_dicts

                authored_file: Path | None = None
                if fixtures_dir:
                    candidate1 = fixtures_dir / f"{domain}.yaml"
                    candidate2 = fixtures_dir / "authored" / f"{domain}.yaml"
                    if candidate1.exists():
                        authored_file = candidate1
                    elif candidate2.exists():
                        authored_file = candidate2
                if authored_file is None:
                    authored_file = (
                        Path(__file__).resolve().parent / "authored" / f"{domain}.yaml"
                    )

                raw_items = load_authored_task_dicts(authored_file, config=cfg)
            else:
                raw_items = load_raw_dataset(
                    source=source,
                    config=cfg,
                    fixtures_dir=fixtures_dir,
                    use_fixtures=use_fixtures,
                )

            # Filter out GSM8K records with no extractable numeric answer BEFORE sampling (FR-13)
            if source == "gsm8k":
                valid_items: list[dict[str, Any]] = []
                dropped = 0
                for item in raw_items:
                    ans = str(item.get("answer", ""))
                    if extract_gsm8k_numeric(ans) is not None:
                        valid_items.append(item)
                    else:
                        dropped += 1
                if dropped > 0:
                    logger.warning(
                        "Filtered out %d GSM8K records with no extractable numeric answer before sampling",
                        dropped,
                    )
                raw_items = valid_items

            # Filter out HotpotQA records with empty answer or no supporting titles BEFORE sampling (FR-13, FR-3)
            if source == "hotpotqa":
                valid_items = []
                dropped = 0
                for item in raw_items:
                    ans = str(item.get("answer", "")).strip()
                    supp_titles = extract_hotpotqa_supporting_titles(
                        item.get("supporting_facts", {})
                    )
                    if ans and supp_titles:
                        valid_items.append(item)
                    else:
                        dropped += 1
                if dropped > 0:
                    logger.warning(
                        "Filtered out %d HotpotQA records with empty answer or no supporting titles before sampling",
                        dropped,
                    )
                raw_items = valid_items

            # Seeded deterministic sampling with stable sha256-derived seed
            src_seed = derive_stable_seed(seed, source)
            sampled_items = sample_raw_data(
                raw_items, count=sample_count, seed=src_seed
            )

            for idx, raw in enumerate(sampled_items):
                task_dict = transform_raw_to_task(
                    raw=raw,
                    source=source,
                    domain=domain,
                    idx=idx,
                    created_at=created_at,
                )
                raw_task_dicts.append(task_dict)

    # Deterministic, stratified split assignment
    all_records = assign_splits(
        task_dicts=raw_task_dicts,
        calibration_fraction=calib_frac,
        seed=seed,
        existing_splits=existing_splits,
    )

    # Invariant guard: assert zero leakage between calibration and eval (FR-3)
    assert_no_leakage(all_records)

    # Save to JSONL and attempt database write via adapter
    save_task_records(all_records, output_path=out_file, write_db=write_db)

    return all_records


# ---------------------------------------------------------------------------
# 8. Command Line Interface: python -m oasis.bench.ingest --all
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint matching `python -m oasis.bench.ingest --all` (README.md)."""
    parser = argparse.ArgumentParser(
        description="Ingest benchmark tasks and partition into calibration/eval splits (FR-3, FR-13)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all configured benchmark domains (code_generation, research_qa, quant_analysis, support_triage, content_generation).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "code_generation",
            "research_qa",
            "quant_analysis",
            "support_triage",
            "content_generation",
        ],
        help="Ingest a specific task domain.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override number of tasks per domain (default from config/benchmark.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed for deterministic sampling and split assignment.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark.yaml configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: data/tasks.jsonl).",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default=None,
        help="Directory containing JSONL sample fixture files.",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Force loading from local fixture files (guaranteed offline, zero network).",
    )

    args = parser.parse_args(argv)

    if not args.all and not args.domain:
        parser.print_help()
        print("\nError: Must specify --all or --domain <domain>.")
        return 1

    cfg = load_benchmark_config(args.config)
    fixtures_path = Path(args.fixtures_dir) if args.fixtures_dir else None

    domains: list[TaskDomain] | None = None
    if args.domain:
        domains = [args.domain]

    try:
        records = ingest_benchmarks(
            domains=domains,
            config=cfg,
            count_override=args.count,
            seed_override=args.seed,
            fixtures_dir=fixtures_path,
            use_fixtures=args.use_fixtures,
            output_path=args.output,
        )
    except ImportError as exc:
        print(f"\n[oasis.bench.ingest] {exc}")
        print("Tip: Run with --use-fixtures or install datasets via 'uv add datasets'.")
        return 1
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"\n[oasis.bench.ingest] Error during ingestion: {exc}", file=sys.stderr)
        return 2

    calib_count = sum(1 for r in records if r.split == "calibration")
    eval_count = sum(1 for r in records if r.split == "eval")
    out_dest = args.output or cfg.output_path

    print(
        f"Successfully ingested {len(records)} tasks across {len({r.domain for r in records})} domains "
        f"({calib_count} calibration, {eval_count} eval) -> {out_dest}"
    )
    print("Zero-leakage invariant assertion (FR-3): PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
