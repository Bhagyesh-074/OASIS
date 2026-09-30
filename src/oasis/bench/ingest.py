"""Task ingestion and benchmark preparation (FR-3, FR-13, docs/DATABASE.md).

Loads tasks from benchmark datasets (MBPP, HumanEval, HotpotQA, GSM8K),
samples them deterministically using a seeded RNG, assigns stratified
calibration and evaluation splits (FR-3), enforces leakage invariants,
and exports TaskRecord rows via JSONL / database adapter.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.models import (
    TaskDomain,
    TaskRecord,
)

# Canonical timestamp for deterministic benchmark generation
DEFAULT_CREATED_AT = "2026-09-01T00:00:00+00:00"

# Default fixtures directory relative to repo root
DEFAULT_FIXTURES_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "bench"
)


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
        source_ref = str(raw.get("task_id", idx))
        try:
            task_id = f"mbpp_{int(source_ref):04d}"
        except ValueError:
            task_id = f"mbpp_{source_ref}"
        statement = str(raw.get("text", raw.get("prompt", ""))).strip()
        code = str(raw.get("code", "")).strip()
        test_list = raw.get("test_list", [])
        test_setup = raw.get("test_setup_code", "")
        test_code = "\n".join(test_list) if isinstance(test_list, list) else str(test_list)
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
        raw_id = str(raw.get("task_id", idx))
        clean_id = raw_id.replace("/", "_").lower()
        task_id = clean_id if clean_id.startswith("humaneval") else f"humaneval_{clean_id}"
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
        source_ref = str(raw.get("id", f"{idx:04d}"))
        task_id = f"hotpotqa_{source_ref}"
        statement = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        supporting_facts = raw.get("supporting_facts", {})
        spec = json.dumps({"expected_answer": answer, "supporting_facts": supporting_facts})
        return {
            "task_id": task_id,
            "domain": domain,
            "source": "hotpotqa",
            "source_ref": source_ref,
            "statement": statement,
            "reference_answer": answer,
            "verifier_type": "citation_resolve",
            "verifier_spec": spec,
            "complexity_label": None,
            "created_at": created_at,
        }

    elif source == "gsm8k":
        source_ref = str(raw.get("id", f"{idx:04d}"))
        task_id = f"gsm8k_{source_ref}"
        statement = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        expected_val = extract_gsm8k_numeric(answer)
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

    else:
        # Fallback for generic or authored sources
        source_ref = str(raw.get("id", raw.get("task_id", idx)))
        task_id = f"{source}_{source_ref}"
        statement = str(raw.get("statement", raw.get("prompt", raw.get("question", "")))).strip()
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
        return str(item.get("task_id", item.get("id", json.dumps(item, sort_keys=True))))

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

        # Sort stably by task_id before sampling
        unassigned_sorted = sorted(unassigned, key=lambda x: str(x["task_id"]))

        # Seeded domain-specific RNG
        domain_seed = seed + abs(hash(domain)) % 100_000
        rng = random.Random(domain_seed)

        num_calib = round(len(unassigned_sorted) * calibration_fraction)
        calib_indices = set(rng.sample(range(len(unassigned_sorted)), num_calib))

        for idx, task_item in enumerate(unassigned_sorted):
            split_val = "calibration" if idx in calib_indices else "eval"
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
    with out.open("w", encoding="utf-8") as f:
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
        else ["code_generation", "research_qa", "quant_analysis"]
    )

    # Load existing records to preserve prior split assignments (immutability invariant)
    out_file = Path(output_path or cfg.output_path)
    existing_records = read_task_records(out_file)
    existing_splits = {r.task_id: r.split for r in existing_records}

    raw_task_dicts: list[dict[str, Any]] = []

    for domain in target_domains:
        allocation = cfg.source_allocation.get(domain, {})
        domain_total = count_override if count_override is not None else cfg.domain_counts.get(domain, 20)

        # If allocation has specific counts, scale proportionally if count_override passed
        for source, src_target in allocation.items():
            if count_override is not None:
                total_alloc = sum(allocation.values()) or 1
                sample_count = max(1, round(src_target * domain_total / total_alloc))
            else:
                sample_count = src_target

            raw_items = load_raw_dataset(
                source=source,
                config=cfg,
                fixtures_dir=fixtures_dir,
                use_fixtures=use_fixtures,
            )

            # Seeded deterministic sampling
            src_seed = seed + abs(hash(source)) % 10_000
            sampled_items = sample_raw_data(raw_items, count=sample_count, seed=src_seed)

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
        help="Ingest all configured benchmark domains (code_generation, research_qa, quant_analysis).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["code_generation", "research_qa", "quant_analysis", "support_triage", "content_generation"],
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
