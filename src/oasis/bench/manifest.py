"""OASIS Benchmark Manifest Builder and Verifier (FR-3, NFR-4).

This module implements deterministic manifest generation and offline verification
for the OASIS benchmark suite. It captures dataset metadata, pinned git revisions,
LF-normalized configuration hashes, and cryptographic hashes of problem statements
and evaluation specifications to guarantee reproducibility (NFR-4) and enforce
calibration/evaluation split isolation with zero leakage (FR-3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.ingest import assert_no_leakage, read_task_records
from oasis.bench.models import TaskRecord

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0.0"


class DatasetManifestEntry(BaseModel):
    """Specification of a pinned benchmark dataset source (FR-3, NFR-4).

    Attributes
    ----------
    repo:
        Repository path on HuggingFace Hub (e.g. 'google-research-datasets/mbpp').
    config:
        Optional dataset subset or configuration name (e.g. 'sanitized', 'distractor').
    split:
        Dataset partition name (e.g. 'test', 'validation').
    revision:
        Pinned git commit hash ensuring exact source versioning (NFR-4).
    """

    model_config = ConfigDict(frozen=True)

    repo: str = Field(..., description="Repository path on HuggingFace Hub")
    config: str | None = Field(
        default=None, description="Dataset config/subset name"
    )
    split: str = Field(..., description="Dataset split name")
    revision: str = Field(..., description="Pinned git commit hash (NFR-4)")


class TaskManifestEntry(BaseModel):
    """Manifest record for an individual benchmark task (FR-3, NFR-4).

    Attributes
    ----------
    task_id:
        Unique task identifier (e.g. 'mbpp_0001', 'gsm8k_0001').
    domain:
        Benchmark task evaluation domain.
    split:
        Held-out partition ('calibration' or 'eval') (FR-3).
    source:
        Origin source dataset or 'authored'.
    source_ref:
        Optional reference ID within the source dataset.
    statement_sha256:
        SHA-256 hex digest of the problem statement normalized to LF.
    spec_sha256:
        SHA-256 hex digest of canonical JSON of {reference_answer, verifier_type,
        verifier_spec, complexity_label}.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique task identifier")
    domain: str = Field(..., description="Benchmark task domain")
    split: str = Field(..., description="Task split partition ('calibration' or 'eval') (FR-3)")
    source: str = Field(..., description="Origin dataset or 'authored'")
    source_ref: str | None = Field(default=None, description="Reference ID in source dataset")
    statement_sha256: str = Field(..., description="SHA-256 of LF-normalized statement")
    spec_sha256: str = Field(..., description="SHA-256 of canonical JSON spec")


class ManifestCounts(BaseModel):
    """Summary counts partitioned by domain, split, and source (FR-3, NFR-4).

    Attributes
    ----------
    domain:
        Task counts per domain.
    split:
        Task counts per split ('calibration' and 'eval').
    source:
        Task counts per data source.
    """

    model_config = ConfigDict(frozen=True)

    domain: dict[str, int] = Field(..., description="Task counts per domain")
    split: dict[str, int] = Field(..., description="Task counts per split (FR-3)")
    source: dict[str, int] = Field(..., description="Task counts per source")


class BenchmarkManifest(BaseModel):
    """Canonical benchmark manifest capturing frozen dataset state (FR-3, NFR-4).

    Attributes
    ----------
    manifest_version:
        Schema version of the manifest.
    config_sha256:
        SHA-256 hex digest of config/benchmark.yaml normalized to LF (NFR-4).
    tasks_sha256:
        SHA-256 hex digest of the tasks JSONL file normalized to LF (NFR-4).
    datasets:
        Pinned HuggingFace dataset specifications sorted by dataset name.
    counts:
        Task counts by domain, split, and source.
    tasks:
        Deterministic list of task entries sorted by task_id.
    """

    model_config = ConfigDict(frozen=True)

    manifest_version: str = Field(
        default=MANIFEST_VERSION, description="Manifest schema version"
    )
    config_sha256: str = Field(
        ..., description="SHA-256 of config/benchmark.yaml (LF-normalized)"
    )
    tasks_sha256: str = Field(
        ..., description="SHA-256 of tasks JSONL file (LF-normalized)"
    )
    datasets: dict[str, DatasetManifestEntry] = Field(
        ..., description="Pinned dataset specifications"
    )
    counts: ManifestCounts = Field(
        ..., description="Counts per domain, split, and source (FR-3)"
    )
    tasks: list[TaskManifestEntry] = Field(
        ..., description="Deterministic sorted task entries"
    )


# ---------------------------------------------------------------------------
# Cryptographic & Canonical Hashing Helpers (NFR-4, FR-3)
# ---------------------------------------------------------------------------


def compute_file_sha256_lf(path: Path | str) -> str:
    """Compute SHA-256 hex digest of file contents normalized to LF (NFR-4).

    Parameters
    ----------
    path:
        Path to file.

    Returns
    -------
    str
        64-character SHA-256 hexadecimal digest.
    """
    raw_bytes = Path(path).read_bytes()
    normalized = raw_bytes.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def compute_statement_sha256(statement: str) -> str:
    """Compute SHA-256 hex digest of a statement normalized to LF (NFR-4).

    Parameters
    ----------
    statement:
        Problem statement text.

    Returns
    -------
    str
        64-character SHA-256 hexadecimal digest.
    """
    normalized = statement.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_canonical_spec_json(
    reference_answer: str | None,
    verifier_type: str | None,
    verifier_spec: str | None,
    complexity_label: str | None,
) -> str:
    """Format canonical JSON with sorted keys and compact separators (NFR-4).

    Parameters
    ----------
    reference_answer:
        Gold reference answer string.
    verifier_type:
        L3 verifier identifier.
    verifier_spec:
        Verifier specification string (JSON or code).
    complexity_label:
        Coarse complexity category.

    Returns
    -------
    str
        Canonical JSON string with sorted keys, compact separators, no trailing newline.
    """
    payload = {
        "complexity_label": complexity_label,
        "reference_answer": (
            reference_answer.replace("\r\n", "\n")
            if reference_answer is not None
            else None
        ),
        "verifier_spec": (
            verifier_spec.replace("\r\n", "\n")
            if verifier_spec is not None
            else None
        ),
        "verifier_type": verifier_type,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_spec_sha256(
    reference_answer: str | None,
    verifier_type: str | None,
    verifier_spec: str | None,
    complexity_label: str | None,
) -> str:
    """Compute SHA-256 hex digest of canonical JSON spec (NFR-4).

    Parameters
    ----------
    reference_answer:
        Gold reference answer.
    verifier_type:
        L3 verifier type.
    verifier_spec:
        Verifier specification.
    complexity_label:
        Task complexity label.

    Returns
    -------
    str
        64-character SHA-256 hexadecimal digest.
    """
    canonical_json = compute_canonical_spec_json(
        reference_answer=reference_answer,
        verifier_type=verifier_type,
        verifier_spec=verifier_spec,
        complexity_label=complexity_label,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest Builder & Serializer (FR-3, NFR-4)
# ---------------------------------------------------------------------------


def extract_dataset_manifest_entries(
    config: BenchmarkConfig,
) -> dict[str, DatasetManifestEntry]:
    """Extract pinned dataset entries from BenchmarkConfig sorted by name (FR-3, NFR-4).

    Parameters
    ----------
    config:
        Validated BenchmarkConfig instance.

    Returns
    -------
    dict[str, DatasetManifestEntry]
        Sorted dictionary of dataset specifications.
    """
    entries: dict[str, DatasetManifestEntry] = {}
    for name in sorted(config.datasets.keys()):
        raw = config.datasets[name]
        repo = str(raw.get("repo") or raw.get("path") or "")
        cfg_name = raw.get("config") or raw.get("name")
        split = str(raw.get("split") or "")
        revision = str(raw.get("revision") or "")
        entries[name] = DatasetManifestEntry(
            repo=repo,
            config=str(cfg_name) if cfg_name is not None else None,
            split=split,
            revision=revision,
        )
    return entries


def compute_manifest_counts(records: list[TaskRecord]) -> ManifestCounts:
    """Compute task counts per domain, split, and source (FR-3, NFR-4).

    Parameters
    ----------
    records:
        List of TaskRecord instances.

    Returns
    -------
    ManifestCounts
        Counts with sorted dictionary keys.
    """
    d_counter = Counter(r.domain for r in records)
    sp_counter = Counter(r.split for r in records)
    src_counter = Counter(r.source for r in records)

    return ManifestCounts(
        domain={k: d_counter[k] for k in sorted(d_counter.keys())},
        split={k: sp_counter[k] for k in sorted(sp_counter.keys())},
        source={k: src_counter[k] for k in sorted(src_counter.keys())},
    )


def build_manifest(
    tasks_path: Path | str,
    config_path: Path | str,
) -> BenchmarkManifest:
    """Build a deterministic BenchmarkManifest from tasks and configuration (FR-3, NFR-4).

    Parameters
    ----------
    tasks_path:
        Path to tasks JSONL file.
    config_path:
        Path to benchmark.yaml file.

    Returns
    -------
    BenchmarkManifest
        Deterministic manifest object.
    """
    t_path = Path(tasks_path)
    c_path = Path(config_path)

    if not t_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {t_path}")
    if not c_path.exists():
        raise FileNotFoundError(f"Config file not found: {c_path}")

    config_sha = compute_file_sha256_lf(c_path)
    tasks_sha = compute_file_sha256_lf(t_path)

    cfg = load_benchmark_config(c_path)
    records = read_task_records(t_path)

    # Invariant guard: assert zero leakage before constructing manifest (FR-3)
    assert_no_leakage(records)

    datasets = extract_dataset_manifest_entries(cfg)
    counts = compute_manifest_counts(records)

    # Build per-task manifest entries sorted by task_id
    task_entries: list[TaskManifestEntry] = []
    for r in records:
        stmt_sha = compute_statement_sha256(r.statement)
        spec_sha = compute_spec_sha256(
            r.reference_answer,
            r.verifier_type,
            r.verifier_spec,
            r.complexity_label,
        )
        task_entries.append(
            TaskManifestEntry(
                task_id=r.task_id,
                domain=r.domain,
                split=r.split,
                source=r.source,
                source_ref=r.source_ref,
                statement_sha256=stmt_sha,
                spec_sha256=spec_sha,
            )
        )
    task_entries.sort(key=lambda t: t.task_id)

    return BenchmarkManifest(
        manifest_version=MANIFEST_VERSION,
        config_sha256=config_sha,
        tasks_sha256=tasks_sha,
        datasets=datasets,
        counts=counts,
        tasks=task_entries,
    )


def save_manifest(
    manifest: BenchmarkManifest,
    output_path: Path | str,
) -> Path:
    """Serialize manifest to deterministic JSON with sorted keys and LF line endings (NFR-4).

    Parameters
    ----------
    manifest:
        BenchmarkManifest instance.
    output_path:
        Destination path for manifest JSON.

    Returns
    -------
    Path
        Path to written manifest file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = manifest.model_dump()
    json_text = json.dumps(data, indent=2, sort_keys=True) + "\n"

    # Enforce pure LF line endings without carriage returns
    dest.write_bytes(json_text.encode("utf-8"))
    return dest


# ---------------------------------------------------------------------------
# Offline Manifest Verifier (FR-3, NFR-4)
# ---------------------------------------------------------------------------


def verify_manifest(
    tasks_path: Path | str = "data/tasks.jsonl",
    config_path: Path | str = "config/benchmark.yaml",
    manifest_path: Path | str = "config/benchmark_manifest.json",
) -> list[str]:
    """Verify tasks file and configuration against committed manifest (FR-3, NFR-4).

    Recomputes hashes, compares against the manifest, re-runs leakage checks (FR-3),
    and validates domain split counts against config. Returns a list of difference
    descriptions categorized as: added, removed, changed, split moved, or revision changed.

    Parameters
    ----------
    tasks_path:
        Path to tasks JSONL file.
    config_path:
        Path to benchmark.yaml config file.
    manifest_path:
        Path to benchmark_manifest.json file.

    Returns
    -------
    list[str]
        List of detected differences. Empty list indicates full verification pass.
    """
    m_path = Path(manifest_path)
    t_path = Path(tasks_path)
    c_path = Path(config_path)

    differences: list[str] = []

    if not m_path.exists():
        return [f"Manifest file does not exist: {m_path}"]
    if not t_path.exists():
        return [f"Tasks file does not exist: {t_path}"]
    if not c_path.exists():
        return [f"Config file does not exist: {c_path}"]

    # 1. Load manifest
    try:
        manifest_data = json.loads(m_path.read_text(encoding="utf-8"))
        manifest = BenchmarkManifest.model_validate(manifest_data)
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
        return [f"Failed to parse manifest at {m_path}: {exc}"]

    # 2. Check configuration and revisions
    recomputed_config_sha = compute_file_sha256_lf(c_path)
    cfg = load_benchmark_config(c_path)
    recomputed_datasets = extract_dataset_manifest_entries(cfg)

    # Check for revision changes and dataset configuration changes
    for ds_name, m_ds in manifest.datasets.items():
        if ds_name not in recomputed_datasets:
            differences.append(
                f"removed: dataset '{ds_name}' present in manifest but missing from config"
            )
        else:
            c_ds = recomputed_datasets[ds_name]
            if c_ds.revision != m_ds.revision:
                differences.append(
                    f"revision changed: dataset '{ds_name}' revision '{m_ds.revision}' -> '{c_ds.revision}'"
                )
            if c_ds.repo != m_ds.repo or c_ds.config != m_ds.config or c_ds.split != m_ds.split:
                differences.append(
                    f"changed: dataset '{ds_name}' spec changed in config"
                )

    for ds_name in recomputed_datasets:
        if ds_name not in manifest.datasets:
            differences.append(
                f"added: dataset '{ds_name}' present in config but missing from manifest"
            )

    if recomputed_config_sha != manifest.config_sha256 and not any(
        "revision changed" in d or "dataset" in d for d in differences
    ):
        differences.append(
            f"changed: config/benchmark.yaml SHA-256 mismatch ('{manifest.config_sha256}' -> '{recomputed_config_sha}')"
        )

    # 3. Read tasks and verify file SHA-256
    recomputed_tasks_sha = compute_file_sha256_lf(t_path)
    records = read_task_records(t_path)

    # 4. Re-run zero leakage invariant check (FR-3)
    try:
        assert_no_leakage(records)
    except AssertionError as exc:
        differences.append(f"leakage check failed (FR-3): {exc}")

    # 5. Check per-domain split counts against config (FR-3)
    domain_records_map: dict[str, list[TaskRecord]] = {}
    for r in records:
        domain_records_map.setdefault(r.domain, []).append(r)

    for domain, domain_records in sorted(domain_records_map.items()):
        n = len(domain_records)
        expected_calib = round(n * cfg.calibration_fraction)
        expected_eval = n - expected_calib

        actual_calib = sum(1 for r in domain_records if r.split == "calibration")
        actual_eval = sum(1 for r in domain_records if r.split == "eval")

        if actual_calib != expected_calib:
            differences.append(
                f"changed: domain '{domain}' calibration split count ({actual_calib}) "
                f"differs from expected ({expected_calib}) (FR-3)"
            )
        if actual_eval != expected_eval:
            differences.append(
                f"changed: domain '{domain}' eval split count ({actual_eval}) "
                f"differs from expected ({expected_eval}) (FR-3)"
            )

    # Validate aggregate counts against manifest
    recomputed_counts = compute_manifest_counts(records)
    if recomputed_counts.domain != manifest.counts.domain:
        differences.append(
            f"changed: domain counts mismatch between tasks and manifest "
            f"({recomputed_counts.domain} vs {manifest.counts.domain})"
        )
    if recomputed_counts.split != manifest.counts.split:
        differences.append(
            f"changed: split counts mismatch between tasks and manifest "
            f"({recomputed_counts.split} vs {manifest.counts.split})"
        )
    if recomputed_counts.source != manifest.counts.source:
        differences.append(
            f"changed: source counts mismatch between tasks and manifest "
            f"({recomputed_counts.source} vs {manifest.counts.source})"
        )

    # 6. Compare task records against manifest entries
    manifest_map = {t.task_id: t for t in manifest.tasks}
    recomputed_map = {r.task_id: r for r in records}

    # Added tasks (in tasks file but not in manifest)
    for tid in sorted(recomputed_map.keys() - manifest_map.keys()):
        differences.append(
            f"added: task '{tid}' present in tasks file but missing from manifest"
        )

    # Removed tasks (in manifest but not in tasks file)
    for tid in sorted(manifest_map.keys() - recomputed_map.keys()):
        differences.append(
            f"removed: task '{tid}' present in manifest but missing from tasks file"
        )

    # Tasks present in both: inspect splits, statements, and specs
    for tid in sorted(manifest_map.keys() & recomputed_map.keys()):
        m_entry = manifest_map[tid]
        r_rec = recomputed_map[tid]

        if r_rec.split != m_entry.split:
            differences.append(
                f"split moved: task '{tid}' moved from '{m_entry.split}' to '{r_rec.split}'"
            )
        if r_rec.domain != m_entry.domain:
            differences.append(
                f"changed: task '{tid}' domain changed ('{m_entry.domain}' -> '{r_rec.domain}')"
            )
        if r_rec.source != m_entry.source:
            differences.append(
                f"changed: task '{tid}' source changed ('{m_entry.source}' -> '{r_rec.source}')"
            )
        if r_rec.source_ref != m_entry.source_ref:
            differences.append(
                f"changed: task '{tid}' source_ref changed ('{m_entry.source_ref}' -> '{r_rec.source_ref}')"
            )

        stmt_sha = compute_statement_sha256(r_rec.statement)
        if stmt_sha != m_entry.statement_sha256:
            differences.append(
                f"changed: task '{tid}' statement SHA-256 changed ('{m_entry.statement_sha256}' -> '{stmt_sha}')"
            )

        spec_sha = compute_spec_sha256(
            r_rec.reference_answer,
            r_rec.verifier_type,
            r_rec.verifier_spec,
            r_rec.complexity_label,
        )
        if spec_sha != m_entry.spec_sha256:
            differences.append(
                f"changed: task '{tid}' spec SHA-256 changed ('{m_entry.spec_sha256}' -> '{spec_sha}')"
            )

    # Overall tasks file SHA check fallback if no individual task difference reported
    if recomputed_tasks_sha != manifest.tasks_sha256 and not any(
        "task" in d for d in differences
    ):
        differences.append(
            f"changed: tasks file SHA-256 mismatch ('{manifest.tasks_sha256}' -> '{recomputed_tasks_sha}')"
        )

    return differences


# ---------------------------------------------------------------------------
# CLI Command Dispatcher (FR-3, NFR-4)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for benchmark manifest build and verify (FR-3, NFR-4).

    Parameters
    ----------
    argv:
        Optional argument list. Defaults to sys.argv[1:].

    Returns
    -------
    int
        0 on success, non-zero on failure or difference detection.
    """
    parser = argparse.ArgumentParser(
        prog="python -m oasis.bench.manifest",
        description="OASIS Benchmark Manifest Builder and Verifier (FR-3, NFR-4).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: build
    build_parser = subparsers.add_parser(
        "build", help="Build config/benchmark_manifest.json from tasks and config (NFR-4)."
    )
    build_parser.add_argument(
        "--tasks",
        type=str,
        default="data/tasks.jsonl",
        help="Path to tasks JSONL file (default: data/tasks.jsonl).",
    )
    build_parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark.yaml",
        help="Path to benchmark.yaml config file (default: config/benchmark.yaml).",
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default="config/benchmark_manifest.json",
        help="Destination path for manifest JSON (default: config/benchmark_manifest.json).",
    )

    # Subcommand: verify
    verify_parser = subparsers.add_parser(
        "verify", help="Verify tasks and config against benchmark manifest (FR-3, NFR-4)."
    )
    verify_parser.add_argument(
        "--tasks",
        type=str,
        default="data/tasks.jsonl",
        help="Path to tasks JSONL file (default: data/tasks.jsonl).",
    )
    verify_parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark.yaml",
        help="Path to benchmark.yaml config file (default: config/benchmark.yaml).",
    )
    verify_parser.add_argument(
        "--manifest",
        type=str,
        default="config/benchmark_manifest.json",
        help="Path to manifest JSON file (default: config/benchmark_manifest.json).",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        try:
            manifest = build_manifest(tasks_path=args.tasks, config_path=args.config)
            dest = save_manifest(manifest=manifest, output_path=args.output)
            print(
                f"Successfully built benchmark manifest with {len(manifest.tasks)} tasks -> {dest}"
            )
            print(f"Config SHA-256: {manifest.config_sha256}")
            print(f"Tasks SHA-256:  {manifest.tasks_sha256}")
            print("Per-domain counts:")
            for d, cnt in manifest.counts.domain.items():
                print(f"  {d}: {cnt}")
            return 0
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"Error building manifest: {exc}", file=sys.stderr)
            return 1

    elif args.command == "verify":
        diffs = verify_manifest(
            tasks_path=args.tasks,
            config_path=args.config,
            manifest_path=args.manifest,
        )
        if diffs:
            print(
                f"Benchmark manifest verification FAILED with {len(diffs)} difference(s):",
                file=sys.stderr,
            )
            for d in diffs:
                print(f"  - {d}", file=sys.stderr)
            return 1
        else:
            print(
                "Benchmark manifest verification PASSED: all hashes, splits, counts, "
                "and zero-leakage invariant verified (FR-3, NFR-4)."
            )
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
