"""Unit tests for task ingestion, loaders, determinism, and stratification (FR-3, FR-13)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from oasis.bench.config import BenchmarkConfig, load_benchmark_config
from oasis.bench.ingest import (
    assign_splits,
    extract_gsm8k_numeric,
    ingest_benchmarks,
    load_raw_from_fixture,
    main,
    read_task_records,
    sample_raw_data,
    transform_raw_to_task,
    write_to_db_adapter,
)
from oasis.bench.models import CONTEXT_MARKER, CitationResolveSpec, TaskRecord


# ---------------------------------------------------------------------------
# 1. Database schema mirroring (docs/DATABASE.md task table)
# ---------------------------------------------------------------------------
def test_task_record_schema_matches_database_spec() -> None:
    """TaskRecord mirrors all 11 columns from the task SQLite table in DATABASE.md."""
    task = TaskRecord(
        task_id="mbpp_0001",
        domain="code_generation",
        source="mbpp",
        source_ref="1",
        statement="Write a function to find min cost path.",
        reference_answer="def min_cost(): pass",
        verifier_type="pytest",
        verifier_spec=json.dumps({"test_code": "assert True"}),
        complexity_label="medium",
        split="calibration",
        created_at="2026-09-01T00:00:00+00:00",
    )

    data = task.model_dump()
    expected_keys = {
        "task_id",
        "domain",
        "source",
        "source_ref",
        "statement",
        "reference_answer",
        "verifier_type",
        "verifier_spec",
        "complexity_label",
        "split",
        "created_at",
    }
    assert set(data.keys()) == expected_keys
    assert task.parsed_verifier_spec() == {"test_code": "assert True"}

    # Invalid domain rejected
    with pytest.raises(ValidationError):
        TaskRecord(
            task_id="invalid_1",
            domain="invalid_domain",  # type: ignore[arg-type]
            source="mbpp",
            statement="test",
            split="calibration",
            created_at="2026-09-01T00:00:00Z",
        )

    # Invalid split rejected
    with pytest.raises(ValidationError):
        TaskRecord(
            task_id="invalid_2",
            domain="code_generation",
            source="mbpp",
            statement="test",
            split="train",  # type: ignore[arg-type]
            created_at="2026-09-01T00:00:00Z",
        )


# ---------------------------------------------------------------------------
# 2. Source transformers & verifier specifications (FR-13)
# ---------------------------------------------------------------------------
def test_mbpp_transformation(fixtures_dir: Path) -> None:
    """MBPP raw records map to code_generation domain with pytest verifier."""
    raw_items = load_raw_from_fixture(fixtures_dir / "mbpp_sample.jsonl")
    assert len(raw_items) >= 2

    task_dict = transform_raw_to_task(
        raw_items[0], source="mbpp", domain="code_generation"
    )
    assert task_dict["task_id"] == "mbpp_0001"
    assert task_dict["domain"] == "code_generation"
    assert task_dict["source"] == "mbpp"
    assert task_dict["verifier_type"] == "pytest"

    # Context marker and first assert presence (including function name)
    assert CONTEXT_MARKER in task_dict["statement"]
    assert "Your code should pass this test:\nassert min_cost" in task_dict["statement"]
    assert "min_cost" in task_dict["statement"]

    spec = json.loads(task_dict["verifier_spec"])
    assert "assert min_cost" in spec["test_code"]


def test_humaneval_transformation(fixtures_dir: Path) -> None:
    """HumanEval raw records map to code_generation domain with pytest verifier."""
    raw_items = load_raw_from_fixture(fixtures_dir / "humaneval_sample.jsonl")
    assert len(raw_items) >= 2

    task_dict = transform_raw_to_task(
        raw_items[0], source="humaneval", domain="code_generation"
    )
    assert task_dict["task_id"] == "humaneval_0"
    assert task_dict["domain"] == "code_generation"
    assert task_dict["source"] == "humaneval"
    assert task_dict["verifier_type"] == "pytest"
    spec = json.loads(task_dict["verifier_spec"])
    assert spec["entry_point"] == "has_close_elements"
    assert "def check(candidate):" in spec["test_code"]


def test_hotpotqa_transformation(fixtures_dir: Path) -> None:
    """HotpotQA raw records map to research_qa domain with distractor statement and citation_resolve verifier (FR-13, FR-3)."""
    raw_items = load_raw_from_fixture(fixtures_dir / "hotpotqa_sample.jsonl")
    assert len(raw_items) >= 2

    task_dict = transform_raw_to_task(
        raw_items[0], source="hotpotqa", domain="research_qa"
    )
    assert task_dict["task_id"] == "hotpotqa_5a7a0b3e5542990178904833"
    assert task_dict["domain"] == "research_qa"
    assert task_dict["source"] == "hotpotqa"
    assert task_dict["source_ref"] == "5a7a0b3e5542990178904833"
    assert task_dict["verifier_type"] == "citation_resolve"

    # Statement must contain instruction line, question, CONTEXT_MARKER, and all context paragraphs
    instruction = "Answer using only the context below and cite the titles of the paragraphs you used."
    assert task_dict["statement"].startswith(instruction)
    assert raw_items[0]["question"] in task_dict["statement"]
    assert CONTEXT_MARKER in task_dict["statement"]

    # Validate verifier_spec with CitationResolveSpec
    spec_dict = json.loads(task_dict["verifier_spec"])
    assert spec_dict["expected_answer"] == "yes"
    assert spec_dict["supporting_titles"] == ["Scott Derrickson", "Ed Wood"]
    assert spec_dict["context_titles"] == [
        "Scott Derrickson",
        "Ed Wood",
        "Doctor Strange (film)",
    ]

    spec_obj = CitationResolveSpec.model_validate_json(task_dict["verifier_spec"])
    assert spec_obj.expected_answer == "yes"
    assert spec_obj.supporting_titles == ["Scott Derrickson", "Ed Wood"]
    assert spec_obj.context_titles == [
        "Scott Derrickson",
        "Ed Wood",
        "Doctor Strange (film)",
    ]

    # Every context title must appear in statement as "[Title]"
    for title in spec_obj.context_titles:
        assert f"[{title}]" in task_dict["statement"]

    # Paragraphs appear in dataset's own order
    pos_marker = task_dict["statement"].index(CONTEXT_MARKER)
    pos1 = task_dict["statement"].index("[Scott Derrickson]")
    pos2 = task_dict["statement"].index("[Ed Wood]")
    pos3 = task_dict["statement"].index("[Doctor Strange (film)]")
    assert pos_marker < pos1 < pos2 < pos3


def test_hotpotqa_validation_and_rejection() -> None:
    """CitationResolveSpec and HotpotQA loader reject records with empty answer or no supporting titles (FR-13, FR-3)."""
    # 1. Pydantic CitationResolveSpec model direct validations
    with pytest.raises(ValidationError):
        CitationResolveSpec(
            expected_answer="",
            context_titles=["T1"],
            supporting_titles=["T1"],
        )

    with pytest.raises(ValidationError):
        CitationResolveSpec(
            expected_answer="   ",
            context_titles=["T1"],
            supporting_titles=["T1"],
        )

    with pytest.raises(ValidationError):
        CitationResolveSpec(
            expected_answer="valid answer",
            context_titles=["T1"],
            supporting_titles=[],
        )

    with pytest.raises(ValidationError):
        CitationResolveSpec(
            expected_answer="valid answer",
            context_titles=["T1"],
            supporting_titles=["", "  "],
        )

    # 2. transform_raw_to_task rejection of empty expected_answer
    with pytest.raises(ValueError, match="Invalid citation_resolve spec"):
        transform_raw_to_task(
            {
                "_id": "empty_ans",
                "question": "Some question?",
                "answer": "",
                "supporting_facts": {"title": ["Doc 1"], "sent_id": [0]},
                "context": {"title": ["Doc 1"], "sentences": [["Some text."]]},
            },
            source="hotpotqa",
            domain="research_qa",
        )

    # 3. transform_raw_to_task rejection of whitespace expected_answer
    with pytest.raises(ValueError, match="Invalid citation_resolve spec"):
        transform_raw_to_task(
            {
                "_id": "space_ans",
                "question": "Some question?",
                "answer": "   \n\t  ",
                "supporting_facts": {"title": ["Doc 1"], "sent_id": [0]},
                "context": {"title": ["Doc 1"], "sentences": [["Some text."]]},
            },
            source="hotpotqa",
            domain="research_qa",
        )

    # 4. transform_raw_to_task rejection of missing supporting_facts / no supporting titles
    with pytest.raises(ValueError, match="Invalid citation_resolve spec"):
        transform_raw_to_task(
            {
                "_id": "no_supp",
                "question": "Some question?",
                "answer": "valid answer",
                "supporting_facts": {"title": [], "sent_id": []},
                "context": {"title": ["Doc 1"], "sentences": [["Some text."]]},
            },
            source="hotpotqa",
            domain="research_qa",
        )

    # 5. transform_raw_to_task rejection of missing question
    with pytest.raises(ValueError, match="missing 'question'"):
        transform_raw_to_task(
            {
                "_id": "no_q",
                "question": "",
                "answer": "valid answer",
                "supporting_facts": {"title": ["Doc 1"], "sent_id": [0]},
                "context": {"title": ["Doc 1"], "sentences": [["Some text."]]},
            },
            source="hotpotqa",
            domain="research_qa",
        )


def test_hotpotqa_ids_and_splits_unchanged(fixtures_dir: Path) -> None:
    """HotpotQA distractor ingestion preserves exact task IDs and split assignments (FR-13, FR-3)."""
    tasks = ingest_benchmarks(
        domains=["research_qa"],
        fixtures_dir=fixtures_dir,
        use_fixtures=True,
        write_db=False,
    )
    assert len(tasks) == 6

    expected_ids = [
        "hotpotqa_5a7a0b3e5542990178904833",
        "hotpotqa_5a8b57f25542995d1e6f1371",
        "hotpotqa_5a8c25345542995d1e6f1402",
        "hotpotqa_5a8d46215542995d1e6f1450",
        "hotpotqa_5a8e73455542995d1e6f1512",
        "hotpotqa_5a8f98105542995d1e6f1590",
    ]
    assert [t.task_id for t in tasks] == expected_ids

    expected_splits = {
        "hotpotqa_5a7a0b3e5542990178904833": "calibration",
        "hotpotqa_5a8b57f25542995d1e6f1371": "eval",
        "hotpotqa_5a8c25345542995d1e6f1402": "eval",
        "hotpotqa_5a8d46215542995d1e6f1450": "eval",
        "hotpotqa_5a8e73455542995d1e6f1512": "eval",
        "hotpotqa_5a8f98105542995d1e6f1590": "calibration",
    }
    actual_splits = {t.task_id: t.split for t in tasks}
    assert actual_splits == expected_splits

    # Every task statement must contain CONTEXT_MARKER and every context title
    for t in tasks:
        assert t.verifier_type == "citation_resolve"
        assert CONTEXT_MARKER in t.statement
        spec = CitationResolveSpec.model_validate_json(t.verifier_spec or "{}")
        assert len(spec.supporting_titles) >= 1
        assert len(spec.context_titles) >= 1
        for title in spec.context_titles:
            assert f"[{title}]" in t.statement


def test_gsm8k_transformation_and_numeric_extraction(fixtures_dir: Path) -> None:
    """GSM8K raw records map to quant_analysis domain with numeric_consistency verifier."""
    raw_items = load_raw_from_fixture(fixtures_dir / "gsm8k_sample.jsonl")
    assert len(raw_items) >= 2

    task_dict = transform_raw_to_task(
        raw_items[0], source="gsm8k", domain="quant_analysis"
    )
    assert task_dict["task_id"] == "gsm8k_73429d5629"
    assert task_dict["domain"] == "quant_analysis"
    assert task_dict["source"] == "gsm8k"
    assert task_dict["source_ref"] == "73429d5629"
    assert task_dict["verifier_type"] == "numeric_consistency"
    spec = json.loads(task_dict["verifier_spec"])
    assert spec["expected_value"] == 72.0
    assert spec["tolerance"] == 1e-4

    # Direct unit tests for numeric extraction
    assert extract_gsm8k_numeric("The answer is 42. #### 42") == 42.0
    assert extract_gsm8k_numeric("Calculated to 1,250.50\n#### 1,250.50") == 1250.5
    assert extract_gsm8k_numeric("Loss is -15. #### -15") == -15.0
    assert extract_gsm8k_numeric("No marker here but ends with 99") == 99.0
    assert extract_gsm8k_numeric("No numbers at all") is None


def test_gsm8k_filtering_and_null_value_rejection(
    fixtures_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Filter out GSM8K records with no extractable numeric answer before sampling, and reject null expected_value."""
    raw_items = load_raw_from_fixture(fixtures_dir / "gsm8k_sample.jsonl")
    # Verify the unextractable fixture row is present
    unextractable = [
        r for r in raw_items if extract_gsm8k_numeric(str(r.get("answer", ""))) is None
    ]
    assert len(unextractable) >= 1

    # Ingesting benchmarks with fixtures drops the unextractable record
    with caplog.at_level("WARNING"):
        records = ingest_benchmarks(
            domains=["quant_analysis"],
            count_override=10,
            fixtures_dir=fixtures_dir,
            use_fixtures=True,
            write_db=False,
        )

    # Ingested records must all have valid numeric consistency specs
    assert len(records) > 0
    for r in records:
        assert r.verifier_type == "numeric_consistency"
        spec = r.parsed_verifier_spec()
        assert spec.get("expected_value") is not None
        assert isinstance(spec["expected_value"], (int, float))

    # Calling transform_raw_to_task directly on unextractable record raises ValueError
    with pytest.raises(
        ValueError,
        match="Cannot emit numeric_consistency spec with null expected_value",
    ):
        transform_raw_to_task(unextractable[0], source="gsm8k", domain="quant_analysis")


def test_different_sources_never_produce_same_id(fixtures_dir: Path) -> None:
    """Two different sources must never produce identical task IDs."""
    # 1. Collision resistance test: different sources with overlapping raw identifiers
    mbpp_task = transform_raw_to_task(
        {
            "task_id": 1,
            "text": "mbpp problem",
            "code": "def f(): pass",
            "test_list": ["assert True"],
        },
        source="mbpp",
        domain="code_generation",
    )
    humaneval_task = transform_raw_to_task(
        {
            "task_id": "HumanEval/1",
            "prompt": "he problem",
            "canonical_solution": "pass",
            "test": "",
            "entry_point": "f",
        },
        source="humaneval",
        domain="code_generation",
    )
    hotpotqa_task = transform_raw_to_task(
        {
            "_id": "1",
            "question": "hotpot problem",
            "answer": "yes",
            "supporting_facts": {"title": ["Doc 1"], "sent_id": [0]},
            "context": {"title": ["Doc 1"], "sentences": [["Context text."]]},
        },
        source="hotpotqa",
        domain="research_qa",
    )
    gsm8k_task = transform_raw_to_task(
        {"question": "gsm problem", "answer": "1 #### 1"},
        source="gsm8k",
        domain="quant_analysis",
    )
    authored_task = transform_raw_to_task(
        {"task_id": "1", "statement": "authored problem"},
        source="authored",
        domain="support_triage",
    )

    generated_ids = [
        mbpp_task["task_id"],
        humaneval_task["task_id"],
        hotpotqa_task["task_id"],
        gsm8k_task["task_id"],
        authored_task["task_id"],
    ]
    # Verify strict uniqueness across distinct sources
    assert len(generated_ids) == len(set(generated_ids))

    # Verify each ID is properly prefixed by its own source
    assert mbpp_task["task_id"] == "mbpp_0001"
    assert humaneval_task["task_id"] == "humaneval_1"
    assert hotpotqa_task["task_id"] == "hotpotqa_1"
    assert gsm8k_task["task_id"].startswith("gsm8k_")
    assert authored_task["task_id"] == "authored_1"

    # 2. Verify disjointness across all sample fixture datasets
    mbpp_raw = load_raw_from_fixture(fixtures_dir / "mbpp_sample.jsonl")
    he_raw = load_raw_from_fixture(fixtures_dir / "humaneval_sample.jsonl")
    hp_raw = load_raw_from_fixture(fixtures_dir / "hotpotqa_sample.jsonl")
    gsm_raw = load_raw_from_fixture(fixtures_dir / "gsm8k_sample.jsonl")

    mbpp_ids = {
        transform_raw_to_task(r, "mbpp", "code_generation")["task_id"] for r in mbpp_raw
    }
    he_ids = {
        transform_raw_to_task(r, "humaneval", "code_generation")["task_id"]
        for r in he_raw
    }
    hp_ids = {
        transform_raw_to_task(r, "hotpotqa", "research_qa")["task_id"] for r in hp_raw
    }
    gsm_valid = [
        r
        for r in gsm_raw
        if extract_gsm8k_numeric(str(r.get("answer", ""))) is not None
    ]
    gsm_ids = {
        transform_raw_to_task(r, "gsm8k", "quant_analysis")["task_id"]
        for r in gsm_valid
    }

    assert mbpp_ids.isdisjoint(he_ids)
    assert mbpp_ids.isdisjoint(hp_ids)
    assert mbpp_ids.isdisjoint(gsm_ids)
    assert he_ids.isdisjoint(hp_ids)
    assert he_ids.isdisjoint(gsm_ids)
    assert hp_ids.isdisjoint(gsm_ids)


# ---------------------------------------------------------------------------
# 3. Deterministic sampling & split stratification (FR-3)
# ---------------------------------------------------------------------------
def test_deterministic_sampling(fixtures_dir: Path) -> None:
    """Same seed and count produce exact same sampled records across repeated calls."""
    raw_items = load_raw_from_fixture(fixtures_dir / "mbpp_sample.jsonl")

    sample1 = sample_raw_data(raw_items, count=3, seed=42)
    sample2 = sample_raw_data(raw_items, count=3, seed=42)
    sample3 = sample_raw_data(raw_items, count=3, seed=999)

    assert sample1 == sample2
    assert len(sample1) == 3
    # Different seeds should produce distinct samples (when pool > count)
    if len(raw_items) > 3:
        assert [x["task_id"] for x in sample1] != [x["task_id"] for x in sample3]


def test_stratification_by_domain() -> None:
    """Split assignment stratifies calibration vs eval proportionally by domain (FR-3)."""
    # 20 tasks per domain across 2 domains = 40 tasks total
    tasks: list[dict[str, Any]] = []
    for i in range(20):
        tasks.append(
            {
                "task_id": f"code_{i:02d}",
                "domain": "code_generation",
                "source": "mbpp",
                "statement": f"Code task {i}",
                "created_at": "2026-09-01T00:00:00Z",
            }
        )
        tasks.append(
            {
                "task_id": f"qa_{i:02d}",
                "domain": "research_qa",
                "source": "hotpotqa",
                "statement": f"QA task {i}",
                "created_at": "2026-09-01T00:00:00Z",
            }
        )

    # Calibration fraction: 0.25 (5 calibration, 15 eval per domain)
    records = assign_splits(tasks, calibration_fraction=0.25, seed=42)
    assert len(records) == 40

    code_records = [r for r in records if r.domain == "code_generation"]
    qa_records = [r for r in records if r.domain == "research_qa"]

    assert sum(1 for r in code_records if r.split == "calibration") == 5
    assert sum(1 for r in code_records if r.split == "eval") == 15

    assert sum(1 for r in qa_records if r.split == "calibration") == 5
    assert sum(1 for r in qa_records if r.split == "eval") == 15


def test_split_assignment_determinism() -> None:
    """assign_splits is fully deterministic given the same seed."""
    tasks = [
        {
            "task_id": f"t_{i:02d}",
            "domain": "code_generation",
            "source": "mbpp",
            "statement": f"Task {i}",
            "created_at": "2026-09-01T00:00:00Z",
        }
        for i in range(12)
    ]

    records1 = assign_splits(tasks, calibration_fraction=0.25, seed=42)
    records2 = assign_splits(tasks, calibration_fraction=0.25, seed=42)
    assert [r.split for r in records1] == [r.split for r in records2]


def test_never_reassign_split_after_first_write() -> None:
    """Never reassign a task's split after first write (immutability invariant)."""
    tasks = [
        {
            "task_id": "task_fixed_1",
            "domain": "code_generation",
            "source": "mbpp",
            "statement": "Fixed task",
            "created_at": "2026-09-01T00:00:00Z",
        },
        {
            "task_id": "task_fixed_2",
            "domain": "code_generation",
            "source": "mbpp",
            "statement": "Another task",
            "created_at": "2026-09-01T00:00:00Z",
        },
    ]

    # Pre-existing split recorded from previous run
    existing_splits = {
        "task_fixed_1": "calibration",
        "task_fixed_2": "eval",
    }

    # Even with seed/fraction that would normally assign differently, existing splits are kept
    records = assign_splits(
        tasks,
        calibration_fraction=0.0,  # normally 0 calibration
        seed=12345,
        existing_splits=existing_splits,
    )

    by_id = {r.task_id: r.split for r in records}
    assert by_id["task_fixed_1"] == "calibration"
    assert by_id["task_fixed_2"] == "eval"


# ---------------------------------------------------------------------------
# 4. Configuration, JSONL persistence, and adapter hook
# ---------------------------------------------------------------------------
def test_config_loading_and_defaults() -> None:
    """BenchmarkConfig loads config/benchmark.yaml and enforces bounds."""
    config = load_benchmark_config()
    assert config.seed == 42
    assert config.calibration_fraction == 0.25
    assert "code_generation" in config.domain_counts
    assert "research_qa" in config.domain_counts
    assert "quant_analysis" in config.domain_counts


def test_jsonl_output_and_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    """Ingestion persists valid JSONL and invokes database adapter."""
    out_file = tmp_path / "tasks_test.jsonl"
    config = BenchmarkConfig(
        seed=42,
        calibration_fraction=0.5,
        domain_counts={"code_generation": 4},
        source_allocation={"code_generation": {"mbpp": 2, "humaneval": 2}},
        output_path=str(out_file),
    )

    records = ingest_benchmarks(
        domains=["code_generation"],
        config=config,
        fixtures_dir=fixtures_dir,
        use_fixtures=True,
        output_path=out_file,
    )

    assert len(records) == 4
    assert out_file.exists()

    # Read back from JSONL
    read_back = read_task_records(out_file)
    assert len(read_back) == 4
    assert [r.task_id for r in read_back] == [r.task_id for r in records]

    # Test database adapter directly (returns 0 until models.py implements writer)
    db_res = write_to_db_adapter(records)
    assert isinstance(db_res, int)


def test_cli_all_with_fixtures(tmp_path: Path, fixtures_dir: Path) -> None:
    """CLI python -m oasis.bench.ingest --all runs cleanly with fixture option."""
    out_file = tmp_path / "cli_tasks.jsonl"
    exit_code = main(
        [
            "--all",
            "--use-fixtures",
            "--fixtures-dir",
            str(fixtures_dir),
            "--output",
            str(out_file),
        ]
    )
    assert exit_code == 0
    assert out_file.exists()

    records = read_task_records(out_file)
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 5. Marker for real download (skipped by default)
# ---------------------------------------------------------------------------
@pytest.mark.real_download
def test_real_download_from_huggingface() -> None:
    """Test downloading live datasets from HuggingFace (skipped by default)."""
    config = load_benchmark_config()
    records = ingest_benchmarks(
        domains=["code_generation"],
        config=config,
        count_override=2,
        use_fixtures=False,
    )
    assert len(records) == 2
