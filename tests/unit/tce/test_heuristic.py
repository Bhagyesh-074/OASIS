"""Unit tests for heuristic Task Complexity Estimator (FR-1, FR-2, FR-3, FR-24, NFR-1, NFR-5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from oasis.rbe.decision_log_client import clear_decision_log, get_decision_log
from oasis.tce.config import load_complexity_config
from oasis.tce.heuristic import (
    _extract_role_text,
    _extract_subtasks_with_cues,
    _reset_model_cache,
    are_models_available_locally,
    build_dependency_graph,
    compute_dependency_density,
    compute_skill_clusters,
    estimate,
    extract_subtasks,
    get_spacy_nlp,
    similarity,
)
from oasis.tce.types import Estimate, SubScores, WeightedContribution


@pytest.fixture(autouse=True)
def _check_models() -> None:
    """Skip tests if spaCy or SBERT models are not cached locally (no network per NFR-1)."""
    if not are_models_available_locally():
        pytest.skip("Models not available locally (no network per NFR-1 / TESTING.md)")


# ---------------------------------------------------------------------------
# 1. spaCy subtask extraction tests (FR-1, FR-2)
# ---------------------------------------------------------------------------
def test_subtask_extraction_simple_one_sentence() -> None:
    """FR-1, FR-2: A one-sentence simple task must yield exactly 1 subtask."""
    simple_cases = [
        "Write a python script to sort a list of numbers.",
        "Summarize this paper on transformer architectures.",
        "Fix the authentication bug in the login endpoint.",
        "Calculate the Sharpe ratio for the given portfolio.",
        "Draft a polite response to the customer enquiry.",
    ]
    for statement in simple_cases:
        subtasks = extract_subtasks(statement)
        assert len(subtasks) == 1, f"Expected 1 subtask for '{statement}', got: {subtasks}"
        assert subtasks[0] == statement


def test_subtask_extraction_multi_clause_and_ordering() -> None:
    """FR-1, FR-2: Multi-clause, coordinated verbs, and ordering markers split into subtasks."""
    # Ordering markers: first, then, finally
    stmt1 = "First fetch the dataset, then clean the missing values, and finally train a classifier."
    subtasks1 = extract_subtasks(stmt1)
    assert len(subtasks1) == 3
    assert any("fetch" in st.lower() for st in subtasks1)
    assert any("clean" in st.lower() for st in subtasks1)
    assert any("train" in st.lower() for st in subtasks1)

    # Line enumerations
    stmt2 = "1. Parse the input YAML.\n2. Validate against schema.\n3. Save to database."
    subtasks2 = extract_subtasks(stmt2)
    assert len(subtasks2) == 3
    assert subtasks2 == [
        "Parse the input YAML.",
        "Validate against schema.",
        "Save to database.",
    ]


def test_subtask_extraction_empty_string() -> None:
    """FR-1: Empty or whitespace statement returns empty list."""
    assert extract_subtasks("") == []
    assert extract_subtasks("   \n\t  ") == []


# ---------------------------------------------------------------------------
# 2. SBERT embeddings & Agglomerative Clustering (FR-2, FR-3, NFR-5)
# ---------------------------------------------------------------------------
def test_skill_clusters_n1() -> None:
    """FR-2: Single subtask yields 1 cluster without error."""
    config = load_complexity_config()
    assert compute_skill_clusters(["Write a python function"], config=config) == 1
    assert compute_skill_clusters([], config=config) == 1


def test_skill_clusters_diverse_tasks() -> None:
    """FR-2, FR-3: Subtasks requiring distinct skills produce multiple clusters."""
    config = load_complexity_config()
    diverse_subtasks = [
        "Implement a low-level memory allocator in C",
        "Design a promotional flyer for social media in Figma",
        "Perform statistical regression on quarterly financial data",
    ]
    clusters = compute_skill_clusters(diverse_subtasks, config=config)
    assert isinstance(clusters, int)
    assert clusters >= 2


# ---------------------------------------------------------------------------
# 3. NetworkX DiGraph & Dependency Density (FR-2)
# ---------------------------------------------------------------------------
def test_dependency_density_bounds() -> None:
    """FR-2: Dependency density is strictly bounded in [0.0, 1.0], with 0.0 for n <= 1."""
    assert compute_dependency_density([]) == 0.0
    assert compute_dependency_density(["Single task"]) == 0.0

    # Independent tasks with no cues or shared nouns
    density_indep = compute_dependency_density(["Write unit test", "Bake chocolate cake"])
    assert 0.0 <= density_indep <= 1.0

    # Sequential tasks with shared nouns and ordering cues
    seq_tasks = [
        "Fetch customer records from the database",
        "Clean and validate the customer records",
        "Generate a PDF summary from the validated customer records",
    ]
    density_seq = compute_dependency_density(seq_tasks)
    assert 0.0 < density_seq <= 1.0


def test_marker_chained_sequential_edges() -> None:
    """FR-2: A marker-chained 4-step statement yields at least n - 1 edges."""
    statement = "First do A, then do B, after that do C, and finally do D."
    est = estimate(statement)
    n = len(est.subtasks)
    assert n == 4

    pipeline = get_spacy_nlp()
    subtasks, cues = _extract_subtasks_with_cues(statement, nlp=pipeline)
    g = build_dependency_graph(subtasks, nlp=pipeline, ordering_indices=cues)
    assert g.number_of_edges() >= n - 1

    # Dependency density must reflect at least n - 1 edges
    possible_edges = n * (n - 1)
    min_expected_density = (n - 1) / possible_edges
    assert est.subscores.dep_density >= min_expected_density


def test_statement_ordering_cues_sequential_edges() -> None:
    """FR-2: Sequential ordering markers (then, finally) produce sequential edges in multi-clause statements."""
    statement = (
        "Write a parser for CSV files, add unit tests, then document the API and finally "
        "benchmark it against pandas."
    )
    est = estimate(statement)
    assert len(est.subtasks) == 4

    pipeline = get_spacy_nlp()
    subtasks, cues = _extract_subtasks_with_cues(statement, nlp=pipeline)
    g = build_dependency_graph(subtasks, nlp=pipeline, ordering_indices=cues)

    # Ordering marker 'then' links subtask 1 ('add unit tests') to subtask 2 ('document the API')
    assert (1, 2) in g.edges()
    # Ordering marker 'finally' links subtask 2 ('document the API') to subtask 3 ('benchmark it against pandas.')
    assert (2, 3) in g.edges()
    assert g.number_of_edges() >= 2
    assert est.subscores.dep_density >= 2.0 / 12.0


# ---------------------------------------------------------------------------
# 4 & 5. Estimate function & Decision Logging (FR-1, FR-2, FR-3, FR-24, NFR-1)
# ---------------------------------------------------------------------------
def test_estimate_schema_and_contributions_sum() -> None:
    """FR-1, FR-2, FR-3: Estimate structure, bounds, and weighted contribution sum."""
    statement = (
        "First fetch the dataset. Next clean the data. "
        "Then train a model. Finally evaluate accuracy."
    )
    result = estimate(statement, domain="code_generation")

    assert isinstance(result, Estimate)
    assert isinstance(result.mvts, int)
    assert 1 <= result.mvts <= 8

    # Sub-scores
    assert isinstance(result.subscores, SubScores)
    assert isinstance(result.subscores.subtask_count, int)
    assert isinstance(result.subscores.skill_clusters, int)
    assert result.subscores.subtask_count >= 1
    assert result.subscores.skill_clusters >= 1
    assert 0.0 <= result.subscores.dep_density <= 1.0

    # Contributions
    assert isinstance(result.contributions, WeightedContribution)
    c = result.contributions
    contrib_sum = c.subtask_count + c.skill_clusters + c.dep_density

    # Sum of contributions matches composite score mapped via score_to_mvts
    config = load_complexity_config()
    expected_mvts = config.score_to_mvts(contrib_sum)
    assert result.mvts == expected_mvts

    # Latency recorded (NFR-1)
    assert result.latency_ms >= 0.0

    # Justification string (FR-24)
    assert result.justification.startswith(f"MVTS={result.mvts}")
    assert "subtasks=" in result.justification
    assert "skill_clusters=" in result.justification
    assert "dep_density=" in result.justification


def test_estimate_decision_logging() -> None:
    """FR-24: Every estimate writes a decision-log record with component='TCE'."""
    clear_decision_log()

    statement = "Build a REST API endpoint for user profile updates."
    result = estimate(statement, domain="code_generation")

    records = get_decision_log()
    assert len(records) >= 1
    tce_records = [r for r in records if r.component == "TCE"]
    assert len(tce_records) >= 1

    last_record = tce_records[-1]
    assert last_record.decision == "estimate_mvts"
    assert last_record.inputs["statement"] == statement
    assert last_record.inputs["domain"] == "code_generation"
    assert last_record.inputs["mvts"] == result.mvts
    assert last_record.justification == result.justification
    assert len(last_record.justification.strip()) > 0


# ---------------------------------------------------------------------------
# 6. Similarity contract for RBE Team Reducer (rbe/reducer.py contract)
# ---------------------------------------------------------------------------
def test_similarity_contract_strings_and_dicts() -> None:
    """Contract: similarity(role_a, role_b) -> float in [0, 1] accepts str or dict."""
    # Strings
    sim_identical = similarity("Backend Engineer", "Backend Engineer")
    assert sim_identical == 1.0

    # Dicts with 'role' key
    sim_dict = similarity({"role": "Backend Engineer"}, {"role": "Backend Engineer"})
    assert sim_dict == 1.0

    # Mixed str and dict
    sim_mixed = similarity("Backend Engineer", {"role": "Backend Engineer"})
    assert sim_mixed == 1.0

    # Semantic similarity ordering
    sim_related = similarity("Software Engineer", "Python Developer")
    sim_unrelated = similarity("Software Engineer", "Pastry Chef")

    assert 0.0 <= sim_related <= 1.0
    assert 0.0 <= sim_unrelated <= 1.0
    assert sim_related > sim_unrelated

    # Empty inputs return 0.0
    assert similarity("", "") == 0.0
    assert similarity({}, {}) == 0.0


def test_extract_role_text() -> None:
    """Role text extraction handles str, dict with 'role', and dict with 'name'."""
    assert _extract_role_text("Engineer") == "Engineer"
    assert _extract_role_text({"role": "Planner"}) == "Planner"
    assert _extract_role_text({"name": "Coder"}) == "Coder"
    assert _extract_role_text({}) == ""


# ---------------------------------------------------------------------------
# 7. Performance: Module-level caching and latency (NFR-1)
# ---------------------------------------------------------------------------
def test_performance_lazy_loading_and_speedup() -> None:
    """NFR-1: spaCy and SBERT lazy-load once; subsequent call is fast relative to first."""
    _reset_model_cache()

    statement = "Write a python function to compute fibonacci numbers."

    # Call 1: cold start (loads models into cache)
    res1 = estimate(statement)
    cold_latency = res1.latency_ms

    # Call 2: warm call (uses module-level cached singletons)
    res2 = estimate(statement)
    warm_latency = res2.latency_ms

    # Warm call must be faster than cold call, or well within supervisory budget (< 500 ms)
    assert warm_latency < max(cold_latency, 500.0)


# ---------------------------------------------------------------------------
# 8. 20 Fixture statements across 5 domains (FR-1, FR-2, FR-3)
# ---------------------------------------------------------------------------
def test_20_fixture_statements_contract_and_determinism() -> None:
    """FR-1, FR-2, FR-3: 20 fixture statements across 5 domains satisfy schema and determinism."""
    fixtures_file = Path(__file__).resolve().parents[2] / "fixtures" / "tce_statements.yaml"
    assert fixtures_file.is_file(), f"Missing fixtures file: {fixtures_file}"

    data = yaml.safe_load(fixtures_file.read_text(encoding="utf-8"))
    statements = data.get("statements", [])
    assert len(statements) == 20, f"Expected 20 statements, got {len(statements)}"

    domains = {s["domain"] for s in statements}
    expected_domains = {
        "code_generation",
        "research_qa",
        "quant_analysis",
        "support_triage",
        "content_generation",
    }
    assert domains == expected_domains

    for item in statements:
        statement_text = item["statement"]
        domain = item["domain"]

        # Call 1
        est1 = estimate(statement_text, domain=domain)

        # Assert schema and bounds
        assert isinstance(est1.mvts, int)
        assert 1 <= est1.mvts <= 8
        assert isinstance(est1.subscores.subtask_count, int)
        assert isinstance(est1.subscores.skill_clusters, int)
        assert est1.subscores.subtask_count >= 1
        assert est1.subscores.skill_clusters >= 1
        assert 0.0 <= est1.subscores.dep_density <= 1.0

        # Contributions present and sum to composite score
        contrib_sum = (
            est1.contributions.subtask_count
            + est1.contributions.skill_clusters
            + est1.contributions.dep_density
        )
        config = load_complexity_config()
        assert est1.mvts == config.score_to_mvts(contrib_sum)

        # Determinism: same input gives exact same output
        est2 = estimate(statement_text, domain=domain)
        assert est1.mvts == est2.mvts
        assert est1.subscores == est2.subscores
        assert est1.contributions == est2.contributions
        assert est1.subtasks == est2.subtasks


def test_domain_monotonicity() -> None:
    """FR-3: A clearly more complex statement scores >= a simpler one in the same domain."""
    fixtures_file = Path(__file__).resolve().parents[2] / "fixtures" / "tce_statements.yaml"
    data = yaml.safe_load(fixtures_file.read_text(encoding="utf-8"))
    statements: list[dict[str, Any]] = data["statements"]

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for item in statements:
        by_domain.setdefault(item["domain"], []).append(item)

    for domain, items in by_domain.items():
        # Sort items by rank (1: low, 2: medium, 3: medium_high, 4: high)
        sorted_items = sorted(items, key=lambda x: int(x.get("rank", 0)))
        scores: list[float] = []

        for item in sorted_items:
            est = estimate(item["statement"], domain=domain)
            score = (
                est.contributions.subtask_count
                + est.contributions.skill_clusters
                + est.contributions.dep_density
            )
            scores.append(score)

        # Monotonicity: each subsequent level has score >= previous level
        for idx in range(len(scores) - 1):
            s_curr, s_next = scores[idx], scores[idx + 1]
            level_curr = sorted_items[idx]["complexity_level"]
            level_next = sorted_items[idx + 1]["complexity_level"]
            assert s_next >= s_curr - 1e-6, (
                f"Monotonicity violation in domain '{domain}': "
                f"level '{level_next}' ({s_next:.4f}) < level '{level_curr}' ({s_curr:.4f})"
            )
