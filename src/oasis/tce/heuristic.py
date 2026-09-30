"""Heuristic Task Complexity Estimator (FR-1, FR-2, FR-3, FR-24, NFR-1, NFR-5).

Decomposes problem statements into candidate subtasks using spaCy linguistic parsing,
computes skill-diversity clusters via agglomerative clustering of SBERT embeddings,
constructs a NetworkX dependency graph over ordering cues and shared entities,
and maps the normalized weighted composite score to an integer Minimum Viable Team Size (MVTS).
Also exports semantic role similarity for RBE team reduction (rbe/reducer.py contract).
"""

from __future__ import annotations

import re
import time
from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]

from oasis.tce._log import log_tce_decision
from oasis.tce.config import ComplexityConfig, load_complexity_config
from oasis.tce.types import Estimate, SubScores, WeightedContribution

# ---------------------------------------------------------------------------
# Module-level model caches for zero-overhead repeated calls (NFR-1)
# ---------------------------------------------------------------------------
_SPACY_NLP: Any | None = None
_SBERT_MODEL: Any | None = None

ORDERING_WORDS = frozenset(
    {
        "then",
        "finally",
        "afterwards",
        "next",
        "subsequently",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
    }
)
ORDERING_PHRASES = ("after that", "following that", "once done", "once completed")
PRONOUN_LEMMAS = frozenset(
    {"it", "its", "they", "them", "their", "theirs", "this", "these", "those"}
)

ORDERING_CUE_REGEX = re.compile(
    r"^(?:(?:and|also)\s+)*(?:then|finally|afterwards|subsequently|first|second|third|fourth|fifth|after\s+that|following\s+that|once\s+done|once\s+completed|(?:next\b(?!\s+(?:step|steps|phase|phases|iteration|iterations|release|releases|task|tasks|action|actions|part|parts|stage|stages))))\b",
    re.IGNORECASE,
)
ENUM_REGEX = re.compile(
    r"^(?:(?:\d+[\.\)]|\([0-9a-zA-Z]+\)|[-*•])\s+|(?:first|second|third|fourth|fifth)[,:]\s*)",
    re.IGNORECASE,
)


def get_spacy_nlp(model_name: str = "en_core_web_sm") -> Any:
    """Lazy-load and cache the spaCy English pipeline (NFR-1).

    Parameters
    ----------
    model_name:
        spaCy pipeline name. Defaults to "en_core_web_sm".

    Returns
    -------
    spacy.language.Language
        Cached spaCy Language pipeline.

    Raises
    ------
    RuntimeError
        If the model is not installed or available locally (offline per NFR-1).
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"spaCy model {model_name!r} not available locally. "
                "Download or install the pipeline offline per NFR-1 / TESTING.md."
            ) from exc
    return _SPACY_NLP


def get_sbert_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Lazy-load and cache the SBERT embedding model (NFR-1, NFR-5).

    Enforces local_files_only to guarantee zero network calls at test/eval time.

    Parameters
    ----------
    model_name:
        Pinned SBERT model identifier from configuration (NFR-5).

    Returns
    -------
    SentenceTransformer
        Cached SentenceTransformer model instance.

    Raises
    ------
    RuntimeError
        If the model weights are not cached locally on disk.
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            _SBERT_MODEL = SentenceTransformer(model_name, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"SBERT model {model_name!r} not available locally. "
                "Ensure model weights are cached on disk per NFR-1 / TESTING.md."
            ) from exc
    return _SBERT_MODEL


def _reset_model_cache() -> None:
    """Reset module-level model singletons (internal helper for latency / lazy-load tests)."""
    global _SPACY_NLP, _SBERT_MODEL
    _SPACY_NLP = None
    _SBERT_MODEL = None


def are_models_available_locally(config: ComplexityConfig | None = None) -> bool:
    """Check whether required spaCy and SBERT models can be loaded locally without network."""
    try:
        get_spacy_nlp()
        cfg = config or load_complexity_config()
        get_sbert_model(cfg.sbert_model)
        return True
    except (RuntimeError, OSError, ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 1. spaCy subtask extraction from clauses and imperatives (FR-1, FR-2)
# ---------------------------------------------------------------------------
def _extract_subtasks_with_cues(
    statement: str, nlp: Any | None = None
) -> tuple[list[str], set[int]]:
    """Internal helper to extract candidate subtasks and identify ordering cue indices.

    Returns
    -------
    tuple[list[str], set[int]]
        (cleaned_subtasks, ordering_indices)
        where ordering_indices contains 0-based subtask indices j (j > 0) that were
        introduced by explicit ordering markers (e.g. 'then', 'finally', 'after that')
        or line enumerations.
    """
    clean_text = statement.strip()
    if not clean_text:
        return [], set()

    # Check for line-by-line enumerations
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    enumerated = [line for line in lines if ENUM_REGEX.match(line)]
    if len(enumerated) >= 2 and len(enumerated) == len(lines):
        clean_subs = [
            ENUM_REGEX.sub("", line).strip()
            for line in lines
            if ENUM_REGEX.sub("", line).strip()
        ]
        # In enumerated lists, steps 1..n-1 sequentially follow previous step
        ord_indices = set(range(1, len(clean_subs)))
        return clean_subs, ord_indices

    pipeline = nlp or get_spacy_nlp()
    doc = pipeline(clean_text)
    raw_subtasks: list[str] = []

    for sent in doc.sents:
        tokens = list(sent)
        split_points: set[int] = set()

        for i, token in enumerate(tokens):
            if i == 0:
                continue

            # Check ordering markers introducing a clause
            is_ordering = False
            if token.text.lower() in ORDERING_WORDS and not (
                token.dep_ in ("amod", "compound") or token.pos_ in ("ADJ", "NOUN")
            ):
                is_ordering = True
            if not is_ordering and i + 1 < len(tokens):
                two_word = f"{token.text.lower()} {tokens[i + 1].text.lower()}"
                if two_word in ORDERING_PHRASES:
                    is_ordering = True

            # Check coordinated verbs with their own dobj where head verb also has dobj
            # or imperative clause following a comma
            token_has_dobj = any(c.dep_ in ("dobj", "pobj", "ccomp") for c in token.children)
            is_coord_verb = False
            if token.pos_ in ("VERB", "AUX") and token_has_dobj and (
                token.dep_ in ("conj", "dep", "parataxis")
                or (token.dep_ == "ROOT" and i > 0 and tokens[i - 1].text in (",", ";"))
            ):
                is_coord_verb = True

            if is_ordering or is_coord_verb:
                # Walk back to conjunction or punctuation to identify clause boundary
                back = i - 1
                while back >= 0 and tokens[back].text.lower() in (
                    "and",
                    "then",
                    "also",
                    "finally",
                    ",",
                    ";",
                ):
                    back -= 1
                split_token_idx = back + 1
                split_points.add(tokens[split_token_idx].idx - sent.start_char)

        if not split_points:
            raw_subtasks.append(sent.text.strip())
        else:
            sorted_splits = sorted(split_points)
            last_pos = 0
            sent_text = sent.text
            for pos in sorted_splits:
                chunk = sent_text[last_pos:pos].strip(" ,;\t\n")
                if chunk:
                    raw_subtasks.append(chunk)
                last_pos = pos
            if last_pos < len(sent_text):
                chunk = sent_text[last_pos:].strip(" ,;\t\n")
                if chunk:
                    raw_subtasks.append(chunk)

    cleaned_subtasks: list[str] = []
    ordering_indices: set[int] = set()

    for idx, raw_st in enumerate(raw_subtasks):
        if idx > 0 and (ORDERING_CUE_REGEX.search(raw_st) or ENUM_REGEX.search(raw_st)):
            ordering_indices.add(len(cleaned_subtasks))

        st_clean = ENUM_REGEX.sub("", raw_st)
        st_clean = re.sub(
            r"^(?:and\s+|then\s+|finally\s+|also\s+|after\s+that\s+)+",
            "",
            st_clean,
            flags=re.IGNORECASE,
        ).strip()
        if st_clean and len(st_clean) >= 3:
            cleaned_subtasks.append(st_clean)

    return (cleaned_subtasks if cleaned_subtasks else [clean_text]), ordering_indices


def extract_subtasks(statement: str, nlp: Any | None = None) -> list[str]:
    """Split a problem statement into candidate subtasks from clauses and imperatives (FR-1, FR-2).

    Detects:
    - Line-by-line enumerations (1., 2., -, etc.)
    - Ordering markers (then, after that, finally, next)
    - Coordinated imperative verbs with distinct direct objects (dobj)
    - Sentence boundaries in multi-sentence instructions

    A simple one-sentence task yields exactly 1 subtask.

    Parameters
    ----------
    statement:
        Raw problem statement text.
    nlp:
        Optional pre-loaded spaCy Language instance.

    Returns
    -------
    list[str]
        List of candidate subtask text segments.
    """
    subtasks, _ = _extract_subtasks_with_cues(statement, nlp=nlp)
    return subtasks


# ---------------------------------------------------------------------------
# 2. SBERT embeddings & Agglomerative Clustering (FR-2, FR-3, NFR-5)
# ---------------------------------------------------------------------------
def compute_skill_clusters(
    subtasks: list[str],
    config: ComplexityConfig,
    embedder: Any | None = None,
) -> int:
    """Compute skill-diversity cluster count via Agglomerative Clustering (FR-2, FR-3).

    Handles n=1 (returns 1) without error. Uses the SBERT model named in config
    and cluster_distance_threshold from config (FR-3, NFR-5).

    Parameters
    ----------
    subtasks:
        List of subtask strings.
    config:
        Validated ComplexityConfig.
    embedder:
        Optional pre-loaded SentenceTransformer instance.

    Returns
    -------
    int
        Number of distinct skill clusters.
    """
    if len(subtasks) <= 1:
        return 1

    model = embedder or get_sbert_model(config.sbert_model)
    embeddings = model.encode(subtasks, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config.cluster_distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return len(set(labels))


# ---------------------------------------------------------------------------
# 3. NetworkX DiGraph & Dependency Density (FR-2)
# ---------------------------------------------------------------------------
def build_dependency_graph(
    subtasks: list[str],
    nlp: Any | None = None,
    ordering_indices: set[int] | None = None,
) -> nx.DiGraph:
    """Construct a NetworkX DiGraph over subtasks from ordering cues and references (FR-2).

    Edges are added from:
    1. Explicit ordering cues (e.g. 'then', 'finally', 'after that') linking sequential steps.
    2. Shared entities and nouns between step i and later step j.
    3. Anaphoric pronoun references ('it', 'them', 'this') referring back to earlier steps.

    Parameters
    ----------
    subtasks:
        List of candidate subtask text segments.
    nlp:
        Optional pre-loaded spaCy Language pipeline.
    ordering_indices:
        Optional set of 0-based indices for subtasks introduced by explicit ordering cues.

    Returns
    -------
    nx.DiGraph
        Directed graph with nodes 0..n-1 and dependency edges.
    """
    n = len(subtasks)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    if n <= 1:
        return g

    pipeline = nlp or get_spacy_nlp()
    docs = [pipeline(st) for st in subtasks]

    # Extract terms (nouns, proper nouns, entities) per subtask
    subtask_terms: list[set[str]] = []
    for doc in docs:
        nouns = {
            t.lemma_.lower()
            for t in doc
            if t.pos_ in ("NOUN", "PROPN") and len(t.lemma_) > 2
        }
        ents = {e.text.lower() for e in doc.ents}
        subtask_terms.append(nouns | ents)

    ord_idx = ordering_indices or set()

    for j in range(n):
        doc_j = docs[j]
        tokens_j = [t.text.lower() for t in doc_j]

        # 1. Ordering cues linking immediately preceding subtask
        is_ord = j in ord_idx
        if not is_ord:
            for w in tokens_j[:2]:
                if w in ORDERING_WORDS:
                    if w == "next":
                        next_tokens = [t for t in doc_j if t.text.lower() == "next"]
                        if next_tokens and next_tokens[0].pos_ == "ADJ":
                            continue
                    is_ord = True
                    break
            if not is_ord and any(subtasks[j].lower().startswith(p) for p in ORDERING_PHRASES):
                is_ord = True

        if j > 0 and is_ord:
            g.add_edge(j - 1, j)

        # 2. Pronoun references linking immediately preceding subtask
        has_pronoun = any(
            (t.lemma_.lower() in PRONOUN_LEMMAS or t.text.lower() in PRONOUN_LEMMAS)
            for t in doc_j
            if t.pos_ == "PRON"
        )
        if j > 0 and has_pronoun:
            g.add_edge(j - 1, j)

        # 3. Shared entities/nouns with earlier subtasks (data dependency i -> j)
        for i in range(j):
            if subtask_terms[i] & subtask_terms[j]:
                g.add_edge(i, j)

    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def compute_dependency_density(
    subtasks: list[str],
    nlp: Any | None = None,
    ordering_indices: set[int] | None = None,
) -> float:
    """Construct a NetworkX DiGraph over subtasks and compute edge density (FR-2).

    Edges are added from:
    1. Explicit ordering cues (e.g. 'then', 'finally', 'after that') linking sequential steps.
    2. Shared entities and nouns between step i and later step j.
    3. Anaphoric pronoun references ('it', 'them', 'this') referring back to earlier steps.

    Density is defined as: edges / (n * (n - 1)); returns 0.0 for n <= 1.

    Parameters
    ----------
    subtasks:
        List of candidate subtask text segments.
    nlp:
        Optional pre-loaded spaCy Language pipeline.
    ordering_indices:
        Optional set of 0-based indices for subtasks introduced by explicit ordering cues.

    Returns
    -------
    float
        Dependency graph edge density bounded in [0.0, 1.0].
    """
    n = len(subtasks)
    if n <= 1:
        return 0.0

    g = build_dependency_graph(subtasks, nlp=nlp, ordering_indices=ordering_indices)
    possible_edges = n * (n - 1)
    density = float(g.number_of_edges() / possible_edges) if possible_edges > 0 else 0.0
    return max(0.0, min(1.0, density))


# ---------------------------------------------------------------------------
# 4 & 5. Main estimate function with decision logging (FR-1, FR-2, FR-3, FR-24, NFR-1)
# ---------------------------------------------------------------------------
def estimate(
    statement: str,
    domain: str | None = None,
    config: ComplexityConfig | None = None,
) -> Estimate:
    """Produce an integer MVTS estimate and sub-score breakdown (FR-1, FR-2, FR-3, FR-24, NFR-1).

    Executes:
    1. spaCy clause and imperative extraction into candidate subtasks.
    2. SBERT embeddings and Agglomerative Clustering for skill clusters.
    3. NetworkX DiGraph for dependency density.
    4. Sub-score normalization using bounds loaded from config (FR-3).
    5. Weighted contribution calculation and monotonic MVTS mapping (FR-1, FR-2).
    6. Decision logging to decision_log audit seam (FR-24).
    7. Supervisory latency measurement in milliseconds (NFR-1).

    Parameters
    ----------
    statement:
        Problem statement or task description.
    domain:
        Optional task domain identifier (e.g., 'code_generation', 'research_qa').
    config:
        Optional pre-loaded ComplexityConfig. Defaults to repository complexity.yaml.

    Returns
    -------
    Estimate
        Validated Estimate response matching POST /v1/estimate in docs/API_SPEC.md.
    """
    start_time = time.perf_counter()
    cfg = config or load_complexity_config()
    pipeline = get_spacy_nlp()

    # 1. Candidate subtasks (FR-1, FR-2)
    subtasks, ordering_indices = _extract_subtasks_with_cues(statement, nlp=pipeline)
    subtask_count = len(subtasks)

    # 2. Skill clusters (FR-2, FR-3, NFR-5)
    skill_clusters = compute_skill_clusters(subtasks, config=cfg)

    # 3. Dependency graph density (FR-2)
    dep_density = compute_dependency_density(
        subtasks, nlp=pipeline, ordering_indices=ordering_indices
    )

    # 4. Documented normalization using constants loaded from config (FR-3)
    norm_cfg = cfg.normalization
    subtask_range = max(1, norm_cfg.max_subtasks - norm_cfg.min_subtasks)
    norm_subtasks = max(0.0, min(1.0, (float(subtask_count) - norm_cfg.min_subtasks) / subtask_range))

    cluster_range = max(1, norm_cfg.max_skill_clusters - norm_cfg.min_skill_clusters)
    norm_clusters = max(0.0, min(1.0, (float(skill_clusters) - norm_cfg.min_skill_clusters) / cluster_range))

    norm_density = max(0.0, min(1.0, dep_density))

    # Compute weighted contributions (FR-2)
    contrib_subtasks = round(cfg.weights.subtask_count * norm_subtasks, 6)
    contrib_clusters = round(cfg.weights.skill_clusters * norm_clusters, 6)
    contrib_density = round(cfg.weights.dep_density * norm_density, 6)

    # Weighted sum
    composite_score = contrib_subtasks + contrib_clusters + contrib_density

    # Map to integer MVTS in range [min, max] (FR-1, FR-3)
    mvts = cfg.score_to_mvts(composite_score)

    subscores = SubScores(
        subtask_count=subtask_count,
        skill_clusters=skill_clusters,
        dep_density=dep_density,
    )
    contributions = WeightedContribution(
        subtask_count=contrib_subtasks,
        skill_clusters=contrib_clusters,
        dep_density=contrib_density,
    )

    # 5. Justification string and audit logging (FR-24)
    justification = (
        f"MVTS={mvts} from subtasks={subtask_count}, "
        f"skill_clusters={skill_clusters}, "
        f"dep_density={dep_density:.2f}"
    )

    log_inputs: dict[str, Any] = {
        "statement": statement,
        "subscores": subscores.model_dump(),
        "weights": cfg.weights.to_dict(),
        "contributions": contributions.model_dump(),
        "composite_score": composite_score,
        "mvts": mvts,
    }
    if domain is not None:
        log_inputs["domain"] = domain

    log_tce_decision(
        decision="estimate_mvts",
        inputs=log_inputs,
        justification=justification,
    )

    # 7. Supervisory latency in ms (NFR-1)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return Estimate(
        mvts=mvts,
        subscores=subscores,
        weights=cfg.weights.to_dict(),
        contributions=contributions,
        subtasks=subtasks,
        justification=justification,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# 6. Role semantic similarity for RBE Team Reducer contract
# ---------------------------------------------------------------------------
def _extract_role_text(role: dict[str, Any] | str) -> str:
    """Extract role string from string or role dict."""
    if isinstance(role, dict):
        text = str(role.get("role", "") or role.get("name", "") or "")
        return text.strip()
    return str(role).strip()


def similarity(
    role_a: dict[str, Any] | str,
    role_b: dict[str, Any] | str,
    config: ComplexityConfig | None = None,
) -> float:
    """Compute semantic similarity in [0.0, 1.0] between two roles (rbe/reducer contract).

    Matches the pluggable pairwise similarity contract expected by TeamReducer
    in src/oasis/rbe/reducer.py.

    Parameters
    ----------
    role_a, role_b:
        Role specification dicts (with 'role' or 'name' key) or role name strings.
    config:
        Optional ComplexityConfig for model identifier (NFR-5).

    Returns
    -------
    float
        Cosine similarity score clamped to [0.0, 1.0].
    """
    text_a = _extract_role_text(role_a)
    text_b = _extract_role_text(role_b)

    if not text_a or not text_b:
        return 0.0
    if text_a.lower() == text_b.lower():
        return 1.0

    cfg = config or load_complexity_config()
    embedder = get_sbert_model(cfg.sbert_model)

    embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
    cos_sim = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, cos_sim))
