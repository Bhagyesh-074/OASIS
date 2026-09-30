"""Unit tests for TCE configuration loader (FR-1, FR-3, NFR-4, NFR-5)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from oasis.tce.config import (
    ComplexityConfig,
    WeightsConfig,
    compute_complexity_config_hash,
    load_complexity_config,
)


def test_load_default_config() -> None:
    """FR-3, NFR-4, NFR-5: Default complexity.yaml loads and validates successfully."""
    config = load_complexity_config()

    assert isinstance(config, ComplexityConfig)
    assert config.version == "0.1.0-uncalibrated"
    assert config.weights.subtask_count == 0.4
    assert config.weights.skill_clusters == 0.4
    assert config.weights.dep_density == 0.2
    assert config.mvts_range == (1, 8)
    assert config.sbert_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert "latest" not in config.sbert_model.lower()
    assert config.cluster_distance_threshold == 0.35

    # Check thresholds cover 1..8 and are monotonic
    assert sorted(config.thresholds.keys()) == list(range(1, 9))
    for k in range(1, 8):
        assert config.thresholds[k] < config.thresholds[k + 1]

    # Check sha256 is valid 64-char hex string
    assert len(config.sha256) == 64
    assert all(c in "0123456789abcdef" for c in config.sha256)


def test_hash_stability(tmp_path: Path) -> None:
    """NFR-4: Config SHA-256 is stable across calls and changes when content changes."""
    hash1 = compute_complexity_config_hash()
    hash2 = compute_complexity_config_hash()
    config = load_complexity_config()

    assert hash1 == hash2
    assert hash1 == config.sha256

    # Verify that modified file produces different hash
    test_yaml = tmp_path / "test_complexity.yaml"
    content = {
        "version": "test-v1",
        "weights": {"subtask_count": 0.5, "skill_clusters": 0.3, "dep_density": 0.2},
        "mvts_range": [1, 8],
        "thresholds": {k: float(k) * 0.1 for k in range(1, 9)},
        "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "cluster_distance_threshold": 0.35,
    }
    raw_text = yaml.dump(content)
    raw_bytes = raw_text.encode("utf-8")
    test_yaml.write_bytes(raw_bytes)

    expected_hash = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()
    assert compute_complexity_config_hash(test_yaml) == expected_hash

    cfg = load_complexity_config(test_yaml)
    assert cfg.sha256 == expected_hash


def test_crlf_and_lf_produce_same_hash(tmp_path: Path) -> None:
    """NFR-4: CRLF and LF copies of identical YAML produce identical SHA-256 hashes."""
    base_content = (
        "version: '0.1.0-test'\n"
        "weights:\n"
        "  subtask_count: 0.4\n"
        "  skill_clusters: 0.4\n"
        "  dep_density: 0.2\n"
        "mvts_range: [1, 8]\n"
        "thresholds:\n"
        "  1: 0.0\n"
        "  2: 0.15\n"
        "  3: 0.3\n"
        "  4: 0.45\n"
        "  5: 0.6\n"
        "  6: 0.7\n"
        "  7: 0.8\n"
        "  8: 0.9\n"
        "sbert_model: 'sentence-transformers/all-MiniLM-L6-v2'\n"
        "cluster_distance_threshold: 0.35\n"
    )
    lf_bytes = base_content.encode("utf-8")
    crlf_bytes = base_content.replace("\n", "\r\n").encode("utf-8")

    assert b"\r\n" in crlf_bytes
    assert b"\r\n" not in lf_bytes

    lf_file = tmp_path / "config_lf.yaml"
    crlf_file = tmp_path / "config_crlf.yaml"

    lf_file.write_bytes(lf_bytes)
    crlf_file.write_bytes(crlf_bytes)

    hash_lf = compute_complexity_config_hash(lf_file)
    hash_crlf = compute_complexity_config_hash(crlf_file)

    assert hash_lf == hash_crlf
    assert hash_lf == hashlib.sha256(lf_bytes).hexdigest()

    cfg_lf = load_complexity_config(lf_file)
    cfg_crlf = load_complexity_config(crlf_file)
    assert cfg_lf.sha256 == hash_lf
    assert cfg_crlf.sha256 == hash_crlf


def test_changing_weight_changes_hash(tmp_path: Path) -> None:
    """NFR-4: Changing one weight in YAML configuration changes the SHA-256 hash."""
    base_yaml = (
        "version: '0.1.0-test'\n"
        "weights:\n"
        "  subtask_count: 0.4\n"
        "  skill_clusters: 0.4\n"
        "  dep_density: 0.2\n"
        "mvts_range: [1, 8]\n"
        "thresholds:\n"
        "  1: 0.0\n"
        "  2: 0.15\n"
        "  3: 0.3\n"
        "  4: 0.45\n"
        "  5: 0.6\n"
        "  6: 0.7\n"
        "  7: 0.8\n"
        "  8: 0.9\n"
        "sbert_model: 'sentence-transformers/all-MiniLM-L6-v2'\n"
        "cluster_distance_threshold: 0.35\n"
    )
    # Change weights while keeping sum = 1.0
    modified_yaml = base_yaml.replace("subtask_count: 0.4", "subtask_count: 0.5").replace(
        "skill_clusters: 0.4", "skill_clusters: 0.3"
    )

    base_file = tmp_path / "base.yaml"
    mod_file = tmp_path / "modified.yaml"

    base_file.write_text(base_yaml, encoding="utf-8")
    mod_file.write_text(modified_yaml, encoding="utf-8")

    hash_base = compute_complexity_config_hash(base_file)
    hash_mod = compute_complexity_config_hash(mod_file)

    assert hash_base != hash_mod

    cfg_base = load_complexity_config(base_file)
    cfg_mod = load_complexity_config(mod_file)
    assert cfg_base.sha256 != cfg_mod.sha256
    assert cfg_base.sha256 == hash_base
    assert cfg_mod.sha256 == hash_mod


def test_weights_validation_negative() -> None:
    """FR-3: Negative weights are rejected."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        WeightsConfig(subtask_count=-0.1, skill_clusters=0.5, dep_density=0.6)


def test_weights_validation_sum() -> None:
    """FR-3: Weights not summing to 1.0 are rejected."""
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        WeightsConfig(subtask_count=0.4, skill_clusters=0.4, dep_density=0.3)

    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        WeightsConfig(subtask_count=0.3, skill_clusters=0.3, dep_density=0.3)


def test_weights_to_dict() -> None:
    """FR-2, FR-3: WeightsConfig converts to standard dict."""
    w = WeightsConfig(subtask_count=0.4, skill_clusters=0.4, dep_density=0.2)
    d = w.to_dict()
    assert d == {"subtask_count": 0.4, "skill_clusters": 0.4, "dep_density": 0.2}


def test_mvts_range_validation() -> None:
    """FR-1: mvts_range must be within [1, 8] with min <= max."""
    weights = WeightsConfig(subtask_count=0.4, skill_clusters=0.4, dep_density=0.2)

    # min < 1 rejected
    with pytest.raises(ValueError, match="1 <= min <= max <= 8"):
        ComplexityConfig(
            version="v1",
            weights=weights,
            sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster_distance_threshold=0.35,
            mvts_range=(0, 8),
            thresholds={k: float(k) for k in range(9)},
        )

    # max > 8 rejected
    with pytest.raises(ValueError, match="1 <= min <= max <= 8"):
        ComplexityConfig(
            version="v1",
            weights=weights,
            sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster_distance_threshold=0.35,
            mvts_range=(1, 9),
            thresholds={k: float(k) for k in range(1, 10)},
        )

    # inverted range rejected
    with pytest.raises(ValueError, match="1 <= min <= max <= 8"):
        ComplexityConfig(
            version="v1",
            weights=weights,
            sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster_distance_threshold=0.35,
            mvts_range=(6, 3),
            thresholds={k: float(k) for k in range(3, 7)},
        )


def test_thresholds_monotonicity() -> None:
    """FR-3: Thresholds must be strictly monotonic increasing."""
    weights = WeightsConfig(subtask_count=0.4, skill_clusters=0.4, dep_density=0.2)
    non_monotonic_thresholds = {
        1: 0.0,
        2: 0.2,
        3: 0.15,  # decrease!
        4: 0.4,
        5: 0.5,
        6: 0.6,
        7: 0.7,
        8: 0.8,
    }
    with pytest.raises(ValueError, match="Thresholds must be strictly monotonic increasing"):
        ComplexityConfig(
            version="v1",
            weights=weights,
            mvts_range=(1, 8),
            thresholds=non_monotonic_thresholds,
            sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster_distance_threshold=0.35,
        )


def test_thresholds_keys_cover_mvts_range() -> None:
    """FR-3: Thresholds keys must cover all integers in mvts_range."""
    weights = WeightsConfig(subtask_count=0.4, skill_clusters=0.4, dep_density=0.2)
    incomplete_thresholds = {
        1: 0.0,
        2: 0.2,
        3: 0.4,
        # missing 4..8
    }
    with pytest.raises(ValueError, match="Thresholds keys must cover all team sizes"):
        ComplexityConfig(
            version="v1",
            weights=weights,
            mvts_range=(1, 8),
            thresholds=incomplete_thresholds,
            sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster_distance_threshold=0.35,
        )


def test_sbert_model_pinned_rejection() -> None:
    """NFR-5: Model alias 'latest' must be rejected."""
    weights = WeightsConfig(subtask_count=0.4, skill_clusters=0.4, dep_density=0.2)
    with pytest.raises(ValueError, match="NFR-5 violation"):
        ComplexityConfig(
            version="v1",
            weights=weights,
            mvts_range=(1, 8),
            thresholds={k: float(k) * 0.1 for k in range(1, 9)},
            sbert_model="all-MiniLM-latest",
            cluster_distance_threshold=0.35,
        )


def test_score_to_mvts_mapping() -> None:
    """FR-1, FR-3: Mapping score to integer MVTS via thresholds and clamping."""
    config = load_complexity_config()

    # Below floor -> clamped to min (1)
    assert config.score_to_mvts(-0.5) == 1
    assert config.score_to_mvts(0.0) == 1
    assert config.score_to_mvts(0.14) == 1

    # Exactly at thresholds
    assert config.score_to_mvts(0.15) == 2
    assert config.score_to_mvts(0.30) == 3
    assert config.score_to_mvts(0.42) == 3
    assert config.score_to_mvts(0.45) == 4
    assert config.score_to_mvts(0.60) == 5
    assert config.score_to_mvts(0.70) == 6
    assert config.score_to_mvts(0.80) == 7
    assert config.score_to_mvts(0.90) == 8

    # Above ceiling -> clamped to max (8)
    assert config.score_to_mvts(1.5) == 8


def test_config_frozen_immutability() -> None:
    """ComplexityConfig is frozen and immutable."""
    from pydantic import ValidationError

    config = load_complexity_config()
    with pytest.raises(ValidationError):
        config.version = "mutated"  # type: ignore[misc]


def test_load_config_file_not_found() -> None:
    """FR-3: FileNotFoundError when config file does not exist."""
    with pytest.raises(FileNotFoundError, match="TCE complexity config not found"):
        load_complexity_config("nonexistent/path/to/complexity.yaml")


def test_load_config_missing_required_keys(tmp_path: Path) -> None:
    """FR-3: Clear error when required keys are missing in YAML."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("version: 'v1'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required configuration key"):
        load_complexity_config(bad_yaml)
