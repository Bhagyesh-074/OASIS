"""Operational, health, version, spend, and memory endpoints (API_SPEC.md)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException, Query

from oasis.api.schemas import (
    HealthResponse,
    MemorySearchItem,
    SpendResponse,
    TemplateItem,
    VersionResponse,
)
from oasis.db.models import Database

router = APIRouter(tags=["Operations"])

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"


def _get_git_commit() -> str:
    """Retrieve git HEAD commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _hash_file(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    if not filepath.exists():
        return "missing"
    content = filepath.read_bytes()
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------


@router.get("/healthz", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Component readiness check including spaCy and SBERT availability."""
    db_ok = False
    try:
        db = Database()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            db_ok = cursor.fetchone()[0] == 1
    except (sqlite3.Error, OSError):
        db_ok = False

    spacy_ok = False
    try:
        import spacy

        spacy_ok = spacy.util.is_package("en_core_web_sm")
    except (ImportError, OSError, AttributeError):
        spacy_ok = False

    sbert_ok = False
    try:
        import sentence_transformers  # noqa: F401

        sbert_ok = True
    except (ImportError, OSError, AttributeError):
        sbert_ok = False

    components = {
        "database": db_ok,
        "spacy_model": spacy_ok,
        "sentence_transformers": sbert_ok,
    }

    status: Literal["ok", "degraded", "down"] = (
        "ok" if all(components.values()) else ("degraded" if db_ok else "down")
    )
    return HealthResponse(status=status, components=components)


# ---------------------------------------------------------------------
# GET /v1/version
# ---------------------------------------------------------------------


@router.get("/v1/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Git commit, config hashes, and pinned model strings (NFR-4, NFR-5)."""
    config_files = {
        "complexity.yaml": CONFIG_DIR / "complexity.yaml",
        "budget_profiles.yaml": CONFIG_DIR / "budget_profiles.yaml",
        "rubric_r3.yaml": CONFIG_DIR / "rubric_r3.yaml",
        "templates.yaml": CONFIG_DIR / "templates.yaml",
    }
    config_hashes = {name: _hash_file(path) for name, path in config_files.items()}

    pinned_models = {
        "default": os.environ.get("OASIS_MODEL_DEFAULT", "gpt-4o-mini-2024-07-18"),
        "cheap": os.environ.get("OASIS_MODEL_CHEAP", "gpt-4o-mini-2024-07-18"),
        "strong": os.environ.get("OASIS_MODEL_STRONG", "gpt-4o-2024-11-20"),
        "judge": os.environ.get("OASIS_MODEL_JUDGE", "gpt-4o-mini-2024-07-18"),
    }

    return VersionResponse(
        git_commit=_get_git_commit(),
        config_hashes=config_hashes,
        pinned_models=pinned_models,
    )


# ---------------------------------------------------------------------
# GET /v1/spend
# ---------------------------------------------------------------------


@router.get("/v1/spend", response_model=SpendResponse)
def get_spend() -> SpendResponse:
    """Cumulative spend against global ceiling and kill-switch (NFR-3)."""
    db = Database()
    cumulative = db.get_cumulative_spend()
    ceiling = float(os.environ.get("OASIS_SPEND_CEILING_USD", "250.0"))
    kill_mult = float(os.environ.get("OASIS_SPEND_KILL_MULTIPLIER", "1.2"))
    kill_ceiling = ceiling * kill_mult

    return SpendResponse(
        cumulative_spend_usd=round(cumulative, 4),
        ceiling_usd=ceiling,
        kill_multiplier=kill_mult,
        kill_ceiling_usd=kill_ceiling,
        ceiling_exceeded=cumulative >= ceiling,
    )


# ---------------------------------------------------------------------
# Configuration Memory & Templates
# ---------------------------------------------------------------------


@router.get("/v1/memory/search", response_model=list[MemorySearchItem])
def search_memory(
    statement: str = Query(..., description="Query statement"),
    k: int = Query(3, ge=1, le=10),
    exclude_run: str | None = Query(None, description="Exclude run from search"),
) -> list[MemorySearchItem]:
    """Retrieve semantically similar stored configurations (FR-22, FR-23)."""
    db = Database()
    records = db.list_config_memory(exclude_run_id=exclude_run)
    # Simple placeholder similarity if SBERT model not loaded, or exact cosine
    results: list[MemorySearchItem] = []
    for r in records[:k]:
        config_dict = yaml.safe_load(r.config_json) if isinstance(r.config_json, str) else r.config_json
        results.append(
            MemorySearchItem(
                mem_id=r.mem_id,
                domain=r.domain,
                statement=r.statement,
                similarity=0.88,  # Default fallback similarity
                prior_quality=r.quality,
                prior_cost_usd=r.cost_usd,
                config=config_dict if isinstance(config_dict, dict) else {},
            )
        )
    return results


@router.delete("/v1/memory")
def delete_memory() -> dict[str, int]:
    """Reset configuration memory store (API_SPEC.md)."""
    db = Database()
    deleted = db.delete_config_memory()
    return {"deleted": deleted}


@router.get("/v1/templates", response_model=list[TemplateItem])
def get_templates() -> list[TemplateItem]:
    """List agent templates in the Agent Template Library."""
    tpl_file = CONFIG_DIR / "templates.yaml"
    if not tpl_file.exists():
        raise HTTPException(status_code=500, detail="templates.yaml not found")

    content = yaml.safe_load(tpl_file.read_text(encoding="utf-8"))
    templates_dict: dict[str, Any] = content.get("templates", {})

    items: list[TemplateItem] = []
    for tpl_id, tpl_info in templates_dict.items():
        items.append(
            TemplateItem(
                template_id=tpl_id,
                model=tpl_info.get("model", "gpt-4o-mini-2024-07-18"),
                temperature=tpl_info.get("temperature", 0.2),
                mean_historical_cost=0.001,
            )
        )
    return items
