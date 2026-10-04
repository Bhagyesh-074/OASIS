"""Contract tests for OASIS API endpoints against API_SPEC.md.

Guards:
- X-API-Key static header authentication.
- Unified error envelope on all failures.
- FR-5: Budget strictly accepts only four dimensions; rejects gpu or ram.
- 202 Accepted on POST /v1/runs.
- GET /v1/runs/{id} schema compliance.
- GET /v1/spend, /v1/version, /v1/templates, /healthz.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oasis.api.app import create_app
from oasis.db.migrate import apply_migrations

AUTH_HEADER = {"X-API-Key": "dev-local-key"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Fixture providing a TestClient backed by a fresh isolated database."""
    test_db = tmp_path / "api_test.db"
    os.environ["OASIS_DB_PATH"] = str(test_db)
    os.environ["OASIS_API_KEY"] = "dev-local-key"
    apply_migrations(test_db)
    app = create_app()
    return TestClient(app)


class TestApiAuthContract:
    """Authentication and error envelope tests."""

    def test_missing_api_key_returns_401_with_error_envelope(self, client: TestClient) -> None:
        """Unauthenticated requests receive 401 with standard error envelope."""
        resp = client.get("/v1/spend")
        assert resp.status_code == 401
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "authentication_required"

    def test_healthz_exempt_from_auth(self, client: TestClient) -> None:
        """GET /healthz does not require X-API-Key."""
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "components" in data


class TestBudgetAndPlanningContract:
    """Budget validation and team planning tests."""

    def test_budget_rejects_gpu_or_ram_negative_fr5(self, client: TestClient) -> None:
        """FR-5 / ADR-002: Unknown dimensions (gpu, ram) are rejected with 422 validation_error."""
        payload = {
            "statement": "Write a python function",
            "domain": "code_generation",
            "budget": {
                "max_tokens": 10000,
                "max_cost_usd": 0.05,
                "max_wall_seconds": 60,
                "max_calls": 5,
                "gpu": "A100",  # FORBIDDEN
            },
        }
        resp = client.post("/v1/plan", json=payload, headers=AUTH_HEADER)
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "validation_error"

    def test_estimate_endpoint(self, client: TestClient) -> None:
        """POST /v1/estimate returns MVTS, subscores, and justification (FR-1, FR-2)."""
        payload = {
            "statement": "Summarize three papers on multi-agent systems and generate a comparison table.",
            "domain": "research_qa",
        }
        resp = client.post("/v1/estimate", json=payload, headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "mvts" in data
        assert 1 <= data["mvts"] <= 8
        assert "subscores" in data
        assert "subtask_count" in data["subscores"]
        assert "justification" in data

    def test_plan_endpoint(self, client: TestClient) -> None:
        """POST /v1/plan returns configured team and enforcement projection."""
        payload = {
            "statement": "Parse log files and produce incident report",
            "domain": "support_triage",
            "budget": {
                "max_tokens": 50000,
                "max_cost_usd": 0.20,
                "max_wall_seconds": 120,
                "max_calls": 20,
            },
        }
        resp = client.post("/v1/plan", json=payload, headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "enforcement" in data
        assert data["enforcement"]["feasible"] is True


class TestRunLifecycleContract:
    """Run initiation, retrieval, decisions, and cancellation."""

    def test_create_run_returns_202_accepted(self, client: TestClient) -> None:
        """POST /v1/runs returns 202 with run_id and events_url."""
        payload = {
            "statement": "Solve arithmetic reasoning",
            "domain": "quant_analysis",
            "budget": {
                "max_tokens": 20000,
                "max_cost_usd": 0.10,
                "max_wall_seconds": 60,
                "max_calls": 10,
            },
            "arm": "full",
            "framework": "langgraph",
            "seed": 42,
        }
        resp = client.post("/v1/runs", json=payload, headers=AUTH_HEADER)
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "queued"
        assert data["events_url"] == f"/v1/runs/{data['run_id']}/events"

        run_id = data["run_id"]

        # GET /v1/runs/{run_id}
        detail_resp = client.get(f"/v1/runs/{run_id}", headers=AUTH_HEADER)
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["run_id"] == run_id
        assert detail["status"] == "queued"
        assert "compliance" in detail
        assert "totals" in detail

        # GET /v1/runs/{run_id}/decisions
        dec_resp = client.get(f"/v1/runs/{run_id}/decisions", headers=AUTH_HEADER)
        assert dec_resp.status_code == 200
        assert "decisions" in dec_resp.json()

        # POST /v1/runs/{run_id}/cancel
        cancel_resp = client.post(f"/v1/runs/{run_id}/cancel", headers=AUTH_HEADER)
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    def test_get_nonexistent_run_returns_404(self, client: TestClient) -> None:
        """Querying a missing run returns 404 with run_not_found code."""
        resp = client.get("/v1/runs/nonexistent_id", headers=AUTH_HEADER)
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "run_not_found"


class TestOpsAndBenchmarkContract:
    """Operational and benchmark endpoint checks."""

    def test_version_endpoint(self, client: TestClient) -> None:
        """GET /v1/version returns git commit and config hashes (NFR-4)."""
        resp = client.get("/v1/version", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "git_commit" in data
        assert "config_hashes" in data
        assert "pinned_models" in data

    def test_spend_endpoint(self, client: TestClient) -> None:
        """GET /v1/spend returns cumulative spend against ceiling (NFR-3)."""
        resp = client.get("/v1/spend", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "cumulative_spend_usd" in data
        assert "ceiling_usd" in data
        assert data["ceiling_exceeded"] is False

    def test_templates_endpoint(self, client: TestClient) -> None:
        """GET /v1/templates lists templates from templates.yaml."""
        resp = client.get("/v1/templates", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        template_ids = [t["template_id"] for t in data]
        assert "fast_cheap" in template_ids
        assert "accurate_slow" in template_ids

    def test_benchmark_job_submission(self, client: TestClient) -> None:
        """POST /v1/benchmark/jobs and GET /v1/metrics/summary."""
        job_payload = {
            "name": "test-ablation-job",
            "task_split": "eval",
            "arms": ["vanilla_fixed", "full"],
            "seeds": [1],
            "concurrency": 4,
        }
        resp = client.post("/v1/benchmark/jobs", json=job_payload, headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

        # Check metrics summary
        summary_resp = client.get("/v1/metrics/summary", headers=AUTH_HEADER)
        assert summary_resp.status_code == 200
        assert "arms" in summary_resp.json()
