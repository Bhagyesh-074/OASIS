"""Contract tests for SQLite schema, migrations, and database invariants (DATABASE.md).

Guards:
- 13 tables created per docs/DATABASE.md.
- WAL mode and foreign_keys=ON enabled.
- FR-15: Score layers stored separately, never collapsed.
- FR-24: Decision log requires non-empty justification.
- Foreign key cascading deletion.
- Snapshot creation via SQLite backup API.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from oasis.db.migrate import apply_migrations, create_snapshot, init_connection
from oasis.db.models import (
    AgentInstanceRecord,
    AgentOutputRecord,
    Database,
    DecisionLogRecord,
    RunRecord,
    ScoreRecord,
    TaskRecord,
    TeamConfigRecord,
)

EXPECTED_TABLES = {
    "task",
    "run",
    "team_config",
    "agent_instance",
    "agent_output",
    "score",
    "budget_event",
    "replacement_event",
    "config_memory",
    "decision_log",
    "injected_failure",
    "oracle_sweep",
    "judge_validation",
    "schema_migrations",
}


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """Fixture initializing a fresh SQLite database with migrations applied."""
    db_file = tmp_path / "oasis_test.db"
    apply_migrations(db_file)
    return Database(db_file)


class TestDatabaseContract:
    """Validate schema compliance and database constraints."""

    def test_all_13_tables_created(self, temp_db: Database) -> None:
        """Every table defined in DATABASE.md must exist in the schema."""
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}

        for expected in EXPECTED_TABLES:
            assert expected in tables, (
                f"Expected table '{expected}' missing from database"
            )

    def test_wal_mode_and_foreign_keys_enabled(self, temp_db: Database) -> None:
        """Connection must have WAL journal mode and foreign_keys=ON."""
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"

            cursor.execute("PRAGMA foreign_keys;")
            fk = cursor.fetchone()[0]
            assert fk == 1

    def test_score_layers_stored_separately_fr15(self, temp_db: Database) -> None:
        """FR-15: L1, L2, L3 layers write to separate rows, never collapsed."""
        # Setup required parent records
        task = TaskRecord(
            task_id="t1",
            domain="code_generation",
            source="mbpp",
            statement="Write a function",
            split="eval",
        )
        temp_db.insert_task(task)

        run = RunRecord(
            run_id="run_fr15",
            task_id="t1",
            arm="full",
            seed=42,
            framework="langgraph",
            mode="live",
            budget_json={
                "max_tokens": 1000,
                "max_cost_usd": 0.1,
                "max_wall_seconds": 60,
                "max_calls": 5,
            },
        )
        temp_db.create_run(run)

        agent = AgentInstanceRecord(
            agent_id="ag1",
            run_id="run_fr15",
            role="coder",
            template_id="fast_cheap",
            model="gpt-4o-mini-2024-07-18",
            temperature=0.2,
            active_from_step=1,
        )
        temp_db.insert_agent_instance(agent)

        output = AgentOutputRecord(
            output_id="out1",
            run_id="run_fr15",
            agent_id="ag1",
            step=1,
            content="def add(a, b): return a + b",
            content_sha="sha256_content",
            prompt_sha="sha256_prompt",
            purpose="productive",
        )
        temp_db.insert_agent_output(output)

        # Insert 3 separate layers for the exact same output
        s_l1 = ScoreRecord(
            score_id="sc_l1",
            output_id="out1",
            layer="l1_embed",
            composite=0.92,
            relevance=0.92,
        )
        s_l2 = ScoreRecord(
            score_id="sc_l2",
            output_id="out1",
            layer="l2_judge",
            composite=0.85,
            factuality=0.88,
            adherence=0.82,
            completeness=0.85,
            scorer_model="gpt-4o-mini-2024-07-18",
            rubric_version="r3",
        )
        s_l3 = ScoreRecord(
            score_id="sc_l3",
            output_id="out1",
            layer="l3_verifier",
            composite=1.0,
            rubric_version="pytest",
        )

        temp_db.insert_score(s_l1)
        temp_db.insert_score(s_l2)
        temp_db.insert_score(s_l3)

        scores = temp_db.get_scores_for_output("out1")
        assert len(scores) == 3
        layers = {s.layer for s in scores}
        assert layers == {"l1_embed", "l2_judge", "l3_verifier"}

    def test_decision_log_non_empty_justification_fr24(self, temp_db: Database) -> None:
        """FR-24: Every decision requires a non-empty justification string."""
        # 1. Valid decision succeeds
        valid = DecisionLogRecord(
            component="RBE",
            decision="halt_budget",
            inputs_json={"remaining_usd": 0.0},
            justification="Cost budget exhausted.",
        )
        temp_db.insert_decision(valid)

        # 2. Empty justification rejected at model/DB layer
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            invalid = DecisionLogRecord(
                component="RBE",
                decision="halt_budget",
                inputs_json={},
                justification="",
            )
            temp_db.insert_decision(invalid)

        # 3. Whitespace-only justification rejected
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            whitespace = DecisionLogRecord(
                component="RBE",
                decision="halt_budget",
                inputs_json={},
                justification="   \t\n  ",
            )
            temp_db.insert_decision(whitespace)

    def test_foreign_key_cascading_deletion(self, temp_db: Database) -> None:
        """Deleting a run cascades to all its child components."""
        task = TaskRecord(
            task_id="t_cascade",
            domain="support_triage",
            source="authored",
            statement="Triage this ticket",
            split="eval",
        )
        temp_db.insert_task(task)

        run = RunRecord(
            run_id="run_casc",
            task_id="t_cascade",
            arm="full",
            seed=1,
            framework="langgraph",
            mode="live",
            budget_json={
                "max_tokens": 500,
                "max_cost_usd": 0.05,
                "max_wall_seconds": 30,
                "max_calls": 3,
            },
        )
        temp_db.create_run(run)

        team = TeamConfigRecord(
            run_id="run_casc",
            team_size=2,
            origin="estimator",
            topology="linear",
            roles_json=[{"role": "triager"}, {"role": "responder"}],
        )
        temp_db.insert_team_config(team)

        assert temp_db.get_team_config("run_casc") is not None

        # Delete the run directly via SQL to verify ON DELETE CASCADE
        with temp_db._write_lock, temp_db.get_connection() as conn:
            conn.execute("DELETE FROM run WHERE run_id = 'run_casc';")

        assert temp_db.get_team_config("run_casc") is None

    def test_database_snapshot_creation(
        self, temp_db: Database, tmp_path: Path
    ) -> None:
        """create_snapshot creates a valid point-in-time backup."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_path = create_snapshot(temp_db.db_path, snapshot_dir=snapshot_dir)

        assert snapshot_path.exists()
        assert snapshot_path.is_file()

        # Connect to snapshot and verify integrity
        snap_conn = init_connection(snapshot_path)
        try:
            cursor = snap_conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            res = cursor.fetchone()[0]
            assert res == "ok"
        finally:
            snap_conn.close()
