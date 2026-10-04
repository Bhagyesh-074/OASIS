"""Applies numbered SQL migrations in order (no Alembic).

Conforms to docs/DATABASE.md operational requirements:
- Plain numbered SQL files applied in order.
- SQLite WAL mode and foreign_keys=ON enabled.
- Snapshots saved to artifacts/db_snapshots/.
"""

from __future__ import annotations

import datetime
import os
import re
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
DEFAULT_DB_PATH = Path(os.environ.get("OASIS_DB_PATH", "./data/oasis.db"))
DEFAULT_SNAPSHOT_DIR = Path("./artifacts/db_snapshots")


def get_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve database path from argument or environment."""
    if db_path is not None:
        return Path(db_path)
    env_path = os.environ.get("OASIS_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def ensure_parent_dir(path: Path) -> None:
    """Ensure directory containing the file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def init_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open SQLite connection with WAL mode and foreign keys enabled."""
    resolved = Path(db_path)
    ensure_parent_dir(resolved)
    conn = sqlite3.connect(str(resolved), timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.row_factory = sqlite3.Row
    return conn


def get_migration_files(schema_dir: Path | None = None) -> list[tuple[int, Path]]:
    """List and sort migration sql files by version prefix."""
    directory = schema_dir or SCHEMA_DIR
    if not directory.exists():
        return []
    migrations: list[tuple[int, Path]] = []
    for file in directory.glob("*.sql"):
        match = re.match(r"^(\d+)_", file.name)
        if match:
            version = int(match.group(1))
            migrations.append((version, file))
    migrations.sort(key=lambda item: item[0])
    return migrations


def get_applied_migrations(conn: sqlite3.Connection) -> set[int]:
    """Retrieve set of applied migration versions."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        """
    )
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
    return {row[0] for row in cursor.fetchall()}


def apply_migrations(
    db_path: Path | str | None = None,
    schema_dir: Path | None = None,
) -> list[str]:
    """Apply all pending numbered SQL migrations in ascending order.

    Returns the list of migration file names applied during this run.
    """
    resolved_path = get_db_path(db_path)
    conn = init_connection(resolved_path)
    applied_names: list[str] = []

    try:
        with conn:
            applied_versions = get_applied_migrations(conn)
            migration_files = get_migration_files(schema_dir)

            for version, filepath in migration_files:
                if version in applied_versions:
                    continue

                sql_script = filepath.read_text(encoding="utf-8")
                # executescript automatically executes PRAGMAs & DDL
                conn.executescript(sql_script)

                now_utc = datetime.datetime.now(datetime.UTC).isoformat()
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);",
                    (version, filepath.name, now_utc),
                )
                applied_names.append(filepath.name)
    finally:
        conn.close()

    return applied_names


def create_snapshot(
    db_path: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
) -> Path:
    """Create a point-in-time SQLite backup snapshot in artifacts/db_snapshots/."""
    source_path = get_db_path(db_path)
    target_dir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    target_path = target_dir / f"oasis_{timestamp}.db"

    source_conn = init_connection(source_path)
    try:
        dest_conn = sqlite3.connect(str(target_path))
        try:
            with dest_conn:
                source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        source_conn.close()

    return target_path


def main(args: Sequence[str] | None = None) -> int:
    """CLI entrypoint for database migration."""
    applied = apply_migrations()
    if applied:
        print(f"Applied {len(applied)} migrations: {', '.join(applied)}")
    else:
        print("Database schema is up to date. No pending migrations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
