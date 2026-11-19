"""OASIS Database and persistence layer.

Provides SQLite WAL-mode schema migrations, snapshotting, typed records,
and thread-safe repository accessors conforming to docs/DATABASE.md.
"""

from oasis.db.migrate import (
    apply_migrations,
    create_snapshot,
    get_db_path,
    init_connection,
)
from oasis.db.models import (
    AgentInstanceRecord,
    AgentOutputRecord,
    BudgetEventRecord,
    ConfigMemoryRecord,
    Database,
    DecisionLogRecord,
    InjectedFailureRecord,
    JudgeValidationRecord,
    OracleSweepRecord,
    ReplacementEventRecord,
    RunRecord,
    ScoreRecord,
    TaskRecord,
    TeamConfigRecord,
)

__all__ = [
    "AgentInstanceRecord",
    "AgentOutputRecord",
    "BudgetEventRecord",
    "ConfigMemoryRecord",
    "Database",
    "DecisionLogRecord",
    "InjectedFailureRecord",
    "JudgeValidationRecord",
    "OracleSweepRecord",
    "ReplacementEventRecord",
    "RunRecord",
    "ScoreRecord",
    "TaskRecord",
    "TeamConfigRecord",
    "apply_migrations",
    "create_snapshot",
    "get_db_path",
    "init_connection",
]
