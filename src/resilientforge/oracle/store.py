"""SQLite-backed structured store for failure records, recovery recipes, and
standing guards.

Three tables:
- `failures`: one row per tool-call/invariant failure occurrence.
- `recipes`: one row per distinct failure signature that has a known fix,
  keyed by signature so a re-seen failure shape updates the same row rather
  than accumulating duplicates.
- `guards` (Phase 2): one row per `(tool_name, argument, kind)` — a proactive
  fix promoted from a proven recipe, applied *before* the first call attempt
  rather than after a failure. Keyed differently from `recipes` on purpose:
  pre-call there's no `error_type`/`error_message` yet, so guards can't be
  looked up by full failure `signature` the way recipes are — matching is by
  which tool/argument is involved, not by which error already happened.

This module only owns raw persistence (schema + CRUD). Recipe domain logic —
building a `Recipe` from a successful recovery, updating `times_applied` /
`success_rate`, and pruning policy — lives in `oracle/recipes.py`. Guard
domain logic (promotion eligibility, describe() text) lives in
`oracle/guards.py`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ResolutionStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RECOVERED = "recovered"
    EXHAUSTED = "exhausted"
    # A recovery attempt hit an invariant with on_violation="abort" — distinct
    # from EXHAUSTED, which implies recovery was attempted and ran out of
    # tries; ABORTED means recovery was deliberately never (fully) attempted.
    ABORTED = "aborted"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT,
    tool_name TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    sanitized_args TEXT NOT NULL DEFAULT '{}',
    signature TEXT NOT NULL,
    resolution_status TEXT NOT NULL DEFAULT 'unresolved',
    fix_applied TEXT,
    fix_verified INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failures_signature ON failures(signature);
CREATE INDEX IF NOT EXISTS idx_failures_workflow_id ON failures(workflow_id);

CREATE TABLE IF NOT EXISTS recipes (
    signature TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    root_cause TEXT,
    fix_strategy TEXT,
    fix_detail TEXT NOT NULL DEFAULT '{}',
    times_applied INTEGER NOT NULL DEFAULT 0,
    times_succeeded INTEGER NOT NULL DEFAULT 0,
    success_rate REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_used TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS guards (
    tool_name TEXT NOT NULL,
    argument TEXT NOT NULL,
    kind TEXT NOT NULL,
    transform TEXT,
    patch_value TEXT,
    source_signature TEXT NOT NULL,
    root_cause TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_applied TEXT,
    times_applied INTEGER NOT NULL DEFAULT 0,
    times_succeeded INTEGER NOT NULL DEFAULT 0,
    success_rate REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (tool_name, argument, kind)
);

CREATE INDEX IF NOT EXISTS idx_guards_tool_name ON guards(tool_name);
"""


@dataclass
class FailureRecord:
    tool_name: str
    signature: str
    workflow_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    fix_applied: dict[str, Any] | None = None
    fix_verified: bool | None = None
    created_at: str = ""
    id: int | None = None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> FailureRecord:
        return cls(
            id=row["id"],
            workflow_id=row["workflow_id"],
            tool_name=row["tool_name"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            sanitized_args=json.loads(row["sanitized_args"]),
            signature=row["signature"],
            resolution_status=ResolutionStatus(row["resolution_status"]),
            fix_applied=json.loads(row["fix_applied"]) if row["fix_applied"] else None,
            fix_verified=bool(row["fix_verified"]) if row["fix_verified"] is not None else None,
            created_at=row["created_at"],
        )


@dataclass
class RecipeRow:
    signature: str
    tool_name: str
    fix_detail: dict[str, Any]
    created_at: str
    last_used: str
    root_cause: str | None = None
    fix_strategy: str | None = None
    times_applied: int = 0
    times_succeeded: int = 0
    success_rate: float = 0.0

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> RecipeRow:
        return cls(
            signature=row["signature"],
            tool_name=row["tool_name"],
            root_cause=row["root_cause"],
            fix_strategy=row["fix_strategy"],
            fix_detail=json.loads(row["fix_detail"]),
            times_applied=row["times_applied"],
            times_succeeded=row["times_succeeded"],
            success_rate=row["success_rate"],
            created_at=row["created_at"],
            last_used=row["last_used"],
        )


@dataclass
class GuardRow:
    tool_name: str
    argument: str
    kind: str  # "transform" | "patch"
    source_signature: str
    created_at: str
    transform: str | None = None
    patch_value: Any = None
    root_cause: str | None = None
    active: bool = True
    last_applied: str | None = None
    times_applied: int = 0
    times_succeeded: int = 0
    success_rate: float = 0.0

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> GuardRow:
        return cls(
            tool_name=row["tool_name"],
            argument=row["argument"],
            kind=row["kind"],
            transform=row["transform"],
            patch_value=json.loads(row["patch_value"]) if row["patch_value"] is not None else None,
            source_signature=row["source_signature"],
            root_cause=row["root_cause"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            last_applied=row["last_applied"],
            times_applied=row["times_applied"],
            times_succeeded=row["times_succeeded"],
            success_rate=row["success_rate"],
        )


class SQLiteStore:
    """Raw CRUD over the `failures` and `recipes` tables."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- failures --------------------------------------------------------

    def insert_failure(self, record: FailureRecord) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO failures (
                workflow_id, tool_name, error_type, error_message,
                sanitized_args, signature, resolution_status,
                fix_applied, fix_verified, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.workflow_id,
                record.tool_name,
                record.error_type,
                record.error_message,
                json.dumps(record.sanitized_args),
                record.signature,
                record.resolution_status.value,
                json.dumps(record.fix_applied) if record.fix_applied is not None else None,
                None if record.fix_verified is None else int(record.fix_verified),
                record.created_at,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def get_failure(self, failure_id: int) -> FailureRecord | None:
        row = self._conn.execute(
            "SELECT * FROM failures WHERE id = ?", (failure_id,)
        ).fetchone()
        return FailureRecord._from_row(row) if row else None

    def update_failure_resolution(
        self,
        failure_id: int,
        status: ResolutionStatus,
        fix_applied: dict[str, Any] | None = None,
        fix_verified: bool | None = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE failures
            SET resolution_status = ?, fix_applied = ?, fix_verified = ?
            WHERE id = ?
            """,
            (
                status.value,
                json.dumps(fix_applied) if fix_applied is not None else None,
                None if fix_verified is None else int(fix_verified),
                failure_id,
            ),
        )
        self._conn.commit()

    def list_failures(
        self,
        signature: str | None = None,
        workflow_id: str | None = None,
        limit: int = 100,
    ) -> list[FailureRecord]:
        clauses = []
        params: list[Any] = []
        if signature is not None:
            clauses.append("signature = ?")
            params.append(signature)
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM failures {where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
        return [FailureRecord._from_row(row) for row in rows]

    # -- recipes -----------------------------------------------------------

    def upsert_recipe(self, recipe: RecipeRow) -> None:
        self._conn.execute(
            """
            INSERT INTO recipes (
                signature, tool_name, root_cause, fix_strategy, fix_detail,
                times_applied, times_succeeded, success_rate, created_at, last_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signature) DO UPDATE SET
                tool_name = excluded.tool_name,
                root_cause = excluded.root_cause,
                fix_strategy = excluded.fix_strategy,
                fix_detail = excluded.fix_detail,
                times_applied = excluded.times_applied,
                times_succeeded = excluded.times_succeeded,
                success_rate = excluded.success_rate,
                last_used = excluded.last_used
            """,
            (
                recipe.signature,
                recipe.tool_name,
                recipe.root_cause,
                recipe.fix_strategy,
                json.dumps(recipe.fix_detail),
                recipe.times_applied,
                recipe.times_succeeded,
                recipe.success_rate,
                recipe.created_at,
                recipe.last_used,
            ),
        )
        self._conn.commit()

    def get_recipe(self, signature: str) -> RecipeRow | None:
        row = self._conn.execute(
            "SELECT * FROM recipes WHERE signature = ?", (signature,)
        ).fetchone()
        return RecipeRow._from_row(row) if row else None

    def list_recipes(self, tool_name: str | None = None, limit: int = 100) -> list[RecipeRow]:
        if tool_name is not None:
            rows = self._conn.execute(
                "SELECT * FROM recipes WHERE tool_name = ? ORDER BY last_used DESC LIMIT ?",
                (tool_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM recipes ORDER BY last_used DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RecipeRow._from_row(row) for row in rows]

    def delete_recipe(self, signature: str) -> None:
        self._conn.execute("DELETE FROM recipes WHERE signature = ?", (signature,))
        self._conn.commit()

    # -- guards --------------------------------------------------------------

    def upsert_guard(self, guard: GuardRow) -> None:
        self._conn.execute(
            """
            INSERT INTO guards (
                tool_name, argument, kind, transform, patch_value,
                source_signature, root_cause, active, created_at,
                last_applied, times_applied, times_succeeded, success_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tool_name, argument, kind) DO UPDATE SET
                transform = excluded.transform,
                patch_value = excluded.patch_value,
                source_signature = excluded.source_signature,
                root_cause = excluded.root_cause,
                active = excluded.active,
                last_applied = excluded.last_applied,
                times_applied = excluded.times_applied,
                times_succeeded = excluded.times_succeeded,
                success_rate = excluded.success_rate
            """,
            (
                guard.tool_name,
                guard.argument,
                guard.kind,
                guard.transform,
                json.dumps(guard.patch_value) if guard.patch_value is not None else None,
                guard.source_signature,
                guard.root_cause,
                int(guard.active),
                guard.created_at,
                guard.last_applied,
                guard.times_applied,
                guard.times_succeeded,
                guard.success_rate,
            ),
        )
        self._conn.commit()

    def get_guard(self, tool_name: str, argument: str, kind: str) -> GuardRow | None:
        row = self._conn.execute(
            "SELECT * FROM guards WHERE tool_name = ? AND argument = ? AND kind = ?",
            (tool_name, argument, kind),
        ).fetchone()
        return GuardRow._from_row(row) if row else None

    def list_guards(
        self,
        tool_name: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[GuardRow]:
        clauses = []
        params: list[Any] = []
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if active_only:
            clauses.append("active = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM guards {where} ORDER BY tool_name, argument LIMIT ?", params
        ).fetchall()
        return [GuardRow._from_row(row) for row in rows]

    def delete_guard(self, tool_name: str, argument: str, kind: str) -> None:
        self._conn.execute(
            "DELETE FROM guards WHERE tool_name = ? AND argument = ? AND kind = ?",
            (tool_name, argument, kind),
        )
        self._conn.commit()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
