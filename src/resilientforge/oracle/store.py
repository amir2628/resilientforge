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
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


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

# -- schema migrations (Phase 5) ----------------------------------------------
#
# Every oracle.db ever created by Phases 1-4 has PRAGMA user_version == 0
# (SQLite's own default — nothing ever stamped it before now). Treated here
# as "implicitly today's schema, version tracking begins now," not "unknown/
# incompatible": there is no actual table shape to migrate FROM, since the
# CREATE TABLE statements above haven't changed since Phase 1. Real future
# schema changes get their own migration function appended to _MIGRATIONS,
# each one assuming the tables already exist (migrations run AFTER
# `executescript(_SCHEMA)` creates the Phase-1 baseline) and only needing to
# handle the delta from the previous version.

_CURRENT_SCHEMA_VERSION = 1


def _migrate_0_to_1(conn: sqlite3.Connection) -> None:
    """No actual table change — see the module note above. This
    migration's entire job is being the thing that runs (and gets
    tested against a simulated pre-Phase-5 database) to prove the
    mechanism works before a real schema change ever needs it."""


_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migrate_0_to_1),
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for target_version, migration in _MIGRATIONS:
        if current < target_version:
            migration(conn)
            conn.execute(f"PRAGMA user_version = {target_version}")
            conn.commit()
            current = target_version
    if current > _CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"this oracle.db was created by a newer version of resilientforge "
            f"(schema version {current}) than this installed version supports "
            f"(schema version {_CURRENT_SCHEMA_VERSION}) — upgrade resilientforge "
            f"before opening it."
        )


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
    """Raw CRUD over the `failures` and `recipes` tables.

    Thread-local connections (Phase 5, found by actually load-testing
    concurrent access, not by inspection): a single shared
    `sqlite3.Connection`, even opened with `check_same_thread=False`,
    is NOT safe to use from multiple threads at once — Python's sqlite3
    module raised `sqlite3.InterfaceError: bad parameter or other API
    misuse` under real concurrent load during development.
    `check_same_thread=False` only disables Python's OWN same-thread
    check; it does not make concurrent operations on that ONE connection
    object thread-safe. Each thread that touches this store gets its own
    connection to the same file instead — safe, and exactly what WAL
    mode (see `_new_connection` below) is designed to support: multiple
    connections to one file, concurrent readers alongside a writer.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Eagerly open+migrate on the constructing thread, so a schema/
        # migration problem surfaces immediately at construction time,
        # not on first use from some other thread later.
        self._ensure_connection()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        # WAL's usual companion: without this, a writer that finds the
        # database briefly locked by another writer/connection raises
        # sqlite3.OperationalError immediately; with it, the writer waits
        # up to 5s and retries instead — the right default for the
        # "many readers + occasional writer, occasionally two writers at
        # once" pattern this store is actually used under (a dashboard
        # reading while an agent writes, or multiple wrap()'d agents/
        # threads sharing one Oracle).
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.executescript(_SCHEMA)
        conn.commit()
        _run_migrations(conn)
        return conn

    def _ensure_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """The CURRENT thread's connection — lazily opened on first
        access from a given thread, transparently, so every existing
        `self._conn.execute(...)` call site below needed no change."""
        return self._ensure_connection()

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

    def record_recipe_success(
        self,
        *,
        signature: str,
        tool_name: str,
        fix_detail: dict[str, Any],
        root_cause: str | None,
        fix_strategy: str | None,
        now: str,
    ) -> RecipeRow:
        """Atomic create-or-increment (Phase 5) — found necessary by
        actually load-testing concurrent access, not by inspection.
        `RecipeManager.record_success` used to read a recipe's current
        counters in Python, compute new ones, then write them back in a
        separate statement; two threads doing this at once could both
        read the same starting value and one increment would be lost.
        This does the whole read-modify-write as ONE atomic SQL
        statement — `times_applied`/`times_succeeded`/`success_rate` are
        computed from `recipes.times_applied` etc. (the CURRENT row,
        inside the same atomic operation), never from a value fetched
        by a separate, earlier SELECT. `root_cause`/`fix_strategy` keep
        their "only overwrite if a new value was actually provided"
        semantics via `COALESCE`."""
        cursor = self._conn.execute(
            """
            INSERT INTO recipes (
                signature, tool_name, root_cause, fix_strategy, fix_detail,
                times_applied, times_succeeded, success_rate, created_at, last_used
            ) VALUES (?, ?, ?, ?, ?, 1, 1, 1.0, ?, ?)
            ON CONFLICT(signature) DO UPDATE SET
                fix_detail = excluded.fix_detail,
                root_cause = COALESCE(excluded.root_cause, recipes.root_cause),
                fix_strategy = COALESCE(excluded.fix_strategy, recipes.fix_strategy),
                times_applied = recipes.times_applied + 1,
                times_succeeded = recipes.times_succeeded + 1,
                success_rate = CAST(recipes.times_succeeded + 1 AS REAL) / (recipes.times_applied + 1),
                last_used = excluded.last_used
            RETURNING *
            """,
            (signature, tool_name, root_cause, fix_strategy, json.dumps(fix_detail), now, now),
        )
        row = cursor.fetchone()
        self._conn.commit()
        return RecipeRow._from_row(row)

    def record_recipe_fast_path_failure(self, signature: str, now: str) -> RecipeRow | None:
        """Atomic increment (Phase 5) — same race fixed as
        `record_recipe_success` above, for the "fast-path replay didn't
        pan out" case. Returns None (no-op) if no recipe exists for
        `signature`, matching `RecipeManager.record_fast_path_failure`'s
        existing contract."""
        cursor = self._conn.execute(
            """
            UPDATE recipes
            SET times_applied = times_applied + 1,
                success_rate = CAST(times_succeeded AS REAL) / (times_applied + 1),
                last_used = ?
            WHERE signature = ?
            RETURNING *
            """,
            (now, signature),
        )
        row = cursor.fetchone()
        self._conn.commit()
        return RecipeRow._from_row(row) if row else None

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

    def record_guard_application(
        self, tool_name: str, argument: str, kind: str, *, succeeded: bool, now: str
    ) -> GuardRow | None:
        """Atomic increment (Phase 5) — same class of fix as
        `record_recipe_success`/`record_recipe_fast_path_failure`
        above, for guard counters. `GuardManager.record_application`
        used to increment `times_applied`/`times_succeeded` on an
        in-memory `StandingGuard` (fetched earlier, before the call was
        even attempted) and write it back — a lost-update race under
        concurrent applications of the same guard. Returns None if the
        guard no longer exists (e.g. pruned/deleted between firing and
        this call)."""
        increment = 1 if succeeded else 0
        cursor = self._conn.execute(
            """
            UPDATE guards
            SET times_applied = times_applied + 1,
                times_succeeded = times_succeeded + ?,
                success_rate = CAST(times_succeeded + ? AS REAL) / (times_applied + 1),
                last_applied = ?
            WHERE tool_name = ? AND argument = ? AND kind = ?
            RETURNING *
            """,
            (increment, increment, now, tool_name, argument, kind),
        )
        row = cursor.fetchone()
        self._conn.commit()
        return GuardRow._from_row(row) if row else None

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
        """Closes the CALLING thread's connection only — a documented
        limitation of the thread-local connection pool above, not an
        oversight. Other threads' connections close when SQLite's own
        WAL-mode durability guarantees (every commit is already durable
        on disk, not buffered in a connection that could lose data on
        exit) make this safe: nothing is lost, only file handles are
        reclaimed later than an exhaustive per-thread close would."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
