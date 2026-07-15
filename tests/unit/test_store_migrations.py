"""Unit tests for oracle/store.py's schema migration mechanism (Phase 5).

`PRAGMA user_version` and the migration runner are internal
implementation details with no public API of their own — these tests
reach into `SQLiteStore._conn` directly (a narrow, deliberate exception
to this codebase's usual public-API-only testing style) since that's the
only way to actually verify a schema version was stamped correctly.
"""

from __future__ import annotations

import sqlite3

import pytest

from resilientforge.oracle.store import _CURRENT_SCHEMA_VERSION, _SCHEMA, SQLiteStore


def test_fresh_database_is_stamped_with_the_current_schema_version(tmp_path):
    store = SQLiteStore(tmp_path / "oracle.db")
    version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == _CURRENT_SCHEMA_VERSION
    store.close()


def test_migration_runs_against_a_simulated_pre_phase_5_database(tmp_path):
    # Build a raw sqlite3 file with today's schema but user_version left
    # at 0 (SQLite's own default) — simulating every oracle.db ever
    # created by Phases 1-4, none of which ever stamped a version.
    db_path = tmp_path / "oracle.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO recipes (signature, tool_name, fix_detail, created_at, last_used) "
        "VALUES ('sig-a', 't', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()

    # Opening it through SQLiteStore, exactly as any real usage would,
    # must run the migration and leave the pre-existing data untouched.
    store = SQLiteStore(db_path)

    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == _CURRENT_SCHEMA_VERSION
    row = store._conn.execute("SELECT * FROM recipes WHERE signature = 'sig-a'").fetchone()
    assert row is not None
    assert row["tool_name"] == "t"
    store.close()


def test_reopening_an_already_migrated_database_is_a_clean_no_op(tmp_path):
    db_path = tmp_path / "oracle.db"
    SQLiteStore(db_path).close()  # first open: runs the migration

    store = SQLiteStore(db_path)  # second open: nothing left to migrate
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == _CURRENT_SCHEMA_VERSION
    store.close()


def test_opening_a_database_from_a_newer_schema_version_raises_clearly(tmp_path):
    db_path = tmp_path / "oracle.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version = {_CURRENT_SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="newer version"):
        SQLiteStore(db_path)
