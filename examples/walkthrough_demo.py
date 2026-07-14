"""A slow, narrated, hands-on walkthrough of ResilientForge — every step
prints what's happening, so you can watch the recovery loop instead of
just trusting a test suite passed.

Run: python examples/walkthrough_demo.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from resilientforge import wrap

ORACLE_PATH = Path(__file__).parent / ".resilientforge_walkthrough"


# =============================================================================
# STEP 1: this is the actual tool an AI agent would call ("function calling").
# It's deliberately simple: book a calendar event. It only accepts a date in
# strict YYYY-MM-DD format. If the date isn't in that format, it crashes.
# =============================================================================

def create_event(date: str, title: str = "Event") -> dict:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise ValueError(f"could not parse date: {date!r} — expected YYYY-MM-DD")
    return {"date": date, "title": title, "status": "created"}


# =============================================================================
# STEP 2: this stands in for asking a real AI model for help. In a real
# setup, this function would call Claude or GPT and ask "how do I fix this
# broken tool call?" Here, it just hard-codes the obvious fix so the demo
# doesn't need an API key.
# =============================================================================

def ask_ai_for_a_fix(context):
    print(f"    >>> ASKING THE AI MODEL FOR HELP. Error was: {context.error_message!r}")
    print("    >>> AI proposes: reparse the 'date' argument as a relative date")
    return {
        "strategy": "reformat_argument",
        "root_cause": "natural-language date string passed where YYYY-MM-DD expected",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dump_database() -> None:
    """Open the actual SQLite file ResilientForge writes to and print its
    raw rows — not the CLI's formatted output, the literal database."""
    db_path = ORACLE_PATH / "oracle.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for table in ("failures", "recipes"):
        print(f"\n--- raw contents of the '{table}' table in {db_path} ---")
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        if not rows:
            print("  (empty)")
        for row in rows:
            for key in row.keys():
                print(f"  {key}: {row[key]}")
            print("  ---")
    conn.close()


def main() -> None:
    if ORACLE_PATH.exists():
        import shutil
        shutil.rmtree(ORACLE_PATH)

    # -- PART 0: what happens with NO ResilientForge at all -----------------
    print_header("PART 0 — calling the raw tool directly, no protection at all")
    print("An AI agent tries to book 'next Friday' as the date.")
    try:
        create_event(date="next Friday", title="Team Meeting")
    except ValueError as exc:
        print(f"  CRASHED: {type(exc).__name__}: {exc}")
        print("  This is what happens today, in most agent code, with no memory of the fix.")

    # -- PART 1: wrap it, same bad input, watch it recover -------------------
    print_header("PART 1 — now wrap the SAME tool with ResilientForge")
    print(f"Local memory (database) will be created at: {ORACLE_PATH}\n")
    wrapped = wrap(create_event, reflect=ask_ai_for_a_fix, oracle_path=ORACLE_PATH)

    print("Call #1: agent tries to book 'next Friday' again (same bad input as Part 0)")
    result = wrapped.invoke(date="next Friday", title="Team Meeting")
    print(f"  SUCCEEDED this time: {result}")

    # -- PART 2: a DIFFERENT bad date, same kind of mistake ------------------
    print_header("PART 2 — a DIFFERENT bad date, same kind of mistake")
    print("Call #2: agent tries to book 'next Tuesday' (never seen this exact value before)")
    print("Watch: no '>>> ASKING THE AI MODEL' line should print below —")
    print("       it should reuse the fix it already learned, instantly.\n")
    result = wrapped.invoke(date="next Tuesday", title="Client Call")
    print(f"  SUCCEEDED, and no AI call was made: {result}")

    # -- PART 3: open the actual database and look at the real rows ---------
    print_header("PART 3 — the actual database file, raw rows, not a summary")
    dump_database()

    wrapped.close()

    print_header("done")
    print(f"The database file is still on disk at: {ORACLE_PATH}")
    print("Open it yourself any time with, e.g.:")
    print(f"  sqlite3 {ORACLE_PATH / 'oracle.db'} 'SELECT * FROM recipes;'")


if __name__ == "__main__":
    main()
