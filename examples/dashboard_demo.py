"""Demo: seeds a small oracle with a few recipes, a standing guard, and
some failure history, then tells you exactly how to look at it in the
dashboard (Phase 4).

This script does NOT start the dashboard itself — `resilientforge
dashboard` blocks until Ctrl+C, which doesn't fit this project's
"every example runs to completion on its own" pattern (see
raw_loop_demo.py, guards_demo.py, walkthrough_demo.py). Run it, then
run the command it prints.

Needs the `dashboard` extra: pip install resilientforge[dashboard]

Run: python examples/dashboard_demo.py
"""

from __future__ import annotations

import re
from pathlib import Path

from resilientforge import GuardManager
from resilientforge.core.recovery import FailureContext
from resilientforge.integrations.raw_tool_loop import wrap_tools

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_event(date: str, title: str = "Event") -> dict:
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


def send_reminder(recipient: str, minutes_before: int) -> dict:
    if not isinstance(minutes_before, int):
        raise TypeError(f"minutes_before must be an int, got {type(minutes_before).__name__}")
    return {"recipient": recipient, "minutes_before": minutes_before, "status": "scheduled"}


def reflect(context: FailureContext) -> dict:
    if context.tool_name == "create_event":
        return {
            "strategy": "reformat_argument",
            "root_cause": "natural-language date string passed where ISO date expected",
            "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
        }
    return {
        "strategy": "coerce_type",
        "root_cause": "minutes_before arrived as a non-numeric string",
        "transforms": [{"argument": "minutes_before", "transform": "coerce_int"}],
    }


def main() -> None:
    oracle_path = Path(__file__).parent / ".resilientforge_dashboard"

    tools = wrap_tools(
        {"create_event": create_event, "send_reminder": send_reminder},
        oracle_path=oracle_path,
        reflect=reflect,
        guard_promotion_min_occurrences=3,
    )

    print("Seeding some recipes, a guard, and failure history...")
    for date in ["next Friday", "next Tuesday", "next Monday", "next Wednesday"]:
        tools["create_event"].invoke(date=date, title="Standup")
    for recipient in ["a@x.com", "b@x.com"]:
        tools["send_reminder"].invoke(recipient=recipient, minutes_before="15")

    # One more failure that never recovers, so the dashboard has an
    # "exhausted" / non-recovered row too, not just successes.
    try:
        tools["send_reminder"].invoke(recipient="c@x.com", minutes_before="not-a-number")
    except Exception:
        pass

    guard = GuardManager(tools["create_event"].oracle).get("create_event", "date", "transform")
    print(f"Guard promoted for create_event(date): {guard is not None}")

    for wrapped in tools.values():
        wrapped.close()

    print(f"\nOracle seeded at: {oracle_path}")
    print("\nNow run:")
    print(f"  resilientforge dashboard --oracle-path {oracle_path}")
    print("and open http://127.0.0.1:8765 in a browser.")


if __name__ == "__main__":
    main()
