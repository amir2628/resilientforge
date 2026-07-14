"""Demo: standing guards (Phase 2) — once a failure shape has recurred
enough times, ResilientForge stops merely recovering from it and starts
*preventing* it outright, and exposes a description you can splice into
your OWN system prompt.

Runs with no API key needed — same hand-rolled `reflect` stand-in as
examples/raw_loop_demo.py.

Run: python examples/guards_demo.py
"""

from __future__ import annotations

import re
from pathlib import Path

from resilientforge import GuardManager
from resilientforge.core.recovery import FailureContext
from resilientforge.integrations.raw_tool_loop import execute_anthropic_tool_use, wrap_tools

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_event(date: str, title: str = "Event") -> dict:
    """Fails unless `date` is already ISO 8601 — the natural-language-date
    failure pattern, same as examples/raw_loop_demo.py."""
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


def hand_rolled_reflect(context: FailureContext) -> dict:
    print(f"    [reflect() called — attempt {context.attempt_number} for {context.tool_name!r}]")
    return {
        "strategy": "reformat_argument",
        "root_cause": "natural-language date string passed where ISO date expected",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


class FakeToolUseBlock:
    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id = id
        self.name = name
        self.input = input


def main() -> None:
    oracle_path = Path(__file__).parent / ".resilientforge_guards"
    print(f"oracle path: {oracle_path}\n")

    tools = wrap_tools(
        {"create_event": create_event},
        oracle_path=oracle_path,
        reflect=hand_rolled_reflect,
        guard_promotion_min_occurrences=3,
    )

    print("--- trials 1-3: recovering reactively, crossing the promotion threshold ---")
    for i, date in enumerate(["next Friday", "next Tuesday", "next Monday"], start=1):
        result = execute_anthropic_tool_use(
            tools, FakeToolUseBlock(id=f"tu_{i}", name="create_event", input={"date": date})
        )
        print(f"  [{i}] date={date!r} -> {result['content']}")

    guard = GuardManager(tools["create_event"].oracle).get("create_event", "date", "transform")
    print(f"\n  guard promoted: {guard is not None}\n")

    print("--- trials 4-5: FRESH dates never seen above — should be PREVENTED, not recovered ---")
    print("  (no reflect() call should print below — the guard fixes the args pre-call)")
    for i, date in enumerate(["next Wednesday", "in 5 days"], start=4):
        result = execute_anthropic_tool_use(
            tools, FakeToolUseBlock(id=f"tu_{i}", name="create_event", input={"date": date})
        )
        print(f"  [{i}] date={date!r} -> {result['content']}")

    print("\n--- guards.describe(): splice this into YOUR OWN system prompt if you want it ---")
    print("    (ResilientForge never does this automatically — no adapter has prompt access)")
    guard_text = GuardManager(tools["create_event"].oracle).describe()
    print(guard_text)

    system_prompt = (
        "You are a scheduling assistant with access to a create_event tool.\n\n"
        "Known recurring issues with this tool, and how they're already handled:\n"
        f"{guard_text}\n"
    )
    print("\n--- the caller-constructed system prompt, guard text spliced in ---")
    print(system_prompt)

    for wrapped in tools.values():
        wrapped.close()


if __name__ == "__main__":
    main()
