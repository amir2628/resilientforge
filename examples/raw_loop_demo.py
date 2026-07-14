"""Demo: wrapping a raw Anthropic-style tool-calling loop with
ResilientForge (PROJECT_SPEC.md §4.5's reference integration).

Runs with no API key needed: `reflect` here is a small hand-rolled stand-in
for a real model call, so the demo is self-contained. For real usage, swap
it for `create_anthropic_reflect()` (see the comment below) — same
`ReflectFn` interface either way, since core/recovery.py and core/engine.py
never care which implementation of `reflect` they're given.

Run: python examples/raw_loop_demo.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from resilientforge.core.recovery import FailureContext
from resilientforge.integrations.raw_tool_loop import (
    execute_anthropic_tool_use,
    execute_openai_tool_call,
    make_json_arg_parser,
    wrap_tools,
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# -- the tools a real agent would expose to the model --------------------------


def create_event(date: str, title: str = "Event") -> dict:
    """A real calendar tool would reject anything that isn't a proper ISO
    date — this is the natural-language-date failure pattern from §1/§4.3."""
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


# -- stand-in for a real model call ---------------------------------------------
#
# A real reflect() would call an LLM. To use Anthropic for real:
#
#   from resilientforge.integrations.raw_tool_loop import create_anthropic_reflect
#   reflect = create_anthropic_reflect(model="claude-sonnet-5")  # reads ANTHROPIC_API_KEY
#
# Both implementations satisfy the same ReflectFn interface — engine.py
# doesn't know or care which one it's given.


def hand_rolled_reflect(context: FailureContext) -> dict:
    print(f"    [reflect() called — attempt {context.attempt_number} for {context.tool_name!r}]")
    if context.tool_name == "parse_tool_call_json":
        return {
            "strategy": "repair_json",
            "transforms": [{"argument": "raw_args", "transform": "repair_common_json_errors"}],
        }
    if context.error_type == "ValueError" and "date" in context.args:
        return {
            "strategy": "reformat_argument",
            "root_cause": "natural-language date string passed where ISO date expected",
            "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
        }
    return {"strategy": "unknown", "argument_patch": {}}


# -- a fake tool_use block, standing in for what a real Message would contain --


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeOpenAIToolCall:
    id: str
    function: FakeFunction


def main() -> None:
    oracle_path = Path(__file__).parent / ".resilientforge"
    print(f"oracle path: {oracle_path}\n")

    tools = wrap_tools(
        {"create_event": create_event},
        oracle_path=oracle_path,
        reflect=hand_rolled_reflect,
    )

    print("--- Anthropic-style tool_use: first call, natural-language date ---")
    result = execute_anthropic_tool_use(
        tools, FakeToolUseBlock(id="tu_1", name="create_event", input={"date": "next Friday"})
    )
    print(f"  result: {result}\n")

    print("--- second call, DIFFERENT natural-language date ---")
    print("  (should recover via the learned recipe — no reflect() call printed above)")
    result = execute_anthropic_tool_use(
        tools, FakeToolUseBlock(id="tu_2", name="create_event", input={"date": "next Tuesday"})
    )
    print(f"  result: {result}\n")

    print("--- OpenAI-style tool_call: malformed JSON args (trailing comma) ---")
    json_parser = make_json_arg_parser(tools["create_event"].oracle, reflect=hand_rolled_reflect)
    result = execute_openai_tool_call(
        tools,
        FakeOpenAIToolCall(
            id="call_1",
            function=FakeFunction(name="create_event", arguments='{"date": "2026-03-05",}'),
        ),
        json_parser=json_parser,
    )
    print(f"  result: {result}\n")

    print("--- recipes learned this run (persisted at oracle path above) ---")
    for recipe in tools["create_event"].recipes.list():
        print(f"  {recipe.signature!r}: applied {recipe.times_applied}x, "
              f"success_rate={recipe.success_rate}")

    for wrapped in tools.values():
        wrapped.close()


if __name__ == "__main__":
    main()
