"""Demo: wrapping a LangGraph ToolNode with ResilientForge
(PROJECT_SPEC.md §4.5's second Phase 1 integration).

Runs with no API key needed — a hand-rolled `reflect` stands in for a real
model call, and tool calls are driven directly (as an AIMessage would
contain them) rather than through a real chat model, so the demo is
self-contained and deterministic.

Run: python examples/langgraph_demo.py
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from resilientforge.core.recovery import FailureContext
from resilientforge.integrations.langgraph_adapter import (
    make_resilientforge_tool_call_wrapper,
    make_tool_node,
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@tool
def create_event(date: str, title: str = "Event") -> str:
    """Create a calendar event; fails unless `date` is ISO 8601 —
    the natural-language-date failure pattern from §1/§4.3."""
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return f"created {title!r} on {date}"


def hand_rolled_reflect(context: FailureContext) -> dict:
    print(f"    [reflect() called — attempt {context.attempt_number} for {context.tool_name!r}]")
    return {
        "strategy": "reformat_argument",
        "root_cause": "natural-language date string passed where ISO date expected",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


def _invoke_tool_call(compiled, tool_name: str, args: dict, call_id: str):
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])
        ]
    }
    return compiled.invoke(state)


def demo_basic_recovery(oracle_path) -> None:
    print("=== Part 1: basic recovery + fast path through make_tool_node() ===")
    node = make_tool_node(
        [create_event], oracle_path=oracle_path, reflect=hand_rolled_reflect
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile()

    print("\n--- first call, natural-language date ---")
    result = _invoke_tool_call(
        compiled, "create_event", {"date": "next Friday", "title": "Standup"}, "call_1"
    )
    print(f"  {result['messages'][-1].content}")

    print("\n--- second call, DIFFERENT natural-language date ---")
    print("  (should recover via the learned recipe — no reflect() call printed above)")
    result = _invoke_tool_call(
        compiled, "create_event", {"date": "next Tuesday", "title": "Retro"}, "call_2"
    )
    print(f"  {result['messages'][-1].content}")


def demo_retry_policy_composition(oracle_path) -> None:
    print("\n=== Part 2: composing with LangGraph's own RetryPolicy ===")
    print("  No reflect() configured, so ResilientForge exhausts with zero attempts.")
    print("  on_exhausted='raise' + handle_tool_errors=False lets a RetryPolicy on this")
    print("  node get a real chance to re-invoke the whole node (see langgraph_adapter.py's")
    print("  module docstring for why handle_tool_errors=False is what makes this work).\n")

    call_count = {"n": 0}
    inner_wrapper = make_resilientforge_tool_call_wrapper(
        oracle_path=oracle_path / "unfixable", reflect=None, on_exhausted="raise"
    )

    def counting_wrapper(request, execute):
        call_count["n"] += 1
        print(f"    [wrap_tool_call invocation #{call_count['n']}]")
        return inner_wrapper(request, execute)

    node = ToolNode([create_event], handle_tool_errors=False, wrap_tool_call=counting_wrapper)
    graph = StateGraph(MessagesState)
    graph.add_node(
        "tools",
        node,
        retry_policy=RetryPolicy(
            max_attempts=3, initial_interval=0.01, jitter=False, retry_on=lambda exc: True
        ),
    )
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile()

    try:
        _invoke_tool_call(compiled, "create_event", {"date": "not a real date"}, "call_3")
    except Exception as exc:
        print(f"\n  graph raised after RetryPolicy also exhausted: {type(exc).__name__}: {exc}")
    print(f"  total wrap_tool_call invocations: {call_count['n']} (== RetryPolicy max_attempts)")


def main() -> None:
    from pathlib import Path

    oracle_path = Path(__file__).parent / ".resilientforge_langgraph"
    demo_basic_recovery(oracle_path)
    demo_retry_policy_composition(oracle_path)


if __name__ == "__main__":
    main()
