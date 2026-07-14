"""Integration tests for integrations/langgraph_adapter.py against a real
compiled LangGraph StateGraph (not a mock of LangGraph itself — only the
model/reflect call is mocked). Confirms composition with
`handle_tool_errors` and `RetryPolicy`,
using facts verified empirically against langgraph 1.2 while building this
adapter (see the module docstring in langgraph_adapter.py):

- `handle_tool_errors=True` makes `execute()` return an error ToolMessage
  instead of raising; `handle_tool_errors=False` makes it raise directly.
  Both must trigger recovery.
- `RetryPolicy`'s default `retry_on` excludes ValueError/TypeError (it's
  scoped to transient failures like connection errors by design) — tests
  that want to observe RetryPolicy actually firing pass an explicit
  `retry_on` rather than relying on the default classification.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import RetryPolicy

from resilientforge.core.invariants import Invariant
from resilientforge.core.recovery import FailureContext
from resilientforge.integrations.langgraph_adapter import (
    make_resilientforge_tool_call_wrapper,
    make_tool_node,
)
from resilientforge.oracle import Oracle
from resilientforge.oracle.guards import GuardManager

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@tool
def create_event(date: str, title: str = "Event") -> str:
    """Create a calendar event; fails unless `date` is ISO 8601."""
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return f"created {title!r} on {date}"


def date_fixing_reflect(context: FailureContext) -> dict:
    return {
        "strategy": "reformat_argument",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


class CountingReflect:
    def __init__(self, fn):
        self.fn = fn
        self.calls: list[FailureContext] = []

    def __call__(self, context: FailureContext) -> dict:
        self.calls.append(context)
        return self.fn(context)


def _build_graph(node, retry_policy: RetryPolicy | None = None):
    graph = StateGraph(MessagesState)
    graph.add_node("tools", node, retry_policy=retry_policy)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def _invoke_tool(compiled, tool_name: str, args: dict, call_id: str = "call_1"):
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])]
    }
    return compiled.invoke(state)


# -- composes with handle_tool_errors, either setting -------------------------


def test_composes_with_handle_tool_errors_true(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    node = make_tool_node(
        [create_event], oracle_path=tmp_path / "oracle", reflect=reflect, handle_tool_errors=True
    )
    compiled = _build_graph(node)

    result = _invoke_tool(compiled, "create_event", {"date": "next Friday", "title": "Standup"})

    tool_message = result["messages"][-1]
    assert tool_message.status != "error"
    assert "created 'Standup'" in tool_message.content
    assert len(reflect.calls) == 1


def test_composes_with_handle_tool_errors_false(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    node = make_tool_node(
        [create_event], oracle_path=tmp_path / "oracle", reflect=reflect, handle_tool_errors=False
    )
    compiled = _build_graph(node)

    result = _invoke_tool(compiled, "create_event", {"date": "next Friday", "title": "Standup"})

    tool_message = result["messages"][-1]
    assert tool_message.status != "error"
    assert len(reflect.calls) == 1


# -- fast path: second occurrence, zero model calls ----------------------------


def test_second_occurrence_resolves_via_fast_path_zero_model_calls(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    node = make_tool_node([create_event], oracle_path=tmp_path / "oracle", reflect=reflect)
    compiled = _build_graph(node)

    _invoke_tool(compiled, "create_event", {"date": "next Friday", "title": "Standup"}, call_id="c1")
    assert len(reflect.calls) == 1

    result = _invoke_tool(compiled, "create_event", {"date": "next Tuesday", "title": "Retro"}, call_id="c2")

    assert len(reflect.calls) == 1  # not called again — recovered via recipe
    tool_message = result["messages"][-1]
    assert tool_message.status != "error"
    assert "created 'Retro'" in tool_message.content


# -- composes with RetryPolicy -------------------------------------------------


def test_on_exhausted_raise_lets_retry_policy_take_over(tmp_path):
    """No `reflect` configured and no matching recipe -> ResilientForge
    exhausts immediately (zero attempts). With on_exhausted="raise", that
    propagates out of the node, so a RetryPolicy configured on this node
    gets a chance to re-invoke the whole node (and thus this wrapper)
    again — verified by counting total wrapper invocations."""
    wrap_calls = {"n": 0}
    inner_wrapper = make_resilientforge_tool_call_wrapper(
        oracle_path=tmp_path / "oracle", reflect=None, on_exhausted="raise"
    )

    def counting_wrapper(request: ToolCallRequest, execute):
        wrap_calls["n"] += 1
        return inner_wrapper(request, execute)

    from langgraph.prebuilt import ToolNode

    node = ToolNode(
        [create_event], handle_tool_errors=False, wrap_tool_call=counting_wrapper
    )
    compiled = _build_graph(
        node,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False, retry_on=lambda exc: True),
    )

    with pytest.raises(Exception):
        _invoke_tool(compiled, "create_event", {"date": "not a real date", "title": "Standup"})

    assert wrap_calls["n"] == 3  # RetryPolicy's max_attempts were exhausted


def test_on_exhausted_default_returns_graceful_error_message(tmp_path):
    node = make_tool_node([create_event], oracle_path=tmp_path / "oracle", reflect=None)
    compiled = _build_graph(node)  # no RetryPolicy — default must not crash the graph

    result = _invoke_tool(compiled, "create_event", {"date": "not a real date", "title": "Standup"})

    tool_message = result["messages"][-1]
    assert tool_message.status == "error"
    assert "ResilientForge" in tool_message.content
    assert "exhausted" in tool_message.content


# -- invariants (operating on the ToolMessage, not a raw value) ----------------


def test_invariant_violation_recovers_via_this_adapter(tmp_path):
    def contains_created(msg) -> bool:
        return "created" in msg.content

    invariant = Invariant(name="has_created_marker", check=contains_created)

    @tool
    def flaky_marker(date: str) -> str:
        """Returns content missing the 'created' marker unless date is ISO."""
        if not _ISO_DATE_RE.match(date):
            return f"pending: {date}"  # no exception — just a bad result
        return f"created on {date}"

    def reflect(context: FailureContext) -> dict:
        return {
            "strategy": "reformat_argument",
            "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
        }

    node = make_tool_node(
        [flaky_marker], invariants=[invariant], oracle_path=tmp_path / "oracle", reflect=reflect
    )
    compiled = _build_graph(node)

    result = _invoke_tool(compiled, "flaky_marker", {"date": "next Friday"})

    tool_message = result["messages"][-1]
    assert "created on" in tool_message.content


def test_invariant_abort_raises_and_does_not_call_reflect(tmp_path):
    reflect = CountingReflect(lambda ctx: {"strategy": "noop"})

    @tool
    def dangerous(action: str) -> str:
        """A tool that can do destructive things."""
        return action

    invariant = Invariant(
        name="no_delete", check=lambda msg: "delete" not in msg.content, on_violation="abort"
    )
    node = make_tool_node(
        [dangerous], invariants=[invariant], oracle_path=tmp_path / "oracle", reflect=reflect
    )
    compiled = _build_graph(node)

    with pytest.raises(Exception):
        _invoke_tool(compiled, "dangerous", {"action": "delete"})

    assert reflect.calls == []


# -- make_tool_node convenience -------------------------------------------------


def test_make_tool_node_builds_a_working_node(tmp_path):
    node = make_tool_node([create_event], oracle_path=tmp_path / "oracle")
    compiled = _build_graph(node)

    result = _invoke_tool(compiled, "create_event", {"date": "2026-03-05", "title": "Standup"})

    tool_message = result["messages"][-1]
    assert tool_message.status != "error"
    assert "created 'Standup'" in tool_message.content


def test_make_tool_node_threads_enable_standing_guards_end_to_end(tmp_path):
    oracle_path = tmp_path / "oracle"
    oracle = Oracle(oracle_path)
    GuardManager(oracle).promote(
        tool_name="create_event", argument="date", kind="transform",
        transform="parse_relative_date_to_iso", source_signature="sig-seed",
    )
    oracle.close()

    # Disabled: the pre-seeded guard must NOT fire — fails outright with no
    # reflect configured, exactly as if no guard existed.
    disabled_node = make_tool_node(
        [create_event], oracle_path=oracle_path, reflect=None, enable_standing_guards=False,
    )
    compiled = _build_graph(disabled_node)
    result = _invoke_tool(compiled, "create_event", {"date": "next Friday"}, call_id="c1")
    assert result["messages"][-1].status == "error"

    # Enabled (the default): the same pre-seeded guard fires and prevents
    # the failure outright.
    enabled_node = make_tool_node([create_event], oracle_path=oracle_path, reflect=None)
    compiled = _build_graph(enabled_node)
    result = _invoke_tool(compiled, "create_event", {"date": "next Tuesday"}, call_id="c2")
    assert result["messages"][-1].status != "error"
