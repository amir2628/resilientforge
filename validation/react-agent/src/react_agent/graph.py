"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.

VALIDATION DEVIATION (see ../../README.md): `ToolNode(TOOLS)` below is
replaced with ResilientForge's `make_tool_node(...)` — this IS the real
integration point a real user would touch to adopt ResilientForge into an
existing LangGraph app, not a special-cased hack. Everything else in this
file is unmodified. Configuration (oracle path, metrics log path) comes from
env vars, read once at import time, so `validation/run_validation.py` doesn't
need to re-edit this file between sessions.
"""

import json
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model
from resilientforge.core.invariants import Invariant
from resilientforge.integrations.langgraph_adapter import make_tool_node
from resilientforge.integrations.raw_tool_loop import create_local_reflect
from resilientforge.telemetry.metrics import MetricEvent


class JsonlMetricsHook:
    """Small dedicated MetricsHook for this validation exercise: appends
    every event as one JSON line, so the full audit trail (every real tool
    call, failure, and recovery attempt with a timestamp) survives across
    the 3 separate validation sessions, not just one process's memory."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __call__(self, event: MetricEvent) -> None:
        with open(self._path, "a") as f:
            f.write(event.model_dump_json(exclude_none=True) + "\n")


def _parsed_tool_message_content(result: Any) -> Any | None:
    content = getattr(result, "content", result)
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _search_result_is_structured(result: Any) -> bool:
    """The search tool's result parses as valid JSON/structured data."""
    return isinstance(_parsed_tool_message_content(result), dict)


def _search_result_has_hits(result: Any) -> bool:
    """The result contains a non-empty list of hits (not silently empty)."""
    parsed = _parsed_tool_message_content(result)
    if not isinstance(parsed, dict):
        return False
    hits = parsed.get("results")
    return isinstance(hits, list) and len(hits) > 0


_SEARCH_INVARIANTS = [
    Invariant(name="search_result_is_structured", check=_search_result_is_structured),
    Invariant(name="search_result_has_hits", check=_search_result_has_hits),
]

_ORACLE_PATH = os.environ.get("RESILIENTFORGE_ORACLE_PATH", ".resilientforge")
_METRICS_LOG_PATH = os.environ.get("RESILIENTFORGE_METRICS_LOG_PATH")

_tool_node = make_tool_node(
    TOOLS,
    invariants=_SEARCH_INVARIANTS,
    oracle_path=_ORACLE_PATH,
    reflect=create_local_reflect(model="qwen2.5:7b"),
    metrics=JsonlMetricsHook(_METRICS_LOG_PATH) if _METRICS_LOG_PATH else None,
)

# Define the function that calls the model


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast( # type: ignore[redundant-cast]
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", _tool_node)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
