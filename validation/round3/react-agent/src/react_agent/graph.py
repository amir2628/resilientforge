"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.

VALIDATION DEVIATION, ROUND 2 (see ../../README.md): same as round 1 --
`ToolNode(TOOLS)` replaced with ResilientForge's `make_tool_node(...)`, the
real integration point. Now wraps TWO tools (`search`, `extract_url_content`)
through the same shared ToolNode/oracle, since `wrap_tool_call` is one
function shared across every tool a ToolNode holds -- invariants below are
written to make sense against either tool's result shape rather than being
tool-specific (make_tool_node has no per-tool invariants knob; a real,
honest limitation worth noting, same spirit as round 1's note on
args-schema invariants).
"""

import json
import os
import re
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
    """Same small dedicated MetricsHook as round 1 -- appends every event
    as one JSON line for a full, timestamped audit trail across sessions."""

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


def _result_is_structured(result: Any) -> bool:
    """Either tool's successful result parses as valid JSON/structured data."""
    return isinstance(_parsed_tool_message_content(result), dict)


def _result_is_non_empty(result: Any) -> bool:
    """search: non-empty `results` list. extract_url_content: non-empty
    `content` string. Whichever key is present for the tool that was
    actually called; vacuously true if neither key is present (not this
    invariant's concern for a result shape it doesn't recognize)."""
    parsed = _parsed_tool_message_content(result)
    if not isinstance(parsed, dict):
        return False
    if "results" in parsed:
        hits = parsed.get("results")
        return isinstance(hits, list) and len(hits) > 0
    if "content" in parsed:
        text = parsed.get("content")
        return isinstance(text, str) and len(text.strip()) > 0
    return False


_HTML_LEAK_RE = re.compile(r"<\s*(html|body|div|script|style|p|span)\b", re.IGNORECASE)


def _extracted_content_is_clean_text(result: Any) -> bool:
    """extract_url_content only: the extracted `content` is non-empty text,
    not raw HTML/binary leaking through unparsed. Vacuously true for any
    result shape without a `content` key (i.e. search's), since this
    invariant isn't about that tool."""
    parsed = _parsed_tool_message_content(result)
    if not isinstance(parsed, dict) or "content" not in parsed:
        return True
    text = parsed.get("content")
    if not isinstance(text, str) or not text.strip():
        return False
    return not _HTML_LEAK_RE.search(text)


_INVARIANTS = [
    Invariant(name="result_is_structured", check=_result_is_structured),
    Invariant(name="result_is_non_empty", check=_result_is_non_empty),
    Invariant(name="extracted_content_is_clean_text", check=_extracted_content_is_clean_text),
]

_ORACLE_PATH = os.environ.get("RESILIENTFORGE_ORACLE_PATH", ".resilientforge")
_METRICS_LOG_PATH = os.environ.get("RESILIENTFORGE_METRICS_LOG_PATH")

_tool_node = make_tool_node(
    TOOLS,
    invariants=_INVARIANTS,
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
