"""Failure-injection scenario: malformed JSON args.

Mirrors the failure OpenAI-style function-calling hands back directly
(arguments as a raw JSON string that doesn't parse — a trailing comma
here) — the exact case integrations/raw_tool_loop.py's
execute_openai_tool_call encounters (see test_execute_openai_tool_call_
recovers_malformed_json_args). Recovered via the repair_common_json_errors
transform from core/recovery.py's TRANSFORM_REGISTRY.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from resilientforge.core.recovery import FailureContext
from tests.failure_injection.harness import FailureScenario


def make_tool() -> Callable[..., Any]:
    def process_order(raw_args: str) -> dict:
        return json.loads(raw_args)

    return process_order


def reflect(context: FailureContext) -> dict:
    return {
        "strategy": "repair_json",
        "root_cause": "trailing comma before a closing bracket makes the JSON invalid",
        "transforms": [{"argument": "raw_args", "transform": "repair_common_json_errors"}],
    }


SCENARIO = FailureScenario(
    name="malformed_json_args",
    description="Trailing comma in tool-call arguments JSON (raw-string args, e.g. OpenAI-style).",
    make_tool=make_tool,
    trials=[
        {"raw_args": '{"item": "widget", "qty": 3,}'},
        {"raw_args": '{"item": "gadget", "qty": 1,}'},
        {"raw_args": '{"item": "gizmo", "qty": 7,}'},
        {"raw_args": '{"item": "sprocket", "qty": 2,}'},
        {"raw_args": '{"item": "cog", "qty": 5,}'},
    ],
    reflect=reflect,
)
