"""Failure-injection scenario: missing required field.

Unlike the other four scenarios, this one is detected by an invariant
(schema validation), not an exception — the tool call itself doesn't
raise, it just returns an incomplete result. Demonstrates the
invariant-driven recovery path from core/engine.py (on_violation=
"recover" is the default), not just the exception-driven path.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from resilientforge.core.invariants import Invariant
from resilientforge.core.recovery import FailureContext
from tests.failure_injection.harness import FailureScenario


class _EventResult(BaseModel):
    title: str
    attendees: list[str]


def make_tool() -> Callable[..., Any]:
    def create_event(title: str, attendees: list[str] | None = None) -> dict:
        if attendees is None:
            return {"title": title}  # missing the required field
        return {"title": title, "attendees": attendees}

    return create_event


def reflect(context: FailureContext) -> dict:
    return {
        "strategy": "add_missing_field",
        "root_cause": "attendees was not provided and has no default in the result",
        "argument_patch": {"attendees": []},
    }


SCENARIO = FailureScenario(
    name="missing_required_field",
    description="Required `attendees` field omitted; caught by a Pydantic-schema invariant, not an exception.",
    make_tool=make_tool,
    trials=[
        {"title": "Standup"},
        {"title": "Retro"},
        {"title": "Planning"},
        {"title": "1:1"},
        {"title": "All-hands"},
    ],
    reflect=reflect,
    invariants=[Invariant.from_pydantic_model("valid_event", _EventResult)],
)
