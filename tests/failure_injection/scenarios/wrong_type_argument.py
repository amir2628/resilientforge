"""PROJECT_SPEC.md §7.3 scenario: wrong argument type — a numeric-only
field arrives as a string (e.g. the model stringified a number). Recovered
via the coerce_int transform, applied to each trial's own literal value.
"""

from __future__ import annotations

from typing import Any, Callable

from resilientforge.core.recovery import FailureContext
from tests.failure_injection.harness import FailureScenario


def make_tool() -> Callable[..., Any]:
    def set_quantity(quantity: int) -> dict:
        if not isinstance(quantity, int):
            raise TypeError(f"quantity must be an int, got {type(quantity).__name__}")
        return {"quantity": quantity, "status": "updated"}

    return set_quantity


def reflect(context: FailureContext) -> dict:
    return {
        "strategy": "coerce_type",
        "root_cause": "quantity arrived as a string instead of an int",
        "transforms": [{"argument": "quantity", "transform": "coerce_int"}],
    }


SCENARIO = FailureScenario(
    name="wrong_type_argument",
    description="A numeric argument (quantity) arrives as a string instead of an int.",
    make_tool=make_tool,
    trials=[
        {"quantity": "5"},
        {"quantity": "42"},
        {"quantity": "100"},
        {"quantity": "7"},
        {"quantity": "250"},
    ],
    reflect=reflect,
)
