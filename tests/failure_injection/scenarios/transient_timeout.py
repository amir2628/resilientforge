"""PROJECT_SPEC.md §7.3 scenario: transient timeout.

Distinct from the other four: no argument is ever wrong, the operation
just needs to be retried — so the fix is an empty patch/no transforms
(Fix already supports this: apply_fix on an all-defaults Fix returns args
unchanged). This exercises the "not a retry/fallback mechanism"
distinction from §2.2 in the narrowest way possible: the value ResilientForge
adds here isn't retry-with-no-change itself (frameworks already do that
in-run per §1), it's that the SAME learned recipe generalizes instantly to
a brand new query on the very next occurrence, with zero model calls.

Each distinct `query` fails exactly once, then succeeds — keyed per-query
so trials don't interfere with each other, and so a successful trial
always needed exactly 1 retry attempt (keeps avg_attempts_to_recovery
comparable across all five scenarios).
"""

from __future__ import annotations

from typing import Any, Callable

from resilientforge.core.recovery import FailureContext
from tests.failure_injection.harness import FailureScenario


def make_tool() -> Callable[..., Any]:
    attempt_counts: dict[str, int] = {}

    def search(query: str) -> dict:
        attempt_counts[query] = attempt_counts.get(query, 0) + 1
        if attempt_counts[query] == 1:
            raise TimeoutError("Request timed out after 30s")
        return {"query": query, "status": "ok"}

    return search


def reflect(context: FailureContext) -> dict:
    return {"strategy": "retry", "root_cause": "transient upstream timeout, no argument is wrong"}


SCENARIO = FailureScenario(
    name="transient_timeout",
    description="A transient upstream timeout that succeeds on a plain retry — no argument correction needed.",
    make_tool=make_tool,
    trials=[
        {"query": "quarterly earnings report"},
        {"query": "annual shareholder letter"},
        {"query": "board meeting minutes"},
        {"query": "product roadmap Q3"},
        {"query": "customer churn analysis"},
    ],
    reflect=reflect,
)
