"""Failure-injection scenario: ambiguous fix candidates (Phase 3).

Unlike the other scenarios, a single "obvious" guess can't reliably fix
this failure: the correct `section` for a desk assignment depends on a
rule (`desk_number % 3`) that isn't derivable from the call arguments
alone — only actually trying a candidate against the real tool reveals
which one is right. This is deliberately NOT solvable by a `transforms`
entry in TRANSFORM_REGISTRY (a transform is a pure function of the
CURRENT value alone, and there is no such pure function here that
doesn't just hardcode the hidden rule, which would defeat the point of
this scenario) and NOT safely cacheable via a plain `argument_patch`
either (the correct `section` differs by desk_number, so blindly
replaying whichever section worked last time is exactly the trap
core/recovery.py's module docstring warns about — `argument_patch` is
only safe to replay when the right value doesn't depend on the specific
occurrence, and here it does).

What DOES generalize is the tool's own real verification: trying a wrong
`section` guess has no problematic real-world effect (this tool is a
pure computation — see `side_effect_free`), so speculative branching
tries every plausible section for real and lets the tool's own answer,
not a cached guess, decide. When a stale recipe from an earlier trial
happens to be wrong for THIS trial's desk_number, real verification
correctly rejects it (the tool validates `section` against `desk_number`,
which is never overwritten by the fix) rather than silently reusing it.

Configured with num_branches == len(_SECTIONS) and side_effect_free=True
so every trial's recovery genuinely covers all 3 sections for real.
Unlike every other scenario in this suite, oracle_hit_rate_after_first is
NOT expected to reach 100% here: filling a candidate batch means
reflect() is consulted every round regardless of whether a recipe
already exists — a real, documented cost of considering more options
(see harness.py's `avg_candidates_considered`). test_recovery_rate.py's
oracle-hit-rate test explicitly exempts num_branches>1 scenarios for
this reason.
"""

from __future__ import annotations

from typing import Any, Callable

from resilientforge.core.recovery import FailureContext
from tests.failure_injection.harness import FailureScenario

_SECTIONS = ["A", "B", "C"]


def _correct_section(desk_number: int) -> str:
    # Stands in for a real-world rule (e.g. which section a floor plan
    # assigns a desk number to) that isn't recoverable from the argument
    # alone — the whole point of this scenario.
    return _SECTIONS[desk_number % len(_SECTIONS)]


def make_tool() -> Callable[..., Any]:
    def assign_desk(desk_number: int, section: str | None = None) -> dict:
        if section is None:
            raise ValueError(f"section is required to assign desk {desk_number}")
        if section != _correct_section(desk_number):
            raise ValueError(f"section {section!r} is wrong for desk {desk_number}")
        return {"desk": f"{section}{desk_number}", "status": "assigned"}

    return assign_desk


def reflect(context: FailureContext) -> dict:
    """Proposes an untried section letter — a stand-in for a model that
    can name several plausible guesses but can't verify which is right
    without actually trying it. Diversifies via `previous_attempts`,
    exactly the mechanism core/engine.py's `_find_fix_candidates`
    documents: within one round, each reflect() call sees every
    candidate proposed so far (including a stale recipe candidate, if
    one was generated first), so it naturally avoids repeating it."""
    tried = {
        fix.argument_patch.get("section")
        for fix in context.previous_attempts
        if fix.argument_patch.get("section")
    }
    for section in _SECTIONS:
        if section not in tried:
            return {
                "strategy": "guess_section",
                "root_cause": "the correct section is not derivable from the desk number alone",
                "argument_patch": {"section": section},
            }
    raise RuntimeError("no untried section left to propose")  # not expected: num_branches == len(_SECTIONS)


SCENARIO = FailureScenario(
    name="ambiguous_fix_candidates",
    description=(
        "Correcting a missing `section` needs trying multiple plausible values for "
        "real — a single guess can be plausible yet wrong, and only actually calling "
        "the tool (not a cached recipe) reveals which one is right."
    ),
    make_tool=make_tool,
    trials=[
        {"desk_number": 300},
        {"desk_number": 301},
        {"desk_number": 302},
        {"desk_number": 303},
        {"desk_number": 304},
        {"desk_number": 305},
    ],
    reflect=reflect,
    num_branches=len(_SECTIONS),
    side_effect_free=True,
)
