"""Failure-injection scenario: natural-language date where ISO 8601 was
expected — the exact example that motivates signature
normalization in the first place ("next Friday" and "next Tuesday" must
normalize to the same failure shape). Recovered via the
parse_relative_date_to_iso transform, which recomputes the correct ISO
date from each trial's own literal value rather than replaying a cached
answer — see core/recovery.py's module docstring for why that matters.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from resilientforge.core.recovery import FailureContext
from tests.failure_injection.harness import FailureScenario

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def make_tool() -> Callable[..., Any]:
    def create_event(date: str, title: str = "Event") -> dict:
        if not _ISO_DATE_RE.match(date):
            raise ValueError(f"could not parse date '{date}'")
        return {"date": date, "title": title, "status": "created"}

    return create_event


def reflect(context: FailureContext) -> dict:
    return {
        "strategy": "reformat_argument",
        "root_cause": "natural-language date string passed where ISO date expected",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


SCENARIO = FailureScenario(
    name="natural_language_date",
    description="Natural-language date string ('next Friday') where ISO 8601 was expected.",
    make_tool=make_tool,
    # Deliberately multi-word phrases only. A single bare word like
    # "tomorrow" hits a *different, already-documented* edge case in
    # core/signature.py's redaction heuristic (see
    # test_single_word_date_value_is_not_collapsed_like_multi_word_ones in
    # test_signature.py) — mixing that concern into this scenario's trials
    # would conflate two different things this suite is meant to measure
    # separately, rather than actually demonstrate a bug in this scenario.
    trials=[
        {"date": "next Friday", "title": "Standup"},
        {"date": "next Tuesday", "title": "Retro"},
        {"date": "in 3 days", "title": "Planning"},
        {"date": "next Monday", "title": "1:1"},
        {"date": "next Wednesday", "title": "All-hands"},
    ],
    reflect=reflect,
)
