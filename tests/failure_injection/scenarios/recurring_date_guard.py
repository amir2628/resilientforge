"""Phase 2 scenario: the "prevention" proof.

Reuses natural_language_date.py's tool/reflect unchanged — the point isn't
a new failure pattern, it's demonstrating that once the SAME failure shape
has recurred enough times, ResilientForge stops merely *recovering* from
it (fail once, fix, retry) and starts *preventing* it outright (the guard
fixes the args before the first attempt, so the call never fails at all).

More trials than natural_language_date.py's 5, deliberately: the first 3
cross the default guard_promotion_min_occurrences threshold (1 reflect
call + 2 fast-path recipe hits), and the remaining 5 use dates never seen
in any prior trial, to prove prevention generalizes to unseen literal
values rather than just replaying whatever was true at promotion time.
"""

from __future__ import annotations

from tests.failure_injection.harness import FailureScenario
from tests.failure_injection.scenarios.natural_language_date import make_tool, reflect

SCENARIO = FailureScenario(
    name="recurring_date_guard",
    description=(
        "Same natural-language-date failure as natural_language_date, but with enough "
        "trials to demonstrate prevention (a standing guard fixing args pre-call) taking "
        "over from reactive recovery, generalizing to dates never seen before."
    ),
    make_tool=make_tool,
    trials=[
        # Trials 1-3: cross the promotion threshold (1 reflect call, then
        # 2 fast-path recipe hits).
        {"date": "next Friday", "title": "Standup"},
        {"date": "next Tuesday", "title": "Retro"},
        {"date": "next Monday", "title": "Planning"},
        # Trials 4-8: fresh dates never used above — proves the guard's
        # prevention generalizes, not just replays a cached value.
        {"date": "next Wednesday", "title": "1:1"},
        {"date": "next Thursday", "title": "All-hands"},
        {"date": "in 2 days", "title": "Sprint Review"},
        {"date": "in 5 days", "title": "Board Meeting"},
        {"date": "in 10 days", "title": "Offsite"},
    ],
    reflect=reflect,
)
