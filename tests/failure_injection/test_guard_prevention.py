"""Phase 2's core proof, mirroring test_recovery_rate.py's style: not just
that ResilientForge recovers from a recurring failure, but that it learns
to prevent it outright once the same shape has recurred enough times.

    pytest tests/failure_injection
"""

from __future__ import annotations

from tests.failure_injection.harness import run_scenario
from tests.failure_injection.scenarios import recurring_date_guard


def test_baseline_still_fails_without_resilientforge(tmp_path):
    report = run_scenario(recurring_date_guard.SCENARIO, tmp_path)
    assert report.baseline_recovery_rate == 0.0


def test_guard_is_promoted_and_all_trials_recover(tmp_path):
    report = run_scenario(recurring_date_guard.SCENARIO, tmp_path)

    assert report.recovery_rate == 1.0, f"recovery_rate={report.recovery_rate:.0%}, expected 100%"
    assert report.guard_promoted is True, "expected a standing guard to be promoted by trial 3"


def test_prevention_rate_is_100_percent_once_the_guard_is_active(tmp_path):
    """The whole point of Phase 2: once the guard is active (from trial 4
    onward, per recurring_date_guard.py's trial design — see that file's
    docstring), every remaining trial — including 5 dates never seen in
    any prior trial — must be prevented outright (zero retries), not
    merely recovered from."""
    report = run_scenario(recurring_date_guard.SCENARIO, tmp_path)

    assert report.prevention_rate == 1.0, (
        f"prevention_rate={report.prevention_rate:.0%}, expected 100% for trials "
        "where a guard was already active"
    )
