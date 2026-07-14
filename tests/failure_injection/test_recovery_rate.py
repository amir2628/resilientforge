"""The failure-injection suite — the project's core
proof, not an afterthought. Required before merging engine changes:

    pytest tests/failure_injection

Writes a stable-format report to tests/failure_injection/reports/latest.md
(gitignored — regenerated on every run) so the README quickstart can
pull real, current numbers rather than hand-written claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.failure_injection.harness import ScenarioReport, format_report, run_scenario
from tests.failure_injection.scenarios import (
    ambiguous_fix_candidates,
    malformed_json_args,
    missing_required_field,
    natural_language_date,
    recurring_date_guard,
    transient_timeout,
    wrong_type_argument,
)

SCENARIOS = [
    malformed_json_args.SCENARIO,
    missing_required_field.SCENARIO,
    natural_language_date.SCENARIO,
    transient_timeout.SCENARIO,
    wrong_type_argument.SCENARIO,
    recurring_date_guard.SCENARIO,
    ambiguous_fix_candidates.SCENARIO,
]

REPORT_PATH = Path(__file__).parent / "reports" / "latest.md"


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> list[ScenarioReport]:
    results = [
        run_scenario(scenario, tmp_path_factory.mktemp(scenario.name)) for scenario in SCENARIOS
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(results) + "\n")
    print("\n" + format_report(results))
    return results


def test_baseline_fails_without_resilientforge(reports: list[ScenarioReport]) -> None:
    """The "before" half of the proof (measurable recovery-rate
    improvement over an unwrapped baseline): every trial here is a
    deliberately malformed call, so with no recovery mechanism at all,
    none of them should succeed."""
    for report in reports:
        assert report.baseline_recovery_rate == 0.0, (
            f"{report.name}: expected 0% baseline recovery, got "
            f"{report.baseline_recovery_rate:.0%}"
        )


def test_recovery_rate_with_resilientforge(reports: list[ScenarioReport]) -> None:
    for report in reports:
        assert report.recovery_rate == 1.0, (
            f"{report.name}: recovery_rate={report.recovery_rate:.0%}, expected 100%"
        )


def test_oracle_hit_rate_approaches_100_percent_after_first_occurrence(
    reports: list[ScenarioReport],
) -> None:
    """This last number is the whole point — it should approach
    100% after the first successful recovery of a given failure shape.

    Exempts num_branches>1 scenarios (currently just
    ambiguous_fix_candidates): speculative branching deliberately
    re-consults reflect() every round to fill its candidate batch, even
    when a recipe already exists — see that scenario's own docstring and
    harness.py's avg_candidates_considered. That's a documented cost of
    considering more options, not a fast-path regression.
    """
    for scenario, report in zip(SCENARIOS, reports):
        if scenario.num_branches > 1:
            continue
        assert report.oracle_hit_rate_after_first == 1.0, (
            f"{report.name}: oracle_hit_rate_after_first="
            f"{report.oracle_hit_rate_after_first:.0%}, expected 100%"
        )
