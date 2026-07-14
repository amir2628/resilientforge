"""Shared harness for the failure-injection suite:
the FailureScenario contract each scenarios/*.py file implements, the
runner that executes baseline-vs-wrapped trials and derives the report
metrics, and the report formatter.

Added because the five scenario files and test_recovery_rate.py need a
shared contract/runner rather than reimplementing the same
instrumentation five times over (same reasoning as
tests/integration/test_engine.py).

Metrics: recovery rate, average attempts-to-recovery, and oracle
hit rate after the first occurrence of each scenario's failure shape —
this last number is the whole point — it should approach 100% after the
first successful recovery of a given failure shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from resilientforge import Invariant, wrap
from resilientforge.core.recovery import FailureContext


@dataclass
class FailureScenario:
    name: str
    description: str
    # A *factory*, not a bare function: several scenarios (transient
    # timeout) need fresh per-run state, and baseline vs. wrapped runs
    # must never share that state.
    make_tool: Callable[[], Callable[..., Any]]
    trials: list[dict[str, Any]]
    reflect: Callable[[FailureContext], dict]
    invariants: list[Invariant] = field(default_factory=list)


@dataclass
class ScenarioReport:
    name: str
    trial_count: int
    baseline_recovery_rate: float
    recovery_rate: float
    avg_attempts_to_recovery: float
    oracle_hit_rate_after_first: float


class _CallCounter:
    """Counts calls to the underlying tool, so we can derive how many
    retry attempts a successful trial needed (total calls - 1)."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.fn(*args, **kwargs)


class _ReflectCounter:
    """Counts reflect() calls, so we can tell whether a given trial was
    resolved via the fast path (oracle hit, zero model calls) or needed
    reflection — the same instrumentation pattern used throughout
    tests/integration/test_engine.py and test_raw_tool_loop.py."""

    def __init__(self, fn: Callable[[FailureContext], dict]) -> None:
        self.fn = fn
        self.count = 0

    def __call__(self, context: FailureContext) -> dict:
        self.count += 1
        return self.fn(context)


def _run_baseline(scenario: FailureScenario) -> float:
    """The "before" number: call the raw, unwrapped tool directly over the
    same trials, with no recovery mechanism at all."""
    tool = scenario.make_tool()
    successes = 0
    for trial in scenario.trials:
        try:
            result = tool(**trial)
        except Exception:
            continue
        if all(inv.evaluate(result) for inv in scenario.invariants):
            successes += 1
    return successes / len(scenario.trials)


def run_scenario(scenario: FailureScenario, oracle_path: Path) -> ScenarioReport:
    """The "after" run: a fresh, empty oracle at `oracle_path`, `trials`
    executed in order through a single wrap()'d tool, so trials after the
    first can actually exercise the fast path."""
    tool = scenario.make_tool()
    call_counter = _CallCounter(tool)
    reflect_counter = _ReflectCounter(scenario.reflect)

    wrapped = wrap(
        call_counter,
        invariants=scenario.invariants,
        oracle_path=oracle_path,
        reflect=reflect_counter,
        tool_name=scenario.name,
    )

    successes = 0
    attempts_per_success: list[int] = []
    fast_path_hits = 0

    for index, trial in enumerate(scenario.trials):
        calls_before = call_counter.count
        reflect_before = reflect_counter.count
        try:
            wrapped.invoke(**trial)
        except Exception:
            continue
        successes += 1
        attempts_per_success.append(call_counter.count - calls_before - 1)
        if index > 0 and reflect_counter.count == reflect_before:
            fast_path_hits += 1

    wrapped.close()

    later_trials = len(scenario.trials) - 1
    return ScenarioReport(
        name=scenario.name,
        trial_count=len(scenario.trials),
        baseline_recovery_rate=_run_baseline(scenario),
        recovery_rate=successes / len(scenario.trials),
        avg_attempts_to_recovery=(
            sum(attempts_per_success) / len(attempts_per_success) if attempts_per_success else 0.0
        ),
        oracle_hit_rate_after_first=(fast_path_hits / later_trials) if later_trials else 0.0,
    )


def format_report(reports: list[ScenarioReport]) -> str:
    header = (
        "| Scenario | Trials | Baseline recovery | Recovery (ResilientForge) "
        "| Avg attempts to recovery | Oracle hit rate (after 1st occurrence) |"
    )
    separator = "|---|---|---|---|---|---|"
    rows = [
        f"| {r.name} | {r.trial_count} | {r.baseline_recovery_rate:.0%} | "
        f"{r.recovery_rate:.0%} | {r.avg_attempts_to_recovery:.1f} | "
        f"{r.oracle_hit_rate_after_first:.0%} |"
        for r in reports
    ]
    return "\n".join([header, separator, *rows])
