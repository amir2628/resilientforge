"""Observability hook for the recovery loop itself (Phase 5) — distinct
from `dashboard/` (which shows the ORACLE's persisted contents: recipes,
guards, failure history after the fact) and distinct from
`tests/failure_injection`'s recovery-rate report (a one-time proof, not
live telemetry). This is for watching a live agent's recovery loop as
it actually runs.

Same vendor-neutral, caller-injects-a-callable pattern as `ReflectFn`
(`core/recovery.py`) and `Invariant.llm_judged`'s `judge` — this module
never imports Prometheus/Datadog/OpenTelemetry/any vendor SDK. This
project explicitly isn't trying to be a full observability/tracing
platform (usable *alongside* Langfuse/Phoenix/LangSmith, not competing
with them) — `MetricsHook` is one small extension point a caller wires
to whatever backend they use; `LoggingMetricsHook` below is a
zero-dependency reference implementation, not the intended production
backend for anyone with real telemetry infrastructure already.

Deliberately NOT exhaustive: this is a known-useful subset of events
(what happened on each real tool call, how a recovery ultimately
resolved, when a guard fired/was promoted/was revoked) — not a full
trace of every internal decision `WrappedAgent` makes. Widen it against
real usage, not speculatively, same discipline `TRANSFORM_REGISTRY`
and `docs/architecture.md`'s embedder section already follow.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

from pydantic import BaseModel


class MetricEvent(BaseModel):
    event_type: Literal[
        "call_result",
        "recovery_resolved",
        "guard_fired",
        "guard_promoted",
        "guard_revoked",
    ]
    tool_name: str
    timestamp: str

    # call_result: one real tool invocation, either the initial attempt
    # or one recovery attempt.
    success: bool | None = None
    error_type: str | None = None
    source: Literal["initial", "recipe", "reflection"] | None = None
    attempt_number: int | None = None

    # recovery_resolved: how one invoke() call that needed recovery
    # ultimately ended.
    resolution: Literal["recovered", "exhausted", "aborted"] | None = None
    total_attempts: int | None = None

    # guard_fired / guard_promoted / guard_revoked
    argument: str | None = None
    kind: str | None = None


MetricsHook = Callable[[MetricEvent], None]


class LoggingMetricsHook:
    """A zero-dependency reference `MetricsHook`, using stdlib
    `logging` — something usable out of the box without forcing a
    vendor choice. `wrap(..., metrics=LoggingMetricsHook())`; configure
    the `resilientforge.metrics` logger the normal stdlib way (handlers,
    formatters, level) to send it wherever you already send logs."""

    def __init__(self, logger_name: str = "resilientforge.metrics", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def __call__(self, event: MetricEvent) -> None:
        self._logger.log(self._level, event.model_dump_json(exclude_none=True))
