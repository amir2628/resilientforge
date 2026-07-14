"""Integration tests for core/engine.py's wrap() orchestration:
the oracle-miss reflection path and the oracle-hit
fast path, end to end through Oracle + signature.py + invariants.py +
recovery.py. The model call is always mocked here — no real API calls.

Added because engine.py needs its own coverage before either adapter
exists, ahead of the adapter-specific integration tests.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from resilientforge import GuardManager, Invariant, InvariantAbortError, RecoveryExhaustedError, wrap
from resilientforge.core.recovery import FailureContext, parse_relative_date_to_iso
from resilientforge.oracle import Oracle

# -- example "tools" used across tests ---------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def flaky_create_event(date: str, title: str = "Event") -> dict:
    """Fails unless `date` is already ISO 8601 — simulating the natural-
    language-date failure scenario."""
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


class _EventResult(BaseModel):
    title: str
    attendees: list[str]


def create_event_maybe_missing_attendees(title: str, attendees: list[str] | None = None) -> dict:
    """Never raises, but the result violates the schema invariant when
    `attendees` wasn't provided."""
    if attendees is None:
        return {"title": title}
    return {"title": title, "attendees": attendees}


def dangerous_tool(action: str) -> dict:
    return {"action": action}


# -- reflect stubs -------------------------------------------------------------


def date_fixing_reflect(context: FailureContext) -> dict:
    return {
        "strategy": "reformat_argument",
        "root_cause": "natural-language date string passed where ISO date expected",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


def missing_attendees_reflect(context: FailureContext) -> dict:
    return {"strategy": "add_missing_field", "argument_patch": {"attendees": []}}


def useless_reflect(context: FailureContext) -> dict:
    # Always proposes a no-op fix — used to exercise max_recovery_attempts.
    return {"strategy": "noop", "argument_patch": {}}


class CountingReflect:
    """A reflect stub that records how many times it was called, so tests
    can assert the fast path makes zero model calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[FailureContext] = []

    def __call__(self, context: FailureContext) -> dict:
        self.calls.append(context)
        return self.fn(context)


# -- success paths --------------------------------------------------------------


def test_successful_call_needs_no_recovery(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=reflect)

    result = wrapped.invoke(date="2026-03-05", title="Standup")

    assert result == {"date": "2026-03-05", "title": "Standup", "status": "created"}
    assert reflect.calls == []


# -- oracle-miss reflection path ------------------------------------------------


def test_oracle_miss_falls_back_to_reflection_and_recovers(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=reflect)

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result["status"] == "created"
    assert result["date"] == parse_relative_date_to_iso("next Friday")
    assert len(reflect.calls) == 1

    # A recipe now exists for this failure shape.
    recipes = wrapped.recipes.list()
    assert len(recipes) == 1
    assert recipes[0].times_applied == 1
    assert recipes[0].times_succeeded == 1


def test_reflection_call_receives_failure_context(tmp_path):
    captured = []

    def inspecting_reflect(context: FailureContext) -> dict:
        captured.append(context)
        return date_fixing_reflect(context)

    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=inspecting_reflect)
    wrapped.invoke(date="next Friday", title="Standup")

    assert len(captured) == 1
    context = captured[0]
    assert context.tool_name == "flaky_create_event"
    assert context.args == {"date": "next Friday", "title": "Standup"}
    assert context.error_type == "ValueError"
    assert "next Friday" in context.error_message
    assert context.attempt_number == 1
    assert context.previous_attempts == []


# -- oracle-hit fast path (the headline acceptance criterion) --------------


def test_second_occurrence_resolves_via_oracle_hit_with_zero_model_calls(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=reflect)

    first = wrapped.invoke(date="next Friday", title="Standup")
    assert len(reflect.calls) == 1

    second = wrapped.invoke(date="next Tuesday", title="Retro")

    assert len(reflect.calls) == 1  # NOT called again — recovered via recipe
    assert second["status"] == "created"
    assert second["date"] == parse_relative_date_to_iso("next Tuesday")
    assert second["date"] != first["date"]  # different literal, correctly recomputed per-occurrence

    recipe = wrapped.recipes.list()[0]
    assert recipe.times_applied == 2
    assert recipe.times_succeeded == 2


def test_fast_path_works_with_no_reflect_configured_when_recipe_preexists(tmp_path):
    seeding_wrapped = wrap(
        flaky_create_event, oracle_path=tmp_path / "oracle", reflect=date_fixing_reflect
    )
    seeding_wrapped.invoke(date="next Friday", title="Standup")
    seeding_wrapped.close()

    # A fresh wrap() over the SAME oracle path, with no reflect at all —
    # zero-config recovery from a recipe a prior run already learned.
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=None)
    result = wrapped.invoke(date="next Tuesday", title="Retro")

    assert result["status"] == "created"
    assert result["date"] == parse_relative_date_to_iso("next Tuesday")


def test_no_recipe_and_no_reflect_exhausts_immediately_with_original_error(tmp_path):
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=None)

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        wrapped.invoke(date="next Friday", title="Standup")

    err = exc_info.value
    assert err.attempts == []
    assert isinstance(err.original_error, ValueError)
    assert err.__cause__ is err.original_error


# -- fast path fails re-verification, falls back to reflection -----------------


def test_recipe_replay_failure_falls_back_to_reflection_same_call(tmp_path):
    seeding_wrapped = wrap(
        flaky_create_event, oracle_path=tmp_path / "oracle", reflect=date_fixing_reflect
    )
    seeding_wrapped.invoke(date="next Friday", title="Standup")
    seeding_wrapped.close()

    # A THIRD occurrence with the same failure shape, but a date string the
    # learned transform can't parse — the fast-path replay itself fails,
    # so this must fall back to reflection within the same invoke() call.
    fallback_reflect = CountingReflect(lambda ctx: {"strategy": "literal", "argument_patch": {"date": "2026-01-01"}})
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=fallback_reflect)

    result = wrapped.invoke(date="not-a-real-date-at-all", title="Standup")

    assert result == {"date": "2026-01-01", "title": "Standup", "status": "created"}
    assert len(fallback_reflect.calls) == 1  # reflection WAS needed this time

    recipe = wrapped.recipes.list()[0]
    # 2 successes from the seed run + this run's fast-path failure + this
    # run's reflection success = 3 applies, 2 successes.
    assert recipe.times_applied == 3
    assert recipe.times_succeeded == 2


# -- invariants: recover, abort, warn -------------------------------------------


def test_invariant_violation_without_exception_triggers_recovery(tmp_path):
    invariant = Invariant.from_pydantic_model("valid_event", _EventResult)
    wrapped = wrap(
        create_event_maybe_missing_attendees,
        invariants=[invariant],
        oracle_path=tmp_path / "oracle",
        reflect=missing_attendees_reflect,
    )

    result = wrapped.invoke(title="Standup")

    assert result == {"title": "Standup", "attendees": []}


def test_invariant_abort_raises_immediately_without_reflection(tmp_path):
    reflect = CountingReflect(lambda ctx: {"strategy": "noop"})
    invariant = Invariant(
        name="no_delete", check=lambda r: r.get("action") != "delete", on_violation="abort"
    )
    wrapped = wrap(
        dangerous_tool, invariants=[invariant], oracle_path=tmp_path / "oracle", reflect=reflect
    )

    with pytest.raises(InvariantAbortError) as exc_info:
        wrapped.invoke(action="delete")

    assert exc_info.value.violated == ["no_delete"]
    assert reflect.calls == []


def test_invariant_warn_returns_result_without_recovery(tmp_path):
    reflect = CountingReflect(lambda ctx: {"strategy": "noop"})
    invariant = Invariant(
        name="prefer_no_delete",
        check=lambda r: r.get("action") != "delete",
        on_violation="warn",
    )
    wrapped = wrap(
        dangerous_tool, invariants=[invariant], oracle_path=tmp_path / "oracle", reflect=reflect
    )

    with pytest.warns(UserWarning, match="prefer_no_delete"):
        result = wrapped.invoke(action="delete")

    assert result == {"action": "delete"}
    assert reflect.calls == []


# -- max_recovery_attempts ------------------------------------------------------


def test_max_recovery_attempts_is_respected_and_history_is_attached(tmp_path):
    reflect = CountingReflect(useless_reflect)
    wrapped = wrap(
        flaky_create_event,
        oracle_path=tmp_path / "oracle",
        reflect=reflect,
        max_recovery_attempts=2,
    )

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        wrapped.invoke(date="next Friday", title="Standup")

    assert len(reflect.calls) == 2
    err = exc_info.value
    assert len(err.attempts) == 2
    assert all(attempt.source == "reflection" for attempt in err.attempts)
    assert all(attempt.error_type == "ValueError" for attempt in err.attempts)
    assert err.tool_name == "flaky_create_event"
    assert err.call_args == {"date": "next Friday", "title": "Standup"}


def test_max_recovery_attempts_zero_exhausts_with_no_attempts(tmp_path):
    wrapped = wrap(
        flaky_create_event,
        oracle_path=tmp_path / "oracle",
        reflect=CountingReflect(date_fixing_reflect),
        max_recovery_attempts=0,
    )

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        wrapped.invoke(date="next Friday", title="Standup")

    assert exc_info.value.attempts == []


# -- wrap() accepts both a bare callable and an object with .invoke() ----------


class _ObjectAgent:
    def invoke(self, date: str, title: str = "Event") -> dict:
        return flaky_create_event(date=date, title=title)


def test_wrap_accepts_bare_callable(tmp_path):
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle")
    assert wrapped.tool_name == "flaky_create_event"
    assert wrapped.invoke(date="2026-03-05") == {
        "date": "2026-03-05",
        "title": "Event",
        "status": "created",
    }


def test_wrap_accepts_object_with_invoke_method(tmp_path):
    wrapped = wrap(_ObjectAgent(), oracle_path=tmp_path / "oracle")
    assert wrapped.tool_name == "_ObjectAgent"
    assert wrapped.invoke(date="2026-03-05") == {
        "date": "2026-03-05",
        "title": "Event",
        "status": "created",
    }


def test_wrap_rejects_non_callable_non_agent():
    with pytest.raises(TypeError):
        wrap(object())


def test_wrap_explicit_tool_name_overrides_default(tmp_path):
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", tool_name="custom_name")
    assert wrapped.tool_name == "custom_name"


# -- lifecycle -------------------------------------------------------------


def test_wrapped_agent_context_manager_closes_cleanly(tmp_path):
    with wrap(flaky_create_event, oracle_path=tmp_path / "oracle") as wrapped:
        wrapped.invoke(date="2026-03-05")
    # No assertion beyond "no exception" — verifies close()/__exit__ wiring.


# -- standing guards (Phase 2) --------------------------------------------------


def test_guard_promotes_after_min_occurrences_and_prevents_next_failure(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap(
        flaky_create_event,
        oracle_path=tmp_path / "oracle",
        reflect=reflect,
        guard_promotion_min_occurrences=3,
    )

    wrapped.invoke(date="next Friday", title="A")  # oracle miss -> reflect
    wrapped.invoke(date="next Tuesday", title="B")  # recipe fast path
    wrapped.invoke(date="next Monday", title="C")  # recipe fast path -> 3rd success promotes a guard

    assert len(reflect.calls) == 1  # only the very first occurrence ever needed a model call
    guard = GuardManager(wrapped.oracle).get("flaky_create_event", "date", "transform")
    assert guard is not None
    assert guard.transform == "parse_relative_date_to_iso"

    # A 4th occurrence, a literal never seen before: the guard fires BEFORE
    # the first attempt, so the call succeeds outright — no failure is even
    # recorded for it, and reflect is still not called again.
    result = wrapped.invoke(date="next Wednesday", title="D")

    assert result["status"] == "created"
    assert result["date"] == parse_relative_date_to_iso("next Wednesday")
    assert len(reflect.calls) == 1


def test_guard_transform_fires_and_prevents_failure(tmp_path):
    oracle_path = tmp_path / "oracle"
    oracle = Oracle(oracle_path)
    GuardManager(oracle).promote(
        tool_name="flaky_create_event",
        argument="date",
        kind="transform",
        transform="parse_relative_date_to_iso",
        source_signature="sig-seed",
    )
    oracle.close()

    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap(flaky_create_event, oracle_path=oracle_path, reflect=reflect)

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result["status"] == "created"
    assert reflect.calls == []  # prevented outright — no recovery was ever needed


def test_guard_is_a_noop_when_transform_cannot_handle_the_value(tmp_path):
    oracle_path = tmp_path / "oracle"
    oracle = Oracle(oracle_path)
    GuardManager(oracle).promote(
        tool_name="flaky_create_event",
        argument="date",
        kind="transform",
        transform="parse_relative_date_to_iso",
        source_signature="sig-seed",
    )
    oracle.close()

    # No reflect configured — if the guard corrupted args or crashed instead
    # of being a clean no-op, this wouldn't cleanly exhaust with the
    # ORIGINAL, unmodified value.
    wrapped = wrap(flaky_create_event, oracle_path=oracle_path, reflect=None)

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        wrapped.invoke(date="not a real date at all", title="Standup")

    assert exc_info.value.call_args == {"date": "not a real date at all", "title": "Standup"}


def test_patch_guard_fills_missing_key_but_never_overwrites_provided_value(tmp_path):
    oracle_path = tmp_path / "oracle"
    oracle = Oracle(oracle_path)
    GuardManager(oracle).promote(
        tool_name="create_event_maybe_missing_attendees",
        argument="attendees",
        kind="patch",
        patch_value=[],
        source_signature="sig-seed",
    )
    oracle.close()

    invariant = Invariant.from_pydantic_model("valid_event", _EventResult)
    wrapped = wrap(
        create_event_maybe_missing_attendees,
        invariants=[invariant],
        oracle_path=oracle_path,
        reflect=None,
        tool_name="create_event_maybe_missing_attendees",
    )

    filled = wrapped.invoke(title="Standup")
    assert filled == {"title": "Standup", "attendees": []}

    preserved = wrapped.invoke(title="Retro", attendees=["a@x.com"])
    assert preserved == {"title": "Retro", "attendees": ["a@x.com"]}


def test_enable_standing_guards_false_disables_precall_application(tmp_path):
    oracle_path = tmp_path / "oracle"
    oracle = Oracle(oracle_path)
    GuardManager(oracle).promote(
        tool_name="flaky_create_event",
        argument="date",
        kind="transform",
        transform="parse_relative_date_to_iso",
        source_signature="sig-seed",
    )
    oracle.close()

    wrapped = wrap(
        flaky_create_event,
        oracle_path=oracle_path,
        reflect=None,
        enable_standing_guards=False,
    )

    with pytest.raises(RecoveryExhaustedError):
        wrapped.invoke(date="next Friday", title="Standup")


def test_guard_promotion_is_scoped_per_workflow_when_workflow_id_is_set(tmp_path):
    oracle = Oracle(tmp_path / "oracle")
    reflect = CountingReflect(date_fixing_reflect)

    wrapped_a = wrap(
        flaky_create_event, oracle=oracle, reflect=reflect,
        workflow_id="workflow-a", guard_promotion_min_occurrences=3,
    )
    wrapped_b = wrap(
        flaky_create_event, oracle=oracle, reflect=reflect,
        workflow_id="workflow-b", guard_promotion_min_occurrences=3,
    )

    # Interleaved: 2 occurrences for workflow-a, 1 for workflow-b — the
    # GLOBAL recipe.times_applied reaches 3, but NEITHER workflow alone has
    # hit the threshold yet.
    wrapped_a.invoke(date="next Friday", title="A1")
    wrapped_b.invoke(date="next Tuesday", title="B1")
    wrapped_a.invoke(date="next Monday", title="A2")

    guards = GuardManager(oracle)
    assert guards.get("flaky_create_event", "date", "transform") is None

    # A 3rd occurrence specifically for workflow-a crosses ITS threshold.
    wrapped_a.invoke(date="next Wednesday", title="A3")

    assert guards.get("flaky_create_event", "date", "transform") is not None
    oracle.close()
