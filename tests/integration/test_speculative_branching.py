"""Integration tests for core/engine.py's Phase 3 speculative branching
(`num_branches` / `side_effect_free`). Own file, not grown into the
already-large test_engine.py, per the Phase 3 plan — test_engine.py's
existing scenarios already cover the `num_branches<=1` path exhaustively;
everything here exercises the NEW candidate-batch machinery layered on
top of it, and the shared `_on_attempt_success`/`_on_attempt_failure`
tail through that new path specifically.

Reminder for every call-count assertion below: `invoke()` always makes
one unconditional real call with the caller's original arguments BEFORE
the recovery loop starts (existing Phase 1/2 behavior, unchanged here) —
so "N real calls" in a test means 1 initial + N-1 from recovery, not N
recovery calls alone.
"""

from __future__ import annotations

import re

import pytest

from resilientforge import GuardManager, Invariant, RecoveryExhaustedError, wrap
from resilientforge.core.recovery import FailureContext

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEAT_RE = re.compile(r"^\d{1,2}[A-F]$")
_SOLD_OUT_SEATS = {"12A", "13A", "14A"}


def _counting(fn):
    """Wrap `fn` to record every REAL invocation, including ones that
    raise — the thing every test here needs to assert on: how many times
    the actual tool ran, not how many candidates were merely considered."""
    calls: list[dict] = []

    def wrapped(**kwargs):
        calls.append(dict(kwargs))
        return fn(**kwargs)

    wrapped.__name__ = fn.__name__
    return wrapped, calls


def flaky_create_event(date: str, title: str = "Event") -> dict:
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


def book_seat(seat: str) -> dict:
    if not _SEAT_RE.match(seat):
        raise ValueError(f"invalid seat format: {seat!r}")
    return {"seat": seat, "status": "booked"}


seat_available = Invariant(
    name="seat_available",
    check=lambda r: r["seat"] not in _SOLD_OUT_SEATS,
    on_violation="recover",
)


def _patch(strategy: str, **argument_patch) -> dict:
    return {"strategy": strategy, "argument_patch": argument_patch}


class SequentialReflect:
    """Returns each fix in `fixes` in order across ALL calls (never
    resets per round) — lets a test span multiple candidate-generation
    rounds with fully predictable, non-repeating proposals. Raises
    IndexError if asked for more fixes than provided — a test running
    past its own fixture is a bug in the test, not something to hide."""

    def __init__(self, fixes: list[dict]) -> None:
        self.fixes = fixes
        self.calls: list[FailureContext] = []

    def __call__(self, context: FailureContext) -> dict:
        self.calls.append(context)
        return self.fixes[len(self.calls) - 1]


# -- the safety boundary: never more than one real call per attempt_number ----


def test_default_never_calls_the_tool_more_than_once_per_attempt(tmp_path):
    tool, calls = _counting(flaky_create_event)
    reflect = SequentialReflect(
        [
            _patch("a", date="bad-1"), _patch("b", date="bad-2"), _patch("c", date="bad-3"),
            _patch("d", date="bad-4"), _patch("e", date="bad-5"), _patch("f", date="bad-6"),
        ]
    )
    wrapped = wrap(
        tool, oracle_path=tmp_path / "oracle", reflect=reflect,
        num_branches=3, side_effect_free=False, max_recovery_attempts=2,
    )

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        wrapped.invoke(date="next Friday", title="Standup")

    assert len(reflect.calls) == 6  # 3 candidates generated per round x 2 rounds
    # 1 initial call + 1 real call per round (never num_branches x that):
    assert len(calls) == 1 + 2
    assert len(exc_info.value.attempts) == 2  # one RecoveryAttempt per round, not per candidate


def test_num_branches_one_is_byte_for_byte_the_phase_1_2_path(tmp_path):
    tool, calls = _counting(flaky_create_event)
    reflect = SequentialReflect([_patch("fix", date="2026-03-05")])
    wrapped = wrap(tool, oracle_path=tmp_path / "oracle", reflect=reflect, num_branches=1)

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result == {"date": "2026-03-05", "title": "Standup", "status": "created"}
    assert len(reflect.calls) == 1  # single-candidate behavior, no batching
    assert len(calls) == 1 + 1  # initial failing call + the one recovery call


# -- proxy ranking (side_effect_free=False, the default) ----------------------


def test_high_success_rate_recipe_is_preferred_over_a_competing_candidate(tmp_path):
    oracle_path = tmp_path / "oracle"
    seed_tool, _ = _counting(flaky_create_event)
    seed_reflect = SequentialReflect([_patch("seed", date="2026-01-01")])
    seed = wrap(seed_tool, oracle_path=oracle_path, reflect=seed_reflect)
    for i in range(5):  # enough successes that recipe.success_rate == 1.0
        seed.invoke(date="next Friday", title=f"seed-{i}")
    seed.close()

    tool, calls = _counting(flaky_create_event)
    reflect = SequentialReflect([_patch("competitor", date="2099-12-31")])
    wrapped = wrap(
        tool, oracle_path=oracle_path, reflect=reflect, num_branches=2, side_effect_free=False,
    )

    result = wrapped.invoke(date="next Tuesday", title="Standup")

    assert result["date"] == "2026-01-01"  # the RECIPE's fix won, not the reflection candidate
    assert len(calls) == 1 + 1  # only ever one real recovery call, even with 2 candidates


def test_purely_reflection_candidates_use_generation_order_as_tie_break(tmp_path):
    tool, calls = _counting(flaky_create_event)
    reflect = SequentialReflect(
        [_patch("first", date="2026-02-02"), _patch("second", date="2026-03-03")]
    )
    wrapped = wrap(
        tool, oracle_path=tmp_path / "oracle", reflect=reflect, num_branches=2, side_effect_free=False,
    )

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result["date"] == "2026-02-02"  # first-generated candidate wins the tie-break
    assert len(reflect.calls) == 2
    assert len(calls) == 1 + 1


# -- side_effect_free=True: real calls, real verification ---------------------


def test_side_effect_free_tries_candidates_for_real_until_one_passes_invariants(tmp_path):
    tool, calls = _counting(book_seat)
    reflect = SequentialReflect(
        [
            _patch("sold_out", seat="12A"),   # applies cleanly, but fails a REAL invariant
            _patch("available", seat="12B"),  # applies cleanly and passes
        ]
    )
    wrapped = wrap(
        tool, invariants=[seat_available], oracle_path=tmp_path / "oracle", reflect=reflect,
        num_branches=2, side_effect_free=True,
    )

    result = wrapped.invoke(seat="bad seat")

    assert result == {"seat": "12B", "status": "booked"}
    # 1 initial (invalid format) + both candidates actually invoked for real:
    assert len(calls) == 3
    assert calls[1]["seat"] == "12A"
    assert calls[2]["seat"] == "12B"


def test_cross_iteration_context_does_not_include_untested_candidates(tmp_path):
    tool, calls = _counting(flaky_create_event)
    reflect = SequentialReflect(
        [
            _patch("r1_tried", date="bad-tried"),       # round 1 generation-order winner: real call, fails
            _patch("r1_untested", date="bad-untested"), # round 1 alternative: NEVER called for real
            _patch("r2_tried", date="2026-05-05"),       # round 2 winner: real call, succeeds
            _patch("r2_untested", date="bad-r2"),
        ]
    )
    wrapped = wrap(
        tool, oracle_path=tmp_path / "oracle", reflect=reflect,
        num_branches=2, side_effect_free=False, max_recovery_attempts=2,
    )

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result["date"] == "2026-05-05"
    assert len(calls) == 3  # 1 initial + 1 real call per round, never per candidate

    # Round 2's FIRST reflect() call must see EXACTLY the one round-1 fix
    # that was actually executed and failed — not the round-1 alternative
    # that was merely considered but never called for real.
    first_round_2_context = reflect.calls[2]
    assert len(first_round_2_context.previous_attempts) == 1
    assert first_round_2_context.previous_attempts[0].argument_patch == {"date": "bad-tried"}


# -- misconfiguration -----------------------------------------------------------


def test_side_effect_free_with_default_num_branches_warns(tmp_path):
    tool, _ = _counting(flaky_create_event)
    with pytest.warns(UserWarning, match="side_effect_free"):
        wrap(tool, oracle_path=tmp_path / "oracle", side_effect_free=True)


# -- recipe / guard write-back after a speculative win -------------------------


def test_speculative_win_writes_back_a_recipe_and_can_promote_a_guard(tmp_path):
    tool, _ = _counting(flaky_create_event)
    reflect = SequentialReflect(
        [_patch("good", date="2026-04-04"), _patch("bad", date="bad")] * 5
    )
    wrapped = wrap(
        tool, oracle_path=tmp_path / "oracle", reflect=reflect,
        num_branches=2, side_effect_free=False, guard_promotion_min_occurrences=3,
    )

    wrapped.invoke(date="next Friday", title="A")
    wrapped.invoke(date="next Tuesday", title="B")
    wrapped.invoke(date="next Monday", title="C")

    recipe = wrapped.recipes.list()[0]
    assert recipe.times_applied == 3
    assert recipe.times_succeeded == 3

    guard = GuardManager(wrapped.oracle).get(wrapped.tool_name, "date", "patch")
    assert guard is not None
    assert guard.patch_value == "2026-04-04"
