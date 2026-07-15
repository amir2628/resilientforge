"""Integration tests for core/engine.py's Phase 4 process isolation
(`isolate` / `call_timeout` / `max_memory_mb` / `max_cpu_seconds`). Own
file, matching test_speculative_branching.py's precedent of not growing
the already-large test_engine.py.

Every tool_fn here is module-level (not a nested closure) — `isolate=True`
requires it to be picklable across the spawn boundary, a real constraint
on every caller, not a test-only workaround.
"""

from __future__ import annotations

import sys
import time

import pytest

from resilientforge import IsolationError, wrap
from resilientforge.core.recovery import FailureContext


def maybe_hang(mode: str) -> dict:
    if mode == "slow":
        time.sleep(10)
    return {"mode": mode, "status": "done"}


def quick_tool(value: int) -> dict:
    return {"value": value, "status": "done"}


def avoid_slow_path_reflect(context: FailureContext) -> dict:
    return {"strategy": "avoid_slow_path", "argument_patch": {"mode": "fast"}}


# -- the whole point: a real hang recovers instead of blocking forever --


def test_hanging_tool_recovers_via_reflect_after_isolated_timeout(tmp_path):
    wrapped = wrap(
        maybe_hang,
        oracle_path=tmp_path / "oracle",
        reflect=avoid_slow_path_reflect,
        isolate=True,
        call_timeout=1.5,  # comfortably above real subprocess-spawn overhead (~0.3s observed),
    )              # well under the 10s hang, so a fast "mode=fast" retry never spuriously times out

    start = time.monotonic()
    result = wrapped.invoke(mode="slow")
    elapsed = time.monotonic() - start

    assert result == {"mode": "fast", "status": "done"}
    assert elapsed < 8.0  # recovered promptly, never blocked on the full 10s hang

    # The recovery went through the exact same write-back path as any
    # other fix — isolation doesn't bypass it.
    recipe = wrapped.recipes.list()[0]
    assert recipe.times_applied == 1
    assert recipe.times_succeeded == 1


# -- isolate=False (default): byte-for-byte unchanged --------------------


def test_isolate_false_is_the_unmodified_phase_1_2_3_path(tmp_path):
    wrapped = wrap(quick_tool, oracle_path=tmp_path / "oracle")
    result = wrapped.invoke(value=42)
    assert result == {"value": 42, "status": "done"}


# -- misconfiguration warnings --------------------------------------------


def test_call_timeout_without_isolate_warns(tmp_path):
    with pytest.warns(UserWarning, match="call_timeout"):
        wrap(quick_tool, oracle_path=tmp_path / "oracle", call_timeout=1.0)


def test_resource_caps_on_windows_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.warns(UserWarning, match="POSIX-only"):
        wrap(quick_tool, oracle_path=tmp_path / "oracle", isolate=True, max_memory_mb=256)


# -- fail-fast picklability check -----------------------------------------


def test_isolate_true_with_unpicklable_tool_fails_fast_without_cloudpickle(tmp_path, monkeypatch):
    # Simulates the `isolation` extra NOT being installed (see
    # test_isolation.py's unit-level equivalent for why sys.modules
    # patching, not just "don't install it", is how this gets tested —
    # cloudpickle IS a real dev-environment dependency here).
    monkeypatch.setitem(sys.modules, "cloudpickle", None)

    def local_closure(value: int) -> dict:
        return {"value": value}

    with pytest.raises(IsolationError, match="picklable"):
        wrap(local_closure, oracle_path=tmp_path / "oracle", isolate=True)


# -- cloudpickle fallback (Phase 5): isolate=True works for closures too --


def make_validating_closure(valid_values: set) -> object:
    """A closure — exactly what stdlib pickle alone could never isolate.
    Captures `valid_values` by value, read-only, never mutated — a
    real constraint of isolate=True, not just this test's design: each
    call gets a completely fresh, independent subprocess, so mutable
    state a closure captures (a counter, a cache) does NOT persist
    across separate isolated calls the way it would for an ordinary
    in-process closure — only what's captured at cloudpickle-serialization
    time for THAT call is visible, every time. See core/isolation.py's
    module docstring."""

    def validate(value: int) -> dict:
        if value not in valid_values:
            raise ValueError(f"{value} not in {sorted(valid_values)}")
        return {"value": value, "status": "done"}

    return validate


def test_isolate_true_works_end_to_end_with_a_closure_via_cloudpickle(tmp_path):
    wrapped = wrap(
        make_validating_closure({7}),
        oracle_path=tmp_path / "oracle",
        reflect=lambda ctx: {"strategy": "coerce_to_valid_value", "argument_patch": {"value": 7}},
        isolate=True,
        tool_name="validating_closure",
    )

    result = wrapped.invoke(value=999)  # not in {7} -> fails -> reflect corrects it -> retried, isolated, succeeds

    assert result == {"value": 7, "status": "done"}
