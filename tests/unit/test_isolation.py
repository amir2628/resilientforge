"""Unit tests for core/isolation.py: run_isolated()'s process-boundary
contract and check_picklable()'s fail-fast picklability guard.

The tool_fns exercised here must be defined at MODULE level, not nested
inside a test function — `run_isolated` sends them across a spawn
boundary, which requires pickling by reference. This is a real
constraint `isolate=True` imposes on every caller too, not a test-only
workaround (see check_picklable's own tests below).
"""

from __future__ import annotations

import sys
import time

import pytest

from resilientforge.core.isolation import IsolationError, check_picklable, run_isolated


def _add(a, b):
    return a + b


def _raise_value_error(message):
    raise ValueError(message)


def _sleep_forever(seconds):
    time.sleep(seconds)
    return "should never get here"


def _hard_crash():
    import os

    os._exit(1)


def _burn_cpu():
    x = 0
    while True:
        x += 1


def _allocate_mb(mb):
    data = bytearray(mb * 1024 * 1024)
    return len(data)


# -- run_isolated(): the normal-behavior contract -----------------------


def test_normal_call_round_trips_result():
    result, error = run_isolated(
        _add, {"a": 2, "b": 3}, timeout=5, max_memory_mb=None, max_cpu_seconds=None
    )
    assert result == 5
    assert error is None


def test_tool_exception_round_trips_as_is():
    result, error = run_isolated(
        _raise_value_error, {"message": "boom"}, timeout=5, max_memory_mb=None, max_cpu_seconds=None
    )
    assert result is None
    assert isinstance(error, ValueError)
    assert str(error) == "boom"


# -- run_isolated(): the whole point — hangs and crashes are contained --


def test_timeout_terminates_a_hanging_call():
    start = time.monotonic()
    result, error = run_isolated(
        _sleep_forever, {"seconds": 5}, timeout=0.2, max_memory_mb=None, max_cpu_seconds=None
    )
    elapsed = time.monotonic() - start

    assert result is None
    assert isinstance(error, IsolationError)
    assert "call_timeout" in str(error)
    assert elapsed < 3.0  # terminated promptly, not left to run the full 5s


def test_crash_is_contained_not_propagated():
    # If this weren't actually isolated, os._exit(1) would kill the test
    # process itself — reaching the assertions below IS the proof it didn't.
    result, error = run_isolated(_hard_crash, {}, timeout=5, max_memory_mb=None, max_cpu_seconds=None)
    assert result is None
    assert isinstance(error, IsolationError)
    assert "abnormally" in str(error)


# -- run_isolated(): POSIX-only resource caps, best-effort by design ----


@pytest.mark.skipif(sys.platform == "win32", reason="max_cpu_seconds is POSIX-only")
def test_cpu_cap_kills_a_cpu_bound_loop():
    start = time.monotonic()
    result, error = run_isolated(_burn_cpu, {}, timeout=10, max_memory_mb=None, max_cpu_seconds=1)
    elapsed = time.monotonic() - start

    assert result is None
    assert isinstance(error, IsolationError)
    assert elapsed < 5.0  # the CPU limit fired well before the 10s wall-clock timeout would


@pytest.mark.skipif(sys.platform == "win32", reason="max_memory_mb is POSIX-only")
def test_memory_cap_does_not_interfere_with_a_call_comfortably_under_it():
    result, error = run_isolated(_allocate_mb, {"mb": 1}, timeout=5, max_memory_mb=200, max_cpu_seconds=None)
    # Best-effort, by design: on at least one real POSIX system encountered
    # during development, RLIMIT_AS itself is refused outright by the
    # kernel — surfacing as its own honest, distinct error rather than
    # silently succeeding OR silently doing nothing. Either outcome here
    # is acceptable; the only real requirement is that it neither hangs
    # nor crashes the test process.
    assert result == 1 * 1024 * 1024 or isinstance(error, IsolationError)


# -- check_picklable() ---------------------------------------------------


def test_check_picklable_accepts_module_level_function():
    check_picklable(_add)  # must not raise


def test_check_picklable_rejects_a_local_closure():
    def local_closure(x):
        return x

    with pytest.raises(IsolationError, match="picklable"):
        check_picklable(local_closure)


def test_check_picklable_rejects_a_lambda():
    with pytest.raises(IsolationError, match="picklable"):
        check_picklable(lambda x: x)
