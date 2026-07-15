"""Process-level isolation for real tool-call invocations (Phase 4).

Scope, stated plainly: this protects the HOST process from a tool call
that hangs or crashes — a timeout or a crash becomes a normal, recoverable
failure instead of taking down the caller. It does **not**, and cannot,
undo a real-world side effect the tool already performed before it hung
or crashed (an HTTP request already sent stays sent) — no code-level
sandbox can do that. See `core/engine.py`'s `isolate` docstring for the
full caller-facing framing; this module is the mechanism, not the policy.

Every tool call dispatched here runs in a freshly-spawned subprocess
(never "fork" — the parent may hold open sqlite3/chromadb connections,
via its own `Oracle`, that must not be duplicated into the child) via the
stdlib `multiprocessing` module directly, not
`concurrent.futures.ProcessPoolExecutor`: a pooled executor's `Future.
cancel()` cannot stop a task that has already started running, which is
exactly the case a timeout needs to handle — `multiprocessing.Process`
exposes a documented, public `terminate()`/`kill()` instead. A fresh
process per call, never reused, so one crashed or resource-limited call
can never poison a later one.
"""

from __future__ import annotations

import multiprocessing
import pickle
import sys
from queue import Empty as _QueueEmpty
from typing import Any, Callable

_RESULT_WAIT_SECONDS = 2.0  # grace period for a queue write to flush after the child exits
_TERMINATE_GRACE_SECONDS = 2.0  # grace period between SIGTERM and escalating to SIGKILL


class IsolationError(Exception):
    """A tool call timed out, crashed its subprocess, or breached a
    resource cap. Normalized so it flows through
    `WrappedAgent._classify_failure` exactly like any other tool
    exception — never means the tool's own real-world side effect (if
    any) was undone."""


def check_picklable(tool_fn: Callable[..., Any]) -> None:
    """Fail fast — at `WrappedAgent` construction, not on the first
    call — if `tool_fn` can't be pickled. `isolate=True` dispatches to a
    spawned subprocess, which requires pickling the target across the
    process boundary: a locally-defined closure or a lambda will not
    work here; a module-level function or a bound method on a picklable
    object will."""
    try:
        pickle.dumps(tool_fn)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise IsolationError(
            f"isolate=True requires tool_fn to be picklable (it runs in a "
            f"separate process), but {tool_fn!r} is not picklable: {exc}. "
            f"A closure or lambda won't work here — use a module-level "
            f"function or a bound method on a picklable object."
        ) from exc


def _apply_resource_limits(max_memory_mb: int | None, max_cpu_seconds: float | None) -> None:
    """Runs inside the spawned child, before `tool_fn` is called.
    POSIX-only (`resource` doesn't exist on Windows) — the caller
    (`core/engine.py`) is responsible for warning at construction time
    if these are requested on an unsupported platform; this function
    silently does nothing there, since by the time we're in the child,
    a warning can no longer usefully reach the caller."""
    if sys.platform == "win32":
        return
    if max_memory_mb is None and max_cpu_seconds is None:
        return
    import resource

    if max_memory_mb is not None:
        limit_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    if max_cpu_seconds is not None:
        limit_seconds = max(int(max_cpu_seconds), 1)
        resource.setrlimit(resource.RLIMIT_CPU, (limit_seconds, limit_seconds))


def _worker(
    result_queue: multiprocessing.Queue,
    tool_fn: Callable[..., Any],
    args: dict[str, Any],
    max_memory_mb: int | None,
    max_cpu_seconds: float | None,
) -> None:
    """The child process's entire body. Module-level (not nested inside
    `run_isolated`) because `multiprocessing`'s spawn context must be
    able to pickle this function itself to send it to the child."""
    try:
        _apply_resource_limits(max_memory_mb, max_cpu_seconds)
    except Exception as exc:
        # Applying the cap itself failed (platform/kernel refused the
        # limit — e.g. some POSIX systems reject certain RLIMIT_AS
        # values outright) — this is an infra problem, not the tool's
        # fault, so it's tagged distinctly and never attributed to it.
        result_queue.put(("limit_error", exc))
        return
    try:
        result = tool_fn(**args)
        result_queue.put(("ok", result))
    except Exception as exc:  # the tool's own exception — sent back as data, not raised here
        result_queue.put(("error", exc))


def run_isolated(
    tool_fn: Callable[..., Any],
    args: dict[str, Any],
    *,
    timeout: float | None,
    max_memory_mb: int | None,
    max_cpu_seconds: float | None,
) -> tuple[Any, Exception | None]:
    """Runs `tool_fn(**args)` in a freshly-spawned subprocess, enforcing
    `timeout` and, POSIX-only, `max_memory_mb`/`max_cpu_seconds`.

    Mirrors `WrappedAgent._call`'s contract exactly: always returns
    `(result, None)` on success or `(None, exception)` on failure, never
    raises. A timeout, a subprocess crash (segfault, `os._exit`, a
    resource-limit signal like `SIGXCPU`), or a dispatch-layer failure
    (e.g. unpicklable args) is normalized into an `IsolationError`; the
    tool's OWN exception, if it simply raised normally inside the
    subprocess, is returned as-is (it must itself be picklable to survive
    the trip back — an exotic custom exception holding unpicklable state
    is the one case this can't faithfully reproduce).
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(
        target=_worker, args=(result_queue, tool_fn, args, max_memory_mb, max_cpu_seconds)
    )

    try:
        process.start()
    except Exception as exc:
        return None, IsolationError(f"could not dispatch tool call to an isolated subprocess: {exc}")

    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
        return None, IsolationError(
            f"tool call exceeded call_timeout={timeout}s and was terminated"
        )

    if process.exitcode != 0:
        return None, IsolationError(
            f"tool call's subprocess exited abnormally (exit code {process.exitcode}) "
            f"— likely a crash, or a resource limit that killed it outright rather "
            f"than raising a Python exception"
        )

    try:
        status, payload = result_queue.get(timeout=_RESULT_WAIT_SECONDS)
    except _QueueEmpty:
        return None, IsolationError(
            "tool call's subprocess exited cleanly but produced no result"
        )

    if status == "ok":
        return payload, None
    if status == "limit_error":
        return None, IsolationError(
            f"could not apply resource limits (max_memory_mb={max_memory_mb}, "
            f"max_cpu_seconds={max_cpu_seconds}) on this platform: {payload}. "
            f"Resource caps are best-effort — the OS/kernel ultimately decides "
            f"whether a given limit is honorable at all."
        )
    return None, payload  # status == "error": the tool's own exception, as-is
