"""LangGraph adapter: composes with LangGraph's own
`handle_tool_errors` and `RetryPolicy` rather than replacing them, via the
`wrap_tool_call` extension point `ToolNode` exposes (LangGraph 1.x).

How `handle_tool_errors` composes (verified empirically against langgraph
1.2, not assumed — see tests/integration/test_langgraph_adapter.py):
`wrap_tool_call` receives `(request, execute)`, where `execute(request)`
runs the underlying tool once. Its failure behavior depends entirely on
the `handle_tool_errors` the *underlying* `ToolNode` was built with:
- `handle_tool_errors=True` (LangGraph's own default): `execute()` does
  NOT raise — it catches the tool's exception itself and returns a
  `ToolMessage` with `status="error"`.
- `handle_tool_errors=False`: `execute()` raises the raw exception
  directly.
This wrapper handles both: a small `_tool_fn` shim normalizes an error
`ToolMessage` into a raised `_ToolCallError`, so a *raw* exception and a
*handle_tool_errors-caught* one both flow through the exact same
failure-detection path in core/engine.py — no duplicated recovery logic
here, `wrap()` from core/engine.py does the real work.

IMPORTANT gotcha, also verified empirically: `handle_tool_errors=True`
doesn't just affect `execute()` — `ToolNode` catches ANY exception raised
out of `wrap_tool_call` as a whole and reformats it into a graceful error
`ToolMessage`, silently, before it ever reaches the graph. That means
`on_exhausted="raise"` and `InvariantAbortError` always-propagates (below)
only actually work if the *underlying* `ToolNode` was built with
`handle_tool_errors=False`. `make_tool_node` defaults to
`handle_tool_errors=False` for exactly this reason — it's a deliberate
deviation from LangGraph's own default (True), not an oversight, because
this adapter's whole point is to be the failure-handling layer; deferring
to LangGraph's separate blanket catch-all on top would silently defeat
the abort/raise guarantees documented below. If you build your own
`ToolNode` around `make_resilientforge_tool_call_wrapper` instead, you
must set `handle_tool_errors=False` yourself to get the same guarantees.

How `RetryPolicy` composes: `RetryPolicy` is a *node*-level, blind retry
(no context, no argument correction — the "in-run retry" this project's
motivating failure patterns note frameworks already do well) set via
`graph.add_node(name, node, retry_policy=RetryPolicy(...))`. Once
ResilientForge's own recovery is exhausted (`RecoveryExhaustedError`),
`on_exhausted` decides what happens next: `"error_message"` (default)
returns a graceful error `ToolMessage` so the graph continues and the
model sees the failure; `"raise"` re-raises, so a `RetryPolicy` on this
node gets a chance to re-invoke the whole node (and this wrapper) as a
last-resort safety net. Note LangGraph's own `retry_on` default excludes
ValueError/TypeError — it's scoped to transient failures by design, so it
naturally doesn't compete with ResilientForge's data-correction recovery
for most failure shapes; pass an explicit `retry_on` if you want it to
catch ResilientForge's own exhaustion too.

`InvariantAbortError` is different from `RecoveryExhaustedError` and
always propagates regardless of `on_exhausted`: "abort" is an explicit
signal the caller chose over "recover"/"warn", and LangGraph (unlike the
raw Anthropic/OpenAI tool loop, which has no equivalent "halt everything"
pathway) can meaningfully act on a propagated exception — softening it
into a tool message the model might just shrug off would undermine why
abort was chosen in the first place.

Phase 4's `isolate` (subprocess-based timeout/crash isolation, see
core/engine.py's `wrap()`) is deliberately **not exposed here** — a real,
structural limitation, not an oversight. `_tool_fn_from_request` below
builds a closure over `execute`/`request` for every tool call, and
`execute` is LangGraph's own live callback, bound to in-process graph
state (checkpointer, tool registry) that cannot be pickled or
meaningfully reconstructed in a separate process — `isolate=True` would
fail `check_picklable` on literally every call. This is unlike
`wrap_tools()` in `integrations/raw_tool_loop.py`, where the wrapped
`tool_fn` is the caller's own plain function, not a closure this module
manufactures, so `isolate=True` works there exactly as it does calling
`wrap()` directly.

Async tools (`make_resilientforge_async_tool_call_wrapper`, wired into
`make_tool_node` alongside the sync wrapper via `awrap_tool_call`): found
and fixed via a real-world validation exercise (`validation/`,
`docs/real_world_validation.md`) against an actual external LangGraph
agent, not by inspection — the sync-only wrapper shipped for all of
Phases 1-5 without ever being tested against an async tool function, and
broke unconditionally (`"StructuredTool does not support sync
invocation"`) on the very first call to one. Root cause: LangGraph's
`ToolNode` exposes both a sync `wrap_tool_call` and an async
`awrap_tool_call` hook, but silently forces the sync path — which cannot
run an async-only tool at all — whenever only `wrap_tool_call` is
registered and the graph is invoked via `graph.ainvoke()` (the ordinary
way to run an agent, and the *only* way to run one containing an
async-only tool, since `graph.invoke()` can't either — a pre-existing
LangGraph-level constraint, not something either wrapper function here
introduces or can paper over). See
`make_resilientforge_async_tool_call_wrapper`'s docstring for how the fix
bridges LangGraph's async `execute()` into `core/engine.py`'s
deliberately-synchronous `wrap()`/`WrappedAgent.invoke()` without adding
any async path to `core/engine.py`/`core/recovery.py` themselves.

`valid_arguments` (both wrapper functions, via `_valid_arguments_from_request`):
another real bug found the same way (`docs/real_world_validation_round2.md`,
`docs/real_world_validation_round3.md`) — a reflection-proposed fix
referencing an argument the tool doesn't actually accept (via
`argument_patch`, e.g. adding `headers` to a tool with only a `url`
parameter — or, found in a later confirmation run, via `transforms[].argument`
targeting the same kind of nonexistent parameter) used to get silently
dropped/skipped, then whatever happened next (success or failure, for
reasons entirely unrelated to the fix) got recorded as if it had worked.
`core/engine.py`'s `wrap()` now rejects any such fix — and one naming an
unregistered `transforms[].transform` — as a whole, before ever calling
the tool or persisting a recipe; see `WrappedAgent._invalid_fix_reasons`.
This module passes `valid_arguments` explicitly (derived from LangChain's
own `BaseTool.args`) because the closures built here are generic
`**kwargs`-accepting shims; introspecting one of those directly (what
`wrap()` falls back to otherwise) would reveal nothing about the real
tool's schema.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Sequence

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

from resilientforge import InvariantAbortError, RecoveryExhaustedError, wrap
from resilientforge.core.invariants import Invariant
from resilientforge.core.recovery import ReflectFn
from resilientforge.oracle import Oracle
from resilientforge.telemetry.metrics import MetricsHook

ExecuteFn = Callable[[ToolCallRequest], Any]
ToolCallWrapperFn = Callable[[ToolCallRequest, ExecuteFn], Any]
AsyncExecuteFn = Callable[[ToolCallRequest], Awaitable[Any]]
AsyncToolCallWrapperFn = Callable[[ToolCallRequest, AsyncExecuteFn], Awaitable[Any]]


class _ToolCallError(Exception):
    """Normalizes an error ToolMessage (handle_tool_errors=True already
    caught the real exception and formatted it as text) into a raised
    exception, so it flows through core/engine.py's existing failure-
    detection path the same way a directly-raised exception would."""


def _tool_fn_from_request(execute: ExecuteFn, request: ToolCallRequest) -> Callable[..., Any]:
    def _tool_fn(**kwargs: Any) -> Any:
        new_request = request.override(tool_call={**request.tool_call, "args": kwargs})
        result = execute(new_request)
        if isinstance(result, ToolMessage) and result.status == "error":
            raise _ToolCallError(str(result.content))
        return result

    return _tool_fn


def _tool_fn_from_async_request(
    execute: AsyncExecuteFn, request: ToolCallRequest
) -> Callable[..., Any]:
    """Same shape as `_tool_fn_from_request`, but bridges an async `execute`
    into the plain sync callable `wrap()` needs, via `asyncio.run` — safe
    ONLY because `make_resilientforge_async_tool_call_wrapper` (the sole
    caller) always runs the whole synchronous `wrap()`/`WrappedAgent.invoke()`
    call inside `asyncio.to_thread`, i.e. a fresh thread with no
    already-running event loop of its own for `asyncio.run` to collide with.
    """

    def _tool_fn(**kwargs: Any) -> Any:
        new_request = request.override(tool_call={**request.tool_call, "args": kwargs})
        result = asyncio.run(execute(new_request))
        if isinstance(result, ToolMessage) and result.status == "error":
            raise _ToolCallError(str(result.content))
        return result

    return _tool_fn


def _valid_arguments_from_request(request: ToolCallRequest) -> set[str] | None:
    """The real tool's accepted parameter names, read from LangChain's own
    `BaseTool.args` (a dict of {param_name: schema}) — used so `wrap()`
    can reject a Fix's `argument_patch` key that isn't one of them (see
    docs/real_world_validation_round2.md). Needed here specifically
    because `_tool_fn_from_request`/`_tool_fn_from_async_request` build a
    generic `**kwargs`-accepting closure per call — introspecting THAT
    closure's own signature (what `wrap()` falls back to when
    `valid_arguments` isn't given) would reveal nothing about the real
    tool's schema. Returns `None` (unknown — don't validate) if `request.tool`
    is unset or doesn't expose `.args`, same "unknown, don't block" default
    `wrap()` itself uses."""
    tool = request.tool
    args = getattr(tool, "args", None)
    if not isinstance(args, dict):
        return None
    return set(args)


def make_resilientforge_tool_call_wrapper(
    invariants: list[Invariant] | None = None,
    oracle_path: str | Path = ".resilientforge",
    max_recovery_attempts: int = 3,
    reflect: ReflectFn | None = None,
    similarity_threshold: float = 0.85,
    workflow_id: str | None = None,
    oracle: Oracle | None = None,
    on_exhausted: Literal["error_message", "raise"] = "error_message",
    enable_standing_guards: bool = True,
    guard_promotion_min_occurrences: int = 3,
    guard_promotion_min_success_rate: float = 0.8,
    num_branches: int = 1,
    side_effect_free: bool = False,
    guard_demotion_min_occurrences: int = 3,
    guard_demotion_max_failure_rate: float = 0.5,
    recipe_min_success_rate: float | None = None,
    recipe_reliability_min_occurrences: int = 3,
    metrics: MetricsHook | None = None,
) -> ToolCallWrapperFn:
    """Build a `wrap_tool_call` callable to pass into your OWN
    `ToolNode(..., wrap_tool_call=...)`. This module never constructs a
    ToolNode itself here — you keep full control over
    `handle_tool_errors`, `tags`, `name`, etc. (`make_tool_node` below is
    a convenience for when you don't need that control).

    If you want `on_exhausted="raise"` or `InvariantAbortError` to
    actually reach your graph (rather than being silently reformatted),
    build your `ToolNode` with `handle_tool_errors=False` — see this
    module's docstring for why that's load-bearing, not optional.

    Note on invariants at this layer: they evaluate whatever `execute()`
    returns on success — typically a `ToolMessage`, not the tool's raw
    return value — so an `Invariant.check` here should look at e.g.
    `result.content`, not assume a bare dict/string.

    `num_branches`/`side_effect_free` (Phase 3, see core/engine.py's
    `wrap()` for the full docstring on `side_effect_free`) apply to every
    tool_call this wrapper handles equally, same caveat as
    `wrap_tools()` in `integrations/raw_tool_loop.py`. Since `execute()`
    here goes through LangGraph's own tool machinery (not a bare Python
    call), a real per-candidate call under `side_effect_free=True` means
    LangGraph's `execute()` runs once per candidate too — vouch for that
    accordingly.

    Phase 4's `isolate`/`call_timeout`/`max_memory_mb`/`max_cpu_seconds`
    are NOT available through this adapter — see this module's docstring
    for why (the tool_fn built here is a closure over LangGraph's own
    live `execute` callback, which cannot be pickled into a subprocess).

    Phase 5's `guard_demotion_*`/`recipe_min_success_rate`/
    `recipe_reliability_min_occurrences`/`metrics` (see core/engine.py's
    `wrap()` for the full docstrings) apply the same way as every other
    param here — one shared behavior across every tool_call this wrapper
    handles.
    """
    resolved_oracle = oracle or Oracle(oracle_path)

    def wrapper(request: ToolCallRequest, execute: ExecuteFn) -> Any:
        tool_name = request.tool_call["name"]
        call_args = dict(request.tool_call.get("args") or {})

        wrapped = wrap(
            _tool_fn_from_request(execute, request),
            invariants=invariants,
            oracle=resolved_oracle,
            max_recovery_attempts=max_recovery_attempts,
            tool_name=tool_name,
            reflect=reflect,
            similarity_threshold=similarity_threshold,
            workflow_id=workflow_id,
            enable_standing_guards=enable_standing_guards,
            guard_promotion_min_occurrences=guard_promotion_min_occurrences,
            guard_promotion_min_success_rate=guard_promotion_min_success_rate,
            num_branches=num_branches,
            side_effect_free=side_effect_free,
            guard_demotion_min_occurrences=guard_demotion_min_occurrences,
            guard_demotion_max_failure_rate=guard_demotion_max_failure_rate,
            recipe_min_success_rate=recipe_min_success_rate,
            recipe_reliability_min_occurrences=recipe_reliability_min_occurrences,
            metrics=metrics,
            valid_arguments=_valid_arguments_from_request(request),
        )
        try:
            return wrapped.invoke(**call_args)
        except InvariantAbortError:
            # Always propagate, regardless of on_exhausted: "abort" is an
            # explicit signal the user chose over "recover"/"warn" — unlike
            # RecoveryExhaustedError, softening it into a graceful tool
            # message the model might just shrug off would undermine why
            # abort was chosen in the first place. LangGraph (unlike the
            # raw Anthropic/OpenAI loop) has a real "propagate and halt"
            # pathway here, so use it.
            raise
        except RecoveryExhaustedError as exc:
            if on_exhausted == "raise":
                raise
            return ToolMessage(
                content=f"ResilientForge: {exc}",
                name=tool_name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

    return wrapper


def make_resilientforge_async_tool_call_wrapper(
    invariants: list[Invariant] | None = None,
    oracle_path: str | Path = ".resilientforge",
    max_recovery_attempts: int = 3,
    reflect: ReflectFn | None = None,
    similarity_threshold: float = 0.85,
    workflow_id: str | None = None,
    oracle: Oracle | None = None,
    on_exhausted: Literal["error_message", "raise"] = "error_message",
    enable_standing_guards: bool = True,
    guard_promotion_min_occurrences: int = 3,
    guard_promotion_min_success_rate: float = 0.8,
    num_branches: int = 1,
    side_effect_free: bool = False,
    guard_demotion_min_occurrences: int = 3,
    guard_demotion_max_failure_rate: float = 0.5,
    recipe_min_success_rate: float | None = None,
    recipe_reliability_min_occurrences: int = 3,
    metrics: MetricsHook | None = None,
) -> AsyncToolCallWrapperFn:
    """The `awrap_tool_call` counterpart to `make_resilientforge_tool_call_wrapper`
    — same parameters, same behavior, build a `wrap_tool_call` callable to
    pass into your OWN `ToolNode(..., awrap_tool_call=...)`.

    Exists because of a real, non-obvious LangGraph gotcha: `ToolNode`
    exposes both a sync `wrap_tool_call` and an async `awrap_tool_call` hook,
    but a `ToolNode` built with only `wrap_tool_call` set still falls back to
    forcing the SYNC path (`tool.invoke(...)`) whenever an async-only tool is
    invoked through `graph.ainvoke()` — which fails unconditionally with
    "StructuredTool does not support sync invocation" (confirmed empirically:
    a pre-existing LangGraph-level constraint, not something either adapter
    function here can paper over from the sync side — a graph containing an
    async-only tool cannot run via `graph.invoke()` at all, with or without
    ResilientForge, since `_execute_tool_sync` calls `tool.invoke()`
    unconditionally). `graph.ainvoke()` is the normal way to run an agent
    with any async tool (react-agent's own test suite only ever calls it),
    so this gap meant the sync-only adapter silently broke on every single
    call for a very common, unremarkable pattern — any I/O-bound tool
    (a web search, an API call, a DB query) defined as `async def`.

    `core/engine.py`'s `WrappedAgent.invoke()` is, by design, fully
    synchronous (see its own docstring) — this wrapper does NOT change
    that, or add any async path to `core/engine.py`/`core/recovery.py`.
    Instead it bridges: the entire synchronous `wrap()`/`WrappedAgent.invoke()`
    call (which may block on real work — reflection model calls, oracle
    sqlite access, retries) runs inside `asyncio.to_thread` so it never
    blocks the event loop the graph itself is running on; the `tool_fn`
    built for it calls the async `execute()` via a fresh `asyncio.run(...)`
    from inside that worker thread — safe specifically because a fresh
    thread has no event loop of its own already running for `asyncio.run`
    to collide with. `make_tool_node` wires both this and the sync wrapper
    onto the same `ToolNode`, so sync tools keep working exactly as before
    (via the sync path when invoked through `.invoke()`) and async tools
    now work too (via this path, whenever `.ainvoke()` is used).

    All other parameters/caveats (invariants seeing `result.content`,
    `num_branches`/`side_effect_free`, `isolate`/`call_timeout`/etc. NOT
    being available, Phase 5 params) are identical to
    `make_resilientforge_tool_call_wrapper` — see its docstring.
    """
    resolved_oracle = oracle or Oracle(oracle_path)

    async def wrapper(request: ToolCallRequest, execute: AsyncExecuteFn) -> Any:
        tool_name = request.tool_call["name"]
        call_args = dict(request.tool_call.get("args") or {})

        wrapped = wrap(
            _tool_fn_from_async_request(execute, request),
            invariants=invariants,
            oracle=resolved_oracle,
            max_recovery_attempts=max_recovery_attempts,
            tool_name=tool_name,
            reflect=reflect,
            similarity_threshold=similarity_threshold,
            workflow_id=workflow_id,
            enable_standing_guards=enable_standing_guards,
            guard_promotion_min_occurrences=guard_promotion_min_occurrences,
            guard_promotion_min_success_rate=guard_promotion_min_success_rate,
            num_branches=num_branches,
            side_effect_free=side_effect_free,
            guard_demotion_min_occurrences=guard_demotion_min_occurrences,
            guard_demotion_max_failure_rate=guard_demotion_max_failure_rate,
            recipe_min_success_rate=recipe_min_success_rate,
            recipe_reliability_min_occurrences=recipe_reliability_min_occurrences,
            metrics=metrics,
            valid_arguments=_valid_arguments_from_request(request),
        )
        try:
            return await asyncio.to_thread(wrapped.invoke, **call_args)
        except InvariantAbortError:
            raise
        except RecoveryExhaustedError as exc:
            if on_exhausted == "raise":
                raise
            return ToolMessage(
                content=f"ResilientForge: {exc}",
                name=tool_name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

    return wrapper


def make_tool_node(
    tools: Sequence[Any],
    invariants: list[Invariant] | None = None,
    oracle_path: str | Path = ".resilientforge",
    max_recovery_attempts: int = 3,
    reflect: ReflectFn | None = None,
    similarity_threshold: float = 0.85,
    workflow_id: str | None = None,
    oracle: Oracle | None = None,
    on_exhausted: Literal["error_message", "raise"] = "error_message",
    handle_tool_errors: Any = False,
    enable_standing_guards: bool = True,
    guard_promotion_min_occurrences: int = 3,
    guard_promotion_min_success_rate: float = 0.8,
    num_branches: int = 1,
    side_effect_free: bool = False,
    guard_demotion_min_occurrences: int = 3,
    guard_demotion_max_failure_rate: float = 0.5,
    recipe_min_success_rate: float | None = None,
    recipe_reliability_min_occurrences: int = 3,
    metrics: MetricsHook | None = None,
    **tool_node_kwargs: Any,
) -> ToolNode:
    """Convenience: build a fully configured ToolNode in one call, for
    the common case where you don't need to customize ToolNode beyond
    what's exposed here.

    `handle_tool_errors` defaults to False here, NOT LangGraph's own
    default of True — see the module docstring's "IMPORTANT gotcha"
    section for why: this wrapper already formats its own graceful error
    ToolMessage on RecoveryExhaustedError (governed by `on_exhausted`),
    and letting LangGraph's separate handle_tool_errors=True catch-all
    sit on top of that would silently swallow `on_exhausted="raise"` and
    InvariantAbortError before they ever reach the graph. Pass
    `handle_tool_errors=True` explicitly if you want LangGraph's own
    catch-all as an extra outer safety net for exceptions ResilientForge
    itself doesn't raise (a genuine bug in a tool, say) — just know it
    will also swallow the abort/raise guarantees documented above.

    Wires up BOTH `wrap_tool_call` (`make_resilientforge_tool_call_wrapper`)
    and `awrap_tool_call` (`make_resilientforge_async_tool_call_wrapper`) on
    the same `ToolNode` — see the latter's docstring for why both are
    needed: `graph.invoke()` (sync) always uses the sync path, `graph.ainvoke()`
    (async — the normal way to run an agent, and the only way to run one
    containing an async-only tool at all) uses the async path. `tools` may
    freely mix sync and async tools; both paths handle either kind. Sharing
    one `Oracle` (`resolved_oracle`) between them is safe specifically
    because of Phase 5's thread-local sqlite connections
    (`oracle/store.py`) — the async path's calls arrive from whatever
    worker thread `asyncio.to_thread` happens to schedule them on, same as
    genuine concurrent callers already had to be safe against.
    """
    resolved_oracle = oracle or Oracle(oracle_path)
    shared_kwargs: dict[str, Any] = dict(
        invariants=invariants,
        max_recovery_attempts=max_recovery_attempts,
        reflect=reflect,
        similarity_threshold=similarity_threshold,
        workflow_id=workflow_id,
        oracle=resolved_oracle,
        on_exhausted=on_exhausted,
        enable_standing_guards=enable_standing_guards,
        guard_promotion_min_occurrences=guard_promotion_min_occurrences,
        guard_promotion_min_success_rate=guard_promotion_min_success_rate,
        num_branches=num_branches,
        side_effect_free=side_effect_free,
        guard_demotion_min_occurrences=guard_demotion_min_occurrences,
        guard_demotion_max_failure_rate=guard_demotion_max_failure_rate,
        recipe_min_success_rate=recipe_min_success_rate,
        recipe_reliability_min_occurrences=recipe_reliability_min_occurrences,
        metrics=metrics,
    )
    wrapper = make_resilientforge_tool_call_wrapper(**shared_kwargs)
    async_wrapper = make_resilientforge_async_tool_call_wrapper(**shared_kwargs)
    return ToolNode(
        tools,
        handle_tool_errors=handle_tool_errors,
        wrap_tool_call=wrapper,
        awrap_tool_call=async_wrapper,
        **tool_node_kwargs,
    )
