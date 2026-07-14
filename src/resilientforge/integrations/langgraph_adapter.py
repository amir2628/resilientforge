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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

from resilientforge import InvariantAbortError, RecoveryExhaustedError, wrap
from resilientforge.core.invariants import Invariant
from resilientforge.core.recovery import ReflectFn
from resilientforge.oracle import Oracle

ExecuteFn = Callable[[ToolCallRequest], Any]
ToolCallWrapperFn = Callable[[ToolCallRequest, ExecuteFn], Any]


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
    """
    resolved_oracle = oracle or Oracle(oracle_path)
    wrapper = make_resilientforge_tool_call_wrapper(
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
    )
    return ToolNode(
        tools,
        handle_tool_errors=handle_tool_errors,
        wrap_tool_call=wrapper,
        **tool_node_kwargs,
    )
