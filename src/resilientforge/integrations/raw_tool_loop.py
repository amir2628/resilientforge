"""Raw tool-calling loop adapter: the reference
Phase 1 integration, working directly against the Anthropic Messages API
tool-use format, with a thin shim for OpenAI's function-calling format.

Two things this module adds on top of core/engine.py's `wrap()`:

1. Translating a tool_use block / tool_call object from each SDK's
   response shape into a `WrappedAgent.invoke()` call, and the result back
   into that SDK's tool-result message shape.
2. `create_anthropic_reflect`: the concrete Anthropic-backed `reflect`
   default that core/recovery.py and core/engine.py deliberately don't
   provide (they stay vendor-neutral, per their own module docstrings) —
   it belongs here, since this module already needs `anthropic` SDK
   wiring for the tool loop itself.

OpenAI's function-calling format hands back arguments as a raw JSON
*string* (`tool_call.function.arguments`), not a pre-parsed dict the way
Anthropic's `tool_use.input` arrives — this is exactly the "malformed JSON
args" failure pattern this project targets. Rather than special-case it,
JSON parsing is itself wrapped with `resilientforge.wrap()` (`make_json_arg_parser`),
so a broken-JSON failure recovers through the same oracle/signature/
recipe machinery as everything else, via the `repair_common_json_errors`
transform added to core/recovery.py's TRANSFORM_REGISTRY alongside this
module, since this is where that failure mode is first concretely
exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resilientforge.core.engine import InvariantAbortError, RecoveryExhaustedError, WrappedAgent, wrap
from resilientforge.core.invariants import Invariant
from resilientforge.core.recovery import FailureContext, Fix, ReflectFn
from resilientforge.oracle import Oracle
from resilientforge.telemetry.metrics import MetricsHook

# -- wiring multiple tools to a shared oracle ---------------------------------


def wrap_tools(
    tools: dict[str, Any],
    invariants: dict[str, list[Invariant]] | None = None,
    oracle_path: str | Path = ".resilientforge",
    max_recovery_attempts: int = 3,
    reflect: ReflectFn | None = None,
    similarity_threshold: float = 0.85,
    workflow_id: str | None = None,
    oracle: Oracle | None = None,
    enable_standing_guards: bool = True,
    guard_promotion_min_occurrences: int = 3,
    guard_promotion_min_success_rate: float = 0.8,
    num_branches: int = 1,
    side_effect_free: bool = False,
    isolate: bool = False,
    call_timeout: float | None = None,
    max_memory_mb: int | None = None,
    max_cpu_seconds: float | None = None,
    guard_demotion_min_occurrences: int = 3,
    guard_demotion_max_failure_rate: float = 0.5,
    recipe_min_success_rate: float | None = None,
    recipe_reliability_min_occurrences: int = 3,
    metrics: MetricsHook | None = None,
) -> dict[str, WrappedAgent]:
    """Wrap every tool in `tools` ({name: callable}), sharing ONE Oracle
    across all of them — recipes learned recovering one tool's failures
    are stored in the same place a sibling tool's failures are, which
    matters since a shared failure shape (e.g. a natural-language date
    argument) can show up across unrelated tools.

    `num_branches`/`side_effect_free` (Phase 3, see core/engine.py's
    `wrap()` for the full docstring) apply the same way to every tool in
    `tools` — there's no per-tool override here, since `side_effect_free`
    is a real safety vouch and per-tool differences would need a
    per-tool call to `wrap()` directly instead.

    `isolate`/`call_timeout`/`max_memory_mb`/`max_cpu_seconds` (Phase 4,
    see core/engine.py's `wrap()` for the full docstring) apply the same
    way — and the same picklability requirement `isolate=True` needs
    applies per tool, checked eagerly when each one is wrapped below.

    `guard_demotion_*`/`recipe_min_success_rate`/
    `recipe_reliability_min_occurrences`/`metrics` (Phase 5, see
    core/engine.py's `wrap()` for the full docstrings) apply the same
    way to every tool — `metrics` in particular means one `MetricsHook`
    sees every tool's events, distinguishable by `MetricEvent.tool_name`.
    """
    shared_oracle = oracle or Oracle(oracle_path)
    invariants = invariants or {}
    return {
        name: wrap(
            fn,
            invariants=invariants.get(name, []),
            oracle=shared_oracle,
            max_recovery_attempts=max_recovery_attempts,
            tool_name=name,
            reflect=reflect,
            similarity_threshold=similarity_threshold,
            workflow_id=workflow_id,
            enable_standing_guards=enable_standing_guards,
            guard_promotion_min_occurrences=guard_promotion_min_occurrences,
            guard_promotion_min_success_rate=guard_promotion_min_success_rate,
            num_branches=num_branches,
            side_effect_free=side_effect_free,
            isolate=isolate,
            call_timeout=call_timeout,
            max_memory_mb=max_memory_mb,
            max_cpu_seconds=max_cpu_seconds,
            guard_demotion_min_occurrences=guard_demotion_min_occurrences,
            guard_demotion_max_failure_rate=guard_demotion_max_failure_rate,
            recipe_min_success_rate=recipe_min_success_rate,
            recipe_reliability_min_occurrences=recipe_reliability_min_occurrences,
            metrics=metrics,
        )
        for name, fn in tools.items()
    }


def _stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


# -- Anthropic Messages API tool-use format -----------------------------------


def execute_anthropic_tool_use(
    wrapped_tools: dict[str, WrappedAgent], tool_use: Any
) -> dict[str, Any]:
    """Execute one `tool_use` content block from an Anthropic Message and
    return the corresponding `tool_result` content block. `tool_use` is
    duck-typed: anything with `.id`, `.name`, `.input` (a dict, matching
    `anthropic.types.ToolUseBlock`) works, real or a test double."""
    wrapped = wrapped_tools.get(tool_use.name)
    if wrapped is None:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": f"unknown tool: {tool_use.name!r}",
            "is_error": True,
        }
    try:
        result = wrapped.invoke(**tool_use.input)
    except (RecoveryExhaustedError, InvariantAbortError) as exc:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": str(exc),
            "is_error": True,
        }
    return {"type": "tool_result", "tool_use_id": tool_use.id, "content": _stringify_result(result)}


# -- OpenAI function-calling format (thin shim) -------------------------------


def _parse_json_args(raw_args: str) -> dict[str, Any]:
    return json.loads(raw_args)


def make_json_arg_parser(
    oracle: Oracle,
    reflect: ReflectFn | None = None,
    max_recovery_attempts: int = 2,
    enable_standing_guards: bool = True,
    guard_promotion_min_occurrences: int = 3,
    guard_promotion_min_success_rate: float = 0.8,
) -> WrappedAgent:
    """A wrapped JSON-parsing step, shared across all tools on the shared
    `oracle` — a broken-JSON recipe generalizes across tools (it's a
    syntactic problem, not a tool-specific one), so this is deliberately
    NOT one-per-tool the way `wrap_tools` is. Guards promote here too, with
    zero special-casing: `(tool_name="parse_tool_call_json",
    argument="raw_args", transform="repair_common_json_errors")` is just
    another guard once the same JSON-escaping mistake recurs enough times.
    """
    return wrap(
        _parse_json_args,
        oracle=oracle,
        tool_name="parse_tool_call_json",
        reflect=reflect,
        max_recovery_attempts=max_recovery_attempts,
        enable_standing_guards=enable_standing_guards,
        guard_promotion_min_occurrences=guard_promotion_min_occurrences,
        guard_promotion_min_success_rate=guard_promotion_min_success_rate,
    )


def _openai_tool_message(tool_call_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if is_error:
        message["is_error"] = True
    return message


def execute_openai_tool_call(
    wrapped_tools: dict[str, WrappedAgent],
    tool_call: Any,
    json_parser: WrappedAgent,
) -> dict[str, Any]:
    """Execute one `tool_call` from an OpenAI chat completion and return
    the corresponding tool message. `tool_call` is duck-typed: anything
    with `.id`, `.function.name`, `.function.arguments` (a raw JSON
    string) works. `json_parser` should come from `make_json_arg_parser`,
    built once per shared oracle and passed in explicitly rather than
    constructed on every call.
    """
    name = tool_call.function.name
    wrapped = wrapped_tools.get(name)
    if wrapped is None:
        return _openai_tool_message(tool_call.id, f"unknown tool: {name!r}", is_error=True)

    try:
        parsed_args = json_parser.invoke(raw_args=tool_call.function.arguments)
    except (RecoveryExhaustedError, InvariantAbortError) as exc:
        return _openai_tool_message(
            tool_call.id, f"could not parse tool arguments: {exc}", is_error=True
        )

    try:
        result = wrapped.invoke(**parsed_args)
    except (RecoveryExhaustedError, InvariantAbortError) as exc:
        return _openai_tool_message(tool_call.id, str(exc), is_error=True)

    return _openai_tool_message(tool_call.id, _stringify_result(result))


# -- default reflect() implementations -----------------------------------------

_FIX_TOOL_NAME = "propose_fix"


def _fix_tool_schema() -> dict[str, Any]:
    return {
        "name": _FIX_TOOL_NAME,
        "description": (
            "Propose a structured fix for a tool-call failure. Use "
            "argument_patch for a literal correction (safe to reuse only "
            "when the right value doesn't depend on this specific call, "
            "e.g. defaulting a missing field). Use transforms (naming one "
            "of the available transforms) when the right value has to be "
            "recomputed from this call's own arguments, e.g. reparsing a "
            "date — that's what keeps a learned fix correct the next time "
            "the same failure shape shows up with a different literal "
            "value."
        ),
        "input_schema": Fix.model_json_schema(),
    }


def _flat_fix_schema() -> dict[str, Any]:
    """A hand-authored, flattened equivalent of `Fix.model_json_schema()`
    — no `$defs`/`$ref`. Claude follows the raw pydantic schema fine (see
    `_fix_tool_schema` above), but empirically (tested against a local
    Qwen2.5 model via Ollama while building `create_local_reflect`)
    smaller/local models' tool-calling gets confused by `$defs`/`$ref`
    indirection and starts inventing its own wrapper structure instead of
    the requested one. Kept separate from `_fix_tool_schema` rather than
    replacing it, since Claude doesn't need this workaround."""
    return {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "description": "short name for the fix strategy, e.g. reformat_argument",
            },
            "root_cause": {
                "type": "string",
                "description": "one-sentence explanation of why the call failed",
            },
            "argument_patch": {
                "type": "object",
                "description": "literal key-value overrides for arguments, or an empty object",
            },
            "transforms": {
                "type": "array",
                "description": "named transforms to apply, or an empty array",
                "items": {
                    "type": "object",
                    "properties": {
                        "argument": {"type": "string"},
                        "transform": {"type": "string"},
                    },
                    "required": ["argument", "transform"],
                },
            },
        },
        "required": ["strategy"],
    }


def _build_reflect_prompt(context: FailureContext) -> str:
    lines = [
        f"A call to tool {context.tool_name!r} failed.",
        f"Arguments: {json.dumps(context.args, default=str)}",
    ]
    if context.error_type:
        lines.append(f"Error type: {context.error_type}")
    if context.error_message:
        lines.append(f"Error message: {context.error_message}")
    if context.available_transforms:
        lines.append("Available transforms: " + ", ".join(context.available_transforms))
    if context.previous_attempts:
        lines.append("Previously tried fixes that did NOT resolve this failure:")
        lines.extend(f"- {prev.model_dump_json()}" for prev in context.previous_attempts)
    lines.append(f"Call the {_FIX_TOOL_NAME} tool with your proposed fix.")
    return "\n".join(lines)


def create_anthropic_reflect(client: Any = None, model: str = "claude-sonnet-5") -> ReflectFn:
    """Build a `reflect` callable backed by a
    real Anthropic Messages API call, forced to invoke a synthetic
    `propose_fix` tool matching Fix's schema.

    `client` defaults to a real `anthropic.Anthropic()` (reads
    ANTHROPIC_API_KEY from the environment) — inject a fake/mock client in
    tests so no network call happens outside the opt-in `live` test tier.
    The import is local to this function so `anthropic` is only
    required if this factory is actually used.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    def _reflect(context: FailureContext) -> dict[str, Any]:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[_fix_tool_schema()],
            tool_choice={"type": "tool", "name": _FIX_TOOL_NAME},
            messages=[{"role": "user", "content": _build_reflect_prompt(context)}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input
        raise RuntimeError(
            f"reflection call did not return a {_FIX_TOOL_NAME!r} tool_use block"
        )

    return _reflect


def create_local_reflect(
    client: Any = None,
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    model: str = "qwen2.5:7b",
) -> ReflectFn:
    """Build a `reflect` callable backed by any locally-hosted,
    OpenAI-compatible chat completions endpoint — developed and verified
    against Ollama (`base_url`/`api_key` default to Ollama's own
    OpenAI-compatibility endpoint, no Ollama-specific code involved), but
    works with anything speaking the same protocol (LM Studio, vLLM's
    `--api-key`/OpenAI-compatible server, etc.).

    Exists for the same reason `create_anthropic_reflect` exists — a
    concrete, ready-to-use `reflect` for people who don't want (or, for
    testing/dogfooding without incurring API cost, can't yet) call a
    paid hosted API. `reflect` was always meant to be vendor-neutral and
    freely swappable (see `core/recovery.py`'s module docstring); this is
    proof by demonstration, not a special case.

    Uses `_flat_fix_schema()`, not `Fix.model_json_schema()` — see that
    function's docstring for why: local models' tool-calling was found,
    empirically, to be far less reliable than Claude's at following a
    schema with `$defs`/`$ref` indirection. `temperature=0` for the same
    reason — this is a structured-extraction task, not creative writing.

    `client` defaults to a real `openai.OpenAI(base_url=..., api_key=...)`
    — inject a fake/mock client in tests so no network call happens
    outside the opt-in `live` test tier. The import is local to this
    function so `openai` is only required if this factory is actually
    used (it already is a base dependency of this package, unlike
    `anthropic` needing no extra install either — this needs no new
    dependency at all).
    """
    if client is None:
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def _reflect(context: FailureContext) -> dict[str, Any]:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": _FIX_TOOL_NAME,
                        "description": _fix_tool_schema()["description"],
                        "parameters": _flat_fix_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": _FIX_TOOL_NAME}},
            messages=[{"role": "user", "content": _build_reflect_prompt(context)}],
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError(
                f"reflection call did not return a {_FIX_TOOL_NAME!r} tool call"
            )
        return json.loads(tool_calls[0].function.arguments)

    return _reflect
