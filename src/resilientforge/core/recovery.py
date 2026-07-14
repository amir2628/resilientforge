"""Fix generation (LLM reflection) and fix application (PROJECT_SPEC.md §4.4
steps 5-6). Oracle lookup and the overall recovery loop live in
`core/engine.py` (step 6 of §9); this module owns two narrower things:

1. `generate_fix`: given a `FailureContext`, ask an injected `reflect`
   callable to propose a `Fix`. Like `Invariant.llm_judged`'s `judge`
   param, `reflect` is caller-supplied rather than hardcoded to a vendor
   (§5.1) — this module has no `anthropic`/`openai` import. A concrete
   default reflector belongs in `integrations/raw_tool_loop.py`, which
   already needs Anthropic wiring for the tool-calling loop itself; that
   keeps this module vendor-neutral and lets tests mock the model call
   entirely (§7.1/§7.2), never touching a real API.
2. `apply_fix`: turn a `Fix` into corrected tool-call arguments.

Why `Fix` has two kinds of correction, not one:
A recipe learned from one occurrence of a failure shape has to stay
correct when replayed (fast path, no LLM call) on a *different*
occurrence of the same shape — e.g. a signature normalizes "next Friday"
and "next Tuesday" to the same shape (§4.3's own example), but the correct
fixed value obviously differs between them. A fix that was just a cached
literal replacement value would be silently wrong on replay.
- `argument_patch`: literal value overrides. Safe to replay only when the
  correct value doesn't depend on the specific occurrence (e.g. "this
  field is missing — default it to []").
- `transforms`: named, deterministic functions (from `TRANSFORM_REGISTRY`)
  re-applied to *this* occurrence's actual argument value at replay time,
  not to a value cached from whenever the recipe was first learned. This
  is what lets a natural-language-date recipe stay correct across
  different literal dates.

`TRANSFORM_REGISTRY` intentionally starts small (§10: this whole area
needs iteration against the failure-injection suite, not front-loaded
guessing) — expand it as failure-injection scenarios (§7.3) demand.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Callable

from pydantic import BaseModel, Field

# -- deterministic transforms -------------------------------------------------


class TransformError(Exception):
    """Raised when a named transform can't be applied to a given value."""


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_IN_N_DAYS_RE = re.compile(r"^in (\d+) days?$")
_NEXT_WEEKDAY_RE = re.compile(r"^next (\w+)$")


def parse_relative_date_to_iso(value: Any, *, today: date | None = None) -> str:
    """A minimal, dependency-free relative-date parser: today/tomorrow/
    yesterday, "in N days", "next <weekday>". Deliberately narrow — this is
    a starting point to prove the transform mechanism, not a general NL
    date parser; widen it against real failure-injection results (§10),
    not speculatively.
    """
    if not isinstance(value, str):
        raise TransformError(f"expected a string date, got {type(value).__name__}")
    text = value.strip().lower()
    today = today or date.today()

    if text == "today":
        return today.isoformat()
    if text == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if text == "yesterday":
        return (today - timedelta(days=1)).isoformat()

    match = _IN_N_DAYS_RE.match(text)
    if match:
        return (today + timedelta(days=int(match.group(1)))).isoformat()

    match = _NEXT_WEEKDAY_RE.match(text)
    if match and match.group(1) in _WEEKDAYS:
        target = _WEEKDAYS[match.group(1)]
        days_ahead = (target - today.weekday() + 7) % 7
        days_ahead = days_ahead or 7  # "next Friday" on a Friday means +7, not today
        return (today + timedelta(days=days_ahead)).isoformat()

    raise TransformError(f"could not parse relative date: {value!r}")


def coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"could not coerce {value!r} to int") from exc


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"could not coerce {value!r} to float") from exc


def coerce_str(value: Any) -> str:
    return str(value)


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def repair_common_json_errors(value: Any) -> str:
    """Best-effort repair of common JSON-escaping mistakes (trailing
    commas, single quotes instead of double) — the malformed-JSON-args
    failure pattern from §1, first concretely exercised by
    integrations/raw_tool_loop.py's OpenAI shim, where function-call
    arguments arrive as a raw string that has to be re-parsed. Returns a
    (hopefully) valid JSON *string* for re-parsing, not the parsed value —
    this transform corrects the input to `json.loads`, it doesn't call it.
    Same philosophy as `parse_relative_date_to_iso`: a narrow starting
    point, not a general JSON repair library; widen against real
    failure-injection results (§10), not speculatively.
    """
    if not isinstance(value, str):
        raise TransformError(f"expected a JSON string, got {type(value).__name__}")
    repaired = _TRAILING_COMMA_RE.sub(r"\1", value)
    return repaired.replace("'", '"')


TRANSFORM_REGISTRY: dict[str, Callable[[Any], Any]] = {
    "parse_relative_date_to_iso": parse_relative_date_to_iso,
    "coerce_int": coerce_int,
    "coerce_float": coerce_float,
    "coerce_str": coerce_str,
    "repair_common_json_errors": repair_common_json_errors,
}


# -- Fix / FailureContext -----------------------------------------------------


class ArgTransform(BaseModel):
    argument: str
    transform: str


class Fix(BaseModel):
    strategy: str
    root_cause: str | None = None
    argument_patch: dict[str, Any] = Field(default_factory=dict)
    transforms: list[ArgTransform] = Field(default_factory=list)


class FailureContext(BaseModel):
    tool_name: str
    args: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None
    signature: str | None = None
    attempt_number: int = 1
    # Fixes already tried this run that didn't resolve the failure, so a
    # reflection call doesn't propose the same broken approach again — the
    # "blind repetition" failure pattern this whole project targets (§1).
    previous_attempts: list[Fix] = Field(default_factory=list)
    available_transforms: list[str] = Field(default_factory=lambda: sorted(TRANSFORM_REGISTRY))


ReflectFn = Callable[[FailureContext], dict[str, Any]]


def generate_fix(context: FailureContext, reflect: ReflectFn) -> Fix:
    """Ask `reflect` (a real model call in production, a mock in tests) to
    propose a fix for `context`, and validate its response into a `Fix`."""
    raw = reflect(context)
    return Fix.model_validate(raw)


def apply_fix(args: dict[str, Any], fix: Fix) -> dict[str, Any]:
    """Apply a Fix to a tool call's arguments, producing the args to retry
    with. Patch entries are applied first (adds/overwrites literal values),
    then transforms recompute specific argument values from what's actually
    in `args` right now — see the module docstring for why that ordering
    and split matters for fast-path replay correctness.
    """
    new_args = dict(args)
    new_args.update(fix.argument_patch)
    for arg_transform in fix.transforms:
        if arg_transform.argument not in new_args:
            continue
        try:
            transform_fn = TRANSFORM_REGISTRY[arg_transform.transform]
        except KeyError as exc:
            raise TransformError(f"unknown transform: {arg_transform.transform!r}") from exc
        new_args[arg_transform.argument] = transform_fn(new_args[arg_transform.argument])
    return new_args
