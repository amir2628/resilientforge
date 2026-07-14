# Writing Invariants

An `Invariant` is a check ResilientForge runs against a tool call's result.
If it fails, `wrap()` treats that as a failure and runs the same recovery
loop it runs for a raised exception (see [`architecture.md`](architecture.md)).

```python
from resilientforge import Invariant

class Invariant(BaseModel):
    name: str
    check: Callable[[Any], bool]
    on_violation: Literal["recover", "abort", "warn"] = "recover"
    severity: Literal["low", "medium", "high"] = "medium"
```

## The three ways to build one

### 1. A plain predicate (the base constructor)

```python
from resilientforge import Invariant

not_negative = Invariant(
    name="quantity_not_negative",
    check=lambda result: result["quantity"] >= 0,
)
```

### 2. Pydantic schema validation

```python
from pydantic import BaseModel
from resilientforge import Invariant

class EventResult(BaseModel):
    title: str
    attendees: list[str]

valid_event = Invariant.from_pydantic_model("valid_event", EventResult)
```

`valid_event.check(result)` returns `False` on any `ValidationError` —
missing fields, wrong types, anything the model doesn't accept.

### 3. LLM-judged (a natural-language rule, evaluated by a model call)

```python
from resilientforge import Invariant

def judge(rule: str, result) -> bool:
    # your model call here — see core/recovery.py's ReflectFn for the
    # same "caller supplies the model call" pattern
    ...

no_destructive_action = Invariant.llm_judged(
    name="no_destructive_fs_action",
    rule="no destructive filesystem action outside the working directory",
    judge=judge,
)
```

`judge` is injected, not hardcoded to a vendor — same reasoning as
`core/recovery.py`'s `ReflectFn`: it keeps this fully mockable in tests
(no real model call needed to test invariant *logic*), and doesn't lock
you into one provider.

## Built-ins

```python
from resilientforge.core.invariants import not_none, is_instance_of

not_none()                                  # result is not None
is_instance_of(dict)                        # isinstance(result, dict)
is_instance_of((int, float), name="numeric")
```

## `on_violation`: what happens when a check fails

- **`"recover"`** (default) — enters the recovery loop: oracle lookup,
  then `reflect` if needed, re-checked after every attempt.
- **`"abort"`** — raises `InvariantAbortError` immediately. No recovery
  is attempted at all. Use this for invariants where a wrong *replayed*
  fix (the fast path, which doesn't get a fresh model judgment) would be
  actively dangerous: *"a recipe applied without a fresh LLM call is, by
  definition, replaying a past action without fresh judgment."* A
  destructive-action check is the canonical example.
- **`"warn"`** — the result is returned as-is, after a Python
  `warnings.warn`. No recovery, no exception. Use this for invariants
  you want visibility into but don't want to block on.

**One integration-specific wrinkle worth knowing**: the LangGraph adapter
always lets `InvariantAbortError` propagate to the graph regardless of
`on_exhausted`, since LangGraph has a real "halt" pathway to use. The raw
Anthropic/OpenAI tool loop has no equivalent — there, an abort is
formatted into an `is_error=True` tool result, since a single tool-call
turn can't meaningfully "halt" a conversation from inside itself. If
`on_violation="abort"` is meant to be a hard stop for your application,
check for that condition in the calling code that drives the loop, not
just in the adapter's return value.

## What `check(result)` actually receives

This differs by integration — see the table in
[`architecture.md`](architecture.md#what-invariants-actually-see). In
short: raw `wrap()`/`wrap_tools()` and the Anthropic/OpenAI adapters pass
the tool's own raw return value; the LangGraph adapter passes whatever
`execute()` returns, typically a `ToolMessage` — so `check` there should
look at `result.content`, not assume a bare dict.

## Multiple invariants

`wrap(tool, invariants=[a, b, c])` evaluates all of them after every
call. If any is violated:
- any `"abort"` invariant wins — raises immediately, regardless of the
  others.
- else, if any is `"recover"` — enters the recovery loop.
- else (all violated invariants are `"warn"`) — warns and returns.

## A complete example

From `tests/failure_injection/scenarios/missing_required_field.py` — the
one failure-injection scenario detected via an invariant rather than an
exception:

```python
from pydantic import BaseModel
from resilientforge import Invariant, wrap

class EventResult(BaseModel):
    title: str
    attendees: list[str]

def create_event(title: str, attendees: list[str] | None = None) -> dict:
    if attendees is None:
        return {"title": title}  # no exception — just an incomplete result
    return {"title": title, "attendees": attendees}

def reflect(context) -> dict:
    return {"strategy": "add_missing_field", "argument_patch": {"attendees": []}}

wrapped = wrap(
    create_event,
    invariants=[Invariant.from_pydantic_model("valid_event", EventResult)],
    reflect=reflect,
)

wrapped.invoke(title="Standup")  # recovers: {"title": "Standup", "attendees": []}
```
