# Architecture

This describes what's actually built (Phase 1). Where reality forced a
deviation or surfaced a real trade-off during the build, it's called out
explicitly below rather than smoothed over.

## Module map

```
src/resilientforge/
├── core/
│   ├── signature.py     # failure normalization/templating (the crux — see below)
│   ├── invariants.py    # Invariant: deterministic + LLM-judged checks
│   ├── recovery.py      # Fix generation (reflection) + application + transforms
│   └── engine.py        # wrap() — ties everything into the recovery loop
├── oracle/
│   ├── store.py          # SQLite: failures + recipes tables (raw CRUD)
│   ├── vector_index.py   # VectorIndex interface + chromadb implementation
│   ├── recipes.py        # Recipe domain model + RecipeManager (times_applied, success_rate, prune)
│   └── __init__.py       # Oracle — the single facade over store.py + vector_index.py
├── integrations/
│   ├── raw_tool_loop.py     # Anthropic tool_use + OpenAI function-calling shim
│   └── langgraph_adapter.py # LangGraph ToolNode via wrap_tool_call
└── cli/main.py           # list / inspect / prune / stats
```

`core/` never imports `anthropic`/`openai`/`langgraph` — the model call used
for reflection is always injected as a `ReflectFn` callable
(`Callable[[FailureContext], dict]`), the same pattern used for
`Invariant.llm_judged`'s `judge` callable. Concrete, vendor-specific
implementations (`create_anthropic_reflect`) live in `integrations/`,
which already needs that SDK wiring for the tool-calling loop itself.
This is why `core/recovery.py` and `core/engine.py` are fully testable
with zero network access and no API key.

## The recovery loop (`core/engine.py`)

`wrap(agent, invariants=[...], reflect=...)` returns a `WrappedAgent`
whose `.invoke(**kwargs)` runs, on every call:

1. Call the tool. Catch an exception, or evaluate `invariants` against
   the result if it didn't raise.
2. If nothing failed, return the result. Done.
3. Otherwise, normalize a failure **signature** from `(tool_name,
   error_type, error_message, args)` via `core/signature.py`.
4. Look up the oracle: exact match on `signature` first; if none, a
   fuzzy vector-similarity search above `similarity_threshold`. A recipe
   match here is tried **at most once per call** — "if no match, or the
   recipe's fix still fails, fall back to reflection" (not "keep
   retrying the recipe").
5. If no recipe (or it already failed once this call), fall back to
   `reflect` — a real or mocked model call that proposes a structured
   `Fix`, given the failure *and* every `Fix` already tried this call
   (so it doesn't repeat a fix that just failed — the "blind repetition"
   pattern this whole project targets).
6. Apply the `Fix`, retry, re-check invariants — **always**; a fix is
   never assumed to have worked.
7. On success: write the fix back as a recipe (`RecipeManager.
   record_success`), updating `times_applied`/`success_rate` if the
   recipe already existed.
8. Exhausted after `max_recovery_attempts`: raise `RecoveryExhaustedError`
   carrying every attempt tried (never a silent failure).

`Invariant.on_violation` controls step 1-2's behavior per invariant:
`"recover"` (default) enters the loop above; `"abort"` raises
`InvariantAbortError` immediately, no recovery attempted; `"warn"`
returns the result as-is after a Python `warnings.warn`.

## Why a `Fix` has two kinds of correction

This is the detail that makes the fast path *correct*, not just fast.
The canonical signature-normalization example is "next Friday" and
"next Tuesday" — different literal values, same failure shape. A recipe
learned from the first occurrence has to still be *right* when replayed
on the second, different one. A cached literal replacement value would
be silently wrong.

- `Fix.argument_patch` — a literal override. Safe to replay only when the
  correct value doesn't depend on the occurrence (e.g. "this field is
  missing, default it to `[]`").
- `Fix.transforms` — named, deterministic functions
  (`core/recovery.py`'s `TRANSFORM_REGISTRY`) re-applied to *this*
  occurrence's actual argument value at replay time, not a value cached
  from whenever the recipe was first learned. This is what lets one
  recipe stay correct across different literal dates, different
  malformed-JSON payloads, different wrong-typed values.

`TRANSFORM_REGISTRY` starts deliberately small (relative-date parsing,
int/float/str coercion, common-JSON-error repair) — the failure-injection
suite, not intuition, is meant to justify expanding it.

## The oracle (`oracle/`)

Two backends behind one `Oracle` facade:
- **SQLite** (`store.py`): `failures` (one row per occurrence) and
  `recipes` (one row per distinct signature that has a known fix) tables.
- **Vector index** (`vector_index.py`): embeddings of the normalized
  signature text, for the fuzzy fallback when there's no exact recipe
  match.

**Deliberate deviation from the obvious default**: chromadb's *default*
embedding function downloads an ONNX model over the network on first use,
which would violate the "unit tests are fast, no network" rule and break
offline installs. `ChromaVectorIndex` defaults to a small, deterministic,
offline hashing (bag-of-words) embedder instead — good enough for
matching near-identical *normalized* signatures (which is what actually
reaches the vector index, after `core/signature.py` has already done the
real work), but not true semantic embedding. It's swappable behind the
same `VectorIndex` interface; revisit if match quality against the
failure-injection suite calls for it.

## Signature normalization (`core/signature.py`) — the crux

The whole value proposition depends on two structurally-identical
failures producing the *same* signature regardless of literal content,
while two *actually different* failures don't collapse together. Two
independent passes:

- `normalize_error_message`: regex-based redaction of literals in
  free-text error messages (dates, uuids, emails, urls, quoted strings,
  numbers — including unit-suffixed ones like `"30s"`, which a naive
  `\b...\b`-bounded regex misses entirely).
- `normalize_args`: recursive, type-based templating of the tool-call
  arguments — dict *keys* are structural and kept as-is; values are
  templated by type (and, for strings, by pattern: uuid/date/email/url
  vs. generic).

**A real, discovered trade-off, not a bug**: a bare single-token quoted
string in an error message (e.g. `'attendees'` in `"missing required
field 'attendees'"`) is *preserved*, not redacted — because the same
regex can't tell a single-word literal value apart from a single-word
field name, and collapsing two different missing-field failures into one
signature risks replaying an unrelated fix. The cost: a single bare word
used as a genuine literal value (e.g. `"tomorrow"` as a date) doesn't
collapse with multi-word phrasings of the same failure (`"next Friday"`)
the way it arguably should. This was found by the failure-injection
suite producing a real 75%-not-100% number, not by inspection — see
`test_single_word_date_value_is_not_collapsed_like_multi_word_ones` in
`tests/unit/test_signature.py`. The heuristic's default is deliberately
the safer failure mode: a missed oracle hit costs one extra model call; a
wrongly-collapsed signature risks misapplying a fix.

## Integrations

### Raw tool loop (`integrations/raw_tool_loop.py`)

The reference implementation. `wrap_tools({name: fn, ...})` wraps a
registry of tools sharing **one** `Oracle` — a recipe learned recovering
one tool's failure is visible to every other tool on the same oracle
(useful: e.g. a JSON-escaping fix is a syntactic fact, not a
tool-specific one).

- `execute_anthropic_tool_use`: Anthropic's `tool_use.input` arrives
  already parsed — this path is the simpler of the two.
- `execute_openai_tool_call` + `make_json_arg_parser`: OpenAI hands back
  function arguments as a raw JSON *string*. Rather than special-case
  "malformed JSON args" (one of this project's first motivating failure
  patterns) as a distinct code path, JSON parsing is itself wrapped with
  `wrap()` too — it recovers through the exact same oracle/signature/recipe
  machinery as everything else, via a `repair_common_json_errors`
  transform. A recipe learned here is shared across tools for the same
  reason as above.
- `create_anthropic_reflect`: the concrete default `reflect`. Forces a
  synthetic `propose_fix` tool call whose schema is
  `Fix.model_json_schema()`, so the response validates directly.

### LangGraph (`integrations/langgraph_adapter.py`)

Hooks into `ToolNode`'s `wrap_tool_call` extension point (LangGraph
1.x) and **reuses `core/engine.py`'s `wrap()` entirely** via a small
calling-convention shim — no duplicated recovery loop.

Two things verified empirically against langgraph 1.2 while building
this (not assumed):

- `execute()`'s failure shape depends on the underlying `ToolNode`'s
  `handle_tool_errors`: `True` (LangGraph's own default) catches the
  tool's exception and returns an error `ToolMessage`; `False` raises
  directly. The adapter normalizes both into the same exception so
  `core/engine.py`'s existing failure detection handles them identically.
- **A real gotcha, found by a failing test, not by inspection**:
  `handle_tool_errors=True` catches *any* exception raised out of
  `wrap_tool_call` as a whole — not just from `execute()` — including
  `on_exhausted="raise"` and `InvariantAbortError`, which are supposed to
  reach the graph. `make_tool_node()` therefore defaults to
  `handle_tool_errors=False`, a deliberate departure from LangGraph's own
  default: deferring to its separate catch-all would silently defeat
  guarantees this adapter explicitly documents. If you build your own
  `ToolNode` around `make_resilientforge_tool_call_wrapper` instead of
  using `make_tool_node`, you must set `handle_tool_errors=False`
  yourself to get the same guarantees.
- `InvariantAbortError` always propagates regardless of `on_exhausted`
  (unlike `RecoveryExhaustedError`): LangGraph has a real "propagate and
  halt" pathway, unlike the raw Anthropic/OpenAI loop, so `abort` uses
  it rather than being softened into a tool message the model might
  shrug off.
- `RetryPolicy`'s default `retry_on` excludes `ValueError`/`TypeError` —
  it's scoped to transient failures (connection errors, 5xx) by design,
  so out of the box it doesn't compete with ResilientForge's
  data-correction recovery for most failure shapes. `on_exhausted="raise"`
  plus an explicit `retry_on` lets `RetryPolicy` act as a last-resort
  safety net after ResilientForge's own recovery is exhausted.

## What invariants actually see

An `Invariant.check` receives whatever the wrapped call's *result* is —
and that shape differs by integration, which matters when writing one:

| Context | What `check(result)` receives |
|---|---|
| Raw tool loop (`wrap()` directly, or `wrap_tools`) | the tool function's raw return value |
| `execute_anthropic_tool_use` / `execute_openai_tool_call` | same — invariants attach to the wrapped tool, evaluated before formatting into a `tool_result` |
| LangGraph adapter | whatever `execute()` returns — typically a `ToolMessage`, so check e.g. `result.content`, not a bare value |

See [`writing_invariants.md`](writing_invariants.md) for concrete examples.

## Testing strategy in practice

Three tiers:

```
pytest tests/unit tests/integration     # fast, no network — default CI gate
pytest tests/failure_injection           # the recovery-rate proof
pytest -m live                           # opt-in, real API calls (not run by default)
```

`tests/failure_injection/harness.py` and `tests/integration/test_engine.py`
were added because the engine and the five failure scenarios needed their
own shared contract/coverage before either adapter existed.
