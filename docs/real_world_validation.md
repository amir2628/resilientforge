# Real-world validation: does the oracle generalize beyond our own failure-injection scenarios?

> **Round 2 addendum (2026-07-15):** this round proved cross-session
> *recurrence recognition* on the one failure shape that occurred (search
> timeouts) but, by its own admission, couldn't test whether signature
> normalization correctly *discriminates* different real failures from
> each other — only one shape ever showed up. A second round, adding a
> second real tool specifically to widen the failure surface, found real
> answers to that question: a genuine false-merge (a 403 and a 402
> collapsed into one signature) and a genuine missed-match (two identical
> "can't decode this PDF" failures split into two signatures over a hex
> byte value). See [`real_world_validation_round2.md`](./real_world_validation_round2.md)
> — kept as a fully separate document/working tree, not merged into this one.

> **Round 3 addendum (2026-07-15):** round 2's two `core/signature.py`
> findings were fixed and confirmed against the exact same real cases that
> exposed them (see [`real_world_validation_round3.md`](./real_world_validation_round3.md)).
> Round 3 also surfaced a related, still-unfixed gap: a `Fix.transforms`
> entry can target an argument that isn't real, the same underlying problem
> as round 2's inert-`argument_patch` finding, just via a different field.

**Date run:** 2026-07-15, 3 sessions back-to-back (same day — see "What this
doesn't tell us").
**External agent:** [`langchain-ai/react-agent`](https://github.com/langchain-ai/react-agent),
commit `7d1f9832f56d6d29ad9ae248caf0b263c5460145` (2026-06-26).
**Setup, exact deviations, reproduction steps:** [`../validation/README.md`](../validation/README.md).
**Scope:** measurement only. Nothing in `src/resilientforge` was changed as
a result of this exercise's *original* question (no normalization gap was
found — see below) — the one code change that did happen (an async-tool
bug fix) was a separate, more fundamental discovery, made and fixed with
explicit sign-off before proceeding, not a silent patch.

## Headline finding: a real bug, not the one we went looking for

Wrapping react-agent's `ToolNode` with `make_tool_node(...)` (the ordinary,
real integration path) failed **unconditionally, on the very first call**,
with `StructuredTool does not support sync invocation`. react-agent's
`search` tool is `async def` — an extremely common pattern for any
I/O-bound tool — and `langgraph_adapter.py`'s wrapper had no support for
async tools at all. Its own test suite exercised only synchronous tools
across every one of its ~10 tests, so this had zero coverage and shipped
broken through Phases 1-5.

Reproduced in isolation (nothing react-agent- or DuckDuckGo-specific):

```python
@tool
async def async_echo(x: str) -> str:
    """Echo x back."""
    return f"got {x}"

node = make_tool_node([async_echo], oracle_path="/tmp/whatever")
# -> "ResilientForge: exhausted 0 recovery attempt(s) for tool 'async_echo':
#     StructuredTool does not support sync invocation."
```

Root cause: LangGraph's `ToolNode` exposes both a sync `wrap_tool_call` and
an async `awrap_tool_call` hook. A `ToolNode` built with only the sync one
still forces the sync execution path (`tool.invoke(...)`) whenever an
async-only tool is invoked through `graph.ainvoke()` — the *only* way to
run a graph containing an async-only tool at all (`graph.invoke()` can't
either — a pre-existing LangGraph-level constraint, confirmed by reading
`_execute_tool_sync`, not something either adapter function could paper
over). `langgraph_adapter.py` never registered an `awrap_tool_call`.

**Fixed**, with the user's explicit sign-off obtained before touching
`src/resilientforge` (this exercise's instructions were "measurement task,
not a fix-it task" — this exception was asked for and granted mid-session,
not assumed): added `make_resilientforge_async_tool_call_wrapper` /
`awrap_tool_call`, bridging LangGraph's async `execute()` into
`core/engine.py`'s deliberately-synchronous `WrappedAgent.invoke()` via
`asyncio.to_thread` (no async path added to `core/engine.py`/
`core/recovery.py` themselves). 3 new regression tests
(`tests/integration/test_langgraph_adapter.py`) cover: an async tool
succeeding via `.ainvoke()`, an async tool recovering via reflect + the
fast path across two calls, and a sync tool still working via `.ainvoke()`
(no regression). Full detail in `langgraph_adapter.py`'s module docstring.
Full suite (253 tests), ruff, and bandit all still pass after the fix.

## The original question: does signature normalization generalize?

With the bug fixed, 3 sessions × 35 real prompts (105 total) ran against
the live, wrapped agent — real local model (`qwen2.5:7b` via Ollama)
driving real DuckDuckGo searches, zero pre-seeding, zero prompts designed
to hit any of the 5 existing synthetic failure-injection scenarios (none
of which apply to a single free-text `query: str` tool anyway — they're
all about a calendar-event-style tool's dates/JSON/missing fields).

**103 of 105 real tool calls succeeded outright, first try.** Exactly
**6** real failures occurred, all the identical underlying shape: `ddgs`'s
own backend occasionally timed out contacting one of the search engines it
tries (`TimeoutException`, e.g. `"error sending request for url
(https://www.mojeek.com/search?q=...) > operation timed out"`) — a
genuinely transient, unscripted, real network failure, not anything
injected.

| # | Session | Query (literal, real) | 1st occurrence? | Path | Resolved? |
|---|---------|------------------------|-----------------|------|-----------|
| 1 | 1 | "most recent stable version of Python release date" | yes | reflection (3 attempts — see below) | yes |
| 2 | 1 | "largest desert in the world by area" | no | recipe → 1 replay failed → reflection | yes |
| 3 | 2 | "difference between Python's asyncio and threading" | no | recipe (1 replay, succeeded) | yes |
| 4 | 2 | "who painted The Starry Night" | no | recipe → 1 replay failed → reflection | yes |
| 5 | 3 | "latest updates in the space industry" | no | recipe (1 replay, succeeded) | yes |
| 6 | 3 | "Brazil vs Germany World Cup wins" | no | recipe (1 replay, succeeded) | yes |

All 6 different literal search queries. All 6 collapsed to the **exact
same** oracle signature:
`tool:search|error_type:TimeoutException|error:error sending request for
url (<URL> > operation timed out|args:{query:<STR>}`.

### The honest metric (Step 4)

Of the **5 recurrences** of this one real, unscripted failure shape (after
its first-ever occurrence, #1), the oracle correctly recognized **all 5**
as matching the existing recipe and attempted it first — **5/5 = 100%**,
across different sessions (i.e. surviving a fresh process and a
freshly-opened oracle.db each time — cross-session persistence genuinely
exercised, not just in-process reuse) and different literal query text
each time. Zero were misses caused by a normalization bug; the one "miss"
(#1) was legitimate — nothing to match yet, since it was the first
occurrence.

**Manual check of every miss/near-miss, as instructed:** #1 was a
legitimate cold-start miss, not a normalization gap — confirmed by reading
`core/signature.py`'s actual output for it. No case was found where a
human would call two failures "the same shape" but the code computed
different signatures. **No GitHub issue opened for `core/signature.py`** —
there was nothing to report on that front from this run.

### Was the fix actually correct, or did it just happen to succeed? (also Step 4)

Checked by hand, and the honest answer is **the latter, and it matters**:
every stored `fix_applied` for this signature has `argument_patch: {}` and
`transforms: []` — the model's `strategy`/`root_cause` text differs
cosmetically between attempts (`"increase_timeout"` vs.
`"retry_with_backoff"`), but the actual fix content is **empty** every
time. There is no argument-level correction possible for "the network
timed out" — the only thing that can help is retrying, and the recipe
that gets replayed is, structurally, a no-op retry. Its `success_rate`
(0.75 — 6 succeeded of 8 applications) measures how often a transient
timeout self-resolves on retry, not whether ResilientForge computed a
correct fix. This is a real, useful thing to know honestly rather than
report as "6/6 recovered, 100% recovery rate" without the caveat: for this
one failure class, "recovery" and "blind retry" are indistinguishable.

One additional, real wrinkle worth recording precisely: occurrence #1's
first reflection attempt actually failed with a *different* error —
`TransformError: unknown transform` — because the local model's very
first proposed fix named a transform that isn't in
`TRANSFORM_REGISTRY` (apparently confusing its own free-text `strategy`
label, `"increase_timeout"`, for an actual registered transform name). The
system correctly treated this as just another failed attempt, fed it back
via `previous_attempts` in the next reflection prompt, and the model
self-corrected to an empty `transforms: []` on its next try. A real,
minor, self-healing quirk of local-model reflection quality — not a
ResilientForge bug.

### Guards: correctly never promoted

Despite this signature recurring 6 times (above the default
`guard_promotion_min_occurrences=3`), **zero standing guards were
promoted** (`oracle.list_guards()` returns empty after all 3 sessions).
This is correct, not a gap: guards apply a proactive *argument* transform
before a call is attempted, and this failure class has no argument-level
fix to promote (see above — the fix is always empty). The system
correctly has nothing to promote into a guard here. Worth stating
explicitly since it means this validation exercised the reactive
(recipe-replay) half of the system thoroughly, but not the standing-guard
half at all.

## What this doesn't tell us

- **Sample size for the actual question asked is tiny.** Only ONE distinct
  real failure shape ever occurred across 105 real tool calls — the
  validation confirms normalization correctly recognized *that one shape*
  recurring, not that it correctly distinguishes many different real
  shapes from each other (we never observed a second distinct real
  failure to test that against). A stronger test would need either far
  more volume, a flakier/more failure-prone real API, or a tool with a
  richer argument shape than a single free-text string.
- **The invariants attached were never exercised.** Every successful
  search returned well-formed, non-empty results — `search_result_is_structured`
  and `search_result_has_hits` never once fired across all 105 calls. This
  run says nothing about invariant-triggered recovery against real data.
  (The third invariant asked for, argument-schema matching, wasn't
  implemented as a ResilientForge `Invariant` at all — see
  `validation/README.md` for why: it's a pre-condition on the call, not a
  post-condition on the result, and LangGraph's own `ToolNode` already
  enforces it structurally.)
- **Guards were never really tested**, for the reason above — this failure
  class has no argument-level fix to promote.
- **One external API, one framework, one local model.** DuckDuckGo (not
  Tavily, per this exercise's own free-alternative deviation), LangGraph,
  and `qwen2.5:7b` specifically. A hosted model (Claude/GPT) might produce
  different reflection quality/error patterns; a different tool's failure
  modes (a database call, a different search API, a multi-argument tool)
  might exercise signature normalization's *argument*-templating logic
  (untested here, since `search` takes exactly one string argument) far
  more than its error-message redaction logic (which is what actually got
  exercised, since every real failure here was a network-error message,
  not an argument-shape difference).
- **All 3 sessions ran back-to-back on the same day**, not spread across
  different days as originally envisioned — a deliberate scope trade
  (explicitly chosen over waiting) — so this doesn't speak to
  longer-horizon oracle staleness/drift, only to same-day cross-process
  persistence, which it did genuinely exercise (each session is a fresh
  Python process with a freshly-opened `oracle.db`).
- **The one bug found and fixed (async tools) was found by accident** —
  it happened to block the very question this exercise set out to answer.
  A real, honest possibility worth naming: there may be other gaps of
  similar shape (untested against a real, external, unscripted user of the
  library) that this one exercise, run once, against one external agent,
  simply didn't happen to surface.
