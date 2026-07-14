# Architecture

This describes what's actually built (Phases 1-3). Where reality forced a
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
│   ├── store.py          # SQLite: failures + recipes + guards tables (raw CRUD)
│   ├── vector_index.py   # VectorIndex interface + chromadb implementation
│   ├── recipes.py        # Recipe domain model + RecipeManager (times_applied, success_rate, prune)
│   ├── guards.py         # Phase 2: StandingGuard + GuardManager (promotion, revoke, describe)
│   └── __init__.py       # Oracle — the single facade over store.py + vector_index.py
├── integrations/
│   ├── raw_tool_loop.py     # Anthropic tool_use + OpenAI function-calling shim
│   └── langgraph_adapter.py # LangGraph ToolNode via wrap_tool_call
└── cli/main.py           # list / inspect / prune / stats / guards *
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

0. **(Phase 2)** Check for an active standing guard matching this tool;
   if one exists, proactively apply it to the args *before* the first
   attempt. See "Standing guards" below.
1. Call the tool (with whatever step 0 produced). Catch an exception, or
   evaluate `invariants` against the result if it didn't raise.
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
   recipe already existed — and, **(Phase 2)**, check whether it's now
   reliable enough to promote into a standing guard.
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

## Standing guards (Phase 2) — prevention, not just recovery

Phase 1's recovery loop always fails once before fixing anything: even the
1000th occurrence of an already-learned failure shape burns one
guaranteed-to-fail call before the recipe kicks in on retry. A standing
guard (`oracle/guards.py`'s `StandingGuard` + `GuardManager`) skips that
wasted call entirely, once a recipe has proven itself — this is what
"invariants checked continuously, not just reactively" and "prevented
rather than merely recovered from" (the two Phase 2 goals) turn into
concretely.

**Promotion.** After a recovery succeeds (`WrappedAgent._maybe_promote_guard`),
if the recipe's been applied at least `guard_promotion_min_occurrences`
times (default 3) at `guard_promotion_min_success_rate` or better (default
0.8), each part of its `Fix` gets promoted into a guard:
`argument_patch` entries become `kind="patch"` guards, `transforms`
entries become `kind="transform"` guards — but only for transforms in
`GUARD_SAFE_TRANSFORMS`, a stricter allowlist than `TRANSFORM_REGISTRY`
(see below).

**Matching is structurally different from recipes, on purpose.** A guard
is keyed by `(tool_name, argument, kind)`, not by a failure `signature`
the way a recipe is — pre-call, before the tool has even been attempted,
there's no `error_type`/`error_message` yet to build a signature from.
This is also why `oracle/store.py`'s `guards` table is deliberately **not**
indexed into the vector store the way `recipes` are: guard matching is
exact-key, never fuzzy, and indexing a guard's pseudo-signature into the
same collection `find_similar_failures` queries would risk it surfacing
as a spurious match during ordinary *recipe* lookup.

**Occurrence counting is dual-mode**, honoring the spec's literal
wording ("recurs N times for a given *workflow*"): scoped to
`workflow_id` via `Oracle.list_failures(signature=, workflow_id=)` when
one was given to `wrap()`, otherwise the recipe's own global
`times_applied`.

**Not every transform is safe to apply proactively.** Four of
`TRANSFORM_REGISTRY`'s five entries are idempotent-or-raise: they leave
an already-valid value unchanged, or raise `TransformError` on input they
can't handle — in which case `WrappedAgent._apply_standing_guards`
treats the guard as a no-op for that call and the original args flow
through unmodified, normal Phase 1 behavior resuming from there.
`coerce_str` is the exception: `str(value)` unconditionally succeeds for
*any* input, so as a proactive guard (unlike as a reactive recipe
replay, where it's already been proven a string was needed) it would
silently stringify an already-correct non-string value on a call that
would otherwise have succeeded fine. `GUARD_SAFE_TRANSFORMS` in
`core/recovery.py` excludes it explicitly, with a regression test
(`test_coerce_str_is_excluded_from_guard_safe_transforms`) asserting
that, so a future always-succeeding transform doesn't get added to the
allowlist without someone re-deriving that reasoning first. Patch-kind
guards don't need an allowlist: they only fill a *missing* key
(`setdefault` semantics), never overwrite a caller-provided value, which
already guarantees "never break an otherwise-fine call" by construction.

**Revocation is sticky.** Once a guard is explicitly revoked (`active =
False`, via `GuardManager.revoke()` or `resilientforge guards revoke`),
automatic promotion refuses to silently reactivate it — an operator's
explicit "no" takes precedence.

**The "system-prompt constraint" flavor** the spec also names is exposed
as `GuardManager.describe()` / `WrappedAgent.describe_guards()` /
`resilientforge guards describe` — plain text a caller can splice into
their own system prompt. It is never auto-injected: neither integration
has any system-prompt or conversation access to begin with (see
"Integrations" below), so this was never architecturally possible as
anything but caller-driven.

`tests/failure_injection/scenarios/recurring_date_guard.py` is the
dedicated proof: 8 trials, the first 3 cross the promotion threshold
reactively, the remaining 5 use dates never seen in any prior trial and
must all be *prevented* (zero retries) — not merely recovered from — to
prove the guard generalizes rather than replaying a cached answer. See
the `prevention_rate` column in the failure-injection report.

## Speculative branching (Phase 3) — multiple candidates, one safety rule

Phase 1/2's recovery loop always considers exactly one `Fix` per attempt.
Phase 3 adds `num_branches` (default `1`, today's behavior byte-for-byte)
and `side_effect_free` (default `False`) to `wrap()`/`WrappedAgent`,
letting a caller ask for several candidate fixes to be generated and
evaluated in one round, instead of committing to the first one proposed.

**No new `Branch`/fork type, no new oracle schema.** `apply_fix(args,
fix)` was already a pure function — it never mutates `args` or calls the
tool — so it already *is* an in-process, diff-based fork; Phase 3 just
generates several `Fix` objects instead of one. `oracle/recipes.py`'s
`recipes` table is still one row per signature: only the eventual
winning `Fix` is ever persisted, through the exact same
`RecipeManager.record_success` → `_maybe_promote_guard` path Phase 1/2
already used (`WrappedAgent._on_attempt_success`/`_on_attempt_failure`,
factored out of `invoke()` specifically so all three paths — the
original single-candidate path and both new multi-candidate paths —
share one bookkeeping implementation and can't silently diverge).

**The safety question this had to resolve first.** Considering multiple
candidates could mean calling the real tool once per candidate — for a
tool with real side effects (booking something, sending something), that
risks duplicate real-world actions. `side_effect_free` is the per-tool,
caller-vouched opt-in that controls this:

- `side_effect_free=False` (the default whenever `num_branches>1`):
  candidates are ranked *without* calling the tool — a recipe candidate
  with `success_rate > 0.5` ranks first (a real, persisted number, never
  fabricated), everything else keeps generation order (recipe, then
  reflection calls in the order made) as a documented tie-break, not a
  claimed confidence ranking. The tool is then called for real **exactly
  once**, with the top-ranked survivor — a structural guarantee
  (`WrappedAgent._try_best_proxy_ranked`), not just a tested claim:
  `num_branches` can be arbitrarily large and the real-call count per
  attempt never changes.
- `side_effect_free=True`: an explicit vouch that `tool_fn` has no
  problematic real-world effect regardless of which arguments it's
  called with — closer to HTTP's "safe" methods (GET/HEAD) than
  "idempotent" ones (PUT/DELETE), which is why the flag isn't named
  `idempotent`: a tool can be idempotent in the classic sense
  ("same input twice is a no-op") while still being completely unsafe to
  call speculatively with several *different* candidate inputs (e.g.
  `set_status`). With this opt-in, `WrappedAgent._try_all_real` calls the
  tool for real once per candidate, in ranked order, until one **fully
  passes invariants** — first-fully-passing wins, not "best passing":
  invariants are boolean, so passing candidates have no finer signal to
  rank by. This is the one path where a candidate is genuinely verified
  against real results rather than filtered by a proxy — every
  tried-and-failed candidate becomes its own `RecoveryAttempt`, a
  documented behavior change: one `attempt_number` round can now produce
  up to `num_branches` real calls, worst case `max_recovery_attempts *
  num_branches` total for one `invoke()`.

**Candidate generation reuses `previous_attempts` for a narrower,
documented purpose.** Each `reflect()` call within a round sees every
candidate proposed *so far this round* (not just prior rounds' real
failures) so a real model naturally diversifies instead of proposing the
same fix `num_branches` times over. This is scoped to *within a round*
only: `RecoveryAttempt`'s existing meaning ("actually executed and
failed") is preserved — a candidate that was generated but never
actually called for real (the default path only calls the winner) never
pollutes a *later round's* `previous_attempts`, only the executed ones
do, exactly as Phase 1/2.

**A real, honest gap, not glossed over**: the original spec's language
("a verifier scores branches against the invariants") is only literally
true on the `side_effect_free=True` path, because invariants evaluate a
*result*, and there is no result for a candidate that's never been
executed. The default (`side_effect_free=False`) path cannot do
invariant-based scoring — it's an honest filter-plus-tie-break over data
that already exists (a recipe's real `success_rate`), not a fabricated
confidence number.

`num_branches`/`side_effect_free` are threaded through every existing
entry point the same way every other `wrap()` parameter already is:
`wrap_tools()` in `integrations/raw_tool_loop.py` and
`make_resilientforge_tool_call_wrapper`/`make_tool_node` in
`integrations/langgraph_adapter.py` — there's no per-tool override in
either adapter; a genuinely different vouch per tool needs a direct
`wrap()` call instead.

`tests/failure_injection/scenarios/ambiguous_fix_candidates.py` is the
dedicated proof: a failure whose correct fix depends on a hidden rule
that isn't derivable from the arguments alone, so neither a
`transforms` entry (a pure function of the current value can't guess a
hidden rule without hardcoding it) nor a plain `argument_patch` (unsafe
here — the right value differs by occurrence) can safely cache the
answer across occurrences. `side_effect_free=True` real verification
recovers 100% of trials anyway, at a real, reported cost: unlike every
other scenario, `oracle_hit_rate_after_first` does **not** reach 100%
here, because filling a candidate batch means `reflect()` is consulted
every round regardless of whether a recipe already exists (see the new
`avg_candidates_considered` column in the failure-injection report,
and `test_recovery_rate.py`'s explicit exemption of `num_branches>1`
scenarios from the oracle-hit-rate assertion).

## The oracle (`oracle/`)

Two backends behind one `Oracle` facade:
- **SQLite** (`store.py`): `failures` (one row per occurrence), `recipes`
  (one row per distinct signature that has a known fix), and `guards`
  (Phase 2 — one row per `(tool_name, argument, kind)`, see "Standing
  guards" above) tables.
- **Vector index** (`vector_index.py`): embeddings of the normalized
  signature text, for the fuzzy fallback when there's no exact recipe
  match. Guards are deliberately never indexed here.

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
`tests/unit/test_guards.py` (Phase 2) follows the same reasoning for
`oracle/guards.py`; `tests/failure_injection/test_guard_prevention.py`
is the dedicated prevention-rate proof, alongside `recurring_date_guard`
being added to the standard six-scenario report in `test_recovery_rate.py`.
`tests/integration/test_speculative_branching.py` (Phase 3) is its own
file rather than growing the already-large `test_engine.py` — it covers
the safety-boundary proof (never more than one real call per attempt
when `side_effect_free=False`, regardless of `num_branches`), proxy
ranking, real-verification rejection of a candidate that applies cleanly
but fails a real invariant, and the misconfiguration warning; alongside
`ambiguous_fix_candidates` being added to the seven-scenario report.
