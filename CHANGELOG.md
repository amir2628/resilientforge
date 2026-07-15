# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Phase 4 (partial) complete: sandboxed isolation and a local dashboard —
  oracle federation was explicitly deferred (the spec itself hedges it as
  "optional" with zero elaboration).
  - **Sandboxed isolation** (`core/isolation.py`'s `run_isolated`,
    `IsolationError`, `check_picklable`): `wrap(..., isolate=True,
    call_timeout=..., max_memory_mb=..., max_cpu_seconds=...)` runs every
    real tool call in a freshly-spawned subprocess, so a hang or crash
    becomes a normal recoverable failure instead of taking down the host
    process — protective isolation of the *caller*, not undoing a
    real-world side effect the tool already performed (no code-level
    sandbox can do that). Requires `tool_fn` to be picklable (checked
    eagerly at construction); a locally-defined closure or lambda won't
    work. `max_memory_mb`/`max_cpu_seconds` are POSIX-only,
    best-effort — confirmed empirically during development that even
    within POSIX, a given resource limit isn't always honorable (macOS
    refused `RLIMIT_AS` outright in testing while `RLIMIT_CPU` worked
    fine), surfaced as its own distinct, honest error rather than
    silently ignored or misattributed to the tool. Deliberately **not**
    exposed through `integrations/langgraph_adapter.py` — that adapter
    builds a closure over LangGraph's own live `execute` callback per
    call, which cannot be pickled into a subprocess; a real structural
    limitation, documented, not an oversight. New
    `tests/unit/test_isolation.py` + `tests/integration/test_isolation.py`.
  - **Local dashboard** (`dashboard/app.py`'s `create_app`, new
    `resilientforge dashboard` CLI command): a read-only, GET-only
    FastAPI app over one oracle's recipes/guards/failures, viewable in a
    browser instead of the terminal. `fastapi`/`uvicorn` are a new
    optional `dashboard` extra (`pip install resilientforge[dashboard]`),
    never a hard dependency — mirrors the existing `langgraph` extra
    exactly, with `resilientforge/__init__.py` importing nothing from
    `dashboard/`. Binds to `127.0.0.1` by default; the entire front end
    is one inlined HTML/CSS/vanilla-JS string, no build step, no CDN. New
    `tests/unit/test_dashboard.py` and `examples/dashboard_demo.py`.
- Phase 3 complete: speculative branching (`wrap(..., num_branches=N,
  side_effect_free=...)`) — instead of committing to the first proposed
  fix, generate up to `N` candidates per recovery attempt. By default
  (`side_effect_free=False`), candidates are ranked without calling the
  tool (a recipe's real `success_rate` if one exists, generation order
  otherwise) and the tool is still called for real exactly once per
  attempt — a structural guarantee, not just a tested claim, so
  `num_branches` never risks a duplicate real-world side effect. With an
  explicit per-tool opt-in (`side_effect_free=True`, a caller's vouch
  that the tool has no problematic real-world effect regardless of
  arguments — deliberately not named `idempotent`, see
  `docs/architecture.md`), candidates are actually called for real, in
  ranked order, until one fully passes invariants — genuine verification
  against real results, not a guess. No new oracle schema: only the
  eventual winning fix is ever persisted, through the same
  `record_success`/guard-promotion path Phase 1/2 already used. New
  `tests/integration/test_speculative_branching.py` and a dedicated
  `ambiguous_fix_candidates` failure-injection scenario (a failure whose
  correct fix depends on a hidden rule undiscoverable except by real
  trial) with a new `avg_candidates_considered` report column.
- Phase 2 complete: standing guards (`oracle/guards.py`'s `StandingGuard` +
  `GuardManager`) — once a fix has proven itself reliably enough times, it's
  promoted into a proactive guard that fixes tool-call arguments *before*
  the first attempt, preventing a recurring failure outright instead of
  merely recovering from it each time. Occurrence counting is scoped per
  `workflow_id` when one is given to `wrap()`, otherwise global. A stricter
  `GUARD_SAFE_TRANSFORMS` allowlist (excluding `coerce_str`, which is only
  safe as a reactive replay, not a proactive one) governs which learned
  transforms are safe to apply before a call is even attempted. New CLI
  commands (`guards list`/`inspect`/`revoke`/`describe`); `describe()`
  exposes guard text for callers to splice into their own system
  prompt — never auto-injected, since neither integration has system-prompt
  access. New `recurring_date_guard` failure-injection scenario and
  `prevention_rate` metric prove guards generalize to unseen literal
  values, not just replay a cached fix.
- Phase 1 (MVP) complete: `wrap()`/`Invariant` core API, the failure oracle
  (SQLite + local vector index behind one `Oracle` interface), failure
  signature normalization, fix generation/application with a deterministic
  transform registry, the recovery engine tying it together, two
  integrations (a raw Anthropic/OpenAI tool-calling loop, LangGraph via
  `ToolNode.wrap_tool_call`), a CLI (`list`/`inspect`/`prune`/`stats`), and
  the failure-injection suite proving recovery works across five real
  failure patterns — see `docs/architecture.md` and the README's recovery-
  rate table for details.
- Initial repository scaffold: package structure, `pyproject.toml`, Apache 2.0
  license, CI config.
