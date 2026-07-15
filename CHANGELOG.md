# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-07-15

### Added
- Phase 5 complete: production-hardening pass across 8 tracks, prompted
  by a direct "is this production ready?" assessment after Phase 4 that
  identified 8 concrete gaps. Two real bugs were found and fixed by
  actually exercising the new tests, not by inspection:
  `oracle/store.py` moved from one shared `sqlite3.Connection` to
  thread-local connections (a shared connection, even with
  `check_same_thread=False`, raised `sqlite3.InterfaceError` under real
  concurrent access) and from Python-level read-modify-write counter
  updates to single atomic SQL statements
  (`record_recipe_success`/`record_recipe_fast_path_failure`/
  `record_guard_application`) after concurrent load testing reliably
  reproduced lost updates (400 concurrent calls landing at 3, then 17,
  before the fix).
  - **Live validation** (`tests/live/`): `create_anthropic_reflect()`'s
    real-client construction path, previously only unit-tested against
    fakes across 4 phases, actually exercised — plus a new
    `create_local_reflect` (`integrations/raw_tool_loop.py`), backed by
    any local OpenAI-compatible endpoint (developed and verified against
    Ollama), added after a real Anthropic account turned out to have
    insufficient API credits mid-verification. Uses a hand-flattened
    tool schema (`_flat_fix_schema`), not `Fix.model_json_schema()`
    directly — empirically, smaller/local models' tool-calling was
    confused by the raw schema's `$defs`/`$ref` indirection in a way
    Claude never was.
  - **Staleness safeguards**: `GuardManager.prune()` (mirrors
    `RecipeManager.prune` exactly — guards had no equivalent before);
    automatic guard demotion (`guard_demotion_min_occurrences`/
    `guard_demotion_max_failure_rate`, always on — a guard that's fired
    enough times with too high a failure rate auto-revokes via the
    existing sticky `revoke()`); opt-in `recipe_min_success_rate` +
    `recipe_reliability_min_occurrences` (skip a recipe that's stopped
    working instead of always trying it first).
  - **Schema migration**: `oracle.db` now stamps `PRAGMA user_version`
    and runs an ordered migration list on open — every pre-Phase-5
    database (implicitly `user_version=0`) migrates cleanly; opening a
    newer database than the code understands raises clearly instead of
    silently misreading it.
  - **Concurrency**: `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout`
    (measured ~4.4x throughput and much lower tail latency under write
    contention vs. the pre-Phase-5 default journal mode — real numbers
    in `docs/architecture.md`), plus the connection/counter-atomicity
    fixes above. New `tests/load/test_concurrency.py`
    (`@pytest.mark.load`, opt-in).
  - **Observability** (`telemetry/metrics.py`, previously an empty
    stub): `wrap(..., metrics=...)`, a vendor-neutral `MetricsHook`
    (same injected-callable pattern as `reflect`) emitting `call_result`/
    `recovery_resolved`/`guard_fired`/`guard_promoted`/`guard_revoked`
    events live as `invoke()` runs — distinct from the dashboard, which
    shows persisted state after the fact. `LoggingMetricsHook` is a
    zero-dependency stdlib-`logging` reference implementation.
  - **Isolation picklability**: new optional `isolation` extra adds
    `cloudpickle` as a fallback when stdlib `pickle` can't serialize
    `tool_fn` (closures/lambdas) — tries stdlib pickle first (unchanged
    fast path), only serializes via cloudpickle (bytes crossing the
    actual process boundary, never cloudpickle objects themselves) when
    that fails and the extra is installed. Found and documented a real,
    non-obvious consequence: mutable state a closure captures does NOT
    persist across separate isolated calls, since each call gets an
    independent subprocess.
  - **Embedder quality**: `tests/unit/test_embedder_quality.py` runs a
    labeled, realistic benchmark against the default hashing embedder
    (recall 1.00, precision ~0.55) and a new optional `semantic` extra
    (`oracle/semantic_embedding.py`'s `SentenceTransformerEmbeddingFunction`,
    ~1GB installed) — reported honestly: the semantic embedder did
    **not** outperform the free default on this benchmark (precision
    ~0.50, slightly worse), a genuine, unflattering result kept exactly
    as measured rather than tuned to look better.
  - Version bumped `0.1.0.dev0` → `0.2.0`, classifier `Pre-Alpha` →
    `Alpha` — a modest, evidence-based bump, not a claim of
    battle-tested production maturity this project doesn't have
    evidence for yet. New dormant `.github/workflows/release.yml`
    (tag-triggered build; publishes only if a `PYPI_API_TOKEN` secret is
    configured).
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
