# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-07-16

First public release. `0.2.0` was an internal development-cycle marker,
never tagged or published — this entry supersedes it as the actual first
released version, covering everything shipping for the first time: five
implemented phases, plus three rounds of validation against a real
external agent that found and fixed real bugs before anything went out.

### Added
- **Phase 1 (MVP)**: `wrap()`/`Invariant` core API, a local failure oracle
  (SQLite + vector index), failure-signature normalization, a
  deterministic fix-transform registry, two integrations (a raw
  Anthropic/OpenAI tool-calling loop, LangGraph via
  `ToolNode.wrap_tool_call`), a CLI (`list`/`inspect`/`prune`/`stats`),
  and a failure-injection suite proving recovery across five real failure
  patterns.
- **Phase 2 (standing guards)**: once a fix has proven itself reliable
  enough times, it's promoted into a proactive guard that fixes tool-call
  arguments *before* the first attempt — preventing a recurring failure
  outright instead of merely recovering from it each time. New
  `guards list`/`inspect`/`revoke`/`describe` CLI commands.
- **Phase 3 (speculative branching)**: `wrap(..., num_branches=N,
  side_effect_free=...)` considers up to `N` candidate fixes per attempt.
  By default the tool is still called for real exactly once per attempt —
  a structural guarantee against duplicate real-world side effects,
  relaxed only via the explicit `side_effect_free=True` opt-in.
- **Phase 4 (sandboxed isolation + local dashboard)**: `isolate=True` runs
  every real tool call in a freshly-spawned subprocess, so a hang or
  crash becomes a normal recoverable failure instead of taking down the
  host process. `resilientforge dashboard` serves a read-only, GET-only,
  localhost-only web view of the oracle's recipes/guards/failure history.
- **Phase 5 (production hardening)**: live model validation
  (`create_anthropic_reflect`'s real-client path, plus a new
  `create_local_reflect` for any local OpenAI-compatible endpoint —
  developed and verified against Ollama), `metrics`/`MetricsHook`
  observability, `oracle.db` schema migration, guard/recipe staleness
  safeguards (`GuardManager.prune`, auto-demotion, opt-in
  `recipe_min_success_rate`), and concurrency fixes (WAL mode,
  thread-local sqlite connections, atomic counter updates) found and
  fixed via real concurrent load testing, not inspection.

### Fixed
Found via 3 rounds of validation against a real external LangGraph agent
(`langchain-ai/react-agent`, wrapped with real tools, zero engineered
failures) — see
[`docs/real_world_validation.md`](docs/real_world_validation.md),
[round 2](docs/real_world_validation_round2.md), and
[round 3 + addendum](docs/real_world_validation_round3.md) for the full
detail behind each of these. Every one was found by testing against
something real rather than this project's own synthetic scenarios, and
fixed and re-confirmed against the exact real case that exposed it:

- `integrations/langgraph_adapter.py` had **no support for async tools at
  all** — any LangGraph agent with an `async def` tool broke
  unconditionally, on the very first call, because the adapter only ever
  registered LangGraph's sync `wrap_tool_call` hook. Fixed with a new
  `awrap_tool_call` path (`make_resilientforge_async_tool_call_wrapper`),
  wired in automatically by `make_tool_node` alongside the existing sync
  wrapper.
- Failure-signature normalization **over-collapsed two different real
  problems into one signature**: an HTTP `403 Forbidden` (bot detection)
  and a `402 Payment Required` (a paywall) — no fix that could help one
  could ever help the other — used to produce the identical oracle
  signature, because `core/signature.py` redacted the status line's
  reason phrase the same way it redacts free text. It now preserves an
  HTTP status line's reason phrase instead.
- Failure-signature normalization also **under-collapsed one real problem
  into two signatures**: two different real PDFs failing to decode as
  UTF-8 text, for the identical underlying reason, produced different
  signatures because the specific failing byte (`0x8f` vs. `0x80`) wasn't
  redacted as a unit — only its leading digit was. Hex-literal redaction
  is now a dedicated normalization pass, ahead of the existing
  decimal-number pass.
- A recovery attempt whose proposed fix referenced an **argument that
  doesn't actually exist on the tool** could get silently recorded as
  "recovered" — e.g. a fix patching a `headers` argument onto a tool that
  only accepts `url` was silently dropped by the tool-calling layer, and
  whatever happened next (success or failure, for reasons unrelated to
  the patch) was misattributed to the fix as if it had worked. A `Fix`'s
  `argument_patch` keys, `transforms[].argument` names, and
  `transforms[].transform` names are now validated — against the real
  tool's parameters and against `TRANSFORM_REGISTRY` respectively —
  before ever being applied to a live retry or persisted as a recipe.
  Surfaces as a new, distinct `ResolutionStatus.FIX_REJECTED` rather than
  a false "recovered".

### Changed
- Version bumped `0.2.0` → `0.3.0` (see version-numbering note above);
  classifier remains `Alpha` — this release adds real, validated fixes
  and one new integration capability (async LangGraph tools), not a
  claim of production stability.
