# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
