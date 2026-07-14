# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
