# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
