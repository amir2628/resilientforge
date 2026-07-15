# Real-world validation: ResilientForge against react-agent

Full write-up: [`../docs/real_world_validation.md`](../docs/real_world_validation.md).
This file documents *setup* and the disclosed deviations from vanilla
react-agent; the findings themselves live in the doc above.

## What's here

- `react-agent/` — a plain clone of
  [`langchain-ai/react-agent`](https://github.com/langchain-ai/react-agent)
  at commit `7d1f9832f56d6d29ad9ae248caf0b263c5460145` (2026-06-26), with
  `.git` removed so the parent repo tracks its files directly (chosen over a
  git submodule for simplicity — the exact commit hash above is what makes
  this reproducible instead).
- `prompts.py` — 35 hand-written prompts (factual/ambiguous/multi-step),
  not designed to trigger any of the 5 existing synthetic failure-injection
  scenarios.
- `run_validation.py` — driver; each invocation is one "session" (see the
  main doc's session log).
- `metrics_log.jsonl` — the full, timestamped audit trail (every real tool
  call, failure, and recovery attempt) across all 3 sessions, via a small
  dedicated `MetricsHook` (`JsonlMetricsHook` in `react-agent/src/react_agent/graph.py`).
- `oracle_export.json` — a plain-JSON export of the final oracle state
  (failures/recipes/guards) after all 3 sessions, so the findings are
  inspectable without needing the (gitignored, regenerable) sqlite file
  itself. Regenerate the real oracle by rerunning `run_validation.py`.
- `session_logs/session_{1,2,3}.jsonl` — per-prompt results (status,
  elapsed time, final answer) for each of the 3 sessions.
- `.venv/`, `.resilientforge/` — gitignored (already covered by the repo's
  existing top-level `.gitignore` patterns).

## Disclosed deviations from vanilla react-agent

Two changes were made to the cloned copy, both driven by "use free
alternatives" (no Tavily/Anthropic keys were available or wanted for this
exercise) — everything else is untouched:

1. **`react-agent/src/react_agent/tools.py`** — the `search` tool now calls
   `ddgs` (DuckDuckGo, free/keyless) instead of Tavily. Still a real, live,
   unpredictable web search — just a different provider. Real errors (rate
   limiting, network failures, empty results) are left to propagate
   unmodified, same as the original.
2. **`react-agent/src/react_agent/graph.py`** — `ToolNode(TOOLS)` replaced
   with ResilientForge's `make_tool_node(...)`. This one **is** the actual
   integration point a real user would touch to adopt ResilientForge into
   an existing LangGraph app — not a workaround, the real Step 2 of this
   exercise. Configuration (oracle path, metrics log path) is read from env
   vars set by `run_validation.py`, so the file itself doesn't need
   re-editing between sessions.

The model is swapped from Anthropic to a local Ollama model
(`MODEL=ollama/qwen2.5:7b`, set by `run_validation.py`) — this needed no
source change at all: `Context.model` already reads from an env var when
unset (react-agent's own designed extension point), and
`langchain.chat_models.init_chat_model` already supports an `ollama`
provider.

**Note on the third invariant the exercise asked for** ("tool call
arguments match the tool's declared schema"): not implemented as a
ResilientForge `Invariant`, because `Invariant.check` evaluates a
*result*, not a call's input arguments — that check is a pre-condition on
the call, not a post-condition on its outcome. It's already enforced
structurally, for free, by LangGraph's own `ToolNode` (schema validation
before `execute()` ever runs); a mismatch there already surfaces as a real
exception through the normal failure-detection path. The two invariants
actually attached (`search_result_is_structured`, `search_result_has_hits`)
are the ones that fit the abstraction as designed.

## A bug found and fixed along the way

Wrapping react-agent's (async) `search` tool immediately broke
unconditionally — a real, general gap in `langgraph_adapter.py` (it never
supported async tools at all, invisible because its own test suite only
ever used sync ones). Fixed in `src/resilientforge/integrations/langgraph_adapter.py`
with the user's explicit sign-off before touching `src/resilientforge` at
all. Full detail in the main doc and in that module's docstring.

## Reproducing

```bash
cd validation
python3 -m venv .venv && source .venv/bin/activate
pip install -e "..[langgraph]"
pip install -e ./react-agent
pip install langchain-ollama ddgs
ollama pull qwen2.5:7b   # or run against any OpenAI-compatible local endpoint
python3 run_validation.py   # run 3x for 3 sessions; oracle/metrics persist across runs
```
