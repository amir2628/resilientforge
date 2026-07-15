# Real-world validation, round 2: does signature normalization discriminate, not just recognize?

Full write-up: [`../../docs/real_world_validation_round2.md`](../../docs/real_world_validation_round2.md).
This file documents *setup* and the disclosed deviations; findings live in
the doc above. Independent of round 1 — nothing in `validation/` (round 1)
was touched.

## What's here

- `react-agent/` — a FRESH, independent clone of
  [`langchain-ai/react-agent`](https://github.com/langchain-ai/react-agent),
  same commit as round 1 (`7d1f9832f56d6d29ad9ae248caf0b263c5460145`,
  2026-06-26 — pinned to the same commit deliberately, so round 1 vs round
  2 differ only in what this exercise changed, not in upstream drift).
  `.git` removed, same reasoning as round 1.
- `prompts.py` — 62 new hand-written prompts (none reused from round 1),
  split into `SEARCH_PROMPTS` (22), `EXTRACTION_PROMPTS` (22, real URLs),
  `BOTH_PROMPTS` (18, model must find something then read it).
- `run_validation.py` — independent driver: own oracle path, own metrics
  log, own session log directory.
- `metrics_log.jsonl`, `oracle_export.json`, `session_logs/session_{1,2,3}.jsonl`
  — same format/purpose as round 1's equivalents.

## Disclosed deviations from vanilla react-agent

1. **`tools.py`** — `search` tool: same Tavily → `ddgs` swap as round 1,
   reused verbatim (free/keyless). NEW: `extract_url_content` tool — real
   HTTP fetch (`httpx`) + real HTML-to-text (`BeautifulSoup4`), strict
   UTF-8 decoding (no silent fallback), a realistic browser `User-Agent`
   header. Nothing about this tool is engineered to fail *or* engineered
   to avoid failing — it's what a reasonably careful (not over-defensive)
   implementation looks like. Confirmed, before ever wiring it into the
   agent, that real pages organically produce a genuinely varied set of
   real failures (403s, DNS failures, redirect loops, `UnicodeDecodeError`
   on real PDF bytes) without any synthetic injection.
2. **`graph.py`** — same real integration point as round 1
   (`ToolNode(TOOLS)` → `make_tool_node(...)`), now wrapping both tools
   through one shared `ToolNode`. Config via env vars, same pattern as
   round 1.

Model: `MODEL=ollama/qwen2.5:7b`, same as round 1, no source change needed.

**Invariants**: the same 2 as round 1, generalized to handle either tool's
result shape (`result_is_structured`, `result_is_non_empty`), plus a new
`extracted_content_is_clean_text` for the extraction tool. All 3 apply to
*every* tool call regardless of which tool was invoked — `make_tool_node`
has no per-tool invariants knob, so each invariant is written to be
vacuously true for a result shape it doesn't recognize. Same architectural
note round 1 made about the args-schema invariant applies here too: not
implemented, since `Invariant.check` evaluates a result, not a call's
input, and LangGraph's own `ToolNode` already enforces argument-schema
validity structurally.

**Real URLs used** (extraction-only and both-in-sequence groups) are
genuine, existing, organically-diverse pages (Wikipedia in 4 languages,
government/NGO sites, project homepages, an arXiv PDF, a Project Gutenberg
text, a historic CERN page, French/German news). No synthetic HTTP-testing
endpoints (e.g. httpbin.org) were ever put in front of the agent — those
were used only during tool development, off to the side, to characterize
what real failure modes were achievable at all.

## Scope note: sessions

The task originally called for 5 sessions spread across different
times/days; reduced to 3 by explicit user request while session 1 was
already running. All 3 ran within roughly a 2-hour window the same day —
see the main doc's "what this still doesn't tell us" for what that does
and doesn't cover.

## Reproducing

```bash
cd validation/round2
python3 -m venv .venv && source .venv/bin/activate
pip install -e "../..[langgraph]"
pip install -e ./react-agent
pip install langchain-ollama ddgs httpx beautifulsoup4
ollama pull qwen2.5:7b
python3 run_validation.py   # run 3x for 3 sessions; oracle/metrics persist across runs
```
