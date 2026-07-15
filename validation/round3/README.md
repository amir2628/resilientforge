# Real-world validation, round 3: confirming the round 2 fixes

Full write-up: [`../../docs/real_world_validation_round3.md`](../../docs/real_world_validation_round3.md).
**This is a confirmation exercise, not new exploration** — every file here
is reused verbatim from `validation/round2/` (same react-agent commit,
same `tools.py`/`graph.py` patches, same `prompts.py`, same invariants).
The only difference is what `resilientforge` package this venv installs:
round 2 ran against the pre-fix source; round 3 runs against the same
source after Part A (`core/signature.py`: HTTP-status false-merge, hex-byte
missed-match) and Part B (`core/engine.py`/`integrations/langgraph_adapter.py`:
`argument_patch` validation) were fixed.

## What's here

Identical in structure and content to `validation/round2/`:
- `react-agent/` — same commit (`7d1f9832f56d6d29ad9ae248caf0b263c5460145`),
  same two disclosed deviations (ddgs search, `extract_url_content`), same
  `make_tool_node` wiring, same 3 invariants.
- `prompts.py` — byte-for-byte identical to round 2's (same 62 prompts).
- `run_validation.py` — same driver, own oracle/metrics/session-log paths.

## Confirmed against the fixed source

```bash
python -c "import resilientforge.core.signature as sig; \
  print(hasattr(sig, '_HEX_LITERAL_RE'), hasattr(sig, '_HTTP_STATUS_TEXT_RE'))"
# -> True True
```

## Reproducing

```bash
cd validation/round3
python3 -m venv .venv && source .venv/bin/activate
pip install -e "../..[langgraph]"        # the FIXED local resilientforge
pip install -e ./react-agent
pip install langchain-ollama ddgs httpx beautifulsoup4
ollama pull qwen2.5:7b
python3 run_validation.py   # run at least twice
```
