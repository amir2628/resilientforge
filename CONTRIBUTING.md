# Contributing

Phase 1 (the MVP) through Phase 5 (production hardening: live model
validation, observability, schema migration, staleness safeguards, and
concurrency fixes) are implemented —
[`docs/architecture.md`](docs/architecture.md) documents what was
actually built, including a few deliberate deviations from the original
plan (each one flagged there and in the commit that made it).

## Ground rules

- Each phase must be fully working, tested, and demoable before the next
  begins — don't start Phase 3 (speculative branching) until it's
  confirmed Phase 1 and 2 acceptance criteria still hold.
- If a design decision turns out to be wrong once real code is written,
  stop and flag it — update `docs/architecture.md` to match reality
  rather than silently diverging.
- Signature normalization (`core/signature.py`) is the crux of the whole
  project. Changes there should be justified by `pytest
  tests/failure_injection` numbers, not intuition.

## Setup

```bash
pip install -e ".[dev,langgraph,dashboard,isolation]"
```

## Tests

```bash
pytest tests/unit tests/integration          # fast, no network — required for every PR
pytest tests/failure_injection                # the recovery-rate proof — required before merging engine/signature/recovery changes
pytest -m live                                 # opt-in, real API calls — not required per-PR
```

Three tiers. Everything in `tests/unit` and
`tests/integration` mocks the model call — no API key needed to run the
default CI gate. `tests/failure_injection` also needs no API key (the
`reflect` in each scenario is a hand-written stand-in, not a real model
call) — it's what generates the recovery-rate numbers in the README.

`tests/live/` (Phase 5) is the only tier that makes a real model call.
Two ways to run it, either works: `tests/live/test_anthropic_reflect.py`
needs `ANTHROPIC_API_KEY` set and a funded account;
`tests/live/test_local_reflect.py` needs a local
[Ollama](https://ollama.com) server running with a model pulled
(`ollama pull qwen2.5:7b` — see that file's docstring for why 7b, not a
smaller one) and needs no API key or cost at all. Each file skips
cleanly with a clear message if its prerequisite isn't available;
neither runs in CI.

## Adding a failure-injection scenario

Each file in `tests/failure_injection/scenarios/` exports one
`SCENARIO = FailureScenario(...)` (see `tests/failure_injection/harness.py`
for the contract): a `make_tool()` factory (a *factory*, not a bare
function — some scenarios need fresh per-run state), a list of `trials`
(kwargs dicts — vary the literal values, keep the failure *shape* the
same, so the suite can measure whether recovery generalizes), and a mock
`reflect`. Add the new scenario to the `SCENARIOS` list in
`test_recovery_rate.py`. Favor scenarios backed by real observed failure
patterns over speculative ones, matching the discipline the current seven
follow.

## Style

- Format/lint with `ruff` (`ruff check .`).
- Type hints throughout; Pydantic models for structured data (invariants,
  fixes, recipes, guards).
- `core/` (signature, invariants, recovery, engine) never imports
  `anthropic`/`openai`/`langgraph` — any real model call is injected as a
  callable (`ReflectFn`, `judge`), never hardcoded. Vendor-specific code
  belongs in `integrations/`.
