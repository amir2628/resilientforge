# Contributing

Phase 1 (the MVP) is implemented — read `PROJECT_SPEC.md` before proposing
changes; it's still the source of truth for scope, phasing, and
architecture decisions, and [`docs/architecture.md`](docs/architecture.md)
documents what was actually built, including a few deliberate deviations
from the original plan (each one flagged there and in the commit that
made it).

## Ground rules

- Phase 1 is done — see `PROJECT_SPEC.md` §8 for its acceptance criteria.
  Do not start Phase 2 work (§3) until it's confirmed all still hold.
- If a design decision in `PROJECT_SPEC.md` turns out to be wrong once
  real code is written, stop and flag it — update the spec and
  `docs/architecture.md` to match reality rather than silently diverging.
- Signature normalization (`core/signature.py`) is the crux of the whole
  project. Changes there should be justified by `pytest
  tests/failure_injection` numbers, not intuition — see §10.

## Setup

```bash
pip install -e ".[dev,langgraph]"
```

## Tests

```bash
pytest tests/unit tests/integration          # fast, no network — required for every PR
pytest tests/failure_injection                # the recovery-rate proof — required before merging engine/signature/recovery changes
pytest -m live                                 # opt-in, real API calls — not required per-PR
```

Three tiers, matching `PROJECT_SPEC.md` §7. Everything in `tests/unit` and
`tests/integration` mocks the model call — no API key needed to run the
default CI gate. `tests/failure_injection` also needs no API key (the
`reflect` in each scenario is a hand-written stand-in, not a real model
call) — it's what generates the recovery-rate numbers in the README.

## Adding a failure-injection scenario

Each file in `tests/failure_injection/scenarios/` exports one
`SCENARIO = FailureScenario(...)` (see `tests/failure_injection/harness.py`
for the contract): a `make_tool()` factory (a *factory*, not a bare
function — some scenarios need fresh per-run state), a list of `trials`
(kwargs dicts — vary the literal values, keep the failure *shape* the
same, so the suite can measure whether recovery generalizes), and a mock
`reflect`. Add the new scenario to the `SCENARIOS` list in
`test_recovery_rate.py`. Favor scenarios backed by real observed failure
patterns over speculative ones, matching the discipline the current five
follow (see `PROJECT_SPEC.md` §1).

## Style

- Format/lint with `ruff` (`ruff check .`).
- Type hints throughout; Pydantic models for structured data (invariants,
  fixes, recipes).
- `core/` (signature, invariants, recovery, engine) never imports
  `anthropic`/`openai`/`langgraph` — any real model call is injected as a
  callable (`ReflectFn`, `judge`), never hardcoded. Vendor-specific code
  belongs in `integrations/`.
