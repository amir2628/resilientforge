"""Live tests: real Anthropic API calls. @pytest.mark.live — opt-in,
deselected by default (pyproject.toml's `addopts = "-m 'not live'"`),
costs real API money per run. Run with:

    ANTHROPIC_API_KEY=sk-... pytest -m live -v

The single biggest gap this file closes: `create_anthropic_reflect` has
been thoroughly unit-tested against hand-rolled fakes since Phase 1 (see
tests/integration/test_raw_tool_loop.py), but its `client=None` branch —
constructing a real `anthropic.Anthropic()` and reading
`ANTHROPIC_API_KEY` from the environment — had never actually executed
anywhere in this codebase's history until these tests. This is the
"dogfooding" proof: a real model proposing a real fix for a real
failure, not a mock standing in for one.

Assertions here are deliberately loose about the EXACT fix a real model
chooses (e.g. a literal date vs. a transform) — a real model isn't
deterministic the way the mocked `reflect()` stubs used everywhere else
in this test suite are. What's asserted is the thing that actually
matters: recovery genuinely happens, and the oracle fast path holds
(zero additional model calls on a second occurrence), regardless of
which specific fix the model proposed.
"""

from __future__ import annotations

import os
import re

import pytest

from resilientforge import wrap
from resilientforge.core.recovery import FailureContext, Fix
from resilientforge.integrations.raw_tool_loop import create_anthropic_reflect

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set — see this file's docstring to run these",
    ),
]

# NOTE: while verifying this file's tests, the Anthropic account available
# had insufficient credits (a real, well-formed request reached the API and
# was rejected only on billing grounds — the create_anthropic_reflect()
# wiring itself was confirmed correct). tests/live/test_local_reflect.py
# covers the same "real model, not mocked" dogfooding goal using a locally-
# hosted model via Ollama instead, and was actually run and verified.

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def flaky_create_event(date: str, title: str = "Event") -> dict:
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}' — expected YYYY-MM-DD")
    return {"date": date, "title": title, "status": "created"}


class CountingReflect:
    """Same idiom as tests/integration/test_engine.py's CountingReflect —
    wraps a real reflect() so we can assert it's NOT called a second time,
    proving the oracle fast path holds even with a real model in the loop."""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[FailureContext] = []

    def __call__(self, context: FailureContext) -> dict:
        self.calls.append(context)
        return self.fn(context)


def test_create_anthropic_reflect_with_real_client_proposes_a_valid_fix():
    reflect = create_anthropic_reflect()  # zero args — the never-before-executed branch

    context = FailureContext(
        tool_name="create_event",
        args={"date": "next Friday", "title": "Standup"},
        error_type="ValueError",
        error_message="could not parse date 'next Friday' — expected YYYY-MM-DD",
    )
    raw = reflect(context)

    fix = Fix.model_validate(raw)  # must be directly usable by generate_fix
    assert fix.strategy
    print(f"\n  real Claude proposed: {fix.model_dump()}")


def test_real_model_recovers_a_genuinely_broken_tool_call(tmp_path):
    wrapped = wrap(
        flaky_create_event,
        oracle_path=tmp_path / "oracle",
        reflect=create_anthropic_reflect(),
    )

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result["status"] == "created"
    assert _ISO_DATE_RE.match(result["date"])
    print(f"\n  real recovery result: {result}")


def test_second_occurrence_resolves_with_zero_additional_model_calls(tmp_path):
    reflect = CountingReflect(create_anthropic_reflect())
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=reflect)

    first = wrapped.invoke(date="next Friday", title="Standup")
    assert len(reflect.calls) == 1
    print(f"\n  first occurrence (real model call): {first}")

    second = wrapped.invoke(date="next Tuesday", title="Retro")

    assert len(reflect.calls) == 1  # NOT called again — real model, still zero extra calls
    assert second["status"] == "created"
    print(f"  second occurrence (oracle fast path, zero model calls): {second}")
