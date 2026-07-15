"""Live tests: a real, locally-hosted model via Ollama (no cloud API key
needed). @pytest.mark.live — opt-in, deselected by default (pyproject.toml's
`addopts = "-m 'not live'"`), since it needs a real Ollama server running
with a model pulled, which CI doesn't have.

This exists alongside tests/live/test_anthropic_reflect.py for a concrete
reason, not duplication: verifying this project's whole real-model
dogfooding story ran into an Anthropic account with insufficient API
credits (the request itself was confirmed well-formed and correctly
authenticated — a billing rejection, not a bug). Using a free, local
model instead let the actual goal — "does a real, non-mocked model
recover a genuinely broken tool call, and does the oracle fast path
still hold?" — get verified anyway.

To run: install Ollama (`brew install ollama`), start it
(`ollama serve`), pull a model (`ollama pull qwen2.5:7b`), then:

    pytest -m live -v tests/live/test_local_reflect.py

Model choice note, from actually testing this: a smaller `qwen2.5:3b`
was tried first and reliably failed to follow the requested tool-call
schema (it invented its own argument structure). `qwen2.5:7b` with
`create_local_reflect`'s flattened schema (see `_flat_fix_schema` in
`integrations/raw_tool_loop.py`) was reliable. This isn't asserted by
these tests (a different/future model could behave differently) — it's
recorded here because it was a real, non-obvious finding.
"""

from __future__ import annotations

import re
import socket
import urllib.request

import pytest

from resilientforge import wrap
from resilientforge.core.recovery import FailureContext, Fix
from resilientforge.integrations.raw_tool_loop import create_local_reflect

_OLLAMA_URL = "http://localhost:11434/api/version"


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(_OLLAMA_URL, timeout=1) as response:
            return response.status == 200
    except (OSError, socket.timeout):
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason=f"no Ollama server reachable at {_OLLAMA_URL} — see this file's docstring to run these",
    ),
]

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def flaky_create_event(date: str, title: str = "Event") -> dict:
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}' — expected YYYY-MM-DD")
    return {"date": date, "title": title, "status": "created"}


class CountingReflect:
    def __init__(self, fn):
        self.fn = fn
        self.calls: list[FailureContext] = []

    def __call__(self, context: FailureContext) -> dict:
        self.calls.append(context)
        return self.fn(context)


def test_create_local_reflect_with_real_client_proposes_a_valid_fix():
    reflect = create_local_reflect()

    context = FailureContext(
        tool_name="create_event",
        args={"date": "next Friday", "title": "Standup"},
        error_type="ValueError",
        error_message="could not parse date 'next Friday' — expected YYYY-MM-DD",
    )
    raw = reflect(context)

    fix = Fix.model_validate(raw)  # must be directly usable by generate_fix
    assert fix.strategy
    print(f"\n  real local model proposed: {fix.model_dump()}")


def test_real_local_model_recovers_a_genuinely_broken_tool_call(tmp_path):
    wrapped = wrap(
        flaky_create_event,
        oracle_path=tmp_path / "oracle",
        reflect=create_local_reflect(),
    )

    result = wrapped.invoke(date="next Friday", title="Standup")

    assert result["status"] == "created"
    assert _ISO_DATE_RE.match(result["date"])
    print(f"\n  real recovery result: {result}")


def test_second_occurrence_resolves_with_zero_additional_model_calls(tmp_path):
    reflect = CountingReflect(create_local_reflect())
    wrapped = wrap(flaky_create_event, oracle_path=tmp_path / "oracle", reflect=reflect)

    first = wrapped.invoke(date="next Friday", title="Standup")
    assert len(reflect.calls) == 1
    print(f"\n  first occurrence (real model call): {first}")

    second = wrapped.invoke(date="next Tuesday", title="Retro")

    assert len(reflect.calls) == 1  # NOT called again — real model, still zero extra calls
    assert second["status"] == "created"
    print(f"  second occurrence (oracle fast path, zero model calls): {second}")
