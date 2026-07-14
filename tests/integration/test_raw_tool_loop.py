"""Integration tests for integrations/raw_tool_loop.py: the Anthropic
tool_use adapter, the OpenAI function-calling shim (including the
malformed-JSON-args recovery path), and the Anthropic-backed reflect
factory — all against fakes/doubles, no real network call."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from resilientforge.core.recovery import FailureContext, Fix
from resilientforge.integrations.raw_tool_loop import (
    create_anthropic_reflect,
    execute_anthropic_tool_use,
    execute_openai_tool_call,
    make_json_arg_parser,
    wrap_tools,
)
from resilientforge.oracle import Oracle

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_event(date: str, title: str = "Event") -> dict:
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


def send_email(to: str, body: str) -> dict:
    return {"to": to, "body": body, "status": "sent"}


def date_fixing_reflect(context: FailureContext) -> dict:
    return {
        "strategy": "reformat_argument",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


def json_repair_reflect(context: FailureContext) -> dict:
    return {
        "strategy": "repair_json",
        "transforms": [{"argument": "raw_args", "transform": "repair_common_json_errors"}],
    }


class CountingReflect:
    def __init__(self, fn):
        self.fn = fn
        self.calls: list[FailureContext] = []

    def __call__(self, context: FailureContext) -> dict:
        self.calls.append(context)
        return self.fn(context)


@dataclass
class _FakeToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeOpenAIToolCall:
    id: str
    function: _FakeFunction


# -- wrap_tools shares one oracle ---------------------------------------------


def test_wrap_tools_shares_one_oracle_across_tools(tmp_path):
    wrapped = wrap_tools(
        {"create_event": create_event, "send_email": send_email},
        oracle_path=tmp_path / "oracle",
    )
    assert wrapped["create_event"].oracle is wrapped["send_email"].oracle
    assert wrapped["create_event"].tool_name == "create_event"
    assert wrapped["send_email"].tool_name == "send_email"


# -- execute_anthropic_tool_use ------------------------------------------------


def test_execute_anthropic_tool_use_success(tmp_path):
    wrapped = wrap_tools({"create_event": create_event}, oracle_path=tmp_path / "oracle")
    block = _FakeToolUseBlock(id="tu_1", name="create_event", input={"date": "2026-03-05"})

    result = execute_anthropic_tool_use(wrapped, block)

    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tu_1"
    assert "is_error" not in result
    assert '"status": "created"' in result["content"]


def test_execute_anthropic_tool_use_unknown_tool(tmp_path):
    wrapped = wrap_tools({"create_event": create_event}, oracle_path=tmp_path / "oracle")
    block = _FakeToolUseBlock(id="tu_1", name="not_a_real_tool", input={})

    result = execute_anthropic_tool_use(wrapped, block)

    assert result["is_error"] is True
    assert "not_a_real_tool" in result["content"]


def test_execute_anthropic_tool_use_recovers_via_reflect(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap_tools(
        {"create_event": create_event}, oracle_path=tmp_path / "oracle", reflect=reflect
    )
    block = _FakeToolUseBlock(id="tu_1", name="create_event", input={"date": "next Friday"})

    result = execute_anthropic_tool_use(wrapped, block)

    assert "is_error" not in result
    assert '"status": "created"' in result["content"]
    assert len(reflect.calls) == 1


def test_execute_anthropic_tool_use_second_occurrence_zero_model_calls(tmp_path):
    reflect = CountingReflect(date_fixing_reflect)
    wrapped = wrap_tools(
        {"create_event": create_event}, oracle_path=tmp_path / "oracle", reflect=reflect
    )

    execute_anthropic_tool_use(
        wrapped, _FakeToolUseBlock(id="tu_1", name="create_event", input={"date": "next Friday"})
    )
    assert len(reflect.calls) == 1

    result = execute_anthropic_tool_use(
        wrapped, _FakeToolUseBlock(id="tu_2", name="create_event", input={"date": "next Tuesday"})
    )

    assert len(reflect.calls) == 1  # not called again — fast path via recipe
    assert "is_error" not in result


def test_execute_anthropic_tool_use_exhausted_returns_error_result(tmp_path):
    wrapped = wrap_tools(
        {"create_event": create_event}, oracle_path=tmp_path / "oracle", reflect=None
    )
    block = _FakeToolUseBlock(id="tu_1", name="create_event", input={"date": "not a date"})

    result = execute_anthropic_tool_use(wrapped, block)

    assert result["is_error"] is True
    assert "exhausted" in result["content"]


def test_execute_anthropic_tool_use_abort_returns_error_result(tmp_path):
    from resilientforge.core.invariants import Invariant

    invariant = Invariant(
        name="no_delete", check=lambda r: r.get("action") != "delete", on_violation="abort"
    )

    def dangerous(action: str) -> dict:
        return {"action": action}

    wrapped = wrap_tools(
        {"dangerous": dangerous},
        invariants={"dangerous": [invariant]},
        oracle_path=tmp_path / "oracle",
    )
    block = _FakeToolUseBlock(id="tu_1", name="dangerous", input={"action": "delete"})

    result = execute_anthropic_tool_use(wrapped, block)

    assert result["is_error"] is True
    assert "aborted" in result["content"]


# -- execute_openai_tool_call (thin shim, including malformed-JSON path) -------


def test_execute_openai_tool_call_success(tmp_path):
    oracle = Oracle(tmp_path / "oracle")
    wrapped = wrap_tools({"create_event": create_event}, oracle=oracle)
    parser = make_json_arg_parser(oracle)
    call = _FakeOpenAIToolCall(
        id="call_1",
        function=_FakeFunction(name="create_event", arguments='{"date": "2026-03-05"}'),
    )

    result = execute_openai_tool_call(wrapped, call, json_parser=parser)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_1"
    assert "is_error" not in result
    assert '"status": "created"' in result["content"]


def test_execute_openai_tool_call_recovers_malformed_json_args(tmp_path):
    oracle = Oracle(tmp_path / "oracle")
    wrapped = wrap_tools({"create_event": create_event}, oracle=oracle)
    reflect = CountingReflect(json_repair_reflect)
    parser = make_json_arg_parser(oracle, reflect=reflect)
    # trailing comma — invalid JSON, exactly the malformed-args pattern
    call = _FakeOpenAIToolCall(
        id="call_1",
        function=_FakeFunction(name="create_event", arguments='{"date": "2026-03-05",}'),
    )

    result = execute_openai_tool_call(wrapped, call, json_parser=parser)

    assert "is_error" not in result
    assert '"status": "created"' in result["content"]
    assert len(reflect.calls) == 1


def test_execute_openai_tool_call_second_malformed_json_resolves_with_zero_model_calls(tmp_path):
    oracle = Oracle(tmp_path / "oracle")
    wrapped = wrap_tools({"create_event": create_event}, oracle=oracle)
    reflect = CountingReflect(json_repair_reflect)
    parser = make_json_arg_parser(oracle, reflect=reflect)

    execute_openai_tool_call(
        wrapped,
        _FakeOpenAIToolCall(
            id="call_1", function=_FakeFunction(name="create_event", arguments='{"date": "2026-03-05",}')
        ),
        json_parser=parser,
    )
    assert len(reflect.calls) == 1

    result = execute_openai_tool_call(
        wrapped,
        _FakeOpenAIToolCall(
            id="call_2", function=_FakeFunction(name="create_event", arguments='{"date": "2026-04-01",}')
        ),
        json_parser=parser,
    )

    assert len(reflect.calls) == 1  # the JSON-repair recipe generalized across occurrences
    assert "is_error" not in result


def test_execute_openai_tool_call_unrepairable_json_reports_parse_error(tmp_path):
    oracle = Oracle(tmp_path / "oracle")
    wrapped = wrap_tools({"create_event": create_event}, oracle=oracle)
    parser = make_json_arg_parser(oracle, reflect=None)
    call = _FakeOpenAIToolCall(
        id="call_1", function=_FakeFunction(name="create_event", arguments="not json at all {{{")
    )

    result = execute_openai_tool_call(wrapped, call, json_parser=parser)

    assert result["is_error"] is True
    assert "could not parse tool arguments" in result["content"]


def test_execute_openai_tool_call_unknown_tool(tmp_path):
    oracle = Oracle(tmp_path / "oracle")
    wrapped = wrap_tools({"create_event": create_event}, oracle=oracle)
    parser = make_json_arg_parser(oracle)
    call = _FakeOpenAIToolCall(
        id="call_1", function=_FakeFunction(name="not_a_real_tool", arguments="{}")
    )

    result = execute_openai_tool_call(wrapped, call, json_parser=parser)

    assert result["is_error"] is True
    assert "not_a_real_tool" in result["content"]


def test_json_arg_parser_shares_recipes_across_different_tools(tmp_path):
    """A broken-JSON recipe is a syntactic fix, not a tool-specific one —
    confirm it generalizes to a SECOND, different tool sharing the oracle."""
    oracle = Oracle(tmp_path / "oracle")
    wrapped = wrap_tools({"create_event": create_event, "send_email": send_email}, oracle=oracle)
    reflect = CountingReflect(json_repair_reflect)
    parser = make_json_arg_parser(oracle, reflect=reflect)

    execute_openai_tool_call(
        wrapped,
        _FakeOpenAIToolCall(
            id="call_1", function=_FakeFunction(name="create_event", arguments='{"date": "2026-03-05",}')
        ),
        json_parser=parser,
    )
    assert len(reflect.calls) == 1

    result = execute_openai_tool_call(
        wrapped,
        _FakeOpenAIToolCall(
            id="call_2",
            function=_FakeFunction(
                name="send_email", arguments='{"to": "a@example.com", "body": "hi",}'
            ),
        ),
        json_parser=parser,
    )

    assert len(reflect.calls) == 1  # still just once — the recipe generalized to a new tool
    assert "is_error" not in result


# -- create_anthropic_reflect (fake client, no network) -------------------------


class _FakeContentBlock:
    def __init__(self, type_, input_=None):
        self.type = type_
        self.input = input_


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeMessagesResource:
    def __init__(self, response_content):
        self.response_content = response_content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(self.response_content)


class _FakeAnthropicClient:
    def __init__(self, response_content):
        self.messages = _FakeMessagesResource(response_content)


def test_create_anthropic_reflect_builds_request_and_parses_response():
    fix_input = {
        "strategy": "reformat_argument",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }
    client = _FakeAnthropicClient([_FakeContentBlock("tool_use", fix_input)])
    reflect = create_anthropic_reflect(client=client, model="claude-sonnet-5")

    context = FailureContext(
        tool_name="create_event",
        args={"date": "next Friday"},
        error_type="ValueError",
        error_message="could not parse date 'next Friday'",
    )
    raw = reflect(context)

    assert raw == fix_input
    fix = Fix.model_validate(raw)  # must be directly usable by generate_fix
    assert fix.strategy == "reformat_argument"

    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["tool_choice"] == {"type": "tool", "name": "propose_fix"}
    assert call["tools"][0]["name"] == "propose_fix"
    assert "create_event" in call["messages"][0]["content"]
    assert "next Friday" in call["messages"][0]["content"]


def test_create_anthropic_reflect_includes_previous_attempts_in_prompt():
    prior_fix = Fix(strategy="noop", argument_patch={})
    client = _FakeAnthropicClient(
        [_FakeContentBlock("tool_use", {"strategy": "x", "argument_patch": {}})]
    )
    reflect = create_anthropic_reflect(client=client)

    context = FailureContext(
        tool_name="t", args={}, attempt_number=2, previous_attempts=[prior_fix]
    )
    reflect(context)

    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "noop" in prompt


def test_create_anthropic_reflect_raises_if_no_tool_use_in_response():
    client = _FakeAnthropicClient([_FakeContentBlock("text", None)])
    reflect = create_anthropic_reflect(client=client)

    with pytest.raises(RuntimeError):
        reflect(FailureContext(tool_name="t", args={}))
