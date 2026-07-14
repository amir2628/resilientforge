"""Unit tests for the Invariant interface."""

from __future__ import annotations

from pydantic import BaseModel

from resilientforge.core.invariants import Invariant, is_instance_of, not_none


# -- base construction / deterministic predicate -----------------------------


def test_plain_predicate_invariant_evaluates_true_and_false():
    inv = Invariant(name="is_positive", check=lambda result: result > 0)

    assert inv.evaluate(5) is True
    assert inv.evaluate(-5) is False


def test_invariant_defaults():
    inv = Invariant(name="x", check=lambda result: True)

    assert inv.on_violation == "recover"
    assert inv.severity == "medium"


def test_invariant_custom_on_violation_and_severity():
    inv = Invariant(name="x", check=lambda result: True, on_violation="abort", severity="high")

    assert inv.on_violation == "abort"
    assert inv.severity == "high"


def test_invariant_rejects_invalid_on_violation():
    import pytest
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        Invariant(name="x", check=lambda result: True, on_violation="not-a-real-option")


# -- Pydantic model validation kind ------------------------------------------


class _EventResult(BaseModel):
    title: str
    attendees: list[str]


def test_from_pydantic_model_passes_for_valid_result():
    inv = Invariant.from_pydantic_model("output_schema_valid", _EventResult)

    assert inv.evaluate({"title": "Standup", "attendees": ["a@x.com"]}) is True


def test_from_pydantic_model_fails_for_invalid_result():
    inv = Invariant.from_pydantic_model("output_schema_valid", _EventResult)

    assert inv.evaluate({"title": "Standup"}) is False  # missing attendees
    assert inv.evaluate({"title": 123, "attendees": []}) is False  # wrong type
    assert inv.evaluate("not even a dict") is False


# -- LLM-judged kind (judge call mocked) -------------------------------------


def test_llm_judged_invariant_uses_injected_judge_not_a_real_model_call():
    calls = []

    def fake_judge(rule: str, result) -> bool:
        calls.append((rule, result))
        return "delete" not in str(result)

    inv = Invariant.llm_judged(
        name="no_destructive_fs_action",
        rule="no destructive filesystem action outside the working directory",
        judge=fake_judge,
    )

    assert inv.evaluate({"action": "read", "path": "a.txt"}) is True
    assert inv.evaluate({"action": "delete", "path": "/etc/passwd"}) is False
    assert calls == [
        ("no destructive filesystem action outside the working directory", {"action": "read", "path": "a.txt"}),
        ("no destructive filesystem action outside the working directory", {"action": "delete", "path": "/etc/passwd"}),
    ]


def test_llm_judged_invariant_coerces_truthy_judge_return_to_bool():
    inv = Invariant.llm_judged(name="x", rule="r", judge=lambda rule, result: 1)
    assert inv.evaluate("anything") is True


# -- built-in common invariants ----------------------------------------------


def test_not_none_builtin():
    inv = not_none()
    assert inv.name == "not_none"
    assert inv.evaluate(0) is True  # falsy but not None
    assert inv.evaluate(None) is False


def test_is_instance_of_builtin_single_type():
    inv = is_instance_of(dict)
    assert inv.name == "is_instance_of_dict"
    assert inv.evaluate({"a": 1}) is True
    assert inv.evaluate([1, 2]) is False


def test_is_instance_of_builtin_multiple_types_and_custom_name():
    inv = is_instance_of((int, float), name="is_numeric")
    assert inv.name == "is_numeric"
    assert inv.evaluate(5) is True
    assert inv.evaluate(5.5) is True
    assert inv.evaluate("5") is False
