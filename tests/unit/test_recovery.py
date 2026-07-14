"""Unit tests for core/recovery.py: fix generation (reflection call mocked
— no real model call in this file) and fix
application, including the transform registry."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from resilientforge.core.recovery import (
    GUARD_SAFE_TRANSFORMS,
    TRANSFORM_REGISTRY,
    ArgTransform,
    FailureContext,
    Fix,
    TransformError,
    apply_fix,
    coerce_float,
    coerce_int,
    coerce_str,
    generate_fix,
    parse_relative_date_to_iso,
    repair_common_json_errors,
)


# -- parse_relative_date_to_iso -----------------------------------------------


def test_parse_relative_date_today_tomorrow_yesterday():
    ref = date(2026, 3, 4)  # a Wednesday
    assert parse_relative_date_to_iso("today", today=ref) == "2026-03-04"
    assert parse_relative_date_to_iso("tomorrow", today=ref) == "2026-03-05"
    assert parse_relative_date_to_iso("yesterday", today=ref) == "2026-03-03"


def test_parse_relative_date_in_n_days():
    ref = date(2026, 3, 4)
    assert parse_relative_date_to_iso("in 3 days", today=ref) == "2026-03-07"
    assert parse_relative_date_to_iso("in 1 day", today=ref) == "2026-03-05"


def test_parse_relative_date_next_weekday_matches_spec_example():
    # "next Friday" and "next Tuesday"
    # must both parse correctly — this is the case the whole Fix.transforms
    # design exists to support.
    ref = date(2026, 3, 4)  # Wednesday
    assert parse_relative_date_to_iso("next Friday", today=ref) == "2026-03-06"
    assert parse_relative_date_to_iso("next Tuesday", today=ref) == "2026-03-10"


def test_parse_relative_date_next_weekday_on_that_weekday_means_a_week_out():
    ref = date(2026, 3, 6)  # a Friday
    assert parse_relative_date_to_iso("next Friday", today=ref) == "2026-03-13"


def test_parse_relative_date_case_and_whitespace_insensitive():
    ref = date(2026, 3, 4)
    assert parse_relative_date_to_iso("  Next FRIDAY  ", today=ref) == "2026-03-06"


def test_parse_relative_date_rejects_unparseable_text():
    with pytest.raises(TransformError):
        parse_relative_date_to_iso("sometime next quarter")


def test_parse_relative_date_rejects_non_string():
    with pytest.raises(TransformError):
        parse_relative_date_to_iso(12345)


# -- coerce_* transforms -------------------------------------------------------


def test_coerce_int_success_and_failure():
    assert coerce_int("42") == 42
    assert coerce_int(42.0) == 42
    with pytest.raises(TransformError):
        coerce_int("not a number")


def test_coerce_float_success_and_failure():
    assert coerce_float("3.14") == pytest.approx(3.14)
    with pytest.raises(TransformError):
        coerce_float(None)


def test_coerce_str_always_succeeds():
    assert coerce_str(42) == "42"
    assert coerce_str(None) == "None"


def test_repair_common_json_errors_strips_trailing_commas():
    repaired = repair_common_json_errors('{"a": 1, "b": 2,}')
    assert json.loads(repaired) == {"a": 1, "b": 2}


def test_repair_common_json_errors_converts_single_to_double_quotes():
    repaired = repair_common_json_errors("{'a': 1}")
    assert json.loads(repaired) == {"a": 1}


def test_repair_common_json_errors_handles_both_at_once():
    repaired = repair_common_json_errors("{'a': 1, 'b': [1, 2,],}")
    assert json.loads(repaired) == {"a": 1, "b": [1, 2]}


def test_repair_common_json_errors_rejects_non_string():
    with pytest.raises(TransformError):
        repair_common_json_errors({"already": "a dict"})


def test_transform_registry_contains_expected_names():
    assert set(TRANSFORM_REGISTRY) == {
        "parse_relative_date_to_iso",
        "coerce_int",
        "coerce_float",
        "coerce_str",
        "repair_common_json_errors",
    }


def test_coerce_str_is_excluded_from_guard_safe_transforms():
    # coerce_str = str(value) unconditionally succeeds for any input, so
    # unlike the other four transforms it can't be applied *proactively*
    # (before a call is even attempted) without risking mangling an
    # already-correct non-string value on a call that would otherwise have
    # succeeded fine. Safe only as a reactive fast-path replay, where it's
    # already been proven a string was needed. See core/recovery.py's
    # GUARD_SAFE_TRANSFORMS docstring for the full reasoning — this test
    # exists so a future always-succeeding transform doesn't get added to
    # the allowlist without someone re-deriving that reasoning first.
    assert "coerce_str" not in GUARD_SAFE_TRANSFORMS
    assert GUARD_SAFE_TRANSFORMS == {
        "parse_relative_date_to_iso",
        "coerce_int",
        "coerce_float",
        "repair_common_json_errors",
    }
    assert GUARD_SAFE_TRANSFORMS < set(TRANSFORM_REGISTRY)


# -- apply_fix -----------------------------------------------------------------


def test_apply_fix_with_literal_patch_only():
    fix = Fix(strategy="add_missing_field", argument_patch={"attendees": []})
    result = apply_fix({"title": "Standup"}, fix)

    assert result == {"title": "Standup", "attendees": []}


def test_apply_fix_with_transform_only():
    fix = Fix(
        strategy="reformat_argument",
        transforms=[ArgTransform(argument="date", transform="parse_relative_date_to_iso")],
    )
    result = apply_fix({"date": "tomorrow"}, fix)

    assert result["date"] == parse_relative_date_to_iso("tomorrow")


def test_apply_fix_transform_recomputes_per_occurrence_not_a_cached_literal():
    """The scenario that motivates transforms existing at all: the SAME Fix
    (as would be replayed from a single stored recipe) must produce the
    CORRECT, different result for two different occurrences of the same
    failure shape."""
    fix = Fix(
        strategy="reformat_argument",
        transforms=[ArgTransform(argument="date", transform="parse_relative_date_to_iso")],
    )

    result_friday = apply_fix({"date": "next Friday"}, fix)
    result_tuesday = apply_fix({"date": "next Tuesday"}, fix)

    assert result_friday["date"] == parse_relative_date_to_iso("next Friday")
    assert result_tuesday["date"] == parse_relative_date_to_iso("next Tuesday")
    assert result_friday["date"] != result_tuesday["date"]


def test_apply_fix_patch_then_transform_ordering():
    fix = Fix(
        strategy="both",
        argument_patch={"date": "in 2 days"},
        transforms=[ArgTransform(argument="date", transform="parse_relative_date_to_iso")],
    )
    # patch sets the raw value, then the transform re-parses it — order matters.
    result = apply_fix({"date": "irrelevant"}, fix)
    assert result["date"] == parse_relative_date_to_iso("in 2 days")


def test_apply_fix_skips_transform_for_absent_argument():
    fix = Fix(
        strategy="reformat_argument",
        transforms=[ArgTransform(argument="date", transform="parse_relative_date_to_iso")],
    )
    result = apply_fix({"title": "Standup"}, fix)

    assert result == {"title": "Standup"}  # no KeyError, no "date" added


def test_apply_fix_unknown_transform_raises_transform_error():
    fix = Fix(
        strategy="x",
        transforms=[ArgTransform(argument="date", transform="not_a_real_transform")],
    )
    with pytest.raises(TransformError):
        apply_fix({"date": "tomorrow"}, fix)


def test_apply_fix_does_not_mutate_original_args():
    original = {"title": "Standup"}
    fix = Fix(strategy="x", argument_patch={"attendees": []})
    apply_fix(original, fix)

    assert original == {"title": "Standup"}


# -- generate_fix (reflection call mocked) -------------------------------------


def test_generate_fix_validates_mocked_reflect_response():
    captured_contexts = []

    def fake_reflect(context: FailureContext) -> dict:
        captured_contexts.append(context)
        return {
            "strategy": "reformat_argument",
            "root_cause": "natural-language date string passed where ISO date expected",
            "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
        }

    context = FailureContext(
        tool_name="create_event",
        args={"date": "next Friday"},
        error_type="ValueError",
        error_message="could not parse date 'next Friday'",
        signature="tool:create_event|error_type:ValueError|args:{date:<STR>}",
    )

    fix = generate_fix(context, fake_reflect)

    assert isinstance(fix, Fix)
    assert fix.strategy == "reformat_argument"
    assert fix.transforms == [ArgTransform(argument="date", transform="parse_relative_date_to_iso")]
    assert captured_contexts == [context]


def test_generate_fix_raises_on_malformed_reflect_response():
    def bad_reflect(context: FailureContext) -> dict:
        return {"root_cause": "missing the required 'strategy' field"}

    context = FailureContext(tool_name="t", args={})
    with pytest.raises(ValidationError):
        generate_fix(context, bad_reflect)


def test_generate_fix_output_is_directly_usable_by_apply_fix():
    def fake_reflect(context: FailureContext) -> dict:
        return {"strategy": "add_missing_field", "argument_patch": {"attendees": []}}

    context = FailureContext(tool_name="create_event", args={"title": "Standup"})
    fix = generate_fix(context, fake_reflect)
    result = apply_fix(context.args, fix)

    assert result == {"title": "Standup", "attendees": []}


# -- FailureContext ------------------------------------------------------------


def test_failure_context_defaults():
    context = FailureContext(tool_name="t", args={})

    assert context.previous_attempts == []
    assert context.attempt_number == 1
    assert context.available_transforms == sorted(TRANSFORM_REGISTRY)


def test_failure_context_carries_previous_attempts_to_avoid_blind_repetition():
    first_attempt = Fix(strategy="coerce_type", argument_patch={"quantity": 5})
    context = FailureContext(
        tool_name="set_quantity",
        args={"quantity": "5"},
        attempt_number=2,
        previous_attempts=[first_attempt],
    )

    assert context.previous_attempts == [first_attempt]
    assert context.attempt_number == 2
