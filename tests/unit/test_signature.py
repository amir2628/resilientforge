"""Unit tests for failure signature normalization/templating.

This is the crux of the whole project (PROJECT_SPEC.md §10): these tests
verify that structurally identical failures produce identical signatures
regardless of literal values, and that structurally *different* failures
(different field names, different types, different tools) do NOT collapse
together.
"""

from __future__ import annotations

from resilientforge.core.signature import (
    build_signature,
    normalize_args,
    normalize_error_message,
    normalize_value,
)


# -- the central case: same shape, different literal values -----------------


def test_natural_language_date_variants_produce_identical_signature():
    """The example straight from PROJECT_SPEC.md §4.3: 'next Friday' and
    'next Tuesday' date-format failures must match as the same shape."""
    sig_friday = build_signature(
        tool_name="create_event",
        error_type="ValueError",
        error_message="could not parse date 'next Friday'",
        args={"date": "next Friday", "title": "Standup"},
    )
    sig_tuesday = build_signature(
        tool_name="create_event",
        error_type="ValueError",
        error_message="could not parse date 'next Tuesday'",
        args={"date": "next Tuesday", "title": "Retro"},
    )

    assert sig_friday == sig_tuesday


def test_single_word_date_value_is_not_collapsed_like_multi_word_ones():
    """A discovered, honest limitation (surfaced by the failure-injection
    suite's natural_language_date scenario, not by inspection): a single
    bare word like "tomorrow" trips the same bare-identifier-preservation
    heuristic that keeps 'missing field X' distinct from 'missing field Y'
    (see test_different_missing_field_name_produces_different_signature) —
    there's no text-only way to tell "a single-word field name" from "a
    single-word literal value" apart. The heuristic's documented default
    is to err toward NOT collapsing when ambiguous (safer: a missed oracle
    hit costs an extra model call, a wrongly-collapsed signature risks
    replaying an unrelated fix — PROJECT_SPEC.md §10). This test exists so
    that trade-off is visible and intentional, not silently rediscovered
    as "flaky" the next time trial data happens to include a single-word
    value."""
    sig_friday = build_signature(
        tool_name="create_event",
        error_type="ValueError",
        error_message="could not parse date 'next Friday'",
        args={"date": "next Friday"},
    )
    sig_tomorrow = build_signature(
        tool_name="create_event",
        error_type="ValueError",
        error_message="could not parse date 'tomorrow'",
        args={"date": "tomorrow"},
    )

    assert sig_friday != sig_tomorrow


def test_json_decode_error_position_varies_but_signature_matches():
    sig_a = build_signature(
        tool_name="create_event",
        error_type="JSONDecodeError",
        error_message="Expecting value: line 1 column 5 (char 4)",
        args={"raw": "{bad json"},
    )
    sig_b = build_signature(
        tool_name="create_event",
        error_type="JSONDecodeError",
        error_message="Expecting value: line 3 column 12 (char 40)",
        args={"raw": "{also bad"},
    )

    assert sig_a == sig_b


def test_transient_timeout_duration_varies_but_signature_matches():
    sig_a = build_signature(
        tool_name="search",
        error_type="TimeoutError",
        error_message="Request timed out after 30s",
        args={"query": "quarterly earnings report"},
    )
    sig_b = build_signature(
        tool_name="search",
        error_type="TimeoutError",
        error_message="Request timed out after 45s",
        args={"query": "annual shareholder letter"},
    )

    assert sig_a == sig_b


def test_wrong_type_argument_same_field_different_literal_int_matches():
    sig_a = build_signature(tool_name="set_quantity", args={"quantity": 5})
    sig_b = build_signature(tool_name="set_quantity", args={"quantity": 9000})

    assert sig_a == sig_b


def test_uuid_and_email_literals_collapse_to_same_signature():
    sig_a = build_signature(
        tool_name="invite_user",
        args={"user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "email": "alice@example.com"},
    )
    sig_b = build_signature(
        tool_name="invite_user",
        args={"user_id": "9c858901-8a57-4791-81fe-4c455b099bc9", "email": "bob@example.org"},
    )

    assert sig_a == sig_b


def test_dict_key_order_does_not_affect_signature():
    sig_a = build_signature(tool_name="t", args={"a": 1, "b": "x"})
    sig_b = build_signature(tool_name="t", args={"b": "x", "a": 1})

    assert sig_a == sig_b


def test_list_argument_order_and_length_do_not_affect_signature():
    sig_a = build_signature(tool_name="t", args={"tags": ["urgent", "billing", "vip"]})
    sig_b = build_signature(tool_name="t", args={"tags": ["support"]})

    assert sig_a == sig_b


# -- structurally different failures must NOT collapse together -------------


def test_different_missing_field_name_produces_different_signature():
    """Regression case: when a required field is absent, its name only
    ever appears in the error message text (it can't appear in `args`,
    since it's missing) — a bare quoted identifier like 'attendees' must
    NOT be redacted the same way a literal value would be, or two
    genuinely different failures (missing attendees vs. missing location)
    would collapse into one signature and risk replaying the wrong fix."""
    sig_attendees = build_signature(
        tool_name="create_event",
        error_type="ValidationError",
        error_message="missing required field 'attendees'",
        args={"title": "Standup"},
    )
    sig_location = build_signature(
        tool_name="create_event",
        error_type="ValidationError",
        error_message="missing required field 'location'",
        args={"title": "Standup"},
    )

    assert sig_attendees != sig_location


def test_different_present_field_name_produces_different_signature():
    sig_attendees = build_signature(tool_name="create_event", args={"attendees": ["a@x.com"]})
    sig_location = build_signature(tool_name="create_event", args={"location": "Room 1"})

    assert sig_attendees != sig_location


def test_different_argument_type_for_same_field_produces_different_signature():
    sig_int = build_signature(tool_name="set_quantity", args={"quantity": 5})
    sig_str = build_signature(tool_name="set_quantity", args={"quantity": "5"})

    assert sig_int != sig_str


def test_different_tool_name_produces_different_signature():
    sig_a = build_signature(tool_name="create_event", args={"x": 1})
    sig_b = build_signature(tool_name="delete_event", args={"x": 1})

    assert sig_a != sig_b


def test_different_error_type_produces_different_signature():
    sig_a = build_signature(tool_name="t", error_type="ValueError", args={})
    sig_b = build_signature(tool_name="t", error_type="TypeError", args={})

    assert sig_a != sig_b


def test_nested_dict_args_normalize_recursively_and_by_structure():
    sig_a = build_signature(
        tool_name="update_profile",
        args={"profile": {"name": "Alice", "age": 30}},
    )
    sig_b = build_signature(
        tool_name="update_profile",
        args={"profile": {"name": "Bob", "age": 41}},
    )
    sig_c = build_signature(
        tool_name="update_profile",
        args={"profile": {"name": "Alice", "nickname": "Al"}},
    )

    assert sig_a == sig_b
    assert sig_a != sig_c  # different key ("age" vs "nickname") is a structural difference


# -- normalize_error_message -------------------------------------------------


def test_normalize_error_message_handles_none_and_empty():
    assert normalize_error_message(None) == ""
    assert normalize_error_message("") == ""


def test_normalize_error_message_redacts_iso_date_and_datetime():
    assert "<DATE>" in normalize_error_message("event starts 2026-03-05")
    assert "<DATETIME>" in normalize_error_message("event starts 2026-03-05T09:00:00Z")


def test_normalize_error_message_redacts_url_and_email():
    assert "<URL>" in normalize_error_message("failed to fetch https://api.example.com/v1/x")
    assert "<EMAIL>" in normalize_error_message("no such user someone@example.com")


def test_normalize_error_message_redacts_standalone_numbers():
    assert normalize_error_message("retry attempt 3 of 5") == "retry attempt <NUM> of <NUM>"


def test_normalize_error_message_redacts_unit_suffixed_numbers():
    # Regression: "30s" has no word boundary between the digits and the
    # trailing unit letter, so a naive \b\d+\b pattern misses it entirely.
    assert normalize_error_message("timed out after 30s") == "timed out after <NUM>s"
    assert normalize_error_message("timed out after 45s") == "timed out after <NUM>s"


def test_normalize_error_message_preserves_digits_embedded_in_identifiers():
    assert normalize_error_message("call to api_v2 failed") == "call to api_v2 failed"


# -- normalize_value / normalize_args ----------------------------------------


def test_normalize_value_type_placeholders():
    assert normalize_value(None) == "<NULL>"
    assert normalize_value(True) == "<BOOL>"
    assert normalize_value(False) == "<BOOL>"
    assert normalize_value(42) == "<INT>"
    assert normalize_value(3.14) == "<FLOAT>"
    assert normalize_value("plain text") == "<STR>"


def test_normalize_value_bool_is_not_classified_as_int():
    # bool is a subclass of int in Python — a common source of bugs.
    assert normalize_value(True) != normalize_value(1)


def test_normalize_value_classifies_uuid_date_datetime_email_url():
    assert normalize_value("3fa85f64-5717-4562-b3fc-2c963f66afa6") == "<UUID>"
    assert normalize_value("2026-03-05") == "<DATE>"
    assert normalize_value("2026-03-05T09:00:00Z") == "<DATETIME>"
    assert normalize_value("alice@example.com") == "<EMAIL>"
    assert normalize_value("https://example.com/path") == "<URL>"


def test_normalize_value_empty_list_and_dict():
    assert normalize_value([]) == "[]"
    assert normalize_value({}) == "{}"


def test_normalize_args_none_is_empty_dict_shape():
    assert normalize_args(None) == "{}"
    assert normalize_args({}) == "{}"


# -- build_signature composition ---------------------------------------------


def test_build_signature_is_deterministic():
    kwargs = dict(
        tool_name="create_event",
        error_type="ValueError",
        error_message="could not parse date 'next Friday'",
        args={"date": "next Friday"},
    )
    assert build_signature(**kwargs) == build_signature(**kwargs)


def test_build_signature_omits_absent_error_type_and_message():
    sig = build_signature(tool_name="t", args={"x": 1})
    assert "error_type:" not in sig
    assert "error:" not in sig
    assert sig.startswith("tool:t|")
