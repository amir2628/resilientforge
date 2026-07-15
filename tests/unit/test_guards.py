"""Unit tests for standing guards: GuardRow/StandingGuard, the guards table
CRUD, and GuardManager's promotion/revocation/describe lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from resilientforge.oracle import Oracle
from resilientforge.oracle.guards import GuardManager, StandingGuard
from resilientforge.oracle.store import GuardRow


@pytest.fixture
def oracle(tmp_path):
    o = Oracle(tmp_path / ".resilientforge")
    yield o
    o.close()


@pytest.fixture
def guards(oracle):
    return GuardManager(oracle)


# -- promote -----------------------------------------------------------------


def test_promote_creates_transform_guard(guards):
    guard = guards.promote(
        tool_name="create_event",
        argument="date",
        kind="transform",
        transform="parse_relative_date_to_iso",
        source_signature="sig-a",
        root_cause="natural-language date string",
    )

    assert guard.tool_name == "create_event"
    assert guard.argument == "date"
    assert guard.kind == "transform"
    assert guard.transform == "parse_relative_date_to_iso"
    assert guard.active is True
    assert guard.times_applied == 0


def test_promote_creates_patch_guard(guards):
    guard = guards.promote(
        tool_name="create_event",
        argument="attendees",
        kind="patch",
        patch_value=[],
        source_signature="sig-b",
    )

    assert guard.kind == "patch"
    assert guard.patch_value == []


def test_promote_updates_existing_active_guard(guards):
    guards.promote(
        tool_name="t", argument="x", kind="transform", transform="coerce_int",
        source_signature="sig-1", root_cause="original cause",
    )
    updated = guards.promote(
        tool_name="t", argument="x", kind="transform", transform="coerce_float",
        source_signature="sig-2",
    )

    assert updated.transform == "coerce_float"
    assert updated.source_signature == "sig-2"
    assert updated.root_cause == "original cause"  # preserved when not re-provided


def test_promote_refuses_to_reactivate_revoked_guard(guards):
    guards.promote(
        tool_name="t", argument="x", kind="transform", transform="coerce_int",
        source_signature="sig-1",
    )
    guards.revoke("t", "x", kind="transform")

    result = guards.promote(
        tool_name="t", argument="x", kind="transform", transform="coerce_int",
        source_signature="sig-2",
    )

    assert result is None
    assert guards.get("t", "x", "transform").active is False


# -- get / list ----------------------------------------------------------------


def test_get_missing_returns_none(guards):
    assert guards.get("nope", "nope", "transform") is None


def test_list_active_filters_by_tool_name(guards):
    guards.promote(tool_name="create_event", argument="date", kind="transform",
                    transform="coerce_int", source_signature="sig-a")
    guards.promote(tool_name="send_email", argument="body", kind="transform",
                    transform="coerce_str", source_signature="sig-b")

    assert len(guards.list_active(tool_name="create_event")) == 1
    assert len(guards.list_active()) == 2


def test_list_active_excludes_revoked(guards):
    guards.promote(tool_name="t", argument="x", kind="patch", patch_value=1,
                    source_signature="sig-a")
    guards.revoke("t", "x")

    assert guards.list_active(tool_name="t") == []


def test_list_all_includes_revoked_when_active_only_false(guards):
    guards.promote(tool_name="t", argument="x", kind="patch", patch_value=1,
                    source_signature="sig-a")
    guards.revoke("t", "x")

    all_guards = guards.list(tool_name="t", active_only=False)
    assert len(all_guards) == 1
    assert all_guards[0].active is False


# -- revoke --------------------------------------------------------------------


def test_revoke_specific_kind_leaves_other_kind_active(guards):
    guards.promote(tool_name="t", argument="x", kind="transform", transform="coerce_int",
                    source_signature="sig-a")
    guards.promote(tool_name="t", argument="x", kind="patch", patch_value=1,
                    source_signature="sig-b")

    revoked = guards.revoke("t", "x", kind="transform")

    assert len(revoked) == 1
    assert revoked[0].kind == "transform"
    remaining = guards.list_active(tool_name="t")
    assert len(remaining) == 1
    assert remaining[0].kind == "patch"


def test_revoke_both_kinds_when_kind_none(guards):
    guards.promote(tool_name="t", argument="x", kind="transform", transform="coerce_int",
                    source_signature="sig-a")
    guards.promote(tool_name="t", argument="x", kind="patch", patch_value=1,
                    source_signature="sig-b")

    revoked = guards.revoke("t", "x")

    assert len(revoked) == 2
    assert guards.list_active(tool_name="t") == []


def test_revoke_no_matching_guards_returns_empty_list(guards):
    assert guards.revoke("nope", "nope") == []


# -- record_application ---------------------------------------------------------


def test_record_application_counter_math(guards):
    guard = guards.promote(tool_name="t", argument="x", kind="transform", transform="coerce_int",
                            source_signature="sig-a")

    guards.record_application([guard], succeeded=True)
    reloaded = guards.get("t", "x", "transform")
    assert reloaded.times_applied == 1
    assert reloaded.times_succeeded == 1
    assert reloaded.success_rate == 1.0
    assert reloaded.last_applied is not None

    guards.record_application([reloaded], succeeded=False)
    reloaded_again = guards.get("t", "x", "transform")
    assert reloaded_again.times_applied == 2
    assert reloaded_again.times_succeeded == 1
    assert reloaded_again.success_rate == 0.5


# -- prune (Phase 5, mirrors RecipeManager.prune) --------------------------------


def test_prune_removes_unreliable_guards_below_success_rate_floor(guards):
    good = guards.promote(tool_name="t", argument="good", kind="transform",
                           transform="coerce_int", source_signature="sig-a")
    bad = guards.promote(tool_name="t", argument="bad", kind="transform",
                          transform="coerce_int", source_signature="sig-b")
    guards.record_application([good], succeeded=True)
    guards.record_application([bad], succeeded=True)
    guards.record_application([bad], succeeded=False)
    guards.record_application([bad], succeeded=False)  # bad: 1/3 = 0.33

    pruned = guards.prune(min_success_rate=0.5)

    assert pruned == [("t", "bad", "transform")]
    assert guards.get("t", "bad", "transform") is None
    assert guards.get("t", "good", "transform") is not None


def test_prune_respects_min_times_applied_before_judging_reliability(guards):
    guard = guards.promote(tool_name="t", argument="x", kind="transform",
                            transform="coerce_int", source_signature="sig-a")
    guards.record_application([guard], succeeded=True)

    pruned = guards.prune(min_success_rate=0.9, min_times_applied=5)

    assert pruned == []


def test_prune_removes_stale_guards_by_age(guards, oracle):
    old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    oracle.upsert_guard(
        GuardRow(
            tool_name="t", argument="old", kind="transform", transform="coerce_int",
            source_signature="sig-a", created_at=old_time, last_applied=old_time,
            times_applied=5, times_succeeded=5, success_rate=1.0,
        )
    )
    fresh = guards.promote(tool_name="t", argument="fresh", kind="transform",
                            transform="coerce_int", source_signature="sig-b")
    guards.record_application([fresh], succeeded=True)

    pruned = guards.prune(max_age_days=30)

    assert pruned == [("t", "old", "transform")]
    assert guards.get("t", "fresh", "transform") is not None


def test_prune_never_treats_a_never_applied_guard_as_stale(guards):
    # last_applied is None until a guard actually fires — pruning by age
    # must not treat "never used yet" as "very stale".
    guards.promote(tool_name="t", argument="x", kind="transform",
                    transform="coerce_int", source_signature="sig-a")

    assert guards.prune(max_age_days=1) == []


def test_prune_dry_run_reports_without_deleting(guards):
    guard = guards.promote(tool_name="t", argument="x", kind="transform",
                            transform="coerce_int", source_signature="sig-a")
    guards.record_application([guard], succeeded=False)

    pruned = guards.prune(min_success_rate=0.5, dry_run=True)

    assert pruned == [("t", "x", "transform")]
    assert guards.get("t", "x", "transform") is not None  # still there — dry run didn't delete


# -- describe --------------------------------------------------------------------


def test_describe_empty(guards):
    assert guards.describe() == "No active guards."


def test_describe_transform_and_patch_guards(guards):
    guards.promote(
        tool_name="create_event", argument="date", kind="transform",
        transform="parse_relative_date_to_iso", source_signature="sig-a",
        root_cause="natural-language date string passed where ISO date expected",
    )
    guards.promote(
        tool_name="create_event", argument="attendees", kind="patch",
        patch_value=[], source_signature="sig-b",
    )

    text = guards.describe()

    assert "create_event(date)" in text
    assert "natural-language date string" in text
    assert "parse_relative_date_to_iso" in text
    assert "create_event(attendees)" in text
    assert "defaults to [] when omitted" in text


def test_describe_filters_by_tool_name(guards):
    guards.promote(tool_name="create_event", argument="date", kind="transform",
                    transform="coerce_int", source_signature="sig-a")
    guards.promote(tool_name="send_email", argument="body", kind="transform",
                    transform="coerce_str", source_signature="sig-b")

    text = guards.describe(tool_name="send_email")

    assert "send_email" in text
    assert "create_event" not in text


def test_describe_excludes_revoked_guards(guards):
    guards.promote(tool_name="t", argument="x", kind="patch", patch_value=1,
                    source_signature="sig-a")
    guards.revoke("t", "x")

    assert guards.describe(tool_name="t") == "No active guards."


# -- GuardRow / StandingGuard round-trip ----------------------------------------


def test_guard_row_round_trip_via_store(oracle):
    row = GuardRow(
        tool_name="t",
        argument="x",
        kind="transform",
        transform="coerce_int",
        source_signature="sig-a",
        created_at="2026-01-01T00:00:00+00:00",
    )
    oracle.upsert_guard(row)

    fetched = oracle.get_guard("t", "x", "transform")
    assert fetched is not None
    assert fetched.transform == "coerce_int"

    guard = StandingGuard._from_row(fetched)
    assert guard.tool_name == "t"
    round_tripped = guard._to_row()
    assert round_tripped.tool_name == "t"
    assert round_tripped.transform == "coerce_int"
