"""Unit tests for the failure oracle: SQLiteStore, ChromaVectorIndex, Oracle,
and the RecipeManager domain logic built on top of it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from resilientforge.oracle import Oracle
from resilientforge.oracle.recipes import RecipeManager
from resilientforge.oracle.store import FailureRecord, RecipeRow, ResolutionStatus, SQLiteStore
from resilientforge.oracle.vector_index import ChromaVectorIndex


# -- SQLiteStore --------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "oracle.db")
    yield s
    s.close()


def _failure(**overrides) -> FailureRecord:
    defaults = dict(
        tool_name="create_event",
        signature="tool:create_event error:invalid_date arg:<DATE>",
        workflow_id="wf-1",
        error_type="ValueError",
        error_message="could not parse date 'next Friday'",
        sanitized_args={"date": "<DATE>"},
        created_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return FailureRecord(**defaults)


def _recipe(**overrides) -> RecipeRow:
    defaults = dict(
        signature="tool:create_event error:invalid_date arg:<DATE>",
        tool_name="create_event",
        root_cause="natural-language date string passed where ISO date expected",
        fix_strategy="reformat_argument",
        fix_detail={"strategy": "parse_natural_language_date"},
        times_applied=1,
        times_succeeded=1,
        success_rate=1.0,
        created_at="2026-01-01T00:00:00+00:00",
        last_used="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return RecipeRow(**defaults)


def test_insert_and_get_failure(store):
    failure_id = store.insert_failure(_failure())
    record = store.get_failure(failure_id)

    assert record is not None
    assert record.id == failure_id
    assert record.tool_name == "create_event"
    assert record.resolution_status == ResolutionStatus.UNRESOLVED
    assert record.sanitized_args == {"date": "<DATE>"}
    assert record.fix_applied is None
    assert record.fix_verified is None


def test_get_failure_missing_returns_none(store):
    assert store.get_failure(999) is None


def test_update_failure_resolution(store):
    failure_id = store.insert_failure(_failure())

    store.update_failure_resolution(
        failure_id,
        ResolutionStatus.RECOVERED,
        fix_applied={"strategy": "reformat_argument"},
        fix_verified=True,
    )

    record = store.get_failure(failure_id)
    assert record.resolution_status == ResolutionStatus.RECOVERED
    assert record.fix_applied == {"strategy": "reformat_argument"}
    assert record.fix_verified is True


def test_list_failures_filters_by_signature_and_workflow(store):
    store.insert_failure(_failure(signature="sig-a", workflow_id="wf-1"))
    store.insert_failure(_failure(signature="sig-b", workflow_id="wf-1"))
    store.insert_failure(_failure(signature="sig-a", workflow_id="wf-2"))

    assert len(store.list_failures(signature="sig-a")) == 2
    assert len(store.list_failures(workflow_id="wf-1")) == 2
    assert len(store.list_failures(signature="sig-a", workflow_id="wf-2")) == 1
    assert len(store.list_failures()) == 3


def test_list_failures_orders_most_recent_first(store):
    first = store.insert_failure(_failure(signature="sig-first"))
    second = store.insert_failure(_failure(signature="sig-second"))

    results = store.list_failures()
    assert [r.id for r in results] == [second, first]


def test_upsert_recipe_insert_then_update(store):
    store.upsert_recipe(_recipe())
    recipe = store.get_recipe(_recipe().signature)
    assert recipe.times_applied == 1
    assert recipe.success_rate == 1.0

    store.upsert_recipe(_recipe(times_applied=4, times_succeeded=3, success_rate=0.75))
    updated = store.get_recipe(_recipe().signature)
    assert updated.times_applied == 4
    assert updated.success_rate == 0.75


def test_get_recipe_missing_returns_none(store):
    assert store.get_recipe("no-such-signature") is None


def test_list_recipes_filters_by_tool_name(store):
    store.upsert_recipe(_recipe(signature="sig-a", tool_name="create_event"))
    store.upsert_recipe(_recipe(signature="sig-b", tool_name="send_email"))

    assert len(store.list_recipes(tool_name="create_event")) == 1
    assert len(store.list_recipes()) == 2


def test_delete_recipe(store):
    store.upsert_recipe(_recipe())
    store.delete_recipe(_recipe().signature)
    assert store.get_recipe(_recipe().signature) is None


def test_data_persists_across_reconnect(tmp_path):
    db_path = tmp_path / "oracle.db"
    store_a = SQLiteStore(db_path)
    failure_id = store_a.insert_failure(_failure())
    store_a.upsert_recipe(_recipe())
    store_a.close()

    store_b = SQLiteStore(db_path)
    assert store_b.get_failure(failure_id) is not None
    assert store_b.get_recipe(_recipe().signature) is not None
    store_b.close()


# -- ChromaVectorIndex ----------------------------------------------------------


@pytest.fixture
def vector_index(tmp_path):
    idx = ChromaVectorIndex(tmp_path / "vectors")
    yield idx
    idx.close()


def test_query_on_empty_index_returns_empty_list(vector_index):
    assert vector_index.query("anything") == []


def test_query_returns_exact_match_first(vector_index):
    vector_index.add(
        "sig-date",
        "tool:create_event error:invalid_date arg:<DATE>",
        metadata={"tool_name": "create_event"},
    )
    vector_index.add(
        "sig-email",
        "tool:send_email error:missing_field arg:<STR>",
        metadata={"tool_name": "send_email"},
    )

    matches = vector_index.query("tool:create_event error:invalid_date arg:<DATE>", top_k=2)

    assert matches[0].id == "sig-date"
    assert matches[0].metadata["tool_name"] == "create_event"
    assert matches[0].score > matches[1].score
    assert matches[0].score == pytest.approx(1.0, abs=1e-4)


def test_add_upserts_rather_than_duplicates(vector_index):
    vector_index.add("sig-date", "tool:create_event error:invalid_date arg:<DATE>")
    vector_index.add(
        "sig-date",
        "tool:create_event error:invalid_date arg:<DATE>",
        metadata={"tool_name": "create_event"},
    )

    matches = vector_index.query("tool:create_event error:invalid_date arg:<DATE>", top_k=5)
    assert len(matches) == 1
    assert matches[0].metadata["tool_name"] == "create_event"


def test_delete_removes_entry(vector_index):
    vector_index.add("sig-date", "tool:create_event error:invalid_date arg:<DATE>")
    vector_index.delete("sig-date")
    assert vector_index.query("tool:create_event error:invalid_date arg:<DATE>") == []


def test_top_k_respects_available_count(vector_index):
    vector_index.add("sig-a", "alpha beta gamma")
    matches = vector_index.query("alpha beta gamma", top_k=5)
    assert len(matches) == 1


# -- Oracle facade --------------------------------------------------------------


@pytest.fixture
def oracle(tmp_path):
    o = Oracle(tmp_path / ".resilientforge")
    yield o
    o.close()


def test_record_failure_persists_and_indexes_signature(oracle):
    record = oracle.record_failure(
        tool_name="create_event",
        signature="tool:create_event error:invalid_date arg:<DATE>",
        workflow_id="wf-1",
        error_type="ValueError",
        error_message="could not parse date",
        sanitized_args={"date": "<DATE>"},
    )

    assert record.id is not None
    assert oracle.get_failure(record.id).tool_name == "create_event"

    matches = oracle.find_similar_failures("tool:create_event error:invalid_date arg:<DATE>")
    assert matches[0].id == "tool:create_event error:invalid_date arg:<DATE>"


def test_upsert_recipe_via_oracle_and_get_recipe(oracle):
    oracle.upsert_recipe(_recipe())

    recipe = oracle.get_recipe(_recipe().signature)
    assert recipe is not None
    assert recipe.fix_strategy == "reformat_argument"

    matches = oracle.find_similar_failures(_recipe().signature)
    assert matches[0].id == _recipe().signature


def test_delete_recipe_removes_from_both_backends(oracle):
    oracle.upsert_recipe(_recipe())
    oracle.delete_recipe(_recipe().signature)

    assert oracle.get_recipe(_recipe().signature) is None
    assert oracle.find_similar_failures(_recipe().signature) == []


def test_oracle_context_manager_closes_cleanly(tmp_path):
    with Oracle(tmp_path / ".resilientforge") as o:
        o.record_failure(tool_name="t", signature="sig", sanitized_args={})
    # No assertion beyond "no exception" — verifies close() doesn't raise
    # and __exit__ is wired correctly.


# -- RecipeManager (oracle/recipes.py) ------------------------------------------


@pytest.fixture
def recipes(oracle):
    return RecipeManager(oracle)


def test_record_success_creates_recipe_on_first_occurrence(recipes):
    recipe = recipes.record_success(
        signature="sig-a",
        tool_name="create_event",
        fix_detail={"strategy": "parse_natural_language_date"},
        root_cause="natural-language date string passed where ISO date expected",
        fix_strategy="reformat_argument",
    )

    assert recipe.times_applied == 1
    assert recipe.times_succeeded == 1
    assert recipe.success_rate == 1.0
    assert recipes.get("sig-a") == recipe


def test_record_success_updates_existing_recipe_stats(recipes):
    recipes.record_success(signature="sig-a", tool_name="t", fix_detail={"v": 1})
    updated = recipes.record_success(signature="sig-a", tool_name="t", fix_detail={"v": 2})

    assert updated.times_applied == 2
    assert updated.times_succeeded == 2
    assert updated.success_rate == 1.0
    assert updated.fix_detail == {"v": 2}  # latest fix_detail wins


def test_record_fast_path_failure_lowers_success_rate_without_new_success(recipes):
    recipes.record_success(signature="sig-a", tool_name="t", fix_detail={})
    result = recipes.record_fast_path_failure("sig-a")

    assert result.times_applied == 2
    assert result.times_succeeded == 1
    assert result.success_rate == 0.5


def test_record_fast_path_failure_on_unknown_signature_returns_none(recipes):
    assert recipes.record_fast_path_failure("no-such-signature") is None


def test_record_success_preserves_root_cause_and_fix_strategy_if_not_reprovided(recipes):
    recipes.record_success(
        signature="sig-a",
        tool_name="t",
        fix_detail={},
        root_cause="original cause",
        fix_strategy="original strategy",
    )
    updated = recipes.record_success(signature="sig-a", tool_name="t", fix_detail={})

    assert updated.root_cause == "original cause"
    assert updated.fix_strategy == "original strategy"


def test_list_recipes_via_manager(recipes):
    recipes.record_success(signature="sig-a", tool_name="create_event", fix_detail={})
    recipes.record_success(signature="sig-b", tool_name="send_email", fix_detail={})

    assert len(recipes.list()) == 2
    assert len(recipes.list(tool_name="create_event")) == 1


def test_prune_removes_unreliable_recipes_below_success_rate_floor(recipes):
    recipes.record_success(signature="sig-good", tool_name="t", fix_detail={})
    recipes.record_success(signature="sig-bad", tool_name="t", fix_detail={})
    recipes.record_fast_path_failure("sig-bad")
    recipes.record_fast_path_failure("sig-bad")  # sig-bad: 1/3 = 0.33

    pruned = recipes.prune(min_success_rate=0.5)

    assert pruned == ["sig-bad"]
    assert recipes.get("sig-bad") is None
    assert recipes.get("sig-good") is not None


def test_prune_respects_min_times_applied_before_judging_reliability(recipes):
    # A single success has success_rate 1.0, so it's never "unreliable" —
    # but confirm a low-min_times_applied still doesn't prune a perfectly
    # successful recipe just because it's only been used once.
    recipes.record_success(signature="sig-a", tool_name="t", fix_detail={})

    pruned = recipes.prune(min_success_rate=0.9, min_times_applied=1)

    assert pruned == []


def test_prune_removes_stale_recipes_by_age(recipes):
    old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    recipes.oracle.upsert_recipe(
        RecipeRow(
            signature="sig-old",
            tool_name="t",
            fix_detail={},
            created_at=old_time,
            last_used=old_time,
            times_applied=5,
            times_succeeded=5,
            success_rate=1.0,
        )
    )
    recipes.record_success(signature="sig-fresh", tool_name="t", fix_detail={})

    pruned = recipes.prune(max_age_days=30)

    assert pruned == ["sig-old"]
    assert recipes.get("sig-fresh") is not None


def test_prune_returns_empty_list_when_nothing_qualifies(recipes):
    recipes.record_success(signature="sig-a", tool_name="t", fix_detail={})
    assert recipes.prune(min_success_rate=0.0, max_age_days=None) == []


def test_prune_dry_run_reports_without_deleting(recipes):
    recipes.record_success(signature="sig-bad", tool_name="t", fix_detail={})
    recipes.record_fast_path_failure("sig-bad")
    recipes.record_fast_path_failure("sig-bad")  # sig-bad: 1/3 = 0.33

    pruned = recipes.prune(min_success_rate=0.5, dry_run=True)

    assert pruned == ["sig-bad"]
    assert recipes.get("sig-bad") is not None  # still there — dry run didn't delete
