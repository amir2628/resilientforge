"""Unit tests for dashboard/app.py: every read-only endpoint, using
FastAPI's standard TestClient idiom (no real port binding needed).

Seeding follows the exact same pattern as test_cli.py's `_seed`/
`_seed_guard` helpers — the dashboard reads through the identical
RecipeManager/GuardManager/Oracle paths the CLI already does.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from resilientforge.dashboard import create_app
from resilientforge.oracle import Oracle
from resilientforge.oracle.guards import GuardManager
from resilientforge.oracle.recipes import RecipeManager


def _seed(oracle_path, tool_name="create_event"):
    signature = f"tool:{tool_name}|error_type:ValueError|args:{{date:<STR>}}"
    with Oracle(oracle_path) as oracle:
        oracle.record_failure(tool_name=tool_name, signature=signature, sanitized_args={"date": "next Friday"})
        RecipeManager(oracle).record_success(
            signature=signature,
            tool_name=tool_name,
            fix_detail={"strategy": "reformat_argument"},
            root_cause="natural-language date string",
            fix_strategy="reformat_argument",
        )
        GuardManager(oracle).promote(
            tool_name=tool_name,
            argument="date",
            kind="transform",
            transform="parse_relative_date_to_iso",
            source_signature=signature,
        )
    return signature


def test_index_serves_html(tmp_path):
    _seed(tmp_path / "oracle")
    with TestClient(create_app(tmp_path / "oracle")) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "ResilientForge Dashboard" in response.text
    assert "text/html" in response.headers["content-type"]


def test_stats_endpoint(tmp_path):
    _seed(tmp_path / "oracle")
    with TestClient(create_app(tmp_path / "oracle")) as client:
        body = client.get("/api/stats").json()
    assert body["recipe_count"] == 1
    assert body["failure_count"] == 1
    assert body["guard_count"] == 1
    assert body["active_guard_count"] == 1
    assert body["failures_by_status"] == {"unresolved": 1}


def test_recipes_endpoint(tmp_path):
    signature = _seed(tmp_path / "oracle")
    with TestClient(create_app(tmp_path / "oracle")) as client:
        body = client.get("/api/recipes").json()
    assert len(body) == 1
    assert body[0]["signature"] == signature
    assert body[0]["tool_name"] == "create_event"
    assert body[0]["success_rate"] == 1.0


def test_recipes_endpoint_filters_by_tool_name(tmp_path):
    _seed(tmp_path / "oracle", tool_name="create_event")
    _seed(tmp_path / "oracle", tool_name="send_email")
    with TestClient(create_app(tmp_path / "oracle")) as client:
        body = client.get("/api/recipes", params={"tool_name": "send_email"}).json()
    assert len(body) == 1
    assert body[0]["tool_name"] == "send_email"


def test_guards_endpoint(tmp_path):
    _seed(tmp_path / "oracle")
    with TestClient(create_app(tmp_path / "oracle")) as client:
        body = client.get("/api/guards").json()
    assert len(body) == 1
    assert body[0]["tool_name"] == "create_event"
    assert body[0]["argument"] == "date"
    assert body[0]["kind"] == "transform"
    assert body[0]["active"] is True


def test_guards_endpoint_active_only_excludes_revoked(tmp_path):
    _seed(tmp_path / "oracle")
    with Oracle(tmp_path / "oracle") as oracle:
        GuardManager(oracle).revoke("create_event", "date")

    with TestClient(create_app(tmp_path / "oracle")) as client:
        active = client.get("/api/guards").json()
        everything = client.get("/api/guards", params={"active_only": False}).json()

    assert active == []
    assert len(everything) == 1
    assert everything[0]["active"] is False


def test_failures_endpoint(tmp_path):
    _seed(tmp_path / "oracle")
    with TestClient(create_app(tmp_path / "oracle")) as client:
        body = client.get("/api/failures").json()
    assert len(body) == 1
    assert body[0]["tool_name"] == "create_event"
    assert body[0]["resolution_status"] == "unresolved"
    assert body[0]["sanitized_args"] == {"date": "next Friday"}


def test_empty_oracle_returns_empty_lists_not_errors(tmp_path):
    with TestClient(create_app(tmp_path / "oracle")) as client:
        assert client.get("/api/recipes").json() == []
        assert client.get("/api/guards").json() == []
        assert client.get("/api/failures").json() == []
        stats = client.get("/api/stats").json()
    assert stats["recipe_count"] == 0
    assert stats["failure_count"] == 0
