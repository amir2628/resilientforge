"""FastAPI app powering `resilientforge dashboard` (Phase 4).

Read-only, GET-only, by design: this dashboard never mutates the oracle
— revoking a guard, pruning a recipe, etc. stay CLI-only operations for
now (see docs/architecture.md's "Local dashboard" section for why that's
a deliberate v1 scope decision, not an oversight). Every endpoint reuses
the exact same read paths `cli/main.py` already uses
(`RecipeManager(oracle).list(...)`, `GuardManager(oracle).list(...)`,
`oracle.list_failures(...)`) — never touching `oracle.store`/
`oracle.vector_index` directly.

This module (and everything under `dashboard/`) is never imported by
`resilientforge/__init__.py` — `fastapi`/`uvicorn` are an optional extra
(`pip install resilientforge[dashboard]`), not a hard dependency; only
`cli/main.py`'s `dashboard` command imports this module, and only lazily,
inside that command's own function body.
"""

from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from resilientforge.dashboard._html import DASHBOARD_HTML
from resilientforge.oracle import FailureRecord, Oracle
from resilientforge.oracle.guards import GuardManager, StandingGuard
from resilientforge.oracle.recipes import Recipe, RecipeManager


def create_app(oracle_path: str | Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.oracle = Oracle(oracle_path)
        try:
            yield
        finally:
            app.state.oracle.close()

    app = FastAPI(
        title="ResilientForge Dashboard",
        description="Read-only view of one oracle's recipes, guards, and failure history.",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/stats")
    def stats() -> dict:
        oracle: Oracle = app.state.oracle
        recipes = oracle.list_recipes(limit=10_000)
        failures = oracle.list_failures(limit=10_000)
        guards = oracle.list_guards(active_only=False, limit=10_000)
        by_status = Counter(f.resolution_status.value for f in failures)
        return {
            "oracle_path": str(oracle_path),
            "recipe_count": len(recipes),
            "failure_count": len(failures),
            "guard_count": len(guards),
            "active_guard_count": sum(1 for g in guards if g.active),
            "failures_by_status": dict(by_status),
        }

    @app.get("/api/recipes", response_model=list[Recipe])
    def recipes(tool_name: str | None = None, limit: int = Query(100, le=10_000)) -> list[Recipe]:
        oracle: Oracle = app.state.oracle
        return RecipeManager(oracle).list(tool_name=tool_name, limit=limit)

    @app.get("/api/guards", response_model=list[StandingGuard])
    def guards(
        tool_name: str | None = None,
        active_only: bool = True,
        limit: int = Query(100, le=10_000),
    ) -> list[StandingGuard]:
        oracle: Oracle = app.state.oracle
        return GuardManager(oracle).list(tool_name=tool_name, active_only=active_only, limit=limit)

    @app.get("/api/failures", response_model=list[FailureRecord])
    def failures(limit: int = Query(100, le=10_000)) -> list[FailureRecord]:
        oracle: Oracle = app.state.oracle
        return oracle.list_failures(limit=limit)

    return app
