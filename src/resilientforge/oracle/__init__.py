"""The failure oracle: a single interface over structured (SQLite) and
semantic (vector) storage of past failures and their fixes.

Callers (recovery.py, engine.py) should only need `Oracle` — they should
not need to know that persistence is split across two backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resilientforge.oracle.store import (
    FailureRecord,
    GuardRow,
    RecipeRow,
    ResolutionStatus,
    SQLiteStore,
)
from resilientforge.oracle.vector_index import ChromaVectorIndex, VectorIndex, VectorMatch

__all__ = [
    "Oracle",
    "FailureRecord",
    "RecipeRow",
    "GuardRow",
    "ResolutionStatus",
    "VectorIndex",
    "ChromaVectorIndex",
    "VectorMatch",
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Oracle:
    def __init__(
        self,
        path: str | Path = ".resilientforge",
        store: SQLiteStore | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.store = store or SQLiteStore(path / "oracle.db")
        self.vector_index = vector_index or ChromaVectorIndex(path / "vectors")

    # -- failures --------------------------------------------------------

    def record_failure(
        self,
        *,
        tool_name: str,
        signature: str,
        workflow_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        sanitized_args: dict[str, Any] | None = None,
    ) -> FailureRecord:
        record = FailureRecord(
            tool_name=tool_name,
            signature=signature,
            workflow_id=workflow_id,
            error_type=error_type,
            error_message=error_message,
            sanitized_args=sanitized_args or {},
            created_at=_utcnow(),
        )
        record.id = self.store.insert_failure(record)
        # Index the raw signature immediately so it's queryable even before
        # any recipe exists for it (a second, still-unresolved occurrence of
        # the same shape should still surface as "seen before").
        self.vector_index.add(id=signature, text=signature, metadata={"tool_name": tool_name})
        return record

    def update_failure_resolution(
        self,
        failure_id: int,
        status: ResolutionStatus,
        fix_applied: dict[str, Any] | None = None,
        fix_verified: bool | None = None,
    ) -> None:
        self.store.update_failure_resolution(failure_id, status, fix_applied, fix_verified)

    def get_failure(self, failure_id: int) -> FailureRecord | None:
        return self.store.get_failure(failure_id)

    def list_failures(
        self,
        signature: str | None = None,
        workflow_id: str | None = None,
        limit: int = 100,
    ) -> list[FailureRecord]:
        return self.store.list_failures(signature=signature, workflow_id=workflow_id, limit=limit)

    # -- semantic lookup ---------------------------------------------------

    def find_similar_failures(self, signature: str, top_k: int = 5) -> list[VectorMatch]:
        return self.vector_index.query(signature, top_k=top_k)

    # -- recipes (raw persistence; domain logic lives in oracle/recipes.py) --

    def upsert_recipe(self, recipe: RecipeRow) -> None:
        self.store.upsert_recipe(recipe)
        self.vector_index.add(
            id=recipe.signature,
            text=recipe.signature,
            metadata={"tool_name": recipe.tool_name},
        )

    def get_recipe(self, signature: str) -> RecipeRow | None:
        return self.store.get_recipe(signature)

    def list_recipes(self, tool_name: str | None = None, limit: int = 100) -> list[RecipeRow]:
        return self.store.list_recipes(tool_name=tool_name, limit=limit)

    def delete_recipe(self, signature: str) -> None:
        self.store.delete_recipe(signature)
        self.vector_index.delete(signature)

    # -- guards (raw persistence; domain logic lives in oracle/guards.py) ----
    #
    # Deliberately NOT indexed into the vector store the way recipes are:
    # guard matching is exact-key (tool_name + argument presence), never
    # fuzzy. Indexing a guard's pseudo-signature into the same collection
    # `find_similar_failures` queries would risk it surfacing as a spurious
    # fuzzy match during ordinary *recipe* lookup, corrupting the primary
    # recovery path's match quality.

    def upsert_guard(self, guard: GuardRow) -> None:
        self.store.upsert_guard(guard)

    def get_guard(self, tool_name: str, argument: str, kind: str) -> GuardRow | None:
        return self.store.get_guard(tool_name, argument, kind)

    def list_guards(
        self,
        tool_name: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[GuardRow]:
        return self.store.list_guards(tool_name=tool_name, active_only=active_only, limit=limit)

    def delete_guard(self, tool_name: str, argument: str, kind: str) -> None:
        self.store.delete_guard(tool_name, argument, kind)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self.store.close()
        self.vector_index.close()

    def __enter__(self) -> Oracle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
