"""Recipe domain model and read/write/prune logic (PROJECT_SPEC.md §4.3).

`oracle/store.py` owns raw persistence of the `recipes` table (`RecipeRow`,
bare CRUD). This module owns the domain behavior on top of it: turning a
successful recovery into a new-or-updated `Recipe`, tracking
`times_applied`/`success_rate` correctly across repeated occurrences of the
same failure signature, and pruning recipes that are stale or unreliable.

Naming note: PROJECT_SPEC.md §4.3's illustrative recipe JSON uses the key
`failure_signature`; this codebase uses `signature` consistently across
`FailureRecord`, `RecipeRow`, `Oracle`, and here, since they all key off the
same normalized string from `core/signature.py`. Not a functional
deviation, just one field name instead of two for the same concept.

All operations go through `Oracle`, not `SQLiteStore` directly, so the
vector index stays in sync with the structured store (PROJECT_SPEC.md §4.3:
the oracle is meant to look like one interface, not two backends callers
have to keep consistent themselves).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from resilientforge.oracle.store import RecipeRow

if TYPE_CHECKING:
    from resilientforge.oracle import Oracle


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Recipe(BaseModel):
    signature: str
    tool_name: str
    fix_detail: dict[str, Any]
    created_at: str
    last_used: str
    root_cause: str | None = None
    fix_strategy: str | None = None
    times_applied: int = 0
    times_succeeded: int = 0
    success_rate: float = 0.0

    @classmethod
    def _from_row(cls, row: RecipeRow) -> Recipe:
        return cls(
            signature=row.signature,
            tool_name=row.tool_name,
            fix_detail=row.fix_detail,
            created_at=row.created_at,
            last_used=row.last_used,
            root_cause=row.root_cause,
            fix_strategy=row.fix_strategy,
            times_applied=row.times_applied,
            times_succeeded=row.times_succeeded,
            success_rate=row.success_rate,
        )

    def _to_row(self) -> RecipeRow:
        return RecipeRow(
            signature=self.signature,
            tool_name=self.tool_name,
            fix_detail=self.fix_detail,
            created_at=self.created_at,
            last_used=self.last_used,
            root_cause=self.root_cause,
            fix_strategy=self.fix_strategy,
            times_applied=self.times_applied,
            times_succeeded=self.times_succeeded,
            success_rate=self.success_rate,
        )


class RecipeManager:
    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle

    def record_success(
        self,
        *,
        signature: str,
        tool_name: str,
        fix_detail: dict[str, Any],
        root_cause: str | None = None,
        fix_strategy: str | None = None,
    ) -> Recipe:
        """A fix for `signature` just worked and passed re-verification —
        create the recipe if this is the first time, or update
        times_applied/times_succeeded/success_rate if it already existed."""
        now = _utcnow().isoformat()
        existing = self.oracle.get_recipe(signature)
        if existing is None:
            recipe = Recipe(
                signature=signature,
                tool_name=tool_name,
                fix_detail=fix_detail,
                root_cause=root_cause,
                fix_strategy=fix_strategy,
                times_applied=1,
                times_succeeded=1,
                success_rate=1.0,
                created_at=now,
                last_used=now,
            )
        else:
            recipe = Recipe._from_row(existing)
            recipe.times_applied += 1
            recipe.times_succeeded += 1
            recipe.success_rate = recipe.times_succeeded / recipe.times_applied
            recipe.fix_detail = fix_detail
            recipe.root_cause = root_cause or recipe.root_cause
            recipe.fix_strategy = fix_strategy or recipe.fix_strategy
            recipe.last_used = now
        self.oracle.upsert_recipe(recipe._to_row())
        return recipe

    def record_fast_path_failure(self, signature: str) -> Recipe | None:
        """A known recipe's fix was replayed on the fast path (no fresh LLM
        call) but did NOT pass re-verification this time — update
        times_applied/success_rate without incrementing times_succeeded.
        Returns None if no recipe exists for this signature (nothing to
        update)."""
        existing = self.oracle.get_recipe(signature)
        if existing is None:
            return None
        recipe = Recipe._from_row(existing)
        recipe.times_applied += 1
        recipe.success_rate = recipe.times_succeeded / recipe.times_applied
        recipe.last_used = _utcnow().isoformat()
        self.oracle.upsert_recipe(recipe._to_row())
        return recipe

    def get(self, signature: str) -> Recipe | None:
        row = self.oracle.get_recipe(signature)
        return Recipe._from_row(row) if row else None

    def list(self, tool_name: str | None = None, limit: int = 100) -> list[Recipe]:
        rows = self.oracle.list_recipes(tool_name=tool_name, limit=limit)
        return [Recipe._from_row(row) for row in rows]

    def prune(
        self,
        *,
        min_success_rate: float = 0.0,
        min_times_applied: int = 1,
        max_age_days: float | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete recipes that are unreliable (success_rate below the floor,
        once they've been applied at least `min_times_applied` times) and/or
        stale (last_used older than `max_age_days`). Returns the signatures
        that were (or, with `dry_run=True`, would be) pruned — the CLI's
        `prune --dry-run` uses this to preview without duplicating the
        selection logic."""
        now = now or _utcnow()
        cutoff = now - timedelta(days=max_age_days) if max_age_days is not None else None

        pruned: list[str] = []
        for recipe in self.list(limit=10_000):
            stale = cutoff is not None and datetime.fromisoformat(recipe.last_used) < cutoff
            unreliable = (
                recipe.times_applied >= min_times_applied
                and recipe.success_rate < min_success_rate
            )
            if stale or unreliable:
                if not dry_run:
                    self.oracle.delete_recipe(recipe.signature)
                pruned.append(recipe.signature)
        return pruned
