"""Standing guard domain model and lifecycle (Phase 2).

`oracle/store.py` owns raw persistence of the `guards` table (`GuardRow`,
bare CRUD). This module owns the domain behavior on top of it: promoting a
proven `Recipe` into a proactive guard, tracking its own `times_applied`/
`success_rate` (distinct from the recipe it was promoted from — a guard's
numbers measure *prevention*, not reactive recovery), and describing active
guards as text for a caller to splice into their own system prompt.

A guard is matched by `(tool_name, argument, kind)`, not by a failure
`signature` the way a `Recipe` is — pre-call, before the tool has even been
attempted, there's no `error_type`/`error_message` yet to build a signature
from. Matching is on which tool/argument is involved, not on which error
already happened.

Revocation is sticky: once an operator explicitly revokes a guard
(`active=False`), `promote()` refuses to silently reactivate it, even if the
underlying recipe keeps succeeding. An explicit "no" from a human takes
precedence over automatic promotion.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from resilientforge.oracle.store import GuardRow

if TYPE_CHECKING:
    from resilientforge.oracle import Oracle


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StandingGuard(BaseModel):
    tool_name: str
    argument: str
    kind: Literal["transform", "patch"]
    source_signature: str
    created_at: str
    transform: str | None = None
    patch_value: Any = None
    root_cause: str | None = None
    active: bool = True
    last_applied: str | None = None
    times_applied: int = 0
    times_succeeded: int = 0
    success_rate: float = 0.0

    @classmethod
    def _from_row(cls, row: GuardRow) -> StandingGuard:
        return cls(
            tool_name=row.tool_name,
            argument=row.argument,
            kind=row.kind,
            transform=row.transform,
            patch_value=row.patch_value,
            source_signature=row.source_signature,
            root_cause=row.root_cause,
            active=row.active,
            created_at=row.created_at,
            last_applied=row.last_applied,
            times_applied=row.times_applied,
            times_succeeded=row.times_succeeded,
            success_rate=row.success_rate,
        )

    def _to_row(self) -> GuardRow:
        return GuardRow(
            tool_name=self.tool_name,
            argument=self.argument,
            kind=self.kind,
            transform=self.transform,
            patch_value=self.patch_value,
            source_signature=self.source_signature,
            root_cause=self.root_cause,
            active=self.active,
            created_at=self.created_at,
            last_applied=self.last_applied,
            times_applied=self.times_applied,
            times_succeeded=self.times_succeeded,
            success_rate=self.success_rate,
        )


class GuardManager:
    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle

    def promote(
        self,
        *,
        tool_name: str,
        argument: str,
        kind: str,
        source_signature: str,
        transform: str | None = None,
        patch_value: Any = None,
        root_cause: str | None = None,
    ) -> StandingGuard | None:
        """Create or update the guard for `(tool_name, argument, kind)`.

        Returns None (a no-op) if a guard for this key already exists and
        was explicitly revoked — see the module docstring on why that's
        sticky rather than silently overridden."""
        existing = self.oracle.get_guard(tool_name, argument, kind)
        if existing is not None and not existing.active:
            return None
        if existing is None:
            guard = StandingGuard(
                tool_name=tool_name,
                argument=argument,
                kind=kind,
                transform=transform,
                patch_value=patch_value,
                source_signature=source_signature,
                root_cause=root_cause,
                created_at=_utcnow(),
            )
        else:
            guard = StandingGuard._from_row(existing)
            guard.transform = transform
            guard.patch_value = patch_value
            guard.source_signature = source_signature
            guard.root_cause = root_cause or guard.root_cause
        self.oracle.upsert_guard(guard._to_row())
        return guard

    def get(self, tool_name: str, argument: str, kind: str) -> StandingGuard | None:
        row = self.oracle.get_guard(tool_name, argument, kind)
        return StandingGuard._from_row(row) if row else None

    def list_active(self, tool_name: str | None = None, limit: int = 100) -> list[StandingGuard]:
        return self.list(tool_name=tool_name, active_only=True, limit=limit)

    def list(
        self,
        tool_name: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[StandingGuard]:
        rows = self.oracle.list_guards(tool_name=tool_name, active_only=active_only, limit=limit)
        return [StandingGuard._from_row(row) for row in rows]

    def revoke(self, tool_name: str, argument: str, kind: str | None = None) -> list[StandingGuard]:
        """Deactivate the active guard(s) matching `(tool_name, argument)`.
        If `kind` is given, only that exact guard. If `kind` is None, every
        active guard for that argument (there can be at most two: one
        "transform", one "patch"). Returns the guards that were revoked."""
        candidates = [
            g
            for g in self.list_active(tool_name=tool_name, limit=10_000)
            if g.argument == argument and (kind is None or g.kind == kind)
        ]
        revoked = []
        for guard in candidates:
            guard.active = False
            self.oracle.upsert_guard(guard._to_row())
            revoked.append(guard)
        return revoked

    def prune(
        self,
        *,
        min_success_rate: float = 0.0,
        min_times_applied: int = 1,
        max_age_days: float | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> list[tuple[str, str, str]]:
        """Delete guards that are unreliable (success_rate below the
        floor, once applied at least `min_times_applied` times) and/or
        stale (`last_applied` older than `max_age_days`) — mirrors
        `RecipeManager.prune` exactly, for parity (Phase 5: guards had
        no equivalent to this at all before). Returns the
        `(tool_name, argument, kind)` triples that were (or, with
        `dry_run=True`, would be) pruned.

        Distinct from `revoke()`: `revoke()` is a sticky, explicit "no"
        an operator chose; `prune()` is unattended maintenance removing
        guards that have quietly stopped being useful — same
        distinction `RecipeManager.prune` already draws for recipes. A
        guard that's never been applied (`last_applied is None`) is
        never treated as stale by age — there's no age to measure yet.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days) if max_age_days is not None else None

        pruned: list[tuple[str, str, str]] = []
        for guard in self.list(active_only=False, limit=10_000):
            stale = (
                cutoff is not None
                and guard.last_applied is not None
                and datetime.fromisoformat(guard.last_applied) < cutoff
            )
            unreliable = (
                guard.times_applied >= min_times_applied
                and guard.success_rate < min_success_rate
            )
            if stale or unreliable:
                if not dry_run:
                    self.oracle.delete_guard(guard.tool_name, guard.argument, guard.kind)
                pruned.append((guard.tool_name, guard.argument, guard.kind))
        return pruned

    def record_application(self, guards: list[StandingGuard], *, succeeded: bool) -> None:
        """Atomically bump times_applied (+1 each)/times_succeeded (+1
        if succeeded)/success_rate/last_applied for every guard that
        fired on one call.

        Phase 5: this used to increment an in-memory `StandingGuard`
        (fetched earlier, before the call was even attempted) and write
        it back — a lost-update race under concurrent applications of
        the same guard, found by actually load-testing it (see
        `oracle/store.py`'s `record_guard_application`, one atomic SQL
        statement per guard now). Mutates the `guard` objects passed in
        with the freshly-read, post-increment values, so callers (e.g.
        `WrappedAgent._maybe_demote_guards`) see the CURRENT numbers —
        same in-place-mutation contract as before, just correct under
        concurrency now."""
        now = _utcnow()
        for guard in guards:
            row = self.oracle.record_guard_application(
                guard.tool_name, guard.argument, guard.kind, succeeded=succeeded, now=now
            )
            if row is None:
                continue  # guard was deleted/pruned between firing and this call
            guard.times_applied = row.times_applied
            guard.times_succeeded = row.times_succeeded
            guard.success_rate = row.success_rate
            guard.last_applied = row.last_applied

    def describe(self, tool_name: str | None = None) -> str:
        """Human/LLM-readable text block for active guards. The caller
        splices this into THEIR OWN system prompt — never auto-injected
        anywhere in this codebase (see integrations/*.py's adapters,
        neither of which has any system-prompt access at all)."""
        guards = self.list_active(tool_name=tool_name, limit=10_000)
        if not guards:
            return "No active guards."
        lines = []
        for guard in guards:
            cause = guard.root_cause or "a recurring issue"
            if guard.kind == "transform":
                action = f"automatic correction ({guard.transform}) is applied as a fallback"
            else:
                action = f"defaults to {guard.patch_value!r} when omitted"
            lines.append(f"- {guard.tool_name}({guard.argument}): {cause}. {action}.")
        return "\n".join(lines)
