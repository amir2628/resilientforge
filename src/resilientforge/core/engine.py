"""The wrap() orchestration: tying the oracle (oracle/), signature
normalization (core/signature.py), invariants (core/invariants.py), and fix
generation/application (core/recovery.py) into the recovery flow.

Scope note on "zero configuration": that promise covers passive
fast-path recovery from recipes already in the oracle. Active,
reflection-based fix generation needs a `reflect` callable — this module
has no default implementation (mirrors core/recovery.py staying
vendor-neutral); a real Anthropic-backed default belongs in
integrations/raw_tool_loop.py, which already needs Anthropic
wiring for the tool-calling loop itself. Without `reflect`, wrap() still
recovers from anything a prior run already learned, but a genuinely novel
failure shape exhausts immediately instead of attempting reflection.
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from resilientforge.core.invariants import Invariant
from resilientforge.core.isolation import check_picklable, run_isolated
from resilientforge.core.recovery import (
    GUARD_SAFE_TRANSFORMS,
    TRANSFORM_REGISTRY,
    FailureContext,
    Fix,
    ReflectFn,
    TransformError,
    apply_fix,
    generate_fix,
)
from resilientforge.core.signature import build_signature
from resilientforge.oracle import Oracle, RecipeRow, ResolutionStatus
from resilientforge.oracle.guards import GuardManager, StandingGuard
from resilientforge.oracle.recipes import RecipeManager
from resilientforge.telemetry.metrics import MetricEvent, MetricsHook

FixSource = Literal["recipe", "reflection"]


@dataclass
class RecoveryAttempt:
    fix: Fix
    source: FixSource
    error_type: str | None = None
    error_message: str | None = None
    # True if this attempt's Fix referenced something that isn't real (an
    # argument_patch key, a transforms[].argument, or a transforms[].transform
    # name) and was rejected before ever reaching a live retry — see
    # WrappedAgent._invalid_fix_reasons.
    rejected: bool = False


@dataclass
class _Candidate:
    """One speculative candidate fix considered within a single
    `attempt_number` round (Phase 3). Transient — never exported,
    never persisted; only the eventual winner's `fix` ever reaches the
    oracle, via the same `_on_attempt_success` path Phase 1/2 already
    used."""

    fix: Fix
    source: FixSource
    applied_args: dict[str, Any]
    proxy_score: float | None = None  # only set for recipe-sourced candidates


class InvariantAbortError(Exception):
    def __init__(self, tool_name: str, violated: list[str]) -> None:
        self.tool_name = tool_name
        self.violated = violated
        super().__init__(
            f"tool {tool_name!r} aborted: invariant(s) failed with on_violation='abort': "
            + ", ".join(violated)
        )


class RecoveryExhaustedError(Exception):
    def __init__(
        self,
        tool_name: str,
        call_args: dict[str, Any],
        original_error: Exception | None,
        attempts: list[RecoveryAttempt],
    ) -> None:
        # NB: deliberately not `self.args` — BaseException.__init__ (called
        # below via super()) overwrites `.args` with its own message tuple,
        # which would silently clobber the tool-call args dict.
        self.tool_name = tool_name
        self.call_args = call_args
        self.original_error = original_error
        self.attempts = attempts
        detail = f": {original_error}" if original_error is not None else ""
        super().__init__(
            f"exhausted {len(attempts)} recovery attempt(s) for tool {tool_name!r}{detail}"
        )


def _resolve_callable(agent: Any) -> Callable[..., Any]:
    for attr in ("invoke", "run"):
        candidate = getattr(agent, attr, None)
        if callable(candidate):
            return candidate
    if callable(agent):
        return agent
    raise TypeError(
        f"wrap() needs an object with .invoke()/.run(), or a plain callable; got {type(agent)!r}"
    )


def _infer_valid_arguments(tool_fn: Callable[..., Any]) -> set[str] | None:
    """Best-effort: the real parameter names `tool_fn` accepts, used to
    reject a Fix's `argument_patch` key that could never actually reach the
    tool (found via a real-world validation exercise, see
    docs/real_world_validation_round2.md — a proposed fix silently no-oped
    instead of erroring, and got recorded as "recovered" for reasons
    unrelated to the fix at all).

    Returns `None` (meaning "unknown — don't validate") when `tool_fn`'s
    signature can't be introspected, or only declares `**kwargs` with no
    named parameters — a generic passthrough shim (e.g.
    `integrations/langgraph_adapter.py`'s per-call closures) tells us
    nothing about the REAL tool's schema; that integration passes
    `valid_arguments` explicitly instead, derived from the actual tool's
    schema. Erring toward `None` (no validation) rather than an empty set
    when we can't tell is deliberate: this is a targeted fix for a
    confirmed real gap, not a license to reject fixes we simply can't
    verify.
    """
    try:
        sig = inspect.signature(tool_fn)
    except (TypeError, ValueError):
        return None
    names = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return names or None


def _default_tool_name(agent: Any, tool_fn: Callable[..., Any]) -> str:
    if agent is tool_fn:
        # `agent` was used directly as the callable (a plain function) —
        # its own name is more useful than the generic method name a
        # bound `.invoke`/`.run` would give (see the `else` branch, where
        # we deliberately do NOT use tool_fn.__name__, which would just be
        # "invoke" or "run" for every wrapped agent).
        name = getattr(tool_fn, "__name__", None)
        if name and name != "<lambda>":
            return name
    return type(agent).__name__


class WrappedAgent:
    def __init__(
        self,
        tool_fn: Callable[..., Any],
        tool_name: str,
        invariants: list[Invariant],
        oracle: Oracle,
        max_recovery_attempts: int,
        reflect: ReflectFn | None,
        similarity_threshold: float,
        workflow_id: str | None,
        enable_standing_guards: bool = True,
        guard_promotion_min_occurrences: int = 3,
        guard_promotion_min_success_rate: float = 0.8,
        num_branches: int = 1,
        side_effect_free: bool = False,
        isolate: bool = False,
        call_timeout: float | None = None,
        max_memory_mb: int | None = None,
        max_cpu_seconds: float | None = None,
        guard_demotion_min_occurrences: int = 3,
        guard_demotion_max_failure_rate: float = 0.5,
        recipe_min_success_rate: float | None = None,
        recipe_reliability_min_occurrences: int = 3,
        metrics: MetricsHook | None = None,
        valid_arguments: set[str] | None = None,
    ) -> None:
        """
        valid_arguments: set[str] | None = None
            The tool's real accepted parameter names, used (via
            `_invalid_fix_reasons`) to reject a Fix that references one
            that isn't real — an `argument_patch` key or a
            `transforms[].argument` — BEFORE it's ever applied to a live
            retry or persisted as a recipe (found via a real-world
            validation exercise — see docs/real_world_validation_round2.md
            and docs/real_world_validation_round3.md — where a fix
            silently no-oped and got recorded as "recovered" for reasons
            unrelated to the fix; round 3 found this same problem slips
            through via `transforms` too, not just `argument_patch`).
            Defaults to `None`, in which case it's inferred from
            `tool_fn`'s own signature (works for a plain function/bound
            method with named parameters); pass this explicitly when
            `tool_fn` is a generic passthrough shim whose own signature
            wouldn't reveal the real tool's schema (see
            `integrations/langgraph_adapter.py`). `None` (whether given
            explicitly or left unresolved) means "unknown — don't
            validate," never "reject everything." A `transforms[].transform`
            name that isn't registered in `TRANSFORM_REGISTRY` at all is
            also rejected the same way, independent of `valid_arguments`
            (that registry is always fully known).

        metrics: MetricsHook | None = None (Phase 5)
            Optional observability hook — a callable receiving a
            `MetricEvent` (see `telemetry/metrics.py`) for real tool
            calls, how a recovery ultimately resolved, and guard fire/
            promote/revoke events. Vendor-neutral, same
            caller-injects-a-callable pattern as `reflect` — this stays
            a no-op unless you provide one.

        guard_demotion_min_occurrences / guard_demotion_max_failure_rate (Phase 5)
            The reverse of guard promotion: once a guard has fired at
            least `guard_demotion_min_occurrences` times and its
            failure rate (`1 - success_rate`) exceeds
            `guard_demotion_max_failure_rate`, it's auto-revoked via
            the same sticky `revoke()` a human would use. Always
            enabled (no separate on/off flag): it only ever acts on a
            guard whose track record has actually gone bad, so it can
            never turn a working scenario into a failing one — a guard
            that's stopped working just stops firing, and recovery
            reverts to Phase 1's reactive per-call fixing, exactly as
            safe as if the guard had never existed.

        recipe_min_success_rate / recipe_reliability_min_occurrences (Phase 5)
            Opt-in (`recipe_min_success_rate` defaults to `None` =
            today's unconditional behavior — a recipe is always tried
            first regardless of its track record, exactly as every
            prior phase did). When set, a recipe whose `success_rate`
            has fallen below this floor (once it's been applied at
            least `recipe_reliability_min_occurrences` times) is
            skipped as a fast-path candidate — falling straight through
            to reflection instead of proposing a fix that's stopped
            working. Kept opt-in rather than defaulting to enabled like
            guard demotion, since this changes the core Phase 1
            fast-path lookup order itself, not just a proactive
            optimization on top of it.

        side_effect_free: bool = False
            Vouches that `tool_fn` has no problematic real-world effect
            regardless of which arguments it's called with, and is
            therefore safe to actually invoke once per speculative
            candidate within a single recovery attempt (not just once
            with the eventual winner).

            This is NOT classic idempotency ("calling twice with the SAME
            input is a no-op the second time" — e.g. PUT). It is closer to
            HTTP's notion of a "safe" method (GET/HEAD): true for
            read-only lookups, pure computations, and validations; FALSE
            for anything that creates, sends, charges, books, or deletes
            for real, even if that operation is itself idempotent in the
            classic sense.

            Only meaningful when `num_branches > 1`. Default False: Phase
            3 never risks a duplicate real-world side effect unless you
            explicitly opt in per-tool.

        isolate: bool = False
            Runs every real `tool_fn` call in a freshly-spawned
            subprocess (Phase 4). A hang past `call_timeout`, or a crash
            (segfault, `os._exit`, a resource-limit signal), becomes a
            normal recoverable failure instead of taking down the host
            process — protective isolation of the CALLER, not of the
            world `tool_fn` touches.

            This does NOT, and cannot, undo a real-world side effect
            `tool_fn` already performed before it hung or crashed — an
            HTTP request already sent stays sent. No code-level sandbox
            can reverse that; only `tool_fn`'s own cooperation (retries,
            idempotency keys, transactions) can make that safe, which is
            exactly what `side_effect_free` already asks a caller to
            vouch for separately.

            Requires `tool_fn` to be picklable (checked eagerly here, at
            construction, not on the first call) — a module-level
            function or a bound method on a picklable object works; a
            locally-defined closure or lambda does not, since the
            subprocess boundary means `tool_fn` has to be pickled across
            it. Default False: every prior phase's behavior is
            unchanged unless you opt in.

        call_timeout: float | None = None
            Wall-clock seconds before an isolated call is terminated.
            Only enforced when `isolate=True` — there is no reliable,
            cross-platform way to preempt arbitrary in-process Python
            code without a process boundary, so this is silently a
            no-op (with a construction-time warning) otherwise.

        max_memory_mb / max_cpu_seconds: int | float | None = None
            POSIX-only (`resource.setrlimit`) resource ceilings applied
            inside the isolated subprocess, before `tool_fn` runs. Only
            enforced when `isolate=True`; a no-op with a construction-
            time warning on Windows. Best-effort even on POSIX systems:
            the kernel ultimately decides whether a given limit is
            honorable at all (e.g. `RLIMIT_AS` is refused outright on
            some POSIX systems) — a limit that can't be applied is
            reported as its own distinct, honest failure, never silently
            skipped and never misattributed to `tool_fn` itself.
        """
        self.tool_fn = tool_fn
        self.tool_name = tool_name
        self.invariants = invariants
        self.oracle = oracle
        self.recipes = RecipeManager(oracle)
        self.guards = GuardManager(oracle)
        self.max_recovery_attempts = max_recovery_attempts
        self.reflect = reflect
        self.similarity_threshold = similarity_threshold
        self.workflow_id = workflow_id
        self.enable_standing_guards = enable_standing_guards
        self.guard_promotion_min_occurrences = guard_promotion_min_occurrences
        self.guard_promotion_min_success_rate = guard_promotion_min_success_rate
        self.guard_demotion_min_occurrences = guard_demotion_min_occurrences
        self.guard_demotion_max_failure_rate = guard_demotion_max_failure_rate
        self.recipe_min_success_rate = recipe_min_success_rate
        self.recipe_reliability_min_occurrences = recipe_reliability_min_occurrences
        self.metrics = metrics
        self.valid_arguments = (
            valid_arguments if valid_arguments is not None else _infer_valid_arguments(tool_fn)
        )
        self.num_branches = num_branches
        self.side_effect_free = side_effect_free
        if side_effect_free and num_branches <= 1:
            warnings.warn(
                "ResilientForge: side_effect_free=True has no effect when "
                "num_branches <= 1 — there's only ever one candidate to consider.",
                stacklevel=2,
            )

        self.isolate = isolate
        self.call_timeout = call_timeout
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        if not isolate and (call_timeout is not None or max_memory_mb is not None or max_cpu_seconds is not None):
            warnings.warn(
                "ResilientForge: call_timeout/max_memory_mb/max_cpu_seconds have "
                "no effect when isolate=False — they're only enforced inside the "
                "isolated subprocess isolate=True dispatches to.",
                stacklevel=2,
            )
        if isolate and sys.platform == "win32" and (max_memory_mb is not None or max_cpu_seconds is not None):
            warnings.warn(
                "ResilientForge: max_memory_mb/max_cpu_seconds are POSIX-only "
                "(resource.setrlimit) and will have no effect on Windows.",
                stacklevel=2,
            )
        if isolate:
            check_picklable(tool_fn)

    def _emit(self, event_type: str, **fields: Any) -> None:
        """No-op unless `metrics` was provided — every call site below
        can unconditionally call this without checking `self.metrics is
        None` itself."""
        if self.metrics is None:
            return
        self.metrics(
            MetricEvent(
                event_type=event_type,
                tool_name=self.tool_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                **fields,
            )
        )

    # -- the recovery loop -------------------------------------------------

    def invoke(self, **kwargs: Any) -> Any:
        current_args = dict(kwargs)
        fired_guards: list[StandingGuard] = []
        if self.enable_standing_guards:
            current_args, fired_guards = self._apply_standing_guards(current_args)

        result, error = self._call(current_args)
        classification = self._classify_failure(result, error)  # may raise InvariantAbortError
        if classification is None:
            self._emit("call_result", success=True, source="initial")
            if fired_guards:
                # Prevention, not recovery: a guard changed the
                # args before the first attempt and that attempt succeeded
                # outright — no failure was ever recorded for this call.
                self.guards.record_application(fired_guards, succeeded=True)
                self._maybe_demote_guards(fired_guards)
                for guard in fired_guards:
                    self._emit("guard_fired", argument=guard.argument, kind=guard.kind, success=True)
            return result
        if fired_guards:
            # The guard fired but wasn't sufficient on its own — record the
            # miss, then fall through to the normal Phase 1 recovery loop
            # below exactly as if no guard had fired (using current_args,
            # which already reflects whatever the guard changed).
            self.guards.record_application(fired_guards, succeeded=False)
            self._maybe_demote_guards(fired_guards)
            for guard in fired_guards:
                self._emit("guard_fired", argument=guard.argument, kind=guard.kind, success=False)
        error_type, error_message = classification
        self._emit("call_result", success=False, source="initial", error_type=error_type)
        original_error = error

        signature = build_signature(
            tool_name=self.tool_name,
            error_type=error_type,
            error_message=error_message,
            args=current_args,
        )
        failure = self.oracle.record_failure(
            tool_name=self.tool_name,
            signature=signature,
            workflow_id=self.workflow_id,
            error_type=error_type,
            error_message=error_message,
            sanitized_args=current_args,
        )

        attempts: list[RecoveryAttempt] = []
        try:
            for attempt_number in range(1, self.max_recovery_attempts + 1):
                if self.num_branches <= 1:
                    # Byte-for-byte the Phase 1/2 path — untouched by
                    # Phase 3's candidate-batch machinery below.
                    fix, source = self._find_fix(
                        signature, current_args, error_type, error_message, attempt_number, attempts
                    )
                    if fix is None:
                        break  # no recipe match and no `reflect` configured — nothing left to try

                    invalid_reasons = self._invalid_fix_reasons(fix)
                    if invalid_reasons:
                        self._on_attempt_rejected(signature, fix, source, invalid_reasons, attempts)
                        continue  # never reaches the tool, never persisted as a recipe

                    new_args, retry_result, retry_error = self._attempt(current_args, fix)
                    retry_classification = self._classify_failure(retry_result, retry_error)

                    if retry_classification is None:
                        self._on_attempt_success(signature, fix, source, failure.id, attempt_number)
                        return retry_result

                    retry_error_type, retry_error_message = retry_classification
                    self._on_attempt_failure(
                        signature, fix, source, retry_error_type, retry_error_message, attempts
                    )
                    current_args = new_args
                    continue

                # Phase 3: speculative branching — consider a batch of up
                # to `num_branches` candidates this round.
                candidates = self._find_fix_candidates(
                    signature, current_args, error_type, error_message, attempt_number, attempts
                )
                if not candidates:
                    break  # nothing new to propose — same exhaustion condition as above

                if not self.side_effect_free:
                    candidate, retry_result, retry_error = self._try_best_proxy_ranked(candidates)
                    retry_classification = self._classify_failure(retry_result, retry_error)
                    if retry_classification is None:
                        self._on_attempt_success(
                            signature, candidate.fix, candidate.source, failure.id, attempt_number
                        )
                        return retry_result
                    retry_error_type, retry_error_message = retry_classification
                    self._on_attempt_failure(
                        signature, candidate.fix, candidate.source,
                        retry_error_type, retry_error_message, attempts,
                    )
                    current_args = candidate.applied_args
                else:
                    winner, winner_result, failed = self._try_all_real(candidates)
                    for failed_candidate, f_error_type, f_error_message in failed:
                        self._on_attempt_failure(
                            signature, failed_candidate.fix, failed_candidate.source,
                            f_error_type, f_error_message, attempts,
                        )
                    if winner is not None:
                        self._on_attempt_success(
                            signature, winner.fix, winner.source, failure.id, attempt_number
                        )
                        return winner_result
                    # Every candidate this round failed for real; carry the
                    # last-tried candidate's applied args forward, mirroring
                    # how the single-candidate path advances `current_args`.
                    current_args = failed[-1][0].applied_args
        except InvariantAbortError:
            self.oracle.update_failure_resolution(failure.id, ResolutionStatus.ABORTED)
            self._emit("recovery_resolved", resolution="aborted", total_attempts=len(attempts))
            raise

        # Every attempt this call made was rejected before ever reaching the
        # tool (see _on_attempt_rejected) — distinct from EXHAUSTED, which
        # implies at least one attempt was a real, live retry. If at least
        # one real attempt happened along the way too, EXHAUSTED is still
        # the accurate status (mixed case: some proposals were invalid,
        # real retries were still tried and still didn't work).
        final_status = (
            ResolutionStatus.FIX_REJECTED
            if attempts and all(a.rejected for a in attempts)
            else ResolutionStatus.EXHAUSTED
        )
        self.oracle.update_failure_resolution(failure.id, final_status)
        self._emit("recovery_resolved", resolution=final_status.value, total_attempts=len(attempts))
        raise RecoveryExhaustedError(
            tool_name=self.tool_name,
            call_args=kwargs,
            original_error=original_error,
            attempts=attempts,
        ) from original_error

    # -- standing guards (Phase 2: "continuous", pre-call checking) ----------

    def _apply_standing_guards(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], list[StandingGuard]]:
        """Proactively apply any active guard for this tool BEFORE the first
        call attempt — this is what "prevented rather than merely recovered
        from" means: skip a guaranteed-to-fail call entirely once a pattern
        is well-established, instead of always failing once before the
        Phase 1 recovery loop kicks in on retry.

        Guards must only ever help or be neutral, never break an otherwise-
        fine call: patch-kind guards only fill a MISSING key (never
        overwrite a caller-provided value); transform-kind guards that raise
        TransformError on this call's actual value are silently skipped
        (this call just didn't need it) rather than propagating.
        """
        guards = self.guards.list_active(tool_name=self.tool_name)
        new_args = dict(args)
        fired: list[StandingGuard] = []
        for guard in (g for g in guards if g.kind == "patch"):
            if guard.argument not in new_args:
                new_args[guard.argument] = guard.patch_value
                fired.append(guard)
        for guard in (g for g in guards if g.kind == "transform"):
            if guard.argument not in new_args:
                continue
            try:
                new_value = TRANSFORM_REGISTRY[guard.transform](new_args[guard.argument])
            except TransformError:
                continue  # this call's value didn't need it — no-op
            if new_value != new_args[guard.argument]:
                new_args[guard.argument] = new_value
                fired.append(guard)
        return new_args, fired

    def _maybe_promote_guard(self, signature: str, fix: Fix, recipe: Any) -> None:
        """Once a recipe has proven itself reliable, promote its fix into a
        standing guard so future calls apply it pre-call instead of failing
        once first. Occurrence counting is dual-mode: scoped to this
        `workflow_id` when one was given to wrap() (using the same
        `Oracle.list_failures` filter the oracle already supports for
        free), otherwise the recipe's own global `times_applied`.
        """
        if self.workflow_id is not None:
            occurrences = len(self.oracle.list_failures(signature=signature, workflow_id=self.workflow_id))
        else:
            occurrences = recipe.times_applied
        if occurrences < self.guard_promotion_min_occurrences:
            return
        if recipe.success_rate < self.guard_promotion_min_success_rate:
            return

        for arg, value in fix.argument_patch.items():
            promoted = self.guards.promote(
                tool_name=self.tool_name,
                argument=arg,
                kind="patch",
                patch_value=value,
                source_signature=signature,
                root_cause=fix.root_cause,
            )
            if promoted is not None:  # None means a no-op (sticky-revoked) — not a real promotion
                self._emit("guard_promoted", argument=arg, kind="patch")
        for arg_transform in fix.transforms:
            if arg_transform.transform not in GUARD_SAFE_TRANSFORMS:
                continue
            promoted = self.guards.promote(
                tool_name=self.tool_name,
                argument=arg_transform.argument,
                kind="transform",
                transform=arg_transform.transform,
                source_signature=signature,
                root_cause=fix.root_cause,
            )
            if promoted is not None:
                self._emit("guard_promoted", argument=arg_transform.argument, kind="transform")

    def _maybe_demote_guards(self, guards: list[StandingGuard]) -> None:
        """The reverse of `_maybe_promote_guard` (Phase 5): once a guard
        has fired enough times and its failure rate has crossed the
        demotion threshold, auto-revoke it via the same sticky
        `revoke()` a human would use — a guard that's stopped working
        shouldn't keep firing just because nothing is watching.
        `guards` here have already had `record_application` update
        their `times_applied`/`success_rate` in place (same objects,
        not fresh copies), so this reads the just-recorded numbers."""
        for guard in guards:
            if guard.times_applied < self.guard_demotion_min_occurrences:
                continue
            failure_rate = 1.0 - guard.success_rate
            if failure_rate > self.guard_demotion_max_failure_rate:
                revoked = self.guards.revoke(guard.tool_name, guard.argument, guard.kind)
                if revoked:
                    self._emit("guard_revoked", argument=guard.argument, kind=guard.kind)

    def describe_guards(self) -> str:
        """Human/LLM-readable text describing this tool's active guards —
        splice into YOUR OWN system prompt if you want the model to see
        them. Never auto-injected anywhere in this codebase (neither
        adapter has system-prompt access to begin with)."""
        return self.guards.describe(tool_name=self.tool_name)

    # -- shared success/failure tail (Phase 3: reused by every selection path) -

    def _on_attempt_success(
        self, signature: str, fix: Fix, source: FixSource, failure_id: int, attempt_number: int
    ) -> None:
        """Write the fix back as a recipe, mark the failure resolved, and
        (if enabled) check whether it's now reliable enough to promote
        into a standing guard. Called identically by the single-candidate
        path and both multi-candidate (speculative branching) paths, so
        this bookkeeping can never silently diverge between them."""
        recipe = self.recipes.record_success(
            signature=signature,
            tool_name=self.tool_name,
            fix_detail=fix.model_dump(),
            root_cause=fix.root_cause,
            fix_strategy=fix.strategy,
        )
        self.oracle.update_failure_resolution(
            failure_id,
            ResolutionStatus.RECOVERED,
            fix_applied=fix.model_dump(),
            fix_verified=True,
        )
        if self.enable_standing_guards:
            self._maybe_promote_guard(signature, fix, recipe)
        self._emit("call_result", success=True, source=source, attempt_number=attempt_number)
        self._emit("recovery_resolved", resolution="recovered", total_attempts=attempt_number)

    def _on_attempt_failure(
        self,
        signature: str,
        fix: Fix,
        source: FixSource,
        error_type: str,
        error_message: str,
        attempts: list[RecoveryAttempt],
    ) -> None:
        """Record the failed attempt, and if it was a recipe fast-path
        replay that didn't pan out, record that against the recipe's own
        track record too. Same reuse rationale as `_on_attempt_success`."""
        attempts.append(
            RecoveryAttempt(
                fix=fix, source=source, error_type=error_type, error_message=error_message
            )
        )
        if source == "recipe":
            self.recipes.record_fast_path_failure(signature)
        self._emit(
            "call_result", success=False, source=source,
            error_type=error_type, attempt_number=len(attempts),
        )

    def _invalid_fix_reasons(self, fix: Fix) -> list[str]:
        """The ONE shared validation every live application of a `Fix`
        goes through — `_attempt` (single-candidate path) and
        `_add_candidate` (Phase 3 speculative branching) both call this
        before `apply_fix`, so there's exactly one place this logic can
        drift, not two that could disagree (found necessary the hard
        way: round 2's real-world validation caught an invalid
        `argument_patch` key; a fix for that alone still let the *same*
        underlying problem — a proposed correction referencing something
        that isn't real — slip through via `transforms` in round 3's
        confirmation run; see docs/real_world_validation_round3.md).

        Checks, all independent (a fix with ANY of these is rejected as a
        whole — never partially applied):
        - an `argument_patch` key that isn't a real tool parameter
        - a `transforms[].argument` that isn't a real tool parameter
        - a `transforms[].transform` name that isn't registered in
          `TRANSFORM_REGISTRY` at all (previously only surfaced as a raw
          `TransformError` from deep inside `apply_fix` — now caught here
          instead, before ever reaching a live retry)

        The two argument-name checks are skipped (never reject) when
        `self.valid_arguments` is `None` — unknown, not "reject
        everything." Transform-name validation always runs regardless,
        since `TRANSFORM_REGISTRY` is always fully known, not something
        that can be "unknown" the way a tool's real parameters can be.
        """
        reasons: list[str] = []
        if self.valid_arguments is not None:
            invalid_patch_keys = sorted(set(fix.argument_patch) - self.valid_arguments)
            if invalid_patch_keys:
                reasons.append(
                    f"argument_patch key(s) not in tool's real parameters: {invalid_patch_keys}"
                )
            invalid_transform_args = sorted(
                {t.argument for t in fix.transforms if t.argument not in self.valid_arguments}
            )
            if invalid_transform_args:
                reasons.append(
                    "transforms[].argument not in tool's real parameters: "
                    f"{invalid_transform_args}"
                )
        unknown_transforms = sorted(
            {t.transform for t in fix.transforms if t.transform not in TRANSFORM_REGISTRY}
        )
        if unknown_transforms:
            reasons.append(f"transforms[].transform not a registered transform: {unknown_transforms}")
        return reasons

    def _on_attempt_rejected(
        self,
        signature: str,
        fix: Fix,
        source: FixSource,
        reasons: list[str],
        attempts: list[RecoveryAttempt],
    ) -> None:
        """A Fix referenced something that can't actually reach the tool
        (see `_invalid_fix_reasons`) — reject it before it's ever applied
        to a live retry or persisted as a recipe (see
        docs/real_world_validation_round2.md /
        docs/real_world_validation_round3.md: the tool-calling layer, or
        `apply_fix`'s own argument-presence guard, previously just
        silently dropped/skipped the invalid reference, and whatever
        happened next — success or failure — got misattributed to this
        fix). Recorded into `attempts` so the next reflection call sees it
        via `previous_attempts` (the model gets a chance to learn its
        proposal referenced something that doesn't exist), and so a
        recipe-sourced rejection still marks `already_tried_recipe` true,
        exactly like a real recipe failure would — this doesn't loop
        forever retrying the same bad recipe."""
        error_message = "; ".join(reasons)
        attempts.append(
            RecoveryAttempt(
                fix=fix,
                source=source,
                error_type="invalid_fix_reference",
                error_message=error_message,
                rejected=True,
            )
        )
        if source == "recipe":
            self.recipes.record_fast_path_failure(signature)
        self._emit(
            "call_result", success=False, source=source,
            error_type="invalid_fix_reference", attempt_number=len(attempts),
        )

    # -- helpers -------------------------------------------------------------

    def _call(self, args: dict[str, Any]) -> tuple[Any, Exception | None]:
        if self.isolate:
            # The one branch point for Phase 4 isolation: every real
            # invocation (the initial attempt, recipe/reflection
            # retries, and Phase 3's speculative real calls) already
            # funnels through this single method, so nothing else in
            # this class needs to change.
            return run_isolated(
                self.tool_fn,
                args,
                timeout=self.call_timeout,
                max_memory_mb=self.max_memory_mb,
                max_cpu_seconds=self.max_cpu_seconds,
            )
        try:
            return self.tool_fn(**args), None
        except Exception as exc:  # intentionally broad: any tool-call failure must be caught
            return None, exc

    def _attempt(
        self, args: dict[str, Any], fix: Fix
    ) -> tuple[dict[str, Any], Any, Exception | None]:
        """Apply `fix` to `args` and call the tool. A failure *applying*
        the fix itself (e.g. an inapplicable transform — see
        core/recovery.py's TransformError) is treated the same as a
        tool-call failure, so the loop can still fall through to
        reflection instead of crashing the whole recovery attempt."""
        try:
            new_args = apply_fix(args, fix)
        except Exception as exc:
            return args, None, exc
        result, error = self._call(new_args)
        return new_args, result, error

    def _classify_failure(
        self, result: Any, error: Exception | None
    ) -> tuple[str, str] | None:
        """Returns (error_type, error_message) if this call should enter/
        continue recovery. Returns None for a genuine success OR for a
        violation whose invariants are all on_violation="warn" (accepted,
        not recovered). Raises InvariantAbortError if any violated
        invariant is on_violation="abort"."""
        if error is not None:
            return type(error).__name__, str(error)

        violated = [inv for inv in self.invariants if not inv.evaluate(result)]
        if not violated:
            return None

        if any(inv.on_violation == "abort" for inv in violated):
            raise InvariantAbortError(self.tool_name, [inv.name for inv in violated])

        names = ", ".join(inv.name for inv in violated)
        if any(inv.on_violation == "recover" for inv in violated):
            return "invariant_violation", f"invariant(s) failed: {names}"

        warnings.warn(
            f"ResilientForge: invariant(s) failed but on_violation='warn': {names}",
            stacklevel=3,
        )
        return None

    def _lookup_recipe_fix(self, signature: str) -> tuple[Fix, RecipeRow] | None:
        """Exact-match recipe lookup first; if none, a fuzzy vector-
        similarity match above `similarity_threshold`. Returns the
        matched recipe's `Fix` and raw `RecipeRow` (for its
        `success_rate`/`times_applied`), or None if no recipe applies.
        Factored out of `_find_fix` so the single-candidate (Phase 1/2)
        path and the multi-candidate (Phase 3) path share one
        implementation — no chance of the two silently diverging.

        Phase 5: if `recipe_min_success_rate` is set (opt-in, None by
        default), a recipe whose `success_rate` has fallen below it —
        once applied at least `recipe_reliability_min_occurrences`
        times, so one early failure can't disqualify a brand-new recipe
        — is treated as if it didn't match at all, falling straight
        through to reflection instead of proposing a fix that's stopped
        working."""
        recipe = self.oracle.get_recipe(signature)
        if recipe is None:
            matches = self.oracle.find_similar_failures(signature, top_k=1)
            if matches and matches[0].score >= self.similarity_threshold:
                recipe = self.oracle.get_recipe(matches[0].id)
        if recipe is None:
            return None
        if (
            self.recipe_min_success_rate is not None
            and recipe.times_applied >= self.recipe_reliability_min_occurrences
            and recipe.success_rate < self.recipe_min_success_rate
        ):
            return None
        return Fix.model_validate(recipe.fix_detail), recipe

    def _find_fix(
        self,
        signature: str,
        current_args: dict[str, Any],
        error_type: str,
        error_message: str,
        attempt_number: int,
        attempts: list[RecoveryAttempt],
    ) -> tuple[Fix | None, FixSource | None]:
        already_tried_recipe = any(a.source == "recipe" for a in attempts)
        if not already_tried_recipe:
            found = self._lookup_recipe_fix(signature)
            if found is not None:
                fix, _recipe_row = found
                return fix, "recipe"

        if self.reflect is None:
            return None, None

        context = FailureContext(
            tool_name=self.tool_name,
            args=current_args,
            error_type=error_type,
            error_message=error_message,
            signature=signature,
            attempt_number=attempt_number,
            previous_attempts=[a.fix for a in attempts],
        )
        return generate_fix(context, self.reflect), "reflection"

    # -- speculative branching (Phase 3) --------------------------------------

    def _add_candidate(
        self,
        candidates: list[_Candidate],
        seen_args: set[str],
        fix: Fix,
        source: FixSource,
        current_args: dict[str, Any],
        proxy_score: float | None,
    ) -> bool:
        """Apply `fix` and, if it applies cleanly and isn't a dupe of an
        already-collected candidate's REALIZED args, add it. Returns
        whether a candidate was added — used by `_find_fix_candidates` to
        know when to stop asking `reflect()` for more.

        Any of `_invalid_fix_reasons` (an `argument_patch` key, a
        `transforms[].argument`, or a `transforms[].transform` name that
        isn't real) is treated the same as any other reason a fix "doesn't
        apply cleanly" — excluded from the candidate batch entirely, same
        as a `TransformError` below. Unlike the single-candidate path
        (`WrappedAgent.invoke`), this doesn't need its own distinct
        `ResolutionStatus`/metric event: an excluded candidate here was
        never a "the call was attempted" event to begin with, exactly like
        an unapplicable transform already wasn't."""
        if self._invalid_fix_reasons(fix):
            return False
        try:
            applied_args = apply_fix(current_args, fix)
        except Exception:
            return False
        key = json.dumps(applied_args, sort_keys=True, default=str)
        if key in seen_args:
            return False
        seen_args.add(key)
        candidates.append(
            _Candidate(fix=fix, source=source, applied_args=applied_args, proxy_score=proxy_score)
        )
        return True

    def _find_fix_candidates(
        self,
        signature: str,
        current_args: dict[str, Any],
        error_type: str,
        error_message: str,
        attempt_number: int,
        attempts: list[RecoveryAttempt],
    ) -> list[_Candidate]:
        """Generate up to `num_branches` distinct candidates for this round:
        the known recipe first (if any, not already tried this call), then
        `reflect()` calls until the batch is full. Each reflect() call sees
        every candidate proposed so far THIS ROUND (via `previous_attempts`)
        so it naturally diversifies rather than repeating itself — this is
        scoped to the round only; cross-round memory is still just the real,
        executed `attempts`, exactly as Phase 1/2."""
        candidates: list[_Candidate] = []
        seen_args: set[str] = set()

        already_tried_recipe = any(a.source == "recipe" for a in attempts)
        if not already_tried_recipe:
            found = self._lookup_recipe_fix(signature)
            if found is not None:
                fix, recipe_row = found
                self._add_candidate(
                    candidates, seen_args, fix, "recipe", current_args, recipe_row.success_rate
                )

        if self.reflect is not None:
            while len(candidates) < self.num_branches:
                context = FailureContext(
                    tool_name=self.tool_name,
                    args=current_args,
                    error_type=error_type,
                    error_message=error_message,
                    signature=signature,
                    attempt_number=attempt_number,
                    previous_attempts=[a.fix for a in attempts] + [c.fix for c in candidates],
                )
                fix = generate_fix(context, self.reflect)
                added = self._add_candidate(
                    candidates, seen_args, fix, "reflection", current_args, None
                )
                if not added:
                    # reflect() proposed something that duplicates an
                    # existing candidate (or can't even apply) — stop
                    # asking rather than loop on a reflect() that keeps
                    # repeating itself.
                    break

        return candidates

    def _rank_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        """Any candidate with a recipe `proxy_score > 0.5` ranks first, by
        descending score (a real, persisted number — never fabricated for
        reflection-sourced candidates, which always have `proxy_score is
        None`). Everything else keeps generation order (recipe, then
        reflection calls in the order made) as a deterministic tie-break —
        NOT a claimed confidence ranking of the model's own proposals."""

        def rank_key(item: tuple[int, _Candidate]) -> tuple[int, float, int]:
            index, candidate = item
            is_good_recipe = candidate.proxy_score is not None and candidate.proxy_score > 0.5
            return (0 if is_good_recipe else 1, -(candidate.proxy_score or 0.0), index)

        return [c for _, c in sorted(enumerate(candidates), key=rank_key)]

    def _try_best_proxy_ranked(
        self, candidates: list[_Candidate]
    ) -> tuple[_Candidate, Any, Exception | None]:
        """side_effect_free=False (the default whenever num_branches>1):
        rank candidates WITHOUT calling the tool, then call the tool for
        real exactly once, with the top-ranked survivor. This is the
        structural guarantee that the default path never risks a duplicate
        real-world side effect, regardless of how large num_branches is."""
        best = self._rank_candidates(candidates)[0]
        result, error = self._call(best.applied_args)
        return best, result, error

    def _try_all_real(
        self, candidates: list[_Candidate]
    ) -> tuple[_Candidate | None, Any, list[tuple[_Candidate, str, str]]]:
        """side_effect_free=True: call the tool for real, once per
        candidate, in ranked order, until one FULLY passes invariants —
        "first fully-passing wins", not "best passing", since invariants
        are boolean and give no finer signal to rank passing candidates by.
        Returns (winning_candidate_or_None, winning_result_or_None, failed)
        where `failed` is every (candidate, error_type, error_message) that
        was actually tried and did NOT pass, in try order — ready for the
        caller to feed straight into `_on_attempt_failure`.

        May raise InvariantAbortError (propagated uncaught, exactly like
        the single-candidate path) if any candidate's real result violates
        an on_violation="abort" invariant; the outer `invoke()` try/except
        already handles that by marking the failure ABORTED and re-raising.
        """
        failed: list[tuple[_Candidate, str, str]] = []
        for candidate in self._rank_candidates(candidates):
            result, error = self._call(candidate.applied_args)
            classification = self._classify_failure(result, error)
            if classification is None:
                return candidate, result, failed
            error_type, error_message = classification
            failed.append((candidate, error_type, error_message))
        return None, None, failed

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self.oracle.close()

    def __enter__(self) -> WrappedAgent:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def wrap(
    agent: Any,
    invariants: list[Invariant] | None = None,
    oracle_path: str | Path = ".resilientforge",
    max_recovery_attempts: int = 3,
    tool_name: str | None = None,
    reflect: ReflectFn | None = None,
    similarity_threshold: float = 0.85,
    workflow_id: str | None = None,
    oracle: Oracle | None = None,
    enable_standing_guards: bool = True,
    guard_promotion_min_occurrences: int = 3,
    guard_promotion_min_success_rate: float = 0.8,
    num_branches: int = 1,
    side_effect_free: bool = False,
    isolate: bool = False,
    call_timeout: float | None = None,
    max_memory_mb: int | None = None,
    max_cpu_seconds: float | None = None,
    guard_demotion_min_occurrences: int = 3,
    guard_demotion_max_failure_rate: float = 0.5,
    recipe_min_success_rate: float | None = None,
    recipe_reliability_min_occurrences: int = 3,
    metrics: MetricsHook | None = None,
    valid_arguments: set[str] | None = None,
) -> WrappedAgent:
    """
    valid_arguments: set[str] | None = None
        See `WrappedAgent.__init__`'s docstring. Defaults to `None`
        (inferred from `agent`'s own signature when it's a plain callable
        with named parameters).

    metrics: MetricsHook | None = None (Phase 5)
        Optional observability hook — see `WrappedAgent.__init__`'s
        docstring and `telemetry/metrics.py` for the full design.

    guard_demotion_min_occurrences / guard_demotion_max_failure_rate (Phase 5)
        Auto-revokes a guard (via the same sticky `revoke()` a human
        would use) once it's fired at least `guard_demotion_min_occurrences`
        times and its failure rate exceeds `guard_demotion_max_failure_rate`.
        Always enabled — it only ever removes a demonstrably-failing
        guard, never a working one.

    recipe_min_success_rate / recipe_reliability_min_occurrences (Phase 5)
        Opt-in (default `None` = today's unconditional behavior). When
        set, a recipe whose `success_rate` has fallen below this floor
        (once applied at least `recipe_reliability_min_occurrences`
        times) is skipped as a fast-path candidate, falling through to
        reflection instead.

    side_effect_free: bool = False
        Vouches that `tool_fn` has no problematic real-world effect
        regardless of which arguments it's called with, and is therefore
        safe to actually invoke once per speculative candidate within a
        single recovery attempt (not just once with the eventual winner).

        This is NOT classic idempotency ("calling twice with the SAME
        input is a no-op the second time" — e.g. PUT). It is closer to
        HTTP's notion of a "safe" method (GET/HEAD): true for read-only
        lookups, pure computations, and validations; FALSE for anything
        that creates, sends, charges, books, or deletes for real, even if
        that operation is itself idempotent in the classic sense.

        Only meaningful when `num_branches > 1`. Default False: Phase 3
        never risks a duplicate real-world side effect unless you
        explicitly opt in per-tool.

    isolate: bool = False
        Runs every real `tool_fn` call in a freshly-spawned subprocess
        (Phase 4). A hang past `call_timeout`, or a crash, becomes a
        normal recoverable failure instead of taking down the host
        process — protective isolation of the CALLER, not of the world
        `tool_fn` touches: this does NOT, and cannot, undo a real-world
        side effect `tool_fn` already performed before it hung or
        crashed. Requires `tool_fn` to be picklable (checked eagerly at
        construction) — a locally-defined closure or lambda won't work.

    call_timeout: float | None = None
        Wall-clock seconds before an isolated call is terminated. Only
        enforced when `isolate=True`.

    max_memory_mb / max_cpu_seconds: int | float | None = None
        POSIX-only, best-effort resource ceilings applied inside the
        isolated subprocess. Only enforced when `isolate=True`; a no-op
        (with a warning) on Windows.

    See `WrappedAgent.__init__`'s docstring for the full detail behind
    each of these.
    """
    tool_fn = _resolve_callable(agent)
    resolved_oracle = oracle or Oracle(oracle_path)
    return WrappedAgent(
        tool_fn=tool_fn,
        tool_name=tool_name or _default_tool_name(agent, tool_fn),
        invariants=invariants or [],
        oracle=resolved_oracle,
        max_recovery_attempts=max_recovery_attempts,
        reflect=reflect,
        similarity_threshold=similarity_threshold,
        workflow_id=workflow_id,
        enable_standing_guards=enable_standing_guards,
        guard_promotion_min_occurrences=guard_promotion_min_occurrences,
        guard_promotion_min_success_rate=guard_promotion_min_success_rate,
        num_branches=num_branches,
        side_effect_free=side_effect_free,
        isolate=isolate,
        call_timeout=call_timeout,
        max_memory_mb=max_memory_mb,
        max_cpu_seconds=max_cpu_seconds,
        guard_demotion_min_occurrences=guard_demotion_min_occurrences,
        guard_demotion_max_failure_rate=guard_demotion_max_failure_rate,
        recipe_min_success_rate=recipe_min_success_rate,
        recipe_reliability_min_occurrences=recipe_reliability_min_occurrences,
        metrics=metrics,
        valid_arguments=valid_arguments,
    )
