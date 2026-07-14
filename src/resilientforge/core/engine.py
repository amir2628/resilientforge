"""The wrap() orchestration: tying the oracle (oracle/), signature
normalization (core/signature.py), invariants (core/invariants.py), and fix
generation/application (core/recovery.py) into the recovery flow described
in PROJECT_SPEC.md §4.4.

Scope note on "zero configuration" (§4.1): that promise covers passive
fast-path recovery from recipes already in the oracle. Active,
reflection-based fix generation needs a `reflect` callable — this module
has no default implementation (mirrors core/recovery.py staying
vendor-neutral); a real Anthropic-backed default belongs in
integrations/raw_tool_loop.py (step 7), which already needs Anthropic
wiring for the tool-calling loop itself. Without `reflect`, wrap() still
recovers from anything a prior run already learned, but a genuinely novel
failure shape exhausts immediately instead of attempting reflection.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from resilientforge.core.invariants import Invariant
from resilientforge.core.recovery import FailureContext, Fix, ReflectFn, apply_fix, generate_fix
from resilientforge.core.signature import build_signature
from resilientforge.oracle import Oracle, ResolutionStatus
from resilientforge.oracle.recipes import RecipeManager

FixSource = Literal["recipe", "reflection"]


@dataclass
class RecoveryAttempt:
    fix: Fix
    source: FixSource
    error_type: str | None = None
    error_message: str | None = None


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
    ) -> None:
        self.tool_fn = tool_fn
        self.tool_name = tool_name
        self.invariants = invariants
        self.oracle = oracle
        self.recipes = RecipeManager(oracle)
        self.max_recovery_attempts = max_recovery_attempts
        self.reflect = reflect
        self.similarity_threshold = similarity_threshold
        self.workflow_id = workflow_id

    # -- the recovery loop -------------------------------------------------

    def invoke(self, **kwargs: Any) -> Any:
        current_args = dict(kwargs)
        result, error = self._call(current_args)
        classification = self._classify_failure(result, error)  # may raise InvariantAbortError
        if classification is None:
            return result
        error_type, error_message = classification
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
                fix, source = self._find_fix(
                    signature, current_args, error_type, error_message, attempt_number, attempts
                )
                if fix is None:
                    break  # no recipe match and no `reflect` configured — nothing left to try

                new_args, retry_result, retry_error = self._attempt(current_args, fix)
                retry_classification = self._classify_failure(retry_result, retry_error)

                if retry_classification is None:
                    self.recipes.record_success(
                        signature=signature,
                        tool_name=self.tool_name,
                        fix_detail=fix.model_dump(),
                        root_cause=fix.root_cause,
                        fix_strategy=fix.strategy,
                    )
                    self.oracle.update_failure_resolution(
                        failure.id,
                        ResolutionStatus.RECOVERED,
                        fix_applied=fix.model_dump(),
                        fix_verified=True,
                    )
                    return retry_result

                retry_error_type, retry_error_message = retry_classification
                attempts.append(
                    RecoveryAttempt(
                        fix=fix,
                        source=source,
                        error_type=retry_error_type,
                        error_message=retry_error_message,
                    )
                )
                if source == "recipe":
                    self.recipes.record_fast_path_failure(signature)
                current_args = new_args
        except InvariantAbortError:
            self.oracle.update_failure_resolution(failure.id, ResolutionStatus.ABORTED)
            raise

        self.oracle.update_failure_resolution(failure.id, ResolutionStatus.EXHAUSTED)
        raise RecoveryExhaustedError(
            tool_name=self.tool_name,
            call_args=kwargs,
            original_error=original_error,
            attempts=attempts,
        ) from original_error

    # -- helpers -------------------------------------------------------------

    def _call(self, args: dict[str, Any]) -> tuple[Any, Exception | None]:
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
            recipe = self.oracle.get_recipe(signature)
            if recipe is None:
                matches = self.oracle.find_similar_failures(signature, top_k=1)
                if matches and matches[0].score >= self.similarity_threshold:
                    recipe = self.oracle.get_recipe(matches[0].id)
            if recipe is not None:
                return Fix.model_validate(recipe.fix_detail), "recipe"

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
) -> WrappedAgent:
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
    )
