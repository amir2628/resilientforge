"""The Invariant interface.

Two supported kinds for Phase 1:
1. Deterministic — a plain Python predicate (the base constructor), or
   Pydantic model validation (`Invariant.from_pydantic_model`).
2. LLM-judged — a short natural-language rule evaluated by a model call
   (`Invariant.llm_judged`). The actual model call is injected as a
   `judge` callable rather than hardcoded to one vendor, which also
   keeps this testable without a real API call.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError

OnViolation = Literal["recover", "abort", "warn"]
Severity = Literal["low", "medium", "high"]

InvariantCheck = Callable[[Any], bool]
JudgeFn = Callable[[str, Any], bool]


class Invariant(BaseModel):
    name: str
    check: InvariantCheck
    on_violation: OnViolation = "recover"
    severity: Severity = "medium"

    def evaluate(self, result: Any) -> bool:
        return bool(self.check(result))

    @classmethod
    def from_pydantic_model(
        cls,
        name: str,
        model: type[BaseModel],
        on_violation: OnViolation = "recover",
        severity: Severity = "medium",
    ) -> Invariant:
        """Deterministic invariant: `result` (a dict, or an object pydantic
        can coerce) must validate against `model`."""

        def _check(result: Any) -> bool:
            try:
                model.model_validate(result)
            except ValidationError:
                return False
            return True

        return cls(name=name, check=_check, on_violation=on_violation, severity=severity)

    @classmethod
    def llm_judged(
        cls,
        name: str,
        rule: str,
        judge: JudgeFn,
        on_violation: OnViolation = "recover",
        severity: Severity = "medium",
    ) -> Invariant:
        """LLM-judged invariant: `judge(rule, result)` evaluates whether
        `result` satisfies the natural-language `rule`. `judge` is the
        caller's model-call implementation (or a mock/stub in tests) — this
        class has no opinion on which model provider it uses."""

        def _check(result: Any) -> bool:
            return bool(judge(rule, result))

        return cls(name=name, check=_check, on_violation=on_violation, severity=severity)


# -- built-in common invariants ----------------------------------------------


def not_none(name: str = "not_none", **kwargs: Any) -> Invariant:
    return Invariant(name=name, check=lambda result: result is not None, **kwargs)


def is_instance_of(
    types: type | tuple[type, ...],
    name: str | None = None,
    **kwargs: Any,
) -> Invariant:
    label = types.__name__ if isinstance(types, type) else "|".join(t.__name__ for t in types)
    return Invariant(
        name=name or f"is_instance_of_{label}",
        check=lambda result: isinstance(result, types),
        **kwargs,
    )
