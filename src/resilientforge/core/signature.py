"""Failure signature normalization/templating.

The crux of the whole project (PROJECT_SPEC.md §10): raw failure data
(args, error messages) is full of run-specific literals — dates, ids, user
text — that would otherwise prevent two structurally identical failures
from producing the same signature and matching in the oracle. This module
strips literal values down to type placeholders (e.g. `<STR>`, `<DATE>`)
while preserving *structure* (argument names, nesting, error type), so
"next Friday" and "next Tuesday" date-format failures normalize to the same
signature, but a failure shaped differently (a different missing field, a
different argument type) does not.

Two independent normalization passes feed the final signature:
- `normalize_error_message`: regex-based redaction of literals in free-text
  error messages (quoted strings, dates, uuids, emails, urls, numbers).
- `normalize_args`: recursive, type-based templating of the structured
  tool-call arguments. Dict *keys* are structural (kept as-is); dict/list
  *values* are templated by type, since the value's specific type (and, for
  strings, whether it looks like a date/uuid/email/url) is what determines
  whether a fix generalizes — the literal value itself never does.

This is intentionally simple regex/type-based matching, not semantic
understanding of the error text. Widening or narrowing this normalization
is expected to need real iteration against the failure-injection suite
(§7.3) — let recovery-rate numbers be the judge of changes here, not
intuition (§10).
"""

from __future__ import annotations

import re
from typing import Any

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# No trailing \b: a unit-suffixed number like "30s" or "45ms" has no word
# boundary between the digits and the following letter, but the numeric
# part is still the literal that needs redacting (leaving "<NUM>s"). The
# leading \b is kept so digits embedded in an identifier (e.g. "v2") are
# left alone, since there's no boundary between a preceding letter and a
# digit either.
_NUMBER_RE = re.compile(r"-?\b\d+\.\d+|-?\b\d+")
_BARE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Anchored variants for classifying a whole string value (args are typed
# leaves, not free text, so we classify the entire value rather than
# scanning for embedded substrings).
_UUID_FULL_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_DATETIME_FULL_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"
)
_DATE_FULL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMAIL_FULL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_URL_FULL_RE = re.compile(r"^https?://\S+$")


def _redact_quoted(match: re.Match[str]) -> str:
    inner = match.group(0)[1:-1]
    if _BARE_IDENTIFIER_RE.match(inner):
        # A single bare token (no spaces/punctuation) inside quotes, e.g.
        # "missing required field 'attendees'", is far more likely to be a
        # structural field/parameter name than a literal value — real
        # user-entered values ("next Friday", a search query, a name) tend
        # to contain spaces or punctuation. Preserve it so two failures
        # differing only in *which* field is missing stay distinct
        # signatures. This is a heuristic, not a guarantee: when genuinely
        # ambiguous, erring toward NOT collapsing is the safer default,
        # since a wrongly-collapsed signature risks replaying an unrelated
        # fix (PROJECT_SPEC.md §10).
        return match.group(0)
    return "<STR>"


def normalize_error_message(message: str | None) -> str:
    """Redact literal values from a free-text error message.

    Order matters: more specific patterns (uuid, datetime, date, email, url)
    run before generic quoted-string collapsing, so e.g. a quoted ISO date
    resolves to `<DATE>` rather than the generic `<STR>`. Remaining
    standalone numbers (e.g. "line 1 column 5") are redacted last.
    """
    if not message:
        return ""
    text = message
    text = _UUID_RE.sub("<UUID>", text)
    text = _DATETIME_RE.sub("<DATETIME>", text)
    text = _DATE_RE.sub("<DATE>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _URL_RE.sub("<URL>", text)
    text = _QUOTED_RE.sub(_redact_quoted, text)
    text = _NUMBER_RE.sub("<NUM>", text)
    return text


def _classify_string(value: str) -> str:
    if _UUID_FULL_RE.match(value):
        return "<UUID>"
    if _DATETIME_FULL_RE.match(value):
        return "<DATETIME>"
    if _DATE_FULL_RE.match(value):
        return "<DATE>"
    if _EMAIL_FULL_RE.match(value):
        return "<EMAIL>"
    if _URL_FULL_RE.match(value):
        return "<URL>"
    return "<STR>"


def normalize_value(value: Any) -> str:
    """Recursively template a single value down to its structural shape.

    Dict keys are preserved (and sorted, for determinism) since argument
    names are structural. Dict/list *values*, and list elements, are
    templated by type — the literal never survives.
    """
    if value is None:
        return "<NULL>"
    if isinstance(value, bool):
        # bool is a subclass of int in Python — must check before int.
        return "<BOOL>"
    if isinstance(value, int):
        return "<INT>"
    if isinstance(value, float):
        return "<FLOAT>"
    if isinstance(value, str):
        return _classify_string(value)
    if isinstance(value, dict):
        items = ", ".join(
            f"{key}:{normalize_value(val)}" for key, val in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        # Collapse to the sorted set of templated element shapes so that
        # two lists differing only in length or in the order/repetition of
        # same-typed elements normalize identically.
        templated = sorted({normalize_value(item) for item in value})
        return "[" + ",".join(templated) + "]"
    return f"<{type(value).__name__.upper()}>"


def normalize_args(args: dict[str, Any] | None) -> str:
    """Template a tool call's arguments dict. Thin wrapper over
    `normalize_value` for a clearer public name at call sites."""
    return normalize_value(args or {})


def build_signature(
    *,
    tool_name: str,
    error_type: str | None = None,
    error_message: str | None = None,
    args: dict[str, Any] | None = None,
) -> str:
    """Build the normalized failure signature used as the oracle lookup key
    and as the text embedded for semantic similarity search.

    Deterministic: identical (tool_name, error_type, normalized error
    shape, normalized args shape) always produces the identical string,
    regardless of literal values or dict/list ordering.
    """
    parts = [f"tool:{tool_name}"]
    if error_type:
        parts.append(f"error_type:{error_type}")
    normalized_message = normalize_error_message(error_message)
    if normalized_message:
        parts.append(f"error:{normalized_message}")
    parts.append(f"args:{normalize_args(args)}")
    return "|".join(parts)
