"""Failure signature normalization/templating.

The crux of the whole project: raw failure data
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
is expected to need real iteration against the failure-injection suite —
let recovery-rate numbers be the judge of changes here, not intuition.

Two gaps found and fixed via a real-world validation exercise (see
`docs/real_world_validation_round2.md`), not by inspection:

1. **False-merge (over-collapsing)**: a quoted HTTP status line like
   `'403 Forbidden'` and `'402 Payment Required'` used to both collapse to
   the generic `<STR>` via `_redact_quoted` — the same mechanism that
   correctly collapses "next Friday"/"next Tuesday", but here it swallowed
   the one piece of information (the reason phrase, e.g. "Forbidden" vs.
   "Payment Required") that determines whether a fix is even possible. The
   fix preserves the whole quoted status line rather than collapsing it to
   `<STR>`; the numeric code itself is still independently redacted to
   `<NUM>` by the later decimal-number pass (same as any other number in
   the message) — discrimination comes from the reason phrase surviving,
   which is sufficient in practice since each standard HTTP status code has
   a distinct, unique reason phrase. Considered a general "short,
   structured, enum-like quoted token" rule instead of one narrowly scoped
   to HTTP status text, but rejected it: a general pattern loose enough to
   catch arbitrary enum-like tokens is also loose enough to accidentally
   preserve real free text that happens to fit the same shape (e.g. a
   quoted movie title like "500 Days of Summer" — digits followed by
   Capitalized Words, indistinguishable from a status line by pattern
   alone), which would under-redact content that should collapse just as
   much as the original bug over-redacted content that shouldn't have.
   `_HTTP_STATUS_TEXT_RE` is deliberately narrow instead: an honestly-scoped
   fix for the concrete case found, not a speculative general rule.
2. **Missed-match (under-collapsing)**: a hex byte literal like `0x8f` in
   a real `UnicodeDecodeError` message only had its leading `0` matched by
   `_NUMBER_RE` (a decimal-digit pattern) — the hex digits themselves
   passed through unredacted, so two structurally identical failures (a
   PDF's binary content failing UTF-8 decoding) produced different
   signatures purely because the specific byte value differed. Fixed by
   redacting whole hex literals (`_HEX_LITERAL_RE`) as their own pass,
   before the decimal-number pass runs.
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
# Hex literals (e.g. Python's `UnicodeDecodeError: ... byte 0x8f ...`) are
# NOT decimal digit runs — `_NUMBER_RE` only ever matches the leading "0",
# leaving the actual hex digits (which vary per byte value, but mean
# nothing structurally) unredacted. Must run BEFORE `_NUMBER_RE` so the
# whole literal is consumed as one unit first.
_HEX_LITERAL_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b")
_BARE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Deliberately narrow (see module docstring's "false-merge" note): an HTTP
# status line, e.g. "403 Forbidden", "402 Payment Required", "500 Internal
# Server Error" — a 3-digit status code (1xx-5xx) followed by one or more
# Capitalized words. Preserved rather than collapsed to <STR>, since the
# status code is exactly the information that determines whether a fix is
# even possible (a bot-detection 403 and a paywall 402 are different
# problems, not the same one with a different literal).
_HTTP_STATUS_TEXT_RE = re.compile(r"^[1-5]\d{2} [A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*$")

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
        # fix.
        return match.group(0)
    if _HTTP_STATUS_TEXT_RE.match(inner):
        # e.g. '403 Forbidden' vs '402 Payment Required' — see module
        # docstring's "false-merge" note and _HTTP_STATUS_TEXT_RE's comment.
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
    text = _HEX_LITERAL_RE.sub("<NUM>", text)
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
