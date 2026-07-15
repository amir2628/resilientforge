# Real-world validation, round 3: confirming the round 2 fixes

> **Addendum (2026-07-16):** the "unexpected finding" below (the same
> inert-fix problem reaching a live call via `fix.transforms` instead of
> `fix.argument_patch`) has been fixed and directly confirmed — see
> "Addendum: confirming the transforms-validation fix" at the end of this
> document. Not a new round4 — this closes out this round's own open
> finding.

**Purpose: confirmation, not exploration.** Rounds 1-2 found and this round's
Part A/B fixed three issues (see below). This round reuses round 2's exact
setup verbatim, pointed at the fixed source, to check whether the same real
cases that exposed the bugs are now handled correctly.

**Date run:** 2026-07-15, 2 sessions back-to-back.
**External agent:** same as rounds 1-2 —
[`langchain-ai/react-agent`](https://github.com/langchain-ai/react-agent),
commit `7d1f9832f56d6d29ad9ae248caf0b263c5460145`.
**Setup:** [`../validation/round3/README.md`](../validation/round3/README.md)
— identical to round 2's `tools.py`/`graph.py`/`prompts.py`, only the
installed `resilientforge` differs (fixed vs. pre-fix).
**Scope:** confirmation only. No further code changes made here — a new,
unexpected finding surfaced (below) and, per this round's own ground rules,
is reported and left unpatched rather than silently fixed.

## Fixes being confirmed

1. **Finding 1 (false-merge)**: `core/signature.py`'s `_redact_quoted` now
   preserves an HTTP status line's reason phrase instead of collapsing it
   to `<STR>` — a 403 and a 402 should no longer share a signature.
2. **Finding 2 (missed-match)**: `_HEX_LITERAL_RE` now redacts hex byte
   literals (e.g. `0x8f`) as a whole unit before the decimal-number pass —
   two PDF-decode failures differing only by byte value should now share
   a signature.
3. **Finding 3 (inert fix)**: `core/engine.py`/`integrations/langgraph_adapter.py`
   now reject a `Fix.argument_patch` key that isn't a real tool parameter,
   before it's ever applied to a live retry or persisted as a recipe — a
   `headers` patch on a `url`-only tool should no longer silently no-op and
   get recorded as "recovered."

## Result 1/3: false-merge fix — confirmed, and more strongly than expected

Across both sessions, `extract_url_content` hit **3 distinct real HTTP
status codes**, and all 3 got their own distinct signature:

| Status | Count | Source | Signature distinct from the others? |
|---|---|---|---|
| 403 Forbidden | 34 (10 + 24) | Wikipedia, w3.org (as before) | — |
| 406 Not Acceptable | 2 | huffpost.com (**new this round**, not seen in round 2) | Yes |
| 404 Not Found | 1 | a Japanese university PDF link (**new this round**) | Yes |

**Honest note, as instructed**: round 2's exact 402 (Le Monde's paywall)
did **not** recur this round — the Le Monde prompts either succeeded or
hit a different error this time (real-world traffic isn't guaranteed to
repeat, and this is reported plainly rather than treated as a pass by
default). The fix is still confirmed, just via different real evidence:
403/404/406 all correctly stayed distinct from each other, which is
actually stronger confirmation (a 3-way split, not just 2-way) that the
fix generalizes beyond the one original pair.

## Result 2/3: missed-match fix — directly confirmed on the identical case

Both real PDFs from round 2 recurred, and now share **one signature**:

| Byte | URL | Signature (abbreviated) |
|---|---|---|
| `0x8f` | `arxiv.org/pdf/2301.00001` (×2) | `...codec can't decode byte <NUM> in position <NUM>...` |
| `0x80` | the degrowth-policy PDF (×1) | `...codec can't decode byte <NUM> in position <NUM>...` |

All 3 identical. This is the cleanest possible confirmation: the exact
same two real files that exposed the bug now collapse correctly.

## Result 3/3: inert-fix rejection — the outcome is confirmed, the mechanism wasn't directly exercised

All 34 `403 Forbidden` occurrences resolved as **`exhausted`** — zero
`recovered`, and **zero recipes were ever created** for this signature.
Before the fix, this signature occasionally (and spuriously) resolved as
`recovered` via an inert `headers` argument_patch (round 2's Finding 3);
that no longer happens.

**Precisely what this does and doesn't prove, checked honestly**: no
`invalid_argument_patch` rejection event appears anywhere in either
session's metrics log — the local model simply didn't propose a `headers`-
style `argument_patch` for this signature during these two runs (model
proposals vary run to run). So the specific *rejection code path* wasn't
directly exercised this round. What **is** directly confirmed is the
outcome that mattered: this failure class can no longer spuriously resolve
as "recovered" the way it did before. A future session proposing that
exact patch again would be a stronger, more direct confirmation — this
round didn't happen to produce one.

## An unexpected finding: the same underlying bug, via a different mechanism

Per this round's ground rules — report and stop rather than patch further.

A **new** real failure this round (`search` tool, `DDGSException`: a
self-signed TLS certificate error from one of `ddgs`'s backend engines)
resolved as `recovered` after exactly one reflection attempt, with this
persisted recipe:

```json
{
  "strategy": "repair_common_json_errors",
  "argument_patch": {},
  "transforms": [{"argument": "verify_ssl", "transform": "coerce_bool"}]
}
```

`search(query: str)` has no `verify_ssl` parameter — same underlying
problem as Finding 3 (a proposed correction targeting an argument that
doesn't exist), but this time via `fix.transforms`, not
`fix.argument_patch`. **Part B's fix only validates `argument_patch` keys
— it does not check `transforms[].argument` against the tool's real
parameters.** `core/recovery.py`'s `apply_fix` already has a guard —
`if arg_transform.argument not in new_args: continue` — that silently
skips a transform whose argument isn't present in the current call's
args, so the retry ran with its original, unchanged `query` and merely
succeeded because the TLS error was transient (the same "recovered but
not because of a real fix" pattern as round 1/2, not a new failure mode,
just a new mechanism for it to slip through).

A second, compounding detail: `"coerce_bool"` isn't in `TRANSFORM_REGISTRY`
at all (only `parse_relative_date_to_iso`, `coerce_int`, `coerce_float`,
`coerce_str`, `repair_common_json_errors` are registered) — had
`"verify_ssl"` actually been present in `args`, this would have raised
`TransformError: unknown transform: 'coerce_bool'` instead. Because the
argument-presence check short-circuits first, that second problem never
even gets evaluated here.

**Not patched.** This is a real, live gap in Part B's fix, found during a
confirmation run, not invented — flagged for the user's review before any
further change to `core/engine.py`, exactly as this round's own rules ask.

## What this still doesn't tell us

- **The Finding 3 rejection mechanism itself wasn't directly triggered**
  this round (see above) — the confirmation is of the outcome, not a
  direct observation of the new code path firing.
- **The `transforms` gap found here is reported, not measured at scale** —
  one real occurrence. Whether it's common or rare in practice is unknown.
- **2 sessions, back-to-back, same day** — same scope-and-time caveats as
  rounds 1/2 apply.
- **This doesn't fix the new `transforms` gap.** Left for the user to
  decide how to proceed, per this round's explicit ground rules.

## Addendum (2026-07-16): confirming the transforms-validation fix

**The fix.** `WrappedAgent._invalid_fix_reasons` (`core/engine.py`) is now
the ONE shared check both live-application call sites go through
(`_attempt` for the single-candidate path, `_add_candidate` for Phase 3
speculative branching) — replacing the argument_patch-only check from
Part B. It rejects a fix, as a whole (never partially applying the valid
part), if ANY of the following hold: an `argument_patch` key isn't a real
tool parameter, a `transforms[].argument` isn't a real tool parameter, or
a `transforms[].transform` name isn't registered in `TRANSFORM_REGISTRY`
at all (round 3's `"coerce_bool"` observation) — using the same
`ResolutionStatus.FIX_REJECTED` path, not a new status. Confirmed this
matches `argument_patch`'s existing whole-fix-rejection behavior — no
inconsistency to flag; nothing in either path ever applied a fix
partially. 3 new regression tests reproduce round 3's exact `verify_ssl`
case, an unregistered-transform-against-a-real-argument case, and confirm
the genuinely legitimate natural-language-date path still recovers
normally. Full suite (264 tests), ruff, and bandit all clean.

**Confirmation session** (not a new round4): `validation/round3/`'s exact
setup, reinstalled against the fixed source, run once more — deliberately
*without* resetting the existing oracle, so the already-persisted bad
`verify_ssl` recipe from the original round 3 sessions was still sitting
there as a live test of whether an old, pre-fix recipe gets caught on
replay, not just a fresh proposal.

**Result: the new rejection path fired for real, 5 separate times**,
against genuinely fresh model-proposed fixes (all for the `403 Forbidden`
signature, all `source: "reflection"`, all 3 attempts per episode
rejected) — 5 failures now show `resolution_status: fix_rejected_invalid_argument`,
`fix_applied: null` (never persisted), and the recipe count stayed at 2
(no new bad recipe created). This is a direct, live observation of the
new code path engaging — not an inferred absence like round 3's original
Result 3/3.

**One thing this run did *not* directly confirm**: the pre-existing bad
`verify_ssl` recipe was never replayed, because no new `DDGSException`
recurred this session (real-world traffic isn't guaranteed to repeat —
same honest caveat as everywhere else in this exercise). So while the
mechanism is confirmed to reject *fresh* invalid proposals, this run
doesn't directly show an *old, already-persisted* invalid recipe being
caught on replay specifically — though there's no structural reason to
expect it to behave differently, since the same check runs regardless of
whether a fix's source is `"reflection"` or `"recipe"`.

## What the addendum still doesn't tell us

- Same limitations as the rest of this document — one confirmation
  session, one local model, one external agent.
- The old bad recipe's rejection-on-replay specifically wasn't observed
  directly (see above) — only reasoned about structurally.
