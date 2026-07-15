# Real-world validation, round 2: does signature normalization discriminate different real failures, not just recognize a repeated one?

> **Round 3 addendum (2026-07-15):** Findings 1 and 2 (the false-merge and
> missed-match below) were fixed in `core/signature.py` and confirmed
> against these exact real cases (403 vs. 402/404/406, and the same two
> real PDFs) in [`real_world_validation_round3.md`](./real_world_validation_round3.md).
> Finding 3 (the inert `argument_patch`) was also fixed, but round 3
> surfaced a related, still-open gap: the same problem can occur via
> `Fix.transforms` instead, which the fix doesn't cover — reported there,
> not yet patched.

**Date run:** 2026-07-15, 3 sessions within a ~2-hour window (reduced from
the originally-planned 5, spread across different times/days, by explicit
user request while session 1 was already running — see "What this still
doesn't tell us").
**External agent:** [`langchain-ai/react-agent`](https://github.com/langchain-ai/react-agent),
commit `7d1f9832f56d6d29ad9ae248caf0b263c5460145` (same commit as round 1,
deliberately, to isolate this round's changes from upstream drift).
**Setup, exact deviations, reproduction:** [`../validation/round2/README.md`](../validation/round2/README.md).
**Relationship to round 1:** [`real_world_validation.md`](./real_world_validation.md)
proved cross-session *recurrence recognition* on one failure shape (search
timeouts) but couldn't test *discrimination* between different real
shapes, since only one ever occurred. This round adds a second, real,
undefensive tool (URL content extraction) specifically to widen the real
failure surface, plus reduces reliance on volume alone. Working trees and
docs are kept fully separate, per the task's own instruction — this is not
merged with round 1's findings.
**Scope:** measurement only. No changes to `core/signature.py`,
`tests/failure_injection`, or `validation/` (round 1). Findings below are
reported and flagged, not silently patched.

## Setup: 62 prompts, 2 real tools, 3 sessions

`search` (DuckDuckGo, same as round 1) and a new `extract_url_content`
(real `httpx` fetch + `BeautifulSoup4` text extraction, strict UTF-8
decode, no engineered failure or engineered robustness) wrapped through
the same `make_tool_node` path. 62 new prompts (22 search-only, 22
extraction-only with real URLs, 18 requiring both tools in sequence),
covering obscure topics, non-English prompts/pages (French, German,
Japanese, Esperanto), one- and two-word prompts, and one deliberately
rambling one. 3 sessions × 62 prompts = 186 real agent runs.

## Headline result: real, organic diversity, no engineering needed

Across 3 sessions, **exactly 4 distinct real oracle signatures** occurred
(all from `extract_url_content` — `search` had zero failures across all
186 runs this round, in contrast to round 1 where search's transient
timeout was the *only* failure ever seen — a useful reminder that which
tool dominates the failure count is circumstantial, not inherent to
either tool):

| Signature (abbreviated) | Count | Real cause |
|---|---|---|
| `HTTPStatusError \| Client error <STR> for url ...` | 43 | Mostly Wikipedia/w3.org **403 Forbidden** (real, consistent bot-detection — not a fingerprinting artifact of a missing User-Agent, confirmed one is set) |
| `UnicodeDecodeError \| ... byte <NUM>x8f ...` | 3 | The same real arXiv PDF (`arxiv.org/pdf/2301.00001`), same byte, all 3 sessions |
| `UnicodeDecodeError \| ... byte <NUM>x80 ...` | 1 | A **different** real PDF (a degrowth-policy PDF the model found itself via search) — same underlying problem as the row above |
| `ConnectTimeout \| ... handshake operation timed out ...` | 1 | A real TLS handshake timeout against a site the model guessed the URL for on its own (`modern-physics.org`) |

48 real recovery episodes total; **47 exhausted, 1 recovered** (the lone
recovery is examined below — it's not what it looks like).

## Finding 1 (the main point of round 2): a real false-merge

**Two genuinely different real problems collapsed into the identical
signature.** Of the 43 `HTTPStatusError` failures, 42 were real `403
Forbidden` responses (bot-detection blocking the request) — but **one was
a real `402 Payment Required`** (`https://www.lemonde.fr/actualite-en-continu/`,
almost certainly a paywall). A human would immediately call these
different underlying problems: one is "the server thinks you're a bot,"
the other is "the content requires payment," and no fix that could ever
help the first (a different `User-Agent`, different headers) could
possibly help the second. Both produced the identical oracle signature:

```
tool:extract_url_content|error_type:HTTPStatusError|error:Client error <STR> for url '<URL>
For more information check: <URL>|args:{url:<URL>}
```

**Root cause, read directly from `core/signature.py`**: the actual HTTP
status text (`'403 Forbidden'` vs. `'402 Payment Required'`) is inside
quotes in the error message, and `_redact_quoted` collapses any quoted
string containing a space down to `<STR>` — exactly the mechanism designed
to stop "next Friday" vs. "next Tuesday" from producing different
signatures (see round 1). Here it also swallows the one piece of
information (the status code) that actually determines whether a fix is
even *possible*. **Not opened as a signature.py fix** — flagging per the
task's own rule, for review before any change.

**A concrete cost of this, not just a theoretical one**: the persisted
recipe for this merged signature gets replayed against the `402` case too
whenever it recurs — a fix that could, at best, only ever help the `403`
population gets tried against a `402` it can never help, and the
recipe's dismal `success_rate=0.025` (1 success in 40 applications)
partly reflects two different populations being averaged together, not
"this fix rarely works."

## Finding 2: a real missed-match (the same problem, split in two)

The opposite error, also real: the 3 arXiv-PDF failures and the 1
degrowth-PDF failure are, by any reasonable human judgment, **the exact
same underlying problem** — asking `extract_url_content` to treat a real
PDF's binary bytes as UTF-8 text, which fails at the very first non-ASCII
byte. Both errors even fail at the identical position (`position 10` —
consistent with both PDFs sharing a standard ASCII header of the same
length before their compressed binary content begins). But they produced
**two different signatures**, because the specific byte value that
triggered the failure (`0x8f` vs. `0x80`) is embedded directly in Python's
`UnicodeDecodeError` message and isn't fully redacted:

```
... codec can't decode byte <NUM>x8f in position <NUM>: invalid start byte
... codec can't decode byte <NUM>x80 in position <NUM>: invalid start byte
```

**Root cause**: `core/signature.py`'s numeric redaction
(`_NUMBER_RE = re.compile(r"-?\b\d+\.\d+|-?\b\d+")`) matches runs of
*decimal* digits. In `0x8f`, it matches only the leading `0` (stopping at
the literal `x`, which isn't a digit) — the hex digits `8f`/`80`
themselves pass through unredacted as part of what looks like ordinary
text, because they follow a non-digit character the regex doesn't treat
as part of the same token. Position numbers in the same message (plain
decimal, e.g. `10`) redact correctly — this is specifically a hex-literal
gap, not a general numeric-redaction failure. **Not patched** — same rule
as Finding 1.

A concrete cost here too: the 3 *identical* arXiv-PDF occurrences (same
URL, same byte, one per session) never once benefited from a fast path —
each independently ran a full 3-attempt reflection cycle before
exhausting, even though it was, in every literal sense, the exact same
failure as before. This particular repeat case's signature *was*
identical across all 3 occurrences (same byte, so no split there) — the
real cost of the split shows up between the arXiv PDF and the *other*
real PDF, which never had a chance to share a signature (and thus a
recipe) despite being the same conceptual problem.

## An honest third observation (not a signature.py finding, but worth flagging clearly)

The **one** episode that did resolve as "recovered" is worth examining
closely, in the same spirit as round 1's "was the fix real, or lucky?"
check. The persisted recipe's fix is `argument_patch: {"headers": {...}}`
— the model proposing to retry with different HTTP headers. **This fix is
structurally inert**: `extract_url_content(url: str)`'s actual signature
takes only `url`. Confirmed directly:

```python
wrapped_tool.ainvoke({"url": "...", "headers": {...}})
# -> still raises the real HTTPStatusError; "headers" is silently dropped,
#    never reaches the HTTP request at all (no error, no effect)
```

So the fix that got "recovered" once (and replayed 39 more times at a
2.5% success rate) can **never have had any real effect** — its one
success and Wikipedia's other 403s the same session are consistent with
Wikipedia's own bot-detection being probabilistic (the identical URL,
identical request, sometimes returns real content and sometimes a 403).
This is the same underlying pattern round 1 found (a "fix" that happens
to coincide with success due to real-world flakiness, not genuine
correction) — but sharper here, since the fix isn't just *empty*, it's
*silently discarded by the tool-calling layer itself*, with no error or
warning that the proposed correction never had anywhere to go. This is
arguably a gap in Fix-application validation (nothing checks that an
`argument_patch` key corresponds to a real tool parameter before
persisting/replaying it as a "working" recipe) — not a `core/signature.py`
issue, and **not patched**, flagged for the same reason as Findings 1–2.

## The metric round 2 set out to get

- **Distinct human-judged failure shapes**: 4 — (a) bot-detection block,
  (b) paywall, (c) PDF/binary content isn't extractable text — one shape,
  even though 2 different real PDFs produced it.
  (d) TLS handshake timeout.
- **Distinct computed signatures**: also 4 — but this number matching (a)
  is a coincidence, not evidence of correctness: shapes (a) and (b) were
  wrongly *merged* (over-collapsed) into one signature, while the two
  instances of shape (c) were wrongly *split* (under-collapsed) into two.
  Two independent, opposite-direction errors happened to cancel out in
  the raw count — a superficial "4 shapes, 4 signatures" read would miss
  both.
- Shape (d) was correctly isolated (its own signature, no merge or split
  observed — though it only occurred once, so this is a weaker result
  than (a)/(c)'s multiple-occurrence evidence).

## What this still doesn't tell us

- **Sessions were 3, within ~2 hours, not 5 spread across days** — an
  explicit, mid-run scope reduction, not an oversight. This says nothing
  about longer-horizon drift (does Wikipedia's block rate change by time
  of day? does a recipe's success rate shift over days?).
- **Both real discrimination findings came from one tool
  (`extract_url_content`)** — `search` had zero failures this round, so
  this round says nothing new about *search*'s signature discrimination
  specifically, only about the new tool's.
- **Sample size per shape is still small.** Shape (b) (paywall) and shape
  (d) (TLS timeout) each occurred exactly once — enough to prove the
  *signature* they got, not enough to know if that signature is stable
  across repeats.
- **This doesn't fix anything.** Both signature.py findings and the
  Fix-validation observation are reported for review, not patched — per
  the task's explicit rule, and consistent with round 1's discipline.
- **One external tool implementation, one set of real URLs, one local
  model.** A different (more or less defensive) extraction tool
  implementation, a different URL set, or a hosted model's reflection
  quality could all surface a different mix of real failures than this
  specific run happened to hit.
