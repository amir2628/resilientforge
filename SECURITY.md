# Security

## Reporting a vulnerability

Please open a GitHub issue (or, for anything sensitive, contact a maintainer
directly) describing the problem and, if possible, a reproduction. This is a
small alpha project without a formal disclosure program yet.

## `pickle`/`cloudpickle` usage (`core/isolation.py`)

`isolate=True` runs a real tool call in a freshly-spawned subprocess. To get
`tool_fn` into that subprocess, this module uses stdlib `pickle` (and, with the
optional `isolation` extra, `cloudpickle` as a fallback for closures/lambdas
that stdlib `pickle` can't handle).

`pickle.loads`/`cloudpickle.loads` can execute arbitrary code if used to
deserialize data from an untrusted source — this is why static analyzers
(e.g. `bandit`'s `B403`) flag any use of `pickle` at all. That risk does not
apply here: the bytes being deserialized are always produced moments earlier,
in the same run, by the same parent process, for its own child — never read
from a file, network socket, or any other externally-controlled input. The
serialize and deserialize sides are the same trust boundary.

## SQL construction (`oracle/store.py`)

`list_failures`/`list_guards` build their `WHERE` clause with an f-string
(flagged by `bandit`'s `B608`). The interpolated fragments are always
hardcoded column/operator strings (e.g. `"signature = ?"`, `"active = 1"`)
assembled by the code itself — every actual value (`signature`, `workflow_id`,
`tool_name`, `limit`) is bound through `sqlite3`'s `?` parameterization, never
string-interpolated. Not a SQL injection vector.
