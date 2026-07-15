"""ResilientForge: persistent, cross-run failure memory for tool-calling agents."""

from importlib.metadata import version as _installed_version

from resilientforge.core.engine import (
    InvariantAbortError,
    RecoveryAttempt,
    RecoveryExhaustedError,
    WrappedAgent,
    wrap,
)
from resilientforge.core.invariants import Invariant
from resilientforge.core.isolation import IsolationError
from resilientforge.oracle.guards import GuardManager, StandingGuard

# Read from installed package metadata (which hatchling derives from
# pyproject.toml's `version` at build/install time — including editable
# installs) rather than a second hardcoded string here. Found this was
# necessary by actually testing a fresh install after the Phase 5
# version bump: this string had silently drifted out of sync with
# pyproject.toml's, exactly the kind of two-sources-of-truth bug this
# fixes structurally, not just for this one release.
__version__ = _installed_version("resilientforge")

__all__ = [
    "wrap",
    "Invariant",
    "WrappedAgent",
    "RecoveryAttempt",
    "RecoveryExhaustedError",
    "InvariantAbortError",
    "IsolationError",
    "GuardManager",
    "StandingGuard",
]
