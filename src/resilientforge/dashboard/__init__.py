"""Local, read-only dashboard over one oracle (Phase 4). Optional extra
— `pip install resilientforge[dashboard]` — never imported by
`resilientforge/__init__.py`, see `dashboard/app.py`'s module docstring.
"""

from __future__ import annotations

from resilientforge.dashboard.app import create_app

__all__ = ["create_app"]
