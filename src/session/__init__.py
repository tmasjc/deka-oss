"""Pure-Python session-state package consumed by the web API and core modules."""

from __future__ import annotations

from src.session.state import SessionState, TurnRecord

__all__ = ["SessionState", "TurnRecord"]
