"""Read-only replay of past tuning sessions.

Loads a session's canonical progress log (``runs/<id>.jsonl``) and
diagnostic sidecar (``runs/<id>.details.jsonl``) into in-memory
``ReplayTurn`` records that the TUI can step through without any
Milvus or LLM connection.
"""

from .loader import (
    ReplayLoadError,
    ReplaySession,
    ReplayTurn,
    load_session,
)

__all__ = [
    "ReplayLoadError",
    "ReplaySession",
    "ReplayTurn",
    "load_session",
]
