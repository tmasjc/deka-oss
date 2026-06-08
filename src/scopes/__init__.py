"""Scope registry: maps human-readable scope names to Milvus collections.

Each session at start picks one ``Scope``; the chosen scope's
``milvus_collection`` overrides ``SearchConfig.collection`` for that session.
The registry is loaded once at process start from ``scopes.yaml``
(see :func:`load_scopes`) and shared across sessions.
"""

from .config import (
    Scope,
    ScopeError,
    ScopeRegistry,
    load_scopes,
)

__all__ = [
    "Scope",
    "ScopeError",
    "ScopeRegistry",
    "load_scopes",
]
