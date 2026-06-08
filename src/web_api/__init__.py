"""FastAPI web API for the Deka query agent.

Wraps the existing :mod:`src.search`, :mod:`src.reflection`, and
:mod:`src.logging` modules in HTTP endpoints for a browser-based UI.
The shared session model lives in :class:`src.session.state.SessionState`.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
