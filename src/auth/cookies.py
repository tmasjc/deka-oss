"""Session-cookie payload helpers.

The actual signed-cookie machinery is provided by Starlette's
:class:`starlette.middleware.sessions.SessionMiddleware`, which is
installed in :mod:`src.web_api.app`. This module just centralises
the cookie name and the payload shape so the route handlers and the
dependency don't have to repeat themselves.

Payload shape: ``{"user_id": "<stable wire id>"}``. Beta-grade — no
expiry, no rotation tooling. If a user's id is removed from
``users.yaml`` after a cookie was issued, the dependency in
:mod:`src.auth.dependencies` rejects the request with 401.
"""

from __future__ import annotations

from typing import Any

# Bound to the deployment, not the user. Changing it invalidates every
# in-flight cookie; do that deliberately (e.g. on a security incident).
COOKIE_NAME = "deka_session"


def session_payload(user_id: str) -> dict[str, Any]:
    """Build the dict to merge into ``request.session`` on login."""
    return {"user_id": user_id}


def user_id_from_session(session: dict[str, Any] | None) -> str | None:
    """Extract the ``user_id`` from a session dict, or ``None`` if not
    set or the dict is missing."""
    if not session:
        return None
    value = session.get("user_id")
    return value if isinstance(value, str) and value else None


__all__ = ["COOKIE_NAME", "session_payload", "user_id_from_session"]
