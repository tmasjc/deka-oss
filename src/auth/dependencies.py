"""FastAPI dependencies for the auth layer.

Every protected route declares ``user: User = Depends(current_user)``;
the dependency resolves the calling cookie session to a :class:`User`
or raises :class:`HTTPException(401)`. The registry itself lives on
``app.state.user_registry`` (set in :mod:`src.web_api.app`).
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from .config import User, UserAuthError, UserRegistry
from .cookies import user_id_from_session


def _registry_from_request(request: Request) -> UserRegistry:
    """Fetch the registry off ``app.state``.

    Mirrors ``app.state.scopes`` access elsewhere in the web API.
    """
    registry: UserRegistry | None = getattr(
        request.app.state, "user_registry", None
    )
    if registry is None:  # pragma: no cover — bootstrap misconfiguration
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registry not initialised",
        )
    return registry


def current_user(request: Request) -> User:
    """Resolve the current user, or raise 401.

    Looks up the cookie session payload first; missing / malformed
    payload → 401. If the cookie payload references an id that has
    since been removed from ``users.yaml``, that's also a 401 — the
    operator deactivated the user, so the cookie is no longer valid.
    """
    user_id = user_id_from_session(getattr(request, "session", None))
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    registry = _registry_from_request(request)
    try:
        return registry.get(user_id)
    except UserAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown user",
        )


__all__ = ["current_user"]
