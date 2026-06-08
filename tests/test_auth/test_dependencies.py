"""Tests for ``src.auth.dependencies.current_user``.

Pure unit tests — no FastAPI app, no HTTP. Build a tiny stand-in for
``Request`` carrying just ``session`` and ``app.state.user_registry``,
which is all the dependency reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from src.auth.config import User, UserRegistry
from src.auth.dependencies import current_user


def _registry(*users: User) -> UserRegistry:
    return UserRegistry(users=tuple(users))


def _alice() -> User:
    return User(
        id="alice",
        token_sha256="a" * 64,
        allowed_scopes=("Parent Full",),
    )


def _request(
    *,
    session: dict[str, Any] | None,
    registry: UserRegistry | None,
) -> Any:
    """Minimal ``Request`` stub.

    ``current_user`` only reads ``request.session`` and
    ``request.app.state.user_registry``; everything else is irrelevant.
    """
    return SimpleNamespace(
        session=session,
        app=SimpleNamespace(state=SimpleNamespace(user_registry=registry)),
    )


def test_returns_user_when_session_payload_matches_registry() -> None:
    alice = _alice()
    req = _request(
        session={"user_id": "alice"},
        registry=_registry(alice),
    )
    assert current_user(req) is alice


def test_raises_401_when_session_is_missing() -> None:
    req = _request(session=None, registry=_registry(_alice()))
    with pytest.raises(HTTPException) as exc_info:
        current_user(req)
    assert exc_info.value.status_code == 401
    assert "Not authenticated" in exc_info.value.detail


def test_raises_401_when_session_lacks_user_id() -> None:
    req = _request(session={}, registry=_registry(_alice()))
    with pytest.raises(HTTPException) as exc_info:
        current_user(req)
    assert exc_info.value.status_code == 401


def test_raises_401_when_session_user_id_blank() -> None:
    req = _request(session={"user_id": "   "}, registry=_registry(_alice()))
    with pytest.raises(HTTPException) as exc_info:
        current_user(req)
    assert exc_info.value.status_code == 401


def test_raises_401_when_user_was_removed_from_registry() -> None:
    """The cookie still references ``alice`` but the operator removed
    her from users.yaml — treat that as a logged-out state."""
    req = _request(
        session={"user_id": "alice"},
        registry=_registry(),  # empty registry
    )
    with pytest.raises(HTTPException) as exc_info:
        current_user(req)
    assert exc_info.value.status_code == 401
    assert "Unknown user" in exc_info.value.detail
