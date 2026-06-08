"""Tests for ``/api/auth/login``, ``/api/auth/logout``, ``/api/auth/me``."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.scopes import Scope, ScopeRegistry
from src.web_api.app import create_app


_PLAINTEXT_ALICE = "a-secret-token-for-alice"
_PLAINTEXT_BOB = "another-token-for-bob"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _registry() -> UserRegistry:
    return UserRegistry(
        users=(
            User(
                id="alice",
                token_sha256=_sha(_PLAINTEXT_ALICE),
                allowed_scopes=("Parent Full",),
            ),
            User(
                id="bob",
                token_sha256=_sha(_PLAINTEXT_BOB),
                allowed_scopes=None,
            ),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name="Parent Full",
                description="full",
                milvus_collection="col_full",
                postgres_table="col_full",
            ),
            Scope(
                name="Therapist Lite",
                description="lite",
                milvus_collection="col_lite",
                postgres_table="col_lite",
            ),
        )
    )


@pytest.fixture
def client() -> TestClient:
    app = create_app(
        users=_registry(),
        scopes=_scopes(),
        session_secret="test-secret-not-for-prod",
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/auth/login
# ---------------------------------------------------------------------------


def test_login_with_valid_token_sets_cookie_and_returns_user_id(
    client: TestClient,
) -> None:
    res = client.post("/api/auth/login", json={"token": _PLAINTEXT_ALICE})
    assert res.status_code == 200
    assert res.json() == {"user_id": "alice"}
    # Starlette's SessionMiddleware sets the cookie on first session write.
    assert "deka_session" in res.cookies or any(
        c.name == "deka_session" for c in client.cookies.jar
    )


def test_login_with_unknown_token_returns_401(client: TestClient) -> None:
    res = client.post("/api/auth/login", json={"token": "totally-fake"})
    assert res.status_code == 401
    assert "Unknown token" in res.json()["detail"]


def test_login_with_empty_payload_returns_422(client: TestClient) -> None:
    res = client.post("/api/auth/login", json={})
    assert res.status_code == 422  # pydantic missing-field


# ---------------------------------------------------------------------------
# /api/auth/me
# ---------------------------------------------------------------------------


def test_me_without_login_returns_401(client: TestClient) -> None:
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_me_after_login_returns_user_id(client: TestClient) -> None:
    client.post("/api/auth/login", json={"token": _PLAINTEXT_ALICE})
    res = client.get("/api/auth/me")
    assert res.status_code == 200
    assert res.json() == {"user_id": "alice"}


# ---------------------------------------------------------------------------
# /api/auth/logout
# ---------------------------------------------------------------------------


def test_logout_clears_cookie_so_subsequent_me_returns_401(
    client: TestClient,
) -> None:
    client.post("/api/auth/login", json={"token": _PLAINTEXT_ALICE})
    res = client.post("/api/auth/logout")
    assert res.status_code == 204

    me = client.get("/api/auth/me")
    assert me.status_code == 401


# ---------------------------------------------------------------------------
# Cross-cutting: per-user identity stays consistent
# ---------------------------------------------------------------------------


def test_login_replaces_existing_session_for_other_user(
    client: TestClient,
) -> None:
    """Logging in as bob while alice's cookie is set should replace the
    user_id payload — the cookie carries one identity at a time."""
    client.post("/api/auth/login", json={"token": _PLAINTEXT_ALICE})
    assert client.get("/api/auth/me").json()["user_id"] == "alice"
    client.post("/api/auth/login", json={"token": _PLAINTEXT_BOB})
    assert client.get("/api/auth/me").json()["user_id"] == "bob"
