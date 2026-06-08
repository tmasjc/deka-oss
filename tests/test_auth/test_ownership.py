"""Cross-user ownership enforcement on the session API.

Two users in the registry; alice creates a session, bob can never
touch it. The 403 surface is what tells the frontend "this exists
but is not yours" so the resume flow in PR3 can route correctly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.search.adapt import AdaptedConfig
from src.search.config import SearchConfig
from src.search.evidence import EvidenceRow, EvidenceTable
from src.scopes import Scope, ScopeRegistry
from src.web_api import app as app_module
from src.web_api.app import create_app
from src.web_api.sessions import SessionStore


_SCOPE = "Test Scope"
_ALICE_TOKEN = "alice-token-for-ownership-tests"
_BOB_TOKEN = "bob-token-for-ownership-tests"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _users() -> UserRegistry:
    return UserRegistry(
        users=(
            User(
                id="alice",
                token_sha256=_sha(_ALICE_TOKEN),
                allowed_scopes=None,
            ),
            User(
                id="bob",
                token_sha256=_sha(_BOB_TOKEN),
                allowed_scopes=None,
            ),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name=_SCOPE,
                description="A scope",
                milvus_collection="col",
                postgres_table="col",
            ),
        )
    )


def _base_config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=3,
        embed_url="http://localhost:8000",
        milvus_uri="http://localhost:19530",
        collection="col",
    )


def _table(query: str, config: SearchConfig) -> EvidenceTable:
    rows = [
        EvidenceRow(
            rank=i,
            pk=f"pk-{i}",
            chunk_id=f"sample_C000000{i}",
            chunk_content=f"line {i}",
            sample_id=f"sample-{i}",
            counselor_id="c1",
            term="2025q1",
            source_paths=["dense"],
            scores={"dense": 0.5, "sparse": 0.0},
        )
        for i in range(1, 4)
    ]
    return EvidenceTable(
        query=query,
        config=config,
        rows=rows,
        per_path_candidates={"dense": [], "sparse": []},
    )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base_config = _base_config()

    def fake_probe(query, config, **_):  # noqa: ANN001
        class P:
            pass

        p = P()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    def fake_run_search(query, config, **_):  # noqa: ANN001
        return _table(query, config)

    monkeypatch.setattr(app_module, "probe_only", fake_probe)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(
        runs_dir=tmp_path,
        reflection_factory=lambda: None,
        extractor_factory=lambda sid: None,
        base_config=base_config,
    )
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]

    return create_app(
        store=store,
        scopes=_scopes(),
        users=_users(),
        session_secret="test-secret",
    )


def _client_for(app, token: str) -> TestClient:
    c = TestClient(app)
    res = c.post("/api/auth/login", json={"token": token})
    assert res.status_code == 200, res.text
    return c


def test_unauthenticated_request_returns_401(app) -> None:
    c = TestClient(app)
    assert c.get("/api/scopes").status_code == 401


def test_alice_creates_session_under_her_subdir(app, tmp_path: Path) -> None:
    alice = _client_for(app, _ALICE_TOKEN)
    res = alice.post(
        "/api/session", json={"query": "hello", "scope": _SCOPE}
    )
    assert res.status_code == 200
    sid = res.json()["session_id"]
    # Ownership marker on disk under runs/alice/.
    owner = (tmp_path / "alice" / f"{sid}.owner").read_text().strip()
    assert owner == "alice"
    # Bob's subdir is untouched.
    assert not (tmp_path / "bob").exists()


def test_bob_cannot_read_alice_session(app) -> None:
    alice = _client_for(app, _ALICE_TOKEN)
    sid = alice.post(
        "/api/session", json={"query": "hello", "scope": _SCOPE}
    ).json()["session_id"]

    bob = _client_for(app, _BOB_TOKEN)
    res = bob.get(f"/api/session/{sid}")
    assert res.status_code == 403


def test_bob_cannot_rate_in_alice_session(app) -> None:
    alice = _client_for(app, _ALICE_TOKEN)
    sid = alice.post(
        "/api/session", json={"query": "hello", "scope": _SCOPE}
    ).json()["session_id"]

    bob = _client_for(app, _BOB_TOKEN)
    res = bob.post(
        f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"}
    )
    assert res.status_code == 403


def test_truly_unknown_session_returns_404(app) -> None:
    alice = _client_for(app, _ALICE_TOKEN)
    res = alice.get("/api/session/000000000000")
    assert res.status_code == 404


def test_scopes_filtered_when_allowlist_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A user with a non-None allowed_scopes only sees those entries."""
    full_scopes = ScopeRegistry(
        scopes=(
            Scope(name="Foo", description="d", milvus_collection="c1", postgres_table="c1"),
            Scope(name="Bar", description="d", milvus_collection="c2", postgres_table="c2"),
        )
    )
    users = UserRegistry(
        users=(
            User(
                id="restricted",
                token_sha256=_sha("restricted-token"),
                allowed_scopes=("Foo",),
            ),
        )
    )
    store = SessionStore(runs_dir=tmp_path, base_config=_base_config())
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]
    app = create_app(
        store=store,
        scopes=full_scopes,
        users=users,
        session_secret="t",
    )
    c = TestClient(app)
    c.post("/api/auth/login", json={"token": "restricted-token"})
    res = c.get("/api/scopes")
    assert res.status_code == 200
    names = [s["name"] for s in res.json()["scopes"]]
    assert names == ["Foo"]


def test_create_session_blocks_when_scope_not_in_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """403 when a user attempts to start a session with a scope they
    aren't allowlisted for, even if the scope exists globally."""

    def fake_probe(query, config, **_):  # noqa: ANN001
        class P:
            pass

        p = P()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    monkeypatch.setattr(app_module, "probe_only", fake_probe)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(
        app_module, "run_search", lambda q, c, **_: _table(q, c)
    )

    full_scopes = ScopeRegistry(
        scopes=(
            Scope(name="Foo", description="d", milvus_collection="c1", postgres_table="c1"),
            Scope(name="Bar", description="d", milvus_collection="c2", postgres_table="c2"),
        )
    )
    users = UserRegistry(
        users=(
            User(
                id="restricted",
                token_sha256=_sha("restricted-token"),
                allowed_scopes=("Foo",),
            ),
        )
    )
    store = SessionStore(runs_dir=tmp_path, base_config=_base_config())
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]
    app = create_app(
        store=store,
        scopes=full_scopes,
        users=users,
        session_secret="t",
    )
    c = TestClient(app)
    c.post("/api/auth/login", json={"token": "restricted-token"})

    res = c.post("/api/session", json={"query": "q", "scope": "Bar"})
    assert res.status_code == 403
