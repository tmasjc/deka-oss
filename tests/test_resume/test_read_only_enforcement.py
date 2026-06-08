"""Read-only enforcement on every mutating endpoint.

The plan: create a normal session, flip ``ctx.read_only = True``
directly on the in-memory context (the production code path that
sets this is the deferred ``DONE_VIEW`` hydrator), then exercise
every endpoint the read-only flag is supposed to gate. Each call
must return ``409`` with a ``DONE_VIEW`` message.

GET endpoints continue to serve the snapshot — a separate test
confirms ``GET /api/session/<sid>`` works on a read-only context.
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


_TOKEN = "alice-readonly-test"
_SCOPE = "Test Scope"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _users() -> UserRegistry:
    return UserRegistry(
        users=(
            User(id="alice", token_sha256=_sha(_TOKEN), allowed_scopes=None),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(name=_SCOPE, description="d", milvus_collection="c1", postgres_table="c1"),
        )
    )


def _base_config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=3,
        embed_url="http://x",
        milvus_uri="http://x",
        collection="c1",
    )


def _fake_table(query: str, config: SearchConfig) -> EvidenceTable:
    rows = [
        EvidenceRow(
            rank=i,
            pk=f"pk-{i}",
            chunk_id=f"sample-{i}_C1",
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
def client_and_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
        return _fake_table(query, config)

    monkeypatch.setattr(app_module, "probe_only", fake_probe)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(runs_dir=tmp_path, base_config=base_config)
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]
    app = create_app(
        store=store,
        scopes=_scopes(),
        users=_users(),
        session_secret="t",
    )
    c = TestClient(app)
    c.post("/api/auth/login", json={"token": _TOKEN})
    return c, store


def _start_then_mark_read_only(client: TestClient, store: SessionStore) -> str:
    res = client.post(
        "/api/session", json={"query": "q", "scope": _SCOPE}
    )
    sid = res.json()["session_id"]
    store.get(sid).read_only = True
    return sid


# Each tuple: (HTTP method, path-template, body-or-None).
_MUTATING_ENDPOINTS: list[tuple[str, str, dict | None]] = [
    ("post", "/api/session/{sid}/rate", {"rank": 1, "rating": "FIT"}),
    ("post", "/api/session/{sid}/config", {"top_k": 5}),
    ("post", "/api/session/{sid}/audit", None),
    ("post", "/api/session/{sid}/drop_path", {"path": "sparse"}),
    ("post", "/api/session/{sid}/recommendation", {"decision": "ignore"}),
    ("post", "/api/session/{sid}/turn/next", None),
    ("post", "/api/session/{sid}/harvest/run", {"confirm": True}),
    ("post", "/api/session/{sid}/refine/derive", None),
    (
        "post",
        "/api/session/{sid}/refine/rubric",
        {"rubric_text": "# minimal\n"},
    ),
    ("post", "/api/session/{sid}/refine/judge", None),
    ("post", "/api/session/{sid}/refine/discard", None),
    ("post", "/api/session/{sid}/refine/finalize", None),
]


@pytest.mark.parametrize("method,path,body", _MUTATING_ENDPOINTS)
def test_read_only_session_returns_409(
    client_and_store, method: str, path: str, body: dict | None
) -> None:
    client, store = client_and_store
    sid = _start_then_mark_read_only(client, store)
    url = path.format(sid=sid)
    fn = getattr(client, method)
    res = fn(url, json=body) if body is not None else fn(url)
    assert res.status_code == 409, (
        f"{method.upper()} {url} expected 409 on read-only ctx, "
        f"got {res.status_code}: {res.text}"
    )
    assert "DONE_VIEW" in res.json()["detail"]


def test_read_only_does_not_block_get_session(client_and_store) -> None:
    client, store = client_and_store
    sid = _start_then_mark_read_only(client, store)
    res = client.get(f"/api/session/{sid}")
    assert res.status_code == 200
    assert res.json()["session_id"] == sid


def test_read_only_does_not_block_delete_session(client_and_store) -> None:
    """Ending a read-only session is allowed — DELETE just drops the
    in-memory ctx; sidecars on disk are untouched. The user should
    always be able to walk away from a session they're reviewing."""
    client, store = client_and_store
    sid = _start_then_mark_read_only(client, store)
    res = client.delete(f"/api/session/{sid}")
    assert res.status_code == 200
    assert store.get(sid) is None
