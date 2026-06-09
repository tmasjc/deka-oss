"""Endpoint tests for the Deka web API.

The session loop (`probe_only`, `adapt_config`, `run_search`) is
monkey-patched so no Milvus / BGE-M3 / LLM service is touched. We
verify the HTTP surface matches the documented contract: session
bootstrap, rating, turn advance, reflection retrieval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.postgres.config import PostgresConfig
from src.search.adapt import AdaptedConfig
from src.search.config import SearchConfig
from src.search.evidence import CandidateRow, EvidenceRow, EvidenceTable
from src.scopes import Scope, ScopeRegistry
from src.web_api import app as app_module
from src.web_api.app import _read_session_scope_and_query, create_app
from src.web_api.sessions import SessionStore


_TEST_USER_ID = "alice"
_TEST_USER_TOKEN = "alice-secret-token-for-tests"


def _test_users() -> UserRegistry:
    sha = hashlib.sha256(_TEST_USER_TOKEN.encode("utf-8")).hexdigest()
    return UserRegistry(
        users=(
            User(
                id=_TEST_USER_ID,
                token_sha256=sha,
                allowed_scopes=None,  # all scopes allowed
            ),
        )
    )


_TEST_SCOPE_NAME = "Test Scope"
_TEST_SCOPE_COLLECTION = "test_scope_collection"


def _test_registry() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name=_TEST_SCOPE_NAME,
                description="A scope used by the API tests.",
                milvus_collection=_TEST_SCOPE_COLLECTION,
                postgres_table=_TEST_SCOPE_COLLECTION,
            ),
            Scope(
                name="Other Scope",
                description="Second scope to verify list endpoint ordering.",
                milvus_collection="other_collection",
                postgres_table="other_collection",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=3,
        embed_url="http://localhost:8000",
        milvus_uri="http://localhost:19530",
        collection="test_collection",
    )


def _fake_table(
    query: str, config: SearchConfig, *, prefix: str = "t1"
) -> EvidenceTable:
    rows = [
        EvidenceRow(
            rank=i,
            pk=f"{prefix}-pk-{i}",
            chunk_id=f"sample-{i}_C0000001",
            chunk_content=f"speaker{i}: hello line\nspeaker{i}: second line",
            sample_id=f"sample-{i}",
            counselor_id="c1",
            term="2025q1",
            source_paths=["dense"],
            scores={"dense": 0.5, "sparse": 0.0},
        )
        for i in range(1, 4)
    ]
    cand = CandidateRow(
        path="sparse",
        rank_in_path=1,
        pk=f"{prefix}-cand-1",
        chunk_id="sample-99_C0000001",
        chunk_content="teacher: candidate text\nparent: more text",
        sample_id="sample-99",
        counselor_id="c1",
        term="2025q1",
        score=0.3,
    )
    return EvidenceTable(
        query=query,
        config=config,
        rows=rows,
        per_path_candidates={"dense": [], "sparse": [cand]},
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """TestClient with all external services patched out."""
    base_config = _base_config()

    calls = {"run_search": 0, "configs": []}

    def fake_probe_only(query, config, **_kwargs):  # noqa: ANN001
        class _Probe:
            pass

        p = _Probe()
        p.query = query
        p.config = config
        # Minimal stats so adapt_config is happy (at least one active path).
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    def fake_run_search(query, config, **_kwargs):  # noqa: ANN001
        calls["run_search"] += 1
        calls["configs"].append(config)
        return _fake_table(query, config, prefix=f"turn{calls['run_search']}")

    monkeypatch.setattr(app_module, "probe_only", fake_probe_only)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(
        runs_dir=tmp_path,
        reflection_factory=lambda: None,
        extractor_factory=lambda sid: None,
        base_config=base_config,
    )
    # Avoid writing progress-log files from tests. The signature now
    # takes (session_id, user_runs) — the lambda accepts both.
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]

    app = create_app(
        store=store,
        scopes=_test_registry(),
        users=_test_users(),
        session_secret="test-secret-not-for-prod",
    )
    test_client = TestClient(app)
    # Authenticate once at fixture time so every request the test
    # makes carries the cookie.
    res = test_client.post("/api/auth/login", json={"token": _TEST_USER_TOKEN})
    assert res.status_code == 200, (
        f"test fixture login failed: {res.status_code} {res.text}"
    )
    test_client.captured_configs = calls["configs"]  # type: ignore[attr-defined]
    test_client.session_store = store  # type: ignore[attr-defined]
    return test_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_start_session_returns_snapshot(client: TestClient) -> None:
    r = client.post(
        "/api/session", json={"query": "  hello world  ", "scope": _TEST_SCOPE_NAME}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "hello world"
    assert len(body["table"]["rows"]) == 3
    assert body["turn_number"] == 1
    assert body["params"]["top_k"] == 3
    assert body["workflow"][0]["status"] == "done"  # START
    assert body["workflow"][1]["status"] == "current"  # TURN 1
    assert body["turn_complete"] is False
    assert body["scope"] == _TEST_SCOPE_NAME
    assert body["precision_trend"] == []
    assert body["breakdown_by_turn"] == []
    assert body["drop_impact_preview"] is None


def test_empty_query_rejected(client: TestClient) -> None:
    r = client.post("/api/session", json={"query": "   ", "scope": _TEST_SCOPE_NAME})
    assert r.status_code == 400


def test_missing_scope_rejected(client: TestClient) -> None:
    r = client.post("/api/session", json={"query": "hello"})
    # Pydantic validation error on required field.
    assert r.status_code == 422


def test_unknown_scope_rejected(client: TestClient) -> None:
    r = client.post("/api/session", json={"query": "hello", "scope": "Bogus Scope"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Bogus Scope" in detail
    assert _TEST_SCOPE_NAME in detail  # available list is surfaced


def test_start_session_applies_scope_collection_to_config(
    client: TestClient,
) -> None:
    """The chosen scope's milvus_collection drives SearchConfig.collection."""
    r = client.post("/api/session", json={"query": "hello", "scope": _TEST_SCOPE_NAME})
    assert r.status_code == 200
    assert r.json()["scope"] == _TEST_SCOPE_NAME
    captured = client.captured_configs  # type: ignore[attr-defined]
    assert captured, "run_search should have been called at least once"
    assert captured[0].collection == _TEST_SCOPE_COLLECTION


def test_scopes_endpoint_lists_configured_scopes(
    client: TestClient,
) -> None:
    r = client.get("/api/scopes")
    assert r.status_code == 200, r.text
    body = r.json()
    names = [s["name"] for s in body["scopes"]]
    assert names == [_TEST_SCOPE_NAME, "Other Scope"]


def test_scopes_endpoint_dto_shape(client: TestClient) -> None:
    r = client.get("/api/scopes")
    assert r.status_code == 200
    first = r.json()["scopes"][0]
    assert set(first.keys()) == {"name", "description", "milvus_collection"}
    assert first["milvus_collection"] == _TEST_SCOPE_COLLECTION


def test_rate_row_then_candidate(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    r = client.post(f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"})
    assert r.status_code == 200
    rows = r.json()["table"]["rows"]
    assert rows[0]["rating"] == "FIT"

    r = client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "NOT_FIT"},
    )
    assert r.status_code == 200
    cands = r.json()["table"]["per_path_candidates"]["sparse"]
    assert cands[0]["rating"] == "NOT_FIT"


def test_rate_with_discard(client: TestClient) -> None:
    """DISCARD is accepted by the rate endpoint for both rows and
    candidates (issue #46)."""
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    r = client.post(f"/api/session/{sid}/rate", json={"rank": 1, "rating": "DISCARD"})
    assert r.status_code == 200, r.text
    assert r.json()["table"]["rows"][0]["rating"] == "DISCARD"

    r = client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "DISCARD"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["table"]["per_path_candidates"]["sparse"][0]["rating"] == "DISCARD"


def test_precision_excludes_discard(client: TestClient) -> None:
    """Precision@K denominator excludes DISCARD chunks — they are an
    escape hatch, not a verdict (issue #46). 1 FIT + 1 NOT_FIT + 1
    DISCARD must report P@K = 0.5, not 1/3."""
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    client.post(f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"})
    client.post(f"/api/session/{sid}/rate", json={"rank": 2, "rating": "NOT_FIT"})
    client.post(f"/api/session/{sid}/rate", json={"rank": 3, "rating": "DISCARD"})

    r = client.post(f"/api/session/{sid}/turn/next")
    assert r.status_code == 200, r.text
    trend = r.json()["snapshot"]["precision_trend"]
    assert trend, "precision_trend should record the completed turn"
    assert trend[0] == pytest.approx(0.5)


def test_rate_requires_selector(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    r = client.post(f"/api/session/{sid}/rate", json={"rating": "FIT"})
    assert r.status_code == 400


def test_next_turn_rejected_when_unrated(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    r = client.post(f"/api/session/{sid}/turn/next")
    assert r.status_code == 409


def test_full_turn_advance(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    # Rate every row + candidate.
    for rank in (1, 2, 3):
        client.post(f"/api/session/{sid}/rate", json={"rank": rank, "rating": "FIT"})
    client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "NOT_FIT"},
    )

    snap = client.get(f"/api/session/{sid}").json()
    assert snap["turn_complete"] is True

    r = client.post(f"/api/session/{sid}/turn/next")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["snapshot"]["turn_number"] == 2
    # New turn loaded — chunk IDs from prefix turn2.
    pks = [row["pk"] for row in body["snapshot"]["table"]["rows"]]
    assert all("turn2" in pk for pk in pks)
    # Reflection is None because the store's reflection factory returns None.
    assert body["reflection"] is None


def test_unknown_session_returns_404(client: TestClient) -> None:
    r = client.get("/api/session/does-not-exist")
    assert r.status_code == 404


def test_client_supplied_session_id_is_honored(client: TestClient) -> None:
    r = client.post(
        "/api/session",
        json={"query": "hello", "scope": _TEST_SCOPE_NAME, "session_id": "abc-123"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["session_id"] == "abc-123"
    # Can be polled via /progress.
    p = client.get("/api/session/abc-123/progress")
    assert p.status_code == 200
    # After successful bootstrap, stage is 'done'.
    assert p.json()["stage"] == "done"


def test_client_supplied_session_id_conflict(client: TestClient) -> None:
    client.post(
        "/api/session",
        json={"query": "hello", "scope": _TEST_SCOPE_NAME, "session_id": "dup"},
    )
    r = client.post(
        "/api/session",
        json={"query": "hello", "scope": _TEST_SCOPE_NAME, "session_id": "dup"},
    )
    assert r.status_code == 409


def test_progress_endpoint_reaches_extracting_spans(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """When extract_span is wired, run_search emits extracting_spans with a total."""

    # Replace run_search with a version that invokes on_stage like the real one.
    def fake_run_search(query, config, **kwargs):  # noqa: ANN001
        on_stage = kwargs.get("on_stage")
        extract = kwargs.get("extract_span")
        if on_stage:
            on_stage("searching_milvus", {})
        table = _fake_table(query, config, prefix="stage")
        if extract is not None and on_stage is not None:
            total = len(table.rows) + sum(
                len(v) for v in table.per_path_candidates.values()
            )
            on_stage("extracting_spans", {"total": total})
            for i, row in enumerate(table.rows, start=1):
                res = extract(row.chunk_content)
                assert res is not None
                row.span_line_indices, row.span_text = res
        return table

    monkeypatch.setattr(app_module, "run_search", fake_run_search)
    # Inject a stub extractor so _build_extract_span wires up.
    from src.web_api import app as _app
    from src.extraction.session import _TurnExtractSpan

    class _StubExtractor:
        model = "stub"
        prompt_version = "v0"
        latency_ms_total = 0.0
        cache_hits = 0

        def extract(self, *, query, chunk_content, prior_fit_spans):  # noqa: ARG002
            return type("R", (), {"span_line_indices": [0]})()

        def derive_span_text(self, chunk_content, indices):  # noqa: ARG002
            return "stub span"

    def build_stub(ctx, query):  # noqa: ANN001
        return _TurnExtractSpan(
            _StubExtractor(),
            ctx.state,
            query,
            on_progress=ctx.progress.set_processed,
        )

    monkeypatch.setattr(_app, "_build_extract_span", build_stub)

    r = client.post(
        "/api/session",
        json={"query": "hello", "scope": _TEST_SCOPE_NAME, "session_id": "prog-1"},
    )
    assert r.status_code == 200, r.text
    p = client.get("/api/session/prog-1/progress").json()
    assert p["stage"] == "done"
    # During bootstrap, total was set on extracting_spans; final total persists
    # on the snapshot because we don't reset after 'done'. Just verify the
    # happy-path shape.
    assert p["error"] is None


def test_end_session(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    r = client.delete(f"/api/session/{sid}")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    r = client.get(f"/api/session/{sid}")
    assert r.status_code == 404


def test_discard_session_removes_files(client: TestClient, tmp_path: Path) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    # SessionStore writes per-user sidecars (e.g. .owner, .log) on create.
    user_runs = tmp_path / _TEST_USER_ID
    sidecars_before = sorted(user_runs.glob(f"{sid}.*"))
    assert sidecars_before, "expected at least one sidecar after session start"

    r = client.post(f"/api/session/{sid}/discard")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True}

    assert not list(user_runs.glob(f"{sid}.*")), (
        "discard should have removed every sidecar"
    )
    # Idempotency: a second discard finds nothing and returns 404.
    r = client.post(f"/api/session/{sid}/discard")
    assert r.status_code == 404


def test_config_edit_happy_path(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    r = client.post(
        f"/api/session/{sid}/config",
        json={"rrf_k": 120, "active_paths": ["dense"]},
    )
    assert r.status_code == 200, r.text
    params = r.json()["params"]
    assert params["rrf_k"] == 120
    assert params["active_paths"] == ["dense"]


def test_config_edit_requires_at_least_one_field(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    r = client.post(f"/api/session/{sid}/config", json={})
    assert r.status_code == 400


def test_config_edit_locked_after_first_turn(client: TestClient) -> None:
    """Once a turn has completed, the seed config is locked. The audit
    flow is the only mid-session config-change channel."""
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    # Pre-turn edits work.
    r = client.post(f"/api/session/{sid}/config", json={"rrf_k": 120})
    assert r.status_code == 200

    # Complete turn 1, advance to turn 2.
    for rank in (1, 2, 3):
        client.post(f"/api/session/{sid}/rate", json={"rank": rank, "rating": "FIT"})
    client.post(f"/api/session/{sid}/turn/next")

    # Now the seed is frozen; the editor must refuse.
    r = client.post(f"/api/session/{sid}/config", json={"rrf_k": 200})
    assert r.status_code == 409
    assert "locked" in r.json()["detail"].lower()


def test_audit_endpoint_flips_audit_mode_active(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    snap_before = client.get(f"/api/session/{sid}").json()
    assert snap_before["audit_mode_active"] is False

    r = client.post(f"/api/session/{sid}/audit")
    assert r.status_code == 200, r.text
    assert r.json()["audit_mode_active"] is True

    # Idempotent — a second call is fine.
    r = client.post(f"/api/session/{sid}/audit")
    assert r.status_code == 200
    assert r.json()["audit_mode_active"] is True


def test_drop_path_requires_audit_mode(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    r = client.post(f"/api/session/{sid}/drop_path", json={"path": "sparse"})
    assert r.status_code == 400
    assert "audit" in r.json()["detail"].lower()


def test_drop_path_requires_all_rated(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    client.post(f"/api/session/{sid}/audit")
    # Audit mode requires fused rows + candidates rated; fused rows aren't
    # done yet, so drop must fail.
    r = client.post(f"/api/session/{sid}/drop_path", json={"path": "sparse"})
    assert r.status_code == 400
    assert "rate" in r.json()["detail"].lower()


def test_drop_path_blocked_by_rule_b_sole_source(client: TestClient) -> None:
    """Every fused row in the fake table is sourced only from dense; rating
    any of them FIT makes a drop of dense a Rule B sole-source violation."""
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    client.post(f"/api/session/{sid}/audit")
    for rank in (1, 2, 3):
        client.post(f"/api/session/{sid}/rate", json={"rank": rank, "rating": "FIT"})
    client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "NOT_FIT"},
    )

    r = client.post(f"/api/session/{sid}/drop_path", json={"path": "dense"})
    assert r.status_code == 409
    assert "FIT" in r.json()["detail"]

    # Config unchanged.
    snap = client.get(f"/api/session/{sid}").json()
    assert sorted(snap["params"]["active_paths"]) == ["dense", "sparse"]


def test_drop_path_succeeds_when_rule_b_clear(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    client.post(f"/api/session/{sid}/audit")
    # Rate fused rows NOT_FIT (no sole-source-dense FIT) and the sparse
    # candidate NOT_FIT, leaving "dense" cleanly droppable.
    for rank in (1, 2, 3):
        client.post(
            f"/api/session/{sid}/rate", json={"rank": rank, "rating": "NOT_FIT"}
        )
    client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "NOT_FIT"},
    )

    r = client.post(f"/api/session/{sid}/drop_path", json={"path": "dense"})
    assert r.status_code == 200, r.text
    snap = r.json()
    assert snap["params"]["active_paths"] == ["sparse"]


def test_config_edit_rejects_empty_active_paths(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    r = client.post(f"/api/session/{sid}/config", json={"active_paths": []})
    assert r.status_code == 400


def test_snapshot_after_turn_advance_has_trend_and_breakdown(
    client: TestClient,
) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]

    for rank in (1, 2, 3):
        client.post(f"/api/session/{sid}/rate", json={"rank": rank, "rating": "FIT"})
    client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "NOT_FIT"},
    )
    r = client.post(f"/api/session/{sid}/turn/next")
    assert r.status_code == 200, r.text

    snap = client.get(f"/api/session/{sid}").json()
    assert snap["precision_trend"] == [1.0]
    assert len(snap["breakdown_by_turn"]) == 1
    entry = snap["breakdown_by_turn"][0]
    assert entry["turn"] == 1
    assert "dense_only" in entry["breakdown"]
    assert entry["breakdown"]["dense_only"]["fit"] == 3


def test_reflection_endpoint_when_none(client: TestClient) -> None:
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    r = client.get(f"/api/session/{sid}/reflection")
    # No completed turns yet.
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Original-content (Postgres) endpoint
# ---------------------------------------------------------------------------


class _StubFetcher:
    """Stand-in for OriginalContentFetcher — records calls, returns canned data."""

    def __init__(
        self,
        *,
        rows: dict[object, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._rows = rows or {}
        self._error = error
        self.calls: list[object] = []
        self.closed = False

    def fetch_original(self, pk: object) -> str | None:
        self.calls.append(pk)
        if self._error is not None:
            raise self._error
        return self._rows.get(pk)

    def close(self) -> None:
        self.closed = True


def _make_client_with_fetcher(
    monkeypatch: pytest.MonkeyPatch,
    fetcher: _StubFetcher | None,
    tmp_path: Path,
    *,
    pg_disabled_by_admin: bool = False,
) -> TestClient:
    base_config = _base_config()

    def fake_probe_only(query, config, **_kwargs):  # noqa: ANN001
        class _Probe:
            pass

        p = _Probe()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    def fake_run_search(query, config, **_kwargs):  # noqa: ANN001
        return _fake_table(query, config, prefix="t1")

    monkeypatch.setattr(app_module, "probe_only", fake_probe_only)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(
        runs_dir=tmp_path,
        reflection_factory=lambda: None,
        extractor_factory=lambda sid: None,
        base_config=base_config,
    )
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]

    app = create_app(
        store=store,
        pg_fetcher=fetcher,  # type: ignore[arg-type]
        pg_disabled_by_admin=pg_disabled_by_admin,
        scopes=_test_registry(),
        users=_test_users(),
        session_secret="test-secret-not-for-prod",
    )
    # Pre-populate the per-table fetcher cache so /expand and refine
    # judge resolve via the stub instead of building a real
    # OriginalContentFetcher against a non-existent Postgres.
    if fetcher is not None:
        app.state.pg_fetcher_by_table = {
            _TEST_SCOPE_COLLECTION: fetcher,
        }
    test_client = TestClient(app)
    res = test_client.post("/api/auth/login", json={"token": _TEST_USER_TOKEN})
    assert res.status_code == 200, (
        f"test fixture login failed: {res.status_code} {res.text}"
    )
    return test_client


def _start_session(client: TestClient) -> str:
    return client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]


def test_original_content_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _StubFetcher(rows={"t1-pk-1": "full original text"})
    client = _make_client_with_fetcher(monkeypatch, fetcher, tmp_path)
    sid = _start_session(client)

    r = client.get(f"/api/session/{sid}/chunks/t1-pk-1/original")

    assert r.status_code == 200, r.text
    assert r.json() == {"pk": "t1-pk-1", "original_content": "full original text"}
    assert fetcher.calls == ["t1-pk-1"]


def test_original_content_numeric_pk_is_coerced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _StubFetcher(rows={42: "by-int"})
    client = _make_client_with_fetcher(monkeypatch, fetcher, tmp_path)
    sid = _start_session(client)

    r = client.get(f"/api/session/{sid}/chunks/42/original")

    assert r.status_code == 200, r.text
    assert r.json() == {"pk": 42, "original_content": "by-int"}
    assert fetcher.calls == [42]


def test_original_content_missing_pk_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _StubFetcher(rows={})
    client = _make_client_with_fetcher(monkeypatch, fetcher, tmp_path)
    sid = _start_session(client)

    r = client.get(f"/api/session/{sid}/chunks/does-not-exist/original")
    assert r.status_code == 404


def test_original_content_unknown_session_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _StubFetcher(rows={"a": "b"})
    client = _make_client_with_fetcher(monkeypatch, fetcher, tmp_path)

    r = client.get("/api/session/no-such/chunks/a/original")
    assert r.status_code == 404


def test_original_content_fetcher_error_returns_502(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.postgres.fetch import PostgresFetchError

    fetcher = _StubFetcher(error=PostgresFetchError("boom"))
    client = _make_client_with_fetcher(monkeypatch, fetcher, tmp_path)
    sid = _start_session(client)

    r = client.get(f"/api/session/{sid}/chunks/t1-pk-1/original")
    assert r.status_code == 502


def test_original_content_disabled_when_no_fetcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _make_client_with_fetcher(monkeypatch, None, tmp_path)
    sid = _start_session(client)

    r = client.get(f"/api/session/{sid}/chunks/t1-pk-1/original")
    assert r.status_code == 503


def test_original_content_returns_403_when_admin_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _make_client_with_fetcher(
        monkeypatch, None, tmp_path, pg_disabled_by_admin=True
    )
    sid = _start_session(client)

    r = client.get(f"/api/session/{sid}/chunks/t1-pk-1/original")

    assert r.status_code == 403
    assert r.json()["detail"] == (
        "Context expansion is disabled by server configuration"
    )


# ---------------------------------------------------------------------------
# Recommendation-decision endpoint
# ---------------------------------------------------------------------------


class _RecordingLogger:
    """Logging fake that captures log_event calls for assertion."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.turn_calls: int = 0

    def log_turn(self, state, turn) -> None:  # noqa: ANN001
        self.turn_calls += 1

    def log_event(self, *, turn: int, kind: str, **payload: Any) -> None:
        self.events.append({"turn": turn, "kind": kind, **payload})


class _RecommendingReflector:
    """Reflection hook returning a payload with path_drop_recommendation."""

    def __init__(self, recommendation: dict[str, Any] | None) -> None:
        self._rec = recommendation

    def reflect(self, state) -> dict[str, Any] | None:  # noqa: ANN001
        result: dict[str, Any] = {
            "observe": "o",
            "diagnose": "d",
            "hypothesis": "h",
            "previous_hypothesis_verdict": None,
            "status": "CONTINUE",
        }
        if self._rec is not None:
            result["path_drop_recommendation"] = dict(self._rec)
        return result


def _make_recommendation_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    recommendation: dict[str, Any] | None,
) -> tuple[TestClient, _RecordingLogger]:
    """Like ``client``, but with a reflection hook that returns the
    given recommendation and a recording logging hook. Returns the
    client and the captured logger so tests can assert on log_event.
    """
    base_config = _base_config()

    def fake_probe_only(query, config, **_kwargs):  # noqa: ANN001
        class _Probe:
            pass

        p = _Probe()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    counter = {"n": 0}

    def fake_run_search(query, config, **_kwargs):  # noqa: ANN001
        counter["n"] += 1
        return _fake_table(query, config, prefix=f"turn{counter['n']}")

    monkeypatch.setattr(app_module, "probe_only", fake_probe_only)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    logger = _RecordingLogger()
    store = SessionStore(
        runs_dir=tmp_path,
        reflection_factory=lambda: _RecommendingReflector(recommendation),
        extractor_factory=lambda sid: None,
        base_config=base_config,
    )
    store._make_logging_hook = lambda session_id, user_runs: logger  # type: ignore[assignment]

    app = create_app(
        store=store,
        scopes=_test_registry(),
        users=_test_users(),
        session_secret="test-secret-not-for-prod",
    )
    test_client = TestClient(app)
    res = test_client.post("/api/auth/login", json={"token": _TEST_USER_TOKEN})
    assert res.status_code == 200, (
        f"test fixture login failed: {res.status_code} {res.text}"
    )
    return test_client, logger


def _advance_one_turn(client: TestClient) -> str:
    """Start a session, rate everything, advance one turn. Returns sid."""
    sid = client.post(
        "/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME}
    ).json()["session_id"]
    for rank in (1, 2, 3):
        client.post(f"/api/session/{sid}/rate", json={"rank": rank, "rating": "FIT"})
    client.post(
        f"/api/session/{sid}/rate",
        json={"path": "sparse", "rank_in_path": 1, "rating": "NOT_FIT"},
    )
    r = client.post(f"/api/session/{sid}/turn/next")
    assert r.status_code == 200, r.text
    return sid


def test_recommendation_apply_drops_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``apply`` drops the recommended path immediately — no audit
    step, no audit_mode_active flip. The snapshot's active_paths
    shrinks by one."""
    rec = {"path": "sparse", "reason": "consistent NOT_FIT", "confidence": "medium"}
    client, logger = _make_recommendation_client(
        monkeypatch, tmp_path, recommendation=rec
    )
    sid = _advance_one_turn(client)

    # Reflection lands on the just-completed turn; before apply the
    # config still has both paths and audit mode is off.
    snap = client.get(f"/api/session/{sid}").json()
    assert snap["audit_mode_active"] is False
    assert sorted(snap["params"]["active_paths"]) == ["dense", "sparse"]

    r = client.post(f"/api/session/{sid}/recommendation", json={"decision": "apply"})
    assert r.status_code == 200, r.text
    body = r.json()
    # The path is dropped; audit mode stays off (recommendation flow
    # no longer touches it).
    assert body["audit_mode_active"] is False
    assert body["params"]["active_paths"] == ["dense"]

    # One-shot safeguard flag is now set on the session.
    store: SessionStore = client.app.state.store  # type: ignore[attr-defined]
    ctx = store.get(sid)
    assert ctx is not None
    assert ctx.state.recommended_drop_applied is True

    # log_event was called with the expected shape.
    assert any(
        e["kind"] == "path_drop_recommendation_decision" and e["decision"] == "apply"
        for e in logger.events
    )


def test_recommendation_ignore_leaves_state_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rec = {"path": "sparse", "reason": "r", "confidence": "low"}
    client, logger = _make_recommendation_client(
        monkeypatch, tmp_path, recommendation=rec
    )
    sid = _advance_one_turn(client)

    r = client.post(f"/api/session/{sid}/recommendation", json={"decision": "ignore"})
    assert r.status_code == 200
    body = r.json()
    assert body["audit_mode_active"] is False
    # Active_paths unchanged on ignore.
    assert sorted(body["params"]["active_paths"]) == ["dense", "sparse"]

    assert any(e["decision"] == "ignore" for e in logger.events)


def test_recommendation_404_when_no_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reflection hook returns no path_drop_recommendation → endpoint 404."""
    client, _ = _make_recommendation_client(monkeypatch, tmp_path, recommendation=None)
    sid = _advance_one_turn(client)

    r = client.post(f"/api/session/{sid}/recommendation", json={"decision": "apply"})
    assert r.status_code == 404


def test_recommendation_double_call_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Single-shot: a second decision on the same recommendation 404s."""
    rec = {"path": "sparse", "reason": "r", "confidence": "medium"}
    client, _ = _make_recommendation_client(monkeypatch, tmp_path, recommendation=rec)
    sid = _advance_one_turn(client)

    r1 = client.post(f"/api/session/{sid}/recommendation", json={"decision": "ignore"})
    assert r1.status_code == 200

    r2 = client.post(f"/api/session/{sid}/recommendation", json={"decision": "apply"})
    assert r2.status_code == 404


def test_recommendation_decision_dto_elides_consumed_recommendation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After the operator decides, /reflection no longer surfaces the
    recommendation — keeps a refresh from re-rendering the banner."""
    rec = {"path": "sparse", "reason": "r", "confidence": "medium"}
    client, _ = _make_recommendation_client(monkeypatch, tmp_path, recommendation=rec)
    sid = _advance_one_turn(client)

    pre = client.get(f"/api/session/{sid}/reflection").json()
    assert pre["path_drop_recommendation"] is not None

    client.post(f"/api/session/{sid}/recommendation", json={"decision": "ignore"})

    post = client.get(f"/api/session/{sid}/reflection").json()
    assert post["path_drop_recommendation"] is None


def test_recommendation_apply_409_when_path_inactive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recommendation names a path that's no longer active → 409, and the
    rejection is logged with note=path_inactive so analytics see the
    attempt (matches the TUI's behaviour)."""
    rec = {"path": "dense", "reason": "r", "confidence": "high"}
    client, logger = _make_recommendation_client(
        monkeypatch, tmp_path, recommendation=rec
    )
    sid = _advance_one_turn(client)

    # Manually mutate the session config to drop dense before the
    # recommendation lands. Easiest: poke state.current_config directly
    # through the session store.
    store: SessionStore = client.app.state.store  # type: ignore[attr-defined]
    ctx = store.get(sid)
    assert ctx is not None
    from src.search.config import with_overrides

    ctx.state.current_config = with_overrides(
        ctx.state.current_config, active_paths=frozenset({"sparse"})
    )

    r = client.post(f"/api/session/{sid}/recommendation", json={"decision": "apply"})
    assert r.status_code == 409
    rejections = [
        e
        for e in logger.events
        if e["kind"] == "path_drop_recommendation_decision"
        and e["decision"] == "apply"
        and e.get("note") == "path_inactive"
    ]
    assert len(rejections) == 1
    # The recommendation stays pending so the operator can re-decide.
    assert ctx.state.turns[-1].reflection is not None
    assert ctx.state.turns[-1].reflection.get("_recommendation_consumed") is not True


def test_recommendation_apply_409_when_last_active_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recommendation would drop the last active path → 409, and the
    rejection is logged with note=last_path."""
    rec = {"path": "sparse", "reason": "r", "confidence": "high"}
    client, logger = _make_recommendation_client(
        monkeypatch, tmp_path, recommendation=rec
    )
    sid = _advance_one_turn(client)

    # Mutate config so only sparse remains active; the recommendation
    # then targets the sole active path.
    store: SessionStore = client.app.state.store  # type: ignore[attr-defined]
    ctx = store.get(sid)
    assert ctx is not None
    from src.search.config import with_overrides

    ctx.state.current_config = with_overrides(
        ctx.state.current_config, active_paths=frozenset({"sparse"})
    )

    r = client.post(f"/api/session/{sid}/recommendation", json={"decision": "apply"})
    assert r.status_code == 409
    assert "last" in r.json()["detail"].lower()
    rejections = [
        e
        for e in logger.events
        if e["kind"] == "path_drop_recommendation_decision"
        and e["decision"] == "apply"
        and e.get("note") == "last_path"
    ]
    assert len(rejections) == 1
    assert ctx.state.turns[-1].reflection.get("_recommendation_consumed") is not True


# ---------------------------------------------------------------------------
# Phase 2 (Harvest) — preflight, run, result
# ---------------------------------------------------------------------------


def _force_convergence(client: TestClient, sid: str) -> None:
    """Drop the dual-gate thresholds and rate one fused row FIT so the
    session reads as converged. Lets harvest tests skip the multi-turn
    drive that real convergence would require.
    """
    from src.session.state import ConvergenceThresholds

    permissive = ConvergenceThresholds(min_fit=1, min_not_fit=0, precision_at_k=0.0)
    store = client.session_store  # type: ignore[attr-defined]
    # _sessions is keyed by session id; the harvest tests own a single
    # logged-in user, so direct lookup is fine.
    ctx = store._sessions[sid]
    ctx.state.convergence = permissive

    r = client.post(f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"})
    assert r.status_code == 200, r.text
    snap = r.json()
    for row in snap["table"]["rows"][1:]:
        if row["rating"] is None:
            client.post(
                f"/api/session/{sid}/rate",
                json={"rank": row["rank"], "rating": "NOT_FIT"},
            )

    r = client.post(f"/api/session/{sid}/turn/next")
    assert r.status_code == 200, r.text
    snap = r.json()["snapshot"]
    assert snap["convergence"]["converged"], snap


def _build_fake_anchor_result() -> Any:
    """Construct a minimal AnchorResult-shaped object for fake runners."""
    from types import SimpleNamespace

    from src.anchor.config import RadiusScheme

    return SimpleNamespace(
        inputs=SimpleNamespace(fits=[SimpleNamespace(pk="fp1", chunk_id="C1")]),
        calibration=SimpleNamespace(
            T=0.42,
            T_prime_out=0.55,
            T_primes=[0.5, 0.55, 0.6],
            deltas=[0.1, 0.13, 0.15],
            n_fit=3,
        ),
        recovery=SimpleNamespace(
            verdict="HEALTHY", recovered=3, total=3, missed_fits=[], per_fit=[]
        ),
        retrieval=SimpleNamespace(
            candidates=[SimpleNamespace(pk="x1"), SimpleNamespace(pk="x2")],
            per_fit_pages=[
                SimpleNamespace(fit_chunk_id="C1", budget_exhausted=False),
            ],
        ),
        write=SimpleNamespace(
            n_records=2,
            jsonl_path="/tmp/fake.phase2.jsonl",
            meta_path="/tmp/fake.phase2.meta.json",
        ),
        not_fit_intrusions=0,
        timings=SimpleNamespace(
            load_ms=1.0,
            calibrate_ms=2.0,
            loo_ms=3.0,
            retrieve_ms=4.0,
            total_ms=10.0,
        ),
        radius_scheme=RadiusScheme.PER_FIT,
        cohort_consistency=[
            {"fit_pk": "fp1", "fit_chunk_id": "C1", "own_chunk_retained": True}
        ],
        quality_gate_dropped=[],
        quality_gate_median_delta_pre_drop=0.13,
        quality_gate_T_pre_drop=0.42,
        quality_gate_multiplier=3.0,
        quality_gate_multiplier_cutoff=0.39,
        quality_gate_median_floor_applied=False,
        n_discard_filtered=0,
        frequency_gate=SimpleNamespace(
            f_configured=2,
            n_fit_after_quality_gate=3,
            kept=2,
            dropped=0,
            qualifying_count_distribution={"min": 2, "median": 2, "max": 3},
            qualifying_count_histogram={2: 1, 3: 1},
        ),
    )


def _wait_for_phase(
    client: TestClient, sid: str, target: str, timeout_s: float = 5.0
) -> dict[str, Any]:
    """Poll ``GET /api/session/{sid}`` until phase reaches ``target``."""
    import time

    deadline = time.monotonic() + timeout_s
    snap: dict[str, Any] = {}
    while time.monotonic() < deadline:
        r = client.get(f"/api/session/{sid}")
        assert r.status_code == 200, r.text
        snap = r.json()
        if snap["phase"] == target:
            return snap
        time.sleep(0.05)
    raise AssertionError(
        f"Phase did not reach {target!r} within {timeout_s}s; last={snap.get('phase')}"
    )


def test_harvest_preflight_requires_convergence(client: TestClient) -> None:
    """Hitting ``/harvest/start`` before the dual gate fires returns 409."""
    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    r = client.post(f"/api/session/{sid}/harvest/start")
    assert r.status_code == 409
    assert "converged" in r.json()["detail"].lower()


def test_harvest_preflight_returns_payload(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """After convergence the preflight returns the FIT count + budgets."""
    from src.anchor.config import HarvestConfig, RadiusScheme

    fake_cfg = HarvestConfig(
        precision_at_k=0.0,
        min_fit=1,
        min_not_fit=1,
        batch_size=64,
        max_k=10000,
        radius_scheme=RadiusScheme.PER_FIT,
        anchor_frequency_gate=2,
        s2c_outlier_multiple=2.0,
    )
    import src.anchor.config as anchor_cfg_module

    monkeypatch.setattr(anchor_cfg_module, "load_harvest_config", lambda **_: fake_cfg)

    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    _force_convergence(client, sid)

    r = client.post(f"/api/session/{sid}/harvest/start")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_fit"] >= 1
    assert body["batch_size"] == 64
    assert body["max_k"] == 10000
    assert body["radius_scheme"] == "per_fit"


def test_harvest_run_completes_and_result_returns_dto(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Run harvest with a fake runner; verify phase rolls to ANCHOR_DONE
    and ``GET /harvest/result`` returns the flattened DTO."""
    fake = _build_fake_anchor_result()

    def fake_run_anchor(session_target, **kwargs):  # noqa: ANN001
        cb = kwargs.get("progress")
        if cb is not None:
            cb("Loading FITs from session...")
        return fake

    import src.anchor.runner as runner_module

    monkeypatch.setattr(runner_module, "run_anchor", fake_run_anchor)

    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    _force_convergence(client, sid)

    r = client.post(f"/api/session/{sid}/harvest/run", json={"confirm": True})
    assert r.status_code == 200, r.text
    assert r.json()["phase"] in ("ANCHOR_RUNNING", "ANCHOR_DONE")

    snap = _wait_for_phase(client, sid, "ANCHOR_DONE")
    harvest_step = next(s for s in snap["workflow"] if s["key"] == "HARVEST")
    assert harvest_step["status"] == "done"
    assert "retained" in (harvest_step["detail"] or "")

    r = client.get(f"/api/session/{sid}/harvest/result")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "HEALTHY"
    assert body["loo_recovered"] == 3
    assert body["loo_total"] == 3
    assert body["retained_chunks"] == 2
    assert body["radius_scheme"] == "per_fit"
    assert body["frequency_gate"]["f_configured"] == 2
    assert body["sidecar_jsonl_path"] == "/tmp/fake.phase2.jsonl"


def test_harvest_run_validation_error_returns_422(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """``AnchorValidationError`` from the worker rolls phase to
    ANCHOR_FAILED and ``GET /harvest/result`` returns 422 with the
    runner's message."""
    from src.anchor.errors import AnchorValidationError

    def fake_run_anchor(session_target, **kwargs):  # noqa: ANN001
        raise AnchorValidationError("LOO recovery FAILED: 1/3 recovered")

    import src.anchor.runner as runner_module

    monkeypatch.setattr(runner_module, "run_anchor", fake_run_anchor)

    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    _force_convergence(client, sid)

    r = client.post(f"/api/session/{sid}/harvest/run", json={"confirm": True})
    assert r.status_code == 200

    _wait_for_phase(client, sid, "ANCHOR_FAILED")
    r = client.get(f"/api/session/{sid}/harvest/result")
    assert r.status_code == 422
    assert "LOO recovery FAILED" in r.json()["detail"]


def test_harvest_run_concurrent_rejected(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A second ``/harvest/run`` while the first is in flight returns 409."""
    import threading

    started = threading.Event()
    release = threading.Event()
    fake = _build_fake_anchor_result()

    def fake_run_anchor(session_target, **kwargs):  # noqa: ANN001
        started.set()
        release.wait(timeout=5.0)
        return fake

    import src.anchor.runner as runner_module

    monkeypatch.setattr(runner_module, "run_anchor", fake_run_anchor)

    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    _force_convergence(client, sid)

    r1 = client.post(f"/api/session/{sid}/harvest/run", json={"confirm": True})
    assert r1.status_code == 200
    assert started.wait(timeout=2.0)

    r2 = client.post(f"/api/session/{sid}/harvest/run", json={"confirm": True})
    assert r2.status_code == 409

    release.set()
    _wait_for_phase(client, sid, "ANCHOR_DONE")


# ---------------------------------------------------------------------------
# Phase 3 (Refine) — preflight, derive, rubric, judge, decisions, finalize
# ---------------------------------------------------------------------------


def _build_fake_rubric_metadata(version: int = 1) -> Any:
    """Construct a real :class:`RubricMetadata` for tests. Using the
    Pydantic model (not a ``SimpleNamespace``) means ``model_copy``
    works for code paths like ``/refine/discard`` that rely on it.
    """
    from src.refine.schema import (
        RubricCheck,
        RubricFitExample,
        RubricMetadata,
        RubricNotFitExample,
    )

    return RubricMetadata(
        query="test query",
        source_session_id="sid-fake",
        derive_model_id="fake-model",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="0" * 64,
        prompt_path="runs/fake.phase3.prompt.md",
        prompt_sha256="1" * 64,
        version=version,
        checks=[
            RubricCheck(
                id="cites_concept",
                description="Mentions the concept",
                required=True,
            )
        ],
        fit_examples=[RubricFitExample(pk="fp1", span_text="positive example text")],
        not_fit_examples=[
            RubricNotFitExample(
                pk="np1",
                span_text="negative example text",
                fails=["cites_concept"],
            )
        ],
    )


def _build_fake_refine_state(version: int = 1) -> Any:
    """RefineRunState-shaped object for tests. ``judge_result`` and
    ``write_result`` start None; tests populate them as needed.
    """
    from types import SimpleNamespace

    meta = _build_fake_rubric_metadata(version)
    derive = SimpleNamespace(
        rubric_text="rubric markdown here",
        metadata=meta,
        derive_model_id="fake-model",
        latency_ms=42.0,
        attempts=1,
    )
    return SimpleNamespace(
        inputs=SimpleNamespace(
            session_id="sid-fake",
            query="test query",
            fits=[],
            not_fits=[],
            rated_pks=frozenset(),
            reflection_diagnoses=[],
        ),
        cfg=SimpleNamespace(
            enabled=True,
            sample_size=50,
            n_bins=10,
            derive_model="fake-derive",
            judge_model="fake-judge",
        ),
        derive_result=derive,
        rubric_text="rubric markdown here",
        rubric_metadata=meta,
        sample=None,
        judge_result=None,
        write_result=None,
        timings={},
        operator_decision="",
        session_id="sid-fake",
    )


def _build_fake_judge_result() -> Any:
    """JudgeResult-shaped object with two verdicts (one KEEP, one DROP)."""
    from types import SimpleNamespace

    verdicts = [
        SimpleNamespace(
            pk="v1",
            nearest_fit_distance=0.1,
            decile=0,
            chunk_content="content 1",
            verdict="KEEP",
            evidence_line_indices=[0],
            failed_check=None,
            reason="ok",
            latency_ms=10.0,
            attempts=1,
            rubric_version=1,
            prompt_sha256="1" * 64,
        ),
        SimpleNamespace(
            pk="v2",
            nearest_fit_distance=0.5,
            decile=5,
            chunk_content="content 2",
            verdict="DROP",
            evidence_line_indices=None,
            failed_check="cites_concept",
            reason="missing",
            latency_ms=12.0,
            attempts=1,
            rubric_version=1,
            prompt_sha256="1" * 64,
        ),
    ]
    return SimpleNamespace(
        verdicts=verdicts,
        parse_error_count=0,
        api_error_count=0,
        total_latency_ms=22.0,
    )


def _drive_to_anchor_done(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> str:
    """Get a session into ANCHOR_DONE so refine endpoints unlock."""
    fake = _build_fake_anchor_result()

    def fake_run_anchor(session_target, **kwargs):  # noqa: ANN001
        return fake

    import src.anchor.runner as runner_module

    monkeypatch.setattr(runner_module, "run_anchor", fake_run_anchor)

    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    _force_convergence(client, sid)
    r = client.post(f"/api/session/{sid}/harvest/run", json={"confirm": True})
    assert r.status_code == 200
    _wait_for_phase(client, sid, "ANCHOR_DONE")
    return sid


def test_refine_preflight_requires_anchor_done(client: TestClient) -> None:
    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    r = client.post(f"/api/session/{sid}/refine/start")
    assert r.status_code == 409


def test_refine_preflight_returns_dto(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(
        enabled=True,
        sample_size=50,
        n_bins=10,
        derive_model="fake-derive",
        judge_model="fake-judge",
    )
    import src.refine.config as cfg_module

    monkeypatch.setattr(cfg_module, "load_refine_config", lambda **_: fake_cfg)

    r = client.post(f"/api/session/{sid}/refine/start")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_size"] == 50
    assert body["derive_model"] == "fake-derive"
    assert body["phase2_count"] >= 0


def test_refine_preflight_disabled_returns_422(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(enabled=False)
    import src.refine.config as cfg_module

    monkeypatch.setattr(cfg_module, "load_refine_config", lambda **_: fake_cfg)

    r = client.post(f"/api/session/{sid}/refine/start")
    assert r.status_code == 422
    assert "disabled" in r.json()["detail"].lower()


def test_refine_derive_completes_and_returns_dto(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    fake_state = _build_fake_refine_state()

    def fake_run_refine_derive(session_target, **kwargs):  # noqa: ANN001
        return fake_state

    import src.refine.runner as runner_module

    monkeypatch.setattr(runner_module, "run_refine_derive", fake_run_refine_derive)

    r = client.post(f"/api/session/{sid}/refine/derive")
    assert r.status_code == 200, r.text
    _wait_for_phase(client, sid, "REFINE_EDITING")

    r = client.get(f"/api/session/{sid}/refine/derive_result")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rubric_text"] == "rubric markdown here"
    assert body["metadata"]["version"] == 1
    assert body["metadata"]["checks"][0]["id"] == "cites_concept"


def test_refine_save_rubric_parse_error(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    fake_state = _build_fake_refine_state()

    def fake_run_refine_derive(session_target, **kwargs):  # noqa: ANN001
        return fake_state

    import src.refine.runner as runner_module

    monkeypatch.setattr(runner_module, "run_refine_derive", fake_run_refine_derive)
    r = client.post(f"/api/session/{sid}/refine/derive")
    assert r.status_code == 200
    _wait_for_phase(client, sid, "REFINE_EDITING")

    # Patch parse to raise.
    from src.refine.errors import RefineParseError
    import src.refine.derive as derive_module

    def bad_parse(text, **kwargs):  # noqa: ANN001
        raise RefineParseError("Missing '## Failed-check enum:' line")

    monkeypatch.setattr(derive_module, "parse_rubric_prompt", bad_parse)

    r = client.post(
        f"/api/session/{sid}/refine/rubric", json={"rubric_text": "garbage"}
    )
    assert r.status_code == 422
    assert "enum" in r.json()["detail"].lower()


def test_refine_save_rubric_bumps_version(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)
    fake_state = _build_fake_refine_state()

    def fake_run_refine_derive(session_target, **kwargs):  # noqa: ANN001
        return fake_state

    import src.refine.runner as runner_module

    monkeypatch.setattr(runner_module, "run_refine_derive", fake_run_refine_derive)
    r = client.post(f"/api/session/{sid}/refine/derive")
    _wait_for_phase(client, sid, "REFINE_EDITING")

    new_meta = _build_fake_rubric_metadata(version=2)
    import src.refine.derive as derive_module

    monkeypatch.setattr(
        derive_module, "parse_rubric_prompt", lambda text, **kw: new_meta
    )

    r = client.post(f"/api/session/{sid}/refine/rubric", json={"rubric_text": "edited"})
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 2


def test_refine_judge_completes_and_verdicts_returned(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    fake_state = _build_fake_refine_state()

    def fake_run_refine_derive(session_target, **kwargs):  # noqa: ANN001
        return fake_state

    def fake_run_refine_judge(state, **kwargs):  # noqa: ANN001
        state.judge_result = _build_fake_judge_result()
        return state

    import src.refine.runner as runner_module
    from types import SimpleNamespace

    monkeypatch.setattr(runner_module, "run_refine_derive", fake_run_refine_derive)
    monkeypatch.setattr(runner_module, "run_refine_judge", fake_run_refine_judge)

    # Stub Postgres config so judge worker doesn't try to connect.
    import src.postgres.config as pg_cfg_module
    import src.postgres.fetch as pg_fetch_module

    monkeypatch.setattr(
        pg_cfg_module,
        "load_postgres_config",
        lambda: PostgresConfig(
            enabled=True,
            dsn="postgresql://stub/db",
            table="",
            id_column="id",
            content_column="original_content",
            connect_timeout=5,
        ),
    )
    monkeypatch.setattr(
        pg_fetch_module,
        "OriginalContentFetcher",
        lambda cfg: SimpleNamespace(
            fetch_original=lambda pk: "content", close=lambda: None
        ),
    )

    r = client.post(f"/api/session/{sid}/refine/derive")
    assert r.status_code == 200
    _wait_for_phase(client, sid, "REFINE_EDITING")

    r = client.post(f"/api/session/{sid}/refine/judge")
    assert r.status_code == 200, r.text
    _wait_for_phase(client, sid, "REFINE_REVIEW")

    r = client.get(f"/api/session/{sid}/refine/judge_result")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["keep_count"] == 1
    assert body["drop_count"] == 1

    r = client.get(f"/api/session/{sid}/refine/verdicts")
    assert r.status_code == 200, r.text
    verdicts = r.json()
    assert len(verdicts) == 2
    assert {v["pk"] for v in verdicts} == {"v1", "v2"}
    assert all("operator_decision" not in v for v in verdicts)


def test_refine_discard_bumps_rubric_version_and_returns_to_editing(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    fake_state = _build_fake_refine_state()

    def fake_run_refine_derive(session_target, **kwargs):  # noqa: ANN001
        return fake_state

    def fake_run_refine_judge(state, **kwargs):  # noqa: ANN001
        state.judge_result = _build_fake_judge_result()
        return state

    import src.refine.runner as runner_module
    from types import SimpleNamespace

    monkeypatch.setattr(runner_module, "run_refine_derive", fake_run_refine_derive)
    monkeypatch.setattr(runner_module, "run_refine_judge", fake_run_refine_judge)

    import src.postgres.config as pg_cfg_module
    import src.postgres.fetch as pg_fetch_module

    monkeypatch.setattr(
        pg_cfg_module,
        "load_postgres_config",
        lambda: PostgresConfig(
            enabled=True,
            dsn="postgresql://stub/db",
            table="",
            id_column="id",
            content_column="original_content",
            connect_timeout=5,
        ),
    )
    monkeypatch.setattr(
        pg_fetch_module,
        "OriginalContentFetcher",
        lambda cfg: SimpleNamespace(
            fetch_original=lambda pk: "content", close=lambda: None
        ),
    )

    client.post(f"/api/session/{sid}/refine/derive")
    _wait_for_phase(client, sid, "REFINE_EDITING")
    client.post(f"/api/session/{sid}/refine/judge")
    _wait_for_phase(client, sid, "REFINE_REVIEW")

    before = client.get(f"/api/session/{sid}/refine/derive_result").json()
    base_version = before["metadata"]["version"]

    r = client.post(f"/api/session/{sid}/refine/discard")
    assert r.status_code == 200, r.text
    new_meta = r.json()
    assert new_meta["version"] == base_version + 1

    snap = client.get(f"/api/session/{sid}").json()
    assert snap["phase"] == "REFINE_EDITING"

    # Verdicts are cleared after discard.
    r = client.get(f"/api/session/{sid}/refine/verdicts")
    assert r.status_code == 404

    # The per-card decision endpoint no longer exists. FastAPI maps a
    # POST to an unregistered subpath under an existing GET route to
    # 405; either 404 or 405 confirms the endpoint is gone.
    r = client.post(
        f"/api/session/{sid}/refine/verdicts/v2/decision",
        json={"decision": "disagree"},
    )
    assert r.status_code in (404, 405)


def test_refine_finalize_returns_summary(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _drive_to_anchor_done(client, monkeypatch)

    fake_state = _build_fake_refine_state()

    def fake_run_refine_derive(session_target, **kwargs):  # noqa: ANN001
        return fake_state

    def fake_run_refine_judge(state, **kwargs):  # noqa: ANN001
        state.judge_result = _build_fake_judge_result()
        return state

    def fake_finalize(state, **kwargs):  # noqa: ANN001
        from types import SimpleNamespace

        state.write_result = SimpleNamespace(
            prompt_path="/tmp/sid.phase3.prompt.md",
            rubric_path="/tmp/sid.phase3.rubric.json",
            evidence_path="/tmp/sid.phase3.evidence.jsonl",
            meta_path="/tmp/sid.phase3.meta.json",
            details_path="/tmp/sid.phase3.details.jsonl",
            n_verdicts=2,
        )
        state.operator_decision = kwargs.get("operator_decision", "agree")
        return state

    import src.refine.runner as runner_module
    from types import SimpleNamespace

    monkeypatch.setattr(runner_module, "run_refine_derive", fake_run_refine_derive)
    monkeypatch.setattr(runner_module, "run_refine_judge", fake_run_refine_judge)
    monkeypatch.setattr(runner_module, "finalize_refine", fake_finalize)

    import src.postgres.config as pg_cfg_module
    import src.postgres.fetch as pg_fetch_module

    monkeypatch.setattr(
        pg_cfg_module,
        "load_postgres_config",
        lambda: PostgresConfig(
            enabled=True,
            dsn="postgresql://stub/db",
            table="",
            id_column="id",
            content_column="original_content",
            connect_timeout=5,
        ),
    )
    monkeypatch.setattr(
        pg_fetch_module,
        "OriginalContentFetcher",
        lambda cfg: SimpleNamespace(
            fetch_original=lambda pk: "content", close=lambda: None
        ),
    )

    client.post(f"/api/session/{sid}/refine/derive")
    _wait_for_phase(client, sid, "REFINE_EDITING")
    client.post(f"/api/session/{sid}/refine/judge")
    _wait_for_phase(client, sid, "REFINE_REVIEW")

    r = client.post(f"/api/session/{sid}/refine/finalize")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["keep_count"] == 1
    assert body["drop_count"] == 1
    assert body["sidecar_paths"]["prompt"].endswith(".prompt.md")
    assert body["sidecar_paths"]["evidence"].endswith(".evidence.jsonl")
    assert body["operator_decision"] == "agree"

    snap = client.get(f"/api/session/{sid}").json()
    assert snap["phase"] == "DONE"


# ---------------------------------------------------------------------------
# Per-scope Postgres table routing (regression: session 5d035450)
#
# A scope's optional ``postgres_table`` overrides the global
# ``postgres.table`` in config.yaml for chunk-content lookups. Without
# this, sessions on the ``parent_mentioned_comps`` Milvus collection
# query ``parent_chunks`` and get zero rows for every PK → all 500
# rubric verdicts return ERROR/empty_chunk_content.
# ---------------------------------------------------------------------------


_OVERRIDE_SCOPE_NAME = "Comps Override"
_OVERRIDE_TABLE = "parent_mentioned_comps"
_GLOBAL_DEFAULT_TABLE = "parent_chunks"


def _registry_with_override() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name=_TEST_SCOPE_NAME,
                description="Default scope; postgres_table matches the global default.",
                milvus_collection=_TEST_SCOPE_COLLECTION,
                postgres_table=_GLOBAL_DEFAULT_TABLE,
            ),
            Scope(
                name=_OVERRIDE_SCOPE_NAME,
                description="Scope whose Milvus PKs live in a non-default table.",
                milvus_collection="comps_milvus_collection",
                postgres_table=_OVERRIDE_TABLE,
            ),
        )
    )


def _make_client_with_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: ScopeRegistry,
    *,
    pg_fetcher: object | None = None,
) -> TestClient:
    """Build a TestClient with a caller-supplied scope registry.

    Mirrors the default ``client`` fixture's external-service stubs so
    sessions advance without touching Milvus/BGE-M3/LLMs.
    """
    base_config = _base_config()

    def fake_probe_only(query, config, **_kwargs):  # noqa: ANN001
        p = type("_P", (), {})()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    def fake_run_search(query, config, **_kwargs):  # noqa: ANN001
        return _fake_table(query, config, prefix="t1")

    monkeypatch.setattr(app_module, "probe_only", fake_probe_only)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(
        runs_dir=tmp_path,
        reflection_factory=lambda: None,
        extractor_factory=lambda sid: None,
        base_config=base_config,
    )
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]

    app = create_app(
        store=store,
        pg_fetcher=pg_fetcher,  # type: ignore[arg-type]
        scopes=registry,
        users=_test_users(),
        session_secret="test-secret-not-for-prod",
    )
    test_client = TestClient(app)
    res = test_client.post("/api/auth/login", json={"token": _TEST_USER_TOKEN})
    assert res.status_code == 200, (
        f"test fixture login failed: {res.status_code} {res.text}"
    )
    return test_client


def test_expand_uses_scope_routed_postgres_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``/expand`` must query the scope-routed Postgres table, not the global default."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    captured_cfgs: list[Any] = []

    class _CapturingFetcher:
        def __init__(self, cfg: Any) -> None:  # noqa: ANN401
            captured_cfgs.append(cfg)
            self.cfg = cfg

        def fetch_original(self, pk: object) -> str:  # noqa: ARG002
            return f"content for table={self.cfg.table}"

        def close(self) -> None:
            pass

    @dataclass
    class _StubPGConfig:
        enabled: bool
        dsn: str
        table: str
        id_column: str
        content_column: str
        connect_timeout: int

    monkeypatch.setattr("src.postgres.fetch.OriginalContentFetcher", _CapturingFetcher)
    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: _StubPGConfig(
            enabled=True,
            dsn="postgresql://stub/db",
            table=_GLOBAL_DEFAULT_TABLE,
            id_column="id",
            content_column="original_content",
            connect_timeout=5,
        ),
    )

    # Provide a singleton default fetcher so the disabled/unconfigured
    # guards pass. The override path must NOT route through it.
    default_fetcher = SimpleNamespace(
        fetch_original=lambda pk: f"default:{pk}", close=lambda: None
    )

    client = _make_client_with_registry(
        monkeypatch,
        tmp_path,
        _registry_with_override(),
        pg_fetcher=default_fetcher,
    )

    sid = client.post(
        "/api/session",
        json={"query": "q", "scope": _OVERRIDE_SCOPE_NAME},
    ).json()["session_id"]

    r = client.get(f"/api/session/{sid}/chunks/some-pk/original")
    assert r.status_code == 200, r.text
    assert r.json()["original_content"] == f"content for table={_OVERRIDE_TABLE}"
    assert captured_cfgs, "expected scope-routed fetcher to be constructed"
    assert captured_cfgs[-1].table == _OVERRIDE_TABLE


def test_expand_uses_scope_routed_table_for_default_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every scope declares its own ``postgres_table`` (no global
    fallback) — the default scope's table is honoured just like any
    other override."""
    from types import SimpleNamespace

    captured_cfgs: list[Any] = []

    class _CapturingFetcher:
        def __init__(self, cfg: Any) -> None:  # noqa: ANN401
            captured_cfgs.append(cfg)

        def fetch_original(self, pk: object) -> str:
            return f"content for table={cfg.table}"  # noqa: F821 — lambdaland

        def close(self) -> None:
            pass

    # Bind the captured config back to fetch_original so the assertion
    # below can verify the table the scope routed to.
    def _make_fetcher(cfg: Any) -> Any:  # noqa: ANN401
        captured_cfgs.append(cfg)

        def _fetch(pk: object) -> str:
            return f"content for table={cfg.table}"

        return SimpleNamespace(fetch_original=_fetch, close=lambda: None)

    monkeypatch.setattr("src.postgres.fetch.OriginalContentFetcher", _make_fetcher)

    client = _make_client_with_registry(
        monkeypatch,
        tmp_path,
        _registry_with_override(),
        pg_fetcher=SimpleNamespace(
            fetch_original=lambda _pk: "global-singleton-not-used",
            close=lambda: None,
        ),
    )
    sid = client.post(
        "/api/session",
        json={"query": "q", "scope": _TEST_SCOPE_NAME},
    ).json()["session_id"]

    r = client.get(f"/api/session/{sid}/chunks/p1/original")
    assert r.status_code == 200, r.text
    assert r.json()["original_content"] == f"content for table={_GLOBAL_DEFAULT_TABLE}"
    assert captured_cfgs and captured_cfgs[-1].table == _GLOBAL_DEFAULT_TABLE


def test_judge_worker_constructs_fetcher_with_scope_routed_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge worker must override ``pg_cfg.table`` when given ``pg_table_override``."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    from src.web_api.app import _run_refine_judge_worker  # type: ignore[attr-defined]

    captured_cfgs: list[Any] = []

    @dataclass
    class _StubPGConfig:
        enabled: bool
        dsn: str
        table: str
        id_column: str
        content_column: str
        connect_timeout: int

    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: _StubPGConfig(
            enabled=True,
            dsn="postgresql://stub/db",
            table=_GLOBAL_DEFAULT_TABLE,
            id_column="id",
            content_column="original_content",
            connect_timeout=5,
        ),
    )

    class _CapturingFetcher:
        def __init__(self, cfg: Any) -> None:  # noqa: ANN401
            captured_cfgs.append(cfg)

        def close(self) -> None:
            pass

    monkeypatch.setattr("src.postgres.fetch.OriginalContentFetcher", _CapturingFetcher)
    monkeypatch.setattr(
        "src.refine.runner.run_refine_judge",
        lambda state, **kwargs: state,
    )

    # Bare-minimum SessionContext stand-in: only attributes touched by
    # the worker are populated.
    progress = SimpleNamespace(
        set_processed=lambda *_: None,
        set_stage=lambda *_a, **_k: None,
        set_error=lambda *_: None,
        snapshot=lambda: {"total": 0},
    )
    state = SimpleNamespace(
        refine_state=SimpleNamespace(judge_result=None),
        scope=_OVERRIDE_SCOPE_NAME,
        set_phase=lambda *_: None,
    )
    ctx = SimpleNamespace(
        state=state,
        progress=progress,
        lock=__import__("threading").Lock(),
        user_id="alice",
    )

    _run_refine_judge_worker(ctx, runs_dir=None, pg_table_override=_OVERRIDE_TABLE)

    assert captured_cfgs, "judge worker must construct a fetcher"
    assert captured_cfgs[-1].table == _OVERRIDE_TABLE


def test_judge_worker_fails_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the global ``postgres.table`` dropped, the judge worker
    requires a scope-routed override — passing ``None`` should mark the
    session REFINE_FAILED rather than silently querying an empty
    table."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    from src.web_api.app import _run_refine_judge_worker  # type: ignore[attr-defined]

    @dataclass
    class _StubPGConfig:
        enabled: bool
        dsn: str
        table: str
        id_column: str
        content_column: str
        connect_timeout: int

    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: _StubPGConfig(
            enabled=True,
            dsn="postgresql://stub/db",
            table="",
            id_column="id",
            content_column="original_content",
            connect_timeout=5,
        ),
    )

    constructed_fetchers: list[Any] = []

    class _CapturingFetcher:
        def __init__(self, cfg: Any) -> None:  # noqa: ANN401
            constructed_fetchers.append(cfg)

        def close(self) -> None:
            pass

    monkeypatch.setattr("src.postgres.fetch.OriginalContentFetcher", _CapturingFetcher)

    set_phase_calls: list[str] = []
    error_messages: list[str] = []
    progress = SimpleNamespace(
        set_processed=lambda *_: None,
        set_stage=lambda *_a, **_k: None,
        set_error=lambda msg: error_messages.append(msg),
        snapshot=lambda: {"total": 0},
    )
    state = SimpleNamespace(
        refine_state=SimpleNamespace(judge_result=None),
        scope=_TEST_SCOPE_NAME,
        set_phase=lambda phase: set_phase_calls.append(phase),
    )
    ctx = SimpleNamespace(
        state=state,
        progress=progress,
        lock=__import__("threading").Lock(),
        user_id="alice",
    )

    _run_refine_judge_worker(ctx, runs_dir=None, pg_table_override=None)

    assert set_phase_calls == ["REFINE_FAILED"]
    assert error_messages and "scope-routed Postgres table" in error_messages[0]
    assert constructed_fetchers == []


def test_spa_serves_index_html_for_root(tmp_path, monkeypatch):
    """When web/dist exists, GET / returns index.html bytes."""
    from src.web_api.app import create_app

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!doctype html><title>x</title>", encoding="utf-8"
    )
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("/*js*/", encoding="utf-8")

    monkeypatch.setenv("DEKA_WEB_DIST", str(dist))

    app = create_app()
    client = TestClient(app)

    # Root falls back to index.html.
    r = client.get("/")
    assert r.status_code == 200
    assert "<!doctype html>" in r.text

    # Unknown client-side route also falls back to index.html (SPA).
    r2 = client.get("/some/client/route")
    assert r2.status_code == 200
    assert "<!doctype html>" in r2.text

    # Asset served from /assets/.
    r3 = client.get("/assets/app.js")
    assert r3.status_code == 200
    assert "/*js*/" in r3.text

    # /api/* still 404s on unknown endpoints (does NOT fall through to index.html).
    r4 = client.get("/api/does-not-exist")
    assert r4.status_code == 404
    assert "<!doctype html>" not in r4.text


def test_read_session_scope_and_query_matches_canonical_writer(tmp_path: Path) -> None:
    # Regression: _read_session_scope_and_query must read the same field
    # name (`scope`) that ProgressLogWriter writes to <sid>.jsonl. A prior
    # mismatch ("user_scope") made every artifacts download 422 because the
    # bound scope resolved to None.
    sid = "abc12345-deadbeef"
    canonical = {
        "turn": 1,
        "timestamp": "2026-05-20T10:00:00Z",
        "session_id": sid,
        "scope": "user_chats",
        "query": "what triggered the regression",
    }
    (tmp_path / f"{sid}.jsonl").write_text(
        json.dumps(canonical) + "\n", encoding="utf-8"
    )

    scope, query = _read_session_scope_and_query(tmp_path, sid)

    assert scope == "user_chats"
    assert query == "what triggered the regression"
