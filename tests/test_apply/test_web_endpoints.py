"""Web API tests for the Phase 4 endpoints.

Patches the apply runner so no Milvus is touched. Reuses the
existing :mod:`tests.test_web_api` fixture infrastructure for auth +
session bootstrap.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.search.config import SearchConfig
from src.search.evidence import EvidenceRow, EvidenceTable
from src.scopes import Scope, ScopeRegistry
from src.web_api import app as app_module
from src.web_api.app import create_app
from src.web_api.sessions import SessionStore


_TEST_USER_ID = "alice"
_TEST_USER_TOKEN = "alice-secret-token-for-tests"
_TEST_SCOPE_NAME = "Test Scope"
_TEST_SCOPE_COLLECTION = "test_scope_collection"


def _test_users() -> UserRegistry:
    sha = hashlib.sha256(_TEST_USER_TOKEN.encode("utf-8")).hexdigest()
    return UserRegistry(
        users=(
            User(
                id=_TEST_USER_ID,
                token_sha256=sha,
                allowed_scopes=None,
            ),
        )
    )


def _test_registry() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name=_TEST_SCOPE_NAME,
                description="test scope",
                milvus_collection=_TEST_SCOPE_COLLECTION,
                postgres_table=_TEST_SCOPE_COLLECTION,
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
        collection="test_collection",
    )


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """User-scoped runs dir with skeleton Phase 3 sidecars in place."""
    user_dir = tmp_path / _TEST_USER_ID
    user_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, runs_root: Path):
    """TestClient with auth + a runs dir under tmp_path."""

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
        from src.search.adapt import AdaptedConfig

        return AdaptedConfig(config=seed, rationale=[], flags=[])

    def fake_run_search(query, config, **_kwargs):  # noqa: ANN001
        return EvidenceTable(
            query=query,
            config=config,
            rows=[
                EvidenceRow(
                    rank=1,
                    pk="pk-1",
                    chunk_id="c1",
                    chunk_content="hello",
                    sample_id="s1",
                    counselor_id="c1",
                    term="2025q1",
                    source_paths=["dense"],
                    scores={"dense": 0.5, "sparse": 0.0},
                ),
            ],
            per_path_candidates={"dense": [], "sparse": []},
        )

    monkeypatch.setattr(app_module, "probe_only", fake_probe_only)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(
        runs_dir=runs_root,
        reflection_factory=lambda: None,
        extractor_factory=lambda sid: None,
        base_config=_base_config(),
    )
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]

    app = create_app(
        store=store,
        scopes=_test_registry(),
        users=_test_users(),
        session_secret="test-secret",
    )
    test_client = TestClient(app)
    r = test_client.post("/api/auth/login", json={"token": _TEST_USER_TOKEN})
    assert r.status_code == 200
    test_client.store = store  # type: ignore[attr-defined]
    return test_client


def _create_session_at_done(client: TestClient) -> str:
    """Spin up a session and force its phase to DONE.

    Bypasses the full Phase 1→3 walk since we're only exercising
    Phase 4 endpoints here. Sessions in DONE are read-only by
    contract; we flip ``read_only=False`` on the in-memory context so
    apply endpoints can mutate.
    """
    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    store: SessionStore = client.store  # type: ignore[attr-defined]
    ctx = store.get(sid)
    assert ctx is not None
    ctx.state.set_phase("DONE")
    ctx.read_only = False
    return sid


def _seed_phase3_sidecars(runs_dir: Path, sid: str) -> None:
    rubric = {"version": 1, "prompt_sha256": "a" * 64}
    (runs_dir / f"{sid}.phase3.rubric.json").write_text(
        json.dumps(rubric), encoding="utf-8"
    )
    details = {
        "search": {
            "collection": _TEST_SCOPE_COLLECTION,
            "embed_url": "http://embed",
            "embed_model_id": "bge-m3-fake",
        }
    }
    (runs_dir / f"{sid}.details.jsonl").write_text(
        json.dumps(details) + "\n", encoding="utf-8"
    )
    evidence = [
        {
            "pk": f"k-{i}",
            "nearest_fit_distance": 0.1 + i * 0.01,
            "decile": i % 10,
            "verdict": "KEEP" if i % 2 == 0 else "DROP",
        }
        for i in range(20)
    ]
    (runs_dir / f"{sid}.phase3.evidence.jsonl").write_text(
        "\n".join(json.dumps(r) for r in evidence) + "\n", encoding="utf-8"
    )
    cohort = [
        {"pk": f"c-{i}", "nearest_fit_distance": 0.1 + i * 0.005} for i in range(30)
    ]
    (runs_dir / f"{sid}.phase2.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cohort) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# /apply/start preflight
# ---------------------------------------------------------------------------


def test_apply_start_requires_done_phase(client: TestClient) -> None:
    r = client.post("/api/session", json={"query": "q", "scope": _TEST_SCOPE_NAME})
    sid = r.json()["session_id"]
    r = client.post(f"/api/session/{sid}/apply/start")
    assert r.status_code == 409


def test_apply_start_returns_preflight(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, runs_root: Path
) -> None:
    sid = _create_session_at_done(client)
    user_runs = runs_root / _TEST_USER_ID
    _seed_phase3_sidecars(user_runs, sid)

    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(
        enabled=True,
        confidence_threshold=0.7,
        min_precision=0.9,
        embedding_dim=4,
        eval_fraction=0.25,
        eval_n_repeats=5,
        kfold_splits=3,
        seed=0,
        embedding_fetch_batch=512,
    )
    import src.apply.config as apply_cfg_module

    monkeypatch.setattr(apply_cfg_module, "load_apply_config", lambda **_: fake_cfg)

    r = client.post(f"/api/session/{sid}/apply/start")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cohort_count"] == 30
    assert body["labels_count"] == 20
    assert body["confidence_threshold"] == 0.7
    assert body["min_precision"] == 0.9


def test_apply_start_returns_422_when_disabled(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    sid = _create_session_at_done(client)
    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(
        enabled=False,
        confidence_threshold=0.7,
        min_precision=0.9,
        embedding_dim=4,
        eval_fraction=0.25,
        eval_n_repeats=5,
        kfold_splits=3,
        seed=0,
        embedding_fetch_batch=512,
    )
    import src.apply.config as apply_cfg_module

    monkeypatch.setattr(apply_cfg_module, "load_apply_config", lambda **_: fake_cfg)

    r = client.post(f"/api/session/{sid}/apply/start")
    assert r.status_code == 422
    assert "disabled" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /apply/train → /apply/eval → /apply/finalize happy path
# ---------------------------------------------------------------------------


def _wait_for_phase(
    client: TestClient, sid: str, target: str, timeout_s: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/api/session/{sid}")
        assert r.status_code == 200, r.text
        if r.json()["phase"] == target:
            return
        time.sleep(0.05)
    raise AssertionError(f"phase never reached {target!r}")


def test_apply_train_eval_finalize_happy_path(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, runs_root: Path
) -> None:
    sid = _create_session_at_done(client)
    user_runs = runs_root / _TEST_USER_ID
    _seed_phase3_sidecars(user_runs, sid)
    # Seed phase2.meta.json so /apply/summary can surface the session
    # start timestamp. The harvest writer normally drops this on Phase 2
    # completion; the test fixture skips Phase 2 entirely, so we write
    # a minimal stand-in. Used to assert session_started_at is wired
    # end-to-end through the summary DTO.
    phase2_ts = "2026-05-14T15:51:33Z"
    (user_runs / f"{sid}.phase2.meta.json").write_text(
        json.dumps({"session_id": sid, "query": "q", "ts": phase2_ts}),
        encoding="utf-8",
    )

    from src.apply.config import ApplyConfig

    cfg = ApplyConfig(
        enabled=True,
        confidence_threshold=0.5,
        min_precision=0.5,  # low bar so test embedding doesn't fail
        eval_fraction=0.25,
        eval_n_repeats=5,
        kfold_splits=3,
        seed=0,
        embedding_dim=4,
        embedding_fetch_batch=512,
    )

    # Build a deterministic embeddings map so the test's training data
    # is linearly separable.
    import numpy as np

    rng = np.random.default_rng(0)
    embeddings: dict[str, list[float]] = {}
    for i in range(20):
        loc = 1.0 if i % 2 == 0 else -1.0
        embeddings[f"k-{i}"] = list(rng.normal(loc=loc, scale=0.2, size=4))
    for i in range(30):
        loc = 1.0 if i < 15 else -1.0
        embeddings[f"c-{i}"] = list(rng.normal(loc=loc, scale=0.2, size=4))

    def fake_fetcher(pks, *, collection, **_kwargs):  # noqa: ANN001
        return {pk: embeddings[pk] for pk in pks if pk in embeddings}

    # Patch the runner's default fetcher + config so the worker thread
    # uses the deterministic ones.
    import src.apply.config as apply_cfg_module
    import src.apply.runner as apply_runner_module

    monkeypatch.setattr(apply_cfg_module, "load_apply_config", lambda **_: cfg)
    monkeypatch.setattr(apply_runner_module, "load_apply_config", lambda **_: cfg)
    monkeypatch.setattr(apply_runner_module, "fetch_embeddings", fake_fetcher)

    r = client.post(f"/api/session/{sid}/apply/train")
    assert r.status_code == 200, r.text
    _wait_for_phase(client, sid, "APPLY_REVIEW")

    r = client.get(f"/api/session/{sid}/apply/eval")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "projection" in body and "eval_at_threshold" in body
    assert body["projection"]["total"] == 30
    # Histogram payload — raw eval scores + labels so the new
    # calibration UI can redraw the score-distribution chart on every
    # τ move without an extra round-trip.
    eval_at = body["eval_at_threshold"]
    assert isinstance(eval_at["eval_scores"], list)
    assert isinstance(eval_at["eval_labels"], list)
    assert len(eval_at["eval_scores"]) == eval_at["eval_n"]
    assert len(eval_at["eval_labels"]) == eval_at["eval_n"]
    assert all(0.0 <= s <= 1.0 for s in eval_at["eval_scores"])
    assert set(eval_at["eval_labels"]).issubset({0, 1})

    r = client.get(f"/api/session/{sid}/apply/eval?threshold=0.6")
    assert r.status_code == 200
    assert abs(r.json()["projection"]["threshold"] - 0.6) < 1e-6

    r = client.post(
        f"/api/session/{sid}/apply/finalize",
        json={"threshold": 0.5, "allow_low_precision": False},
    )
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["operator_decision"] == "agree"
    assert "labels" in summary["sidecar_paths"]
    # The summary surfaces the original query + session wall-clock
    # bookends used by the post-apply DONE view headline.
    assert summary["query"] == "q"
    assert summary["session_started_at"] == phase2_ts
    assert summary["session_ended_at"] is not None
    assert summary["session_ended_at"] >= phase2_ts

    snap = client.get(f"/api/session/{sid}").json()
    assert snap["phase"] == "DONE"
    assert (user_runs / f"{sid}.phase4.labels.jsonl").exists()

    # Re-hit /apply/summary post-finalize and assert the timing fields
    # remain populated (the endpoint is read-only and must be idempotent).
    r = client.get(f"/api/session/{sid}/apply/summary")
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["query"] == "q"
    assert again["session_started_at"] == phase2_ts
    assert again["session_ended_at"] is not None


def test_apply_summary_handles_missing_phase2_meta(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, runs_root: Path
) -> None:
    """Without phase2.meta.json, session_started_at is None and the
    endpoint still returns 200 (not 500). Phase 4 meta is written by
    finalize so session_ended_at remains populated."""
    sid = _create_session_at_done(client)
    user_runs = runs_root / _TEST_USER_ID
    _seed_phase3_sidecars(user_runs, sid)
    # Intentionally omit phase2.meta.json.

    from src.apply.config import ApplyConfig

    cfg = ApplyConfig(
        enabled=True,
        confidence_threshold=0.5,
        min_precision=0.5,
        eval_fraction=0.25,
        eval_n_repeats=5,
        kfold_splits=3,
        seed=0,
        embedding_dim=4,
        embedding_fetch_batch=512,
    )

    import numpy as np

    rng = np.random.default_rng(0)
    embeddings: dict[str, list[float]] = {}
    for i in range(20):
        loc = 1.0 if i % 2 == 0 else -1.0
        embeddings[f"k-{i}"] = list(rng.normal(loc=loc, scale=0.2, size=4))
    for i in range(30):
        loc = 1.0 if i < 15 else -1.0
        embeddings[f"c-{i}"] = list(rng.normal(loc=loc, scale=0.2, size=4))

    def fake_fetcher(pks, *, collection, **_kwargs):  # noqa: ANN001
        return {pk: embeddings[pk] for pk in pks if pk in embeddings}

    import src.apply.config as apply_cfg_module
    import src.apply.runner as apply_runner_module

    monkeypatch.setattr(apply_cfg_module, "load_apply_config", lambda **_: cfg)
    monkeypatch.setattr(apply_runner_module, "load_apply_config", lambda **_: cfg)
    monkeypatch.setattr(apply_runner_module, "fetch_embeddings", fake_fetcher)

    r = client.post(f"/api/session/{sid}/apply/train")
    assert r.status_code == 200, r.text
    _wait_for_phase(client, sid, "APPLY_REVIEW")

    r = client.post(
        f"/api/session/{sid}/apply/finalize",
        json={"threshold": 0.5, "allow_low_precision": False},
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/api/session/{sid}/apply/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_started_at"] is None
    # phase4.meta.json was written by finalize, so end is still present
    assert body["session_ended_at"] is not None
    # Query is set at session creation, independent of phase2.meta.json
    assert body["query"] == "q"


def test_apply_cancel_clears_state_and_returns_to_done(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, runs_root: Path
) -> None:
    sid = _create_session_at_done(client)
    user_runs = runs_root / _TEST_USER_ID
    _seed_phase3_sidecars(user_runs, sid)

    from src.apply.config import ApplyConfig

    cfg = ApplyConfig(
        enabled=True,
        confidence_threshold=0.5,
        min_precision=0.5,
        eval_fraction=0.25,
        eval_n_repeats=5,
        kfold_splits=3,
        seed=0,
        embedding_dim=4,
        embedding_fetch_batch=512,
    )

    import numpy as np

    rng = np.random.default_rng(0)
    embeddings: dict[str, list[float]] = {}
    for i in range(20):
        loc = 1.0 if i % 2 == 0 else -1.0
        embeddings[f"k-{i}"] = list(rng.normal(loc=loc, scale=0.2, size=4))
    for i in range(30):
        loc = 1.0 if i < 15 else -1.0
        embeddings[f"c-{i}"] = list(rng.normal(loc=loc, scale=0.2, size=4))

    def fake_fetcher(pks, *, collection, **_kwargs):  # noqa: ANN001
        return {pk: embeddings[pk] for pk in pks if pk in embeddings}

    import src.apply.config as apply_cfg_module
    import src.apply.runner as apply_runner_module

    monkeypatch.setattr(apply_cfg_module, "load_apply_config", lambda **_: cfg)
    monkeypatch.setattr(apply_runner_module, "load_apply_config", lambda **_: cfg)
    monkeypatch.setattr(apply_runner_module, "fetch_embeddings", fake_fetcher)

    client.post(f"/api/session/{sid}/apply/train")
    _wait_for_phase(client, sid, "APPLY_REVIEW")

    r = client.post(f"/api/session/{sid}/apply/cancel")
    assert r.status_code == 200
    assert r.json()["phase"] == "DONE"
