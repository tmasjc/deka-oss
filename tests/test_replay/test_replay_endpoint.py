"""Tests for ``POST /api/session/<sid>/replay`` + ``/replay/advance``.

Replay Mode walks a past session read-only starting at Phase 1.
These tests cover the TUNING-only case (no phase-2/3/4 sidecars on
disk) end-to-end through the FastAPI app — phases that *did* run
get their own coverage in ``smoke_tests/replay_mode.py`` against
the on-disk operator session.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.search.adapt import AdaptedConfig
from src.search.config import SearchConfig
from src.search.evidence import EvidenceRow, EvidenceTable
from src.scopes import Scope, ScopeRegistry
from src.session.state import ConvergenceThresholds
from src.web_api import app as app_module
from src.web_api import resume as resume_module
from src.web_api import sessions as sessions_module
from src.web_api.app import create_app
from src.web_api.sessions import SessionStore


_ALICE_TOKEN = "alice-token-replay"
_SCOPE = "Test Scope"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _users() -> UserRegistry:
    return UserRegistry(
        users=(User(id="alice", token_sha256=_sha(_ALICE_TOKEN), allowed_scopes=None),)
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name=_SCOPE,
                description="d",
                milvus_collection="c1",
                postgres_table="c1",
            ),
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


def _fake_table(query: str, config: SearchConfig, *, prefix: str) -> EvidenceTable:
    rows = [
        EvidenceRow(
            rank=i,
            pk=f"{prefix}-pk-{i}",
            chunk_id=f"sample-{i}_C0000001",
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
        return _fake_table(query, config, prefix="t1")

    monkeypatch.setattr(app_module, "probe_only", fake_probe)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    permissive = ConvergenceThresholds(min_fit=1, min_not_fit=0, precision_at_k=0.5)
    monkeypatch.setattr(
        sessions_module,
        "_resolve_convergence_thresholds",
        lambda _overrides: permissive,
    )
    monkeypatch.setattr(
        resume_module,
        "_convergence_from_disk",
        lambda _sid, _runs: permissive,
    )

    store = SessionStore(runs_dir=tmp_path, base_config=base_config)
    return create_app(
        store=store,
        scopes=_scopes(),
        users=_users(),
        session_secret="t",
    )


def _client(app, token: str) -> TestClient:
    c = TestClient(app)
    res = c.post("/api/auth/login", json={"token": token})
    assert res.status_code == 200
    return c


def _drive_to_convergence(client: TestClient) -> str:
    """Create + converge a session, then DELETE the in-memory ctx so
    only the on-disk sidecars remain."""
    res = client.post("/api/session", json={"query": "the query", "scope": _SCOPE})
    assert res.status_code == 200, res.text
    sid = res.json()["session_id"]
    for rank in (1, 2, 3):
        client.post(f"/api/session/{sid}/rate", json={"rank": rank, "rating": "FIT"})
    advance = client.post(f"/api/session/{sid}/turn/next")
    assert advance.status_code == 200, advance.text
    client.delete(f"/api/session/{sid}")
    return sid


def _file_mtimes(user_dir: Path, sid: str) -> dict[str, float]:
    """Snapshot every session-data sidecar's mtime.

    Excludes ``<sid>.log`` — the per-run diagnostic log is operational
    output (request/lifecycle traces from the FastAPI app), not
    session data. Replay legitimately appends to it because attaching
    the context reopens the run-scoped logger.
    """
    out: dict[str, float] = {}
    for entry in user_dir.iterdir():
        if not entry.is_file() or not entry.name.startswith(sid):
            continue
        if entry.name.endswith(".log"):
            continue
        out[entry.name] = entry.stat().st_mtime
    return out


# ---------------------------------------------------------------------------


def test_replay_lands_at_phase1_read_only(app) -> None:
    alice = _client(app, _ALICE_TOKEN)
    sid = _drive_to_convergence(alice)

    res = alice.post(f"/api/session/{sid}/replay")
    assert res.status_code == 200, res.text
    snap = res.json()
    assert snap["session_id"] == sid
    assert snap["phase"] == "TUNING"
    assert snap["replay"] is True
    assert snap["read_only"] is True
    # Turn 1 is the active turn — turn_number reads 1 even though the
    # original session converged after turn 1 (state.turns=[] at the
    # cursor entry, current_table populated).
    assert snap["turn_number"] == 1
    assert len(snap["table"]["rows"]) == 3


def test_replay_rejects_mutations(app) -> None:
    """The existing ``_require_writable`` guard 409s every mutation
    while ``read_only=True``. Replay sets ``read_only=True`` at
    attach, so we get the same guarantee without extra wiring."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _drive_to_convergence(alice)
    alice.post(f"/api/session/{sid}/replay")

    rate = alice.post(f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"})
    assert rate.status_code == 409

    nxt = alice.post(f"/api/session/{sid}/turn/next")
    assert nxt.status_code == 409

    audit = alice.post(f"/api/session/{sid}/audit")
    assert audit.status_code == 409


def test_replay_advance_terminates_at_done(app) -> None:
    """A TUNING-only session has phase_sequence = [TUNING, DONE].
    One advance from the single phase-1 turn drops us into DONE; the
    next is idempotent."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _drive_to_convergence(alice)
    alice.post(f"/api/session/{sid}/replay")

    advance = alice.post(f"/api/session/{sid}/replay/advance")
    assert advance.status_code == 200
    snap = advance.json()
    # Cursor was already at the final phase-1 turn, so this transition
    # walks past TUNING straight to DONE (no phase 2/3/4 sidecars exist).
    assert snap["phase"] == "DONE"
    assert snap["replay"] is True

    again = alice.post(f"/api/session/{sid}/replay/advance")
    assert again.status_code == 200
    assert again.json()["phase"] == "DONE"


def test_replay_advance_requires_replay_session(app) -> None:
    """``/replay/advance`` 409s when called on a non-replay session —
    prevents accidental drift if the frontend dispatches the wrong
    advance hook."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _drive_to_convergence(alice)
    # Plain resume — not replay.
    alice.post(f"/api/session/{sid}/resume")
    res = alice.post(f"/api/session/{sid}/replay/advance")
    assert res.status_code == 409


def test_replay_preserves_disk_state(app) -> None:
    """Replay + a full walk must not modify any sidecar on disk."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _drive_to_convergence(alice)
    user_dir = app.state.store.runs_dir_for("alice")
    # Settle mtimes — _drive_to_convergence wrote the canonical jsonl,
    # so freeze a stable baseline by touching every file to the same
    # epoch before recording.
    fixed = 1_700_000_000.0
    for entry in user_dir.iterdir():
        if entry.is_file() and entry.name.startswith(sid):
            os.utime(entry, (fixed, fixed))
    before = _file_mtimes(user_dir, sid)

    alice.post(f"/api/session/{sid}/replay")
    # Walk forward until terminal (idempotent past DONE).
    for _ in range(5):
        alice.post(f"/api/session/{sid}/replay/advance")

    after = _file_mtimes(user_dir, sid)
    # Same set of files, same mtimes — replay touches nothing on disk.
    assert before.keys() == after.keys()
    for name, mtime in before.items():
        assert after[name] == mtime, f"{name} mtime changed by replay"
