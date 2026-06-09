"""Tests for ``GET /api/sessions``.

The listing endpoint walks ``runs/<user_id>/``, classifies each
session via the disk classifier, drops abandoned ones, and returns
the rest sorted by ``last_modified`` desc. Cross-user isolation is
enforced upstream (the ``current_user`` dep + the per-user
``runs_dir_for`` resolver).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.search.config import SearchConfig
from src.scopes import Scope, ScopeRegistry
from src.web_api.app import create_app
from src.web_api.sessions import SessionStore


_ALICE_TOKEN = "alice-token-for-list-tests"
_BOB_TOKEN = "bob-token-for-list-tests"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _users() -> UserRegistry:
    return UserRegistry(
        users=(
            User(id="alice", token_sha256=_sha(_ALICE_TOKEN), allowed_scopes=None),
            User(id="bob", token_sha256=_sha(_BOB_TOKEN), allowed_scopes=None),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name="Foo", description="d", milvus_collection="c1", postgres_table="c1"
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


@pytest.fixture
def app(tmp_path: Path):
    store = SessionStore(runs_dir=tmp_path, base_config=_base_config())
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]
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


def _seed_session(
    runs: Path,
    user: str,
    sid: str,
    *,
    target: str,
    query: str = "q",
    scope: str = "Foo",
    n_turns: int = 1,
    mtime_offset: float = 0.0,
) -> None:
    """Create on-disk artifacts that match a given resume target.

    ``target`` is a ``ResumeTarget`` value string. Picks the minimum
    set of files needed for the classifier to land on that branch.
    """
    user_dir = runs / user
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / f"{sid}.owner").write_text(f"{user}\n", encoding="utf-8")
    canonical = user_dir / f"{sid}.jsonl"
    lines = []
    for i in range(1, n_turns + 1):
        lines.append(
            json.dumps(
                {
                    "turn": i,
                    "session_id": sid,
                    "scope": scope,
                    "query": query,
                }
            )
        )
    if target == "POST_TUNING":
        lines.append(json.dumps({"event": "converged", "turn": n_turns, "ts": "x"}))
    canonical.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if target in ("POST_HARVEST", "POST_RUBRIC", "DONE_VIEW"):
        (user_dir / f"{sid}.phase2.meta.json").write_text(
            json.dumps({"verdict": "PASSED"}), encoding="utf-8"
        )
    if target in ("POST_RUBRIC", "DONE_VIEW"):
        (user_dir / f"{sid}.phase3.rubric.json").write_text(
            json.dumps({"version": 1}), encoding="utf-8"
        )
        (user_dir / f"{sid}.phase3.evidence.jsonl").write_text("{}\n", encoding="utf-8")
    if target == "DONE_VIEW":
        (user_dir / f"{sid}.phase3.meta.json").write_text(
            json.dumps({"operator_decision": "agree"}), encoding="utf-8"
        )
        # Phase 4 finalised → DONE_VIEW (without labels.jsonl the
        # classifier would return APPLY_PENDING).
        (user_dir / f"{sid}.phase4.labels.jsonl").write_text("{}\n", encoding="utf-8")

    # Bump mtime to a deterministic value so sort-order tests stay stable.
    if mtime_offset > 0.0:
        ts = time.time() + mtime_offset
        for entry in user_dir.glob(f"{sid}.*"):
            os.utime(entry, (ts, ts))


# ---------------------------------------------------------------------------


def test_returns_empty_list_when_user_has_no_sessions(app) -> None:
    alice = _client(app, _ALICE_TOKEN)
    res = alice.get("/api/sessions")
    assert res.status_code == 200
    assert res.json() == []


def test_lists_only_callers_sessions(app, tmp_path: Path) -> None:
    _seed_session(tmp_path, "alice", "s_alice_1", target="POST_TUNING")
    _seed_session(tmp_path, "bob", "s_bob_1", target="POST_TUNING")

    alice = _client(app, _ALICE_TOKEN)
    res = alice.get("/api/sessions")
    assert res.status_code == 200
    body = res.json()
    assert [item["session_id"] for item in body] == ["s_alice_1"]

    bob = _client(app, _BOB_TOKEN)
    assert [item["session_id"] for item in bob.get("/api/sessions").json()] == [
        "s_bob_1"
    ]


def test_hides_abandoned_sessions(app, tmp_path: Path) -> None:
    """A session whose canonical jsonl carries no convergence marker
    (and no phase 2 / 3 sidecars) classifies as abandoned and never
    appears in the listing."""
    user_dir = tmp_path / "alice"
    user_dir.mkdir()
    sid = "abandoned"
    (user_dir / f"{sid}.owner").write_text("alice\n", encoding="utf-8")
    (user_dir / f"{sid}.jsonl").write_text(
        json.dumps({"turn": 1, "scope": "Foo", "query": "q"}) + "\n",
        encoding="utf-8",
    )
    alice = _client(app, _ALICE_TOKEN)
    assert alice.get("/api/sessions").json() == []


@pytest.mark.parametrize(
    "target",
    ["POST_TUNING", "POST_HARVEST", "POST_RUBRIC", "DONE_VIEW"],
)
def test_each_resume_target_produces_a_listing_row(
    app, tmp_path: Path, target: str
) -> None:
    _seed_session(tmp_path, "alice", "sX", target=target, query="hello")
    alice = _client(app, _ALICE_TOKEN)
    body = alice.get("/api/sessions").json()
    assert len(body) == 1
    row = body[0]
    assert row["session_id"] == "sX"
    assert row["resume_target"] == target
    assert row["query"] == "hello"
    assert row["scope"] == "Foo"
    assert row["n_turns"] >= 1
    assert row["last_modified"].endswith("Z")


def test_sorted_by_last_modified_desc(app, tmp_path: Path) -> None:
    _seed_session(tmp_path, "alice", "older", target="POST_TUNING", mtime_offset=10.0)
    _seed_session(tmp_path, "alice", "newer", target="POST_TUNING", mtime_offset=100.0)
    alice = _client(app, _ALICE_TOKEN)
    body = alice.get("/api/sessions").json()
    assert [row["session_id"] for row in body] == ["newer", "older"]


def test_unauthenticated_returns_401(app) -> None:
    res = TestClient(app).get("/api/sessions")
    assert res.status_code == 401


def test_has_rubric_reflects_prompt_sidecar(app, tmp_path: Path) -> None:
    _seed_session(tmp_path, "alice", "no_rubric", target="POST_TUNING")
    _seed_session(tmp_path, "alice", "with_rubric", target="POST_TUNING")
    (tmp_path / "alice" / "with_rubric.phase3.prompt.md").write_text(
        "# rubric\n", encoding="utf-8"
    )

    alice = _client(app, _ALICE_TOKEN)
    body = alice.get("/api/sessions").json()
    by_sid = {row["session_id"]: row for row in body}
    assert by_sid["no_rubric"]["has_rubric"] is False
    assert by_sid["with_rubric"]["has_rubric"] is True


def test_has_artifacts_reflects_phase4_labels_sidecar(app, tmp_path: Path) -> None:
    _seed_session(tmp_path, "alice", "no_artifacts", target="POST_TUNING")
    _seed_session(tmp_path, "alice", "with_artifacts", target="POST_TUNING")
    (tmp_path / "alice" / "with_artifacts.phase4.labels.jsonl").write_text(
        '{"pk": "x-y-1-pre-p", "verdict": "KEEP", "p_keep": 0.9}\n',
        encoding="utf-8",
    )

    alice = _client(app, _ALICE_TOKEN)
    body = alice.get("/api/sessions").json()
    by_sid = {row["session_id"]: row for row in body}
    assert by_sid["no_artifacts"]["has_artifacts"] is False
    assert by_sid["with_artifacts"]["has_artifacts"] is True
