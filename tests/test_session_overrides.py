"""Tests for per-session config overrides.

Covers:
- ``src/config_loader.py`` — the merge + sidecar load helpers.
- ``src/{search,anchor,refine,apply}/config.py`` — each phase loader
  accepts a ``session_overrides`` dict and the override wins.
- ``src/web_api/schemas.py:SessionOverrides`` — the Pydantic allow-list
  rejects fixed/infrastructure keys with 422.
- ``POST /api/session`` — happy-path writes the sidecar; 422 on
  disallowed keys; the sidecar is read back by the Phase-1 loader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.anchor.config import load_harvest_config
from src.apply.config import load_apply_config
from src.auth.config import User, UserRegistry
from src.config_loader import (
    ConfigFileError,
    load_section,
    load_session_overrides,
    session_overrides_path,
)
from src.refine.config import load_refine_config
from src.search.adapt import AdaptedConfig
from src.search.config import SearchConfig, load_default_config
from src.search.evidence import EvidenceRow, EvidenceTable
from src.scopes import Scope, ScopeRegistry
from src.web_api import app as app_module
from src.web_api.app import create_app
from src.web_api.schemas import SessionOverrides
from src.web_api.sessions import SessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_USER_ID = "alice"
_USER_TOKEN = "alice-secret-token-for-tests"
_SCOPE_NAME = "Test Scope"
_SCOPE_COLLECTION = "test_collection"


def _users() -> UserRegistry:
    sha = hashlib.sha256(_USER_TOKEN.encode("utf-8")).hexdigest()
    return UserRegistry(
        users=(
            User(id=_USER_ID, token_sha256=sha, allowed_scopes=None),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(
                name=_SCOPE_NAME,
                description="Test scope",
                milvus_collection=_SCOPE_COLLECTION,
                postgres_table=_SCOPE_COLLECTION,
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
        collection=_SCOPE_COLLECTION,
        min_survivors=1,
    )


def _fake_table(query: str, config: SearchConfig) -> EvidenceTable:
    rows = [
        EvidenceRow(
            rank=1,
            pk="pk-1",
            chunk_id="sample-1_C0000001",
            chunk_content="老师: hi\n家长: hello",
            sample_id="sample-1",
            counselor_id="c1",
            term="2025q1",
            source_paths=["dense"],
            scores={"dense": 0.5, "sparse": 0.0},
        )
    ]
    return EvidenceTable(query=query, config=config, rows=rows, per_path_candidates={"dense": [], "sparse": []})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """TestClient with external services patched out. Runs dir = tmp_path."""

    def fake_probe(query, config, **_):  # noqa: ANN001
        class _P:
            pass

        p = _P()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 1},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    captured = {"configs": []}

    def fake_run_search(query, config, **_):  # noqa: ANN001
        captured["configs"].append(config)
        return _fake_table(query, config)

    monkeypatch.setattr(app_module, "probe_only", fake_probe)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(
        runs_dir=tmp_path,
        reflection_factory=lambda: None,
        extractor_factory=lambda sid: None,
        base_config=_base_config(),
    )
    store._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]

    app = create_app(
        store=store,
        scopes=_scopes(),
        users=_users(),
        session_secret="test-secret",
    )
    tc = TestClient(app)
    r = tc.post("/api/auth/login", json={"token": _USER_TOKEN})
    assert r.status_code == 200, r.text
    tc.captured_configs = captured["configs"]  # type: ignore[attr-defined]
    tc.runs_dir = tmp_path  # type: ignore[attr-defined]
    return tc


# ---------------------------------------------------------------------------
# config_loader helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


def test_load_section_merges_overrides(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path, {"search": {"rrf_k": 60, "top_k": 10, "per_path_limit": 20}}
    )
    merged = load_section("search", explicit=path, overrides={"top_k": 50})
    assert merged["top_k"] == 50
    assert merged["rrf_k"] == 60  # untouched
    assert merged["per_path_limit"] == 20


def test_load_section_no_overrides_returns_yaml_block(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"search": {"rrf_k": 7, "top_k": 4}})
    merged = load_section("search", explicit=path)
    assert merged == {"rrf_k": 7, "top_k": 4}


def test_load_session_overrides_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_session_overrides("no-such-sid", tmp_path) == {}


def test_load_session_overrides_reads_sidecar(tmp_path: Path) -> None:
    user_dir = tmp_path / _USER_ID
    user_dir.mkdir()
    sidecar = user_dir / "abc.overrides.json"
    sidecar.write_text(json.dumps({"search": {"top_k": 99}}))
    out = load_session_overrides("abc", tmp_path, user_id=_USER_ID)
    assert out == {"search": {"top_k": 99}}


def test_load_session_overrides_rejects_malformed_json(tmp_path: Path) -> None:
    sidecar = tmp_path / "abc.overrides.json"
    sidecar.write_text("{not json")
    with pytest.raises(ConfigFileError):
        load_session_overrides("abc", tmp_path)


def test_session_overrides_path_scoped(tmp_path: Path) -> None:
    p = session_overrides_path("abc", tmp_path, user_id="alice")
    assert p == tmp_path / "alice" / "abc.overrides.json"


# ---------------------------------------------------------------------------
# Per-phase loaders accept overrides
# ---------------------------------------------------------------------------


def test_search_loader_applies_overrides(tmp_path: Path) -> None:
    body = {
        "search": {
            "rrf_k": 60,
            "per_path_limit": 20,
            "top_k": 10,
            "min_survivors": 1,
            "active_paths": ["dense", "sparse"],
            "embed_url": "http://localhost:8000",
            "milvus_uri": "http://localhost:19530",
            "http_timeout": 30,
        }
    }
    path = _write_yaml(tmp_path, body)
    cfg = load_default_config(path=path, session_overrides={"top_k": 5})
    assert cfg.top_k == 5
    # min_survivors stays at YAML default
    assert cfg.min_survivors == 1


def test_harvest_loader_applies_overrides(tmp_path: Path) -> None:
    body = {
        "harvest": {
            "min_fit": 10,
            "min_not_fit": 3,
            "precision_at_k": 0.7,
            "batch_size": 4000,
            "max_k": 500000,
            "s2c_outlier_multiple": 3.0,
            "radius_scheme": "per_fit",
            "anchor_frequency_gate": 2,
        }
    }
    path = _write_yaml(tmp_path, body)
    cfg = load_harvest_config(path=path, session_overrides={"min_fit": 25})
    assert cfg.min_fit == 25
    assert cfg.precision_at_k == 0.7


def test_refine_loader_applies_overrides(tmp_path: Path) -> None:
    body = {
        "refine": {
            "enabled": True,
            "sample_size": 500,
            "n_bins": 10,
            "seed": 0,
            "max_fit_examples": 6,
            "max_not_fit_examples": 6,
            "meta_prompt_path": "harness/prompts/RUBRIC_DERIVE.md",
            "derive_model": "x",
            "derive_base_url": "http://example.invalid",
            "derive_temperature": 1.0,
            "judge_model": "x",
            "judge_base_url": "http://example.invalid",
            "judge_concurrency": 4,
            "judge_qps_limit": 1.0,
            "judge_tpm_limit": 100000.0,
            "judge_timeout_seconds": 30,
            "judge_max_retries": 2,
            "api_key_env": "FAKE_KEY",
            "auto_drop_known_intruders": True,
        }
    }
    path = _write_yaml(tmp_path, body)
    cfg = load_refine_config(path=path, session_overrides={"sample_size": 250})
    assert cfg.sample_size == 250


def test_apply_loader_applies_overrides(tmp_path: Path) -> None:
    body = {
        "apply": {
            "enabled": True,
            "confidence_threshold": 0.7,
            "min_precision": 0.75,
            "eval_fraction": 0.2,
            "kfold_splits": 5,
            "seed": 0,
            "embedding_dim": 1024,
            "embedding_fetch_batch": 2000,
        }
    }
    path = _write_yaml(tmp_path, body)
    cfg = load_apply_config(
        path=path, session_overrides={"confidence_threshold": 0.85}
    )
    assert cfg.confidence_threshold == 0.85
    assert cfg.min_precision == 0.75


# ---------------------------------------------------------------------------
# Pydantic allow-list (security boundary)
# ---------------------------------------------------------------------------


def test_session_overrides_accepts_allow_listed_keys() -> None:
    SessionOverrides(
        search={"top_k": 50, "active_paths": ["dense"]},
        harvest={
            "min_fit": 20,
            "precision_at_k": 0.85,
            "s2c_outlier_multiple": 2.5,
            "anchor_frequency_gate": 1,
        },
        refine={
            "sample_size": 200,
            "n_bins": 10,
            "seed": 7,
            "max_fit_examples": 4,
            "max_not_fit_examples": 4,
            "auto_drop_known_intruders": False,
        },
        apply={"confidence_threshold": 0.6, "min_precision": 0.6},
    )


@pytest.mark.parametrize(
    "section,bad_key",
    [
        ("search", "milvus_uri"),
        ("search", "embed_url"),
        ("search", "http_timeout"),
        ("harvest", "batch_size"),
        ("harvest", "max_k"),
        ("refine", "judge_model"),
        ("refine", "derive_base_url"),
        ("refine", "api_key_env"),
        ("refine", "enabled"),
        ("apply", "embedding_dim"),
        ("apply", "eval_fraction"),
    ],
)
def test_session_overrides_rejects_fixed_keys(section: str, bad_key: str) -> None:
    with pytest.raises(ValidationError) as ei:
        SessionOverrides(**{section: {bad_key: "anything"}})
    assert bad_key in str(ei.value)


def test_to_sidecar_dict_strips_empty_sections() -> None:
    ov = SessionOverrides(
        search={"top_k": 1},
        harvest=None,
        refine={},  # empty dict treated as no overrides for that phase
        apply={"enabled": False},
    )
    out = ov.to_sidecar_dict()
    assert out == {"search": {"top_k": 1}, "apply": {"enabled": False}}


# ---------------------------------------------------------------------------
# POST /api/session — happy path + 422 boundary
# ---------------------------------------------------------------------------


def test_start_session_writes_overrides_sidecar(client: TestClient) -> None:
    r = client.post(
        "/api/session",
        json={
            "query": "hello",
            "scope": _SCOPE_NAME,
            "session_id": "deadbeef",
            "overrides": {"search": {"top_k": 7}},
        },
    )
    assert r.status_code == 200, r.text
    sidecar = client.runs_dir / _USER_ID / "deadbeef.overrides.json"  # type: ignore[attr-defined]
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload == {"search": {"top_k": 7}}
    # The Phase 1 search config seen by run_search reflects the override.
    captured = client.captured_configs  # type: ignore[attr-defined]
    assert captured[0].top_k == 7


def test_start_session_without_overrides_writes_no_sidecar(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/session",
        json={
            "query": "hello",
            "scope": _SCOPE_NAME,
            "session_id": "nooverrides",
        },
    )
    assert r.status_code == 200, r.text
    sidecar = client.runs_dir / _USER_ID / "nooverrides.overrides.json"  # type: ignore[attr-defined]
    assert not sidecar.exists()


def test_start_session_rejects_fixed_parameter_in_overrides(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/session",
        json={
            "query": "hello",
            "scope": _SCOPE_NAME,
            "session_id": "evil",
            "overrides": {"search": {"milvus_uri": "http://evil.invalid"}},
        },
    )
    assert r.status_code == 422
    sidecar = client.runs_dir / _USER_ID / "evil.overrides.json"  # type: ignore[attr-defined]
    assert not sidecar.exists()


def test_config_defaults_endpoint_returns_allow_listed_keys(
    client: TestClient,
) -> None:
    # /api/config/defaults reads config.yaml from the repo root, which
    # exists in this worktree, so we just check the response shape.
    r = client.get("/api/config/defaults")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"search", "harvest", "refine", "apply"}
    # None of the fixed-key shouldn't-be-here values are present.
    for k in ("milvus_uri", "embed_url", "http_timeout"):
        assert k not in body["search"]
    for k in ("derive_model", "judge_model", "api_key_env"):
        assert k not in body["refine"]
