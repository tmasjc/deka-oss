"""Tests for the pre-flight check module + endpoint (issue #33).

Covers:

- Each ``check_*`` function in isolation, with the network-bound ones
  exercised via the ``*_probe`` test seams on :class:`PreflightContext`.
- The ``run_preflight`` orchestrator's ordering + parallelism.
- The ``POST /api/session/preflight`` endpoint's 200 / 400 surfaces.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.search.config import SearchConfig
from src.scopes import Scope, ScopeRegistry
from src.web_api import app as app_module
from src.web_api.app import create_app
from src.web_api.preflight import (
    CHECK_FUNCTIONS,
    PreflightCheckResult,
    PreflightContext,
    check_embed_service,
    check_llm_reflection,
    check_llm_refine_derive,
    check_milvus,
    check_postgres,
    check_prompts,
    check_scopes_yaml,
    check_users_yaml,
    first_failure,
    run_preflight,
)
from src.web_api.sessions import SessionStore


_ALICE_TOKEN = "alice-token-preflight"
_SCOPE = "Test Scope"
_COLLECTION = "c1"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _users() -> UserRegistry:
    return UserRegistry(
        users=(
            User(id="alice", token_sha256=_sha(_ALICE_TOKEN), allowed_scopes=None),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(name=_SCOPE, description="d", milvus_collection=_COLLECTION, postgres_table=_COLLECTION),
        )
    )


def _ctx(**overrides) -> PreflightContext:
    """Build a PreflightContext seeded with passing defaults.

    Each test overrides only the field it needs to flex, keeping
    failures isolated and intent obvious.
    """
    defaults: dict = {
        "scope_name": _SCOPE,
        "users_registry": _users(),
        "scopes_registry": _scopes(),
        "load_section_fn": lambda section: {
            "search": {"embed_url": "http://embed", "milvus_uri": "http://milvus"},
            "reflection": {"api_key_env": "TEST_REFLECTION_KEY"},
            "refine": {"api_key_env": "TEST_REFINE_KEY"},
            "extraction": {"api_key_env": "TEST_EXTRACT_KEY"},
        }[section],
        "embed_probe": lambda url, t: True,
        "milvus_probe": lambda uri, t: [_COLLECTION, "other"],
        "postgres_probe": lambda cfg: (True, "ok"),
    }
    defaults.update(overrides)
    return PreflightContext(**defaults)


# ---------------------------------------------------------------------------
# Per-check unit tests
# ---------------------------------------------------------------------------


def test_users_yaml_passes_when_registry_present() -> None:
    r = check_users_yaml(_ctx())
    assert r.status == "ok"
    assert "1 user(s)" in r.detail


def test_users_yaml_fails_when_registry_missing() -> None:
    r = check_users_yaml(_ctx(users_registry=None))
    assert r.status == "fail"
    assert r.code == "USERS_YAML_MISSING"


def test_scopes_yaml_reports_count() -> None:
    r = check_scopes_yaml(_ctx())
    assert r.status == "ok"
    assert "1 scope(s)" in r.detail


def test_embed_service_passes_when_probe_returns_true() -> None:
    r = check_embed_service(_ctx())
    assert r.status == "ok"
    assert "http://embed" in r.detail


def test_embed_service_fails_on_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_probe(url, t):
        raise requests.ConnectionError("connection refused")

    r = check_embed_service(_ctx(embed_probe=raising_probe))
    assert r.status == "fail"
    assert r.code == "EMBED_UNREACHABLE"
    assert "connection refused" in r.detail


def test_embed_service_fails_when_url_missing() -> None:
    ctx = _ctx(
        load_section_fn=lambda section: {"search": {"embed_url": ""}}.get(
            section, {}
        )
    )
    r = check_embed_service(ctx)
    assert r.status == "fail"
    assert r.code == "EMBED_URL_MISSING"


def test_milvus_passes_when_collection_present() -> None:
    r = check_milvus(_ctx())
    assert r.status == "ok"
    assert _COLLECTION in r.detail


def test_milvus_fails_when_collection_absent() -> None:
    r = check_milvus(_ctx(milvus_probe=lambda uri, t: ["something_else"]))
    assert r.status == "fail"
    assert r.code == "MILVUS_NO_COLLECTION"
    assert _COLLECTION in r.detail


def test_milvus_fails_when_unreachable() -> None:
    def raising_probe(uri, t):
        raise RuntimeError("no route to host")

    r = check_milvus(_ctx(milvus_probe=raising_probe))
    assert r.status == "fail"
    assert r.code == "MILVUS_UNREACHABLE"


def test_postgres_disabled_reports_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When postgres.enabled=false, the check must pass with a
    'disabled' note rather than failing — the operator deliberately
    turned the feature off."""
    from src.postgres.config import PostgresConfig

    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: PostgresConfig(
            enabled=False,
            dsn="",
            table="",
            id_column="",
            content_column="",
            connect_timeout=0,
        ),
    )
    r = check_postgres(_ctx())
    assert r.status == "ok"
    assert "disabled" in r.detail


def test_postgres_fail_table_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.postgres.config import PostgresConfig

    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: PostgresConfig(
            enabled=True,
            dsn="postgresql://x",
            table="parent_chunks",
            id_column="id",
            content_column="content",
            connect_timeout=5,
        ),
    )
    r = check_postgres(
        _ctx(postgres_probe=lambda cfg: (False, "table 'parent_chunks' does not exist"))
    )
    assert r.status == "fail"
    assert r.code == "POSTGRES_NO_TABLE"


def test_postgres_fail_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.postgres.config import PostgresConfig

    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: PostgresConfig(
            enabled=True,
            dsn="postgresql://x",
            table="parent_chunks",
            id_column="id",
            content_column="content",
            connect_timeout=5,
        ),
    )
    r = check_postgres(
        _ctx(postgres_probe=lambda cfg: (False, "connection refused"))
    )
    assert r.status == "fail"
    assert r.code == "POSTGRES_UNREACHABLE"


def test_llm_key_passes_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_REFLECTION_KEY", "secret")
    r = check_llm_reflection(_ctx())
    assert r.status == "ok"
    assert "TEST_REFLECTION_KEY" in r.detail


def test_llm_key_fails_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_REFINE_KEY", raising=False)
    r = check_llm_refine_derive(_ctx())
    assert r.status == "fail"
    assert r.code == "MISSING_LLM_KEY"
    assert r.env_var == "TEST_REFINE_KEY"
    assert "Refine's derive step" in r.detail


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_run_preflight_returns_results_in_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for k in ("TEST_REFLECTION_KEY", "TEST_REFINE_KEY", "TEST_EXTRACT_KEY"):
        monkeypatch.setenv(k, "secret")
    monkeypatch.setattr(
        "src.postgres.config.load_postgres_config",
        lambda: __import__(
            "src.postgres.config", fromlist=["PostgresConfig"]
        ).PostgresConfig(
            enabled=False,
            dsn="",
            table="",
            id_column="",
            content_column="",
            connect_timeout=0,
        ),
    )
    results = run_preflight(_ctx())
    names = [r.name for r in results]
    assert names == list(CHECK_FUNCTIONS.keys())
    assert all(r.status == "ok" for r in results)


def test_first_failure_returns_first_in_order() -> None:
    checks = [
        PreflightCheckResult(name="a", status="ok", detail=""),
        PreflightCheckResult(
            name="b", status="fail", detail="bad", code="X"
        ),
        PreflightCheckResult(
            name="c", status="fail", detail="also bad", code="Y"
        ),
    ]
    f = first_failure(checks)
    assert f is not None
    assert f.name == "b"


def test_first_failure_returns_none_when_all_pass() -> None:
    checks = [PreflightCheckResult(name="a", status="ok", detail="")]
    assert first_failure(checks) is None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _base_config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=3,
        embed_url="http://x",
        milvus_uri="http://x",
        collection=_COLLECTION,
    )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """App fixture that monkeypatches ``run_preflight`` so the endpoint
    test doesn't need a live Milvus / Postgres / embed service.

    Per-test overrides set the patched return value so the 200 / 400
    paths are both exercised.
    """
    store = SessionStore(runs_dir=tmp_path, base_config=_base_config())
    return create_app(
        store=store,
        scopes=_scopes(),
        users=_users(),
        session_secret="t",
    )


def _client(app, token: str = _ALICE_TOKEN) -> TestClient:
    c = TestClient(app)
    res = c.post("/api/auth/login", json={"token": token})
    assert res.status_code == 200
    return c


def test_preflight_endpoint_returns_200_when_all_pass(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_ok = [
        PreflightCheckResult(name=name, status="ok", detail="ok")
        for name in CHECK_FUNCTIONS.keys()
    ]
    monkeypatch.setattr(app_module, "run_preflight", lambda ctx: all_ok)

    alice = _client(app)
    res = alice.post(
        "/api/session/preflight", json={"scope": _SCOPE}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["all_passed"] is True
    assert [c["name"] for c in body["checks"]] == list(CHECK_FUNCTIONS.keys())
    assert all(c["status"] == "ok" for c in body["checks"])


def test_preflight_endpoint_returns_400_with_typed_body_on_failure(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    mixed = [
        PreflightCheckResult(name="prompts", status="ok", detail="4/4"),
        PreflightCheckResult(name="users.yaml", status="ok", detail="1 user"),
        PreflightCheckResult(
            name="scopes.yaml", status="ok", detail="1 scope"
        ),
        PreflightCheckResult(
            name="embed_service",
            status="fail",
            code="EMBED_UNREACHABLE",
            detail="connection refused",
        ),
    ] + [
        PreflightCheckResult(name=n, status="ok", detail="ok")
        for n in (
            "milvus.collection",
            "postgres",
            "llm.reflection",
            "llm.refine.derive",
            "llm.refine.judge",
            "llm.span_extractor",
        )
    ]
    monkeypatch.setattr(app_module, "run_preflight", lambda ctx: mixed)

    alice = _client(app)
    res = alice.post(
        "/api/session/preflight", json={"scope": _SCOPE}
    )
    assert res.status_code == 400, res.text
    body = res.json()["detail"]
    assert body["code"] == "EMBED_UNREACHABLE"
    assert body["phase"] == "embed_service"
    assert body["env_var"] == ""
    assert "connection refused" in body["detail"]
    assert len(body["checks"]) == len(CHECK_FUNCTIONS)
    fail_row = next(c for c in body["checks"] if c["name"] == "embed_service")
    assert fail_row["status"] == "fail"


def test_preflight_endpoint_failure_includes_env_var_for_missing_key(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The typed-error body's ``env_var`` field is what the UI uses to
    name *exactly* which env var to set. Verify it round-trips."""
    mixed = [
        PreflightCheckResult(name=n, status="ok", detail="ok")
        for n in (
            "users.yaml",
            "scopes.yaml",
            "embed_service",
            "milvus.collection",
            "postgres",
            "llm.reflection",
        )
    ] + [
        PreflightCheckResult(
            name="llm.refine.derive",
            status="fail",
            code="MISSING_LLM_KEY",
            env_var="OPENROUTER_API_KEY",
            detail="Refine's derive step needs OPENROUTER_API_KEY in env or .env.",
        ),
        PreflightCheckResult(name="llm.refine.judge", status="ok", detail="ok"),
        PreflightCheckResult(name="llm.span_extractor", status="ok", detail="ok"),
    ]
    monkeypatch.setattr(app_module, "run_preflight", lambda ctx: mixed)

    alice = _client(app)
    res = alice.post(
        "/api/session/preflight", json={"scope": _SCOPE}
    )
    assert res.status_code == 400, res.text
    body = res.json()["detail"]
    assert body["code"] == "MISSING_LLM_KEY"
    assert body["env_var"] == "OPENROUTER_API_KEY"
    assert body["phase"] == "llm.refine.derive"


def test_preflight_endpoint_unknown_scope_returns_422(app) -> None:
    alice = _client(app)
    res = alice.post(
        "/api/session/preflight", json={"scope": "Bogus Scope"}
    )
    assert res.status_code == 422


def test_preflight_endpoint_requires_auth(app) -> None:
    res = TestClient(app).post(
        "/api/session/preflight", json={"scope": _SCOPE}
    )
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# prompts probe
# ---------------------------------------------------------------------------


_REQUIRED_PROMPTS = (
    "SYSTEM.md",
    "REFLECTION.md",
    "EXTRACTION.md",
    "RUBRIC_DERIVE.md",
)


def _seed_prompts(dir_: Path) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for name in _REQUIRED_PROMPTS:
        (dir_ / name).write_text("# stub\n", encoding="utf-8")


def test_prompts_passes_when_dir_and_files_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    _seed_prompts(prompts_dir)
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(prompts_dir))
    r = check_prompts(_ctx())
    assert r.status == "ok"
    assert str(prompts_dir) in r.detail


def test_prompts_fails_when_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(missing))
    r = check_prompts(_ctx())
    assert r.status == "fail"
    assert r.code == "PROMPTS_DIR_MISSING"
    assert r.env_var == "DEKA_PROMPTS_DIR"
    assert str(missing) in r.detail


def test_prompts_fails_when_required_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    _seed_prompts(prompts_dir)
    (prompts_dir / "REFLECTION.md").unlink()
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(prompts_dir))
    r = check_prompts(_ctx())
    assert r.status == "fail"
    assert r.code == "PROMPTS_FILE_MISSING"
    assert r.env_var == "DEKA_PROMPTS_DIR"
    assert "REFLECTION.md" in r.detail


def test_run_preflight_runs_prompts_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    _seed_prompts(prompts_dir)
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(prompts_dir))
    monkeypatch.setenv("TEST_REFLECTION_KEY", "x")
    monkeypatch.setenv("TEST_REFINE_KEY", "x")
    monkeypatch.setenv("TEST_EXTRACT_KEY", "x")
    results = run_preflight(_ctx())
    assert [r.name for r in results][0] == "prompts"
