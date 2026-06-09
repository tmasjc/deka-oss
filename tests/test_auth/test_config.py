"""Tests for ``src.auth.config`` — invited-user loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.auth.config import (
    User,
    UserAuthError,
    UserRegistry,
    _resolved_path_for_test,
    load_users,
)
from src.scopes import Scope, ScopeRegistry


def _scope_registry(*names: str) -> ScopeRegistry:
    return ScopeRegistry(
        scopes=tuple(
            Scope(
                name=n,
                description=f"desc for {n}",
                milvus_collection=f"col_{n.lower().replace(' ', '_')}",
                postgres_table=f"tbl_{n.lower().replace(' ', '_')}",
            )
            for n in names
        )
    )


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_users(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "users.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy path + lookup
# ---------------------------------------------------------------------------


def test_load_users_reads_full_entries(tmp_path: Path) -> None:
    sha = _sha("a-secret-token")
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{sha}"
    allowed_scopes: ["Parent Full"]
  - id: bob
    token_sha256: "{_sha("b")}"
""",
    )
    registry = load_users(
        path, scope_registry=_scope_registry("Parent Full", "Therapist Lite")
    )

    assert isinstance(registry, UserRegistry)
    assert registry.ids() == ["alice", "bob"]
    assert registry.get("alice") == User(
        id="alice",
        token_sha256=sha,
        allowed_scopes=("Parent Full",),
    )
    # Bob omitted allowed_scopes → None means "any scope".
    assert registry.get("bob").allowed_scopes is None


def test_find_by_token_sha_returns_user_or_none(tmp_path: Path) -> None:
    sha = _sha("plaintext")
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{sha}"
""",
    )
    registry = load_users(path, scope_registry=_scope_registry("Parent Full"))
    assert registry.find_by_token_sha(sha).id == "alice"
    assert registry.find_by_token_sha(_sha("nope")) is None


def test_get_unknown_id_raises(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("x")}"
""",
    )
    registry = load_users(path, scope_registry=_scope_registry("Parent Full"))
    with pytest.raises(UserAuthError, match="Unknown user id"):
        registry.get("ghost")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_missing_users_key(tmp_path: Path) -> None:
    path = _write_users(tmp_path, "scopes: []\n")
    with pytest.raises(UserAuthError, match="missing top-level 'users' key"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_empty_users_list(tmp_path: Path) -> None:
    path = _write_users(tmp_path, "users: []\n")
    with pytest.raises(UserAuthError, match="non-empty list"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_duplicate_id(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("a")}"
  - id: alice
    token_sha256: "{_sha("b")}"
""",
    )
    with pytest.raises(UserAuthError, match="duplicate user id 'alice'"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("a")}"
    role: admin
""",
    )
    with pytest.raises(UserAuthError, match="unknown keys"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_blank_id(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: "   "
    token_sha256: "{_sha("a")}"
""",
    )
    with pytest.raises(UserAuthError, match="must be a non-empty string"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_short_or_non_hex_token_sha(tmp_path: Path) -> None:
    for bad in ("deadbeef", "g" * 64, "A" * 64):
        path = _write_users(
            tmp_path,
            f"""
users:
  - id: alice
    token_sha256: "{bad}"
""",
        )
        with pytest.raises(UserAuthError, match="64-char lowercase hex"):
            load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_normalises_uppercase_sha_no_wait_rejects(tmp_path: Path) -> None:
    """We require lowercase hex specifically. An uppercase digest is
    a foot-gun (case-sensitive comparison at login time would silently
    miss valid tokens), so the loader rejects it instead of normalising.
    """
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("a").upper()}"
""",
    )
    with pytest.raises(UserAuthError, match="64-char lowercase hex"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_allowed_scopes_not_in_registry(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("a")}"
    allowed_scopes: ["Bogus", "Parent Full"]
""",
    )
    with pytest.raises(UserAuthError, match="not declared in scopes"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_empty_allowed_scopes_list(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("a")}"
    allowed_scopes: []
""",
    )
    with pytest.raises(UserAuthError, match="non-empty list, or omitted"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


def test_rejects_duplicate_allowed_scope(tmp_path: Path) -> None:
    path = _write_users(
        tmp_path,
        f"""
users:
  - id: alice
    token_sha256: "{_sha("a")}"
    allowed_scopes: ["Parent Full", "Parent Full"]
""",
    )
    with pytest.raises(UserAuthError, match="duplicate"):
        load_users(path, scope_registry=_scope_registry("Parent Full"))


# ---------------------------------------------------------------------------
# Path-resolution precedence
# ---------------------------------------------------------------------------


def test_path_precedence_explicit_arg_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.yaml"
    monkeypatch.setenv("DEKA_USERS_FILE", str(tmp_path / "env.yaml"))
    assert _resolved_path_for_test(explicit) == explicit


def test_path_precedence_env_beats_repo_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "env.yaml"
    env_path.write_text("users: []\n", encoding="utf-8")
    monkeypatch.setenv("DEKA_USERS_FILE", str(env_path))
    assert _resolved_path_for_test() == env_path


def test_path_precedence_falls_back_to_example_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither an explicit arg nor the env var is set and no
    repo-local users.yaml exists, the loader falls back to the
    committed users.yaml.example so a fresh checkout is bootable."""
    monkeypatch.delenv("DEKA_USERS_FILE", raising=False)
    resolved = _resolved_path_for_test()
    # The fallback file ships with the repo.
    assert resolved.name in ("users.yaml", "users.yaml.example")
    assert resolved.exists()


# ---------------------------------------------------------------------------
# users.yaml.example sanity (it must load against an actual scope file)
# ---------------------------------------------------------------------------


def test_committed_users_example_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ``load_users`` with no overrides on a fresh checkout
    must return a non-empty registry — the committed example file
    is what bootstraps a new deployment."""
    monkeypatch.delenv("DEKA_USERS_FILE", raising=False)
    # Fall back to whichever scope registry the repo ships.
    registry = load_users()
    assert len(registry) >= 1
