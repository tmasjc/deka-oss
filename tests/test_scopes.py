"""Tests for the scopes config loader and registry.

Covers :func:`load_scopes` (file resolution, validation, env override)
and :class:`ScopeRegistry` (lookup, ordering, error semantics).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.scopes import (
    Scope,
    ScopeError,
    ScopeRegistry,
    load_scopes,
)


VALID_YAML = """
scopes:
  - name: Parent Full
    description: Full parent transcripts
    milvus_collection: parent_chunks
    postgres_table: parent_chunks
  - name: Child Full
    description: Full child transcripts
    milvus_collection: child_chunks
    postgres_table: child_chunks
"""


# ---------------------------------------------------------------------------
# load_scopes — happy path
# ---------------------------------------------------------------------------


def test_load_returns_registry_with_entries(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")

    reg = load_scopes(f)

    assert isinstance(reg, ScopeRegistry)
    assert len(reg) == 2
    assert reg.names() == ["Parent Full", "Child Full"]


def test_load_preserves_yaml_order(tmp_path: Path) -> None:
    yaml_text = """
scopes:
  - name: Z scope
    description: last in list
    milvus_collection: z_coll
    postgres_table: z_coll
  - name: A scope
    description: first alphabetically but second in list
    milvus_collection: a_coll
    postgres_table: a_coll
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    reg = load_scopes(f)

    assert reg.names() == ["Z scope", "A scope"]


def test_registry_get_returns_scope(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")

    reg = load_scopes(f)
    scope = reg.get("Parent Full")

    assert isinstance(scope, Scope)
    assert scope.name == "Parent Full"
    assert scope.description == "Full parent transcripts"
    assert scope.milvus_collection == "parent_chunks"
    assert scope.postgres_table == "parent_chunks"


def test_postgres_table_distinct_from_milvus_collection(tmp_path: Path) -> None:
    """A scope may name a Postgres table that differs from its Milvus
    collection — the loader keeps them separate, no inference."""
    yaml_text = """
scopes:
  - name: Comps
    description: comp-keyword mentions
    milvus_collection: parent_mentioned_comps_milvus
    postgres_table: parent_mentioned_comps_pg
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")

    reg = load_scopes(f)
    scope = reg.get("Comps")

    assert scope.milvus_collection == "parent_mentioned_comps_milvus"
    assert scope.postgres_table == "parent_mentioned_comps_pg"


def test_postgres_table_empty_string_rejected(tmp_path: Path) -> None:
    yaml_text = """
scopes:
  - name: Bad
    description: ok
    milvus_collection: some_coll
    postgres_table: ""
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ScopeError, match="non-empty string"):
        load_scopes(f)


def test_postgres_table_whitespace_rejected(tmp_path: Path) -> None:
    yaml_text = """
scopes:
  - name: Bad
    description: ok
    milvus_collection: some_coll
    postgres_table: "   "
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ScopeError, match="non-empty string"):
        load_scopes(f)


def test_registry_iter_yields_scopes_in_order(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")

    reg = load_scopes(f)
    names = [s.name for s in reg]

    assert names == ["Parent Full", "Child Full"]


# ---------------------------------------------------------------------------
# load_scopes — error paths
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "nope.yaml"
    with pytest.raises(ScopeError, match="not found"):
        load_scopes(f)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text("scopes: [::not yaml::]", encoding="utf-8")
    with pytest.raises(ScopeError, match="Invalid YAML"):
        load_scopes(f)


def test_top_level_not_mapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text("- just a list\n- of strings", encoding="utf-8")
    with pytest.raises(ScopeError, match="top-level mapping"):
        load_scopes(f)


def test_missing_scopes_key_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text("other_key: 1", encoding="utf-8")
    with pytest.raises(ScopeError, match="'scopes'"):
        load_scopes(f)


def test_scopes_not_list_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text("scopes:\n  some: mapping", encoding="utf-8")
    with pytest.raises(ScopeError, match="non-empty list"):
        load_scopes(f)


def test_empty_scopes_list_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text("scopes: []", encoding="utf-8")
    with pytest.raises(ScopeError, match="non-empty list"):
        load_scopes(f)


def test_entry_not_mapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text("scopes:\n  - just a string", encoding="utf-8")
    with pytest.raises(ScopeError, match="must be a mapping"):
        load_scopes(f)


@pytest.mark.parametrize(
    "missing", ["name", "description", "milvus_collection", "postgres_table"]
)
def test_missing_required_field_raises(tmp_path: Path, missing: str) -> None:
    fields = {
        "name": "Parent Full",
        "description": "Full parent transcripts",
        "milvus_collection": "parent_chunks",
        "postgres_table": "parent_chunks",
    }
    fields.pop(missing)
    body = "\n".join(f"    {k}: {v}" for k, v in fields.items())
    f = tmp_path / "scopes.yaml"
    f.write_text(f"scopes:\n  -\n{body}\n", encoding="utf-8")
    with pytest.raises(ScopeError, match="missing required keys"):
        load_scopes(f)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    yaml_text = """
scopes:
  - name: Parent Full
    description: ok
    milvus_collection: parent_chunks
    postgres_table: parent_chunks
    extra_key: nope
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ScopeError, match="unknown keys"):
        load_scopes(f)


@pytest.mark.parametrize(
    "field_name", ["name", "description", "milvus_collection", "postgres_table"]
)
def test_empty_string_field_rejected(tmp_path: Path, field_name: str) -> None:
    fields = {
        "name": "Parent Full",
        "description": "ok",
        "milvus_collection": "parent_chunks",
        "postgres_table": "parent_chunks",
    }
    fields[field_name] = ""
    body = "\n".join(f'    {k}: "{v}"' for k, v in fields.items())
    f = tmp_path / "scopes.yaml"
    f.write_text(f"scopes:\n  -\n{body}\n", encoding="utf-8")
    with pytest.raises(ScopeError, match="non-empty string"):
        load_scopes(f)


def test_whitespace_string_rejected(tmp_path: Path) -> None:
    yaml_text = """
scopes:
  - name: "   "
    description: ok
    milvus_collection: parent_chunks
    postgres_table: parent_chunks
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ScopeError, match="non-empty string"):
        load_scopes(f)


def test_duplicate_name_rejected(tmp_path: Path) -> None:
    yaml_text = """
scopes:
  - name: Same Name
    description: first
    milvus_collection: a_coll
    postgres_table: a_coll
  - name: Same Name
    description: second
    milvus_collection: b_coll
    postgres_table: b_coll
"""
    f = tmp_path / "scopes.yaml"
    f.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ScopeError, match="duplicate scope name"):
        load_scopes(f)


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


def test_registry_get_unknown_raises(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")

    reg = load_scopes(f)
    with pytest.raises(ScopeError, match="Unknown scope"):
        reg.get("Bogus")


def test_registry_names_lists_all_in_order(tmp_path: Path) -> None:
    f = tmp_path / "scopes.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")

    reg = load_scopes(f)
    assert reg.names() == ["Parent Full", "Child Full"]


# ---------------------------------------------------------------------------
# File resolution: env override + .yaml/.example fallback
# ---------------------------------------------------------------------------


def test_env_override_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "from_env.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")

    monkeypatch.setenv("DEKA_SCOPES_FILE", str(f))
    reg = load_scopes()  # no explicit path

    assert reg.names() == ["Parent Full", "Child Full"]


def test_explicit_path_beats_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(VALID_YAML, encoding="utf-8")

    other_yaml = """
scopes:
  - name: Other
    description: from env file
    milvus_collection: other_coll
    postgres_table: other_coll
"""
    env_file = tmp_path / "env_file.yaml"
    env_file.write_text(other_yaml, encoding="utf-8")
    monkeypatch.setenv("DEKA_SCOPES_FILE", str(env_file))

    reg = load_scopes(explicit)

    assert reg.names() == ["Parent Full", "Child Full"]


def test_committed_example_file_is_valid() -> None:
    """The committed scopes.yaml.example must parse cleanly.

    Smoke test for the operator-facing template — guards against landing
    a broken example file.
    """
    repo_root = Path(__file__).resolve().parents[1]
    example = repo_root / "scopes.yaml.example"
    if not example.exists():
        pytest.skip("scopes.yaml.example not yet committed")
    reg = load_scopes(example)
    assert len(reg) >= 1
