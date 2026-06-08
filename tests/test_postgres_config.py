"""Unit tests for the ``postgres`` config loader and its enable flag."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.postgres.config import load_postgres_config
from src.search.errors import ConfigError


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_enabled_defaults_true_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEKA_POSTGRES_DSN", raising=False)
    path = _write_yaml(
        tmp_path,
        """
postgres:
  dsn: postgresql://u:p@h/db
  id_column: pk
  content_column: original_content
  connect_timeout: 5
""",
    )
    cfg = load_postgres_config(path)
    assert cfg.enabled is True
    assert cfg.dsn == "postgresql://u:p@h/db"
    assert cfg.connect_timeout == 5
    # ``table`` is intentionally blank in the loaded config — it is
    # resolved per-session from the chosen scope's
    # ``postgres_table`` (see ``scopes.yaml``).
    assert cfg.table == ""


def test_disabled_config_does_not_require_other_keys(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "postgres:\n  enabled: false\n")
    cfg = load_postgres_config(path)
    assert cfg.enabled is False
    # Sentinel empty strings are expected when disabled — callers must not
    # use them because no fetcher will be constructed.
    assert cfg.dsn == ""
    assert cfg.table == ""
    assert cfg.id_column == ""
    assert cfg.content_column == ""
    assert cfg.connect_timeout == 0


def test_enabled_true_still_requires_keys(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "postgres:\n  enabled: true\n")
    with pytest.raises(ConfigError, match="missing required keys"):
        load_postgres_config(path)


def test_enabled_true_rejects_empty_dsn(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
postgres:
  enabled: true
  dsn: ""
  id_column: pk
  content_column: original_content
  connect_timeout: 5
""",
    )
    with pytest.raises(ConfigError, match="'dsn' must be a non-empty string"):
        load_postgres_config(path)


def test_non_bool_enabled_rejected(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "postgres:\n  enabled: yes-please\n")
    with pytest.raises(ConfigError, match="'enabled' must be a boolean"):
        load_postgres_config(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
postgres:
  enabled: false
  bogus: 1
""",
    )
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_postgres_config(path)
