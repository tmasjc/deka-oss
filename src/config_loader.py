"""Shared entry point for the unified ``config.yaml``.

Every domain (search, extraction, reflection, anchor, convergence, TUI)
keeps its own strict validator, but they all read from the same file
via :func:`load_section`. The loader prefers a local, gitignored
``config.yaml`` and falls back to the committed ``config.yaml.example``
so fresh checkouts work without an explicit copy step.

Per-session overrides are persisted at
``runs/<user_id>/<sid>.overrides.json`` and merged on top of the YAML
defaults at load time. The merge is a shallow dict-merge: every
override value is a scalar, list, or bool — never a nested mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRIMARY = _REPO_ROOT / "config.yaml"
_FALLBACK = _REPO_ROOT / "config.yaml.example"


def resolve_config_path(explicit: Path | None = None) -> Path:
    """Return the config file to read. Explicit paths bypass the fallback."""
    if explicit is not None:
        return explicit
    if _PRIMARY.exists():
        return _PRIMARY
    return _FALLBACK


def load_section(
    section: str,
    explicit: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the unified YAML and return ``raw[section]`` as a dict.

    When ``overrides`` is given, its keys take precedence over the YAML
    values for the returned section. The merge is shallow — callers must
    not pass nested mappings; the curated per-session override set is
    scalars/lists/bools only.

    Raises :class:`ConfigFileError` when the file cannot be read or the
    section is missing. Validation of required/unknown keys is left to
    the caller so each domain reports errors in its own idiom.
    """
    path = resolve_config_path(explicit)
    if not path.exists():
        raise ConfigFileError(f"Required config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigFileError(f"{path} must contain a YAML mapping at the root")
    if section not in raw:
        raise ConfigFileError(
            f"{path} missing required section '{section}'. "
            f"Found sections: {sorted(raw)}"
        )
    block = raw[section]
    if not isinstance(block, dict):
        raise ConfigFileError(
            f"{path}: section '{section}' must be a mapping; got {type(block).__name__}"
        )
    if overrides:
        block = {**block, **overrides}
    return block


def session_overrides_path(
    session_id: str,
    runs_dir: Path,
    user_id: str | None = None,
) -> Path:
    """Return the canonical sidecar path for a session's overrides file."""
    base = runs_dir / user_id if user_id else runs_dir
    return base / f"{session_id}.overrides.json"


def load_session_overrides(
    session_id: str,
    runs_dir: Path,
    user_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{section: overrides_dict}`` from the sidecar, or ``{}``.

    Missing files are treated as "no overrides" — not an error. A
    malformed file raises :class:`ConfigFileError` so a corrupt sidecar
    fails loudly rather than silently reverting to YAML defaults.
    """
    path = session_overrides_path(session_id, runs_dir, user_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigFileError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigFileError(f"{path} must contain a JSON object at the root")
    result: dict[str, dict[str, Any]] = {}
    for section, block in data.items():
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ConfigFileError(
                f"{path}: section '{section}' must be a JSON object; got {type(block).__name__}"
            )
        result[section] = block
    return result


class ConfigFileError(RuntimeError):
    """Raised when the unified config file cannot be read or parsed."""
