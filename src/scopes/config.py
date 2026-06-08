"""Scope config loading and validation.

Reads the operator-curated list of scopes from ``scopes.yaml``
(falling back to the committed ``scopes.yaml.example`` template).
Each scope is a ``{name, description, milvus_collection, postgres_table}``
record — the session-start picker offers the names, and the chosen scope's
``milvus_collection`` selects the Milvus corpus while ``postgres_table``
selects the Postgres table used for chunk-content lookups.

The loader honours the ``DEKA_SCOPES_FILE`` env var so deployments
can point at an alternate path without editing the repo file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_ENV_PATH = "DEKA_SCOPES_FILE"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIMARY = _REPO_ROOT / "scopes.yaml"
_FALLBACK = _REPO_ROOT / "scopes.yaml.example"

_REQUIRED_ENTRY_KEYS = frozenset(
    {"name", "description", "milvus_collection", "postgres_table"}
)
_ALLOWED_ENTRY_KEYS = _REQUIRED_ENTRY_KEYS


class ScopeError(RuntimeError):
    """Raised when the scopes file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Scope:
    """A single corpus scope.

    ``name`` is both the display label and the stable wire key. Renaming
    a ``name`` is a breaking change for in-flight client links.

    ``milvus_collection`` selects the Milvus collection for retrieval;
    ``postgres_table`` selects the Postgres table used for chunk-content
    lookups (the "expand" interaction + the Phase 3 judge). Both are
    required — there is no global fallback in ``config.yaml``.
    """

    name: str
    description: str
    milvus_collection: str
    postgres_table: str


@dataclass(frozen=True)
class ScopeRegistry:
    """Ordered collection of available scopes.

    Order matches the YAML source so the picker UI renders entries in the
    operator-chosen order.
    """

    scopes: tuple[Scope, ...]

    def names(self) -> list[str]:
        return [s.name for s in self.scopes]

    def get(self, name: str) -> Scope:
        for scope in self.scopes:
            if scope.name == name:
                return scope
        raise ScopeError(
            f"Unknown scope {name!r}; available: {self.names()}"
        )

    def __iter__(self):
        return iter(self.scopes)

    def __len__(self) -> int:
        return len(self.scopes)


def _resolve_path(explicit: Path | None) -> Path:
    """Pick which file to read.

    Precedence: explicit argument > ``DEKA_SCOPES_FILE`` env >
    repo-local ``scopes.yaml`` > committed ``scopes.yaml.example``.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get(_ENV_PATH)
    if env:
        return Path(env)
    if _PRIMARY.exists():
        return _PRIMARY
    return _FALLBACK


def load_scopes(path: Path | None = None) -> ScopeRegistry:
    """Load and validate the scopes config.

    Raises :class:`ScopeError` on any missing-file, parse, or schema
    failure. Validation is strict: every entry must carry exactly the
    three required keys, no extras, and no empty/whitespace strings.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise ScopeError(f"scopes file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ScopeError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScopeError(
            f"{resolved}: top-level mapping with key 'scopes' required"
        )
    if "scopes" not in raw:
        raise ScopeError(
            f"{resolved}: missing top-level 'scopes' key"
        )
    entries = raw["scopes"]
    if not isinstance(entries, list) or not entries:
        raise ScopeError(
            f"{resolved}: 'scopes' must be a non-empty list"
        )

    scopes: list[Scope] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ScopeError(
                f"{resolved}: scopes[{index}] must be a mapping"
            )
        missing = _REQUIRED_ENTRY_KEYS - entry.keys()
        if missing:
            raise ScopeError(
                f"{resolved}: scopes[{index}] missing required keys: "
                f"{sorted(missing)}"
            )
        unknown = entry.keys() - _ALLOWED_ENTRY_KEYS
        if unknown:
            raise ScopeError(
                f"{resolved}: scopes[{index}] has unknown keys: "
                f"{sorted(unknown)}"
            )
        for key in _REQUIRED_ENTRY_KEYS:
            value = entry[key]
            if not isinstance(value, str) or not value.strip():
                raise ScopeError(
                    f"{resolved}: scopes[{index}].{key} must be a "
                    f"non-empty string"
                )
        name = entry["name"]
        if name in seen:
            raise ScopeError(
                f"{resolved}: duplicate scope name {name!r}"
            )
        seen.add(name)
        scopes.append(
            Scope(
                name=name,
                description=entry["description"],
                milvus_collection=entry["milvus_collection"],
                postgres_table=entry["postgres_table"],
            )
        )
    return ScopeRegistry(scopes=tuple(scopes))


def _resolved_path_for_test(explicit: Path | None = None) -> Path:
    """Internal accessor used by tests to assert path-resolution priority."""
    return _resolve_path(explicit)


__all__ = [
    "Scope",
    "ScopeError",
    "ScopeRegistry",
    "load_scopes",
]
