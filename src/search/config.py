"""Search configuration loading and validation.

Reads the tunable parameter set from the unified ``config.yaml``
(section ``search``) and applies environment-variable overrides for
service endpoints. See ``docs/INFRA.md`` for the endpoint contracts and
the documented env vars, and ``harness/constraints/RULES.md`` for the
tunable axes.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config_loader import ConfigFileError, load_section

from .errors import ConfigError
from .evidence import PathName

log = logging.getLogger(__name__)

_SECTION = "search"

_ENV_EMBED_URL = "DEKA_EMBED_URL"
_ENV_MILVUS_URI = "DEKA_MILVUS_URI"

_ALLOWED_TOP_KEYS = frozenset(
    {
        "rrf_k",
        "per_path_limit",
        "top_k",
        "min_survivors",
        "active_paths",
        "embed_url",
        "milvus_uri",
        "http_timeout",
    }
)
# Every allowed key is required — the Milvus collection is no longer
# declared globally (it lives on the chosen scope, see
# ``scopes.yaml``); session creation overrides ``collection`` via
# :func:`with_overrides` before any retrieval call.
_REQUIRED_TOP_KEYS = _ALLOWED_TOP_KEYS

_VALID_PATHS: frozenset[PathName] = frozenset({"dense", "sparse"})
_ALL_ACTIVE: frozenset[PathName] = frozenset({"dense", "sparse"})


@dataclass(frozen=True)
class SearchConfig:
    """A single tuning-turn configuration.

    Fusion is always RRFRanker. The tunable axes are ``rrf_k``,
    ``per_path_limit``, ``top_k``, and ``active_paths`` (the subset of
    retrieval paths that feed fusion) — see
    ``harness/constraints/RULES.md``. Service endpoints are not tunable
    but are carried here so callers do not need a second config object.

    Field defaults exist only to keep in-process construction (tests,
    replay of sessions whose JSONL predates endpoint capture) ergonomic.
    The production loader :func:`load_default_config` is strict and
    requires every field to be set in ``defaults.yaml``.
    """

    rrf_k: int
    per_path_limit: int
    top_k: int
    active_paths: frozenset[PathName] = field(default_factory=lambda: _ALL_ACTIVE)
    embed_url: str = "http://localhost:8000"
    milvus_uri: str = "http://localhost:19530"
    # Resolved per-session from the chosen scope's
    # ``milvus_collection`` (see ``scopes.yaml`` +
    # :func:`src.web_api.sessions.SessionStore.create`). Empty here is
    # never a runtime value — every code path that touches Milvus
    # operates on a ``SessionConfig`` produced via :func:`with_overrides`
    # which fills this in. Tests and replay paths set it explicitly.
    collection: str = ""
    http_timeout: int = 30
    # Minimum fused-row count after the short-chunk + sample-id filters.
    # If a turn falls below this, ``run_search`` automatically doubles
    # ``per_path_limit`` (up to 3 attempts) so the operator always has a
    # candidate set to rate even when the chosen scope's collection is
    # small or filter-heavy. See harness/constraints/RULES.md.
    # Default 1 (no auto-retry) for in-process construction in tests; the
    # production loader requires an explicit value in ``config.yaml``.
    min_survivors: int = 1


def load_default_config(
    path: Path | None = None,
    session_overrides: dict[str, Any] | None = None,
) -> SearchConfig:
    """Load the starting configuration from the unified config.

    Raises :class:`ConfigError` on any parse or validation failure.
    Endpoint fields are taken from environment variables if set, falling
    back to the YAML values documented in ``docs/INFRA.md``.

    ``collection`` is intentionally left blank here — it is resolved at
    session-creation time from the chosen scope's
    ``milvus_collection`` (see ``scopes.yaml``).

    ``session_overrides`` is a curated dict of per-session knobs (see
    :mod:`src.web_api.schemas.SessionOverrides`); when present its
    values shadow the YAML defaults for this load only.
    """

    try:
        raw = load_section(_SECTION, explicit=path, overrides=session_overrides)
    except ConfigFileError as exc:
        raise ConfigError(str(exc)) from exc

    missing = _REQUIRED_TOP_KEYS - raw.keys()
    if missing:
        raise ConfigError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}. "
            f"Required: {sorted(_REQUIRED_TOP_KEYS)}"
        )
    unknown = raw.keys() - _ALLOWED_TOP_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown keys in config section '{_SECTION}': {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_TOP_KEYS)}"
        )

    return _build_config(raw)


def with_overrides(base: SearchConfig, **overrides: Any) -> SearchConfig:
    """Return a new ``SearchConfig`` with selected fields replaced.

    Thin wrapper around :func:`dataclasses.replace` so reflection-step
    code does not need to import the dataclass helper directly.
    """

    return dataclasses.replace(base, **overrides)


def _build_config(raw: dict[str, Any]) -> SearchConfig:
    rrf_k = raw.get("rrf_k")
    if not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ConfigError("'rrf_k' must be a positive integer")

    per_path_limit = raw.get("per_path_limit")
    if not isinstance(per_path_limit, int) or per_path_limit <= 0:
        raise ConfigError("'per_path_limit' must be a positive integer")

    top_k = raw.get("top_k")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ConfigError("'top_k' must be a positive integer")

    if per_path_limit < top_k:
        log.warning(
            "per_path_limit (%d) is smaller than top_k (%d); "
            "fused ranking may be under-populated",
            per_path_limit,
            top_k,
        )

    min_survivors = raw.get("min_survivors")
    if not isinstance(min_survivors, int) or min_survivors <= 0:
        raise ConfigError("'min_survivors' must be a positive integer")
    if min_survivors > top_k:
        raise ConfigError(
            f"'min_survivors' ({min_survivors}) cannot exceed 'top_k' ({top_k}); "
            "asking for more survivors than available rating slots is incoherent"
        )

    active_paths = _coerce_active_paths(raw.get("active_paths"))

    http_timeout = raw.get("http_timeout")
    if not isinstance(http_timeout, int) or http_timeout <= 0:
        raise ConfigError("'http_timeout' must be a positive integer")

    for key in ("embed_url", "milvus_uri"):
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ConfigError(f"'{key}' must be a non-empty string")

    return SearchConfig(
        rrf_k=rrf_k,
        per_path_limit=per_path_limit,
        top_k=top_k,
        active_paths=active_paths,
        embed_url=os.environ.get(_ENV_EMBED_URL, raw["embed_url"]),
        milvus_uri=os.environ.get(_ENV_MILVUS_URI, raw["milvus_uri"]),
        collection="",
        http_timeout=http_timeout,
        min_survivors=min_survivors,
    )


def _coerce_active_paths(raw: Any) -> frozenset[PathName]:
    """Normalize the ``active_paths`` YAML value to a validated frozenset.

    Accepts ``None`` (fall back to all active paths), a list, or a set.
    Rejects empty collections, duplicates, and unknown path names —
    those are user-facing errors, not silent coercions.
    """
    if raw is None:
        return _ALL_ACTIVE
    if not isinstance(raw, (list, set, frozenset, tuple)):
        raise ConfigError(
            f"'active_paths' must be a list of path names; got {type(raw).__name__}"
        )
    items = list(raw)
    if not items:
        raise ConfigError("'active_paths' must contain at least one path name")
    if len(set(items)) != len(items):
        raise ConfigError(f"'active_paths' contains duplicates: {items}")
    unknown = [p for p in items if p not in _VALID_PATHS]
    if unknown:
        raise ConfigError(
            f"'active_paths' contains unknown path(s) {unknown}; "
            f"allowed: {sorted(_VALID_PATHS)}"
        )
    return frozenset(items)
