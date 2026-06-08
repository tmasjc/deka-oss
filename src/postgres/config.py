"""Postgres configuration loading and validation.

Reads the ``postgres`` section of the unified ``config.yaml`` and
applies the ``DEKA_POSTGRES_DSN`` environment-variable override for the
DSN.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from src.config_loader import ConfigFileError, load_section

from src.search.errors import ConfigError

log = logging.getLogger(__name__)

_SECTION = "postgres"

_ENV_DSN = "DEKA_POSTGRES_DSN"

_ALLOWED_TOP_KEYS = frozenset(
    {
        "enabled",
        "dsn",
        "id_column",
        "content_column",
        "connect_timeout",
    }
)
# ``table`` is no longer declared globally — it lives on the chosen
# scope's ``postgres_table`` (see ``scopes.yaml``). Callers
# that need a concrete table (the /expand fetcher, refine judge,
# preflight) clone this config with the scope override before issuing
# the query.
_REQUIRED_WHEN_ENABLED = frozenset(
    {
        "dsn",
        "id_column",
        "content_column",
        "connect_timeout",
    }
)


@dataclass(frozen=True)
class PostgresConfig:
    """Connection + schema info for original-content lookups.

    ``table`` is empty in the loaded config — it is resolved per-session
    from the scope's ``postgres_table`` and patched in via
    :func:`dataclasses.replace` before any query is issued.
    """

    enabled: bool
    dsn: str
    table: str
    id_column: str
    content_column: str
    connect_timeout: int


def load_postgres_config(path: Path | None = None) -> PostgresConfig:
    """Load the Postgres config block. Raises :class:`ConfigError` on failure.

    When ``enabled: false``, connection fields are optional — an admin
    disabling the feature shouldn't need to keep a valid DSN around.
    Disabled configs return sentinel empty strings that callers must
    not read (the fetcher won't be constructed).
    """

    try:
        raw = load_section(_SECTION, explicit=path)
    except ConfigFileError as exc:
        raise ConfigError(str(exc)) from exc

    unknown = raw.keys() - _ALLOWED_TOP_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown keys in config section '{_SECTION}': {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_TOP_KEYS)}"
        )

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("'enabled' must be a boolean")

    if not enabled:
        return PostgresConfig(
            enabled=False,
            dsn="",
            table="",
            id_column="",
            content_column="",
            connect_timeout=0,
        )

    missing = _REQUIRED_WHEN_ENABLED - raw.keys()
    if missing:
        raise ConfigError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}. "
            f"Required when enabled: {sorted(_REQUIRED_WHEN_ENABLED)}"
        )

    for key in ("dsn", "id_column", "content_column"):
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ConfigError(f"'{key}' must be a non-empty string")

    connect_timeout = raw.get("connect_timeout")
    if not isinstance(connect_timeout, int) or connect_timeout <= 0:
        raise ConfigError("'connect_timeout' must be a positive integer")

    dsn = os.environ.get(_ENV_DSN, raw["dsn"])

    return PostgresConfig(
        enabled=True,
        dsn=dsn,
        table="",
        id_column=raw["id_column"],
        content_column=raw["content_column"],
        connect_timeout=connect_timeout,
    )
