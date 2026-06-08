"""Fetch ``original_content`` from a scope-routed Postgres table.

The fetcher owns a psycopg3 connection pool and is held per-table on
FastAPI ``app.state.pg_fetcher_by_table`` (one entry per scope
``postgres_table``). Its single ``fetch_original`` method runs a
parameterized SELECT keyed by the Milvus primary key.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

from .config import PostgresConfig

log = logging.getLogger(__name__)


class PostgresFetchError(RuntimeError):
    """Raised when Postgres is unreachable or a query fails."""


class OriginalContentFetcher:
    """Look up a chunk's original text by primary key."""

    def __init__(self, config: PostgresConfig, *, pool_max_size: int = 4) -> None:
        self._config = config
        self._pool_max_size = pool_max_size
        self._pool: ConnectionPool | None = None
        self._query = sql.SQL(
            "SELECT {content} FROM {table} WHERE {id_col} = %s LIMIT 1"
        ).format(
            content=sql.Identifier(config.content_column),
            table=sql.Identifier(config.table),
            id_col=sql.Identifier(config.id_column),
        )
        self._bulk_query = sql.SQL(
            "SELECT {id_col}, {content} FROM {table} WHERE {id_col} = ANY(%s)"
        ).format(
            id_col=sql.Identifier(config.id_column),
            content=sql.Identifier(config.content_column),
            table=sql.Identifier(config.table),
        )

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self._config.dsn,
                min_size=0,
                max_size=self._pool_max_size,
                timeout=self._config.connect_timeout,
                kwargs={"connect_timeout": self._config.connect_timeout},
                open=True,
            )
        return self._pool

    def fetch_original(self, pk: int | str) -> str | None:
        """Return the row's original_content, or ``None`` if no row matches.

        Raises :class:`PostgresFetchError` on any Postgres / pool failure.
        """
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(self._query, (pk,))
                    row: tuple[Any, ...] | None = cur.fetchone()
        except psycopg.Error as exc:
            raise PostgresFetchError(str(exc)) from exc
        except Exception as exc:  # pool timeouts etc.
            raise PostgresFetchError(str(exc)) from exc

        if row is None:
            return None
        value = row[0]
        if value is None:
            return None
        if not isinstance(value, str):
            raise PostgresFetchError(
                f"Expected str in column {self._config.content_column!r}; "
                f"got {type(value).__name__}"
            )
        return value

    def fetch_originals(
        self, pks: list[str], *, batch_size: int = 2000
    ) -> dict[str, str]:
        """Bulk lookup. Returns ``pk -> original_content`` for matches.

        PKs whose row is absent or whose content is null are simply
        missing from the result, so callers can detect gaps by set
        difference against the input.
        """
        out: dict[str, str] = {}
        if not pks:
            return out
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(pks), batch_size):
                        chunk = pks[i : i + batch_size]
                        cur.execute(self._bulk_query, (chunk,))
                        for pk, content in cur.fetchall():
                            if isinstance(content, str):
                                out[str(pk)] = content
        except psycopg.Error as exc:
            raise PostgresFetchError(str(exc)) from exc
        except Exception as exc:
            raise PostgresFetchError(str(exc)) from exc
        return out

    def close(self) -> None:
        """Close the connection pool, if it was opened."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
