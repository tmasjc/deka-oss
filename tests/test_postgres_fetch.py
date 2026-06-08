"""Unit tests for the parent_chunks Postgres fetcher.

The real ConnectionPool is replaced by a fake so these tests do not
require a running Postgres.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import psycopg
import pytest

from src.postgres.config import PostgresConfig
from src.postgres.fetch import OriginalContentFetcher, PostgresFetchError


class _FakeCursor:
    def __init__(
        self,
        row: tuple[Any, ...] | None,
        *,
        raise_on_execute: Exception | None = None,
        rows_by_param: dict[Any, list[tuple[Any, ...]]] | None = None,
    ) -> None:
        self._row = row
        self._raise = raise_on_execute
        self._rows_by_param = rows_by_param or {}
        self._last_rows: list[tuple[Any, ...]] = []
        self.executed: list[tuple[Any, tuple[Any, ...]]] = []

    def execute(self, query: Any, params: tuple[Any, ...]) -> None:
        self.executed.append((query, params))
        if self._raise is not None:
            raise self._raise
        key = params[0]
        if isinstance(key, list):
            key = tuple(key)
        self._last_rows = list(self._rows_by_param.get(key, []))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._last_rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._conn = _FakeConnection(cursor)
        self.closed = False

    @contextmanager
    def connection(self):
        yield self._conn

    def close(self) -> None:
        self.closed = True


def _config() -> PostgresConfig:
    return PostgresConfig(
        enabled=True,
        dsn="postgresql://localhost:5432/deka",
        table="parent_chunks",
        id_column="id",
        content_column="original_content",
        connect_timeout=5,
    )


def _install(fetcher: OriginalContentFetcher, pool: _FakePool) -> None:
    fetcher._pool = pool  # type: ignore[attr-defined]


def test_fetch_original_returns_string() -> None:
    cur = _FakeCursor(row=("hello world",))
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    result = fetcher.fetch_original(42)

    assert result == "hello world"
    assert cur.executed[0][1] == (42,)


def test_fetch_original_returns_none_when_missing() -> None:
    cur = _FakeCursor(row=None)
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    assert fetcher.fetch_original("missing-pk") is None


def test_fetch_original_returns_none_when_column_is_null() -> None:
    cur = _FakeCursor(row=(None,))
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    assert fetcher.fetch_original(1) is None


def test_fetch_original_raises_on_psycopg_error() -> None:
    cur = _FakeCursor(row=None, raise_on_execute=psycopg.Error("boom"))
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    with pytest.raises(PostgresFetchError):
        fetcher.fetch_original(1)


def test_fetch_original_raises_on_non_string_value() -> None:
    cur = _FakeCursor(row=(12345,))
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    with pytest.raises(PostgresFetchError):
        fetcher.fetch_original(1)


def test_fetch_originals_returns_dict_for_found_rows() -> None:
    cur = _FakeCursor(
        row=None,
        rows_by_param={
            ("a-1", "b-2", "c-3"): [("a-1", "hello"), ("c-3", "world")],
        },
    )
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    result = fetcher.fetch_originals(["a-1", "b-2", "c-3"])

    assert result == {"a-1": "hello", "c-3": "world"}


def test_fetch_originals_skips_null_content() -> None:
    cur = _FakeCursor(
        row=None,
        rows_by_param={("a-1", "b-2"): [("a-1", "hi"), ("b-2", None)]},
    )
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    assert fetcher.fetch_originals(["a-1", "b-2"]) == {"a-1": "hi"}


def test_fetch_originals_empty_input_short_circuits() -> None:
    cur = _FakeCursor(row=None)
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    assert fetcher.fetch_originals([]) == {}
    assert cur.executed == []


def test_fetch_originals_batches_large_input() -> None:
    cur = _FakeCursor(
        row=None,
        rows_by_param={
            tuple(f"p{i}" for i in range(3)): [(f"p{i}", f"c{i}") for i in range(3)],
            tuple(f"p{i}" for i in range(3, 5)): [
                (f"p{i}", f"c{i}") for i in range(3, 5)
            ],
        },
    )
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    pks = [f"p{i}" for i in range(5)]
    result = fetcher.fetch_originals(pks, batch_size=3)

    assert result == {f"p{i}": f"c{i}" for i in range(5)}
    assert len(cur.executed) == 2


def test_fetch_originals_raises_on_psycopg_error() -> None:
    cur = _FakeCursor(row=None, raise_on_execute=psycopg.Error("boom"))
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    with pytest.raises(PostgresFetchError):
        fetcher.fetch_originals(["x"])


def test_close_closes_pool() -> None:
    cur = _FakeCursor(row=("x",))
    pool = _FakePool(cur)
    fetcher = OriginalContentFetcher(_config())
    _install(fetcher, pool)

    fetcher.close()

    assert pool.closed is True
    assert fetcher._pool is None  # type: ignore[attr-defined]
