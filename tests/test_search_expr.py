"""Unit tests for the Milvus ``expr`` builder used by chunk deduplication.

Covers :func:`_build_expr` and :func:`_render_pk_literal` in
``src/search/search.py``. These helpers translate a session's ``seen_pks``
set into a Milvus boolean expression of the form ``id not in [...]`` that
is injected into every ``AnnSearchRequest.expr`` and per-path
``client.search(filter=...)`` call so RRF reaches deeper into each path's
candidate pool.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.search.search import (
    _MAX_EXPR_PKS,
    _build_expr,
    _render_pk_literal,
)


# ---------------------------------------------------------------------------
# _render_pk_literal
# ---------------------------------------------------------------------------


def test_render_pk_literal_int() -> None:
    assert _render_pk_literal(42) == "42"


def test_render_pk_literal_varchar() -> None:
    assert _render_pk_literal("37901377-30006850-16") == '"37901377-30006850-16"'


def test_render_pk_literal_varchar_with_double_quote() -> None:
    # json.dumps must escape double quotes so the Milvus parser
    # doesn't terminate the literal prematurely.
    assert _render_pk_literal('a"b') == '"a\\"b"'


def test_render_pk_literal_varchar_with_backslash() -> None:
    assert _render_pk_literal("a\\b") == '"a\\\\b"'


def test_render_pk_literal_non_ascii_preserved() -> None:
    # ensure_ascii=False keeps CJK characters literal.
    rendered = _render_pk_literal("样本-123")
    assert rendered == '"样本-123"'


# ---------------------------------------------------------------------------
# _build_expr
# ---------------------------------------------------------------------------


def test_build_expr_none_returns_none() -> None:
    assert _build_expr(None) is None


def test_build_expr_empty_frozenset_returns_none() -> None:
    assert _build_expr(frozenset()) is None


def test_build_expr_single_varchar() -> None:
    expr = _build_expr(frozenset({"37901377-30006850-16"}))
    assert expr == 'id not in ["37901377-30006850-16"]'


def test_build_expr_single_int() -> None:
    expr = _build_expr(frozenset({42}))
    assert expr == "id not in [42]"


def test_build_expr_many_varchar() -> None:
    pks = {f"pk-{i}" for i in range(500)}
    expr = _build_expr(frozenset(pks))
    assert expr is not None
    assert expr.startswith("id not in [")
    assert expr.endswith("]")
    # Round-trip: strip prefix/suffix, parse the JSON list, compare as a set.
    rendered = expr[len("id not in [") : -1]
    parsed = json.loads("[" + rendered + "]")
    assert set(parsed) == pks


def test_build_expr_mixed_types_rejected_or_stable() -> None:
    # Set membership allows both int and str in principle; validate
    # the builder doesn't crash and renders each literal correctly.
    expr = _build_expr(frozenset({1, "a"}))
    assert expr is not None
    assert expr.startswith("id not in [")
    # Both forms should be present.
    assert "1" in expr
    assert '"a"' in expr


def test_build_expr_custom_pk_field() -> None:
    expr = _build_expr(frozenset({"x"}), pk_field="chunk_pk")
    assert expr == 'chunk_pk not in ["x"]'


def test_build_expr_ceiling_truncates_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    over = {f"pk-{i}" for i in range(_MAX_EXPR_PKS + 10)}
    with caplog.at_level(logging.WARNING, logger="src.search.search"):
        expr = _build_expr(frozenset(over))
    assert expr is not None
    # The rendered list must hold exactly _MAX_EXPR_PKS entries.
    rendered = expr[len("id not in [") : -1]
    parsed = json.loads("[" + rendered + "]")
    assert len(parsed) == _MAX_EXPR_PKS
    assert any("exceeds ceiling" in rec.message for rec in caplog.records)


def test_build_expr_escaping_roundtrips() -> None:
    # A PK containing a double quote must survive re-parsing as JSON.
    pk = 'weird"pk'
    expr = _build_expr(frozenset({pk}))
    assert expr is not None
    rendered = expr[len("id not in [") : -1]
    parsed = json.loads("[" + rendered + "]")
    assert parsed == [pk]
