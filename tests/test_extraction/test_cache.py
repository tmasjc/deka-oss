"""Tests for ``CacheKey`` and ``SpanCache``."""

from __future__ import annotations

import json
from pathlib import Path


from src.extraction.cache import CacheKey, SpanCache
from src.extraction.schema import SpanExtractionResult


def _key(**overrides: str) -> CacheKey:
    base = {
        "model_id": "openai/gpt-5.4-mini",
        "prompt_version": "v1",
        "query": "家长觉得课程有点难",
        "chunk_content": "老师：hi\n家长：hello",
    }
    base.update(overrides)
    return CacheKey(**base)


class TestCacheKey:
    def test_sha256_stable(self) -> None:
        assert _key().sha256() == _key().sha256()

    def test_different_query_different_hash(self) -> None:
        assert _key().sha256() != _key(query="other").sha256()

    def test_different_content_different_hash(self) -> None:
        assert _key().sha256() != _key(chunk_content="老师：x\n家长：y").sha256()

    def test_different_model_different_hash(self) -> None:
        assert _key().sha256() != _key(model_id="other/model").sha256()

    def test_different_prompt_version_different_hash(self) -> None:
        assert _key().sha256() != _key(prompt_version="v2").sha256()


class TestSpanCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = SpanCache(tmp_path / "cache.jsonl")
        assert cache.get(_key()) is None

    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = SpanCache(tmp_path / "cache.jsonl")
        result = SpanExtractionResult(span_line_indices=[1, 2], reason="ok")
        cache.put(_key(), result)
        assert cache.get(_key()) == result

    def test_different_keys_isolated(self, tmp_path: Path) -> None:
        cache = SpanCache(tmp_path / "cache.jsonl")
        a = SpanExtractionResult(span_line_indices=[0], reason="a")
        b = SpanExtractionResult(span_line_indices=[2], reason="b")
        cache.put(_key(), a)
        cache.put(_key(query="different"), b)
        assert cache.get(_key()) == a
        assert cache.get(_key(query="different")) == b

    def test_put_writes_slim_entry_with_span_text(self, tmp_path: Path) -> None:
        cache = SpanCache(tmp_path / "cache.jsonl")
        key = _key()  # chunk_content = "老师：hi\n家长：hello"
        cache.put(key, SpanExtractionResult(span_line_indices=[0], reason="first line"))
        line = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8"))
        assert set(line.keys()) == {"span_line_indices", "reason", "span_text"}
        assert line["span_text"] == "老师：hi"

    def test_put_empty_span_text(self, tmp_path: Path) -> None:
        cache = SpanCache(tmp_path / "cache.jsonl")
        cache.put(_key(), SpanExtractionResult(span_line_indices=[], reason="no span"))
        line = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8"))
        assert line["span_text"] == ""

    def test_cache_parent_auto_created(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c.jsonl"
        cache = SpanCache(deep)
        assert deep.parent.exists()
        assert cache.path == deep

    def test_second_put_appends(self, tmp_path: Path) -> None:
        cache = SpanCache(tmp_path / "cache.jsonl")
        cache.put(_key(), SpanExtractionResult(span_line_indices=[0], reason="a"))
        cache.put(_key(query="q2"), SpanExtractionResult(span_line_indices=[1], reason="b"))
        lines = (tmp_path / "cache.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
