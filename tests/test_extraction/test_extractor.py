"""Tests for ``SpanExtractor`` with a mocked OpenAI client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.extraction.cache import SpanCache
from src.extraction.errors import ExtractionError
from src.extraction.extractor import SpanExtractor, _extract_json


def _mock_response(data: dict[str, Any] | str) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(data) if isinstance(data, dict) else data
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_extractor(tmp_path: Path, client: MagicMock | None = None) -> SpanExtractor:
    cache = SpanCache(tmp_path / "span_cache.jsonl")
    return SpanExtractor(
        client=client or MagicMock(),
        cache=cache,
    )


class TestExtractHappyPath:
    def test_returns_parsed_result(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1, 2], "reason": "clear concept expression"}
        )
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(
            query="query",
            chunk_content="a\nb\nc\nd",
            prior_fit_spans=[],
        )

        assert result.span_line_indices == [1, 2]
        assert result.reason == "clear concept expression"

    def test_empty_span_accepted(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [], "reason": "no clean span"}
        )
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        assert result.span_line_indices == []

    def test_calls_with_temperature_zero_and_json_format(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "ok"}
        )
        extractor = _make_extractor(tmp_path, client)
        extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert len(call_kwargs["messages"]) == 2
        # Qwen-family chain-of-thought reasoning is hard-disabled on
        # DashScope so the extractor's tiny JSON output doesn't pay
        # the thinking-token latency tax (issue #54 follow-up).
        assert call_kwargs["extra_body"] == {"enable_thinking": False}


class TestCacheIntegration:
    def test_cache_hit_avoids_llm_call(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0, 1], "reason": "ok"}
        )
        extractor = _make_extractor(tmp_path, client)

        r1 = extractor.extract(query="q", chunk_content="a\nb\nc", prior_fit_spans=[])
        r2 = extractor.extract(query="q", chunk_content="a\nb\nc", prior_fit_spans=[])

        assert r1 == r2
        assert client.chat.completions.create.call_count == 1
        assert extractor.cache_hits == 1
        assert extractor.call_count == 2

    def test_different_chunks_miss(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response({"span_line_indices": [0], "reason": "a"}),
            _mock_response({"span_line_indices": [1], "reason": "b"}),
        ]
        extractor = _make_extractor(tmp_path, client)

        a = extractor.extract(query="q", chunk_content="A\nB", prior_fit_spans=[])
        b = extractor.extract(query="q", chunk_content="C\nD", prior_fit_spans=[])

        assert a.span_line_indices == [0]
        assert b.span_line_indices == [1]
        assert client.chat.completions.create.call_count == 2

    def test_prior_spans_do_not_affect_cache_key(self, tmp_path: Path) -> None:
        """Cache key is (model, prompt_version, query, chunk_content) only —
        prior spans change the prompt but not the cache identity, so the
        second call returns the cached answer regardless of the evolving
        few-shot context."""
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "ok"}
        )
        extractor = _make_extractor(tmp_path, client)

        extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        extractor.extract(
            query="q", chunk_content="a\nb", prior_fit_spans=["a previous span"]
        )

        assert client.chat.completions.create.call_count == 1


class TestErrorHandling:
    def test_bad_json_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response("not json at all")
        extractor = _make_extractor(tmp_path, client)

        with pytest.raises(ExtractionError):
            extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])

    def test_non_contiguous_indices_now_accepted(self, tmp_path: Path) -> None:
        # Regression: contiguity was relaxed — [0, 3] is sorted + unique
        # so it must pass validation and reach the cache.
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0, 3], "reason": "two disjoint hits"}
        )
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(
            query="q", chunk_content="a\nb\nc\nd", prior_fit_spans=[]
        )
        assert result.span_line_indices == [0, 3]

    def test_too_long_span_rejected(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0, 1, 2, 3], "reason": "bad"}
        )
        extractor = _make_extractor(tmp_path, client)

        with pytest.raises(ExtractionError, match="validation"):
            extractor.extract(query="q", chunk_content="a\nb\nc\nd", prior_fit_spans=[])

    def test_api_failure_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network")
        extractor = _make_extractor(tmp_path, client)

        with pytest.raises(ExtractionError, match="network"):
            extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])

    def test_empty_content_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        extractor = _make_extractor(tmp_path, client)
        with pytest.raises(ExtractionError, match="empty"):
            extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])


class TestRetry:
    """One automatic retry on ExtractionError before surfacing failure."""

    def test_retries_once_on_api_failure_then_succeeds(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("transient network blip"),
            _mock_response({"span_line_indices": [1], "reason": "ok"}),
        ]
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(
            query="q", chunk_content="a\nb\nc", prior_fit_spans=[]
        )
        assert result.span_line_indices == [1]
        assert client.chat.completions.create.call_count == 2

    def test_retries_once_on_validation_failure_then_succeeds(
        self, tmp_path: Path
    ) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response({"span_line_indices": [0, 1, 2, 3], "reason": "bad"}),
            _mock_response({"span_line_indices": [0], "reason": "ok"}),
        ]
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(
            query="q", chunk_content="a\nb\nc\nd", prior_fit_spans=[]
        )
        assert result.span_line_indices == [0]
        assert client.chat.completions.create.call_count == 2

    def test_caps_at_two_attempts_then_raises(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("net 1"),
            RuntimeError("net 2"),
        ]
        extractor = _make_extractor(tmp_path, client)

        with pytest.raises(ExtractionError, match="net 2"):
            extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        assert client.chat.completions.create.call_count == 2


class TestOutOfRangeIndices:
    """Issue #51: bounds-check the LLM's ``span_line_indices`` against
    the chunk's actual line count. The Pydantic schema can't see
    ``chunk_content``, so the check has to live in ``extract()`` —
    raising ``ExtractionError`` so the existing retry-once →
    drop-chunk pipeline absorbs the failure instead of letting an
    ``IndexError`` escape on the downstream ``derive_span_text``
    call."""

    def test_raises_after_retries_when_indices_out_of_range(
        self, tmp_path: Path
    ) -> None:
        # Chunk has 3 lines (valid indices 0..2); LLM returns [3] on
        # both attempts — model schema accepts it (single, non-negative,
        # sorted) but the bounds check trips it.
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response({"span_line_indices": [3], "reason": "bad-1"}),
            _mock_response({"span_line_indices": [3], "reason": "bad-2"}),
        ]
        extractor = _make_extractor(tmp_path, client)

        with pytest.raises(ExtractionError, match="out-of-range"):
            extractor.extract(query="q", chunk_content="a\nb\nc", prior_fit_spans=[])
        assert client.chat.completions.create.call_count == 2
        # Cache never receives a malformed result.
        assert extractor.cache_hits == 0

    def test_retries_when_indices_out_of_range_then_succeeds(
        self, tmp_path: Path
    ) -> None:
        # Bug case: first attempt returns an out-of-range index, second
        # attempt returns a valid one. The 2-attempt budget rescues the
        # turn without dropping the chunk.
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response({"span_line_indices": [3], "reason": "bad"}),
            _mock_response({"span_line_indices": [1], "reason": "ok"}),
        ]
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(
            query="q", chunk_content="a\nb\nc", prior_fit_spans=[]
        )
        assert result.span_line_indices == [1]
        assert client.chat.completions.create.call_count == 2

    def test_accepts_indices_at_upper_boundary(self, tmp_path: Path) -> None:
        # Regression guard against off-by-one: for a 3-line chunk the
        # last valid index is 2. The guard must not reject it.
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [2], "reason": "last line"}
        )
        extractor = _make_extractor(tmp_path, client)

        result = extractor.extract(
            query="q", chunk_content="a\nb\nc", prior_fit_spans=[]
        )
        assert result.span_line_indices == [2]
        assert client.chat.completions.create.call_count == 1


class TestDeriveSpanText:
    def test_empty_indices_returns_empty_string(self) -> None:
        assert SpanExtractor.derive_span_text("a\nb\nc", []) == ""

    def test_single_line(self) -> None:
        assert SpanExtractor.derive_span_text("老师：x\n家长：y", [1]) == "家长：y"

    def test_two_lines(self) -> None:
        assert (
            SpanExtractor.derive_span_text("老师：x\n家长：y\n老师：z", [0, 1])
            == "老师：x\n家长：y"
        )


class TestExtractJson:
    def test_plain_json_passthrough(self) -> None:
        raw = '{"span_line_indices": [0], "reason": "ok"}'
        assert _extract_json(raw) == raw

    def test_think_tags_stripped(self) -> None:
        raw = '<think>reasoning</think>\n{"span_line_indices": [0], "reason": "ok"}'
        assert _extract_json(raw) == '{"span_line_indices": [0], "reason": "ok"}'

    def test_prose_wrapper_extracted(self) -> None:
        raw = 'Here it is:\n{"span_line_indices": [], "reason": "x"}\nDone.'
        assert _extract_json(raw) == '{"span_line_indices": [], "reason": "x"}'


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a unified config.yaml with a valid ``extraction`` section."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "extraction:\n"
        "  model: openai/gpt-5.4-mini\n"
        "  base_url: https://openrouter.ai/api/v1\n"
        "  prompt_version: v1\n"
        "  api_key_env: OPENROUTER_API_KEY\n"
        f"  cache_root: {tmp_path / 'span_cache.jsonl'}\n",
        encoding="utf-8",
    )
    return cfg


class TestConstructorPaths:
    def test_missing_api_key_raises(self, tmp_path: Path, monkeypatch) -> None:
        """No client, no api_key, no env var -> ExtractionError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        cfg = _write_minimal_config(tmp_path)
        with pytest.raises(ExtractionError, match="No API key"):
            SpanExtractor(cache=SpanCache(tmp_path / "c"), config_path=cfg)

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        """A strict loader: no file -> ExtractionError with explicit message."""
        with pytest.raises(ExtractionError, match="not found"):
            SpanExtractor(
                cache=SpanCache(tmp_path / "c"),
                config_path=tmp_path / "missing.yaml",
            )

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        """Partial config -> ExtractionError listing missing keys."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "extraction:\n"
            "  model: openai/gpt-5.4-mini\n"
            "  base_url: x\n"
            "  prompt_version: v1\n",
            encoding="utf-8",
        )
        with pytest.raises(ExtractionError, match="missing required keys"):
            SpanExtractor(cache=SpanCache(tmp_path / "c"), config_path=cfg)

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "extraction:\n"
            "  model: m\n"
            "  base_url: b\n"
            "  prompt_version: v1\n"
            "  api_key_env: K\n"
            "  cache_root: /tmp/c.jsonl\n"
            "  stray: nope\n",
            encoding="utf-8",
        )
        with pytest.raises(ExtractionError, match="unknown keys"):
            SpanExtractor(cache=SpanCache(tmp_path / "c"), config_path=cfg)

    def test_loader_accepts_secondary_keys(self, tmp_path: Path, monkeypatch) -> None:
        """Dual-vendor config with all three ``secondary_*`` keys
        present must load without error and surface
        ``has_secondary == True`` (issue #54)."""
        from src.extraction.extractor import _load_config

        monkeypatch.setenv("OPENROUTER_API_KEY", "x")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "y")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "extraction:\n"
            "  model: openai/gpt-5.4-mini\n"
            "  base_url: https://openrouter.ai/api/v1\n"
            "  prompt_version: v1\n"
            "  api_key_env: OPENROUTER_API_KEY\n"
            f"  cache_root: {tmp_path / 'span_cache.jsonl'}\n"
            "  secondary_model: qwen-plus\n"
            "  secondary_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
            "  secondary_api_key_env: DASHSCOPE_API_KEY\n",
            encoding="utf-8",
        )
        loaded = _load_config(cfg)
        assert loaded.has_secondary is True
        assert loaded.secondary_model == "qwen-plus"
        assert loaded.secondary_api_key_env == "DASHSCOPE_API_KEY"

    def test_loader_rejects_partial_secondary_keys(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Setting some but not all ``secondary_*`` keys must be
        rejected — silent fallback to single-vendor on partial
        config would mask the misconfig."""
        from src.extraction.extractor import _load_config

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "extraction:\n"
            "  model: m\n"
            "  base_url: b\n"
            "  prompt_version: v1\n"
            "  api_key_env: K\n"
            f"  cache_root: {tmp_path / 'span_cache.jsonl'}\n"
            "  secondary_model: qwen-plus\n",
            encoding="utf-8",
        )
        with pytest.raises(ExtractionError, match="dual-vendor extraction requires"):
            _load_config(cfg)

    def test_loader_single_vendor_still_works(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Backwards-compat sanity: a config without any
        ``secondary_*`` keys still loads cleanly and reports
        ``has_secondary == False``."""
        from src.extraction.extractor import _load_config

        monkeypatch.setenv("OPENROUTER_API_KEY", "x")
        cfg = _write_minimal_config(tmp_path)
        loaded = _load_config(cfg)
        assert loaded.has_secondary is False
        assert loaded.secondary_model is None

    def test_reset_stats(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "ok"}
        )
        extractor = _make_extractor(tmp_path, client)
        extractor.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        assert extractor.call_count == 1
        extractor.reset_stats()
        assert extractor.call_count == 0
        assert extractor.cache_hits == 0
        assert extractor.latency_ms_total == 0.0
