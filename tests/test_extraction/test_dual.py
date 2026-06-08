"""Tests for ``DualSpanExtractor`` (issue #54)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.extraction.cache import SpanCache
from src.extraction.dual import DualSpanExtractor
from src.extraction.errors import ExtractionError
from src.extraction.extractor import SpanExtractor


def _mock_response(data: dict[str, Any]) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_extractor(
    tmp_path: Path,
    *,
    model: str,
    client: MagicMock | None = None,
    cache: SpanCache | None = None,
) -> SpanExtractor:
    """Build a SpanExtractor with an explicit model_override so the
    two halves of a dual pair don't collide on cache keys."""
    if cache is None:
        cache = SpanCache(tmp_path / f"span_cache_{model.replace('/', '_')}.jsonl")
    return SpanExtractor(
        client=client or MagicMock(),
        cache=cache,
        model_override=model,
    )


class TestDualSpanExtractor:
    def test_both_succeed_returns_sorted_union(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1, 2], "reason": "A picks 1,2"}
        )
        client_b = MagicMock()
        client_b.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [2, 3], "reason": "B picks 2,3"}
        )
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        result = dual.extract(
            query="q", chunk_content="a\nb\nc\nd", prior_fit_spans=[]
        )

        assert result.span_line_indices == [1, 2, 3]
        assert "[primary]" in result.reason
        assert "[secondary]" in result.reason

    def test_union_can_exceed_single_vendor_cap(self, tmp_path: Path) -> None:
        # Each vendor returns 3 lines (the schema's per-vendor cap);
        # disjoint sets union to 6 — exceeds the cap, but
        # ``model_construct`` bypasses the validator so the union is
        # returned cleanly.
        client_a = MagicMock()
        client_a.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0, 2, 4], "reason": "A"}
        )
        client_b = MagicMock()
        client_b.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1, 3, 5], "reason": "B"}
        )
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        result = dual.extract(
            query="q",
            chunk_content="a\nb\nc\nd\ne\nf",
            prior_fit_spans=[],
        )

        assert result.span_line_indices == [0, 1, 2, 3, 4, 5]

    def test_primary_failure_uses_secondary_alone(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.side_effect = RuntimeError("net down")
        client_b = MagicMock()
        client_b.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1], "reason": "B alone"}
        )
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        result = dual.extract(
            query="q", chunk_content="a\nb\nc", prior_fit_spans=[]
        )
        assert result.span_line_indices == [1]
        # Primary exhausted its 2-attempt budget; secondary called once.
        assert client_a.chat.completions.create.call_count == 2
        assert client_b.chat.completions.create.call_count == 1

    def test_secondary_failure_uses_primary_alone(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "A alone"}
        )
        client_b = MagicMock()
        client_b.chat.completions.create.side_effect = RuntimeError("rate limit")
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        result = dual.extract(
            query="q", chunk_content="a\nb\nc", prior_fit_spans=[]
        )
        assert result.span_line_indices == [0]
        assert client_a.chat.completions.create.call_count == 1
        assert client_b.chat.completions.create.call_count == 2

    def test_both_fail_raises_extraction_error(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.side_effect = RuntimeError("a-down")
        client_b = MagicMock()
        client_b.chat.completions.create.side_effect = RuntimeError("b-down")
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        with pytest.raises(ExtractionError, match="[Bb]oth vendors failed"):
            dual.extract(query="q", chunk_content="a\nb\nc", prior_fit_spans=[])

    def test_calls_run_in_parallel(self, tmp_path: Path) -> None:
        # Each vendor blocks for ~120ms before responding. If the
        # calls were sequential total time would be ~240ms; in
        # parallel it should be ~120ms. Generous threshold to avoid
        # CI flake.
        delay = 0.12

        def slow_response(*_args, **_kwargs):
            time.sleep(delay)
            return _mock_response({"span_line_indices": [0], "reason": "ok"})

        client_a = MagicMock()
        client_a.chat.completions.create.side_effect = slow_response
        client_b = MagicMock()
        client_b.chat.completions.create.side_effect = slow_response
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        started = time.perf_counter()
        dual.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        elapsed = time.perf_counter() - started

        assert elapsed < delay * 1.8, (
            f"Expected parallel execution (~{delay:.2f}s); "
            f"got {elapsed:.3f}s — looks sequential."
        )

    def test_stats_aggregate_across_vendors(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "A"}
        )
        client_b = MagicMock()
        client_b.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1], "reason": "B"}
        )
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        # Two distinct chunks → 2 calls per vendor → 4 total.
        dual.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        dual.extract(query="q", chunk_content="c\nd", prior_fit_spans=[])

        assert dual.call_count == 4
        assert dual.cache_hits == 0
        assert dual.model == "vendor-a+vendor-b"
        # Latency is summed across vendors (parallel but cumulative).
        assert dual.latency_ms_total == pytest.approx(
            primary.latency_ms_total + secondary.latency_ms_total,
            abs=0.01,
        )

    def test_cache_hits_counted_per_vendor(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "A"}
        )
        client_b = MagicMock()
        client_b.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1], "reason": "B"}
        )
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        # First call populates both vendor caches; second call hits both.
        dual.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])
        dual.extract(query="q", chunk_content="a\nb", prior_fit_spans=[])

        assert dual.cache_hits == 2  # one per vendor on the second call
        assert client_a.chat.completions.create.call_count == 1
        assert client_b.chat.completions.create.call_count == 1

    def test_prompt_version_mismatch_rejected_at_construction(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", cache=cache)
        secondary = _make_extractor(tmp_path, model="vendor-b", cache=cache)
        # Force a mismatch by mutating the loaded prompt version.
        monkeypatch.setattr(secondary, "_prompt_version", "v-different")

        with pytest.raises(ExtractionError, match="prompt_version"):
            DualSpanExtractor(primary, secondary)

    def test_same_model_rejected_at_construction(self, tmp_path: Path) -> None:
        # Two extractors built with the same model would share cache
        # keys — guard at construction so the misconfig surfaces
        # immediately rather than silently halving cache utility.
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="same-model", cache=cache)
        secondary = _make_extractor(tmp_path, model="same-model", cache=cache)

        with pytest.raises(ExtractionError, match="distinct models"):
            DualSpanExtractor(primary, secondary)

    def test_reason_carries_both_justifications(self, tmp_path: Path) -> None:
        client_a = MagicMock()
        client_a.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [0], "reason": "primary says alpha"}
        )
        client_b = MagicMock()
        client_b.chat.completions.create.return_value = _mock_response(
            {"span_line_indices": [1], "reason": "secondary says beta"}
        )
        cache = SpanCache(tmp_path / "span_cache.jsonl")
        primary = _make_extractor(tmp_path, model="vendor-a", client=client_a, cache=cache)
        secondary = _make_extractor(
            tmp_path, model="vendor-b", client=client_b, cache=cache
        )
        dual = DualSpanExtractor(primary, secondary)

        result = dual.extract(
            query="q", chunk_content="a\nb\nc", prior_fit_spans=[]
        )
        assert "primary says alpha" in result.reason
        assert "secondary says beta" in result.reason

    def test_derive_span_text_static(self) -> None:
        # The static derive_span_text passes through to the
        # single-extractor implementation — no surprises for the
        # _TurnExtractSpan caller.
        assert (
            DualSpanExtractor.derive_span_text("a\nb\nc", [0, 2]) == "a\nc"
        )


class TestDefaultFactoryWiring:
    """The session factory in ``src.web_api.sessions`` returns a
    ``DualSpanExtractor`` iff the extraction config has the
    secondary keys set (issue #54)."""

    def test_returns_dual_when_secondary_configured(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from src.extraction.extractor import ExtractionConfig
        from src.web_api import sessions as sessions_mod

        def fake_load_config(path=None):
            return ExtractionConfig(
                model="openai/gpt-5.4-mini",
                base_url="https://openrouter.ai/api/v1",
                prompt_version="v1",
                api_key_env="OPENROUTER_API_KEY",
                cache_root=tmp_path / "fallback.jsonl",
                secondary_model="qwen-plus",
                secondary_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                secondary_api_key_env="DASHSCOPE_API_KEY",
            )

        monkeypatch.setattr(sessions_mod, "_load_extraction_config", fake_load_config)
        # SpanExtractor.__init__ itself also reads the config via
        # extractor._load_config — patch that too, otherwise the
        # constructor sees the real config.yaml.
        from src.extraction import extractor as extractor_mod

        monkeypatch.setattr(extractor_mod, "_load_config", fake_load_config)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-a")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-b")

        result = sessions_mod._default_extractor_factory(tmp_path, "sess-1")
        assert isinstance(result, DualSpanExtractor)
        assert result.model == "openai/gpt-5.4-mini+qwen-plus"

    def test_returns_single_when_no_secondary(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from src.extraction.extractor import ExtractionConfig
        from src.web_api import sessions as sessions_mod

        def fake_load_config(path=None):
            return ExtractionConfig(
                model="openai/gpt-5.4-mini",
                base_url="https://openrouter.ai/api/v1",
                prompt_version="v1",
                api_key_env="OPENROUTER_API_KEY",
                cache_root=tmp_path / "fallback.jsonl",
            )

        monkeypatch.setattr(sessions_mod, "_load_extraction_config", fake_load_config)
        from src.extraction import extractor as extractor_mod

        monkeypatch.setattr(extractor_mod, "_load_config", fake_load_config)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        result = sessions_mod._default_extractor_factory(tmp_path, "sess-1")
        assert isinstance(result, SpanExtractor)
        assert not isinstance(result, DualSpanExtractor)
